# backtest/data/request_aggregator.py
"""
Module: Data Request Aggregator
Purpose: Consolidate data requests from multiple calculation modules to minimize API calls
Features: Request batching, intelligent merging, prefetch optimization, async execution
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
from enum import Enum
from typing import Union

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Types of data that can be requested"""
    BARS = "bars"
    TRADES = "trades"
    QUOTES = "quotes"
    ALIGNED_TRADES = "aligned_trades"


@dataclass
class DataNeed:
    """Represents a data requirement from a calculation module"""
    module_name: str
    symbol: str
    data_type: DataType
    timeframe: str  # For bars: '1min', '5min', etc. For tick data: 'tick'
    start_time: datetime
    end_time: datetime
    priority: int = 5  # 1-10, higher = more important
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure timezone aware datetimes"""
        if self.start_time.tzinfo is None:
            self.start_time = self.start_time.replace(tzinfo=timezone.utc)
        if self.end_time.tzinfo is None:
            self.end_time = self.end_time.replace(tzinfo=timezone.utc)


@dataclass
class AggregatedRequest:
    """Consolidated request covering multiple module needs"""
    symbol: str
    data_type: DataType
    timeframe: str
    start_time: datetime
    end_time: datetime
    requesting_modules: List[str] = field(default_factory=list)
    module_needs: List[DataNeed] = field(default_factory=list)
    
    def covers(self, need: DataNeed) -> bool:
        """Check if this request covers a specific need"""
        return (
            self.symbol == need.symbol and
            self.data_type == need.data_type and
            self.timeframe == need.timeframe and
            self.start_time <= need.start_time and
            self.end_time >= need.end_time
        )


class RequestAggregator:
    """
    Aggregates data requests from multiple calculation modules to minimize API calls.
    
    Key Features:
    1. Collects all data needs before fetching
    2. Merges overlapping requests
    3. Handles different data types (bars, trades, quotes)
    4. Distributes fetched data back to requesting modules
    5. Supports async batch operations
    """
    
    def __init__(self, data_manager=None, extend_window_pct: float = 0.1):
        """
        Initialize the aggregator.
        
        Args:
            data_manager: PolygonDataManager instance (optional, can be set later)
            extend_window_pct: Percentage to extend time windows for efficiency
        """
        self.data_manager = data_manager
        self.extend_window_pct = extend_window_pct
        
        # Track pending needs by symbol and data type
        self.pending_needs: Dict[str, List[DataNeed]] = defaultdict(list)
        
        # Cache for fetched data during a batch operation
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        # Statistics
        self.stats = {
            'total_needs': 0,
            'aggregated_requests': 0,
            'api_calls_saved': 0,
            'data_points_fetched': 0,
            'cache_reuse_count': 0
        }
        
        logger.info(f"RequestAggregator initialized with {extend_window_pct*100}% window extension")
    
    def set_data_manager(self, data_manager):
        """Set or update the data manager"""
        self.data_manager = data_manager
        logger.info("Data manager set for RequestAggregator")
    
    def register_need(self, need: DataNeed):
        """
        Register a data need from a calculation module.
        
        Args:
            need: DataNeed object describing what data is required
        """
        key = f"{need.symbol}_{need.data_type.value}_{need.timeframe}"
        self.pending_needs[key].append(need)
        self.stats['total_needs'] += 1
        
        logger.debug(
            f"Registered need from {need.module_name}: "
            f"{need.symbol} {need.data_type.value} {need.timeframe} "
            f"[{need.start_time} to {need.end_time}]"
        )
    
    def register_needs(self, needs: List[DataNeed]):
        """Register multiple needs at once"""
        for need in needs:
            self.register_need(need)
    
    def clear_needs(self):
        """Clear all pending needs and cache"""
        self.pending_needs.clear()
        self.data_cache.clear()
        # Reset stats but keep running totals
        self.stats.update({
            'total_needs': 0,
            'aggregated_requests': 0,
            'api_calls_saved': 0
        })
        logger.info("Cleared all pending needs and cache")
    
    def _aggregate_needs(self) -> List[AggregatedRequest]:
        """
        Aggregate pending needs into optimized requests.
        
        Returns:
            List of AggregatedRequest objects
        """
        aggregated = []
        
        for key, needs in self.pending_needs.items():
            if not needs:
                continue
            
            # Sort needs by start time
            sorted_needs = sorted(needs, key=lambda n: n.start_time)
            
            # Use a greedy algorithm to merge overlapping/adjacent requests
            current_request = None
            
            for need in sorted_needs:
                if current_request is None:
                    # Start new aggregated request
                    current_request = AggregatedRequest(
                        symbol=need.symbol,
                        data_type=need.data_type,
                        timeframe=need.timeframe,
                        start_time=need.start_time,
                        end_time=need.end_time,
                        requesting_modules=[need.module_name],
                        module_needs=[need]
                    )
                else:
                    # Check if we should merge with current request
                    # Merge if overlapping or within extension window
                    extension = timedelta(
                        seconds=(need.end_time - need.start_time).total_seconds() * self.extend_window_pct
                    )
                    
                    if need.start_time <= current_request.end_time + extension:
                        # Merge into current request
                        current_request.end_time = max(current_request.end_time, need.end_time)
                        current_request.start_time = min(current_request.start_time, need.start_time)
                        if need.module_name not in current_request.requesting_modules:
                            current_request.requesting_modules.append(need.module_name)
                        current_request.module_needs.append(need)
                    else:
                        # Save current and start new request
                        aggregated.append(current_request)
                        current_request = AggregatedRequest(
                            symbol=need.symbol,
                            data_type=need.data_type,
                            timeframe=need.timeframe,
                            start_time=need.start_time,
                            end_time=need.end_time,
                            requesting_modules=[need.module_name],
                            module_needs=[need]
                        )
            
            # Don't forget the last request
            if current_request:
                aggregated.append(current_request)
        
        # Apply window extension to all requests
        for req in aggregated:
            duration = req.end_time - req.start_time
            extension = timedelta(seconds=duration.total_seconds() * self.extend_window_pct)
            req.start_time -= extension
            req.end_time += extension
        
        self.stats['aggregated_requests'] = len(aggregated)
        self.stats['api_calls_saved'] = self.stats['total_needs'] - len(aggregated)
        
        logger.info(
            f"Aggregated {self.stats['total_needs']} needs into "
            f"{len(aggregated)} requests (saved {self.stats['api_calls_saved']} API calls)"
        )
        
        return aggregated
    
    async def fetch_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch all aggregated data requests concurrently.
        
        Returns:
            Dict mapping module_name -> data_key -> DataFrame
            where data_key is f"{symbol}_{data_type}_{timeframe}"
        """
        if not self.data_manager:
            raise ValueError("Data manager not set. Call set_data_manager() first.")
        
        # Get aggregated requests
        aggregated_requests = self._aggregate_needs()
        
        if not aggregated_requests:
            logger.warning("No data needs to fetch")
            return {}
        
        # Create async tasks for all requests
        tasks = []
        request_map = {}  # Track which task corresponds to which request
        
        for i, req in enumerate(aggregated_requests):
            if req.data_type == DataType.BARS:
                # For bars, use the load_bars method
                task = self.data_manager.load_bars(
                    symbol=req.symbol,
                    start_time=req.start_time,
                    end_time=req.end_time,
                    timeframe=req.timeframe
                )
            elif req.data_type == DataType.TRADES:
                task = self.data_manager.load_trades(
                    symbol=req.symbol,
                    start_time=req.start_time,
                    end_time=req.end_time
                )
            elif req.data_type == DataType.QUOTES:
                task = self.data_manager.load_quotes(
                    symbol=req.symbol,
                    start_time=req.start_time,
                    end_time=req.end_time
                )
            else:
                logger.error(f"Unknown data type: {req.data_type}")
                continue
            
            tasks.append(task)
            request_map[i] = req
        
        # Execute all tasks concurrently
        logger.info(f"Executing {len(tasks)} data fetches concurrently...")
        start_time = datetime.now()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        fetch_duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Completed {len(tasks)} fetches in {fetch_duration:.2f} seconds")
        
        # Process results and build response structure
        module_data = defaultdict(dict)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching data: {result}")
                # Create empty DataFrame for failed requests
                req = request_map[i]
                for need in req.module_needs:
                    data_key = f"{need.symbol}_{need.data_type.value}_{need.timeframe}"
                    module_data[need.module_name][data_key] = pd.DataFrame()
                continue
            
            req = request_map[i]
            if result is not None and not result.empty:
                # Cache the full result
                cache_key = f"{req.symbol}_{req.data_type.value}_{req.timeframe}"
                self.data_cache[cache_key] = result
                self.stats['data_points_fetched'] += len(result)
                
                # Distribute to requesting modules
                for need in req.module_needs:
                    # Filter to the specific time range needed
                    filtered_data = self._filter_data(result, need.start_time, need.end_time)
                    
                    data_key = f"{need.symbol}_{need.data_type.value}_{need.timeframe}"
                    module_data[need.module_name][data_key] = filtered_data
                    
                    logger.debug(
                        f"Distributed {len(filtered_data)} data points to "
                        f"{need.module_name} for {data_key}"
                    )
            else:
                # Handle empty results
                for need in req.module_needs:
                    data_key = f"{need.symbol}_{need.data_type.value}_{need.timeframe}"
                    module_data[need.module_name][data_key] = pd.DataFrame()
                    logger.warning(f"No data available for {data_key}")
        
        return dict(module_data)
    
    def _filter_data(self, df: pd.DataFrame, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Filter dataframe to specific time range"""
        if df.empty:
            return df
        
        # Ensure timezone aware comparison
        if start_time.tzinfo is None:
            start_time = pd.Timestamp(start_time).tz_localize('UTC')
        else:
            start_time = pd.Timestamp(start_time)
            
        if end_time.tzinfo is None:
            end_time = pd.Timestamp(end_time).tz_localize('UTC')
        else:
            end_time = pd.Timestamp(end_time)
        
        mask = (df.index >= start_time) & (df.index <= end_time)
        return df[mask].copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        return {
            **self.stats,
            'pending_needs_count': sum(len(needs) for needs in self.pending_needs.values()),
            'unique_symbols': len(set(
                need.symbol 
                for needs in self.pending_needs.values() 
                for need in needs
            )),
            'cached_datasets': len(self.data_cache)
        }
    
    def create_request_report(self) -> str:
        """Create a detailed report of aggregation efficiency"""
        report = []
        report.append("=" * 60)
        report.append("REQUEST AGGREGATION REPORT")
        report.append("=" * 60)
        
        stats = self.get_stats()
        report.append(f"\nTotal Needs Registered: {stats['total_needs']}")
        report.append(f"Aggregated Requests: {stats['aggregated_requests']}")
        report.append(f"API Calls Saved: {stats['api_calls_saved']}")
        report.append(f"Efficiency: {(stats['api_calls_saved'] / max(1, stats['total_needs']) * 100):.1f}%")
        report.append(f"Data Points Fetched: {stats['data_points_fetched']:,}")
        
        # Show aggregation details
        if self.pending_needs:
            report.append("\n\nAGGREGATION DETAILS:")
            report.append("-" * 40)
            
            for key, needs in self.pending_needs.items():
                if not needs:
                    continue
                
                report.append(f"\n{key}:")
                report.append(f"  Original Requests: {len(needs)}")
                
                # Show time range coverage
                min_start = min(n.start_time for n in needs)
                max_end = max(n.end_time for n in needs)
                report.append(f"  Combined Range: {min_start} to {max_end}")
                
                # Show requesting modules
                modules = list(set(n.module_name for n in needs))
                report.append(f"  Modules: {', '.join(modules)}")
        
        return "\n".join(report)
    
    def get_pending_needs_summary(self) -> Dict[str, Any]:
        """Get a summary of pending needs by module and data type"""
        summary = {
            'by_module': defaultdict(lambda: {'bars': 0, 'trades': 0, 'quotes': 0}),
            'by_symbol': defaultdict(lambda: {'bars': 0, 'trades': 0, 'quotes': 0}),
            'total_time_range': {},
            'priority_distribution': defaultdict(int)
        }
        
        for needs in self.pending_needs.values():
            for need in needs:
                # By module
                summary['by_module'][need.module_name][need.data_type.value] += 1
                
                # By symbol
                summary['by_symbol'][need.symbol][need.data_type.value] += 1
                
                # Priority distribution
                summary['priority_distribution'][need.priority] += 1
                
                # Total time range per symbol
                if need.symbol not in summary['total_time_range']:
                    summary['total_time_range'][need.symbol] = {
                        'start': need.start_time,
                        'end': need.end_time
                    }
                else:
                    summary['total_time_range'][need.symbol]['start'] = min(
                        summary['total_time_range'][need.symbol]['start'],
                        need.start_time
                    )
                    summary['total_time_range'][need.symbol]['end'] = max(
                        summary['total_time_range'][need.symbol]['end'],
                        need.end_time
                    )
        
        return dict(summary)


# Example integration with the new modular PolygonDataManager
async def example_with_modular_data_manager():
    """Example showing integration with the new modular structure"""
    from polygon_data_manager import PolygonDataManager
    
    # Create data manager with the new modular structure
    data_manager = PolygonDataManager(
        api_key='your_api_key',
        cache_dir='./cache',
        memory_cache_size=100,
        file_cache_hours=24
    )
    
    # Create aggregator
    aggregator = RequestAggregator(
        data_manager=data_manager,
        extend_window_pct=0.15  # 15% extension
    )
    
    # Simulate needs from multiple modules
    base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    
    needs = [
        # Trend Analysis Module
        DataNeed(
            module_name="TrendAnalysis",
            symbol="AAPL",
            data_type=DataType.BARS,
            timeframe="1min",
            start_time=base_time - timedelta(hours=2),
            end_time=base_time,
            priority=8
        ),
        DataNeed(
            module_name="TrendAnalysis",
            symbol="AAPL",
            data_type=DataType.BARS,
            timeframe="5min",
            start_time=base_time - timedelta(hours=4),
            end_time=base_time,
            priority=7
        ),
        
        # Order Flow Module
        DataNeed(
            module_name="OrderFlow",
            symbol="AAPL",
            data_type=DataType.TRADES,
            timeframe="tick",
            start_time=base_time - timedelta(minutes=30),
            end_time=base_time,
            priority=9
        ),
        DataNeed(
            module_name="OrderFlow",
            symbol="AAPL",
            data_type=DataType.QUOTES,
            timeframe="tick",
            start_time=base_time - timedelta(minutes=30),
            end_time=base_time,
            priority=9
        ),
        
        # Volume Analysis Module (overlapping with TrendAnalysis)
        DataNeed(
            module_name="VolumeAnalysis",
            symbol="AAPL",
            data_type=DataType.BARS,
            timeframe="1min",
            start_time=base_time - timedelta(hours=1, minutes=30),
            end_time=base_time + timedelta(minutes=30),
            priority=8
        ),
        DataNeed(
            module_name="BidAskImbalance",
            symbol=symbol,
            data_type=DataType.ALIGNED_TRADES,
            timeframe="tick",
            start_time=start_time,
            end_time=end_time
        ),
    ]
    
    # Register all needs
    aggregator.register_needs(needs)
    
    # Show pre-fetch report
    print(aggregator.create_request_report())
    
    # Get summary
    summary = aggregator.get_pending_needs_summary()
    print("\n\nPENDING NEEDS SUMMARY:")
    print(f"By Module: {dict(summary['by_module'])}")
    print(f"By Symbol: {dict(summary['by_symbol'])}")
    print(f"Priority Distribution: {dict(summary['priority_distribution'])}")
    
    # Fetch all data
    module_data = await aggregator.fetch_all_data()
    
    # Show results
    print("\n\nFETCH RESULTS:")
    print("-" * 40)
    for module_name, data_dict in module_data.items():
        print(f"\n{module_name}:")
        for data_key, df in data_dict.items():
            if not df.empty:
                print(f"  {data_key}: {len(df)} rows [{df.index.min()} to {df.index.max()}]")
            else:
                print(f"  {data_key}: No data available")
    
    # Final stats
    print("\n\nFINAL STATISTICS:")
    final_stats = aggregator.get_stats()
    print(f"Total API calls saved: {final_stats['api_calls_saved']}")
    print(f"Total data points fetched: {final_stats['data_points_fetched']:,}")
    print(f"Cached datasets: {final_stats['cached_datasets']}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_with_modular_data_manager())