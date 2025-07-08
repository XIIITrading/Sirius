# modules/data/polygon_bridge.py
"""
Module: Polygon-HVN Bridge
Purpose: Connect Polygon.io data fetching to HVN calculation engine
Features: Historical data loading, real-time streaming, automatic HVN updates
Performance: Optimized for intraday trading with minimal latency
Architecture: Data adapter layer - filters and prepares data for calculation engines
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from collections import deque
import logging
import sys
import os

# Polygon Connection
from polygon import DataFetcher, PolygonWebSocketClient
from polygon.config import PolygonConfig

# Import HVN components
from market_review.calculations.volume.hvn_engine import HVNEngine, HVNResult
from market_review.calculations.volume.volume_profile import PriceLevel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LiveHVNState:
    """Container for live HVN state and recent data"""
    symbol: str
    last_calculation: datetime
    hvn_result: HVNResult
    recent_bars: pd.DataFrame  # Last N bars for trend analysis
    current_price: float
    last_update: datetime
    end_datetime: Optional[datetime] = None
    is_near_hvn: bool = False
    approaching_levels: List[PriceLevel] = None


class PolygonHVNBridge:
    """
    Bridge between Polygon.io data and HVN calculations.
    Handles both historical analysis and real-time updates.
    
    This is the DATA ADAPTER LAYER that:
    1. Receives data requests with specific parameters
    2. Fetches raw data from Polygon.io
    3. Filters and validates data according to request
    4. Passes clean data to calculation engines
    5. Returns complete analysis results
    """
    
    def __init__(self,
                 hvn_levels: int = 100,
                 hvn_percentile: float = 80.0,
                 lookback_days: int = 14,
                 update_interval_minutes: int = 5,
                 cache_enabled: bool = True):
        """
        Initialize the Polygon-HVN bridge.
        
        Args:
            hvn_levels: Number of price levels for volume profile
            hvn_percentile: Percentile threshold for HVN identification
            lookback_days: Days of historical data for HVN calculation
            update_interval_minutes: How often to recalculate HVN
            cache_enabled: Use Polygon's local cache
        """
        # Initialize components
        try:
            self.fetcher = DataFetcher(config=PolygonConfig({'cache_enabled': cache_enabled}))
        except Exception as e:
            logger.error(f"Failed to initialize DataFetcher: {e}")
            raise
            
        self.ws_client = None
        self.hvn_engine = HVNEngine(
            levels=hvn_levels,
            percentile_threshold=hvn_percentile
        )
        
        # Configuration
        self.lookback_days = lookback_days
        self.update_interval = timedelta(minutes=update_interval_minutes)
        
        # State tracking
        self.live_states: Dict[str, LiveHVNState] = {}
        self.callbacks: Dict[str, List[Callable]] = {
            'hvn_update': [],
            'proximity_alert': [],
            'volume_surge': []
        }
        
        # Real-time data buffers
        self.tick_buffers: Dict[str, deque] = {}
        self.aggregation_tasks: Dict[str, asyncio.Task] = {}
        
    def calculate_hvn(self, 
                     symbol: str,
                     end_date: Optional[datetime] = None,
                     timeframe: str = '5min') -> LiveHVNState:
        """
        Calculate HVN for a symbol using historical data.
        
        THIS IS THE MAIN DATA FILTERING METHOD
        - Fetches raw data from Polygon
        - Filters to exact date range requested
        - Prepares data for calculation engines
        - Returns complete analysis
        
        Args:
            symbol: Stock ticker
            end_date: End date for calculation (default: now)
            timeframe: Data timeframe for analysis
            
        Returns:
            LiveHVNState with complete HVN analysis
        """
        # Handle default end_date
        if end_date is None:
            end_date = datetime.now()
            logger.info(f"Using current time as end_date: {end_date}")
        
        # Ensure end_date is timezone-aware (UTC) for comparison with Polygon data
        if end_date.tzinfo is None:
            # If naive datetime, assume it's UTC
            end_date = end_date.replace(tzinfo=timezone.utc)
            logger.info(f"Converting naive datetime to UTC: {end_date}")
        
        # Calculate start date based on lookback period
        start_date = end_date - timedelta(days=self.lookback_days)
        
        # Ensure start_date is also timezone-aware
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        
        logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")
        
        try:
            # Fetch raw data from Polygon
            df = self.fetcher.fetch_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
                validate=True,
                fill_gaps=True
            )
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")
        
        if df.empty:
            raise ValueError(f"No data available for {symbol} in the requested date range")
        
        # ===== CRITICAL DATA FILTERING SECTION =====
        # This ensures we provide EXACTLY what was requested
        
        # Check if the dataframe index is timezone-aware
        if df.index.tz is not None:
            # If df has timezone info, ensure our dates match
            if end_date.tzinfo is None:
                end_date = pd.Timestamp(end_date).tz_localize('UTC')
            else:
                end_date = pd.Timestamp(end_date)
                
            if start_date.tzinfo is None:
                start_date = pd.Timestamp(start_date).tz_localize('UTC')
            else:
                start_date = pd.Timestamp(start_date)
        else:
            # If df doesn't have timezone info, use naive datetimes
            end_date = pd.Timestamp(end_date).tz_localize(None)
            start_date = pd.Timestamp(start_date).tz_localize(None)
        
        # Filter to ensure data doesn't exceed requested end_date
        original_count = len(df)
        df = df[df.index <= end_date]
        
        # Also filter start date to ensure exact window
        df = df[df.index >= start_date]
        
        filtered_count = len(df)
        if filtered_count == 0:
            raise ValueError(f"No data available for {symbol} after filtering to date range")
        
        logger.info(f"Data filtering: {original_count} bars fetched, {filtered_count} bars after filtering")
        logger.info(f"Filtered data range: {df.index[0]} to {df.index[-1]}")
        
        # Ensure proper column names for calculation engines
        df = df.rename(columns={
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        # Add timestamp column (calculation engines expect this)
        df['timestamp'] = df.index
        
        # Validate data quality before passing to calculation engine
        if df['volume'].sum() == 0:
            logger.warning(f"No volume data for {symbol} in the requested period")
        
        # Pass CLEAN, FILTERED data to calculation engine
        try:
            hvn_result = self.hvn_engine.analyze(df)
        except Exception as e:
            logger.error(f"HVN calculation failed for {symbol}: {e}")
            raise ValueError(f"HVN calculation failed: {str(e)}")
        
        # Get current price from the FILTERED data (last available price)
        current_price = df['close'].iloc[-1]
        
        # Calculate ATR for proximity detection
        try:
            atr = self.hvn_engine.calculate_atr(df)
        except Exception as e:
            logger.warning(f"ATR calculation failed: {e}, using default")
            atr = df['high'].iloc[-20:].mean() - df['low'].iloc[-20:].mean()
        
        # Check proximity to HVN clusters
        proximity_map = {}
        if hvn_result.clusters:
            proximity_map = self.hvn_engine.check_proximity(
                current_price, hvn_result.clusters, atr
            )
        
        # Get approaching levels
        approaching_levels = []
        if hvn_result.filtered_levels:
            approaching_levels = self._get_approaching_levels(
                current_price, hvn_result.filtered_levels, atr
            )
        
        # Create state with properly filtered data
        state = LiveHVNState(
            symbol=symbol,
            last_calculation=datetime.now(),
            hvn_result=hvn_result,
            recent_bars=df, # Keep all the bars pulled for calculation to display
            current_price=current_price,
            last_update=datetime.now(),
            end_datetime=end_date.to_pydatetime() if isinstance(end_date, pd.Timestamp) else end_date,  # Convert back to datetime
            is_near_hvn=any(proximity_map.values()) if proximity_map else False,
            approaching_levels=approaching_levels
        )
        
        # Cache the state for this symbol
        self.live_states[symbol] = state
        
        logger.info(f"HVN calculation complete for {symbol}: {len(hvn_result.filtered_levels)} HVN levels found")
        
        return state
    
    def _get_approaching_levels(self,
                               current_price: float,
                               levels: List[PriceLevel],
                               atr: float,
                               threshold_multiplier: float = 2.0) -> List[PriceLevel]:
        """Get HVN levels within threshold distance of current price."""
        threshold = atr * threshold_multiplier
        approaching = []
        
        for level in levels:
            # Calculate minimum distance to level
            distance = min(
                abs(current_price - level.low),
                abs(current_price - level.high),
                abs(current_price - level.center)
            )
            
            if distance <= threshold:
                approaching.append(level)
                
        # Sort by distance to current price
        return sorted(approaching, key=lambda x: abs(x.center - current_price))
    
    async def start_live_updates(self, symbols: Union[str, List[str]]):
        """
        Start real-time updates for symbols.
        
        Args:
            symbols: Single symbol or list of symbols
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            
        # Calculate initial HVN for all symbols
        for symbol in symbols:
            if symbol not in self.live_states:
                try:
                    self.calculate_hvn(symbol)
                except Exception as e:
                    logger.error(f"Failed to calculate initial HVN for {symbol}: {e}")
        
        # Initialize WebSocket client
        try:
            self.ws_client = PolygonWebSocketClient()
            await self.ws_client.connect()
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            raise
        
        # Initialize tick buffers
        for symbol in symbols:
            self.tick_buffers[symbol] = deque(maxlen=1000)
        
        # Subscribe to trades and minute aggregates
        try:
            await self.ws_client.subscribe(
                symbols=symbols,
                channels=['T', 'AM'],  # Trades and Minute Aggregates
                callback=self._handle_live_data
            )
        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {e}")
            raise
        
        # Start aggregation tasks
        for symbol in symbols:
            self.aggregation_tasks[symbol] = asyncio.create_task(
                self._aggregation_loop(symbol)
            )
        
        # Start listening
        logger.info(f"Starting live updates for {symbols}")
        try:
            await self.ws_client.listen()
        except Exception as e:
            logger.error(f"WebSocket listening error: {e}")
            await self.stop()
            raise
    
    async def _handle_live_data(self, data: Dict):
        """Handle incoming WebSocket data."""
        try:
            event_type = data.get('event_type')
            symbol = data.get('symbol')
            
            if not symbol or symbol not in self.tick_buffers:
                return
                
            if event_type == 'trade':
                # Store trade for aggregation
                self.tick_buffers[symbol].append({
                    'price': data['price'],
                    'size': data['size'],
                    'timestamp': datetime.fromtimestamp(data['timestamp'] / 1000)
                })
                
                # Update current price
                if symbol in self.live_states:
                    self.live_states[symbol].current_price = data['price']
                    self.live_states[symbol].last_update = datetime.now()
                    
                    # Check proximity alerts
                    self._check_proximity_alerts(symbol)
                    
            elif event_type == 'aggregate':
                # Handle minute bars
                await self._update_recent_bars(symbol, data)
                
        except Exception as e:
            logger.error(f"Error handling live data: {e}")
    
    async def _update_recent_bars(self, symbol: str, bar_data: Dict):
        """Update recent bars with new minute data."""
        if symbol not in self.live_states:
            return
            
        try:
            state = self.live_states[symbol]
            
            # Create new bar
            new_bar = pd.DataFrame([{
                'timestamp': pd.Timestamp.fromtimestamp(bar_data['timestamp'] / 1000, tz='UTC'),
                'open': bar_data['open'],
                'high': bar_data['high'],
                'low': bar_data['low'],
                'close': bar_data['close'],
                'volume': bar_data['volume']
            }])
            new_bar.set_index('timestamp', inplace=True)
            
            # Append to recent bars
            state.recent_bars = pd.concat([state.recent_bars, new_bar]).tail(1000)
            
            # Check for volume surge
            self._check_volume_surge(symbol, bar_data['volume'])
            
        except Exception as e:
            logger.error(f"Error updating recent bars for {symbol}: {e}")
    
    async def _aggregation_loop(self, symbol: str):
        """Periodic HVN recalculation loop."""
        while True:
            try:
                await asyncio.sleep(self.update_interval.total_seconds())
                
                # Check if recalculation needed
                state = self.live_states.get(symbol)
                if state:
                    time_since_calc = datetime.now() - state.last_calculation
                    if time_since_calc >= self.update_interval:
                        logger.info(f"Recalculating HVN for {symbol}")
                        new_state = self.calculate_hvn(symbol)
                        
                        # Trigger update callbacks
                        for callback in self.callbacks['hvn_update']:
                            try:
                                callback(symbol, new_state)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                                
            except asyncio.CancelledError:
                logger.info(f"Aggregation loop cancelled for {symbol}")
                break
            except Exception as e:
                logger.error(f"Error in aggregation loop for {symbol}: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    def _check_proximity_alerts(self, symbol: str):
        """Check and trigger proximity alerts."""
        state = self.live_states.get(symbol)
        if not state or not state.hvn_result.clusters:
            return
            
        try:
            # Recalculate proximity
            atr = self.hvn_engine.calculate_atr(state.recent_bars)
            proximity_map = self.hvn_engine.check_proximity(
                state.current_price,
                state.hvn_result.clusters,
                atr
            )
            
            was_near = state.is_near_hvn
            state.is_near_hvn = any(proximity_map.values())
            
            # Trigger alert if newly approaching
            if state.is_near_hvn and not was_near:
                for callback in self.callbacks['proximity_alert']:
                    try:
                        callback(symbol, state, proximity_map)
                    except Exception as e:
                        logger.error(f"Proximity alert callback error: {e}")
                        
        except Exception as e:
            logger.error(f"Error checking proximity for {symbol}: {e}")
    
    def _check_volume_surge(self, symbol: str, current_volume: float):
        """Check for volume surges near HVN levels."""
        state = self.live_states.get(symbol)
        if not state or not state.is_near_hvn:
            return
            
        try:
            # Calculate average volume
            if len(state.recent_bars) >= 20:
                avg_volume = state.recent_bars['volume'].rolling(20).mean().iloc[-1]
                
                # Check for surge (2x average)
                if current_volume > avg_volume * 2:
                    for callback in self.callbacks['volume_surge']:
                        try:
                            callback(symbol, current_volume, avg_volume)
                        except Exception as e:
                            logger.error(f"Volume surge callback error: {e}")
                            
        except Exception as e:
            logger.error(f"Error checking volume surge for {symbol}: {e}")
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for events."""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.info(f"Registered callback for {event_type}")
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def get_hvn_levels_near_price(self, 
                                 symbol: str,
                                 price_range_percent: float = 2.0) -> List[PriceLevel]:
        """
        Get HVN levels within percent range of current price.
        
        Args:
            symbol: Stock ticker
            price_range_percent: Percentage range to check
            
        Returns:
            List of nearby HVN levels
        """
        state = self.live_states.get(symbol)
        if not state or not state.hvn_result.filtered_levels:
            return []
            
        current_price = state.current_price
        range_size = current_price * (price_range_percent / 100)
        
        nearby_levels = []
        for level in state.hvn_result.filtered_levels:
            if (current_price - range_size <= level.center <= current_price + range_size):
                nearby_levels.append(level)
                
        return sorted(nearby_levels, key=lambda x: abs(x.center - current_price))
    
    def get_volume_trend_at_level(self, symbol: str, level: PriceLevel) -> str:
        """Get volume trend approaching a specific level."""
        state = self.live_states.get(symbol)
        if not state:
            return 'unknown'
            
        try:
            return self.hvn_engine.calculate_volume_trend(
                state.recent_bars,
                level.center,
                bars=14
            )
        except Exception as e:
            logger.error(f"Error calculating volume trend: {e}")
            return 'unknown'
    
    async def stop(self):
        """Stop all live updates and cleanup."""
        logger.info("Stopping Polygon-HVN bridge...")
        
        # Cancel aggregation tasks
        for symbol, task in self.aggregation_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info(f"Cancelled aggregation task for {symbol}")
            
        # Close WebSocket
        if self.ws_client:
            try:
                await self.ws_client.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting WebSocket: {e}")
            
        logger.info("Polygon-HVN bridge stopped")


# Example usage and testing
async def main():
    """Example usage of PolygonHVNBridge"""
    
    # Initialize bridge
    bridge = PolygonHVNBridge(
        hvn_levels=100,
        hvn_percentile=80.0,
        lookback_days=14,
        update_interval_minutes=5
    )
    
    # Test 1: Live data (current time)
    print("Test 1: Calculating HVN with live data...")
    state = bridge.calculate_hvn('TSLA')
    
    print(f"\nLive HVN Analysis:")
    print(f"  Current price: ${state.current_price:.2f}")
    print(f"  Price range: ${state.hvn_result.price_range[0]:.2f} - ${state.hvn_result.price_range[1]:.2f}")
    print(f"  HVN clusters found: {len(state.hvn_result.clusters)}")
    print(f"  Data end time: {state.end_datetime}")
    
    # Test 2: Historical data (specific end time)
    print("\n\nTest 2: Calculating HVN with historical data...")
    historical_end = datetime.now() - timedelta(days=1)
    state2 = bridge.calculate_hvn('TSLA', end_date=historical_end)
    
    print(f"\nHistorical HVN Analysis:")
    print(f"  Current price: ${state2.current_price:.2f}")
    print(f"  Price range: ${state2.hvn_result.price_range[0]:.2f} - ${state2.hvn_result.price_range[1]:.2f}")
    print(f"  HVN clusters found: {len(state2.hvn_result.clusters)}")
    print(f"  Data end time: {state2.end_datetime}")
    print(f"  Actual data range: {state2.recent_bars.index[0]} to {state2.recent_bars.index[-1]}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())