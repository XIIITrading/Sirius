# src/services/ranking_engine.py

from typing import List, Dict, Tuple, Optional
from datetime import date, datetime, timezone, timedelta
from dataclasses import dataclass
import math
import pytz
from polygon import RESTClient

from src.models.level_models import PremarketLevel, RankedLevel, RankedLevelWithDetails
from src.services.level_service import LevelService

@dataclass
class MarketData:
    """Market data needed for ranking calculations."""
    ticker: str
    current_price: float  # Live price (for reference/display)
    atr: float  # ATR based on configured timeframe
    benchmark_price: float  # 12:00 UTC price for distance calculations

class RankingEngine:
    """
    Core ranking algorithm for pre-market levels.
    
    Implements the confluence-based scoring system:
    - Type Weight (40%): HVN > Daily MS > H1 OB
    - Confluence Factor (35%): Nearby levels boost score
    - Distance Score (25%): Distance from 12:00 UTC benchmark price
    """
    
    def __init__(self, polygon_api_key: str = None, atr_timeframe: str = "15min"):
        self.level_service = LevelService()
        self.polygon_client = RESTClient(polygon_api_key) if polygon_api_key else None
        self._last_known_price = None  # For fallback calculations
        
        # ATR timeframe configuration
        self.atr_timeframe = atr_timeframe
        self.timeframe_configs = {
            "5min": {
                "multiplier": 5,
                "timespan": "minute",
                "lookback_hours": 3,
                "periods_per_day": 78,  # ~78 five-minute periods in 6.5 hour day
                "description": "5-Minute"
            },
            "15min": {
                "multiplier": 15,
                "timespan": "minute",
                "lookback_hours": 6,
                "periods_per_day": 26,  # ~26 fifteen-minute periods in 6.5 hour day
                "description": "15-Minute"
            },
            "30min": {
                "multiplier": 30,
                "timespan": "minute",
                "lookback_hours": 12,
                "periods_per_day": 13,  # ~13 thirty-minute periods in 6.5 hour day
                "description": "30-Minute"
            },
            "1hour": {
                "multiplier": 1,
                "timespan": "hour",
                "lookback_hours": 24,
                "periods_per_day": 6.5,  # 6.5 hours in a trading day
                "description": "1-Hour"
            }
        }
        
        if atr_timeframe not in self.timeframe_configs:
            raise ValueError(f"Unsupported ATR timeframe: {atr_timeframe}. Supported: {list(self.timeframe_configs.keys())}")
        
        # Type weights (normalized to 40 points max)
        self.type_weights = {
            'HVN': 40.0,        # 100% of type weight
            'Daily MS': 32.0,   # 80% of type weight  
            'H1 OB': 24.0       # 60% of type weight
        }
        
        # Position multipliers (closer positions get higher priority)
        self.use_position_multipliers = True  # Can be turned off if needed
        
        # ATR multiplier for zone calculation
        self.atr_multiplier = 1.0
        
        # Confluence detection range (in ATR units)
        self.confluence_range = 1.5
    
    def calculate_position(self, level_price: float, benchmark_price: float, 
                          all_levels: List[PremarketLevel]) -> str:
        """
        Dynamically calculate position (Above 1, Above 2, Below 1, Below 2)
        based on level's position relative to benchmark price.
        """
        is_above = level_price > benchmark_price
        
        # Get all levels on the same side of benchmark
        if is_above:
            same_side_levels = [l for l in all_levels if l.price > benchmark_price]
            # Sort descending (closest to benchmark first)
            same_side_levels.sort(key=lambda l: l.price)
        else:
            same_side_levels = [l for l in all_levels if l.price < benchmark_price]
            # Sort ascending (closest to benchmark first)
            same_side_levels.sort(key=lambda l: l.price, reverse=True)
        
        # Find position in sorted list
        position_index = next((i for i, l in enumerate(same_side_levels) if l.price == level_price), -1)
        
        if position_index == 0:
            return f"{'Above' if is_above else 'Below'} 1"
        elif position_index == 1:
            return f"{'Above' if is_above else 'Below'} 2"
        else:
            # For levels beyond the first 2, return a generic position
            return f"{'Above' if is_above else 'Below'} {position_index + 1}"
    
    def get_position_multiplier(self, position: str) -> float:
        """
        Get multiplier based on position.
        Only the two closest levels on each side get a bonus.
        """
        if not self.use_position_multipliers:
            return 1.0
            
        position_multipliers = {
            'Above 1': 1.2,
            'Above 2': 1.1,
            'Below 1': 1.2,
            'Below 2': 1.1
        }
        
        return position_multipliers.get(position, 1.0)
    
    def get_benchmark_price_polygon(self, ticker: str, target_date: date) -> float:
        """
        Get the price at 12:00 UTC using Polygon.io API.
        """
        if not self.polygon_client:
            raise ValueError("Polygon API key not provided")
        
        # Convert to datetime for 12:00 UTC
        dt_utc = datetime.combine(target_date, datetime.min.time()).replace(
            hour=12, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )
        
        # Get a range around 12:00 UTC (11:55 to 12:05)
        from_time = dt_utc - timedelta(minutes=5)
        to_time = dt_utc + timedelta(minutes=5)
        
        # Convert to Unix timestamps (milliseconds)
        from_timestamp = int(from_time.timestamp() * 1000)
        to_timestamp = int(to_time.timestamp() * 1000)
        
        try:
            # Get minute bars around 12:00 UTC using timestamps
            minute_bars = []
            for bar in self.polygon_client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="minute",
                from_=from_timestamp,
                to=to_timestamp,
                limit=10
            ):
                minute_bars.append(bar)
            
            if minute_bars:
                # Find the bar closest to 12:00 UTC
                target_ts = int(dt_utc.timestamp() * 1000)
                closest_bar = min(minute_bars, key=lambda x: abs(x.timestamp - target_ts))
                price = closest_bar.close
                self._last_known_price = price
                print(f"   Found price at 12:00 UTC: ${price:.2f}")
                return price
            else:
                # If no minute data, try to get the daily data
                print(f"   No minute data at 12:00 UTC, trying daily data")
                date_str = target_date.strftime('%Y-%m-%d')
                
                try:
                    daily = self.polygon_client.get_daily_open_close_agg(ticker, date_str)
                    if daily and daily.close:
                        price = daily.close
                        self._last_known_price = price
                        print(f"   Using daily close price: ${price:.2f}")
                        return price
                except:
                    # Try listing daily aggregates
                    daily_bars = []
                    for bar in self.polygon_client.list_aggs(
                        ticker=ticker,
                        multiplier=1,
                        timespan="day",
                        from_=date_str,
                        to=date_str,
                        limit=1
                    ):
                        daily_bars.append(bar)
                    
                    if daily_bars:
                        price = daily_bars[0].close
                        self._last_known_price = price
                        print(f"   Using daily bar close: ${price:.2f}")
                        return price
                    
                raise ValueError(f"No price data found for {ticker} on {target_date}")
                
        except Exception as e:
            print(f"Error fetching price data: {e}")
            raise
    
    def calculate_atr(self, ticker: str, target_date: date) -> float:
        """
        Calculate 14-period ATR using the configured timeframe.
        """
        if not self.polygon_client:
            raise ValueError("Polygon API key not provided")
        
        config = self.timeframe_configs[self.atr_timeframe]
        
        try:
            # We need data ending at 12:00 UTC on target date
            dt_utc_end = datetime.combine(target_date, datetime.min.time()).replace(
                hour=12, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
            )
            
            # Start time based on configured lookback
            dt_utc_start = dt_utc_end - timedelta(hours=config['lookback_hours'])
            
            # Convert to Unix timestamps (milliseconds)
            from_timestamp = int(dt_utc_start.timestamp() * 1000)
            to_timestamp = int(dt_utc_end.timestamp() * 1000)
            
            print(f"   Fetching {config['description']} data from {dt_utc_start} to {dt_utc_end}")
            
            # Get bars using timestamps
            bars = []
            try:
                for bar in self.polygon_client.list_aggs(
                    ticker=ticker,
                    multiplier=config['multiplier'],
                    timespan=config['timespan'],
                    from_=from_timestamp,
                    to=to_timestamp,
                    limit=100
                ):
                    bars.append(bar)
            except Exception as e:
                print(f"   Error with timestamps, trying date format: {e}")
                # Fallback to date strings
                from_date = dt_utc_start.strftime('%Y-%m-%d')
                to_date = dt_utc_end.strftime('%Y-%m-%d')
                
                for bar in self.polygon_client.list_aggs(
                    ticker=ticker,
                    multiplier=config['multiplier'],
                    timespan=config['timespan'],
                    from_=from_date,
                    to=to_date,
                    limit=200
                ):
                    bars.append(bar)
            
            print(f"   Retrieved {len(bars)} {config['description'].lower()} bars")
            
            if len(bars) < 15:  # Need at least 15 bars (14 + 1 for previous close)
                # Try extending the range
                extended_lookback = config['lookback_hours'] * 4
                print(f"   Insufficient data, extending range to {extended_lookback} hours")
                dt_utc_start = dt_utc_end - timedelta(hours=extended_lookback)
                from_timestamp = int(dt_utc_start.timestamp() * 1000)
                
                bars = []
                for bar in self.polygon_client.list_aggs(
                    ticker=ticker,
                    multiplier=config['multiplier'],
                    timespan=config['timespan'],
                    from_=from_timestamp,
                    to=to_timestamp,
                    limit=200
                ):
                    bars.append(bar)
                
                print(f"   Retrieved {len(bars)} {config['description'].lower()} bars with extended range")
            
            if len(bars) < 15:
                print(f"   Still insufficient {config['description'].lower()} data, falling back to daily ATR")
                return self.calculate_daily_atr_fallback(ticker, target_date)
            
            # Sort bars by timestamp to ensure correct order
            bars.sort(key=lambda x: x.timestamp)
            
            # Only use bars up to 12:00 UTC on target date
            target_timestamp = int(dt_utc_end.timestamp() * 1000)
            bars = [bar for bar in bars if bar.timestamp <= target_timestamp]
            
            # Calculate True Range for each period
            true_ranges = []
            for i in range(1, len(bars)):
                high_low = bars[i].high - bars[i].low
                high_close = abs(bars[i].high - bars[i-1].close)
                low_close = abs(bars[i].low - bars[i-1].close)
                true_range = max(high_low, high_close, low_close)
                true_ranges.append(true_range)
            
            # Calculate 14-period ATR (using the last 14 true ranges)
            if len(true_ranges) >= 14:
                atr = sum(true_ranges[-14:]) / 14
            else:
                # Use all available true ranges
                atr = sum(true_ranges) / len(true_ranges)
            
            print(f"   {config['description']} ATR (14-period): ${atr:.2f}")
            return round(atr, 2)
            
        except Exception as e:
            print(f"Error calculating {config['description'].lower()} ATR: {e}")
            # Fallback to daily ATR
            return self.calculate_daily_atr_fallback(ticker, target_date)
    
    def calculate_daily_atr_fallback(self, ticker: str, target_date: date) -> float:
        """
        Fallback method to calculate ATR using daily bars, then divide to approximate
        the configured timeframe.
        """
        try:
            print("   Calculating daily ATR as fallback...")
            
            config = self.timeframe_configs[self.atr_timeframe]
            
            end_date = target_date
            start_date = end_date - timedelta(days=30)
            
            # Use simple date strings for daily data
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            daily_bars = []
            for bar in self.polygon_client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=from_date,
                to=to_date,
                limit=30
            ):
                daily_bars.append(bar)
            
            if len(daily_bars) >= 14:
                # Calculate True Range for each day
                true_ranges = []
                for i in range(1, len(daily_bars)):
                    high_low = daily_bars[i].high - daily_bars[i].low
                    high_close = abs(daily_bars[i].high - daily_bars[i-1].close)
                    low_close = abs(daily_bars[i].low - daily_bars[i-1].close)
                    true_range = max(high_low, high_close, low_close)
                    true_ranges.append(true_range)
                
                # Calculate 14-period daily ATR
                daily_atr = sum(true_ranges[-14:]) / 14
                # Approximate timeframe ATR
                timeframe_atr = daily_atr / config['periods_per_day']
            else:
                # Ultimate fallback: use simple average range
                avg_range = sum(bar.high - bar.low for bar in daily_bars) / len(daily_bars)
                timeframe_atr = avg_range / config['periods_per_day']
            
            print(f"   Approximated {config['description']} ATR: ${timeframe_atr:.2f}")
            return round(timeframe_atr, 2)
            
        except Exception as e:
            print(f"Error in daily ATR fallback: {e}")
            # Ultimate fallback - return a reasonable default based on ticker price
            if hasattr(self, '_last_known_price') and self._last_known_price:
                # 0.5% of price as a reasonable default
                default_atr = self._last_known_price * 0.005
                print(f"   Using default ATR (0.5% of price): ${default_atr:.2f}")
                return round(default_atr, 2)
            else:
                raise ValueError("Unable to calculate ATR")
    
    def get_market_data_polygon(self, ticker: str, target_date: date) -> MarketData:
        """
        Fetch all market data using Polygon API.
        """
        if not self.polygon_client:
            raise ValueError("Polygon API key not provided")
        
        try:
            # Get benchmark price at 12:00 UTC
            benchmark_price = self.get_benchmark_price_polygon(ticker, target_date)
            
            # Get current/latest price
            try:
                snapshot = self.polygon_client.get_snapshot_all("stocks")
                ticker_data = next((s for s in snapshot if s.ticker == ticker), None)
                current_price = ticker_data.day.close if ticker_data else benchmark_price
            except:
                # If snapshot fails, use benchmark price
                current_price = benchmark_price
            
            # Calculate ATR using configured timeframe
            atr = self.calculate_atr(ticker, target_date)
            
            return MarketData(
                ticker=ticker,
                current_price=current_price,
                atr=atr,
                benchmark_price=benchmark_price
            )
            
        except Exception as e:
            print(f"Error fetching market data for {ticker}: {e}")
            raise
    
    def rank_levels(self, ticker: str, target_date: date, market_data: MarketData) -> List[RankedLevel]:
        """
        Main ranking method - processes all levels for a ticker.
        
        Args:
            ticker: Stock symbol
            target_date: Date to process
            market_data: Current price, ATR, and 12:00 UTC benchmark price
            
        Returns:
            List of RankedLevel objects sorted by rank
        """
        config = self.timeframe_configs[self.atr_timeframe]
        
        print(f"\n🔄 Starting ranking process for {ticker}")
        print(f"   Current Price: ${market_data.current_price:.2f}")
        print(f"   12:00 UTC Benchmark: ${market_data.benchmark_price:.2f}")
        print(f"   {config['description']} ATR: ${market_data.atr:.2f}")
        
        # Clear any existing rankings for this ticker/date
        self.level_service.clear_existing_rankings(ticker, target_date)
        
        # Get all premarket levels
        levels = self.level_service.get_premarket_levels_for_analysis(ticker, target_date)
        
        if not levels:
            print(f"⚠️  No levels found for {ticker} on {target_date}")
            return []
        
        # Calculate scores for each level
        scored_levels = []
        for level in levels:
            # Calculate position dynamically
            position = self.calculate_position(level.price, market_data.benchmark_price, levels)
            
            score_components = self._calculate_level_score(level, levels, market_data)
            total_score = sum(score_components.values())
            
            # Apply position multiplier
            position_multiplier = self.get_position_multiplier(position)
            total_score *= position_multiplier
            
            scored_levels.append({
                'level': level,
                'score': total_score,
                'components': score_components,
                'position': position,
                'position_multiplier': position_multiplier
            })
        
        # Sort by score (highest first)
        scored_levels.sort(key=lambda x: x['score'], reverse=True)
        
        # Create ranked levels
        ranked_levels = []
        for rank, scored in enumerate(scored_levels, 1):
            level = scored['level']
            position = scored['position']
            
            # Calculate zone boundaries using configured ATR
            zone_width = market_data.atr * self.atr_multiplier
            zone_high = level.price + zone_width
            zone_low = level.price - zone_width
            
            # Generate TradingView variable name using calculated position
            tv_variable = self._generate_tv_variable(level.level_type, position)
            
            # Create RankedLevel object
            ranked_level = RankedLevel(
                premarket_level_id=level.id,
                date=target_date,
                ticker=ticker,
                rank=rank,
                confluence_score=round(scored['score'], 2),
                zone_high=round(zone_high, 2),
                zone_low=round(zone_low, 2),
                tv_variable=tv_variable,
                current_price=market_data.current_price,
                atr_value=market_data.atr
            )
            
            ranked_levels.append(ranked_level)
            
            # Print ranking details
            print(f"\n   Rank #{rank}: {level.level_type} {position} @ ${level.price:.2f}")
            print(f"   Score: {scored['score']:.2f} (multiplier: {scored['position_multiplier']})")
            print(f"   Components: {scored['components']}")
            print(f"   Zone: ${zone_low:.2f} - ${zone_high:.2f} (±${zone_width:.2f})")
            print(f"   Distance from benchmark: ${abs(level.price - market_data.benchmark_price):.2f}")
        
        # Save to database
        saved_levels = self.level_service.save_ranked_levels(ranked_levels)
        
        print(f"\n✅ Ranking complete! Processed {len(saved_levels)} levels")
        return saved_levels
    
    def _calculate_level_score(self, level: PremarketLevel, all_levels: List[PremarketLevel], 
                              market_data: MarketData) -> Dict[str, float]:
        """
        Calculate individual score components for a level.
        
        Returns dict with score breakdown:
        - type_weight: Score from level type (40% max)
        - confluence_score: Points for nearby levels (35% max)
        - distance_score: Proximity to 12:00 UTC benchmark price (25% max)
        """
        components = {}
        
        # 1. Type Weight (40% of total)
        components['type_weight'] = self.type_weights.get(level.level_type, 0)
        
        # 2. Confluence Factor (35% of total)
        components['confluence_score'] = self._calculate_confluence(level, all_levels, market_data.atr)
        
        # 3. Distance Score (25% of total) - NOW USES BENCHMARK PRICE
        components['distance_score'] = self._calculate_distance_score(
            level.price, 
            market_data.benchmark_price
        )
        
        return components
    
    def _calculate_confluence(self, target_level: PremarketLevel, all_levels: List[PremarketLevel], 
                             atr: float) -> float:
        """
        Calculate confluence score based on nearby levels.
        
        Confluence rules:
        - Levels within 1.5 ATR are considered confluent
        - Different level types in confluence score higher
        - Maximum 35 points
        """
        confluence_distance = atr * self.confluence_range
        confluence_count = 0
        different_type_bonus = 0
        
        for level in all_levels:
            if level.id == target_level.id:
                continue
                
            # Check if within confluence range
            distance = abs(level.price - target_level.price)
            if distance <= confluence_distance:
                confluence_count += 1
                
                # Bonus for different type confluence
                if level.level_type != target_level.level_type:
                    different_type_bonus += 2
        
        # Calculate score (max 35)
        base_score = min(confluence_count * 4, 20)  # Max 20 from count
        bonus_score = min(different_type_bonus, 15)  # Max 15 from type diversity
        
        return min(base_score + bonus_score, 35)
    
    def _calculate_distance_score(self, level_price: float, benchmark_price: float) -> float:
        """
        Calculate distance score based on proximity to 12:00 UTC benchmark price.
        
        Closer levels get higher scores (max 25 points).
        Uses exponential decay based on percentage distance.
        """
        # Calculate percentage distance from benchmark
        pct_distance = abs(level_price - benchmark_price) / benchmark_price * 100
        
        # Exponential decay scoring
        # At 0% distance: 25 points
        # At 5% distance: ~12.5 points  
        # At 10% distance: ~6.25 points
        score = 25 * math.exp(-pct_distance / 5)
        
        return round(score, 2)
    
    def _generate_tv_variable(self, level_type: str, position: str) -> str:
        """Generate TradingView variable name."""
        type_map = {
            'HVN': 'hvn',
            'Daily MS': 'ds',
            'H1 OB': 'ob'
        }
        
        # Extract position code from dynamic position string
        if 'Above 1' in position:
            pos_code = 'a1'
        elif 'Above 2' in position:
            pos_code = 'a2'
        elif 'Below 1' in position:
            pos_code = 'b1'
        elif 'Below 2' in position:
            pos_code = 'b2'
        else:
            # For positions beyond 2, use generic codes
            if 'Above' in position:
                pos_code = 'ax'
            elif 'Below' in position:
                pos_code = 'bx'
            else:
                pos_code = 'unk'
        
        type_code = type_map.get(level_type, 'unk')
        
        return f"{type_code}_{pos_code}"
    
    def process_multiple_tickers(self, tickers: List[str], target_date: date, 
                                market_data_dict: Dict[str, MarketData] = None) -> Dict[str, List[RankedLevel]]:
        """
        Process multiple tickers in batch.
        
        Args:
            tickers: List of ticker symbols
            target_date: Date to process
            market_data_dict: Optional pre-fetched market data. If None, fetches from Polygon.
            
        Returns:
            Dict of ticker -> List[RankedLevel]
        """
        results = {}
        
        for ticker in tickers:
            try:
                # Get market data if not provided
                if market_data_dict and ticker in market_data_dict:
                    market_data = market_data_dict[ticker]
                else:
                    print(f"Fetching market data from Polygon for {ticker}...")
                    market_data = self.get_market_data_polygon(ticker, target_date)
                
                # Run ranking
                ranked_levels = self.rank_levels(ticker, target_date, market_data)
                results[ticker] = ranked_levels
                
            except Exception as e:
                print(f"⚠️  Error processing {ticker}: {e}")
                continue
        
        return results