"""
M5 Market Structure Plugin Implementation
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import pandas as pd
import logging

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backtest.plugins.base_plugin import BacktestPlugin
from modules.calculations.market_structure.m5_market_structure import (
    M5MarketStructureAnalyzer, 
    MarketStructureSignal
)
from data.polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)


class M5MarketStructurePlugin(BacktestPlugin):
    """Plugin for 5-minute market structure analysis using fractals"""
    
    def __init__(self):
        self.data_manager = PolygonDataManager()
        self.config = {
            'fractal_length': 3,       # Smaller for 5-min timeframe
            'buffer_size': 100,        # 100 5-min bars = 500 minutes
            'min_candles_required': 15,
            'lookback_minutes': 600    # 10 hours of 5-min data
        }
        
    @property
    def name(self) -> str:
        return "5-Min Market Structure"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def get_config(self) -> Dict[str, Any]:
        """Get plugin configuration"""
        return self.config.copy()
    
    def validate_inputs(self, symbol: str, entry_time: datetime, direction: str) -> bool:
        """Validate input parameters"""
        if not symbol or not isinstance(symbol, str):
            return False
        
        if not isinstance(entry_time, datetime):
            return False
            
        if direction not in ['LONG', 'SHORT']:
            return False
            
        return True
    
    async def run_analysis(self, symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
        """
        Run the complete M5 market structure analysis.
        """
        try:
            # 1. Fetch required data (1-minute bars)
            bars_1min = await self._fetch_data(symbol, entry_time)
            
            if bars_1min.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # 2. Aggregate to 5-minute bars
            bars_5min = self._aggregate_to_5min(bars_1min)
            
            logger.info(f"Aggregated {len(bars_1min)} 1-min bars to {len(bars_5min)} 5-min bars")
            
            # 3. Run analysis
            analyzer = M5MarketStructureAnalyzer(
                fractal_length=self.config['fractal_length'],
                buffer_size=self.config['buffer_size'],
                min_candles_required=self.config['min_candles_required']
            )
            
            # Process historical candles up to entry time
            signal = self._process_bars(analyzer, symbol, bars_5min, entry_time)
            
            # 4. Format results
            return self._format_results(signal, entry_time, direction)
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise
    
    async def _fetch_data(self, symbol: str, entry_time: datetime) -> pd.DataFrame:
        """Fetch required 1-minute bar data"""
        # Calculate time range
        start_time = entry_time - timedelta(minutes=self.config['lookback_minutes'])
        
        logger.info(f"Fetching {symbol} 1-min data from {start_time} to {entry_time}")
        
        # Use PolygonDataManager to fetch data
        bars = await self.data_manager.load_bars(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time,
            timeframe='1min',
            use_cache=True
        )
        
        return bars
    
    def _aggregate_to_5min(self, bars_1min: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 1-minute bars to 5-minute bars"""
        if bars_1min.empty:
            return pd.DataFrame()
        
        # Resample to 5-minute bars
        bars_5min = bars_1min.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return bars_5min
    
    def _process_bars(self, analyzer: M5MarketStructureAnalyzer, 
                     symbol: str, bars: pd.DataFrame, 
                     entry_time: datetime) -> Optional[MarketStructureSignal]:
        """Process bars and get signal at entry time"""
        candles = []
        
        # Convert bars to candle format
        for timestamp, row in bars.iterrows():
            # Stop at entry time
            if timestamp >= entry_time:
                break
                
            candle_dict = {
                't': timestamp,
                'o': float(row['open']),
                'h': float(row['high']),
                'l': float(row['low']),
                'c': float(row['close']),
                'v': float(row['volume'])
            }
            candles.append(candle_dict)
        
        logger.info(f"Processing {len(candles)} 5-minute candles")
        
        # Process all historical candles
        signal = analyzer.process_historical_candles(symbol, candles)
        
        # If no signal, get current analysis
        if not signal:
            signal = analyzer.get_current_analysis(symbol)
            
        return signal
    
    def _format_results(self, signal: Optional[MarketStructureSignal], 
                       entry_time: datetime, direction: str) -> Dict[str, Any]:
        """Format results for display"""
        if not signal:
            # No signal - return neutral
            return self._create_neutral_result(entry_time)
        
        # Map signal direction
        direction_map = {
            'BULL': 'BULLISH',
            'BEAR': 'BEARISH',
            'NEUTRAL': 'NEUTRAL'
        }
        
        signal_direction = direction_map.get(signal.signal, 'NEUTRAL')
        metrics = signal.metrics
        
        # Build display data
        display_rows = []
        
        # Main info
        display_rows.append(['Timeframe', '5-Minute'])
        display_rows.append(['Structure Type', signal.structure_type or 'None'])
        display_rows.append(['Current Trend', metrics.get('current_trend', 'NEUTRAL')])
        display_rows.append(['Signal Strength', f"{signal.strength:.0f}%"])
        
        # Fractal info
        if metrics.get('last_high_fractal'):
            display_rows.append(['Last High Fractal', f"${metrics['last_high_fractal']:.2f}"])
        if metrics.get('last_low_fractal'):
            display_rows.append(['Last Low Fractal', f"${metrics['last_low_fractal']:.2f}"])
        
        # Break info
        if metrics.get('last_break_type'):
            display_rows.append(['Last Break', metrics['last_break_type']])
            if metrics.get('last_break_price'):
                display_rows.append(['Break Price', f"${metrics['last_break_price']:.2f}"])
        
        # Statistics
        display_rows.append(['Total Fractals', str(metrics.get('fractal_count', 0))])
        display_rows.append(['Structure Breaks', str(metrics.get('structure_breaks', 0))])
        display_rows.append(['5-Min Bars Processed', str(metrics.get('candles_processed', 0))])
        
        # Summary text - indicate timeframe
        summary = f"5M: {signal.structure_type or 'No Signal'}"
        if signal.structure_type:
            summary += f" - {signal_direction}"
            if signal.structure_type == 'CHoCH':
                summary += " (Reversal)"
            elif signal.structure_type == 'BOS':
                summary += " (Continuation)"
        
        return {
            'plugin_name': self.name,
            'timestamp': entry_time,
            'signal': {
                'direction': signal_direction,
                'strength': float(signal.strength),
                'confidence': float(signal.strength)  # Use strength as confidence
            },
            'details': {
                'timeframe': '5-minute',
                'structure_type': signal.structure_type,
                'current_trend': metrics.get('current_trend'),
                'last_high_fractal': metrics.get('last_high_fractal'),
                'last_low_fractal': metrics.get('last_low_fractal'),
                'last_break_type': metrics.get('last_break_type'),
                'last_break_price': metrics.get('last_break_price'),
                'fractal_count': metrics.get('fractal_count', 0),
                'structure_breaks': metrics.get('structure_breaks', 0),
                'trend_changes': metrics.get('trend_changes', 0),
                'candles_processed': metrics.get('candles_processed', 0),
                'reason': signal.reason
            },
            'display_data': {
                'summary': summary,
                'description': signal.reason,
                'table_data': display_rows,
                'chart_markers': self._get_chart_markers(metrics)
            }
        }
    
    def _create_neutral_result(self, entry_time: datetime) -> Dict[str, Any]:
        """Create a neutral result when no signal is found"""
        return {
            'plugin_name': self.name,
            'timestamp': entry_time,
            'signal': {
                'direction': 'NEUTRAL',
                'strength': 50.0,
                'confidence': 50.0
            },
            'details': {
                'timeframe': '5-minute',
                'structure_type': None,
                'current_trend': 'NEUTRAL',
                'reason': 'No clear 5-minute market structure signal'
            },
            'display_data': {
                'summary': '5M: No Signal',
                'description': 'No clear market structure detected on 5-minute timeframe',
                'table_data': [
                    ['Timeframe', '5-Minute'],
                    ['Structure Type', 'None'],
                    ['Current Trend', 'NEUTRAL'],
                    ['Signal Strength', '50%']
                ]
            }
        }
    
    def _get_chart_markers(self, metrics: Dict) -> List[Dict]:
        """Get chart markers for visualization"""
        markers = []
        
        # Add fractal markers
        if metrics.get('last_high_fractal'):
            markers.append({
                'type': 'high_fractal',
                'price': metrics['last_high_fractal'],
                'label': 'H5',
                'timeframe': '5min'
            })
            
        if metrics.get('last_low_fractal'):
            markers.append({
                'type': 'low_fractal', 
                'price': metrics['last_low_fractal'],
                'label': 'L5',
                'timeframe': '5min'
            })
            
        return markers