# backtest/plugins/m15_statistical_trend/__init__.py
"""
15-Minute Statistical Trend Plugin - Simplified Pass-through
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import sys
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backtest.data.polygon_data_manager import PolygonDataManager
from backtest.calculations.trend.statistical_trend_15min import StatisticalTrend15MinSimplified

logger = logging.getLogger(__name__)

# Plugin metadata
PLUGIN_NAME = "15-Min Statistical Trend"
PLUGIN_VERSION = "2.0.0"

# Initialize data manager (shared instance)
_data_manager = None

def _get_data_manager():
    """Get or create data manager instance"""
    global _data_manager
    if _data_manager is None:
        _data_manager = PolygonDataManager(
            extend_window_bars=2  # Small extension for 15-min bars
        )
        _data_manager.set_current_plugin(PLUGIN_NAME)
    return _data_manager

async def run_analysis(symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    Simple pass-through for 15-minute statistical trend analysis.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time (UTC)
        direction: 'LONG' or 'SHORT'
        
    Returns:
        Analysis results with regime, bias, and signal data
    """
    try:
        # 1. Initialize calculator
        analyzer = StatisticalTrend15MinSimplified(lookback_periods=10)  # 150 minutes
        
        # 2. Calculate data range (need 10 bars minimum)
        start_time = entry_time - timedelta(minutes=150)  # 10 * 15-minute bars
        
        # 3. Fetch data
        logger.info(f"Fetching 15-min bars for {symbol} from {start_time} to {entry_time}")
        data_manager = _get_data_manager()
        bars_df = await data_manager.load_bars(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time,
            timeframe='15min'
        )
        
        if bars_df.empty or len(bars_df) < 10:
            raise ValueError(f"Insufficient data: got {len(bars_df)} bars, need at least 10")
        
        # 4. Run analysis
        signal = analyzer.analyze(symbol, bars_df, entry_time)
        
        # 5. Format and return results
        return _format_results(signal, direction)
        
    except Exception as e:
        logger.error(f"Error in 15-Min Statistical Trend analysis: {e}")
        return {
            'plugin_name': PLUGIN_NAME,
            'timestamp': entry_time,
            'error': str(e),
            'signal': {
                'direction': 'NEUTRAL',
                'strength': 0,
                'confidence': 0
            }
        }

def _format_results(signal, direction: str) -> Dict[str, Any]:
    """Format MarketRegimeSignal into expected output structure"""
    
    # Map regime to direction for dashboard
    direction_map = {
        'BULL MARKET': 'BULLISH',
        'BEAR MARKET': 'BEARISH',
        'RANGE BOUND': 'NEUTRAL'
    }
    
    # Check alignment
    aligned = False
    if direction == 'LONG' and signal.regime == 'BULL MARKET':
        aligned = True
    elif direction == 'SHORT' and signal.regime == 'BEAR MARKET':
        aligned = True
    
    # Build description
    description = f"{signal.regime} - {signal.daily_bias}"
    if signal.volatility_state != 'NORMAL':
        description += f" ({signal.volatility_state} volatility)"
    if aligned:
        description += f" ✅ Aligned with {direction}"
    else:
        description += f" ⚠️ Not aligned with {direction}"
    
    return {
        'plugin_name': PLUGIN_NAME,
        'timestamp': signal.timestamp,
        'signal': {
            'direction': direction_map.get(signal.regime, 'NEUTRAL'),
            'strength': signal.trend_strength,
            'confidence': signal.confidence
        },
        'details': {
            'regime': signal.regime,
            'daily_bias': signal.daily_bias,
            'volatility_state': signal.volatility_state,
            'volume_trend': signal.volume_trend,
            'volatility_adjusted_strength': signal.volatility_adjusted_strength,
            'aligned': aligned,
            'price': signal.price
        },
        'display_data': {
            'summary': f"{signal.regime} - {signal.daily_bias} ({signal.confidence:.0f}% confidence)",
            'description': description,
            'table_data': [
                ['Market Regime', signal.regime],
                ['Daily Bias', signal.daily_bias],
                ['Confidence', f"{signal.confidence:.0f}%"],
                ['Trend Strength', f"{signal.trend_strength:.1f}%"],
                ['Vol-Adjusted Strength', f"{signal.volatility_adjusted_strength:.2f}"],
                ['Volatility State', signal.volatility_state],
                ['Volume Trend', signal.volume_trend],
                ['Current Price', f"${signal.price:.2f}"],
                ['Trade Alignment', '✅ Yes' if aligned else '❌ No']
            ]
        }
    }