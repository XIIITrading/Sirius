# backtest/plugins/tick_flow/__init__.py
"""
Tick Flow Analysis Plugin
Real-time order flow analysis using tick-by-tick trade classification.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import pandas as pd

from modules.calculations.volume.tick_flow import TickFlowAnalyzer, VolumeSignal
from backtest.data.polygon_data_manager import PolygonDataManager

# Configure logging
logger = logging.getLogger(__name__)

# Plugin metadata
PLUGIN_NAME = "Tick Flow Analysis"
PLUGIN_VERSION = "1.0.0"

# Configuration
CONFIG = {
    'buffer_size': 1000,  # Number of trades to analyze
    'large_trade_multiplier': 3.0,  # Multiple of avg size for large trades
    'momentum_threshold': 60.0,  # % threshold for bull/bear signal
    'min_trades_required': 50,  # Minimum trades for valid signal
    'lookback_minutes': 5  # Minutes of data to fetch before entry
}

# Create module-level instances
_data_manager = PolygonDataManager()
_data_manager.set_current_plugin(PLUGIN_NAME)

_analyzer = TickFlowAnalyzer(
    buffer_size=CONFIG['buffer_size'],
    large_trade_multiplier=CONFIG['large_trade_multiplier'],
    momentum_threshold=CONFIG['momentum_threshold'],
    min_trades_required=CONFIG['min_trades_required']
)


async def run_analysis(symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    Run Tick Flow analysis.
    
    This is the single entry point for the plugin.
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time (UTC)
        direction: 'LONG' or 'SHORT'
        
    Returns:
        Complete analysis results formatted for display
    """
    try:
        # Validate inputs
        if not _validate_inputs(symbol, entry_time, direction):
            raise ValueError("Invalid input parameters")
        
        # Ensure timezone aware
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        
        # 1. Fetch trades from data layer
        trades_df = await _fetch_trades(symbol, entry_time)
        
        if trades_df.empty:
            raise ValueError(f"No trade data available for {symbol}")
        
        # 2. Process trades through analyzer
        last_signal = None
        signals_generated = []
        trades_processed = 0
        
        logger.info(f"Processing {len(trades_df)} trades for {symbol}")
        
        for timestamp, trade in trades_df.iterrows():
            # Stop at entry time
            if timestamp >= entry_time:
                break
            
            trades_processed += 1
            
            # Convert trade data to expected format
            trade_data = {
                'timestamp': int(timestamp.timestamp() * 1000),
                'price': float(trade['price']),
                'size': float(trade['size']),
                'conditions': trade.get('conditions', [])
            }
            
            # Process trade
            signal = _analyzer.process_trade(symbol, trade_data)
            
            if signal:
                last_signal = signal
                signals_generated.append(signal)
        
        logger.info(f"Processed {trades_processed} trades, generated {len(signals_generated)} signals")
        
        # 3. Get final analysis
        if not last_signal:
            # Force analysis even if below min trades
            last_signal = _analyzer.get_current_analysis(symbol)
            
        if not last_signal:
            raise ValueError("Insufficient trades for analysis")
        
        # 4. Format and return results
        return _format_results(last_signal, entry_time, direction, signals_generated)
        
    except Exception as e:
        logger.error(f"Error in Tick Flow analysis: {e}")
        # Return error result
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


def _validate_inputs(symbol: str, entry_time: datetime, direction: str) -> bool:
    """Validate input parameters"""
    if not symbol or not isinstance(symbol, str):
        return False
    
    if not isinstance(entry_time, datetime):
        return False
        
    if direction not in ['LONG', 'SHORT']:
        return False
        
    return True


async def _fetch_trades(symbol: str, entry_time: datetime) -> pd.DataFrame:
    """Fetch required trade data from data layer"""
    # Calculate time range - fetch trades BEFORE entry time
    start_time = entry_time - timedelta(minutes=CONFIG['lookback_minutes'])
    
    logger.info(f"Fetching {symbol} trades from {start_time} to {entry_time}")
    
    # Fetch trade data - data layer handles all caching automatically
    trades = await _data_manager.load_trades(
        symbol=symbol,
        start_time=start_time,
        end_time=entry_time
    )
    
    return trades


def _format_results(signal: VolumeSignal, entry_time: datetime, 
                    direction: str, signals_generated: List[VolumeSignal]) -> Dict[str, Any]:
    """Format results for display"""
    
    # Get statistics
    stats = _analyzer.get_statistics()
    metrics = signal.metrics
    
    # Build summary display
    summary_rows = [
        ['Signal', f"{signal.signal} ({signal.strength:.0f}%)"],
        ['Buy Volume %', f"{metrics['buy_volume_pct']:.1f}%"],
        ['Buy/Sell Trades', f"{metrics['buy_trades']}/{metrics['sell_trades']}"],
        ['Buy/Sell Volume', f"{metrics['buy_volume']:,.0f}/{metrics['sell_volume']:,.0f}"],
        ['Large Buys/Sells', f"{metrics['large_buy_trades']}/{metrics['large_sell_trades']}"],
        ['Momentum Score', f"{metrics['momentum_score']:+.1f}"],
        ['Trade Rate', f"{metrics['trade_rate']:.1f} trades/sec"],
        ['Avg Trade Size', f"{metrics['avg_trade_size']:,.0f}"]
    ]
    
    # Calculate signal history - now signals have correct timestamps
    signal_history = []
    if signals_generated:
        # Get last 10 signals
        for sig in signals_generated[-10:]:
            signal_history.append([
                sig.timestamp.strftime('%H:%M:%S'),  # Now uses correct timestamp from trades
                sig.signal,
                f"{sig.strength:.0f}%",
                f"{sig.metrics['buy_volume_pct']:.0f}%"
            ])
    
    # Determine confidence based on alignment
    confidence = signal.strength
    aligned = False
    
    if (direction == 'LONG' and signal.signal == 'BULLISH') or \
       (direction == 'SHORT' and signal.signal == 'BEARISH'):
        aligned = True
    elif (direction == 'LONG' and signal.signal == 'BEARISH') or \
         (direction == 'SHORT' and signal.signal == 'BULLISH'):
        confidence = max(0, 100 - confidence)  # Invert if opposite
    
    # Create description
    description = signal.reason
    if aligned:
        description += f" | ✅ Aligned with {direction}"
    else:
        description += f" | ⚠️ Contradicts {direction}"
    
    # Determine emoji for signal
    emoji_map = {
        'BULLISH': '🟢',
        'BEARISH': '🔴',
        'NEUTRAL': '⚪'
    }
    emoji = emoji_map.get(signal.signal, '⚪')
    
    return {
        'plugin_name': PLUGIN_NAME,
        'timestamp': entry_time,
        'signal': {
            'direction': signal.signal,
            'strength': float(signal.strength),
            'confidence': float(confidence)
        },
        'details': {
            'trades_analyzed': metrics['total_trades'],
            'buy_volume_pct': metrics['buy_volume_pct'],
            'momentum_score': metrics['momentum_score'],
            'large_buy_trades': metrics['large_buy_trades'],
            'large_sell_trades': metrics['large_sell_trades'],
            'trade_rate': metrics['trade_rate'],
            'price_trend': metrics['price_trend'],
            'trades_processed': stats['trades_processed'],
            'signals_generated': stats['signals_generated'],
            'aligned': aligned
        },
        'display_data': {
            'summary': f"{emoji} {signal.signal} - {signal.strength:.0f}%",
            'description': description,
            'table_data': summary_rows,
            'signal_history': {
                'headers': ['Time', 'Signal', 'Strength', 'Buy %'],
                'rows': signal_history
            } if signal_history else None,
            'chart_markers': _get_chart_markers(metrics)
        }
    }


def _get_chart_markers(metrics: Dict) -> List[Dict]:
    """Get chart markers for visualization"""
    markers = []
    
    # Buy/sell pressure markers
    if metrics['buy_volume_pct'] > 70:
        markers.append({
            'type': 'heavy_buying',
            'label': 'HEAVY BUY',
            'color': 'green'
        })
    elif metrics['buy_volume_pct'] < 30:
        markers.append({
            'type': 'heavy_selling',
            'label': 'HEAVY SELL',
            'color': 'red'
        })
    
    # Large trade markers
    if metrics['large_buy_trades'] > metrics['large_sell_trades'] * 2:
        markers.append({
            'type': 'large_buys',
            'label': 'LARGE BUYS',
            'color': 'darkgreen'
        })
    elif metrics['large_sell_trades'] > metrics['large_buy_trades'] * 2:
        markers.append({
            'type': 'large_sells',
            'label': 'LARGE SELLS',
            'color': 'darkred'
        })
        
    return markers


# Optional: Export configuration function for UI
def get_config() -> Dict[str, Any]:
    """Get plugin configuration for UI settings"""
    return CONFIG.copy()


# Optional: Update configuration
def update_config(new_config: Dict[str, Any]):
    """Update plugin configuration"""
    global CONFIG, _analyzer
    
    CONFIG.update(new_config)
    
    # Recreate analyzer with new config
    _analyzer = TickFlowAnalyzer(
        buffer_size=CONFIG['buffer_size'],
        large_trade_multiplier=CONFIG['large_trade_multiplier'],
        momentum_threshold=CONFIG['momentum_threshold'],
        min_trades_required=CONFIG['min_trades_required']
    )