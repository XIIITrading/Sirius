# backtest/debug/compare_results.py
"""
Compare results between debug tool and backtest engine
"""

import asyncio
from datetime import datetime, timezone
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.debug.debug_m1_market_structure import M1MarketStructureDebugger
from backtest.core.engine import BacktestEngine, BacktestConfig
from backtest.plugins.plugin_loader import PluginLoader
from data.polygon_data_manager import PolygonDataManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


async def compare_m1_market_structure(symbol: str, entry_time_str: str):
    """Compare debug tool results with backtest engine results"""
    
    # Parse entry time
    entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
    entry_time = entry_time.replace(tzinfo=timezone.utc)
    
    print(f"\n{'='*80}")
    print(f"COMPARING M1 MARKET STRUCTURE RESULTS")
    print(f"Symbol: {symbol}")
    print(f"Entry Time: {entry_time}")
    print(f"{'='*80}\n")
    
    # 1. Run debug tool
    print("1. Running debug tool...")
    debugger = M1MarketStructureDebugger(
        symbol=symbol,
        entry_time=entry_time,
        lookback_minutes=120,
        fractal_length=5
    )
    debug_results = await debugger.run_debug()
    
    debug_final_state = debug_results.get('final_state', {})
    print(f"\nDebug tool results:")
    print(f"  Trend: {debug_final_state.get('trend')}")
    print(f"  Last signal: {debug_final_state.get('last_signal')}")
    print(f"  Strength: {debug_final_state.get('strength')}")
    
    if debug_final_state.get('metrics'):
        metrics = debug_final_state['metrics']
        print(f"  Last break type: {metrics.get('last_break_type')}")
        print(f"  Last high fractal: {metrics.get('last_high_fractal')}")
        print(f"  Last low fractal: {metrics.get('last_low_fractal')}")
    
    # 2. Run backtest engine
    print("\n2. Running backtest engine...")
    
    # Load plugins
    plugin_loader = PluginLoader()
    plugins = plugin_loader.load_all_plugins()
    plugin_registry = plugin_loader.get_registry()
    
    # Create engine
    data_manager = PolygonDataManager()
    engine = BacktestEngine(
        data_manager=data_manager,
        plugin_registry=plugin_registry
    )
    engine.enable_debug_mode()  # Enable debug mode
    
    # Register M1 market structure adapter
    adapter_configs = plugin_loader.get_adapter_configs()
    m1_config = adapter_configs.get('m1_market_structure')
    if m1_config:
        adapter = m1_config['adapter_class'](**m1_config['adapter_config'])
        engine.register_adapter('m1_market_structure', adapter)
    else:
        print("ERROR: M1 market structure adapter not found!")
        return
    
    # Create backtest config
    config = BacktestConfig(
        symbol=symbol,
        entry_time=entry_time,
        direction='LONG',  # Doesn't matter for this test
        enabled_calculations=['m1_market_structure']
    )
    
    # Run backtest
    result = await engine.run_backtest(config)
    
    # Get M1 signal
    m1_signal = None
    for signal in result.entry_signals:
        if signal.name == "1-Min Market Structure":
            m1_signal = signal
            break
    
    if m1_signal:
        print(f"\nBacktest engine results:")
        print(f"  Direction: {m1_signal.direction}")
        print(f"  Strength: {m1_signal.strength}")
        
        metadata = m1_signal.metadata
        print(f"  Structure type: {metadata.get('structure_type')}")
        print(f"  Current trend: {metadata.get('current_trend')}")
        print(f"  Last break type: {metadata.get('last_break_type')}")
        print(f"  Last high fractal: {metadata.get('last_high_fractal')}")
        print(f"  Last low fractal: {metadata.get('last_low_fractal')}")
        print(f"  Reason: {metadata.get('reason')}")
    else:
        print("\nNo M1 market structure signal found in backtest results!")
    
    # 3. Compare results
    print(f"\n{'='*80}")
    print("COMPARISON:")
    print(f"{'='*80}")
    
    if debug_final_state and m1_signal:
        debug_trend = debug_final_state.get('trend', 'UNKNOWN')
        backtest_direction = m1_signal.direction.replace('ISH', '')  # Remove 'ISH' suffix
        
        match = debug_trend == backtest_direction
        print(f"Trends match: {match}")
        print(f"  Debug tool: {debug_trend}")
        print(f"  Backtest: {backtest_direction}")
        
        if not match:
            print("\n⚠️ MISMATCH DETECTED!")
            print("Possible causes:")
            print("1. Data range differences - check the lookback period")
            print("2. Entry time handling - debug tool processes UP TO entry, backtest might differ")
            print("3. Signal at entry - backtest gets signal AT entry time")
            print("4. Initialization differences - check if both use same fractal length")
            
            print("\nRecommendations:")
            print("- Enable debug mode in GUI to see detailed logs")
            print("- Check the exact data range being processed")
            print("- Verify the last few candles before entry")
            print("- Check if signals are being generated at expected times")
    else:
        print("Unable to compare - missing data from one of the tools")


async def main():
    """Main function"""
    if len(sys.argv) < 3:
        print("Usage: python compare_results.py SYMBOL 'YYYY-MM-DD HH:MM:SS'")
        print("Example: python compare_results.py CRCL '2024-06-27 13:35:00'")
        sys.exit(1)
    
    symbol = sys.argv[1]
    entry_time = sys.argv[2]
    
    await compare_m1_market_structure(symbol, entry_time)


if __name__ == "__main__":
    asyncio.run(main())