# backtest/test_plugin_interpreter_full.py
"""
Comprehensive test tool for plugin calculations and interpreter output
Shows: Core Calculation → Plugin Output → Interpreter Output
"""

import asyncio
import logging
import json
import importlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtest.data.polygon_data_manager import PolygonDataManager
from backtest.processing.bt_results_interpreter import BTResultsInterpreter, SignalType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Plugin to calculation module mapping - WITH CORRECT PLUGIN PATHS
PLUGIN_MAPPING = {
    "1-Min EMA Crossover": {
        "plugin_module": "backtest.plugins.m1_ema",  # Correct module path
        "calc_module": "backtest.calculations.indicators.m1_ema",
        "calculator": "M1EMACalculator",
        "timeframe": "1min",
        "signal_type": SignalType.EMA
    },
    "5-Min EMA Crossover": {
        "plugin_module": "backtest.plugins.m5_ema",
        "calc_module": "backtest.calculations.indicators.m5_ema",
        "calculator": "M5EMACalculator",
        "timeframe": "5min",
        "signal_type": SignalType.EMA
    },
    "15-Min EMA Crossover": {
        "plugin_module": "backtest.plugins.m15_ema",
        "calc_module": "backtest.calculations.indicators.m15_ema",
        "calculator": "M15EMACalculator",
        "timeframe": "15min",
        "signal_type": SignalType.EMA
    },
    "1-Min Market Structure": {
        "plugin_module": "backtest.plugins.m1_market_structure",
        "calc_module": "backtest.calculations.market_structure.m1_structure",
        "calculator": "M1MarketStructure",
        "timeframe": "1min",
        "signal_type": SignalType.MARKET_STRUCTURE
    },
    "5-Min Market Structure": {
        "plugin_module": "backtest.plugins.m5_market_structure",
        "calc_module": "backtest.calculations.market_structure.m5_structure",
        "calculator": "M5MarketStructure",
        "timeframe": "5min",
        "signal_type": SignalType.MARKET_STRUCTURE
    },
    "15-Min Market Structure": {
        "plugin_module": "backtest.plugins.m15_market_structure",
        "calc_module": "backtest.calculations.market_structure.m15_structure",
        "calculator": "M15MarketStructure",
        "timeframe": "15min",
        "signal_type": SignalType.MARKET_STRUCTURE
    },
    "1-Min Statistical Trend": {
        "plugin_module": "backtest.plugins.m1_statistical_trend",
        "calc_module": "backtest.calculations.statistical.m1_statistical_trend",
        "calculator": "M1StatisticalTrend",
        "timeframe": "1min",
        "signal_type": SignalType.STATISTICAL_TREND
    },
    "5-Min Statistical Trend": {
        "plugin_module": "backtest.plugins.m5_statistical_trend",
        "calc_module": "backtest.calculations.statistical.m5_statistical_trend",
        "calculator": "M5StatisticalTrend",
        "timeframe": "5min",
        "signal_type": SignalType.STATISTICAL_TREND
    },
    "15-Min Statistical Trend": {
        "plugin_module": "backtest.plugins.m15_statistical_trend",
        "calc_module": "backtest.calculations.statistical.m15_statistical_trend",
        "calculator": "M15StatisticalTrend",
        "timeframe": "15min",
        "signal_type": SignalType.STATISTICAL_TREND
    },
    "1-Min Bid/Ask Analysis": {
        "plugin_module": "backtest.plugins.m1_bid_ask",
        "calc_module": "backtest.calculations.indicators.m1_bid_ask",
        "calculator": "M1BidAskAnalyzer",
        "timeframe": "1min",
        "signal_type": SignalType.BID_ASK_IMBALANCE,
        "needs_quotes": True
    },
    "Bid/Ask Imbalance Analysis": {
        "plugin_module": "backtest.plugins.bid_ask_imbalance",
        "calc_module": "backtest.calculations.microstructure.bid_ask_imbalance",
        "calculator": "BidAskImbalanceCalculator",
        "timeframe": "1min",
        "signal_type": SignalType.BID_ASK_IMBALANCE,
        "needs_quotes": True
    },
    "Tick Flow Analysis": {
        "plugin_module": "backtest.plugins.tick_flow",
        "calc_module": "backtest.calculations.volume.tick_flow",
        "calculator": "TickFlowAnalyzer",
        "timeframe": "1min",
        "signal_type": SignalType.TICK_FLOW,
        "needs_trades": True
    },
    "Bid/Ask Ratio Tracker": {
        "plugin_module": "backtest.plugins.bid_ask_ratio",
        "calc_module": "backtest.calculations.bid_ask_ratio",
        "calculator": "BidAskRatioTracker",
        "timeframe": "1min",
        "signal_type": SignalType.BUY_SELL_RATIO,
        "needs_quotes": True
    },
    "Impact Success": {
        "plugin_module": "backtest.plugins.impact_success",
        "calc_module": "backtest.calculations.large_order_tracker",
        "calculator": "LargeOrderTracker",
        "timeframe": "1min",
        "signal_type": SignalType.LARGE_ORDER_IMPACT,
        "needs_trades": True
    },
    "Large Orders Grid": {
        "plugin_module": "backtest.plugins.large_orders_grid",
        "calc_module": "backtest.calculations.large_order_tracker",
        "calculator": "LargeOrderTracker",
        "timeframe": "1min",
        "signal_type": SignalType.LARGE_ORDER_IMPACT,
        "needs_trades": True
    }
}


class PluginInterpreterTester:
    """Test complete flow from calculation to interpreter"""
    
    def __init__(self):
        # Initialize data manager
        logger.info("Initializing data manager...")
        self.data_manager = PolygonDataManager()
        
        # Initialize interpreter
        self.interpreter = BTResultsInterpreter()
        
    async def test_plugin(self, plugin_name: str, symbol: str, 
                         entry_time: datetime, direction: str):
        """Test a specific plugin through the complete flow"""
        
        if plugin_name not in PLUGIN_MAPPING:
            print(f"❌ Unknown plugin: {plugin_name}")
            return
            
        plugin_info = PLUGIN_MAPPING[plugin_name]
        
        print("\n" + "="*80)
        print(f"TESTING: {plugin_name}")
        print(f"Symbol: {symbol} | Direction: {direction} | Time: {entry_time}")
        print("="*80 + "\n")
        
        try:
            # Step 1: Load and run the plugin
            print("📊 Step 1: Running Plugin...")
            plugin_result = await self._run_plugin(plugin_name, plugin_info, symbol, entry_time, direction)
            
            if plugin_result:
                print("\n" + "-"*60)
                print("PLUGIN OUTPUT:")
                print("-"*60)
                self._display_plugin_output(plugin_result)
            
            # Step 2: Run the core calculation directly
            print("\n📊 Step 2: Running Core Calculation...")
            calc_result = await self._run_calculation(plugin_info, symbol, entry_time)
            
            if calc_result:
                print("\n" + "-"*60)
                print("CORE CALCULATION OUTPUT:")
                print("-"*60)
                self._display_calculation_output(calc_result, plugin_info['signal_type'])
            
            # Step 3: Run through interpreter
            if calc_result:
                print("\n📊 Step 3: Running Through Interpreter...")
                standardized = self.interpreter.interpret_signal(
                    signal_data=calc_result,
                    signal_type=plugin_info['signal_type'],
                    symbol=symbol,
                    timeframe=plugin_info['timeframe']
                )
                
                print("\n" + "-"*60)
                print("INTERPRETER OUTPUT:")
                print("-"*60)
                self._display_interpreter_output(standardized)
                
                # Step 4: Compare outputs
                print("\n" + "="*60)
                print("COMPARISON SUMMARY:")
                print("="*60)
                self._compare_outputs(plugin_result, calc_result, standardized, plugin_info)
                
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    async def _run_plugin(self, plugin_name: str, plugin_info: Dict, symbol: str, 
                         entry_time: datetime, direction: str) -> Optional[Dict]:
        """Run the plugin and return its output"""
        try:
            # Use the plugin_module from mapping
            plugin_module_path = plugin_info['plugin_module']
            
            plugin_module = importlib.import_module(plugin_module_path)
            
            # Set data manager if needed
            if hasattr(plugin_module, 'set_data_manager'):
                plugin_module.set_data_manager(self.data_manager)
            
            # Run analysis
            result = await plugin_module.run_analysis(symbol, entry_time, direction)
            return result
            
        except Exception as e:
            print(f"  ⚠️ Plugin error: {e}")
            return None
    
    async def _run_calculation(self, plugin_info: Dict, symbol: str, 
                              entry_time: datetime) -> Optional[Any]:
        """Run the core calculation module directly"""
        try:
            # Import calculation module
            calc_module = importlib.import_module(plugin_info['calc_module'])
            CalculatorClass = getattr(calc_module, plugin_info['calculator'])
            
            # Determine data needs
            if plugin_info.get('needs_quotes'):
                # Load quotes data
                data = await self._load_quotes(symbol, entry_time, plugin_info['timeframe'])
            elif plugin_info.get('needs_trades'):
                # Load trades data
                data = await self._load_trades(symbol, entry_time)
            else:
                # Load bars data
                data = await self._load_bars(symbol, entry_time, plugin_info['timeframe'])
            
            if data is None or (hasattr(data, 'empty') and data.empty):
                print("  ⚠️ No data available for calculation")
                return None
            
            # Create calculator and run
            calculator = CalculatorClass()
            
            # Different calculators have different method names
            if hasattr(calculator, 'calculate'):
                result = calculator.calculate(data)
            elif hasattr(calculator, 'analyze'):
                result = calculator.analyze(data)
            elif hasattr(calculator, 'process'):
                result = calculator.process(data)
            elif hasattr(calculator, 'add_trade'):
                # For trade-based calculators
                for _, trade in data.iterrows():
                    calculator.add_trade(trade)
                result = calculator.get_current_stats()
            else:
                print(f"  ⚠️ Calculator {plugin_info['calculator']} has no known analysis method")
                return None
                
            return result
            
        except Exception as e:
            print(f"  ⚠️ Calculation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _load_bars(self, symbol: str, entry_time: datetime, timeframe: str):
        """Load bar data"""
        lookback = 120 if timeframe == '1min' else 60 if timeframe == '5min' else 30
        start_time = entry_time - timedelta(minutes=lookback)
        
        return await self.data_manager.load_bars(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time,
            timeframe=timeframe
        )
    
    async def _load_quotes(self, symbol: str, entry_time: datetime, timeframe: str):
        """Load quote data"""
        lookback = 10  # 10 minutes of quotes
        start_time = entry_time - timedelta(minutes=lookback)
        
        return await self.data_manager.load_quotes(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time
        )
    
    async def _load_trades(self, symbol: str, entry_time: datetime):
        """Load trade data"""
        lookback = 10  # 10 minutes of trades
        start_time = entry_time - timedelta(minutes=lookback)
        
        return await self.data_manager.load_trades(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time
        )
    
    def _display_plugin_output(self, result: Dict):
        """Display plugin output in a structured way"""
        # Signal info
        signal = result.get('signal', {})
        print(f"Signal Direction: {signal.get('direction', 'N/A')}")
        print(f"Signal Strength: {signal.get('strength', 'N/A')}")
        if 'normalized_strength' in signal:
            print(f"Normalized Strength: {signal.get('normalized_strength')}")
        print(f"Confidence: {signal.get('confidence', 'N/A')}")
        
        # Display data
        display = result.get('display_data', {})
        if display:
            print(f"\nSummary: {display.get('summary', 'N/A')}")
            print(f"Description: {display.get('description', 'N/A')}")
        
        # Error check
        if 'error' in result:
            print(f"\n❌ Error: {result['error']}")
    
    def _display_calculation_output(self, result: Any, signal_type: SignalType):
        """Display calculation output based on type"""
        if hasattr(result, '__dict__'):
            # Object with attributes
            attrs = vars(result)
            for key, value in attrs.items():
                if not key.startswith('_'):
                    print(f"{key}: {value}")
        elif isinstance(result, dict):
            # Dictionary result
            for key, value in result.items():
                print(f"{key}: {value}")
        else:
            # Other format
            print(f"Result: {result}")
    
    def _display_interpreter_output(self, standardized):
        """Display interpreter output"""
        print(f"Bull/Bear Signal: {standardized.bull_bear_signal}")
        print(f"Strength: {standardized.strength}")
        print(f"Confidence: {standardized.confidence}")
        print(f"Original Signal: {standardized.original_signal}")
        if standardized.sub_type:
            print(f"Sub Type: {standardized.sub_type}")
        print(f"Reason: {standardized.reason}")
        
        # Show database format
        db_data = self.interpreter.prepare_for_supabase(standardized)
        print(f"\nDatabase Summary:")
        print(f"  Indicator Type: {db_data['summary']['indicator_type']}")
        print(f"  Bull/Bear: {db_data['summary']['bull_bear_signal']}")
        print(f"  Metrics: {len(db_data['details'])} values")
    
    def _compare_outputs(self, plugin_result: Dict, calc_result: Any, 
                        standardized: Any, plugin_info: Dict):
        """Compare outputs and highlight discrepancies"""
        if not plugin_result:
            print("⚠️ No plugin result to compare")
            return
            
        plugin_signal = plugin_result.get('signal', {})
        
        print("\n📊 ALIGNMENT CHECK:")
        
        # Direction alignment
        plugin_dir = plugin_signal.get('direction', 'N/A')
        interp_dir = standardized.bull_bear_signal if standardized else 'N/A'
        
        if plugin_dir != 'N/A' and plugin_dir == interp_dir:
            print(f"✅ Direction: {plugin_dir} → {interp_dir}")
        elif plugin_dir != 'N/A':
            print(f"⚠️  Direction Mismatch: Plugin={plugin_dir}, Interpreter={interp_dir}")
        else:
            print(f"⚠️  No plugin direction to compare")
            
        # Strength alignment - handle N/A values
        plugin_strength = plugin_signal.get('strength', 'N/A')
        interp_strength = standardized.strength if standardized else 'N/A'
        
        if plugin_strength != 'N/A' and interp_strength != 'N/A':
            try:
                if abs(float(plugin_strength) - float(interp_strength)) < 1:
                    print(f"✅ Strength: {plugin_strength} → {interp_strength}")
                else:
                    print(f"⚠️  Strength Mismatch: Plugin={plugin_strength}, Interpreter={interp_strength}")
            except (ValueError, TypeError):
                print(f"⚠️  Could not compare strengths: Plugin={plugin_strength}, Interpreter={interp_strength}")
        else:
            print(f"⚠️  No strength values to compare")
        
        # Check if 4-category signals are handled
        if plugin_dir in ['WEAK_BULLISH', 'WEAK_BEARISH', 'BULLISH', 'BEARISH']:
            if interp_dir in ['WEAK_BULLISH', 'WEAK_BEARISH', 'BULLISH', 'BEARISH']:
                print(f"✅ 4-Category Signal Preserved: {plugin_dir}")
            else:
                print(f"❌ 4-Category Signal Lost: {plugin_dir} → {interp_dir}")
                print("   The interpreter needs to handle 4-category signals!")


async def main():
    """Main test function"""
    import argparse
    
    # List all available plugins
    plugin_list = list(PLUGIN_MAPPING.keys())
    
    parser = argparse.ArgumentParser(
        description="Test Plugin Calculation and Interpreter Flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Plugins:
{chr(10).join(f'  {i+1}. {p}' for i, p in enumerate(plugin_list))}

Examples:
  # Test specific plugin by name
  python test_plugin_interpreter_full.py -p "1-Min EMA Crossover" -s AAPL
  
  # Test by number
  python test_plugin_interpreter_full.py -p 1 -s TSLA
  
  # Test all plugins
  python test_plugin_interpreter_full.py --all -s SPY
        """
    )
    
    parser.add_argument('-p', '--plugin', 
                       help='Plugin name or number to test')
    
    parser.add_argument('-s', '--symbol', 
                       default='AAPL',
                       help='Stock symbol to analyze')
    
    parser.add_argument('-d', '--direction',
                       choices=['LONG', 'SHORT'],
                       default='LONG',
                       help='Trade direction')
    
    parser.add_argument('-t', '--time',
                       help='Entry time (YYYY-MM-DD HH:MM:SS) in UTC')
    
    parser.add_argument('--all',
                       action='store_true',
                       help='Test all plugins')
    
    parser.add_argument('--list',
                       action='store_true',
                       help='List all available plugins')
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        print("\nAvailable Plugins:")
        for i, plugin in enumerate(plugin_list, 1):
            print(f"  {i}. {plugin}")
        return
    
    # Parse entry time
    if args.time:
        try:
            entry_time = datetime.strptime(args.time, '%Y-%m-%d %H:%M:%S')
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"Error: Invalid time format. Use YYYY-MM-DD HH:MM:SS")
            return
    else:
        entry_time = datetime.now(timezone.utc) - timedelta(hours=1)
    
    # Create tester
    tester = PluginInterpreterTester()
    
    # Determine which plugins to test
    if args.all:
        plugins_to_test = plugin_list
    elif args.plugin:
        # Check if it's a number
        try:
            plugin_num = int(args.plugin)
            if 1 <= plugin_num <= len(plugin_list):
                plugins_to_test = [plugin_list[plugin_num - 1]]
            else:
                print(f"Error: Plugin number must be between 1 and {len(plugin_list)}")
                return
        except ValueError:
            # It's a name
            if args.plugin in plugin_list:
                plugins_to_test = [args.plugin]
            else:
                print(f"Error: Unknown plugin '{args.plugin}'")
                print("Use --list to see available plugins")
                return
    else:
        print("Error: Please specify a plugin with -p or use --all")
        return
    
    # Run tests
    for plugin in plugins_to_test:
        await tester.test_plugin(plugin, args.symbol, entry_time, args.direction)
        if len(plugins_to_test) > 1:
            await asyncio.sleep(2)  # Pause between plugins


if __name__ == "__main__":
    print("\n🔬 Plugin Calculation & Interpreter Test Tool")
    print("Shows: Core Calculation → Plugin Output → Interpreter Output")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()