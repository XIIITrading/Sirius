# backtest/plugins/m15_ema/test.py
"""
Test CLI for M15 EMA Crossover Plugin
Tests the complete workflow: Dashboard -> Plugin -> DataManager -> CircuitBreaker -> Signal
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backtest.data.polygon_data_manager import PolygonDataManager
from backtest.data.protected_data_manager import ProtectedDataManager
from backtest.calculations.indicators.m15_ema import M15EMACalculator, validate_15min_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PluginTester:
    """Simulates the dashboard calling the plugin"""
    
    def __init__(self):
        """Initialize the test environment"""
        logger.info("Initializing M15 EMA Plugin Tester...")
        
        # Create data manager with circuit breaker protection
        self.polygon_manager = PolygonDataManager()
        self.data_manager = ProtectedDataManager(
            polygon_data_manager=self.polygon_manager,
            circuit_breaker_config={
                'failure_threshold': 0.5,
                'consecutive_failures': 3,
                'recovery_timeout': 60,
                'rate_limits': {
                    'bars': {'per_minute': 100, 'burst': 10}
                }
            }
        )
        
        # Plugin configuration
        self.plugin_config = {
            'name': '15-Min EMA Crossover',
            'version': '1.0.0',
            'required_bars': 390,  # 26 fifteen-minute bars * 15
            'ema_short': 9,
            'ema_long': 21
        }
        
        logger.info("Plugin Tester initialized")
    
    async def run_analysis(self, symbol: str, entry_time: datetime, direction: str):
        """
        Simulate the complete plugin workflow
        
        This mimics what the dashboard would do:
        1. Call plugin to analyze
        2. Plugin fetches data via data manager
        3. Data goes through circuit breaker
        4. Plugin runs calculation
        5. Returns signal to dashboard
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting 15-minute EMA analysis for {symbol} at {entry_time}")
        logger.info(f"Direction: {direction}")
        logger.info(f"{'='*60}\n")
        
        try:
            # Step 1: Plugin sets itself as current plugin
            self.data_manager.set_current_plugin(self.plugin_config['name'])
            
            # Step 2: Plugin will handle its own data fetching internally
            # We'll simulate what the plugin does
            required_1min_bars = self.plugin_config['required_bars']
            lookback_days = 3
            max_lookback_days = 10
            bars_1min = None
            
            logger.info(f"Attempting to fetch {required_1min_bars} 1-minute bars")
            
            while lookback_days <= max_lookback_days:
                start_time = entry_time - timedelta(days=lookback_days)
                
                logger.info(f"Trying {lookback_days}-day lookback: {start_time} to {entry_time}")
                
                bars_1min = await self.data_manager.load_bars(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=entry_time,
                    timeframe='1min'
                )
                
                if not bars_1min.empty:
                    logger.info(f"Fetched {len(bars_1min)} bars with {lookback_days}-day lookback")
                    
                    if len(bars_1min) >= required_1min_bars:
                        # Trim to exactly what we need
                        bars_1min = bars_1min.iloc[-required_1min_bars:]
                        logger.info(f"Successfully obtained required {required_1min_bars} bars")
                        break
                
                lookback_days += 2
            
            # Check circuit breaker status
            circuit_status = self.data_manager.get_circuit_status()
            logger.info(f"Circuit Breaker State: {circuit_status['state']}")
            logger.info(f"API Calls: {circuit_status['metrics']['total_requests']}")
            logger.info(f"Cache Hits: {circuit_status['metrics'].get('cache_hits', 0)}")
            
            if bars_1min is None or bars_1min.empty:
                logger.error("No data received from data manager")
                return self._create_error_signal(entry_time, "No data available")
            
            if len(bars_1min) < required_1min_bars:
                logger.error(f"Insufficient data: only {len(bars_1min)} bars")
                return self._create_error_signal(entry_time, f"Insufficient data: only {len(bars_1min)} bars")
            
            logger.info(f"Using exactly {len(bars_1min)} 1-minute bars for calculation")
            
            # Step 4: Validate data
            is_valid, error_msg = validate_15min_data(bars_1min, timeframe='1min')
            if not is_valid:
                logger.error(f"Data validation failed: {error_msg}")
                return self._create_error_signal(entry_time, error_msg)
            
            # Step 5: Plugin runs calculation
            calculator = M15EMACalculator()
            result = calculator.calculate(bars_1min, timeframe='1min')
            
            if not result:
                return self._create_error_signal(entry_time, "Insufficient data for calculation")
            
            # Step 6: Plugin formats signal for dashboard
            signal = self._format_signal(result, entry_time, direction)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}", exc_info=True)
            return self._create_error_signal(entry_time, str(e))
    
    def _format_signal(self, result, entry_time: datetime, direction: str) -> dict:
        """Format calculation result as dashboard signal"""
        
        # Determine signal alignment
        alignment = "Neutral"
        if result.signal != "NEUTRAL":
            if (direction == "LONG" and result.signal == "BULL") or \
               (direction == "SHORT" and result.signal == "BEAR"):
                alignment = "Aligned ✓"
            else:
                alignment = "Opposed ✗"
        
        # Map internal signal to dashboard format
        signal_map = {
            'BULL': 'BULLISH',
            'BEAR': 'BEARISH',
            'NEUTRAL': 'NEUTRAL'
        }
        
        return {
            'plugin_name': self.plugin_config['name'],
            'timestamp': entry_time,
            'signal': {
                'direction': signal_map.get(result.signal, 'NEUTRAL'),
                'strength': float(result.signal_strength),
                'confidence': float(result.signal_strength)
            },
            'details': {
                'timeframe': '15-minute',
                'ema_9': result.ema_9,
                'ema_21': result.ema_21,
                'spread': result.spread,
                'spread_pct': result.spread_pct,
                'trend_strength': result.trend_strength,
                'is_crossover': result.is_crossover,
                'crossover_type': result.crossover_type,
                'price_position': result.price_position,
                'bars_processed': result.bars_processed,
                'last_close': result.last_15min_close,
                'last_volume': result.last_15min_volume,
                'alignment': alignment,
                'bars_used': self.plugin_config['required_bars']
            },
            'display_data': {
                'summary': f"15-Min EMA 9/21 - {signal_map[result.signal]}",
                'description': result.reason,
                'table_data': [
                    ['Timeframe', '15-Minute'],
                    ['EMA 9', f'${result.ema_9:.2f}'],
                    ['EMA 21', f'${result.ema_21:.2f}'],
                    ['Spread', f'${result.spread:.2f} ({result.spread_pct:.2f}%)'],
                    ['Trend Strength', f'{result.trend_strength:.0f}%'],
                    ['Signal Strength', f'{result.signal_strength:.0f}%'],
                    ['15m Bars', f'{result.bars_processed}'],
                    ['1m Bars Used', f'{self.plugin_config["required_bars"]}'],
                    ['Price Position', result.price_position.title()],
                    ['Last Close', f'${result.last_15min_close:.2f}'],
                    ['Last Volume', f'{result.last_15min_volume:,.0f}'],
                    ['Direction Alignment', alignment]
                ]
            }
        }
    
    def _create_error_signal(self, entry_time: datetime, error_msg: str) -> dict:
        """Create error signal response"""
        return {
            'plugin_name': self.plugin_config['name'],
            'timestamp': entry_time,
            'error': error_msg,
            'signal': {
                'direction': 'NEUTRAL',
                'strength': 0,
                'confidence': 0
            }
        }
    
    def display_results(self, signal: dict):
        """Display signal results in formatted output"""
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        
        if 'error' in signal:
            print(f"\n❌ ERROR: {signal['error']}")
            return
        
        # Signal summary with color coding
        sig = signal['signal']
        details = signal.get('details', {})
        
        # Color codes
        if sig['direction'] == 'BULLISH':
            color = '\033[92m'  # Green
            emoji = '🟢'
        elif sig['direction'] == 'BEARISH':
            color = '\033[91m'  # Red
            emoji = '🔴'
        else:
            color = '\033[93m'  # Yellow
            emoji = '⚪'
        reset = '\033[0m'
        
        print(f"\n{emoji} {color}SIGNAL: {sig['direction']}{reset}")
        print(f"   Strength: {sig['strength']:.0f}%")
        print(f"   Confidence: {sig['confidence']:.0f}%")
        
        # Details
        if details:
            print("\n📈 DETAILS:")
            print(f"   Timeframe: {details['timeframe']}")
            print(f"   EMA 9: ${details['ema_9']:.2f}")
            print(f"   EMA 21: ${details['ema_21']:.2f}")
            print(f"   Spread: ${details['spread']:.2f} ({details['spread_pct']:.2f}%)")
            print(f"   Trend Strength: {details['trend_strength']:.0f}%")
            
            if details.get('is_crossover'):
                print(f"   🔄 Crossover: {details['crossover_type']}")
            
            print(f"   Price Position: {details.get('price_position', 'N/A')}")
            print(f"   Last Close: ${details.get('last_close', 0):.2f}")
            print(f"   Last Volume: {details.get('last_volume', 0):,.0f}")
            print(f"   15m Bars Processed: {details.get('bars_processed', 'N/A')}")
            print(f"   1m Bars Used: {details.get('bars_used', 'N/A')}")
            print(f"   Alignment: {details.get('alignment', 'N/A')}")
        
        # Display data
        display = signal.get('display_data', {})
        if display:
            print(f"\n📋 SUMMARY: {display['summary']}")
            print(f"   {display['description']}")
    
    async def get_data_report(self):
        """Get data manager statistics"""
        stats = self.data_manager.get_cache_stats()
        circuit = self.data_manager.get_circuit_status()
        
        print("\n" + "="*60)
        print("DATA MANAGER REPORT")
        print("="*60)
        
        print("\n📊 API Statistics:")
        print(f"   Total Requests: {circuit['metrics']['total_requests']}")
        print(f"   Successful: {circuit['metrics']['successful_requests']}")
        print(f"   Failed: {circuit['metrics']['failed_requests']}")
        print(f"   Cache Hit Rate: {stats['api_stats']['cache_hit_rate']:.1f}%")
        
        print("\n🔌 Circuit Breaker:")
        print(f"   State: {circuit['state']}")
        print(f"   Failure Rate: {circuit['failure_rate']:.1%}")
        
        print("\n💾 Cache Status:")
        print(f"   Memory Cache Items: {stats['memory_cache']['cached_items']}")
        print(f"   File Cache Size: {stats['file_cache']['total_size_mb']:.1f} MB")


async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(
        description="Test M15 EMA Crossover Plugin Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with current time
  python test.py -s AAPL -d LONG
  
  # Test with specific time
  python test.py -s TSLA -d SHORT -t "2025-01-15 10:30:00"
  
  # Test multiple scenarios
  python test.py -s SPY -d LONG --scenarios
        """
    )
    
    parser.add_argument('-s', '--symbol', 
                       default='AAPL',
                       help='Stock symbol to analyze')
    
    parser.add_argument('-d', '--direction',
                       choices=['LONG', 'SHORT'],
                       default='LONG',
                       help='Trade direction')
    
    parser.add_argument('-t', '--time',
                       help='Entry time (YYYY-MM-DD HH:MM:SS) in UTC')
    
    parser.add_argument('--scenarios',
                       action='store_true',
                       help='Run multiple test scenarios')
    
    args = parser.parse_args()
    
    # Parse entry time
    if args.time:
        try:
            entry_time = datetime.strptime(args.time, '%Y-%m-%d %H:%M:%S')
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"Error: Invalid time format. Use YYYY-MM-DD HH:MM:SS")
            return
    else:
        # Use current time minus 1 hour (to ensure market data)
        entry_time = datetime.now(timezone.utc) - timedelta(hours=1)
    
    # Create tester
    tester = PluginTester()
    
    if args.scenarios:
        # Run multiple scenarios
        print("\nRunning multiple test scenarios...")
        
        scenarios = [
            ('AAPL', 'LONG', entry_time),
            ('AAPL', 'SHORT', entry_time),
            ('TSLA', 'LONG', entry_time - timedelta(days=1)),
            ('SPY', 'LONG', entry_time - timedelta(hours=4)),
            ('NVDA', 'SHORT', entry_time - timedelta(hours=2)),
        ]
        
        for symbol, direction, time in scenarios:
            signal = await tester.run_analysis(symbol, time, direction)
            tester.display_results(signal)
            await asyncio.sleep(1)  # Respect rate limits
    else:
        # Run single test
        signal = await tester.run_analysis(args.symbol, entry_time, args.direction)
        tester.display_results(signal)
    
    # Show data report
    await tester.get_data_report()


if __name__ == "__main__":
    print("\n🚀 M15 EMA Crossover Plugin Test CLI")
    print("Testing complete workflow: Dashboard → Plugin → DataManager → CircuitBreaker → Signal")
    print(f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()