# backtest/plugins/m1_statistical_trend/test.py
"""
Interactive test interface for 1-Minute Statistical Trend Plugin
Supports both historical point-in-time and live streaming tests
"""

import asyncio
import argparse
from datetime import datetime, timedelta, timezone
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backtest.plugins.m1_statistical_trend import run_analysis
from modules.calculations.trend.statistical_trend_1min import StatisticalTrend1Min

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InteractiveTest:
    """Interactive test environment for the plugin"""
    
    def __init__(self):
        self.calculator = StatisticalTrend1Min()
        self.running = False
        
    async def test_historical(self, symbol: str, test_time: datetime, direction: str):
        """Test with historical data at specific point in time"""
        print(f"\n{'='*60}")
        print(f"HISTORICAL TEST - 1-MIN STATISTICAL TREND")
        print(f"{'='*60}")
        print(f"Symbol: {symbol}")
        print(f"Time: {test_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Direction: {direction}")
        print(f"{'='*60}\n")
        
        try:
            # Call plugin analysis
            result = await run_analysis(symbol, test_time, direction)
            
            # Display results
            self._display_results(result)
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            logger.error(f"Test failed: {e}", exc_info=True)
    
    async def test_live(self, symbol: str):
        """Start live monitoring test"""
        print(f"\n{'='*60}")
        print(f"LIVE MONITORING TEST - 1-MIN STATISTICAL TREND")
        print(f"{'='*60}")
        print(f"Symbol: {symbol}")
        print(f"Starting at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"{'='*60}\n")
        print("Press Ctrl+C to stop...\n")
        
        self.running = True
        
        try:
            # Callback for live updates
            async def on_signal_update(signal):
                print(f"\n[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] New Signal:")
                print(f"  Signal: {signal.signal}")
                print(f"  Price: ${signal.price:.2f}")
                print(f"  Confidence: {signal.confidence:.1f}%")
                print(f"  P-Value: {signal.p_value:.4f}")
                print(f"  Volatility: {signal.volatility:.2f}%")
                
                if signal.signal != 'NEUTRAL':
                    print(f"  🔔 ACTION: {signal.signal} signal detected!")
            
            # Start live monitoring
            await self.calculator.start_live_monitoring([symbol], on_signal_update)
            
            # Keep running until interrupted
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\nStopping live test...")
            self.running = False
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            logger.error(f"Live test failed: {e}", exc_info=True)
    
    def _display_results(self, result: dict):
        """Display analysis results - no calculations, just formatting"""
        if 'error' in result:
            print(f"❌ ERROR: {result['error']}")
            return
        
        # Main signal info
        signal = result['signal']
        print(f"SIGNAL: {signal['direction']}")
        print(f"Confidence: {signal['confidence']:.1f}%")
        print(f"Strength: {signal['strength']:.1f}%")
        
        # Details
        details = result.get('details', {})
        print(f"\nDETAILS:")
        print(f"Raw Signal: {details.get('raw_signal', 'N/A')}")
        print(f"P-Value: {details.get('p_value', 0):.4f}")
        print(f"Volatility: {details.get('volatility', 0):.2f}%")
        
        if details.get('aligned'):
            print(f"Alignment: ✅ Aligned")
        else:
            print(f"Alignment: ⚠️ Not aligned")
        
        # Display data
        display = result.get('display_data', {})
        if 'table_data' in display:
            print(f"\nSTATISTICAL TESTS:")
            print("-" * 50)
            for row in display['table_data']:
                print(f"{row[0]:<20} {row[1]}")
        
        print(f"\n{display.get('description', '')}")


async def interactive_menu():
    """Interactive command-line menu"""
    test = InteractiveTest()
    
    while True:
        print("\n" + "="*50)
        print("1-MIN STATISTICAL TREND TEST MENU")
        print("="*50)
        print("1. Test historical (specific time)")
        print("2. Test current time")
        print("3. Start live monitoring")
        print("4. Exit")
        print("="*50)
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            # Historical test
            symbol = input("Enter symbol (e.g., AAPL): ").strip().upper()
            date_str = input("Enter datetime (YYYY-MM-DD HH:MM:SS): ").strip()
            direction = input("Enter direction (LONG/SHORT): ").strip().upper()
            
            try:
                test_time = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                test_time = test_time.replace(tzinfo=timezone.utc)
                await test.test_historical(symbol, test_time, direction)
            except ValueError:
                print("❌ Invalid datetime format")
                
        elif choice == '2':
            # Current time test
            symbol = input("Enter symbol (e.g., AAPL): ").strip().upper()
            direction = input("Enter direction (LONG/SHORT): ").strip().upper()
            
            test_time = datetime.now(timezone.utc) - timedelta(minutes=1)
            await test.test_historical(symbol, test_time, direction)
            
        elif choice == '3':
            # Live monitoring
            symbol = input("Enter symbol (e.g., AAPL): ").strip().upper()
            await test.test_live(symbol)
            
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("❌ Invalid option")


def parse_arguments():
    """Parse command line arguments for quick testing"""
    parser = argparse.ArgumentParser(
        description='Test 1-Minute Statistical Trend Plugin'
    )
    
    parser.add_argument(
        '-s', '--symbol',
        type=str,
        help='Stock symbol (e.g., AAPL)'
    )
    
    parser.add_argument(
        '-t', '--time',
        type=str,
        help='Test time "YYYY-MM-DD HH:MM:SS" (default: 1 min ago)'
    )
    
    parser.add_argument(
        '-d', '--direction',
        type=str,
        choices=['LONG', 'SHORT'],
        default='LONG',
        help='Trade direction'
    )
    
    parser.add_argument(
        '-l', '--live',
        action='store_true',
        help='Start live monitoring'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Start interactive menu'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_arguments()
    
    if args.interactive or not args.symbol:
        # Interactive mode
        await interactive_menu()
    else:
        # Command line mode
        test = InteractiveTest()
        
        if args.live:
            # Live monitoring
            await test.test_live(args.symbol.upper())
        else:
            # Historical test
            if args.time:
                test_time = datetime.strptime(args.time, "%Y-%m-%d %H:%M:%S")
                test_time = test_time.replace(tzinfo=timezone.utc)
            else:
                test_time = datetime.now(timezone.utc) - timedelta(minutes=1)
            
            await test.test_historical(
                args.symbol.upper(),
                test_time,
                args.direction
            )


if __name__ == "__main__":
    # Example usage:
    # python test.py -i                                    # Interactive menu
    # python test.py -s AAPL -d LONG                      # Test AAPL LONG at current time
    # python test.py -s TSLA -t "2024-01-15 10:30:00"    # Test specific time
    # python test.py -s SPY -l                            # Live monitoring
    
    asyncio.run(main())