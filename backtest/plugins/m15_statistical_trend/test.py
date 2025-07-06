# backtest/plugins/m15_statistical_trend/test.py
"""
Mock Backtest Tool - Tests the simplified 15-minute statistical trend plugin
"""

import asyncio
import argparse
from datetime import datetime, timezone
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent paths
current_file = Path(__file__).resolve()
sirius_dir = current_file.parent.parent.parent.parent
sys.path.insert(0, str(sirius_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the plugin
from backtest.plugins.m15_statistical_trend import run_analysis


async def test_statistical_trend(symbol: str, test_time: datetime, direction: str):
    """Run 15-minute statistical trend analysis test"""
    
    print(f"\n{'='*60}")
    print(f"15-MINUTE STATISTICAL TREND ANALYSIS")
    print(f"Symbol: {symbol}")
    print(f"Test Time: {test_time}")
    print(f"Direction: {direction}")
    print(f"{'='*60}\n")
    
    try:
        # Run the analysis
        print("Running market regime analysis...")
        result = await run_analysis(symbol, test_time, direction)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            return
        
        # Display results
        print(f"\nANALYSIS COMPLETE")
        print(f"{'='*60}")
        
        # Main regime signal
        details = result['details']
        signal = result['signal']
        
        print(f"\nMARKET REGIME: {details['regime']}")
        print(f"Daily Bias: {details['daily_bias']}")
        print(f"Signal Direction: {signal['direction']}")
        print(f"Confidence: {signal['confidence']:.0f}%")
        print(f"Trend Strength: {signal['strength']:.1f}%")
        print(f"Volatility State: {details['volatility_state']}")
        print(f"Volume Trend: {details['volume_trend']}")
        print(f"Vol-Adjusted Strength: {details['volatility_adjusted_strength']:.2f}")
        print(f"Current Price: ${details['price']:.2f}")
        
        # Alignment
        if details['aligned']:
            print(f"\n✅ REGIME ALIGNED with {direction} trade")
        else:
            print(f"\n⚠️ REGIME NOT ALIGNED with {direction} trade")
        
        # Display summary and description
        display = result['display_data']
        print(f"\nSUMMARY: {display['summary']}")
        print(f"DESCRIPTION: {display['description']}")
        
        # Display table
        print("\nDETAILED ANALYSIS:")
        print("-" * 50)
        for row in display['table_data']:
            print(f"{row[0]:.<25} {row[1]}")
        
        print(f"\n{'='*60}\n")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


class MockBacktestTool:
    """
    Simulates a backtest tool calling the plugin
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.trades = []
        
    async def analyze_trade(self, entry_time: datetime, direction: str) -> Dict[str, Any]:
        """
        Simulate analyzing a trade entry
        """
        print(f"\n{'='*60}")
        print(f"MOCK BACKTEST TOOL - TRADE ANALYSIS")
        print(f"{'='*60}")
        print(f"Symbol: {self.symbol}")
        print(f"Entry Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Direction: {direction}")
        print(f"{'='*60}\n")
        
        # Call the plugin
        print("Calling 15-Min Statistical Trend plugin...")
        result = await run_analysis(self.symbol, entry_time, direction)
        
        # Store trade
        self.trades.append({
            'time': entry_time,
            'direction': direction,
            'result': result
        })
        
        return result
    
    def display_results(self, result: Dict[str, Any]):
        """Display plugin results"""
        
        if 'error' in result:
            print(f"\n❌ ERROR: {result['error']}")
            return
        
        print("\n" + "="*60)
        print("PLUGIN RESULTS")
        print("="*60)
        
        # Signal data
        signal = result['signal']
        print(f"\nSIGNAL:")
        print(f"  Direction: {signal['direction']}")
        print(f"  Strength: {signal['strength']:.1f}%")
        print(f"  Confidence: {signal['confidence']:.0f}%")
        
        # Details
        details = result['details']
        print(f"\nDETAILS:")
        print(f"  Market Regime: {details['regime']}")
        print(f"  Daily Bias: {details['daily_bias']}")
        print(f"  Volatility: {details['volatility_state']}")
        print(f"  Volume Trend: {details['volume_trend']}")
        print(f"  Price: ${details['price']:.2f}")
        
        # Display data
        display = result['display_data']
        print(f"\nSUMMARY: {display['summary']}")
        print(f"DESCRIPTION: {display['description']}")
        
        # Table
        print("\nANALYSIS TABLE:")
        print("-" * 40)
        for row in display['table_data']:
            print(f"{row[0]:.<25} {row[1]}")
        
        # Trade decision
        print("\n" + "="*60)
        if details['aligned']:
            print("✅ TRADE DECISION: PROCEED - Regime aligned with direction")
        else:
            print("⚠️  TRADE DECISION: CAUTION - Regime not aligned with direction")
        print("="*60)
    
    async def run_batch_analysis(self, test_times: list):
        """Run analysis on multiple trades"""
        print(f"\nRunning batch analysis for {len(test_times)} trades...\n")
        
        for entry_time, direction in test_times:
            result = await self.analyze_trade(entry_time, direction)
            self.display_results(result)
            print("\n" + "-"*80 + "\n")
            await asyncio.sleep(0.5)  # Simulate processing delay
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary of all trades"""
        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)
        print(f"Total Trades: {len(self.trades)}")
        
        aligned = sum(1 for t in self.trades 
                     if 'details' in t['result'] and t['result']['details'].get('aligned', False))
        
        print(f"Aligned Trades: {aligned}/{len(self.trades)} ({aligned/len(self.trades)*100:.0f}%)")
        
        # Group by regime
        regimes = {}
        for trade in self.trades:
            if 'details' in trade['result']:
                regime = trade['result']['details']['regime']
                regimes[regime] = regimes.get(regime, 0) + 1
        
        print("\nRegime Distribution:")
        for regime, count in regimes.items():
            print(f"  {regime}: {count} ({count/len(self.trades)*100:.0f}%)")


async def main():
    """Run the test with CLI arguments"""
    parser = argparse.ArgumentParser(
        description='Test 15-Min Statistical Trend Plugin'
    )
    
    parser.add_argument(
        '-s', '--symbol',
        type=str,
        default='AAPL',
        help='Stock symbol (default: AAPL)'
    )
    
    parser.add_argument(
        '-t', '--time',
        type=str,
        default=None,
        help='Analysis time "YYYY-MM-DD HH:MM:SS" (default: batch test)'
    )
    
    parser.add_argument(
        '-d', '--direction',
        type=str,
        choices=['LONG', 'SHORT'],
        default='LONG',
        help='Trade direction (default: LONG)'
    )
    
    parser.add_argument(
        '--batch', 
        action='store_true',
        help='Run batch test with backtest tool'
    )
    
    args = parser.parse_args()
    
    if args.batch or (not args.time and not args.batch):
        # Use mock backtest tool
        backtest_tool = MockBacktestTool(args.symbol.upper())
        
        # Create test scenarios
        base_time = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        test_trades = [
            (base_time, 'LONG'),
            (base_time.replace(hour=14, minute=30), 'LONG'),
            (base_time.replace(hour=15, minute=30), 'SHORT'),
            (base_time.replace(hour=13, minute=0), 'SHORT'),
            (base_time.replace(hour=17, minute=30), 'LONG'),
        ]
        
        await backtest_tool.run_batch_analysis(test_trades)
    
    else:
        # Single test
        if args.time:
            try:
                test_time = datetime.strptime(args.time, "%Y-%m-%d %H:%M:%S")
                test_time = test_time.replace(tzinfo=timezone.utc)
            except ValueError:
                print(f"ERROR: Invalid datetime format: {args.time}")
                print("Please use format: YYYY-MM-DD HH:MM:SS")
                return
            
            # Run single test
            await test_statistical_trend(
                symbol=args.symbol.upper(),
                test_time=test_time,
                direction=args.direction
            )


if __name__ == "__main__":
    asyncio.run(main())