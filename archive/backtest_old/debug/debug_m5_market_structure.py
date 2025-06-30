# backtest/debug/debug_m5_market_structure.py
"""
Debug tool for M5 Market Structure calculations
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import pandas as pd
import logging
import sys
from pathlib import Path

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.debug.base_debug import BaseCalculationDebugger
from data.polygon_data_manager import PolygonDataManager
from modules.calculations.market_structure.m5_market_structure import (
    M5MarketStructureAnalyzer, 
    MarketStructureSignal
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class M5MarketStructureDebugger(BaseCalculationDebugger):
    """Debug tool for 5-minute market structure analysis"""
    
    def __init__(self, symbol: str, entry_time: datetime, 
                 lookback_minutes: int = 600, forward_minutes: int = 15,
                 fractal_length: int = 3):
        super().__init__(symbol, entry_time, lookback_minutes, forward_minutes)
        self.fractal_length = fractal_length
        self.data_manager = PolygonDataManager()
        
    @property
    def calculation_name(self) -> str:
        return "m5_market_structure"
    
    @property
    def timeframe(self) -> str:
        return "5-minute"
    
    async def run_debug(self) -> Dict[str, Any]:
        """Run debug analysis for M5 market structure"""
        # Load data
        start_time = self.entry_time - timedelta(minutes=self.lookback_minutes)
        end_time = self.entry_time + timedelta(minutes=self.forward_minutes)
        
        logger.info(f"\n{'='*100}")
        logger.info(f"M5 MARKET STRUCTURE DEBUG: {self.symbol}")
        logger.info(f"Entry Time: {self.entry_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"Data Range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%H:%M')} UTC")
        logger.info(f"Fractal Length: {self.fractal_length} (requires {self.fractal_length} bars on each side)")
        logger.info(f"Note: Analyzing 5-minute aggregated bars")
        logger.info(f"{'='*100}\n")
        
        # Load 1-minute bars and aggregate to 5-minute
        bars_1min = await self.data_manager.load_bars(
            symbol=self.symbol,
            start_time=start_time,
            end_time=end_time,
            timeframe='1min',
            use_cache=True
        )
        
        if bars_1min.empty:
            logger.error("No data available")
            return {}
        
        # Aggregate to 5-minute bars
        bars_5min = bars_1min.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        logger.info(f"Aggregated {len(bars_1min)} 1-minute bars into {len(bars_5min)} 5-minute bars")
        
        # Create analyzer
        analyzer = M5MarketStructureAnalyzer(
            fractal_length=self.fractal_length,
            buffer_size=100,
            min_candles_required=15
        )
        
        # Process bars and track everything
        results = {
            'symbol': self.symbol,
            'entry_time': self.entry_time,
            'fractal_length': self.fractal_length,
            'timeframe': '5-minute',
            'candles': [],
            'fractals': [],
            'structure_breaks': [],
            'signals': [],
            'final_state': None
        }
        
        # Process each 5-minute bar
        for idx, (timestamp, row) in enumerate(bars_5min.iterrows()):
            # Create candle data
            candle_dict = {
                't': timestamp,
                'o': float(row['open']),
                'h': float(row['high']),
                'l': float(row['low']),
                'c': float(row['close']),
                'v': float(row['volume'])
            }
            
            # Store candle info
            candle_info = {
                'index': idx,
                'time': timestamp.strftime('%H:%M'),
                'timestamp': timestamp,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': int(row['volume']),
                'is_entry': timestamp <= self.entry_time < timestamp + timedelta(minutes=5),
                'is_future': timestamp > self.entry_time
            }
            results['candles'].append(candle_info)
            
            # Only process up to entry time
            if timestamp > self.entry_time:
                continue
            
            # Process candle
            signal = analyzer.process_candle(self.symbol, candle_dict, is_complete=True)
            
            # Track new fractals (similar to M1 but for 5-min data)
            # ... (similar fractal tracking logic as M1)
            
            # Track signals
            if signal and signal.structure_type in ['BOS', 'CHoCH']:
                results['signals'].append({
                    'timestamp': timestamp,
                    'time': timestamp.strftime('%H:%M'),
                    'type': signal.structure_type,
                    'direction': signal.signal,
                    'strength': signal.strength,
                    'reason': signal.reason,
                    'metrics': signal.metrics
                })
        
        # Get final state
        final_analysis = analyzer.get_current_analysis(self.symbol)
        if final_analysis:
            results['final_state'] = {
                'trend': final_analysis.signal,
                'last_signal': final_analysis.structure_type,
                'strength': final_analysis.strength,
                'metrics': final_analysis.metrics,
                'reason': final_analysis.reason
            }
        
        # Display results
        self.display_results(results)
        
        return results
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display debug results for M5 market structure"""
        logger.info("5-MINUTE MARKET STRUCTURE ANALYSIS:")
        logger.info("-" * 100)
        
        # Similar display logic to M1 but adapted for 5-minute timeframe
        # Show 5-minute bars, fractals detected on 5-min timeframe, etc.
        
        # Key differences:
        # - Time displayed as HH:MM (no seconds)
        # - Fractal detection happens 3+ bars later (15+ minutes)
        # - Signals are less frequent but more significant


async def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug M5 Market Structure")
    parser.add_argument("symbol", help="Stock symbol")
    parser.add_argument("entry_time", help="Entry time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--lookback", type=int, default=600, help="Lookback minutes (default: 600)")
    parser.add_argument("--fractal-length", type=int, default=3, help="Fractal length (default: 3)")
    parser.add_argument("--save", action="store_true", help="Save debug report to file")
    
    args = parser.parse_args()
    
    # Parse entry time
    entry_time = datetime.strptime(args.entry_time, "%Y-%m-%d %H:%M:%S")
    entry_time = entry_time.replace(tzinfo=timezone.utc)
    
    # Create debugger
    debugger = M5MarketStructureDebugger(
        symbol=args.symbol,
        entry_time=entry_time,
        lookback_minutes=args.lookback,
        fractal_length=args.fractal_length
    )
    
    # Run debug
    results = await debugger.run_debug()
    
    # Save if requested
    if args.save:
        debugger.save_debug_report(results)


if __name__ == "__main__":
    asyncio.run(main())