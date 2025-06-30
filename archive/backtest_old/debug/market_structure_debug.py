# backtest/debug/market_structure_debug.py
"""
Market Structure Debug Visualization Tool
Shows exactly where fractals are detected and how signals are generated
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging
from pathlib import Path
import json

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.polygon_data_manager import PolygonDataManager
from modules.calculations.market_structure.m1_market_structure import (
    MarketStructureAnalyzer, 
    MarketStructureSignal,
    Fractal
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class MarketStructureDebugger:
    """Debug visualization for market structure calculations"""
    
    def __init__(self, symbol: str, entry_time: datetime, 
                 lookback_minutes: int = 120, forward_minutes: int = 0):
        """
        Initialize debugger
        
        Args:
            symbol: Stock symbol
            entry_time: Entry time (UTC)
            lookback_minutes: Minutes of historical data to analyze
            forward_minutes: Minutes of forward data (for validation)
        """
        self.symbol = symbol
        self.entry_time = entry_time.replace(tzinfo=timezone.utc) if entry_time.tzinfo is None else entry_time
        self.lookback_minutes = lookback_minutes
        self.forward_minutes = forward_minutes
        self.data_manager = PolygonDataManager()
        
    async def run_debug(self, fractal_length: int = 5) -> Dict:
        """Run debug analysis"""
        # Load data
        start_time = self.entry_time - timedelta(minutes=self.lookback_minutes)
        end_time = self.entry_time + timedelta(minutes=self.forward_minutes)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"MARKET STRUCTURE DEBUG: {self.symbol}")
        logger.info(f"Entry Time: {self.entry_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"Analyzing: {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')} UTC")
        logger.info(f"Fractal Length: {fractal_length}")
        logger.info(f"{'='*80}\n")
        
        # Load 1-minute bars
        bars = await self.data_manager.load_bars(
            symbol=self.symbol,
            start_time=start_time,
            end_time=end_time,
            timeframe='1min',
            use_cache=True
        )
        
        if bars.empty:
            logger.error("No data available")
            return {}
        
        # Create analyzer with debug enabled
        analyzer = MarketStructureAnalyzer(
            fractal_length=fractal_length,
            buffer_size=200,
            min_candles_required=fractal_length * 2 + 1
        )
        
        # Process bars and track fractals
        fractal_history = []
        structure_breaks = []
        candle_data = []
        
        # Convert bars to list of dicts
        for idx, (timestamp, row) in enumerate(bars.iterrows()):
            candle_dict = {
                't': timestamp,
                'o': float(row['open']),
                'h': float(row['high']),
                'l': float(row['low']),
                'c': float(row['close']),
                'v': float(row['volume']),
                'idx': idx
            }
            
            # Store candle data
            candle_data.append({
                'time': timestamp.strftime('%H:%M:%S'),
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': int(row['volume']),
                'is_entry': timestamp <= self.entry_time < timestamp + timedelta(minutes=1)
            })
            
            # Only process up to entry time
            if timestamp > self.entry_time:
                candle_data[-1]['future'] = True
                continue
                
            # Process candle
            signal = analyzer.process_candle(self.symbol, candle_dict, is_complete=True)
            
            # Track fractals after processing
            if self.symbol in analyzer.high_fractals:
                for fractal in analyzer.high_fractals[self.symbol]:
                    if not any(f['bar_index'] == fractal.bar_index and f['type'] == 'high' 
                              for f in fractal_history):
                        fractal_history.append({
                            'bar_index': fractal.bar_index,
                            'timestamp': fractal.timestamp,
                            'price': fractal.price,
                            'type': 'high',
                            'broken': fractal.broken,
                            'break_time': fractal.break_time,
                            'break_type': fractal.break_type
                        })
            
            if self.symbol in analyzer.low_fractals:
                for fractal in analyzer.low_fractals[self.symbol]:
                    if not any(f['bar_index'] == fractal.bar_index and f['type'] == 'low' 
                              for f in fractal_history):
                        fractal_history.append({
                            'bar_index': fractal.bar_index,
                            'timestamp': fractal.timestamp,
                            'price': fractal.price,
                            'type': 'low',
                            'broken': fractal.broken,
                            'break_time': fractal.break_time,
                            'break_type': fractal.break_type
                        })
            
            # Track structure breaks
            if signal and signal.structure_type in ['BOS', 'CHoCH']:
                structure_breaks.append({
                    'timestamp': timestamp,
                    'type': signal.structure_type,
                    'direction': signal.signal,
                    'strength': signal.strength,
                    'reason': signal.reason
                })
        
        # Get final analysis
        final_analysis = analyzer.get_current_analysis(self.symbol)
        
        # Display results
        self._display_results(candle_data, fractal_history, structure_breaks, 
                            final_analysis, analyzer)
        
        return {
            'candles': candle_data,
            'fractals': fractal_history,
            'breaks': structure_breaks,
            'final_analysis': final_analysis
        }
    
    def _display_results(self, candles: List[Dict], fractals: List[Dict], 
                        breaks: List[Dict], final_analysis: Optional[MarketStructureSignal],
                        analyzer: MarketStructureAnalyzer):
        """Display debug results in readable format"""
        
        # 1. Show detected fractals
        logger.info("DETECTED FRACTALS:")
        logger.info("-" * 80)
        logger.info(f"{'Time':^10} {'Type':^6} {'Price':^10} {'Status':^12} {'Break Type':^10}")
        logger.info("-" * 80)
        
        for fractal in sorted(fractals, key=lambda x: x['timestamp']):
            time_str = fractal['timestamp'].strftime('%H:%M:%S')
            status = "BROKEN" if fractal['broken'] else "ACTIVE"
            break_type = fractal['break_type'] or "-"
            
            logger.info(f"{time_str:^10} {fractal['type']:^6} {fractal['price']:^10.2f} "
                       f"{status:^12} {break_type:^10}")
        
        # 2. Show price action with fractal markers
        logger.info(f"\nPRICE ACTION WITH FRACTAL MARKERS:")
        logger.info("-" * 100)
        logger.info(f"{'Time':^10} {'Open':^8} {'High':^8} {'Low':^8} {'Close':^8} "
                   f"{'Volume':^10} {'Fractal':^15} {'Break':^15}")
        logger.info("-" * 100)
        
        # Create fractal lookup by bar index
        high_fractals = {f['bar_index']: f for f in fractals if f['type'] == 'high'}
        low_fractals = {f['bar_index']: f for f in fractals if f['type'] == 'low'}
        
        # Display candles with markers
        for idx, candle in enumerate(candles):
            time_str = candle['time']
            
            # Check for fractals at this index
            fractal_marker = ""
            if idx in high_fractals:
                fractal_marker = f"HIGH@{high_fractals[idx]['price']:.2f}"
            elif idx in low_fractals:
                fractal_marker = f"LOW@{low_fractals[idx]['price']:.2f}"
            
            # Check for breaks at this time
            break_marker = ""
            for brk in breaks:
                if brk['timestamp'].strftime('%H:%M:%S') == time_str:
                    break_marker = f"{brk['type']}-{brk['direction']}"
            
            # Mark entry bar
            if candle.get('is_entry'):
                time_str = f">{time_str}<"
            
            # Mark future bars
            if candle.get('future'):
                time_str = f"[{candle['time']}]"
            
            logger.info(f"{time_str:^10} {candle['open']:^8.2f} {candle['high']:^8.2f} "
                       f"{candle['low']:^8.2f} {candle['close']:^8.2f} "
                       f"{candle['volume']:^10,d} {fractal_marker:^15} {break_marker:^15}")
        
        # 3. Show structure breaks
        logger.info(f"\nSTRUCTURE BREAKS:")
        logger.info("-" * 80)
        if breaks:
            for brk in breaks:
                logger.info(f"{brk['timestamp'].strftime('%H:%M:%S')} - "
                           f"{brk['type']} {brk['direction']} - {brk['reason']}")
        else:
            logger.info("No structure breaks detected")
        
        # 4. Show final state
        logger.info(f"\nFINAL ANALYSIS AT ENTRY TIME:")
        logger.info("-" * 80)
        if final_analysis:
            metrics = final_analysis.metrics
            logger.info(f"Current Trend: {metrics['current_trend']}")
            logger.info(f"Last High Fractal: ${metrics['last_high_fractal']:.2f}" 
                       if metrics['last_high_fractal'] else "Last High Fractal: None")
            logger.info(f"Last Low Fractal: ${metrics['last_low_fractal']:.2f}"
                       if metrics['last_low_fractal'] else "Last Low Fractal: None")
            logger.info(f"Last Break: {metrics['last_break_type']} at ${metrics['last_break_price']:.2f}"
                       if metrics['last_break_type'] else "No recent breaks")
            logger.info(f"Signal: {final_analysis.signal} ({final_analysis.strength:.0f}% strength)")
            logger.info(f"Reason: {final_analysis.reason}")
        
        # 5. Show active (unbroken) fractals at entry
        logger.info(f"\nACTIVE FRACTALS AT ENTRY TIME:")
        logger.info("-" * 80)
        active_highs = [f for f in fractals if f['type'] == 'high' and not f['broken']]
        active_lows = [f for f in fractals if f['type'] == 'low' and not f['broken']]
        
        if active_highs:
            latest_high = max(active_highs, key=lambda x: x['timestamp'])
            logger.info(f"Most Recent High Fractal: ${latest_high['price']:.2f} "
                       f"at {latest_high['timestamp'].strftime('%H:%M:%S')}")
        
        if active_lows:
            latest_low = max(active_lows, key=lambda x: x['timestamp'])
            logger.info(f"Most Recent Low Fractal: ${latest_low['price']:.2f} "
                       f"at {latest_low['timestamp'].strftime('%H:%M:%S')}")
    
    async def save_debug_report(self, output_path: str = None):
        """Save debug report to file"""
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"debug_market_structure_{self.symbol}_{timestamp}.json"
        
        results = await self.run_debug()
        
        # Convert datetime objects to strings
        for fractal in results.get('fractals', []):
            if isinstance(fractal['timestamp'], datetime):
                fractal['timestamp'] = fractal['timestamp'].isoformat()
            if fractal['break_time'] and isinstance(fractal['break_time'], datetime):
                fractal['break_time'] = fractal['break_time'].isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nDebug report saved to: {output_path}")


async def debug_market_structure(symbol: str, entry_time: str, 
                               lookback_hours: float = 2.0,
                               fractal_length: int = 5):
    """
    Quick debug function for market structure
    
    Args:
        symbol: Stock symbol
        entry_time: Entry time string (YYYY-MM-DD HH:MM:SS)
        lookback_hours: Hours of historical data
        fractal_length: Fractal detection length
    """
    # Parse entry time
    entry_dt = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
    
    # Create debugger
    debugger = MarketStructureDebugger(
        symbol=symbol,
        entry_time=entry_dt,
        lookback_minutes=int(lookback_hours * 60),
        forward_minutes=5  # Show 5 minutes after entry for context
    )
    
    # Run debug
    await debugger.run_debug(fractal_length=fractal_length)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python market_structure_debug.py SYMBOL 'YYYY-MM-DD HH:MM:SS' [lookback_hours] [fractal_length]")
        print("Example: python market_structure_debug.py CRCL '2024-06-27 13:35:00' 2 5")
        sys.exit(1)
    
    symbol = sys.argv[1]
    entry_time = sys.argv[2]
    lookback_hours = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    fractal_length = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    
    asyncio.run(debug_market_structure(
        symbol=symbol,
        entry_time=entry_time,
        lookback_hours=lookback_hours,
        fractal_length=fractal_length
    ))