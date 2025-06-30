# backtest/debug/debug_m1_market_structure.py
"""
Debug tool for M1 Market Structure calculations
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
from modules.calculations.market_structure.m1_market_structure import (
    MarketStructureAnalyzer, 
    MarketStructureSignal,
    Fractal
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class M1MarketStructureDebugger(BaseCalculationDebugger):
    """Debug tool for 1-minute market structure analysis"""
    
    def __init__(self, symbol: str, entry_time: datetime, 
                 lookback_minutes: int = 120, forward_minutes: int = 5,
                 fractal_length: int = 5):
        super().__init__(symbol, entry_time, lookback_minutes, forward_minutes)
        self.fractal_length = fractal_length
        self.data_manager = PolygonDataManager()
        
    @property
    def calculation_name(self) -> str:
        return "m1_market_structure"
    
    @property
    def timeframe(self) -> str:
        return "1-minute"
    
    async def run_debug(self) -> Dict[str, Any]:
        """Run debug analysis for M1 market structure"""
        # Load data
        start_time = self.entry_time - timedelta(minutes=self.lookback_minutes)
        end_time = self.entry_time + timedelta(minutes=self.forward_minutes)
        
        logger.info(f"\n{'='*100}")
        logger.info(f"M1 MARKET STRUCTURE DEBUG: {self.symbol}")
        logger.info(f"Entry Time: {self.entry_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"Data Range: {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')} UTC")
        logger.info(f"Fractal Length: {self.fractal_length} (requires {self.fractal_length} bars on each side)")
        logger.info(f"{'='*100}\n")
        
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
        
        # Create analyzer
        analyzer = MarketStructureAnalyzer(
            fractal_length=self.fractal_length,
            buffer_size=200,
            min_candles_required=self.fractal_length * 2 + 1
        )
        
        # Process bars and track everything
        results = {
            'symbol': self.symbol,
            'entry_time': self.entry_time,
            'fractal_length': self.fractal_length,
            'candles': [],
            'fractals': [],
            'structure_breaks': [],
            'signals': [],
            'final_state': None
        }
        
        # Process each bar
        for idx, (timestamp, row) in enumerate(bars.iterrows()):
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
                'time': timestamp.strftime('%H:%M:%S'),
                'timestamp': timestamp,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': int(row['volume']),
                'is_entry': timestamp <= self.entry_time < timestamp + timedelta(minutes=1),
                'is_future': timestamp > self.entry_time
            }
            results['candles'].append(candle_info)
            
            # Only process up to entry time
            if timestamp > self.entry_time:
                continue
            
            # Process candle
            signal = analyzer.process_candle(self.symbol, candle_dict, is_complete=True)
            
            # Track new fractals
            if self.symbol in analyzer.high_fractals:
                for fractal in analyzer.high_fractals[self.symbol]:
                    if not any(f['bar_index'] == fractal.bar_index and f['type'] == 'high' 
                              for f in results['fractals']):
                        results['fractals'].append({
                            'bar_index': fractal.bar_index,
                            'candle_index': fractal.bar_index,
                            'timestamp': fractal.timestamp,
                            'time': fractal.timestamp.strftime('%H:%M:%S'),
                            'price': fractal.price,
                            'type': 'high',
                            'broken': fractal.broken,
                            'break_time': fractal.break_time,
                            'break_type': fractal.break_type
                        })
            
            if self.symbol in analyzer.low_fractals:
                for fractal in analyzer.low_fractals[self.symbol]:
                    if not any(f['bar_index'] == fractal.bar_index and f['type'] == 'low' 
                              for f in results['fractals']):
                        results['fractals'].append({
                            'bar_index': fractal.bar_index,
                            'candle_index': fractal.bar_index,
                            'timestamp': fractal.timestamp,
                            'time': fractal.timestamp.strftime('%H:%M:%S'),
                            'price': fractal.price,
                            'type': 'low',
                            'broken': fractal.broken,
                            'break_time': fractal.break_time,
                            'break_type': fractal.break_type
                        })
            
            # Track signals
            if signal and signal.structure_type in ['BOS', 'CHoCH']:
                results['signals'].append({
                    'timestamp': timestamp,
                    'time': timestamp.strftime('%H:%M:%S'),
                    'type': signal.structure_type,
                    'direction': signal.signal,
                    'strength': signal.strength,
                    'reason': signal.reason,
                    'metrics': signal.metrics
                })
                
                # Track structure break
                results['structure_breaks'].append({
                    'timestamp': timestamp,
                    'time': timestamp.strftime('%H:%M:%S'),
                    'type': signal.structure_type,
                    'direction': signal.signal
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
        
        # Get active fractals at entry
        results['active_fractals_at_entry'] = {
            'high': [f for f in results['fractals'] if f['type'] == 'high' and not f['broken']],
            'low': [f for f in results['fractals'] if f['type'] == 'low' and not f['broken']]
        }
        
        # Display results
        self.display_results(results)
        
        return results
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display debug results for M1 market structure"""
        
        # 1. Fractal Detection Timeline
        logger.info("FRACTAL DETECTION TIMELINE:")
        logger.info("-" * 100)
        logger.info("Note: Fractals are detected AFTER they are confirmed (5+ bars later)")
        logger.info(f"{'Bar#':^6} {'Time':^10} {'Type':^6} {'Price':^10} {'Status':^12} {'Break Type':^10} Notes")
        logger.info("-" * 100)
        
        for fractal in sorted(results['fractals'], key=lambda x: x['timestamp']):
            bar_num = fractal['bar_index']
            time_str = fractal['time']
            status = "BROKEN" if fractal['broken'] else "ACTIVE"
            break_type = fractal['break_type'] or "-"
            
            # Calculate detection delay
            candle_time = results['candles'][bar_num]['time']
            detection_bar = bar_num + self.fractal_length + 1
            if detection_bar < len(results['candles']):
                detection_time = results['candles'][detection_bar]['time']
                note = f"Detected at {detection_time}"
            else:
                note = "Detection pending"
            
            logger.info(f"{bar_num:^6} {time_str:^10} {fractal['type']:^6} "
                       f"{fractal['price']:^10.2f} {status:^12} {break_type:^10} {note}")
        
        # 2. Price Action with Markers
        logger.info(f"\nPRICE ACTION WITH FRACTAL & SIGNAL MARKERS:")
        logger.info("-" * 120)
        logger.info(f"{'Bar#':^6} {'Time':^10} {'Open':^8} {'High':^8} {'Low':^8} {'Close':^8} "
                   f"{'Volume':^10} {'Fractal':^20} {'Signal':^20}")
        logger.info("-" * 120)
        
        # Create lookups
        high_fractals = {f['bar_index']: f for f in results['fractals'] if f['type'] == 'high'}
        low_fractals = {f['bar_index']: f for f in results['fractals'] if f['type'] == 'low'}
        signals_by_time = {s['time']: s for s in results['signals']}
        
        for candle in results['candles']:
            if candle['is_future']:
                continue
                
            idx = candle['index']
            time_str = candle['time']
            
            # Check for fractals
            fractal_marker = ""
            if idx in high_fractals:
                status = "!" if high_fractals[idx]['broken'] else ""
                fractal_marker = f"HIGH@{high_fractals[idx]['price']:.2f}{status}"
            elif idx in low_fractals:
                status = "!" if low_fractals[idx]['broken'] else ""
                fractal_marker = f"LOW@{low_fractals[idx]['price']:.2f}{status}"
            
            # Check for signals
            signal_marker = ""
            if time_str in signals_by_time:
                sig = signals_by_time[time_str]
                signal_marker = f"{sig['type']}-{sig['direction']}"
            
            # Mark entry bar
            if candle.get('is_entry'):
                time_str = f">{time_str}<"
            
            logger.info(f"{idx:^6} {time_str:^10} {candle['open']:^8.2f} {candle['high']:^8.2f} "
                       f"{candle['low']:^8.2f} {candle['close']:^8.2f} "
                       f"{candle['volume']:^10,d} {fractal_marker:^20} {signal_marker:^20}")
        
        # 3. Structure Breaks Summary
        logger.info(f"\nSTRUCTURE BREAKS DETECTED:")
        logger.info("-" * 80)
        if results['structure_breaks']:
            for brk in results['structure_breaks']:
                logger.info(f"{brk['time']} - {brk['type']:^6} {brk['direction']:^8}")
        else:
            logger.info("No structure breaks detected before entry")
        
        # 4. Final State at Entry
        logger.info(f"\nFINAL STATE AT ENTRY TIME:")
        logger.info("-" * 80)
        if results['final_state']:
            state = results['final_state']
            metrics = state['metrics']
            logger.info(f"Current Trend: {state['trend']}")
            logger.info(f"Signal Strength: {state['strength']:.0f}%")
            logger.info(f"Last Structure: {metrics.get('last_break_type', 'None')}")
            if metrics.get('last_high_fractal'):
                logger.info(f"Last High Fractal: ${metrics['last_high_fractal']:.2f}")
            if metrics.get('last_low_fractal'):
                logger.info(f"Last Low Fractal: ${metrics['last_low_fractal']:.2f}")
            logger.info(f"Reason: {state['reason']}")
        
        # 5. Active Fractals at Entry
        logger.info(f"\nACTIVE (UNBROKEN) FRACTALS AT ENTRY:")
        logger.info("-" * 80)
        active = results['active_fractals_at_entry']
        
        if active['high']:
            latest_high = max(active['high'], key=lambda x: x['timestamp'])
            logger.info(f"Most Recent High: ${latest_high['price']:.2f} at {latest_high['time']} "
                       f"(Bar #{latest_high['bar_index']})")
            logger.info(f"Total Active Highs: {len(active['high'])}")
        else:
            logger.info("No active high fractals")
        
        if active['low']:
            latest_low = max(active['low'], key=lambda x: x['timestamp'])
            logger.info(f"Most Recent Low: ${latest_low['price']:.2f} at {latest_low['time']} "
                       f"(Bar #{latest_low['bar_index']})")
            logger.info(f"Total Active Lows: {len(active['low'])}")
        else:
            logger.info("No active low fractals")
        
        # 6. Signal Analysis
        logger.info(f"\nSIGNAL ANALYSIS:")
        logger.info("-" * 80)
        if results['signals']:
            bull_signals = [s for s in results['signals'] if s['direction'] == 'BULL']
            bear_signals = [s for s in results['signals'] if s['direction'] == 'BEAR']
            bos_signals = [s for s in results['signals'] if s['type'] == 'BOS']
            choch_signals = [s for s in results['signals'] if s['type'] == 'CHoCH']
            
            logger.info(f"Total Signals: {len(results['signals'])}")
            logger.info(f"Bullish: {len(bull_signals)} ({len(bull_signals)/len(results['signals'])*100:.0f}%)")
            logger.info(f"Bearish: {len(bear_signals)} ({len(bear_signals)/len(results['signals'])*100:.0f}%)")
            logger.info(f"BOS (Continuation): {len(bos_signals)}")
            logger.info(f"CHoCH (Reversal): {len(choch_signals)}")
            
            # Last signal details
            if results['signals']:
                last_signal = results['signals'][-1]
                logger.info(f"\nLast Signal Before Entry:")
                logger.info(f"  Time: {last_signal['time']}")
                logger.info(f"  Type: {last_signal['type']}")
                logger.info(f"  Direction: {last_signal['direction']}")
                logger.info(f"  Strength: {last_signal['strength']:.0f}%")
                logger.info(f"  Reason: {last_signal['reason']}")


async def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug M1 Market Structure")
    parser.add_argument("symbol", help="Stock symbol")
    parser.add_argument("entry_time", help="Entry time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--lookback", type=int, default=120, help="Lookback minutes (default: 120)")
    parser.add_argument("--fractal-length", type=int, default=5, help="Fractal length (default: 5)")
    parser.add_argument("--save", action="store_true", help="Save debug report to file")
    
    args = parser.parse_args()
    
    # Parse entry time
    entry_time = datetime.strptime(args.entry_time, "%Y-%m-%d %H:%M:%S")
    entry_time = entry_time.replace(tzinfo=timezone.utc)
    
    # Create debugger
    debugger = M1MarketStructureDebugger(
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