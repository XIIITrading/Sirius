# backtest/processing/bt_results_interpreter.py
"""
Backtest Results Interpreter
Normalizes all signal types from various calculation modules into a standardized format
for Supabase storage without modifying the original calculation code.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import json
from enum import Enum


class SignalType(Enum):
    """Types of signals from different calculation modules"""
    EMA = "ema"
    MARKET_STRUCTURE = "market_structure"
    BID_ASK_IMBALANCE = "bid_ask_imbalance"
    BUY_SELL_RATIO = "buy_sell_ratio"
    LARGE_ORDER_IMPACT = "large_order_impact"
    MICROSTRUCTURE = "microstructure"
    TRADE_SIZE_DIST = "trade_size_distribution"
    STATISTICAL_TREND = "statistical_trend"
    TICK_FLOW = "tick_flow"
    VOLUME_ANALYSIS = "volume_analysis"
    MARKET_CONTEXT = "market_context"
    HVN = "hvn"


@dataclass
class StandardizedSignal:
    """Standardized signal format for all indicator types"""
    # Core fields
    timestamp: datetime
    symbol: str
    timeframe: str
    indicator_type: str
    
    # Standardized signal
    bull_bear_signal: str  # BULL/BEAR/NEUTRAL
    strength: float  # 0-100 normalized
    confidence: float  # 0-1 normalized
    
    # Original signal info
    original_signal: str  # Preserve original signal format
    sub_type: Optional[str] = None  # e.g., BOS/CHoCH for market structure
    
    # Explanation
    reason: str = ""
    warnings: List[str] = None
    
    # Raw data preservation
    raw_metrics: Dict[str, Any] = None
    

class BTResultsInterpreter:
    """
    Interprets and normalizes signals from all backtest calculation modules
    """
    
    def __init__(self):
        self.processed_count = 0
        self.errors = []
        
    def interpret_signal(self, signal_data: Any, signal_type: SignalType, 
                        symbol: str, timeframe: str) -> StandardizedSignal:
        """
        Main entry point - routes signal to appropriate interpreter
        
        Args:
            signal_data: Raw signal object from calculation module
            signal_type: Type of signal (from SignalType enum)
            symbol: Trading symbol
            timeframe: Timeframe (1min, 5min, 15min, etc.)
            
        Returns:
            StandardizedSignal ready for database storage
        """
        
        interpreters = {
            SignalType.EMA: self._interpret_ema,
            SignalType.MARKET_STRUCTURE: self._interpret_market_structure,
            SignalType.BID_ASK_IMBALANCE: self._interpret_bid_ask,
            SignalType.BUY_SELL_RATIO: self._interpret_buy_sell_ratio,
            SignalType.LARGE_ORDER_IMPACT: self._interpret_large_order,
            SignalType.MICROSTRUCTURE: self._interpret_microstructure,
            SignalType.TRADE_SIZE_DIST: self._interpret_trade_size,
            SignalType.STATISTICAL_TREND: self._interpret_statistical_trend,
            SignalType.TICK_FLOW: self._interpret_tick_flow,
            SignalType.VOLUME_ANALYSIS: self._interpret_volume,
            SignalType.MARKET_CONTEXT: self._interpret_market_context,
            SignalType.HVN: self._interpret_hvn
        }
        
        interpreter = interpreters.get(signal_type)
        if not interpreter:
            raise ValueError(f"Unknown signal type: {signal_type}")
            
        try:
            standardized = interpreter(signal_data, symbol, timeframe)
            self.processed_count += 1
            return standardized
        except Exception as e:
            self.errors.append({
                'signal_type': signal_type.value,
                'error': str(e),
                'timestamp': datetime.now()
            })
            raise
    
    def _normalize_bull_bear(self, signal: str) -> str:
        """Normalize various bull/bear formats to BULL/BEAR/NEUTRAL"""
        signal_upper = str(signal).upper()
        
        # Direct mappings
        if signal_upper in ['BULL', 'BEAR', 'NEUTRAL']:
            return signal_upper
            
        # Bullish variants
        if any(x in signal_upper for x in ['BULLISH', 'BUY', 'LONG', 'ACCUMULATION']):
            return 'BULL'
            
        # Bearish variants
        if any(x in signal_upper for x in ['BEARISH', 'SELL', 'SHORT', 'DISTRIBUTION']):
            return 'BEAR'
            
        # Neutral variants
        if any(x in signal_upper for x in ['NEUTRAL', 'RANGE', 'MIXED', 'FLAT']):
            return 'NEUTRAL'
            
        return 'NEUTRAL'  # Default
    
    def _normalize_strength(self, strength: Any, max_value: float = 100) -> float:
        """Normalize strength to 0-100 scale"""
        if strength is None:
            return 50.0
            
        try:
            value = float(strength)
            if max_value == 1:  # Already 0-1 scale
                return min(100, value * 100)
            return min(100, max(0, value))
        except:
            return 50.0
    
    def _normalize_confidence(self, confidence: Any) -> float:
        """Normalize confidence to 0-1 scale"""
        if confidence is None:
            return 0.5
            
        try:
            value = float(confidence)
            if value > 1:  # Assume it's a percentage
                return min(1.0, value / 100)
            return min(1.0, max(0, value))
        except:
            return 0.5
    
    def _interpret_ema(self, signal_data: Any, symbol: str, timeframe: str) -> StandardizedSignal:
        """Interpret EMA signals (M1/M5/M15)"""
        return StandardizedSignal(
            timestamp=getattr(signal_data, 'timestamp', datetime.now()),
            symbol=symbol,
            timeframe=timeframe,
            indicator_type=SignalType.EMA.value,
            bull_bear_signal=self._normalize_bull_bear(signal_data.signal),
            strength=self._normalize_strength(signal_data.signal_strength),
            confidence=self._normalize_confidence(signal_data.signal_strength / 100),
            original_signal=signal_data.signal,
            reason=signal_data.reason,
            raw_metrics={
                'ema_9': signal_data.ema_9,
                'ema_21': signal_data.ema_21,
                'spread': signal_data.spread,
                'spread_pct': signal_data.spread_pct,
                'trend_strength': signal_data.trend_strength,
                'is_crossover': getattr(signal_data, 'is_crossover', False),
                'price_position': getattr(signal_data, 'price_position', '')
            }
        )
    
    def _interpret_market_structure(self, signal_data: Any, symbol: str, timeframe: str) -> StandardizedSignal:
        """Interpret Market Structure signals"""
        return StandardizedSignal(
            timestamp=signal_data.timestamp,
            symbol=symbol,
            timeframe=timeframe,
            indicator_type=SignalType.MARKET_STRUCTURE.value,
            bull_bear_signal=self._normalize_bull_bear(signal_data.signal),
            strength=self._normalize_strength(signal_data.strength),
            confidence=self._normalize_confidence(signal_data.strength / 100),
            original_signal=signal_data.signal,
            sub_type=signal_data.structure_type,  # BOS or CHoCH
            reason=signal_data.reason,
            raw_metrics=signal_data.metrics
        )
    
    def _interpret_bid_ask(self, signal_data: Any, symbol: str, timeframe: str) -> StandardizedSignal:
        """Interpret Bid/Ask Imbalance signals"""
        # Convert bull/bear scores to signal
        net_score = signal_data.bull_score - signal_data.bear_score
        if net_score > 0:
            signal = 'BULL'
        elif net_score < 0:
            signal = 'BEAR'
        else:
            signal = 'NEUTRAL'
            
        # Normalize score (-4 to +4) to strength (0-100)
        strength = abs(net_score) * 25  # Maps to 0, 25, 50, 75, 100
        
        return StandardizedSignal(
            timestamp=signal_data.timestamp,
            symbol=symbol,
            timeframe=timeframe,
            indicator_type=SignalType.BID_ASK_IMBALANCE.value,
            bull_bear_signal=signal,
            strength=strength,
            confidence=signal_data.confidence,
            original_signal=signal_data.signal_type,
            sub_type=signal_data.signal_strength,
            reason=signal_data.reason,
            warnings=signal_data.warnings,
            raw_metrics={
                'bull_score': signal_data.bull_score,
                'bear_score': signal_data.bear_score,
                'size_imbalance': signal_data.components.size_imbalance,
                'size_ratio': signal_data.components.size_ratio,
                'spread': signal_data.components.current_spread,
                'spread_ratio': signal_data.components.spread_ratio_1min,
                'liquidity_state': signal_data.components.liquidity_state,
                'trade_count': signal_data.trade_count
            }
        )
    
    def _interpret_buy_sell_ratio(self, signal_data: Any, symbol: str, timeframe: str) -> StandardizedSignal:
        """Interpret Buy/Sell Ratio signals"""
        return StandardizedSignal(
            timestamp=signal_data['timestamp'],
            symbol=symbol,
            timeframe=timeframe,
            indicator_type=SignalType.BUY_SELL_RATIO.value,
            bull_bear_signal=self._normalize_bull_bear(signal_data['signal']),
            strength=self._normalize_strength(signal_data['strength']),
            confidence=self._normalize_confidence(signal_data['strength'] / 100),
            original_signal=signal_data['signal'],
            reason=signal_data['reason'],
            raw_metrics=signal_data.get('metrics', {})
        )
    
    def _interpret_large_order(self, signal_data: Any, symbol: str, timeframe: str) -> StandardizedSignal:
        """Interpret Large Order Impact signals"""
        # Extract from current stats
        stats = signal_data  # Assuming get_current_stats() output
        net_pressure = stats.get('net_pressure', 0)
        
        if net_pressure > 0:
            signal = 'BULL'
        elif net_pressure < 0:
            signal = 'BEAR'
        else:
            signal = 'NEUTRAL'
            
        # Normalize pressure to strength
        strength = min(100, abs(net_pressure) / 1000)  # Adjust scale as needed
        
        return StandardizedSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            indicator_type=SignalType.LARGE_ORDER_IMPACT.value,
            bull_bear_signal=signal,
            strength=strength,
            confidence=0.7,  # Default confidence
            original_signal=stats.get('pressure_direction', 'NEUTRAL'),
            reason=stats.get('interpretation', ''),
            raw_metrics=stats
        )
    
    def _interpret_microstructure(self, signal_data: Any, symbol: str, timeframe: str) -> StandardizedSignal:
        """Interpret Microstructure Momentum signals"""
        net_signal = signal_data.net_signal
        
        if net_signal > 0:
            signal = 'BULL'
        elif net_signal < 0:
            signal = 'BEAR'
        else:
            signal = 'NEUTRAL'
            
        # Map net signal (-2 to +2) to strength
        strength = abs(net_signal) * 50  # 0, 50, 100
        
        return StandardizedSignal(
            timestamp=signal_data.timestamp,
            symbol=symbol,
            timeframe=timeframe,
            indicator_type=SignalType.MICROSTRUCTURE.value,
            bull_bear_signal=signal,
            strength=strength,
            confidence=signal_data.confidence,
            original_signal=f"Bull: {signal_data.bull_score}, Bear: {signal_data.bear_score}",
            reason=' | '.join(signal_data.reasons),
            warnings=signal_data.warnings,
            raw_metrics={
                'bull_score': signal_data.bull_score,
                'bear_score': signal_data.bear_score,
                'net_signal': signal_data.net_signal,
                'bid_size': signal_data.metrics.bid_size,
                'ask_size': signal_data.metrics.ask_size,
                'size_imbalance': signal_data.metrics.size_imbalance,
                'spread': signal_data.metrics.current_spread,
                'microstructure_quality': signal_data.metrics.microstructure_quality
            }
        )
    
    def _interpret_trade_size(self, signal_data: Any, symbol: str, timeframe: str) -> StandardizedSignal:
        """Interpret Trade Size Distribution signals"""
        # Similar to bid/ask interpretation
        net_score = signal_data.bull_score - signal_data.bear_score
        if net_score > 0:
            signal = 'BULL'
        elif net_score < 0:
            signal = 'BEAR'
        else:
            signal = 'NEUTRAL'
            
        strength = abs(net_score) * 25
        
        return StandardizedSignal(
            timestamp=signal_data.timestamp,
            symbol=symbol,
            timeframe=timeframe,
            indicator_type=SignalType.TRADE_SIZE_DIST.value,
            bull_bear_signal=signal,
            strength=strength,
            confidence=signal_data.confidence,
            original_signal=signal_data.signal_type,
            sub_type=signal_data.signal_strength,
            reason=signal_data.reason,
            warnings=signal_data.warnings,
            raw_metrics={
                'bull_score': signal_data.bull_score,
                'bear_score': signal_data.bear_score,
                'volume_weighted_ratio': signal_data.components.volume_weighted_ratio,
                'zscore': signal_data.components.zscore,
                'institutional_score': signal_data.components.institutional_score,
                'trade_count': signal_data.trade_count
            }
        )
    
    def _interpret_statistical_trend(self, signal_data: Any, symbol: str, timeframe: str) -> StandardizedSignal:
        """Interpret Statistical Trend signals (various formats)"""
        # Handle different formats based on timeframe
        if hasattr(signal_data, 'signal'):
            # 1-min and 5-min format
            original_signal = signal_data.signal
            
            # Map signals like 'STRONG BUY' to BULL
            if 'BUY' in original_signal or 'UP' in original_signal:
                signal = 'BULL'
            elif 'SELL' in original_signal or 'DOWN' in original_signal:
                signal = 'BEAR'
            else:
                signal = 'NEUTRAL'
                
        elif hasattr(signal_data, 'regime'):
            # 15-min format
            original_signal = signal_data.regime
            if signal_data.daily_bias in ['LONG ONLY', 'LONG BIAS']:
                signal = 'BULL'
            elif signal_data.daily_bias in ['SHORT ONLY', 'SHORT BIAS']:
                signal = 'BEAR'
            else:
                signal = 'NEUTRAL'
        else:
            signal = 'NEUTRAL'
            original_signal = 'UNKNOWN'
            
        return StandardizedSignal(
            timestamp=signal_data.timestamp,
            symbol=symbol,
            timeframe=timeframe,
            indicator_type=SignalType.STATISTICAL_TREND.value,
            bull_bear_signal=signal,
            strength=self._normalize_strength(signal_data.confidence),
            confidence=self._normalize_confidence(signal_data.confidence / 100),
            original_signal=original_signal,
            sub_type=getattr(signal_data, 'bias', None) or getattr(signal_data, 'daily_bias', None),
            reason=getattr(signal_data, 'reason', ''),
            raw_metrics={
                'trend_strength': signal_data.trend_strength,
                'volatility_adjusted_strength': signal_data.volatility_adjusted_strength,
                'volume_confirmation': getattr(signal_data, 'volume_confirmation', None),
                'volatility_state': getattr(signal_data, 'volatility_state', None)
            }
        )
    
    def _interpret_tick_flow(self, signal_data: Any, symbol: str, timeframe: str) -> StandardizedSignal:
        """Interpret Tick Flow signals"""
        return self._interpret_volume(signal_data, symbol, timeframe)  # Same format
    
    def _interpret_volume(self, signal_data: Any, symbol: str, timeframe: str) -> StandardizedSignal:
        """Interpret Volume Analysis signals (standard VolumeSignal format)"""
        return StandardizedSignal(
            timestamp=signal_data.timestamp,
            symbol=symbol,
            timeframe=timeframe,
            indicator_type=SignalType.VOLUME_ANALYSIS.value,
            bull_bear_signal=self._normalize_bull_bear(signal_data.signal),
            strength=self._normalize_strength(signal_data.strength),
            confidence=self._normalize_confidence(signal_data.strength / 100),
            original_signal=signal_data.signal,
            reason=signal_data.reason,
            raw_metrics=signal_data.metrics
        )
    
    def _interpret_market_context(self, signal_data: Any, symbol: str, timeframe: str) -> StandardizedSignal:
        """Interpret Market Context signals"""
        return self._interpret_volume(signal_data, symbol, timeframe)  # Same format
    
    def _interpret_hvn(self, signal_data: Any, symbol: str, timeframe: str) -> StandardizedSignal:
        """Interpret HVN signals"""
        # HVN doesn't produce bull/bear signals directly, it's support/resistance
        # We'll interpret based on price position relative to HVN levels
        
        # This is a placeholder - actual implementation would need price data
        return StandardizedSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            indicator_type=SignalType.HVN.value,
            bull_bear_signal='NEUTRAL',
            strength=50,
            confidence=0.8,
            original_signal='HVN_LEVELS',
            reason=f"Found {len(signal_data.clusters)} HVN clusters",
            raw_metrics={
                'clusters': len(signal_data.clusters),
                'price_range': signal_data.price_range,
                'hvn_unit': signal_data.hvn_unit,
                'filtered_levels': len(signal_data.filtered_levels)
            }
        )
    
    def prepare_for_supabase(self, standardized_signal: StandardizedSignal) -> Dict[str, Any]:
        """
        Prepare standardized signal for Supabase insertion
        
        Returns dict ready for database insertion
        """
        # Main summary record
        summary = {
            'timestamp': standardized_signal.timestamp.isoformat(),
            'symbol': standardized_signal.symbol,
            'timeframe': standardized_signal.timeframe,
            'indicator_type': standardized_signal.indicator_type,
            'bull_bear_signal': standardized_signal.bull_bear_signal,
            'strength': standardized_signal.strength,
            'confidence': standardized_signal.confidence,
            'original_signal': standardized_signal.original_signal,
            'sub_type': standardized_signal.sub_type,
            'reason': standardized_signal.reason,
            'warnings': json.dumps(standardized_signal.warnings) if standardized_signal.warnings else None
        }
        
        # Detail records for metrics
        details = []
        if standardized_signal.raw_metrics:
            for key, value in standardized_signal.raw_metrics.items():
                details.append({
                    'metric_name': key,
                    'metric_value': str(value),  # Convert to string for flexibility
                    'metric_type': type(value).__name__
                })
                
        return {
            'summary': summary,
            'details': details
        }
    
    def batch_interpret(self, signals: List[Dict[str, Any]]) -> List[StandardizedSignal]:
        """
        Process multiple signals at once
        
        Args:
            signals: List of dicts with 'data', 'type', 'symbol', 'timeframe'
            
        Returns:
            List of StandardizedSignal objects
        """
        results = []
        
        for signal_info in signals:
            try:
                standardized = self.interpret_signal(
                    signal_data=signal_info['data'],
                    signal_type=SignalType(signal_info['type']),
                    symbol=signal_info['symbol'],
                    timeframe=signal_info['timeframe']
                )
                results.append(standardized)
            except Exception as e:
                print(f"Error processing signal: {e}")
                continue
                
        return results
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'processed_count': self.processed_count,
            'error_count': len(self.errors),
            'errors': self.errors[-10:]  # Last 10 errors
        }


# Example usage
if __name__ == "__main__":
    # Create interpreter
    interpreter = BTResultsInterpreter()
    
    # Example: Interpret an EMA signal
    class MockEMASignal:
        signal = 'BULL'
        signal_strength = 75.5
        timestamp = datetime.now()
        ema_9 = 150.25
        ema_21 = 149.80
        spread = 0.45
        spread_pct = 0.30
        trend_strength = 80
        reason = "EMA 9 > EMA 21 with strong trend"
        
    ema_signal = MockEMASignal()
    standardized = interpreter.interpret_signal(
        signal_data=ema_signal,
        signal_type=SignalType.EMA,
        symbol='AAPL',
        timeframe='5min'
    )
    
    print("Standardized Signal:")
    print(f"  Bull/Bear: {standardized.bull_bear_signal}")
    print(f"  Strength: {standardized.strength}")
    print(f"  Confidence: {standardized.confidence}")
    
    # Prepare for database
    db_data = interpreter.prepare_for_supabase(standardized)
    print("\nDatabase Format:")
    print("Summary:", db_data['summary'])
    print("Details:", db_data['details'])