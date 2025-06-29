# backtest/core/signal_aggregator.py
"""
Signal aggregation using point & call system.
Combines multiple calculation signals into consensus.
"""

from typing import List, Dict, Any
from collections import Counter
import numpy as np
import logging

from ..adapters.base import StandardSignal

logger = logging.getLogger(__name__)


class SignalAggregator:
    """
    Aggregates signals from multiple calculations using point & call system.
    Provides consensus direction and confidence metrics.
    """
    
    def __init__(self):
        """Initialize signal aggregator"""
        self.aggregation_history = []
        
    def aggregate_signals(self, signals: List[StandardSignal]) -> Dict[str, Any]:
        """
        Aggregate multiple signals into consensus.
        
        Args:
            signals: List of StandardSignal from various calculations
            
        Returns:
            Dictionary with aggregation results
        """
        if not signals:
            return {
                'consensus_direction': 'NEUTRAL',
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'total_signals': 0,
                'agreement_score': 0,
                'average_strength': 0,
                'average_confidence': 0,
                'signals_by_category': {}
            }
            
        # Count directions
        directions = [s.direction for s in signals]
        direction_counts = Counter(directions)
        
        bullish_count = direction_counts.get('BULLISH', 0)
        bearish_count = direction_counts.get('BEARISH', 0)
        neutral_count = direction_counts.get('NEUTRAL', 0)
        total_signals = len(signals)
        
        # Determine consensus (simple majority)
        if bullish_count > bearish_count and bullish_count > neutral_count:
            consensus_direction = 'BULLISH'
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            consensus_direction = 'BEARISH'
        else:
            consensus_direction = 'NEUTRAL'
            
        # Calculate agreement score (0-100)
        if consensus_direction != 'NEUTRAL':
            agreeing_count = bullish_count if consensus_direction == 'BULLISH' else bearish_count
            agreement_score = (agreeing_count / total_signals) * 100
        else:
            # For neutral, agreement is how many are neutral
            agreement_score = (neutral_count / total_signals) * 100
            
        # Calculate weighted metrics
        total_strength = sum(s.strength for s in signals if s.direction != 'NEUTRAL')
        total_confidence = sum(s.confidence for s in signals if s.direction != 'NEUTRAL')
        non_neutral_count = bullish_count + bearish_count
        
        average_strength = total_strength / non_neutral_count if non_neutral_count > 0 else 0
        average_confidence = total_confidence / non_neutral_count if non_neutral_count > 0 else 0
        
        # Group signals by category
        signals_by_category = self._group_by_category(signals)
        
        # Advanced metrics
        result = {
            'consensus_direction': consensus_direction,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'total_signals': total_signals,
            'agreement_score': round(agreement_score, 2),
            'average_strength': round(average_strength, 2),
            'average_confidence': round(average_confidence, 2),
            'signals_by_category': signals_by_category,
            'strong_signals': self._identify_strong_signals(signals),
            'conflicting_signals': self._identify_conflicts(signals),
            'weighted_score': self._calculate_weighted_score(signals)
        }
        
        # Store for analysis
        self.aggregation_history.append({
            'timestamp': signals[0].timestamp if signals else None,
            'result': result
        })
        
        return result
        
    def _group_by_category(self, signals: List[StandardSignal]) -> Dict[str, Dict]:
        """Group signals by category (trend, volume, order flow)"""
        categories = {
            'trend': [],
            'volume': [],
            'order_flow': [],
            'momentum': [],
            'market_structure': []
        }
        
        for signal in signals:
            name_lower = signal.name.lower()
            
            if 'trend' in name_lower or 'statistical' in name_lower:
                categories['trend'].append(signal)
            elif 'volume' in name_lower or 'tick_flow' in name_lower:
                categories['volume'].append(signal)
            elif 'delta' in name_lower or 'bid_ask' in name_lower or 'trade_size' in name_lower:
                categories['order_flow'].append(signal)
            elif 'momentum' in name_lower:
                categories['momentum'].append(signal)
            else:
                categories['market_structure'].append(signal)
                
        # Summarize each category
        result = {}
        for category, cat_signals in categories.items():
            if cat_signals:
                cat_directions = [s.direction for s in cat_signals]
                cat_counter = Counter(cat_directions)
                
                result[category] = {
                    'total': len(cat_signals),
                    'bullish': cat_counter.get('BULLISH', 0),
                    'bearish': cat_counter.get('BEARISH', 0),
                    'neutral': cat_counter.get('NEUTRAL', 0),
                    'consensus': max(cat_counter, key=cat_counter.get)
                }
                
        return result
        
    def _identify_strong_signals(self, signals: List[StandardSignal]) -> List[Dict]:
        """Identify particularly strong signals"""
        strong_signals = []
        
        for signal in signals:
            if signal.strength >= 80 and signal.confidence >= 70:
                strong_signals.append({
                    'name': signal.name,
                    'direction': signal.direction,
                    'strength': signal.strength,
                    'confidence': signal.confidence
                })
                
        return strong_signals
        
    def _identify_conflicts(self, signals: List[StandardSignal]) -> List[Dict]:
        """Identify conflicting signals"""
        conflicts = []
        
        # Find signals with opposite directions and high confidence
        for i, sig1 in enumerate(signals):
            for sig2 in signals[i+1:]:
                if (sig1.direction == 'BULLISH' and sig2.direction == 'BEARISH' or
                    sig1.direction == 'BEARISH' and sig2.direction == 'BULLISH'):
                    if sig1.confidence >= 60 and sig2.confidence >= 60:
                        conflicts.append({
                            'signal1': sig1.name,
                            'signal2': sig2.name,
                            'direction1': sig1.direction,
                            'direction2': sig2.direction,
                            'confidence1': sig1.confidence,
                            'confidence2': sig2.confidence
                        })
                        
        return conflicts
        
    def _calculate_weighted_score(self, signals: List[StandardSignal]) -> float:
        """
        Calculate weighted directional score (-100 to +100).
        Positive = bullish, Negative = bearish
        """
        if not signals:
            return 0
            
        weighted_sum = 0
        total_weight = 0
        
        for signal in signals:
            # Weight by both strength and confidence
            weight = (signal.strength / 100) * (signal.confidence / 100)
            
            if signal.direction == 'BULLISH':
                weighted_sum += weight * 100
            elif signal.direction == 'BEARISH':
                weighted_sum -= weight * 100
            # NEUTRAL contributes 0
            
            if signal.direction != 'NEUTRAL':
                total_weight += weight
                
        if total_weight > 0:
            return round(weighted_sum / total_weight, 2)
        else:
            return 0
            
    def get_consensus_direction(self, signals: List[StandardSignal]) -> str:
        """Get just the consensus direction"""
        result = self.aggregate_signals(signals)
        return result['consensus_direction']
        
    def get_category_consensus(self, signals: List[StandardSignal], 
                             category: str) -> str:
        """Get consensus for specific category"""
        result = self.aggregate_signals(signals)
        cat_data = result['signals_by_category'].get(category, {})
        return cat_data.get('consensus', 'NEUTRAL')