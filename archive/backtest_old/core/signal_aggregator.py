# backtest/core/signal_aggregator.py
"""
Signal aggregation using Point & Call voting system.
Implements democratic voting where each calculation gets one vote.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass 
class StandardSignal:
    """Standardized signal format from all calculations"""
    name: str                    # Calculation name
    timestamp: datetime          # UTC timestamp
    direction: str              # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: float            # 0-100
    confidence: float          # 0-100
    metadata: Dict[str, Any]   # Calculation-specific data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'name': self.name,
            'timestamp': self.timestamp.isoformat(),
            'direction': self.direction,
            'strength': self.strength,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class SignalAggregator:
    """
    Aggregates signals using Point & Call voting system.
    Each calculation gets 1 vote regardless of strength/confidence.
    """
    
    def __init__(self):
        """Initialize aggregator"""
        self.calculation_categories = {
            'trend': ['Statistical Trend 1min', 'Statistical Trend 5min', 'Statistical Trend 15min'],
            'order_flow': ['Bid/Ask Imbalance', 'Trade Size Distribution', 'Cumulative Delta', 'Micro Momentum'],
            'volume': ['Tick Flow', 'Volume Analysis 1min', 'Market Context', 'Session Profile', 'Cluster Analyzer'],
            'market_structure': ['HVN Engine', 'Volume Profile', 'Ranking Engine']
        }
        
    def aggregate_signals(self, signals: List[StandardSignal]) -> Dict[str, Any]:
        """
        Aggregate signals using democratic voting.
        
        Args:
            signals: List of signals from all calculations
            
        Returns:
            Dictionary with consensus results
        """
        if not signals:
            return {
                'consensus_direction': 'NEUTRAL',
                'agreement_score': 0,
                'vote_breakdown': {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0},
                'category_consensus': {},
                'average_strength': 0,
                'average_confidence': 0,
                'total_calculations': 0,
                'participating_calculations': 0
            }
            
        # Count votes
        vote_counts = {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0}
        category_votes = {cat: {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0} 
                         for cat in self.calculation_categories}
        
        # Strength and confidence aggregation (non-neutral only)
        non_neutral_strengths = []
        non_neutral_confidences = []
        
        # Process each signal
        for signal in signals:
            # Count vote
            vote_counts[signal.direction] += 1
            
            # Track by category
            category = self._get_signal_category(signal.name)
            if category:
                category_votes[category][signal.direction] += 1
                
            # Collect strength/confidence for non-neutral signals
            if signal.direction != 'NEUTRAL':
                non_neutral_strengths.append(signal.strength)
                non_neutral_confidences.append(signal.confidence)
                
        # Determine consensus
        total_votes = sum(vote_counts.values())
        consensus_direction = self._determine_consensus(vote_counts)
        
        # Calculate agreement score (0-100)
        if total_votes > 0:
            max_votes = max(vote_counts.values())
            agreement_score = (max_votes / total_votes) * 100
        else:
            agreement_score = 0
            
        # Average strength and confidence
        avg_strength = sum(non_neutral_strengths) / len(non_neutral_strengths) if non_neutral_strengths else 0
        avg_confidence = sum(non_neutral_confidences) / len(non_neutral_confidences) if non_neutral_confidences else 0
        
        # Category consensus
        category_consensus = {}
        for category, votes in category_votes.items():
            if sum(votes.values()) > 0:
                category_consensus[category] = self._determine_consensus(votes)
            else:
                category_consensus[category] = 'NO_DATA'
                
        return {
            'consensus_direction': consensus_direction,
            'agreement_score': round(agreement_score, 1),
            'vote_breakdown': vote_counts,
            'category_consensus': category_consensus,
            'average_strength': round(avg_strength, 1),
            'average_confidence': round(avg_confidence, 1),
            'total_calculations': 15,  # Always 15 in our system
            'participating_calculations': total_votes,
            'signals': [s.to_dict() for s in signals]  # Include raw signals
        }
        
    def _determine_consensus(self, vote_counts: Dict[str, int]) -> str:
        """
        Determine consensus direction from votes.
        Simple majority wins. Ties go to NEUTRAL.
        """
        bull_votes = vote_counts.get('BULLISH', 0)
        bear_votes = vote_counts.get('BEARISH', 0)
        neutral_votes = vote_counts.get('NEUTRAL', 0)
        
        if bull_votes > bear_votes and bull_votes > neutral_votes:
            return 'BULLISH'
        elif bear_votes > bull_votes and bear_votes > neutral_votes:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
            
    def _get_signal_category(self, signal_name: str) -> Optional[str]:
        """Get category for a signal name"""
        for category, names in self.calculation_categories.items():
            if signal_name in names:
                return category
        return None
        
    def get_consensus_direction(self, signals: List[StandardSignal]) -> str:
        """Quick method to just get consensus direction"""
        result = self.aggregate_signals(signals)
        return result['consensus_direction']