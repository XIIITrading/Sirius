# backtest/data/trade_quote_aligner.py
"""
Module: Trade Quote Aligner
Purpose: Align trade executions with quote data for accurate buy/sell classification
Features: Time-based alignment, confidence scoring, quote staleness detection, interpolation
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TradeSide(Enum):
    """Classification of trade side"""
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"
    MIDPOINT = "midpoint"


class AlignmentMethod(Enum):
    """Methods for aligning trades with quotes"""
    EXACT = "exact"              # Exact timestamp match
    BACKWARD = "backward"        # Most recent quote before trade
    FORWARD = "forward"          # Next quote after trade
    INTERPOLATED = "interpolated" # Interpolated between quotes


@dataclass
class AlignmentResult:
    """Result of aligning a single trade with quotes"""
    trade_time: datetime
    trade_price: float
    trade_size: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    spread: float
    spread_pct: float
    trade_side: TradeSide
    confidence: float
    alignment_method: AlignmentMethod
    quote_age_ms: float  # How old the quote is relative to trade
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentReport:
    """Summary report for trade/quote alignment"""
    total_trades: int
    aligned_trades: int
    failed_alignments: int
    avg_quote_age_ms: float
    max_quote_age_ms: float
    side_distribution: Dict[str, int]
    confidence_distribution: Dict[str, int]
    alignment_methods_used: Dict[str, int]
    avg_spread: float
    avg_spread_pct: float
    warnings: List[str] = field(default_factory=list)


class TradeQuoteAligner:
    """
    Aligns trade executions with quote data for accurate market microstructure analysis.
    
    Key Features:
    1. Time-based alignment with configurable tolerance
    2. Multiple classification algorithms (Lee-Ready, tick test, quote rule)
    3. Confidence scoring based on quote staleness and spread
    4. Interpolation for missing quotes
    5. Special handling for crossed/locked markets
    """
    
    def __init__(self,
                 max_quote_age_ms: float = 1000,    # 1 second
                 interpolation_limit_ms: float = 5000,  # 5 seconds
                 min_confidence_threshold: float = 0.5,
                 spread_outlier_pct: float = 5.0):
        """
        Initialize the aligner with configuration.
        
        Args:
            max_quote_age_ms: Maximum age for a quote to be considered valid
            interpolation_limit_ms: Maximum gap for interpolation
            min_confidence_threshold: Minimum confidence for classification
            spread_outlier_pct: Spread % to flag as outlier
        """
        self.max_quote_age_ms = max_quote_age_ms
        self.interpolation_limit_ms = interpolation_limit_ms
        self.min_confidence_threshold = min_confidence_threshold
        self.spread_outlier_pct = spread_outlier_pct
        
        logger.info(f"TradeQuoteAligner initialized with max_quote_age={max_quote_age_ms}ms")
    
    def align_trades_quotes(self, 
                           trades_df: pd.DataFrame,
                           quotes_df: pd.DataFrame,
                           method: str = 'backward') -> Tuple[pd.DataFrame, AlignmentReport]:
        """
        Align trades with quotes and classify buy/sell.
        
        Args:
            trades_df: DataFrame with trades (index=timestamp, columns=[price, size])
            quotes_df: DataFrame with quotes (index=timestamp, columns=[bid, ask, bid_size, ask_size])
            method: Alignment method ('backward', 'forward', 'nearest')
            
        Returns:
            Tuple of (aligned DataFrame, alignment report)
        """
        if trades_df.empty or quotes_df.empty:
            logger.warning("Empty trades or quotes DataFrame provided")
            return pd.DataFrame(), self._create_empty_report()
        
        # Ensure sorted by time
        trades_df = trades_df.sort_index()
        quotes_df = quotes_df.sort_index()
        
        # Perform alignment
        logger.info(f"Aligning {len(trades_df)} trades with {len(quotes_df)} quotes using {method} method")
        
        # Use pandas merge_asof for efficient time-based alignment
        aligned_df = self._perform_alignment(trades_df, quotes_df, method)
        
        # Classify trades
        aligned_df = self._classify_trades(aligned_df)
        
        # Calculate confidence scores
        aligned_df = self._calculate_confidence(aligned_df)
        
        # Generate report
        report = self._generate_report(aligned_df, len(trades_df))
        
        return aligned_df, report
    
    def _perform_alignment(self, trades_df: pd.DataFrame, 
                          quotes_df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Perform the actual alignment using merge_asof"""
        # Prepare data for merge
        trades_reset = trades_df.reset_index()
        quotes_reset = quotes_df.reset_index()
        
        # Rename columns to avoid conflicts
        trades_reset.columns = ['trade_time'] + [f'trade_{col}' for col in trades_df.columns]
        quotes_reset.columns = ['quote_time'] + [f'quote_{col}' for col in quotes_df.columns]
        
        # Perform time-based merge
        if method == 'backward':
            direction = 'backward'
        elif method == 'forward':
            direction = 'forward'
        else:
            direction = 'nearest'
        
        aligned = pd.merge_asof(
            trades_reset.sort_values('trade_time'),
            quotes_reset.sort_values('quote_time'),
            left_on='trade_time',
            right_on='quote_time',
            direction=direction,
            tolerance=pd.Timedelta(milliseconds=self.interpolation_limit_ms)
        )
        
        # Calculate quote age
        aligned['quote_age_ms'] = (
            aligned['trade_time'] - aligned['quote_time']
        ).dt.total_seconds() * 1000
        
        # Set alignment method
        aligned['alignment_method'] = AlignmentMethod.BACKWARD.value
        
        # Handle missing quotes with interpolation
        if aligned['quote_bid'].isna().any():
            aligned = self._interpolate_missing_quotes(aligned, quotes_reset)
        
        return aligned
    
    def _interpolate_missing_quotes(self, aligned_df: pd.DataFrame, 
                                   quotes_df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing quotes where possible"""
        missing_mask = aligned_df['quote_bid'].isna()
        
        if not missing_mask.any():
            return aligned_df
        
        logger.info(f"Interpolating {missing_mask.sum()} missing quotes")
        
        for idx in aligned_df[missing_mask].index:
            trade_time = aligned_df.loc[idx, 'trade_time']
            
            # Find surrounding quotes
            before_quotes = quotes_df[quotes_df['quote_time'] < trade_time]
            after_quotes = quotes_df[quotes_df['quote_time'] > trade_time]
            
            if len(before_quotes) > 0 and len(after_quotes) > 0:
                before = before_quotes.iloc[-1]
                after = after_quotes.iloc[0]
                
                # Check if gap is within interpolation limit
                gap_ms = (after['quote_time'] - before['quote_time']).total_seconds() * 1000
                
                if gap_ms <= self.interpolation_limit_ms:
                    # Linear interpolation
                    weight = ((trade_time - before['quote_time']).total_seconds() / 
                             (after['quote_time'] - before['quote_time']).total_seconds())
                    
                    aligned_df.loc[idx, 'quote_bid'] = (
                        before['quote_bid'] * (1 - weight) + after['quote_bid'] * weight
                    )
                    aligned_df.loc[idx, 'quote_ask'] = (
                        before['quote_ask'] * (1 - weight) + after['quote_ask'] * weight
                    )
                    aligned_df.loc[idx, 'quote_bid_size'] = int(
                        before['quote_bid_size'] * (1 - weight) + after['quote_bid_size'] * weight
                    )
                    aligned_df.loc[idx, 'quote_ask_size'] = int(
                        before['quote_ask_size'] * (1 - weight) + after['quote_ask_size'] * weight
                    )
                    aligned_df.loc[idx, 'alignment_method'] = AlignmentMethod.INTERPOLATED.value
                    aligned_df.loc[idx, 'quote_age_ms'] = 0  # Interpolated, so age is 0
        
        return aligned_df
    
    def _classify_trades(self, aligned_df: pd.DataFrame) -> pd.DataFrame:
        """Classify trades as buy/sell using multiple methods"""
        # Initialize classification columns
        aligned_df['spread'] = aligned_df['quote_ask'] - aligned_df['quote_bid']
        aligned_df['spread_pct'] = (aligned_df['spread'] / aligned_df['quote_bid']) * 100
        aligned_df['midpoint'] = (aligned_df['quote_bid'] + aligned_df['quote_ask']) / 2
        
        # Method 1: Lee-Ready Algorithm (primary method)
        aligned_df['trade_side'] = self._lee_ready_classification(aligned_df)
        
        # Method 2: Effective spread classification (backup)
        aligned_df['effective_spread_side'] = self._effective_spread_classification(aligned_df)
        
        # Method 3: Tick test (additional signal)
        aligned_df['tick_test'] = self._tick_test_classification(aligned_df)
        
        # Combine methods for final classification
        aligned_df = self._combine_classifications(aligned_df)
        
        return aligned_df
    
    def _lee_ready_classification(self, df: pd.DataFrame) -> pd.Series:
        """
        Lee-Ready algorithm for trade classification.
        - If trade price > midpoint: BUY
        - If trade price < midpoint: SELL
        - If trade price = midpoint: use tick test
        """
        conditions = [
            df['trade_price'] > df['midpoint'],
            df['trade_price'] < df['midpoint'],
            df['trade_price'] == df['midpoint']
        ]
        
        choices = [
            TradeSide.BUY.value,
            TradeSide.SELL.value,
            TradeSide.MIDPOINT.value
        ]
        
        return pd.Series(np.select(conditions, choices, default=TradeSide.UNKNOWN.value), index=df.index)
    
    def _effective_spread_classification(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify based on effective spread.
        - If closer to bid: SELL
        - If closer to ask: BUY
        """
        bid_distance = abs(df['trade_price'] - df['quote_bid'])
        ask_distance = abs(df['trade_price'] - df['quote_ask'])
        
        conditions = [
            bid_distance < ask_distance,
            ask_distance < bid_distance,
            bid_distance == ask_distance
        ]
        
        choices = [
            TradeSide.SELL.value,
            TradeSide.BUY.value,
            TradeSide.MIDPOINT.value
        ]
        
        return pd.Series(np.select(conditions, choices, default=TradeSide.UNKNOWN.value), index=df.index)
    
    def _tick_test_classification(self, df: pd.DataFrame) -> pd.Series:
        """
        Tick test: compare with previous trade price.
        - If price increased: BUY
        - If price decreased: SELL
        """
        price_change = df['trade_price'].diff()
        
        conditions = [
            price_change > 0,
            price_change < 0,
            price_change == 0
        ]
        
        choices = [
            TradeSide.BUY.value,
            TradeSide.SELL.value,
            TradeSide.UNKNOWN.value
        ]
        
        return pd.Series(np.select(conditions, choices, default=TradeSide.UNKNOWN.value), index=df.index)
    
    def _combine_classifications(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine multiple classification methods"""
        # For midpoint trades, use tick test
        midpoint_mask = df['trade_side'] == TradeSide.MIDPOINT.value
        df.loc[midpoint_mask, 'trade_side'] = df.loc[midpoint_mask, 'tick_test']
        
        # For remaining unknowns, use effective spread
        unknown_mask = df['trade_side'] == TradeSide.UNKNOWN.value
        df.loc[unknown_mask, 'trade_side'] = df.loc[unknown_mask, 'effective_spread_side']
        
        return df
    
    def _calculate_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate confidence score for each classification"""
        df['confidence'] = 1.0
        
        # Factor 1: Quote staleness (0-1, where 1 is fresh)
        staleness_factor = 1 - (df['quote_age_ms'].clip(0, self.max_quote_age_ms) / self.max_quote_age_ms)
        df['confidence'] *= staleness_factor
        
        # Factor 2: Spread reasonableness (0-1, where 1 is normal spread)
        normal_spread_mask = df['spread_pct'] <= self.spread_outlier_pct
        df.loc[~normal_spread_mask, 'confidence'] *= 0.5
        
        # Factor 3: Price position clarity (0-1, where 1 is clear buy/sell)
        price_position = np.minimum(
            abs(df['trade_price'] - df['quote_bid']),
            abs(df['trade_price'] - df['quote_ask'])
        ) / df['spread']
        price_clarity = 1 - (2 * price_position.clip(0, 0.5))  # 1 at bid/ask, 0 at midpoint
        df['confidence'] *= price_clarity
        
        # Factor 4: Crossed/locked market penalty
        crossed_mask = df['quote_bid'] >= df['quote_ask']
        df.loc[crossed_mask, 'confidence'] *= 0.3
        
        # Factor 5: Interpolated data penalty
        interpolated_mask = df['alignment_method'] == AlignmentMethod.INTERPOLATED.value
        df.loc[interpolated_mask, 'confidence'] *= 0.7
        
        # Ensure confidence is between 0 and 1
        df['confidence'] = df['confidence'].clip(0, 1)
        
        return df
    
    def _generate_report(self, aligned_df: pd.DataFrame, total_trades: int) -> AlignmentReport:
        """Generate summary report of alignment results"""
        if aligned_df.empty:
            return self._create_empty_report()
        
        # Calculate statistics
        aligned_trades = len(aligned_df[aligned_df['quote_bid'].notna()])
        failed_alignments = total_trades - aligned_trades
        
        # Side distribution
        side_dist = aligned_df['trade_side'].value_counts().to_dict()
        
        # Confidence distribution
        conf_bins = pd.cut(aligned_df['confidence'], 
                          bins=[0, 0.25, 0.5, 0.75, 1.0],
                          labels=['Low', 'Medium', 'High', 'Very High'])
        conf_dist = conf_bins.value_counts().to_dict()
        
        # Alignment method distribution
        method_dist = aligned_df['alignment_method'].value_counts().to_dict()
        
        # Generate warnings
        warnings = []
        
        high_staleness = aligned_df['quote_age_ms'] > self.max_quote_age_ms
        if high_staleness.any():
            warnings.append(f"{high_staleness.sum()} trades have stale quotes (>{self.max_quote_age_ms}ms)")
        
        low_confidence = aligned_df['confidence'] < self.min_confidence_threshold
        if low_confidence.any():
            warnings.append(f"{low_confidence.sum()} trades have low confidence (<{self.min_confidence_threshold})")
        
        wide_spreads = aligned_df['spread_pct'] > self.spread_outlier_pct
        if wide_spreads.any():
            warnings.append(f"{wide_spreads.sum()} trades have unusually wide spreads (>{self.spread_outlier_pct}%)")
        
        return AlignmentReport(
            total_trades=total_trades,
            aligned_trades=aligned_trades,
            failed_alignments=failed_alignments,
            avg_quote_age_ms=aligned_df['quote_age_ms'].mean() if aligned_trades > 0 else 0,
            max_quote_age_ms=aligned_df['quote_age_ms'].max() if aligned_trades > 0 else 0,
            side_distribution=side_dist,
            confidence_distribution=conf_dist,
            alignment_methods_used=method_dist,
            avg_spread=aligned_df['spread'].mean() if aligned_trades > 0 else 0,
            avg_spread_pct=aligned_df['spread_pct'].mean() if aligned_trades > 0 else 0,
            warnings=warnings
        )
    
    def _create_empty_report(self) -> AlignmentReport:
        """Create empty report when no data available"""
        return AlignmentReport(
            total_trades=0,
            aligned_trades=0,
            failed_alignments=0,
            avg_quote_age_ms=0,
            max_quote_age_ms=0,
            side_distribution={},
            confidence_distribution={},
            alignment_methods_used={},
            avg_spread=0,
            avg_spread_pct=0,
            warnings=["No data available for alignment"]
        )
    
    def calculate_order_flow_metrics(self, aligned_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate order flow metrics from aligned trades.
        
        Returns dict with:
        - buy_volume / sell_volume
        - buy_count / sell_count
        - net_order_flow
        - buy_sell_ratio
        - large_buy_volume / large_sell_volume
        """
        if aligned_df.empty:
            return {}
        
        # Filter by confidence
        confident_trades = aligned_df[aligned_df['confidence'] >= self.min_confidence_threshold]
        
        # Calculate metrics
        buy_trades = confident_trades[confident_trades['trade_side'] == TradeSide.BUY.value]
        sell_trades = confident_trades[confident_trades['trade_side'] == TradeSide.SELL.value]
        
        buy_volume = buy_trades['trade_size'].sum()
        sell_volume = sell_trades['trade_size'].sum()
        
        # Large trades (top 10% by size)
        size_threshold = confident_trades['trade_size'].quantile(0.9)
        large_buys = buy_trades[buy_trades['trade_size'] >= size_threshold]
        large_sells = sell_trades[sell_trades['trade_size'] >= size_threshold]
        
        metrics = {
            'buy_volume': int(buy_volume),
            'sell_volume': int(sell_volume),
            'buy_count': len(buy_trades),
            'sell_count': len(sell_trades),
            'net_order_flow': int(buy_volume - sell_volume),
            'buy_sell_ratio': buy_volume / max(1, sell_volume),
            'large_buy_volume': int(large_buys['trade_size'].sum()),
            'large_sell_volume': int(large_sells['trade_size'].sum()),
            'avg_buy_size': buy_trades['trade_size'].mean() if len(buy_trades) > 0 else 0,
            'avg_sell_size': sell_trades['trade_size'].mean() if len(sell_trades) > 0 else 0,
            'confidence_rate': len(confident_trades) / len(aligned_df) if len(aligned_df) > 0 else 0
        }
        
        return metrics
    
    def create_summary_report(self, report: AlignmentReport, metrics: Dict[str, Any]) -> str:
        """Create human-readable summary report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TRADE/QUOTE ALIGNMENT REPORT")
        lines.append("=" * 80)
        
        # Alignment statistics
        lines.append("\nAlignment Statistics:")
        lines.append(f"  Total Trades: {report.total_trades:,}")
        lines.append(f"  Successfully Aligned: {report.aligned_trades:,} ({report.aligned_trades/max(1,report.total_trades)*100:.1f}%)")
        lines.append(f"  Failed Alignments: {report.failed_alignments:,}")
        
        # Quote quality
        lines.append("\nQuote Quality:")
        lines.append(f"  Average Quote Age: {report.avg_quote_age_ms:.1f} ms")
        lines.append(f"  Maximum Quote Age: {report.max_quote_age_ms:.1f} ms")
        lines.append(f"  Average Spread: ${report.avg_spread:.4f} ({report.avg_spread_pct:.2f}%)")
        
        # Classification distribution
        lines.append("\nTrade Classification:")
        for side, count in report.side_distribution.items():
            pct = count / max(1, report.aligned_trades) * 100
            lines.append(f"  {side.upper()}: {count:,} ({pct:.1f}%)")
        
        # Confidence distribution
        lines.append("\nConfidence Distribution:")
        for level, count in sorted(report.confidence_distribution.items()):
            lines.append(f"  {level}: {count:,}")
        
        # Order flow metrics
        if metrics:
            lines.append("\nOrder Flow Metrics:")
            lines.append(f"  Buy Volume: {metrics['buy_volume']:,}")
            lines.append(f"  Sell Volume: {metrics['sell_volume']:,}")
            lines.append(f"  Net Order Flow: {metrics['net_order_flow']:,}")
            lines.append(f"  Buy/Sell Ratio: {metrics['buy_sell_ratio']:.2f}")
            lines.append(f"  Large Buy Volume: {metrics['large_buy_volume']:,}")
            lines.append(f"  Large Sell Volume: {metrics['large_sell_volume']:,}")
            lines.append(f"  Confidence Rate: {metrics['confidence_rate']*100:.1f}%")
        
        # Warnings
        if report.warnings:
            lines.append("\nWarnings:")
            for warning in report.warnings:
                lines.append(f"  ⚠️  {warning}")
        
        return "\n".join(lines)