# backtest/data/trade_quote_aligner.py
"""
Module: Trade Quote Aligner
Purpose: Align trade executions with quote data for accurate buy/sell classification
Features: Time-based alignment, confidence scoring, quote staleness detection, interpolation
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return asdict(self)


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
        
        # Validate required columns
        if not self._validate_dataframes(trades_df, quotes_df):
            return pd.DataFrame(), self._create_empty_report()
        
        # Ensure sorted by time
        trades_df = trades_df.sort_index()
        quotes_df = quotes_df.sort_index()
        
        # Perform alignment
        logger.info(f"Aligning {len(trades_df)} trades with {len(quotes_df)} quotes using {method} method")
        
        # Use pandas merge_asof for efficient time-based alignment
        aligned_df = self._perform_alignment(trades_df, quotes_df, method)
        
        if aligned_df.empty:
            logger.warning("Alignment resulted in empty DataFrame")
            return aligned_df, self._create_empty_report()
        
        # Classify trades
        aligned_df = self._classify_trades(aligned_df)
        
        # Calculate confidence scores
        aligned_df = self._calculate_confidence(aligned_df)
        
        # Generate report
        report = self._generate_report(aligned_df, len(trades_df))
        
        return aligned_df, report
    
    def _validate_dataframes(self, trades_df: pd.DataFrame, quotes_df: pd.DataFrame) -> bool:
        """Validate that dataframes have required columns"""
        required_trade_cols = ['price', 'size']
        required_quote_cols = ['bid', 'ask', 'bid_size', 'ask_size']
        
        # Check trades
        missing_trade_cols = [col for col in required_trade_cols if col not in trades_df.columns]
        if missing_trade_cols:
            logger.error(f"Trades DataFrame missing required columns: {missing_trade_cols}")
            return False
        
        # Check quotes
        missing_quote_cols = [col for col in required_quote_cols if col not in quotes_df.columns]
        if missing_quote_cols:
            logger.error(f"Quotes DataFrame missing required columns: {missing_quote_cols}")
            return False
        
        return True
    
    def _perform_alignment(self, trades_df: pd.DataFrame, 
                          quotes_df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Perform the actual alignment using merge_asof"""
        # Prepare data for merge
        trades_reset = trades_df.reset_index()
        quotes_reset = quotes_df.reset_index()
        
        # Ensure index name is standardized
        trades_reset.rename(columns={trades_reset.columns[0]: 'trade_time'}, inplace=True)
        quotes_reset.rename(columns={quotes_reset.columns[0]: 'quote_time'}, inplace=True)
        
        # Rename other columns to avoid conflicts
        for col in trades_df.columns:
            if col in trades_reset.columns:
                trades_reset.rename(columns={col: f'trade_{col}'}, inplace=True)
        
        for col in quotes_df.columns:
            if col in quotes_reset.columns:
                quotes_reset.rename(columns={col: f'quote_{col}'}, inplace=True)
        
        # Perform time-based merge
        direction_map = {
            'backward': 'backward',
            'forward': 'forward',
            'nearest': 'nearest'
        }
        direction = direction_map.get(method, 'backward')
        
        try:
            aligned = pd.merge_asof(
                trades_reset.sort_values('trade_time'),
                quotes_reset.sort_values('quote_time'),
                left_on='trade_time',
                right_on='quote_time',
                direction=direction,
                tolerance=pd.Timedelta(milliseconds=self.interpolation_limit_ms)
            )
            
            # Calculate quote age (handle NaT values)
            if 'quote_time' in aligned.columns:
                quote_age = (aligned['trade_time'] - aligned['quote_time']).dt.total_seconds() * 1000
                aligned['quote_age_ms'] = quote_age.fillna(self.interpolation_limit_ms)
            else:
                aligned['quote_age_ms'] = self.interpolation_limit_ms
            
            # Set alignment method
            aligned['alignment_method'] = AlignmentMethod.BACKWARD.value
            
            # Handle missing quotes with interpolation
            if 'quote_bid' in aligned.columns and aligned['quote_bid'].isna().any():
                aligned = self._interpolate_missing_quotes(aligned, quotes_reset)
            
            return aligned
            
        except Exception as e:
            logger.error(f"Error during alignment: {e}")
            return pd.DataFrame()
    
    def _interpolate_missing_quotes(self, aligned_df: pd.DataFrame, 
                                   quotes_df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing quotes where possible"""
        if 'quote_bid' not in aligned_df.columns:
            return aligned_df
            
        missing_mask = aligned_df['quote_bid'].isna()
        
        if not missing_mask.any():
            return aligned_df
        
        logger.info(f"Interpolating {missing_mask.sum()} missing quotes")
        
        # Work with a copy to avoid SettingWithCopyWarning
        aligned_df = aligned_df.copy()
        
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
        # Work with a copy
        aligned_df = aligned_df.copy()
        
        # Check if we have necessary columns
        if 'quote_bid' not in aligned_df.columns or 'quote_ask' not in aligned_df.columns:
            aligned_df['trade_side'] = TradeSide.UNKNOWN.value
            return aligned_df
        
        # Initialize classification columns
        aligned_df['spread'] = aligned_df['quote_ask'] - aligned_df['quote_bid']
        aligned_df['spread_pct'] = (aligned_df['spread'] / aligned_df['quote_bid'].replace(0, np.nan)) * 100
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
        if 'trade_price' not in df.columns or 'midpoint' not in df.columns:
            return pd.Series(TradeSide.UNKNOWN.value, index=df.index)
            
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
        if not all(col in df.columns for col in ['trade_price', 'quote_bid', 'quote_ask']):
            return pd.Series(TradeSide.UNKNOWN.value, index=df.index)
            
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
        if 'trade_price' not in df.columns:
            return pd.Series(TradeSide.UNKNOWN.value, index=df.index)
            
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
        if 'trade_side' in df.columns and 'tick_test' in df.columns:
            midpoint_mask = df['trade_side'] == TradeSide.MIDPOINT.value
            df.loc[midpoint_mask, 'trade_side'] = df.loc[midpoint_mask, 'tick_test']
        
        # For remaining unknowns, use effective spread
        if 'trade_side' in df.columns and 'effective_spread_side' in df.columns:
            unknown_mask = df['trade_side'] == TradeSide.UNKNOWN.value
            df.loc[unknown_mask, 'trade_side'] = df.loc[unknown_mask, 'effective_spread_side']
        
        return df
    
    def _calculate_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate confidence score for each classification"""
        df = df.copy()
        df['confidence'] = 1.0
        
        # Factor 1: Quote staleness (0-1, where 1 is fresh)
        if 'quote_age_ms' in df.columns:
            staleness_factor = 1 - (df['quote_age_ms'].clip(0, self.max_quote_age_ms) / self.max_quote_age_ms)
            df['confidence'] *= staleness_factor
        
        # Factor 2: Spread reasonableness (0-1, where 1 is normal spread)
        if 'spread_pct' in df.columns:
            normal_spread_mask = df['spread_pct'] <= self.spread_outlier_pct
            df.loc[~normal_spread_mask, 'confidence'] *= 0.5
        
        # Factor 3: Price position clarity (0-1, where 1 is clear buy/sell)
        if all(col in df.columns for col in ['trade_price', 'quote_bid', 'quote_ask', 'spread']):
            with np.errstate(divide='ignore', invalid='ignore'):
                price_position = np.minimum(
                    abs(df['trade_price'] - df['quote_bid']),
                    abs(df['trade_price'] - df['quote_ask'])
                ) / df['spread'].replace(0, np.nan)
                price_clarity = 1 - (2 * price_position.clip(0, 0.5))
                price_clarity = price_clarity.fillna(0.5)  # Default for zero spread
                df['confidence'] *= price_clarity
        
        # Factor 4: Crossed/locked market penalty
        if 'quote_bid' in df.columns and 'quote_ask' in df.columns:
            crossed_mask = df['quote_bid'] >= df['quote_ask']
            df.loc[crossed_mask, 'confidence'] *= 0.3
        
        # Factor 5: Interpolated data penalty
        if 'alignment_method' in df.columns:
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
        aligned_trades = len(aligned_df[aligned_df['quote_bid'].notna()]) if 'quote_bid' in aligned_df else 0
        failed_alignments = total_trades - aligned_trades
        
        # Side distribution
        side_dist = {}
        if 'trade_side' in aligned_df:
            side_dist = aligned_df['trade_side'].value_counts().to_dict()
        
        # Confidence distribution
        conf_dist = {}
        if 'confidence' in aligned_df:
            conf_bins = pd.cut(aligned_df['confidence'], 
                              bins=[0, 0.25, 0.5, 0.75, 1.0],
                              labels=['Low', 'Medium', 'High', 'Very High'])
            conf_dist = conf_bins.value_counts().to_dict()
        
        # Alignment method distribution
        method_dist = {}
        if 'alignment_method' in aligned_df:
            method_dist = aligned_df['alignment_method'].value_counts().to_dict()
        
        # Calculate averages
        avg_quote_age_ms = aligned_df['quote_age_ms'].mean() if 'quote_age_ms' in aligned_df and aligned_trades > 0 else 0
        max_quote_age_ms = aligned_df['quote_age_ms'].max() if 'quote_age_ms' in aligned_df and aligned_trades > 0 else 0
        avg_spread = aligned_df['spread'].mean() if 'spread' in aligned_df and aligned_trades > 0 else 0
        avg_spread_pct = aligned_df['spread_pct'].mean() if 'spread_pct' in aligned_df and aligned_trades > 0 else 0
        
        # Generate warnings
        warnings = []
        
        if 'quote_age_ms' in aligned_df:
            high_staleness = aligned_df['quote_age_ms'] > self.max_quote_age_ms
            if high_staleness.any():
                warnings.append(f"{high_staleness.sum()} trades have stale quotes (>{self.max_quote_age_ms}ms)")
        
        if 'confidence' in aligned_df:
            low_confidence = aligned_df['confidence'] < self.min_confidence_threshold
            if low_confidence.any():
                warnings.append(f"{low_confidence.sum()} trades have low confidence (<{self.min_confidence_threshold})")
        
        if 'spread_pct' in aligned_df:
            wide_spreads = aligned_df['spread_pct'] > self.spread_outlier_pct
            if wide_spreads.any():
                warnings.append(f"{wide_spreads.sum()} trades have unusually wide spreads (>{self.spread_outlier_pct}%)")
        
        return AlignmentReport(
            total_trades=total_trades,
            aligned_trades=aligned_trades,
            failed_alignments=failed_alignments,
            avg_quote_age_ms=avg_quote_age_ms,
            max_quote_age_ms=max_quote_age_ms,
            side_distribution=side_dist,
            confidence_distribution=conf_dist,
            alignment_methods_used=method_dist,
            avg_spread=avg_spread,
            avg_spread_pct=avg_spread_pct,
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
            return self._create_empty_metrics()
        
        # Check required columns
        if 'trade_side' not in aligned_df or 'trade_size' not in aligned_df:
            logger.warning("Missing required columns for order flow metrics")
            return self._create_empty_metrics()
        
        # Filter by confidence if available
        if 'confidence' in aligned_df:
            confident_trades = aligned_df[aligned_df['confidence'] >= self.min_confidence_threshold]
        else:
            confident_trades = aligned_df
        
        # Calculate metrics
        buy_trades = confident_trades[confident_trades['trade_side'] == TradeSide.BUY.value]
        sell_trades = confident_trades[confident_trades['trade_side'] == TradeSide.SELL.value]
        
        buy_volume = int(buy_trades['trade_size'].sum())
        sell_volume = int(sell_trades['trade_size'].sum())
        
        # Large trades (top 10% by size)
        size_threshold = 0
        if len(confident_trades) > 0:
            size_threshold = confident_trades['trade_size'].quantile(0.9)
        
        large_buys = buy_trades[buy_trades['trade_size'] >= size_threshold] if size_threshold > 0 else pd.DataFrame()
        large_sells = sell_trades[sell_trades['trade_size'] >= size_threshold] if size_threshold > 0 else pd.DataFrame()
        
        metrics = {
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'buy_count': len(buy_trades),
            'sell_count': len(sell_trades),
            'net_order_flow': buy_volume - sell_volume,
            'buy_sell_ratio': buy_volume / max(1, sell_volume),
            'large_buy_volume': int(large_buys['trade_size'].sum()) if len(large_buys) > 0 else 0,
            'large_sell_volume': int(large_sells['trade_size'].sum()) if len(large_sells) > 0 else 0,
            'avg_buy_size': float(buy_trades['trade_size'].mean()) if len(buy_trades) > 0 else 0,
            'avg_sell_size': float(sell_trades['trade_size'].mean()) if len(sell_trades) > 0 else 0,
            'confidence_rate': len(confident_trades) / len(aligned_df) if len(aligned_df) > 0 else 0,
            'size_threshold': float(size_threshold)
        }
        
        return metrics
    
    def _create_empty_metrics(self) -> Dict[str, Any]:
        """Create empty metrics dictionary"""
        return {
            'buy_volume': 0,
            'sell_volume': 0,
            'buy_count': 0,
            'sell_count': 0,
            'net_order_flow': 0,
            'buy_sell_ratio': 0.0,
            'large_buy_volume': 0,
            'large_sell_volume': 0,
            'avg_buy_size': 0.0,
            'avg_sell_size': 0.0,
            'confidence_rate': 0.0,
            'size_threshold': 0.0
        }
    
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
        if report.confidence_distribution:
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
            lines.append(f"  Avg Buy Size: {metrics['avg_buy_size']:.1f}")
            lines.append(f"  Avg Sell Size: {metrics['avg_sell_size']:.1f}")
            lines.append(f"  Confidence Rate: {metrics['confidence_rate']*100:.1f}%")
        
        # Warnings
        if report.warnings:
            lines.append("\nWarnings:")
            for warning in report.warnings:
                lines.append(f"  ⚠️  {warning}")
        
        return "\n".join(lines)


# Integration with the new modular PolygonDataManager
async def align_trades_with_polygon_data(data_manager, symbol: str, 
                                       start_time: datetime, end_time: datetime,
                                       aligner_config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, AlignmentReport, Dict[str, Any]]:
    """
    Helper function to fetch and align trades/quotes using the modular PolygonDataManager
    
    Args:
        data_manager: PolygonDataManager instance
        symbol: Stock symbol
        start_time: Start time for data
        end_time: End time for data
        aligner_config: Optional configuration for TradeQuoteAligner
        
    Returns:
        Tuple of (aligned DataFrame, alignment report, order flow metrics)
    """
    # Fetch data using the modular data manager
    trades_df = await data_manager.load_trades(symbol, start_time, end_time)
    quotes_df = await data_manager.load_quotes(symbol, start_time, end_time)
    
    # Create aligner with config
    config = aligner_config or {}
    aligner = TradeQuoteAligner(**config)
    
    # Perform alignment
    aligned_df, report = aligner.align_trades_quotes(trades_df, quotes_df)
    
    # Calculate metrics
    metrics = aligner.calculate_order_flow_metrics(aligned_df)
    
    return aligned_df, report, metrics


if __name__ == "__main__":
    # Example usage with the new modular structure
    import asyncio
    from polygon_data_manager import PolygonDataManager
    
    async def main():
        # Create data manager
        data_manager = PolygonDataManager(
            api_key='your_api_key',
            cache_dir='./cache'
        )
        
        # Define time range
        symbol = 'AAPL'
        end_time = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
        start_time = end_time - timedelta(hours=1)
        
        # Fetch and align
        aligned_df, report, metrics = await align_trades_with_polygon_data(
            data_manager, symbol, start_time, end_time,
            aligner_config={
                'max_quote_age_ms': 500,
                'min_confidence_threshold': 0.7
            }
        )
        
        # Create summary
        aligner = TradeQuoteAligner()
        summary = aligner.create_summary_report(report, metrics)
        print(summary)
        
        # Export results if needed
        if not aligned_df.empty:
            aligned_df.to_csv(f'{symbol}_aligned_trades.csv')
            print(f"\nAligned trades saved to {symbol}_aligned_trades.csv")
    
    asyncio.run(main())