# backtest/data/data_validator.py
"""
Module: Data Validator
Purpose: Validate market data quality before calculations
Features: Gap detection, price validation, volume checks, quality scoring
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Severity levels for validation issues"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ValidationType(Enum):
    """Types of data to validate (renamed to avoid conflict with request_aggregator.DataType)"""
    BARS = "bars"
    TRADES = "trades"
    QUOTES = "quotes"


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    timestamp: Optional[datetime]
    issue_type: str
    description: str
    level: ValidationLevel
    affected_rows: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report for a dataset"""
    data_type: ValidationType
    symbol: str
    start_time: datetime
    end_time: datetime
    row_count: int
    issues: List[ValidationIssue] = field(default_factory=list)
    quality_score: float = 100.0
    passed: bool = True
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_issue(self, issue: ValidationIssue):
        """Add an issue and update quality score"""
        self.issues.append(issue)
        
        # Deduct points based on severity
        if issue.level == ValidationLevel.WARNING:
            self.quality_score -= 5
        elif issue.level == ValidationLevel.ERROR:
            self.quality_score -= 10
            self.passed = False
        elif issue.level == ValidationLevel.CRITICAL:
            self.quality_score -= 20
            self.passed = False
        
        self.quality_score = max(0, self.quality_score)
    
    def get_issues_by_level(self, level: ValidationLevel) -> List[ValidationIssue]:
        """Get all issues of a specific level"""
        return [issue for issue in self.issues if issue.level == level]


class DataValidator:
    """
    Validates market data quality with configurable thresholds.
    
    Key validations:
    - Timestamp gaps and consistency
    - Price sanity checks
    - Volume validation
    - Quote quality (bid/ask spread)
    - Data completeness
    """
    
    def __init__(self,
                 max_gap_minutes: int = 5,
                 max_price_change_pct: float = 10.0,
                 max_spread_pct: float = 5.0,
                 min_quality_score: float = 80.0):
        """
        Initialize validator with thresholds.
        
        Args:
            max_gap_minutes: Maximum allowed gap between timestamps
            max_price_change_pct: Maximum allowed price change percentage
            max_spread_pct: Maximum allowed bid-ask spread percentage
            min_quality_score: Minimum score to pass validation
        """
        self.max_gap_minutes = max_gap_minutes
        self.max_price_change_pct = max_price_change_pct
        self.max_spread_pct = max_spread_pct
        self.min_quality_score = min_quality_score
        
        # Market hours (EST converted to UTC)
        self.market_open = 14  # 9:30 AM EST = 14:30 UTC
        self.market_close = 21  # 4:00 PM EST = 21:00 UTC
        
        logger.info(f"DataValidator initialized with thresholds: "
                   f"gap={max_gap_minutes}min, price_change={max_price_change_pct}%, "
                   f"spread={max_spread_pct}%")
    
    def validate_bars(self, df: pd.DataFrame, symbol: str, 
                     timeframe: str = '1min') -> ValidationReport:
        """
        Validate bar (OHLCV) data.
        
        Checks:
        - Timestamp gaps
        - OHLC relationship (O<=H, O>=L, L<=C<=H)
        - Price continuity
        - Volume validity
        - Zero/negative prices
        """
        report = ValidationReport(
            data_type=ValidationType.BARS,
            symbol=symbol,
            start_time=df.index.min() if len(df) > 0 else datetime.now(timezone.utc),
            end_time=df.index.max() if len(df) > 0 else datetime.now(timezone.utc),
            row_count=len(df)
        )
        
        if df.empty:
            report.add_issue(ValidationIssue(
                timestamp=None,
                issue_type="EMPTY_DATA",
                description="No data provided for validation",
                level=ValidationLevel.ERROR,
                affected_rows=0
            ))
            return report
        
        # 1. Check timestamp gaps
        self._check_timestamp_gaps(df, report, timeframe)
        
        # 2. Check OHLC relationships
        self._check_ohlc_relationships(df, report)
        
        # 3. Check price continuity
        self._check_price_continuity(df, report)
        
        # 4. Check volume
        self._check_volume(df, report)
        
        # 5. Check for zero/negative prices
        self._check_price_validity(df, report)
        
        # 6. Check for duplicate timestamps
        self._check_duplicate_timestamps(df, report)
        
        # Summary statistics
        report.summary = {
            'total_bars': len(df),
            'missing_bars': report.summary.get('missing_bars', 0),
            'price_issues': len([i for i in report.issues if 'price' in i.issue_type.lower()]),
            'volume_issues': len([i for i in report.issues if 'volume' in i.issue_type.lower()]),
            'avg_spread': ((df['high'] - df['low']) / df['low'] * 100).mean() if len(df) > 0 else 0
        }
        
        return report
    
    def validate_trades(self, df: pd.DataFrame, symbol: str) -> ValidationReport:
        """
        Validate trade tick data.
        
        Checks:
        - Timestamp ordering
        - Price validity
        - Size validity
        - Excessive gaps
        - Duplicate trades
        """
        report = ValidationReport(
            data_type=ValidationType.TRADES,
            symbol=symbol,
            start_time=df.index.min() if len(df) > 0 else datetime.now(timezone.utc),
            end_time=df.index.max() if len(df) > 0 else datetime.now(timezone.utc),
            row_count=len(df)
        )
        
        if df.empty:
            report.add_issue(ValidationIssue(
                timestamp=None,
                issue_type="EMPTY_DATA",
                description="No trade data provided",
                level=ValidationLevel.ERROR,
                affected_rows=0
            ))
            return report
        
        # 1. Check timestamp ordering
        self._check_timestamp_ordering(df, report)
        
        # 2. Check trade prices
        self._check_trade_prices(df, report)
        
        # 3. Check trade sizes
        self._check_trade_sizes(df, report)
        
        # 4. Check for suspicious patterns
        self._check_trade_patterns(df, report)
        
        # Summary
        report.summary = {
            'total_trades': len(df),
            'avg_price': df['price'].mean() if 'price' in df else 0,
            'total_volume': df['size'].sum() if 'size' in df else 0,
            'price_range': (df['price'].max() - df['price'].min()) if 'price' in df else 0,
            'trades_per_minute': len(df) / max(1, (df.index.max() - df.index.min()).total_seconds() / 60)
        }
        
        return report
    
    def validate_quotes(self, df: pd.DataFrame, symbol: str) -> ValidationReport:
        """
        Validate quote (NBBO) data.
        
        Checks:
        - Bid < Ask relationship
        - Spread reasonableness
        - Quote staleness
        - Size validity
        - Crossed markets
        """
        report = ValidationReport(
            data_type=ValidationType.QUOTES,
            symbol=symbol,
            start_time=df.index.min() if len(df) > 0 else datetime.now(timezone.utc),
            end_time=df.index.max() if len(df) > 0 else datetime.now(timezone.utc),
            row_count=len(df)
        )
        
        if df.empty:
            report.add_issue(ValidationIssue(
                timestamp=None,
                issue_type="EMPTY_DATA",
                description="No quote data provided",
                level=ValidationLevel.ERROR,
                affected_rows=0
            ))
            return report
        
        # 1. Check bid/ask relationship
        self._check_bid_ask_relationship(df, report)
        
        # 2. Check spread
        self._check_spread(df, report)
        
        # 3. Check quote sizes
        self._check_quote_sizes(df, report)
        
        # 4. Check for stale quotes
        self._check_quote_staleness(df, report)
        
        # Summary
        if 'bid' in df and 'ask' in df:
            spreads = df['ask'] - df['bid']
            spread_pct = (spreads / df['bid'] * 100)
            
            report.summary = {
                'total_quotes': len(df),
                'avg_spread': spreads.mean(),
                'avg_spread_pct': spread_pct.mean(),
                'max_spread_pct': spread_pct.max(),
                'crossed_quotes': len(df[df['bid'] >= df['ask']]),
                'zero_spreads': len(df[spreads == 0])
            }
        
        return report
    
    def _check_timestamp_gaps(self, df: pd.DataFrame, report: ValidationReport, 
                            timeframe: str):
        """Check for gaps in timestamps"""
        if len(df) < 2:
            return
        
        # Calculate expected frequency
        freq_map = {
            '1min': 1,
            '5min': 5,
            '15min': 15,
            '30min': 30,
            '1hour': 60
        }
        expected_minutes = freq_map.get(timeframe, 1)
        
        # Check gaps
        time_diffs = df.index.to_series().diff()
        gaps = time_diffs[time_diffs > timedelta(minutes=expected_minutes * 1.5)]
        
        # Filter out expected gaps (market close, weekends)
        for gap_time, gap_duration in gaps.items():
            gap_minutes = gap_duration.total_seconds() / 60
            
            # Check if gap is during market hours
            hour = gap_time.hour
            if hour >= self.market_open and hour < self.market_close and gap_time.weekday() < 5:
                if gap_minutes > self.max_gap_minutes:
                    report.add_issue(ValidationIssue(
                        timestamp=gap_time,
                        issue_type="TIMESTAMP_GAP",
                        description=f"Gap of {gap_minutes:.1f} minutes detected",
                        level=ValidationLevel.WARNING if gap_minutes < 15 else ValidationLevel.ERROR,
                        affected_rows=1,
                        metadata={'gap_minutes': gap_minutes}
                    ))
        
        report.summary['missing_bars'] = len(gaps)
    
    def _check_ohlc_relationships(self, df: pd.DataFrame, report: ValidationReport):
        """Check OHLC logical relationships"""
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return
        
        # High should be >= all other prices
        invalid_high = df[(df['high'] < df['open']) | 
                         (df['high'] < df['close']) | 
                         (df['high'] < df['low'])]
        
        if len(invalid_high) > 0:
            report.add_issue(ValidationIssue(
                timestamp=invalid_high.index[0],
                issue_type="INVALID_HIGH",
                description=f"High price lower than other prices in {len(invalid_high)} bars",
                level=ValidationLevel.ERROR,
                affected_rows=len(invalid_high)
            ))
        
        # Low should be <= all other prices
        invalid_low = df[(df['low'] > df['open']) | 
                        (df['low'] > df['close']) | 
                        (df['low'] > df['high'])]
        
        if len(invalid_low) > 0:
            report.add_issue(ValidationIssue(
                timestamp=invalid_low.index[0],
                issue_type="INVALID_LOW",
                description=f"Low price higher than other prices in {len(invalid_low)} bars",
                level=ValidationLevel.ERROR,
                affected_rows=len(invalid_low)
            ))
    
    def _check_price_continuity(self, df: pd.DataFrame, report: ValidationReport):
        """Check for unrealistic price jumps"""
        if 'close' not in df.columns or len(df) < 2:
            return
        
        price_changes = df['close'].pct_change() * 100
        large_moves = price_changes[price_changes.abs() > self.max_price_change_pct]
        
        for timestamp, change in large_moves.items():
            if pd.notna(change):
                report.add_issue(ValidationIssue(
                    timestamp=timestamp,
                    issue_type="PRICE_SPIKE",
                    description=f"Price changed by {change:.1f}%",
                    level=ValidationLevel.WARNING if abs(change) < 15 else ValidationLevel.ERROR,
                    affected_rows=1,
                    metadata={'price_change_pct': change}
                ))
    
    def _check_volume(self, df: pd.DataFrame, report: ValidationReport):
        """Check volume validity"""
        if 'volume' not in df.columns:
            return
        
        # Check for negative volume
        negative_volume = df[df['volume'] < 0]
        if len(negative_volume) > 0:
            report.add_issue(ValidationIssue(
                timestamp=negative_volume.index[0],
                issue_type="NEGATIVE_VOLUME",
                description=f"Negative volume in {len(negative_volume)} bars",
                level=ValidationLevel.ERROR,
                affected_rows=len(negative_volume)
            ))
        
        # Check for suspiciously high volume
        if len(df) > 10:
            volume_mean = df['volume'].mean()
            volume_std = df['volume'].std()
            if volume_std > 0:
                outliers = df[df['volume'] > volume_mean + 5 * volume_std]
                if len(outliers) > 0:
                    report.add_issue(ValidationIssue(
                        timestamp=outliers.index[0],
                        issue_type="VOLUME_OUTLIER",
                        description=f"Unusually high volume in {len(outliers)} bars",
                        level=ValidationLevel.INFO,
                        affected_rows=len(outliers)
                    ))
    
    def _check_price_validity(self, df: pd.DataFrame, report: ValidationReport):
        """Check for zero or negative prices"""
        price_cols = ['open', 'high', 'low', 'close']
        existing_cols = [col for col in price_cols if col in df.columns]
        
        for col in existing_cols:
            zero_prices = df[df[col] == 0]
            if len(zero_prices) > 0:
                report.add_issue(ValidationIssue(
                    timestamp=zero_prices.index[0],
                    issue_type="ZERO_PRICE",
                    description=f"Zero {col} price in {len(zero_prices)} bars",
                    level=ValidationLevel.CRITICAL,
                    affected_rows=len(zero_prices)
                ))
            
            negative_prices = df[df[col] < 0]
            if len(negative_prices) > 0:
                report.add_issue(ValidationIssue(
                    timestamp=negative_prices.index[0],
                    issue_type="NEGATIVE_PRICE",
                    description=f"Negative {col} price in {len(negative_prices)} bars",
                    level=ValidationLevel.CRITICAL,
                    affected_rows=len(negative_prices)
                ))
    
    def _check_duplicate_timestamps(self, df: pd.DataFrame, report: ValidationReport):
        """Check for duplicate timestamps"""
        duplicates = df.index.duplicated()
        if duplicates.any():
            dup_count = duplicates.sum()
            report.add_issue(ValidationIssue(
                timestamp=df.index[duplicates][0],
                issue_type="DUPLICATE_TIMESTAMP",
                description=f"Found {dup_count} duplicate timestamps",
                level=ValidationLevel.ERROR,
                affected_rows=dup_count
            ))
    
    def _check_timestamp_ordering(self, df: pd.DataFrame, report: ValidationReport):
        """Check if timestamps are in order"""
        if len(df) < 2:
            return
        
        # Check if index is monotonic increasing
        if not df.index.is_monotonic_increasing:
            out_of_order = (~df.index.to_series().diff().dt.total_seconds().fillna(0) >= 0).sum()
            report.add_issue(ValidationIssue(
                timestamp=None,
                issue_type="TIMESTAMP_ORDER",
                description=f"Timestamps not in order, {out_of_order} instances",
                level=ValidationLevel.ERROR,
                affected_rows=out_of_order
            ))
    
    def _check_trade_prices(self, df: pd.DataFrame, report: ValidationReport):
        """Validate trade prices"""
        if 'price' not in df.columns:
            return
        
        # Check for zero/negative prices
        invalid_prices = df[(df['price'] <= 0)]
        if len(invalid_prices) > 0:
            report.add_issue(ValidationIssue(
                timestamp=invalid_prices.index[0],
                issue_type="INVALID_TRADE_PRICE",
                description=f"Invalid prices in {len(invalid_prices)} trades",
                level=ValidationLevel.CRITICAL,
                affected_rows=len(invalid_prices)
            ))
        
        # Check for price spikes in trades
        if len(df) > 10:
            price_changes = df['price'].pct_change() * 100
            spikes = price_changes[price_changes.abs() > 2.0]  # 2% spike in single trade
            
            if len(spikes) > 0:
                max_spike = price_changes.abs().max()
                report.add_issue(ValidationIssue(
                    timestamp=spikes.index[0],
                    issue_type="TRADE_PRICE_SPIKE",
                    description=f"Large price movements in {len(spikes)} trades (max: {max_spike:.1f}%)",
                    level=ValidationLevel.WARNING,
                    affected_rows=len(spikes),
                    metadata={'max_spike_pct': max_spike}
                ))
    
    def _check_trade_sizes(self, df: pd.DataFrame, report: ValidationReport):
        """Validate trade sizes"""
        if 'size' not in df.columns:
            return
        
        # Check for invalid sizes
        invalid_sizes = df[df['size'] <= 0]
        if len(invalid_sizes) > 0:
            report.add_issue(ValidationIssue(
                timestamp=invalid_sizes.index[0],
                issue_type="INVALID_TRADE_SIZE",
                description=f"Invalid trade sizes in {len(invalid_sizes)} trades",
                level=ValidationLevel.ERROR,
                affected_rows=len(invalid_sizes)
            ))
    
    def _check_trade_patterns(self, df: pd.DataFrame, report: ValidationReport):
        """Check for suspicious trading patterns"""
        if len(df) < 100:
            return
        
        # Check trades per second
        time_range = (df.index.max() - df.index.min()).total_seconds()
        if time_range > 0:
            trades_per_second = len(df) / time_range
            
            # Flag if more than 100 trades per second (possible data issue)
            if trades_per_second > 100:
                report.add_issue(ValidationIssue(
                    timestamp=None,
                    issue_type="EXCESSIVE_TRADE_FREQUENCY",
                    description=f"Very high trade frequency: {trades_per_second:.1f} trades/second",
                    level=ValidationLevel.WARNING,
                    affected_rows=len(df),
                    metadata={'trades_per_second': trades_per_second}
                ))
    
    def _check_bid_ask_relationship(self, df: pd.DataFrame, report: ValidationReport):
        """Check bid/ask validity"""
        if not all(col in df.columns for col in ['bid', 'ask']):
            return
        
        # Check for crossed quotes (bid >= ask)
        crossed = df[df['bid'] >= df['ask']]
        if len(crossed) > 0:
            report.add_issue(ValidationIssue(
                timestamp=crossed.index[0],
                issue_type="CROSSED_QUOTES",
                description=f"Bid >= Ask in {len(crossed)} quotes",
                level=ValidationLevel.ERROR,
                affected_rows=len(crossed)
            ))
        
        # Check for zero bid/ask
        zero_quotes = df[(df['bid'] == 0) | (df['ask'] == 0)]
        if len(zero_quotes) > 0:
            report.add_issue(ValidationIssue(
                timestamp=zero_quotes.index[0],
                issue_type="ZERO_QUOTES",
                description=f"Zero bid or ask in {len(zero_quotes)} quotes",
                level=ValidationLevel.ERROR,
                affected_rows=len(zero_quotes)
            ))
    
    def _check_spread(self, df: pd.DataFrame, report: ValidationReport):
        """Check spread reasonableness"""
        if not all(col in df.columns for col in ['bid', 'ask']):
            return
        
        # Calculate spread percentage
        spreads = df['ask'] - df['bid']
        spread_pct = (spreads / df['bid'] * 100).replace([np.inf, -np.inf], np.nan)
        
        # Check for excessive spreads
        wide_spreads = spread_pct[spread_pct > self.max_spread_pct]
        if len(wide_spreads) > 0:
            max_spread = spread_pct.max()
            report.add_issue(ValidationIssue(
                timestamp=wide_spreads.index[0],
                issue_type="WIDE_SPREAD",
                description=f"Wide spreads in {len(wide_spreads)} quotes (max: {max_spread:.1f}%)",
                level=ValidationLevel.WARNING,
                affected_rows=len(wide_spreads),
                metadata={'max_spread_pct': max_spread}
            ))
        
        # Check for zero spreads (locked market)
        zero_spreads = spreads[spreads == 0]
        if len(zero_spreads) > 10:  # Some zero spreads are normal
            report.add_issue(ValidationIssue(
                timestamp=zero_spreads.index[0],
                issue_type="LOCKED_MARKET",
                description=f"Zero spread (locked market) in {len(zero_spreads)} quotes",
                level=ValidationLevel.INFO,
                affected_rows=len(zero_spreads)
            ))
    
    def _check_quote_sizes(self, df: pd.DataFrame, report: ValidationReport):
        """Check quote size validity"""
        size_cols = [col for col in ['bid_size', 'ask_size'] if col in df.columns]
        
        for col in size_cols:
            invalid_sizes = df[df[col] <= 0]
            if len(invalid_sizes) > 0:
                report.add_issue(ValidationIssue(
                    timestamp=invalid_sizes.index[0],
                    issue_type="INVALID_QUOTE_SIZE",
                    description=f"Invalid {col} in {len(invalid_sizes)} quotes",
                    level=ValidationLevel.WARNING,
                    affected_rows=len(invalid_sizes)
                ))
    
    def _check_quote_staleness(self, df: pd.DataFrame, report: ValidationReport):
        """Check for stale quotes"""
        if len(df) < 2:
            return
        
        # Check time between quote updates
        time_diffs = df.index.to_series().diff()
        stale_quotes = time_diffs[time_diffs > timedelta(seconds=5)]
        
        if len(stale_quotes) > 0:
            max_staleness = stale_quotes.max().total_seconds()
            report.add_issue(ValidationIssue(
                timestamp=stale_quotes.index[0],
                issue_type="STALE_QUOTES",
                description=f"Stale quotes detected, {len(stale_quotes)} instances (max: {max_staleness:.1f}s)",
                level=ValidationLevel.INFO,
                affected_rows=len(stale_quotes),
                metadata={'max_staleness_seconds': max_staleness}
            ))
    
    def validate_data_quality(self, 
                            bars_df: Optional[pd.DataFrame] = None,
                            trades_df: Optional[pd.DataFrame] = None,
                            quotes_df: Optional[pd.DataFrame] = None,
                            symbol: str = "UNKNOWN") -> Dict[str, ValidationReport]:
        """
        Validate all provided data types and return comprehensive report.
        
        Returns:
            Dict mapping data type to validation report
        """
        reports = {}
        
        if bars_df is not None:
            reports['bars'] = self.validate_bars(bars_df, symbol)
        
        if trades_df is not None:
            reports['trades'] = self.validate_trades(trades_df, symbol)
        
        if quotes_df is not None:
            reports['quotes'] = self.validate_quotes(quotes_df, symbol)
        
        return reports
    
    def create_summary_report(self, reports: Dict[str, ValidationReport]) -> str:
        """Create a human-readable summary of all validation reports"""
        lines = []
        lines.append("=" * 80)
        lines.append("DATA VALIDATION SUMMARY")
        lines.append("=" * 80)
        
        overall_score = 0
        total_reports = 0
        
        for data_type, report in reports.items():
            lines.append(f"\n{data_type.upper()} VALIDATION:")
            lines.append("-" * 40)
            lines.append(f"Symbol: {report.symbol}")
            lines.append(f"Time Range: {report.start_time} to {report.end_time}")
            lines.append(f"Rows: {report.row_count:,}")
            lines.append(f"Quality Score: {report.quality_score:.1f}%")
            lines.append(f"Status: {'PASSED' if report.passed else 'FAILED'}")
            
            if report.issues:
                lines.append(f"\nIssues Found ({len(report.issues)} total):")
                
                # Group by level
                for level in ValidationLevel:
                    level_issues = report.get_issues_by_level(level)
                    if level_issues:
                        lines.append(f"  {level.value}: {len(level_issues)}")
                        for issue in level_issues[:3]:  # Show first 3
                            lines.append(f"    - {issue.description}")
            
            if report.summary:
                lines.append("\nSummary Statistics:")
                for key, value in report.summary.items():
                    if isinstance(value, float):
                        lines.append(f"  {key}: {value:.2f}")
                    else:
                        lines.append(f"  {key}: {value}")
            
            overall_score += report.quality_score
            total_reports += 1
        
        if total_reports > 0:
            lines.append("\n" + "=" * 80)
            lines.append(f"OVERALL QUALITY SCORE: {overall_score/total_reports:.1f}%")
            lines.append("=" * 80)
        
        return "\n".join(lines)


# Integration with the new modular PolygonDataManager
async def validate_fetched_data(data_manager, symbol: str, start_time: datetime, end_time: datetime):
    """
    Example of how to integrate DataValidator with the new modular PolygonDataManager
    """
    from .polygon_data_manager import PolygonDataManager
    
    # Fetch data
    bars = await data_manager.load_bars(symbol, start_time, end_time)
    trades = await data_manager.load_trades(symbol, start_time, end_time)
    quotes = await data_manager.load_quotes(symbol, start_time, end_time)
    
    # Validate
    validator = DataValidator()
    reports = validator.validate_data_quality(
        bars_df=bars if not bars.empty else None,
        trades_df=trades if not trades.empty else None,
        quotes_df=quotes if not quotes.empty else None,
        symbol=symbol
    )
    
    # Get summary
    summary = validator.create_summary_report(reports)
    
    return reports, summary