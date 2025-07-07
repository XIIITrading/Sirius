TradeQuoteAligner Documentation
Overview
The TradeQuoteAligner is a module that synchronizes trade executions with quote (NBBO) data for accurate buy/sell classification and market microstructure analysis.
Input Data Requirements
Trades DataFrame

Index: DatetimeIndex (timezone-aware, preferably UTC)
Required Columns:

price (float): Trade execution price
size (int): Trade size/volume


Optional Columns:

exchange (str): Exchange code
conditions (str): Trade conditions



Quotes DataFrame

Index: DatetimeIndex (timezone-aware, preferably UTC)
Required Columns:

bid (float): Best bid price
ask (float): Best ask price
bid_size (int): Size at best bid
ask_size (int): Size at best ask


Optional Columns:

bid_exchange (int): Bid exchange code
ask_exchange (int): Ask exchange code



Output Schema
Aligned DataFrame
The output is a DataFrame with a RangeIndex (0, 1, 2, ...) where each row represents an aligned trade.
Column Schema
Column NameTypeDescriptionTrade Informationtrade_timedatetimeOriginal trade timestamp (timezone-aware)trade_pricefloatTrade execution pricetrade_sizeintTrade size/volumetrade_conditionsstrTrade condition codes (if available)trade_exchangeint/strExchange where trade executedQuote Informationquote_timedatetimeTimestamp of synchronized quotequote_bidfloatBid price at time of tradequote_askfloatAsk price at time of tradequote_bid_sizeintBid size at time of tradequote_ask_sizeintAsk size at time of tradequote_bid_exchangeintBid exchange codequote_ask_exchangeintAsk exchange codeAlignment Metricsquote_age_msfloatMilliseconds between trade and quotealignment_methodstrHow quote was aligned ('backward', 'forward', 'interpolated')Calculated FieldsspreadfloatAsk - Bidspread_pctfloatSpread as percentage of bidmidpointfloat(Bid + Ask) / 2Classificationtrade_sidestrClassification result: 'buy', 'sell', 'midpoint', 'unknown'effective_spread_sidestrAlternative classification method resulttick_teststrTick test classification resultconfidencefloatConfidence score (0.0 to 1.0)
Important Notes
Index Structure
⚠️ CRITICAL: The output DataFrame has a numeric RangeIndex, NOT a DatetimeIndex. The actual timestamps are in the trade_time column.
python# WRONG - This will fail:
aligned_df.index.floor('1min')

# CORRECT - Use this:
aligned_df.set_index('trade_time').index.floor('1min')
# OR
pd.to_datetime(aligned_df['trade_time']).dt.floor('1min')
Trade Side Classification

'buy': Trade executed at or above midpoint (buyer-initiated)
'sell': Trade executed at or below midpoint (seller-initiated)
'midpoint': Trade executed exactly at midpoint
'unknown': Could not classify (missing quote data)

Confidence Score Factors
The confidence score (0-1) is based on:

Quote staleness: Lower confidence for older quotes
Spread reasonableness: Lower confidence for abnormally wide spreads
Price position clarity: Higher confidence when clearly at bid or ask
Market conditions: Lower confidence for crossed/locked markets
Interpolation: Lower confidence for interpolated quotes

Usage Example
pythonfrom backtest.data.trade_quote_aligner import TradeQuoteAligner
import pandas as pd

# Initialize aligner
aligner = TradeQuoteAligner(
    max_quote_age_ms=1000,      # Max 1 second quote age
    min_confidence_threshold=0.5 # Minimum confidence for metrics
)

# Align trades with quotes
aligned_df, report = aligner.align_trades_quotes(trades_df, quotes_df)

# Process aligned trades (example: aggregate to minute bars)
# IMPORTANT: Set the index to trade_time first!
aligned_df = aligned_df.set_index('trade_time')

# Now you can use time-based operations
minute_groups = aligned_df.groupby(aligned_df.index.floor('1min'))

for minute, trades in minute_groups:
    buy_volume = trades[trades['trade_side'] == 'buy']['trade_size'].sum()
    sell_volume = trades[trades['trade_side'] == 'sell']['trade_size'].sum()
    print(f"{minute}: Buy={buy_volume}, Sell={sell_volume}")
Alignment Report
The AlignmentReport object contains:

total_trades: Total number of input trades
aligned_trades: Successfully aligned trades
failed_alignments: Trades without matching quotes
avg_quote_age_ms: Average age of quotes used
side_distribution: Dict of trade classifications
confidence_distribution: Dict of confidence levels
warnings: List of any data quality issues

Common Pitfalls

Timezone Issues: Ensure both trades and quotes DataFrames have timezone-aware DatetimeIndex
Index Confusion: Remember output has RangeIndex, not DatetimeIndex
Missing Quotes: Some trades may have NaN quote fields if no matching quote found
Confidence Filtering: Always check confidence scores before aggregating

Performance Considerations

Uses pandas merge_asof for efficient time-based matching
Can handle millions of trades/quotes
Memory usage scales with data size
Consider chunking for very large datasets (>10M rows)