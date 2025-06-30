# M15 Market Structure Plugin

This plugin implements fractal-based market structure analysis for 15-minute timeframe backtesting.

## Overview

The M15 Market Structure plugin detects market structure changes using fractal patterns on 15-minute charts. It aggregates 1-minute data to create 15-minute candles and identifies two main types of market structure events:

- **BOS (Break of Structure)**: Indicates trend continuation when price breaks above a previous high in an uptrend or below a previous low in a downtrend.
- **CHoCH (Change of Character)**: Signals potential trend reversal when price breaks below a previous low in an uptrend or above a previous high in a downtrend.

## Configuration

The plugin uses the following parameters optimized for 15-minute analysis:

- `fractal_length`: Number of bars on each side for fractal detection (default: 2)
- `buffer_size`: Number of 15-minute candles to maintain in memory (default: 60 = 900 minutes/15 hours)
- `min_candles_required`: Minimum candles needed for valid analysis (default: 10)

## Signal Output

The plugin generates signals with the following information:

### Direction
- `BULLISH`: Bullish market structure detected on 15-minute timeframe
- `BEARISH`: Bearish market structure detected on 15-minute timeframe
- `NEUTRAL`: No clear structure

### Metadata
- `structure_type`: Type of structure break (BOS or CHoCH)
- `current_trend`: Current market trend (BULL, BEAR, or NEUTRAL)
- `last_high_fractal`: Most recent high fractal price
- `last_low_fractal`: Most recent low fractal price
- `last_break_type`: Type of the last structure break
- `last_break_time`: Timestamp of the last break
- `last_break_price`: Price at which the last break occurred
- `fractal_count`: Total number of fractals detected
- `structure_breaks`: Total number of structure breaks
- `trend_changes`: Number of trend reversals (CHoCH events)
- `candles_processed`: Number of 15-minute candles analyzed
- `timeframe`: Always '15-minute' for this plugin
- `reason`: Human-readable explanation of the signal

## Database Schema

The plugin stores results in the `bt_m15_market_structure` table with all the metadata fields listed above.

## Data Aggregation

The plugin automatically aggregates 1-minute bars from the shared data cache into 15-minute bars:
- Open: First open of the 15-minute period
- High: Maximum high of the 15-minute period
- Low: Minimum low of the 15-minute period
- Close: Last close of the 15-minute period
- Volume: Sum of volumes in the 15-minute period
- VWAP: Volume-weighted average price for the period

## Usage

The plugin is automatically loaded by the backtest system. To use it:

1. Include "m15_market_structure" in your enabled calculations
2. Run the backtest with sufficient historical data (at least 1800 1-minute bars recommended for proper 15-minute aggregation)
3. Results will be stored in the database and included in the aggregated signals

## Technical Details

### Fractal Detection (15-Minute Optimized)
A fractal is identified when a 15-minute candle has:
- **High Fractal**: Higher high than `fractal_length` candles on both sides
- **Low Fractal**: Lower low than `fractal_length` candles on both sides

### Signal Strength
- CHoCH signals: Base strength of 80 (major trend reversal on higher timeframe)
- BOS signals: Base strength of 60 (strong trend continuation on higher timeframe)
- Additional strength based on break magnitude (up to 20 points)

### Data Requirements
- Requires 1-minute OHLCV data (automatically aggregated to 15-minute)
- No trades or quotes data needed
- Minimum 1800 minutes (30 hours) of 1-minute data recommended
- Minimum 10 15-minute candles for initial signal generation

## Differences from M1/M5 Market Structure

1. **Timeframe**: Analyzes 15-minute charts (highest timeframe)
2. **Fractal Length**: Shortest fractal length (2 vs 3 for M5, 5 for M1)
3. **Signal Strength**: Highest base strength due to higher timeframe significance
4. **Data Requirements**: Requires most historical 1-minute data for aggregation
5. **Noise Reduction**: Most resistant to market noise
6. **Signal Frequency**: Generates fewer but more significant signals
7. **Trend Reliability**: Best for identifying major market structure changes

## Multi-Timeframe Analysis

When used together with M1 and M5 market structure plugins:
- **M1**: Catches early micro-structure changes (scalping)
- **M5**: Identifies short-term structure changes (day trading)
- **M15**: Confirms major structure changes (swing trading)

The consensus across timeframes provides high-confidence trading signals.