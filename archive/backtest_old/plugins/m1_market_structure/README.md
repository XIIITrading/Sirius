# M1 Market Structure Plugin

This plugin implements fractal-based market structure analysis for 1-minute timeframe backtesting.

## Overview

The M1 Market Structure plugin detects market structure changes using fractal patterns to identify swing points. It identifies two main types of market structure events:

- **BOS (Break of Structure)**: Indicates trend continuation when price breaks above a previous high in an uptrend or below a previous low in a downtrend.
- **CHoCH (Change of Character)**: Signals potential trend reversal when price breaks below a previous low in an uptrend or above a previous high in a downtrend.

## Configuration

The plugin uses the following parameters:

- `fractal_length`: Number of bars on each side for fractal detection (default: 5)
- `buffer_size`: Number of candles to maintain in memory (default: 200)
- `min_candles_required`: Minimum candles needed for valid analysis (default: 21)

## Signal Output

The plugin generates signals with the following information:

### Direction
- `BULLISH`: Bullish market structure detected
- `BEARISH`: Bearish market structure detected
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
- `candles_processed`: Number of candles analyzed
- `reason`: Human-readable explanation of the signal

## Database Schema

The plugin stores results in the `bt_m1_market_structure` table with all the metadata fields listed above.

## Usage

The plugin is automatically loaded by the backtest system. To use it:

1. Include "m1_market_structure" in your enabled calculations
2. Run the backtest with sufficient historical data (at least 200 1-minute bars recommended)
3. Results will be stored in the database and included in the aggregated signals

## Technical Details

### Fractal Detection
A fractal is identified when a candle has:
- **High Fractal**: Higher high than `fractal_length` candles on both sides
- **Low Fractal**: Lower low than `fractal_length` candles on both sides

### Signal Strength
- CHoCH signals: Base strength of 70 (trend reversal)
- BOS signals: Base strength of 50 (trend continuation)
- Additional strength based on break magnitude (up to 30 points)

### Data Requirements
- Requires 1-minute OHLCV data
- No trades or quotes data needed
- Minimum 21 candles for initial signal generation