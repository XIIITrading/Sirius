# M5 Market Structure Plugin

This plugin implements fractal-based market structure analysis for 5-minute timeframe backtesting.

## Overview

The M5 Market Structure plugin detects market structure changes using fractal patterns on 5-minute charts. It aggregates 1-minute data to create 5-minute candles and identifies two main types of market structure events:

- **BOS (Break of Structure)**: Indicates trend continuation when price breaks above a previous high in an uptrend or below a previous low in a downtrend.
- **CHoCH (Change of Character)**: Signals potential trend reversal when price breaks below a previous low in an uptrend or above a previous high in a downtrend.

## Configuration

The plugin uses the following parameters optimized for 5-minute analysis:

- `fractal_length`: Number of bars on each side for fractal detection (default: 3)
- `buffer_size`: Number of 5-minute candles to maintain in memory (default: 100 = 500 minutes)
- `min_candles_required`: Minimum candles needed for valid analysis (default: 15)

## Signal Output

The plugin generates signals with the following information:

### Direction
- `BULLISH`: Bullish market structure detected on 5-minute timeframe
- `BEARISH`: Bearish market structure detected on 5-minute timeframe
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
- `candles_processed`: Number of 5-minute candles analyzed
- `timeframe`: Always '5-minute' for this plugin
- `reason`: Human-readable explanation of the signal

## Database Schema

The plugin stores results in the `bt_m5_market_structure` table with all the metadata fields listed above.

## Data Aggregation

The plugin automatically aggregates 1-minute bars from the shared data cache into 5-minute bars:
- Open: First open of the 5-minute period
- High: Maximum high of the 5-minute period
- Low: Minimum low of the 5-minute period
- Close: Last close of the 5-minute period
- Volume: Sum of volumes in the 5-minute period

## Usage

The plugin is automatically loaded by the backtest system. To use it:

1. Include "m5_market_structure" in your enabled calculations
2. Run the backtest with sufficient historical data (at least 600 1-minute bars recommended for proper 5-minute aggregation)
3. Results will be stored in the database and included in the aggregated signals

## Technical Details

### Fractal Detection (5-Minute Optimized)
A fractal is identified when a 5-minute candle has:
- **High Fractal**: Higher high than `fractal_length` candles on both sides
- **Low Fractal**: Lower low than `fractal_length` candles on both sides

### Signal Strength
- CHoCH signals: Base strength of 75 (trend reversal on higher timeframe)
- BOS signals: Base strength of 55 (trend continuation on higher timeframe)
- Additional strength based on break magnitude (up to 25 points)

### Data Requirements
- Requires 1-minute OHLCV data (automatically aggregated to 5-minute)
- No trades or quotes data needed
- Minimum 600 minutes (10 hours) of 1-minute data recommended
- Minimum 15 5-minute candles for initial signal generation

## Differences from M1 Market Structure

1. **Timeframe**: Analyzes 5-minute charts instead of 1-minute
2. **Fractal Length**: Optimized with shorter fractal length (3 vs 5)
3. **Signal Strength**: Slightly higher base strength due to higher timeframe
4. **Data Requirements**: Requires more historical 1-minute data for aggregation
5. **Noise Reduction**: Less susceptible to market noise compared to 1-minute analysis