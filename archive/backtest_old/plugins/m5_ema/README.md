# M5 EMA Crossover Plugin

This plugin implements a 5-minute EMA crossover strategy for backtesting.

## Overview

The M5 EMA plugin analyzes 5-minute candles using 9/21 EMA crossover to determine trend direction:
- **BULLISH**: 9 EMA > 21 EMA and price above 9 EMA
- **BEARISH**: 9 EMA < 21 EMA and price below 9 EMA  
- **NEUTRAL**: Price on wrong side of 9 EMA

## Configuration

- `buffer_size`: Number of 5-minute candles to maintain (default: 50)
- `ema_short`: Short EMA period (default: 9)
- `ema_long`: Long EMA period (default: 21)
- `min_candles_required`: Minimum candles for valid signal (default: 21)

## Data Requirements

- **Timeframe**: 5-minute bars
- **Lookback**: 300 minutes (5 hours)
- **Trades**: Not required
- **Quotes**: Not required

## Output

The plugin generates signals with:
- Direction: BULLISH/BEARISH/NEUTRAL
- Strength: 0-100 based on EMA spread
- Confidence: Same as strength
- Detailed metrics including EMA values, spread, and trend strength

## Storage

Results are stored in the `bt_m5_ema` table with full signal details and EMA metrics.