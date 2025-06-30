# M15 EMA Crossover Plugin

This plugin implements a 15-minute EMA crossover strategy for backtesting.

## Overview

The M15 EMA plugin analyzes 15-minute candles using 9/21 EMA crossover to determine trend direction:
- **BULLISH**: 9 EMA > 21 EMA and price above 9 EMA
- **BEARISH**: 9 EMA < 21 EMA and price below 9 EMA  
- **NEUTRAL**: Price on wrong side of 9 EMA

## Configuration

- `buffer_size`: Number of 15-minute candles to maintain (default: 40)
- `ema_short`: Short EMA period (default: 9)
- `ema_long`: Long EMA period (default: 21)
- `min_candles_required`: Minimum candles for valid signal (default: 21)

## Data Requirements

- **Timeframe**: 15-minute bars
- **Lookback**: 900 minutes (15 hours)
- **Trades**: Not required
- **Quotes**: Not required

## Features

- Detects EMA crossovers within 75-minute window (5 bars)
- Calculates trend strength based on EMA spread
- Tracks price position relative to 9 EMA
- Provides detailed signal reasoning

## Output

The plugin generates signals with:
- Direction: BULLISH/BEARISH/NEUTRAL
- Strength: 0-100 based on EMA spread
- Confidence: Same as strength
- Detailed metrics including EMA values, spread, and trend strength

## Storage

Results are stored in the `bt_m15_ema` table with full signal details and EMA metrics.

## Time Considerations

15-minute bars provide a medium-term view of the market, suitable for:
- Swing trading strategies
- Position entries with wider stops
- Trend confirmation across multiple timeframes