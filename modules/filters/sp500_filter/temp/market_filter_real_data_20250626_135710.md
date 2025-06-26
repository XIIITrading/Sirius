# Market Filter - Real S&P 500 Data Test

**Generated:** 2025-06-26 13:57:10
**Data Source:** Polygon.io via SP500Bridge
**Test Mode:** Direct script execution with real data

## Test Configuration

**Tickers Tested:** AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, JPM, BRK-B, UNH, XOM, JNJ, V, PG, MA
**Total Tickers:** 15

## Summary

| Metric | Value |
|--------|-------|
| Tickers Scanned | 15 |
| Passed Filters | 11 |
| Pass Rate | 73.3% |
| Avg Interest Score | 62.12 |
| Score Std Dev | 7.66 |

## Filter Criteria (Relaxed for Testing)

| Criterion | Value |
|-----------|-------|
| Price Range | $5.0 - $500.0 |
| Min Avg Volume | 500,000 shares |
| Min PM Volume Ratio | 0.1% |
| Min Dollar Volume | $1,000,000 |
| Min ATR | $0.5 |
| Min ATR % | 0.5% |

## Top 11 Stocks by Interest Score

| Rank | Ticker | Price | Score | PM Volume | PM % | Avg Volume | ATR | ATR % | $ Volume |
|:----:|:------:|------:|------:|----------:|-----:|-----------:|----:|------:|---------:|
| 1 | 🥇 **NVDA** | $156.01 | **75.6** | 139,238,043 | 107.62% | 129.4M | $3.49 | 2.24% | $20185.0M |
| 2 | 🥈 **AMZN** | $215.61 | **72.5** | 24,468,221 | 92.14% | 26.6M | $5.10 | 2.37% | $5725.9M |
| 3 | 🥉 **V** | $344.46 | 67.4 | 4,868,588 | 79.45% | 6.1M | $8.98 | 2.61% | $2110.8M |
| 4 | JPM | $289.30 | 65.9 | 5,190,413 | 88.79% | 5.8M | $4.75 | 1.64% | $1691.2M |
| 5 | GOOGL | $172.45 | 63.5 | 18,460,358 | 69.62% | 26.5M | $4.76 | 2.76% | $4572.7M |
| 6 | AAPL | $200.09 | 61.0 | 28,912,711 | 76.42% | 37.8M | $3.87 | 1.93% | $7570.6M |
| 7 | PG | $158.09 | 59.3 | 3,795,366 | 72.59% | 5.2M | $1.92 | 1.21% | $826.6M |
| 8 | XOM | $109.95 | 56.6 | 8,097,956 | 52.60% | 15.4M | $2.52 | 2.29% | $1692.7M |
| 9 | MSFT | $494.57 | 55.8 | 8,555,807 | 63.74% | 13.4M | $6.59 | 1.33% | $6638.1M |
| 10 | UNH | $303.04 | 53.7 | 3,927,634 | 45.34% | 8.7M | $6.48 | 2.14% | $2625.1M |
| 11 | JNJ | $152.50 | 51.9 | 3,099,239 | 53.98% | 5.7M | $1.92 | 1.26% | $875.6M |

## Score Components (Top Stock)

**NVDA** - Total Score: 75.56

| Component | Raw Value | Score | Weight | Contribution |
|-----------|-----------|------:|-------:|-------------:|
| PM Vol Ratio | 107.62% | 100.0 | 40% | 40.0 |
| ATR % | 2.24% | 2.2 | 25% | 0.6 |
| Dollar Vol | $20,185,005,309 | 100.0 | 20% | 20.0 |
| PM Vol Abs | 139,238,043 | 100.0 | 10% | 10.0 |
| Price-ATR | Yes | 100.0 | 5% | 5.0 |

---
*Real data test generated at 2025-06-26 13:57:10*
*For full S&P 500 scan, use sp500_bridge.py directly*
