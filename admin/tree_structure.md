# Project Structure
Last updated: 2025-06-26 10:16:32

```
Sirius/
│   ├── .env
│   ├── README.md # Project documentation
│   ├── git_commit.py
│   ├── git_replace.py
│   ├── main.py # Main application entry point
│   ├── admin/
│   │   ├── tree_structure.md
│   │   ├── update_tree.py
│   ├── modules/
│   │   ├── calculations/
│   │   │   ├── market-structure/
│   │   │   │   ├── __init__.py
│   │   │   ├── volume/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── cluster_analyzer.py
│   │   │   │   ├── hvn_engine.py
│   │   │   │   ├── ranking_engine.py
│   │   │   │   ├── session_profile.py
│   │   │   │   ├── volume_profile.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── polygon_bridge.py
│   │   ├── filters/
│   │   ├── scanners/
│   │   ├── tests/
│   │   ├── ui/
│   │   │   ├── charts/
│   │   │   ├── dashboards/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── hvn_dashboard.py
│   │   │   ├── grids/
│   ├── polygon/
│   │   ├── README.md # Project documentation
│   │   ├── __init__.py
│   │   ├── api_validator.py
│   │   ├── config.py # Configuration settings
│   │   ├── core.py
│   │   ├── exceptions.py
│   │   ├── fetcher.py
│   │   ├── rate_limiter.py
│   │   ├── storage.py
│   │   ├── utils.py
│   │   ├── websocket.py
│   │   ├── data/
│   │   │   ├── rate_limit_stats.json
│   │   │   ├── cache/
│   │   │   │   ├── polygon_cache.db
│   │   │   ├── logs/
│   │   │   │   ├── polygon.log
│   │   │   ├── parquet/
│   │   │   │   ├── symbols/
│   │   │   │   │   ├── AAPL/
│   │   │   │   │   │   ├── AAPL_15min.parquet
│   │   │   │   │   │   ├── AAPL_1day.parquet
│   │   │   │   │   │   ├── AAPL_1hour.parquet
│   │   │   │   │   │   ├── AAPL_1min.parquet
│   │   │   │   │   │   ├── AAPL_5min.parquet
│   │   │   │   │   ├── AMD/
│   │   │   │   │   │   ├── AMD_15min.parquet
│   │   │   │   │   │   ├── AMD_1day.parquet
│   │   │   │   │   │   ├── AMD_1hour.parquet
│   │   │   │   │   │   ├── AMD_1min.parquet
│   │   │   │   │   ├── AMZN/
│   │   │   │   │   │   ├── AMZN_1day.parquet
│   │   │   │   │   │   ├── AMZN_1min.parquet
│   │   │   │   │   ├── CRCL/
│   │   │   │   │   │   ├── CRCL_15min.parquet
│   │   │   │   │   ├── GOOG/
│   │   │   │   │   │   ├── GOOG_15min.parquet
│   │   │   │   │   │   ├── GOOG_1day.parquet
│   │   │   │   │   │   ├── GOOG_1min.parquet
│   │   │   │   │   ├── GOOGL/
│   │   │   │   │   │   ├── GOOGL_1day.parquet
│   │   │   │   │   ├── MSFT/
│   │   │   │   │   │   ├── MSFT_1day.parquet
│   │   │   │   │   │   ├── MSFT_1min.parquet
│   │   │   │   │   ├── NVDA/
│   │   │   │   │   │   ├── NVDA_15min.parquet
│   │   │   │   │   │   ├── NVDA_1day.parquet
│   │   │   │   │   │   ├── NVDA_1hour.parquet
│   │   │   │   │   │   ├── NVDA_1min.parquet
│   │   │   │   │   ├── OKLO/
│   │   │   │   │   │   ├── OKLO_15min.parquet
│   │   │   │   │   │   ├── OKLO_1day.parquet
│   │   │   │   │   │   ├── OKLO_1hour.parquet
│   │   │   │   │   │   ├── OKLO_1min.parquet
│   │   │   │   │   ├── PLTR/
│   │   │   │   │   │   ├── PLTR_15min.parquet
│   │   │   │   │   │   ├── PLTR_1day.parquet
│   │   │   │   │   │   ├── PLTR_1hour.parquet
│   │   │   │   │   │   ├── PLTR_1min.parquet
│   │   │   │   │   ├── SPY/
│   │   │   │   │   │   ├── SPY_5min.parquet
│   │   │   │   │   ├── TSLA/
│   │   │   │   │   │   ├── TSLA_15min.parquet
│   │   │   │   │   │   ├── TSLA_1d.parquet
│   │   │   │   │   │   ├── TSLA_1day.parquet
│   │   │   │   │   │   ├── TSLA_1hour.parquet
│   │   │   │   │   │   ├── TSLA_1min.parquet
│   │   │   │   │   │   ├── TSLA_5min.parquet
│   │   ├── docs/
│   │   │   ├── config_overview.txt
│   │   │   ├── core_overview.txt
│   │   │   ├── exceptions_overview.txt
│   │   │   ├── fetcher_overview.txt
│   │   │   ├── polygon_components_breakdown.txt
│   │   │   ├── polygon_pre_synopsis.txt
│   │   │   ├── polygon_structure.txt
│   │   │   ├── rate_limiter.txt
│   │   │   ├── storage_overview.txt
│   │   │   ├── utils_overview.txt
│   │   │   ├── validators_overview.txt
│   │   │   ├── websocket.txt
│   │   │   ├── websocket_client.txt
│   │   ├── polygon_server/
│   │   │   ├── __init__.py
│   │   │   ├── config.py # Configuration settings
│   │   │   ├── models.py
│   │   │   ├── requirements.txt # Python dependencies
│   │   │   ├── server.py
│   │   │   ├── start_server.py
│   │   │   ├── endpoints/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── health.py
│   │   │   │   ├── rest.py
│   │   │   │   ├── websocket.py
│   │   │   ├── utils/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── json_encoder.py
│   │   ├── tests/
│   │   │   ├── test_monitor_websocket.py
│   │   │   ├── test_websocket_live.py
│   │   ├── validators/
│   │   │   ├── __init__.py
│   │   │   ├── anomalies.py
│   │   │   ├── api_features.py
│   │   │   ├── data_quality.py
│   │   │   ├── gaps.py
│   │   │   ├── market_hours.py
│   │   │   ├── ohlcv.py
│   │   │   ├── symbol.py
```
