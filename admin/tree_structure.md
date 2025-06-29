# Project Structure
Last updated: 2025-06-29 00:11:43

```
Sirius/
│   ├── .env
│   ├── README.md # Project documentation
│   ├── git_commit.py
│   ├── git_replace.py
│   ├── main.py # Main application entry point
│   ├── requirements.txt # Python dependencies
│   ├── sp500_filter_summary.md
│   ├── admin/
│   │   ├── tree_structure.md
│   │   ├── update_tree.py
│   │   ├── prompts/
│   │   │   ├── bactest_modularize.md
│   │   │   ├── calculation.txt
│   ├── backtest/
│   ├── modules/
│   │   ├── calculations/
│   │   │   ├── backtest/
│   │   │   ├── market-structure/
│   │   │   │   ├── __init__.py
│   │   │   ├── momentum/
│   │   │   ├── order_flow/
│   │   │   │   ├── bid_ask_imbal.py
│   │   │   │   ├── cum_delta.py
│   │   │   │   ├── micro_momentum.py
│   │   │   │   ├── trade_size_distro.py
│   │   │   ├── trend/
│   │   │   │   ├── statistical_trend_15min.py
│   │   │   │   ├── statistical_trend_1min.py
│   │   │   │   ├── statistical_trend_5min.py
│   │   │   ├── volume/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── cluster_analyzer.py
│   │   │   │   ├── hvn_engine.py
│   │   │   │   ├── market_context.py
│   │   │   │   ├── ranking_engine.py
│   │   │   │   ├── session_profile.py
│   │   │   │   ├── tick_flow.py
│   │   │   │   ├── volume_analysis_1min.py
│   │   │   │   ├── volume_profile.py
│   │   ├── claude/
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── polygon_bridge.py
│   │   │   ├── supabase_client.py
│   │   ├── filters/
│   │   │   ├── sp500_filter/
│   │   │   │   ├── market_filter.py
│   │   │   │   ├── sp500_bridge.py
│   │   │   │   ├── sp500_tickers.py
│   │   │   │   ├── test_integration.py
│   │   │   │   ├── update_sp500_list.py
│   │   │   │   ├── temp/
│   │   │   │   │   ├── market_filter_scan_20250627_073514.md
│   │   ├── integrations/
│   │   │   ├── claude_dialog.py
│   │   │   ├── claude_integration.py
│   │   ├── ui/
│   │   │   ├── components/
│   │   │   │   ├── dual_hvn_chart.py
│   │   │   ├── dashboards/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── backtest_dashboard.py
│   │   │   │   ├── entry_dashboard.py
│   │   │   │   ├── hvn_dashboard.py
│   │   │   │   ├── scanner_results_viewer.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── result_formatter.py
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
│   │   │   │   │   ├── A/
│   │   │   │   │   │   ├── A_1d.parquet
│   │   │   │   │   ├── AAL/
│   │   │   │   │   │   ├── AAL_1d.parquet
│   │   │   │   │   │   ├── AAL_1min.parquet
│   │   │   │   │   ├── AAP/
│   │   │   │   │   │   ├── AAP_1d.parquet
│   │   │   │   │   ├── AAPL/
│   │   │   │   │   │   ├── AAPL_15min.parquet
│   │   │   │   │   │   ├── AAPL_1d.parquet
│   │   │   │   │   │   ├── AAPL_1day.parquet
│   │   │   │   │   │   ├── AAPL_1hour.parquet
│   │   │   │   │   │   ├── AAPL_1min.parquet
│   │   │   │   │   │   ├── AAPL_5min.parquet
│   │   │   │   │   ├── ABBV/
│   │   │   │   │   │   ├── ABBV_1d.parquet
│   │   │   │   │   ├── ABT/
│   │   │   │   │   │   ├── ABT_1d.parquet
│   │   │   │   │   │   ├── ABT_1min.parquet
│   │   │   │   │   ├── ACGL/
│   │   │   │   │   │   ├── ACGL_1d.parquet
│   │   │   │   │   ├── ACN/
│   │   │   │   │   │   ├── ACN_1d.parquet
│   │   │   │   │   │   ├── ACN_1min.parquet
│   │   │   │   │   ├── ADBE/
│   │   │   │   │   │   ├── ADBE_1d.parquet
│   │   │   │   │   │   ├── ADBE_1min.parquet
│   │   │   │   │   ├── ADI/
│   │   │   │   │   │   ├── ADI_1d.parquet
│   │   │   │   │   ├── ADM/
│   │   │   │   │   │   ├── ADM_1d.parquet
│   │   │   │   │   ├── ADP/
│   │   │   │   │   │   ├── ADP_1d.parquet
│   │   │   │   │   │   ├── ADP_1min.parquet
│   │   │   │   │   ├── ADSK/
│   │   │   │   │   │   ├── ADSK_1d.parquet
│   │   │   │   │   │   ├── ADSK_1min.parquet
│   │   │   │   │   ├── AEE/
│   │   │   │   │   │   ├── AEE_1d.parquet
│   │   │   │   │   ├── AEP/
│   │   │   │   │   │   ├── AEP_1d.parquet
│   │   │   │   │   │   ├── AEP_1min.parquet
│   │   │   │   │   ├── AES/
│   │   │   │   │   │   ├── AES_1d.parquet
│   │   │   │   │   │   ├── AES_1min.parquet
│   │   │   │   │   ├── AFL/
│   │   │   │   │   │   ├── AFL_1d.parquet
│   │   │   │   │   ├── AIG/
│   │   │   │   │   │   ├── AIG_1d.parquet
│   │   │   │   │   ├── AIZ/
│   │   │   │   │   │   ├── AIZ_1d.parquet
│   │   │   │   │   ├── AJG/
│   │   │   │   │   │   ├── AJG_1d.parquet
│   │   │   │   │   ├── AKAM/
│   │   │   │   │   │   ├── AKAM_1d.parquet
│   │   │   │   │   ├── ALB/
│   │   │   │   │   │   ├── ALB_1d.parquet
│   │   │   │   │   │   ├── ALB_1min.parquet
│   │   │   │   │   ├── ALGN/
│   │   │   │   │   │   ├── ALGN_1d.parquet
│   │   │   │   │   ├── ALK/
│   │   │   │   │   │   ├── ALK_1d.parquet
│   │   │   │   │   │   ├── ALK_1min.parquet
│   │   │   │   │   ├── ALL/
│   │   │   │   │   │   ├── ALL_1d.parquet
│   │   │   │   │   ├── ALLE/
│   │   │   │   │   │   ├── ALLE_1d.parquet
│   │   │   │   │   ├── AMAT/
│   │   │   │   │   │   ├── AMAT_1d.parquet
│   │   │   │   │   │   ├── AMAT_1min.parquet
│   │   │   │   │   ├── AMCR/
│   │   │   │   │   │   ├── AMCR_1d.parquet
│   │   │   │   │   │   ├── AMCR_1min.parquet
│   │   │   │   │   ├── AMD/
│   │   │   │   │   │   ├── AMD_15min.parquet
│   │   │   │   │   │   ├── AMD_1d.parquet
│   │   │   │   │   │   ├── AMD_1day.parquet
│   │   │   │   │   │   ├── AMD_1hour.parquet
│   │   │   │   │   │   ├── AMD_1min.parquet
│   │   │   │   │   ├── AME/
│   │   │   │   │   │   ├── AME_1d.parquet
│   │   │   │   │   ├── AMGN/
│   │   │   │   │   │   ├── AMGN_1d.parquet
│   │   │   │   │   ├── AMP/
│   │   │   │   │   │   ├── AMP_1d.parquet
│   │   │   │   │   ├── AMT/
│   │   │   │   │   │   ├── AMT_1d.parquet
│   │   │   │   │   ├── AMTM/
│   │   │   │   │   │   ├── AMTM_1d.parquet
│   │   │   │   │   │   ├── AMTM_1min.parquet
│   │   │   │   │   ├── AMZN/
│   │   │   │   │   │   ├── AMZN_15min.parquet
│   │   │   │   │   │   ├── AMZN_1d.parquet
│   │   │   │   │   │   ├── AMZN_1day.parquet
│   │   │   │   │   │   ├── AMZN_1min.parquet
│   │   │   │   │   ├── ANET/
│   │   │   │   │   │   ├── ANET_15min.parquet
│   │   │   │   │   │   ├── ANET_1d.parquet
│   │   │   │   │   │   ├── ANET_1min.parquet
│   │   │   │   │   ├── ANSS/
│   │   │   │   │   │   ├── ANSS_1d.parquet
│   │   │   │   │   ├── AON/
│   │   │   │   │   │   ├── AON_1d.parquet
│   │   │   │   │   ├── AOS/
│   │   │   │   │   │   ├── AOS_1d.parquet
│   │   │   │   │   ├── APA/
│   │   │   │   │   │   ├── APA_1d.parquet
│   │   │   │   │   │   ├── APA_1min.parquet
│   │   │   │   │   ├── APD/
│   │   │   │   │   │   ├── APD_1d.parquet
│   │   │   │   │   ├── APH/
│   │   │   │   │   │   ├── APH_1d.parquet
│   │   │   │   │   │   ├── APH_1min.parquet
│   │   │   │   │   ├── APO/
│   │   │   │   │   │   ├── APO_1d.parquet
│   │   │   │   │   ├── APTV/
│   │   │   │   │   │   ├── APTV_1d.parquet
│   │   │   │   │   ├── ARE/
│   │   │   │   │   │   ├── ARE_1d.parquet
│   │   │   │   │   ├── ATO/
│   │   │   │   │   │   ├── ATO_1d.parquet
│   │   │   │   │   ├── AVB/
│   │   │   │   │   │   ├── AVB_1d.parquet
│   │   │   │   │   ├── AVGO/
│   │   │   │   │   │   ├── AVGO_15min.parquet
│   │   │   │   │   │   ├── AVGO_1d.parquet
│   │   │   │   │   │   ├── AVGO_1min.parquet
│   │   │   │   │   ├── AVY/
│   │   │   │   │   │   ├── AVY_1d.parquet
│   │   │   │   │   ├── AWK/
│   │   │   │   │   │   ├── AWK_1d.parquet
│   │   │   │   │   ├── AXON/
│   │   │   │   │   │   ├── AXON_1d.parquet
│   │   │   │   │   ├── AXP/
│   │   │   │   │   │   ├── AXP_1d.parquet
│   │   │   │   │   │   ├── AXP_1min.parquet
│   │   │   │   │   ├── AZO/
│   │   │   │   │   │   ├── AZO_1d.parquet
│   │   │   │   │   ├── BA/
│   │   │   │   │   │   ├── BA_1d.parquet
│   │   │   │   │   │   ├── BA_1min.parquet
│   │   │   │   │   ├── BAC/
│   │   │   │   │   │   ├── BAC_1d.parquet
│   │   │   │   │   │   ├── BAC_1min.parquet
│   │   │   │   │   ├── BALL/
│   │   │   │   │   │   ├── BALL_1d.parquet
│   │   │   │   │   ├── BAX/
│   │   │   │   │   │   ├── BAX_1d.parquet
│   │   │   │   │   ├── BBWI/
│   │   │   │   │   │   ├── BBWI_1d.parquet
│   │   │   │   │   ├── BBY/
│   │   │   │   │   │   ├── BBY_1d.parquet
│   │   │   │   │   │   ├── BBY_1min.parquet
│   │   │   │   │   ├── BDX/
│   │   │   │   │   │   ├── BDX_1d.parquet
│   │   │   │   │   ├── BEN/
│   │   │   │   │   │   ├── BEN_1d.parquet
│   │   │   │   │   │   ├── BEN_1min.parquet
│   │   │   │   │   ├── BG/
│   │   │   │   │   │   ├── BG_1d.parquet
│   │   │   │   │   ├── BIIB/
│   │   │   │   │   │   ├── BIIB_1d.parquet
│   │   │   │   │   ├── BIO/
│   │   │   │   │   │   ├── BIO_1d.parquet
│   │   │   │   │   ├── BK/
│   │   │   │   │   │   ├── BK_1d.parquet
│   │   │   │   │   ├── BKNG/
│   │   │   │   │   │   ├── BKNG_1d.parquet
│   │   │   │   │   ├── BKR/
│   │   │   │   │   │   ├── BKR_1d.parquet
│   │   │   │   │   │   ├── BKR_1min.parquet
│   │   │   │   │   ├── BLDR/
│   │   │   │   │   │   ├── BLDR_1d.parquet
│   │   │   │   │   ├── BLK/
│   │   │   │   │   │   ├── BLK_1d.parquet
│   │   │   │   │   ├── BMY/
│   │   │   │   │   │   ├── BMY_1d.parquet
│   │   │   │   │   │   ├── BMY_1min.parquet
│   │   │   │   │   ├── BR/
│   │   │   │   │   │   ├── BR_1d.parquet
│   │   │   │   │   ├── BRO/
│   │   │   │   │   │   ├── BRO_1d.parquet
│   │   │   │   │   ├── BSX/
│   │   │   │   │   │   ├── BSX_1d.parquet
│   │   │   │   │   ├── BWA/
│   │   │   │   │   │   ├── BWA_1d.parquet
│   │   │   │   │   ├── BX/
│   │   │   │   │   │   ├── BX_1d.parquet
│   │   │   │   │   │   ├── BX_1min.parquet
│   │   │   │   │   ├── BXP/
│   │   │   │   │   │   ├── BXP_1d.parquet
│   │   │   │   │   ├── C/
│   │   │   │   │   │   ├── C_1d.parquet
│   │   │   │   │   │   ├── C_1min.parquet
│   │   │   │   │   ├── CAG/
│   │   │   │   │   │   ├── CAG_1d.parquet
│   │   │   │   │   ├── CAH/
│   │   │   │   │   │   ├── CAH_1d.parquet
│   │   │   │   │   ├── CARR/
│   │   │   │   │   │   ├── CARR_1d.parquet
│   │   │   │   │   ├── CAT/
│   │   │   │   │   │   ├── CAT_1d.parquet
│   │   │   │   │   │   ├── CAT_1min.parquet
│   │   │   │   │   ├── CB/
│   │   │   │   │   │   ├── CB_1d.parquet
│   │   │   │   │   ├── CBOE/
│   │   │   │   │   │   ├── CBOE_1d.parquet
│   │   │   │   │   ├── CBRE/
│   │   │   │   │   │   ├── CBRE_1d.parquet
│   │   │   │   │   ├── CCI/
│   │   │   │   │   │   ├── CCI_1d.parquet
│   │   │   │   │   ├── CCL/
│   │   │   │   │   │   ├── CCL_1d.parquet
│   │   │   │   │   │   ├── CCL_1min.parquet
│   │   │   │   │   ├── CDNS/
│   │   │   │   │   │   ├── CDNS_1d.parquet
│   │   │   │   │   │   ├── CDNS_1min.parquet
│   │   │   │   │   ├── CDW/
│   │   │   │   │   │   ├── CDW_1d.parquet
│   │   │   │   │   ├── CE/
│   │   │   │   │   │   ├── CE_1d.parquet
│   │   │   │   │   ├── CEG/
│   │   │   │   │   │   ├── CEG_15min.parquet
│   │   │   │   │   │   ├── CEG_1d.parquet
│   │   │   │   │   │   ├── CEG_1min.parquet
│   │   │   │   │   ├── CF/
│   │   │   │   │   │   ├── CF_1d.parquet
│   │   │   │   │   ├── CFG/
│   │   │   │   │   │   ├── CFG_1d.parquet
│   │   │   │   │   ├── CHD/
│   │   │   │   │   │   ├── CHD_1d.parquet
│   │   │   │   │   ├── CHRW/
│   │   │   │   │   │   ├── CHRW_1d.parquet
│   │   │   │   │   ├── CHTR/
│   │   │   │   │   │   ├── CHTR_1d.parquet
│   │   │   │   │   ├── CI/
│   │   │   │   │   │   ├── CI_1d.parquet
│   │   │   │   │   ├── CINF/
│   │   │   │   │   │   ├── CINF_1d.parquet
│   │   │   │   │   ├── CL/
│   │   │   │   │   │   ├── CL_1d.parquet
│   │   │   │   │   ├── CLX/
│   │   │   │   │   │   ├── CLX_1d.parquet
│   │   │   │   │   │   ├── CLX_1min.parquet
│   │   │   │   │   ├── CMA/
│   │   │   │   │   │   ├── CMA_1d.parquet
│   │   │   │   │   ├── CMCSA/
│   │   │   │   │   │   ├── CMCSA_1d.parquet
│   │   │   │   │   │   ├── CMCSA_1min.parquet
│   │   │   │   │   ├── CME/
│   │   │   │   │   │   ├── CME_1d.parquet
│   │   │   │   │   ├── CMG/
│   │   │   │   │   │   ├── CMG_1d.parquet
│   │   │   │   │   │   ├── CMG_1min.parquet
│   │   │   │   │   ├── CMI/
│   │   │   │   │   │   ├── CMI_1d.parquet
│   │   │   │   │   ├── CMS/
│   │   │   │   │   │   ├── CMS_1d.parquet
│   │   │   │   │   ├── CNC/
│   │   │   │   │   │   ├── CNC_1d.parquet
│   │   │   │   │   ├── CNP/
│   │   │   │   │   │   ├── CNP_1d.parquet
│   │   │   │   │   ├── COF/
│   │   │   │   │   │   ├── COF_1d.parquet
│   │   │   │   │   │   ├── COF_1min.parquet
│   │   │   │   │   ├── COO/
│   │   │   │   │   │   ├── COO_1d.parquet
│   │   │   │   │   ├── COP/
│   │   │   │   │   │   ├── COP_1d.parquet
│   │   │   │   │   │   ├── COP_1min.parquet
│   │   │   │   │   ├── COR/
│   │   │   │   │   │   ├── COR_1d.parquet
│   │   │   │   │   ├── COST/
│   │   │   │   │   │   ├── COST_1d.parquet
│   │   │   │   │   ├── CPAY/
│   │   │   │   │   │   ├── CPAY_1d.parquet
│   │   │   │   │   ├── CPB/
│   │   │   │   │   │   ├── CPB_1d.parquet
│   │   │   │   │   ├── CPRT/
│   │   │   │   │   │   ├── CPRT_1d.parquet
│   │   │   │   │   │   ├── CPRT_1min.parquet
│   │   │   │   │   ├── CPT/
│   │   │   │   │   │   ├── CPT_1d.parquet
│   │   │   │   │   ├── CRCL/
│   │   │   │   │   │   ├── CRCL_15min.parquet
│   │   │   │   │   │   ├── CRCL_1min.parquet
│   │   │   │   │   ├── CRL/
│   │   │   │   │   │   ├── CRL_1d.parquet
│   │   │   │   │   ├── CRM/
│   │   │   │   │   │   ├── CRM_1d.parquet
│   │   │   │   │   │   ├── CRM_1min.parquet
│   │   │   │   │   ├── CRWD/
│   │   │   │   │   │   ├── CRWD_1d.parquet
│   │   │   │   │   ├── CSCO/
│   │   │   │   │   │   ├── CSCO_1d.parquet
│   │   │   │   │   │   ├── CSCO_1min.parquet
│   │   │   │   │   ├── CSGP/
│   │   │   │   │   │   ├── CSGP_1d.parquet
│   │   │   │   │   ├── CSX/
│   │   │   │   │   │   ├── CSX_1d.parquet
│   │   │   │   │   ├── CTAS/
│   │   │   │   │   │   ├── CTAS_1d.parquet
│   │   │   │   │   ├── CTRA/
│   │   │   │   │   │   ├── CTRA_1d.parquet
│   │   │   │   │   │   ├── CTRA_1min.parquet
│   │   │   │   │   ├── CTSH/
│   │   │   │   │   │   ├── CTSH_1d.parquet
│   │   │   │   │   ├── CTVA/
│   │   │   │   │   │   ├── CTVA_1d.parquet
│   │   │   │   │   ├── CVS/
│   │   │   │   │   │   ├── CVS_1d.parquet
│   │   │   │   │   ├── CVX/
│   │   │   │   │   │   ├── CVX_1d.parquet
│   │   │   │   │   │   ├── CVX_1min.parquet
│   │   │   │   │   ├── CZR/
│   │   │   │   │   │   ├── CZR_1d.parquet
│   │   │   │   │   ├── D/
│   │   │   │   │   │   ├── D_1d.parquet
│   │   │   │   │   ├── DAL/
│   │   │   │   │   │   ├── DAL_1d.parquet
│   │   │   │   │   │   ├── DAL_1min.parquet
│   │   │   │   │   ├── DD/
│   │   │   │   │   │   ├── DD_1d.parquet
│   │   │   │   │   │   ├── DD_1min.parquet
│   │   │   │   │   ├── DE/
│   │   │   │   │   │   ├── DE_1d.parquet
│   │   │   │   │   ├── DECK/
│   │   │   │   │   │   ├── DECK_1d.parquet
│   │   │   │   │   │   ├── DECK_1min.parquet
│   │   │   │   │   ├── DG/
│   │   │   │   │   │   ├── DG_1d.parquet
│   │   │   │   │   │   ├── DG_1min.parquet
│   │   │   │   │   ├── DGX/
│   │   │   │   │   │   ├── DGX_1d.parquet
│   │   │   │   │   ├── DHI/
│   │   │   │   │   │   ├── DHI_1d.parquet
│   │   │   │   │   │   ├── DHI_1min.parquet
│   │   │   │   │   ├── DHR/
│   │   │   │   │   │   ├── DHR_1d.parquet
│   │   │   │   │   ├── DIS/
│   │   │   │   │   │   ├── DIS_1d.parquet
│   │   │   │   │   │   ├── DIS_1min.parquet
│   │   │   │   │   ├── DLR/
│   │   │   │   │   │   ├── DLR_1d.parquet
│   │   │   │   │   │   ├── DLR_1min.parquet
│   │   │   │   │   ├── DLTR/
│   │   │   │   │   │   ├── DLTR_1d.parquet
│   │   │   │   │   ├── DOC/
│   │   │   │   │   │   ├── DOC_1d.parquet
│   │   │   │   │   ├── DOV/
│   │   │   │   │   │   ├── DOV_1d.parquet
│   │   │   │   │   ├── DOW/
│   │   │   │   │   │   ├── DOW_1d.parquet
│   │   │   │   │   │   ├── DOW_1min.parquet
│   │   │   │   │   ├── DPZ/
│   │   │   │   │   │   ├── DPZ_1d.parquet
│   │   │   │   │   ├── DRI/
│   │   │   │   │   │   ├── DRI_1d.parquet
│   │   │   │   │   ├── DTE/
│   │   │   │   │   │   ├── DTE_1d.parquet
│   │   │   │   │   ├── DUK/
│   │   │   │   │   │   ├── DUK_1d.parquet
│   │   │   │   │   ├── DVA/
│   │   │   │   │   │   ├── DVA_1d.parquet
│   │   │   │   │   ├── DVN/
│   │   │   │   │   │   ├── DVN_1d.parquet
│   │   │   │   │   │   ├── DVN_1min.parquet
│   │   │   │   │   ├── DXCM/
│   │   │   │   │   │   ├── DXCM_1d.parquet
│   │   │   │   │   ├── EA/
│   │   │   │   │   │   ├── EA_1d.parquet
│   │   │   │   │   ├── EBAY/
│   │   │   │   │   │   ├── EBAY_1d.parquet
│   │   │   │   │   │   ├── EBAY_1min.parquet
│   │   │   │   │   ├── ECL/
│   │   │   │   │   │   ├── ECL_1d.parquet
│   │   │   │   │   ├── ED/
│   │   │   │   │   │   ├── ED_1d.parquet
│   │   │   │   │   ├── EFX/
│   │   │   │   │   │   ├── EFX_1d.parquet
│   │   │   │   │   │   ├── EFX_1min.parquet
│   │   │   │   │   ├── EG/
│   │   │   │   │   │   ├── EG_1d.parquet
│   │   │   │   │   ├── EIX/
│   │   │   │   │   │   ├── EIX_1d.parquet
│   │   │   │   │   ├── EL/
│   │   │   │   │   │   ├── EL_1d.parquet
│   │   │   │   │   │   ├── EL_1min.parquet
│   │   │   │   │   ├── ELV/
│   │   │   │   │   │   ├── ELV_1d.parquet
│   │   │   │   │   ├── EMN/
│   │   │   │   │   │   ├── EMN_1d.parquet
│   │   │   │   │   ├── EMR/
│   │   │   │   │   │   ├── EMR_1d.parquet
│   │   │   │   │   ├── ENPH/
│   │   │   │   │   │   ├── ENPH_1d.parquet
│   │   │   │   │   │   ├── ENPH_1min.parquet
│   │   │   │   │   ├── EOG/
│   │   │   │   │   │   ├── EOG_1d.parquet
│   │   │   │   │   ├── EPAM/
│   │   │   │   │   │   ├── EPAM_1d.parquet
│   │   │   │   │   ├── EQIX/
│   │   │   │   │   │   ├── EQIX_1d.parquet
│   │   │   │   │   ├── EQR/
│   │   │   │   │   │   ├── EQR_1d.parquet
│   │   │   │   │   ├── EQT/
│   │   │   │   │   │   ├── EQT_1d.parquet
│   │   │   │   │   │   ├── EQT_1min.parquet
│   │   │   │   │   ├── ERIE/
│   │   │   │   │   │   ├── ERIE_1d.parquet
│   │   │   │   │   ├── ES/
│   │   │   │   │   │   ├── ES_1d.parquet
│   │   │   │   │   ├── ESS/
│   │   │   │   │   │   ├── ESS_1d.parquet
│   │   │   │   │   ├── ETN/
│   │   │   │   │   │   ├── ETN_1d.parquet
│   │   │   │   │   │   ├── ETN_1min.parquet
│   │   │   │   │   ├── ETR/
│   │   │   │   │   │   ├── ETR_1d.parquet
│   │   │   │   │   ├── ETSY/
│   │   │   │   │   │   ├── ETSY_1d.parquet
│   │   │   │   │   │   ├── ETSY_1min.parquet
│   │   │   │   │   ├── EVRG/
│   │   │   │   │   │   ├── EVRG_1d.parquet
│   │   │   │   │   ├── EW/
│   │   │   │   │   │   ├── EW_1d.parquet
│   │   │   │   │   ├── EXC/
│   │   │   │   │   │   ├── EXC_1d.parquet
│   │   │   │   │   ├── EXPD/
│   │   │   │   │   │   ├── EXPD_1d.parquet
│   │   │   │   │   ├── EXPE/
│   │   │   │   │   │   ├── EXPE_1d.parquet
│   │   │   │   │   ├── EXR/
│   │   │   │   │   │   ├── EXR_1d.parquet
│   │   │   │   │   ├── F/
│   │   │   │   │   │   ├── F_1d.parquet
│   │   │   │   │   │   ├── F_1min.parquet
│   │   │   │   │   ├── FANG/
│   │   │   │   │   │   ├── FANG_1d.parquet
│   │   │   │   │   │   ├── FANG_1min.parquet
│   │   │   │   │   ├── FAST/
│   │   │   │   │   │   ├── FAST_1d.parquet
│   │   │   │   │   │   ├── FAST_1min.parquet
│   │   │   │   │   ├── FCX/
│   │   │   │   │   │   ├── FCX_1d.parquet
│   │   │   │   │   │   ├── FCX_1min.parquet
│   │   │   │   │   ├── FDS/
│   │   │   │   │   │   ├── FDS_1d.parquet
│   │   │   │   │   ├── FDX/
│   │   │   │   │   │   ├── FDX_1d.parquet
│   │   │   │   │   │   ├── FDX_1min.parquet
│   │   │   │   │   ├── FE/
│   │   │   │   │   │   ├── FE_1d.parquet
│   │   │   │   │   ├── FFIV/
│   │   │   │   │   │   ├── FFIV_1d.parquet
│   │   │   │   │   ├── FI/
│   │   │   │   │   │   ├── FI_1d.parquet
│   │   │   │   │   │   ├── FI_1min.parquet
│   │   │   │   │   ├── FICO/
│   │   │   │   │   │   ├── FICO_1d.parquet
│   │   │   │   │   ├── FIS/
│   │   │   │   │   │   ├── FIS_1d.parquet
│   │   │   │   │   ├── FITB/
│   │   │   │   │   │   ├── FITB_1d.parquet
│   │   │   │   │   ├── FMC/
│   │   │   │   │   │   ├── FMC_1d.parquet
│   │   │   │   │   │   ├── FMC_1min.parquet
│   │   │   │   │   ├── FOX/
│   │   │   │   │   │   ├── FOX_1d.parquet
│   │   │   │   │   │   ├── FOX_1min.parquet
│   │   │   │   │   ├── FOXA/
│   │   │   │   │   │   ├── FOXA_1d.parquet
│   │   │   │   │   ├── FRT/
│   │   │   │   │   │   ├── FRT_1d.parquet
│   │   │   │   │   ├── FSLR/
│   │   │   │   │   │   ├── FSLR_1d.parquet
│   │   │   │   │   │   ├── FSLR_1min.parquet
│   │   │   │   │   ├── FTNT/
│   │   │   │   │   │   ├── FTNT_1d.parquet
│   │   │   │   │   │   ├── FTNT_1min.parquet
│   │   │   │   │   ├── FTV/
│   │   │   │   │   │   ├── FTV_1d.parquet
│   │   │   │   │   │   ├── FTV_1min.parquet
│   │   │   │   │   ├── GD/
│   │   │   │   │   │   ├── GD_1d.parquet
│   │   │   │   │   ├── GDDY/
│   │   │   │   │   │   ├── GDDY_1d.parquet
│   │   │   │   │   ├── GE/
│   │   │   │   │   │   ├── GE_1d.parquet
│   │   │   │   │   ├── GEHC/
│   │   │   │   │   │   ├── GEHC_1d.parquet
│   │   │   │   │   ├── GEN/
│   │   │   │   │   │   ├── GEN_1d.parquet
│   │   │   │   │   │   ├── GEN_1min.parquet
│   │   │   │   │   ├── GEV/
│   │   │   │   │   │   ├── GEV_1d.parquet
│   │   │   │   │   ├── GILD/
│   │   │   │   │   │   ├── GILD_1d.parquet
│   │   │   │   │   │   ├── GILD_1min.parquet
│   │   │   │   │   ├── GIS/
│   │   │   │   │   │   ├── GIS_1d.parquet
│   │   │   │   │   ├── GL/
│   │   │   │   │   │   ├── GL_1d.parquet
│   │   │   │   │   ├── GLW/
│   │   │   │   │   │   ├── GLW_1d.parquet
│   │   │   │   │   ├── GM/
│   │   │   │   │   │   ├── GM_1d.parquet
│   │   │   │   │   │   ├── GM_1min.parquet
│   │   │   │   │   ├── GNRC/
│   │   │   │   │   │   ├── GNRC_1d.parquet
│   │   │   │   │   │   ├── GNRC_1min.parquet
│   │   │   │   │   ├── GOOG/
│   │   │   │   │   │   ├── GOOG_15min.parquet
│   │   │   │   │   │   ├── GOOG_1d.parquet
│   │   │   │   │   │   ├── GOOG_1day.parquet
│   │   │   │   │   │   ├── GOOG_1min.parquet
│   │   │   │   │   ├── GOOGL/
│   │   │   │   │   │   ├── GOOGL_15min.parquet
│   │   │   │   │   │   ├── GOOGL_1d.parquet
│   │   │   │   │   │   ├── GOOGL_1day.parquet
│   │   │   │   │   │   ├── GOOGL_1min.parquet
│   │   │   │   │   ├── GPC/
│   │   │   │   │   │   ├── GPC_1d.parquet
│   │   │   │   │   ├── GPN/
│   │   │   │   │   │   ├── GPN_1d.parquet
│   │   │   │   │   ├── GRMN/
│   │   │   │   │   │   ├── GRMN_1d.parquet
│   │   │   │   │   ├── GS/
│   │   │   │   │   │   ├── GS_1d.parquet
│   │   │   │   │   ├── GWW/
│   │   │   │   │   │   ├── GWW_1d.parquet
│   │   │   │   │   ├── HAL/
│   │   │   │   │   │   ├── HAL_1d.parquet
│   │   │   │   │   │   ├── HAL_1min.parquet
│   │   │   │   │   ├── HAS/
│   │   │   │   │   │   ├── HAS_1d.parquet
│   │   │   │   │   ├── HBAN/
│   │   │   │   │   │   ├── HBAN_1d.parquet
│   │   │   │   │   │   ├── HBAN_1min.parquet
│   │   │   │   │   ├── HCA/
│   │   │   │   │   │   ├── HCA_1d.parquet
│   │   │   │   │   ├── HD/
│   │   │   │   │   │   ├── HD_1d.parquet
│   │   │   │   │   ├── HES/
│   │   │   │   │   │   ├── HES_1d.parquet
│   │   │   │   │   ├── HIG/
│   │   │   │   │   │   ├── HIG_1d.parquet
│   │   │   │   │   │   ├── HIG_1min.parquet
│   │   │   │   │   ├── HII/
│   │   │   │   │   │   ├── HII_1d.parquet
│   │   │   │   │   ├── HLT/
│   │   │   │   │   │   ├── HLT_1d.parquet
│   │   │   │   │   ├── HOLX/
│   │   │   │   │   │   ├── HOLX_1d.parquet
│   │   │   │   │   ├── HON/
│   │   │   │   │   │   ├── HON_1d.parquet
│   │   │   │   │   │   ├── HON_1min.parquet
│   │   │   │   │   ├── HPE/
│   │   │   │   │   │   ├── HPE_1d.parquet
│   │   │   │   │   │   ├── HPE_1min.parquet
│   │   │   │   │   ├── HPQ/
│   │   │   │   │   │   ├── HPQ_1d.parquet
│   │   │   │   │   │   ├── HPQ_1min.parquet
│   │   │   │   │   ├── HRL/
│   │   │   │   │   │   ├── HRL_1d.parquet
│   │   │   │   │   │   ├── HRL_1min.parquet
│   │   │   │   │   ├── HSIC/
│   │   │   │   │   │   ├── HSIC_1d.parquet
│   │   │   │   │   ├── HST/
│   │   │   │   │   │   ├── HST_1d.parquet
│   │   │   │   │   ├── HSY/
│   │   │   │   │   │   ├── HSY_1d.parquet
│   │   │   │   │   ├── HUBB/
│   │   │   │   │   │   ├── HUBB_1d.parquet
│   │   │   │   │   ├── HUM/
│   │   │   │   │   │   ├── HUM_1d.parquet
│   │   │   │   │   ├── HWM/
│   │   │   │   │   │   ├── HWM_1d.parquet
│   │   │   │   │   ├── IBM/
│   │   │   │   │   │   ├── IBM_1d.parquet
│   │   │   │   │   │   ├── IBM_1min.parquet
│   │   │   │   │   ├── ICE/
│   │   │   │   │   │   ├── ICE_1d.parquet
│   │   │   │   │   ├── IDXX/
│   │   │   │   │   │   ├── IDXX_1d.parquet
│   │   │   │   │   ├── IEX/
│   │   │   │   │   │   ├── IEX_1d.parquet
│   │   │   │   │   ├── IFF/
│   │   │   │   │   │   ├── IFF_1d.parquet
│   │   │   │   │   ├── ILMN/
│   │   │   │   │   │   ├── ILMN_1d.parquet
│   │   │   │   │   │   ├── ILMN_1min.parquet
│   │   │   │   │   ├── INCY/
│   │   │   │   │   │   ├── INCY_1d.parquet
│   │   │   │   │   ├── INTC/
│   │   │   │   │   │   ├── INTC_15min.parquet
│   │   │   │   │   │   ├── INTC_1d.parquet
│   │   │   │   │   │   ├── INTC_1min.parquet
│   │   │   │   │   ├── INTU/
│   │   │   │   │   │   ├── INTU_1d.parquet
│   │   │   │   │   ├── INVH/
│   │   │   │   │   │   ├── INVH_1d.parquet
│   │   │   │   │   ├── IP/
│   │   │   │   │   │   ├── IP_1d.parquet
│   │   │   │   │   │   ├── IP_1min.parquet
│   │   │   │   │   ├── IPG/
│   │   │   │   │   │   ├── IPG_1d.parquet
│   │   │   │   │   ├── IQV/
│   │   │   │   │   │   ├── IQV_1d.parquet
│   │   │   │   │   ├── IR/
│   │   │   │   │   │   ├── IR_1d.parquet
│   │   │   │   │   │   ├── IR_1min.parquet
│   │   │   │   │   ├── IRM/
│   │   │   │   │   │   ├── IRM_1d.parquet
│   │   │   │   │   ├── ISRG/
│   │   │   │   │   │   ├── ISRG_1d.parquet
│   │   │   │   │   ├── IT/
│   │   │   │   │   │   ├── IT_1d.parquet
│   │   │   │   │   ├── ITW/
│   │   │   │   │   │   ├── ITW_1d.parquet
│   │   │   │   │   ├── IVZ/
│   │   │   │   │   │   ├── IVZ_1d.parquet
│   │   │   │   │   ├── J/
│   │   │   │   │   │   ├── J_1d.parquet
│   │   │   │   │   ├── JBHT/
│   │   │   │   │   │   ├── JBHT_1d.parquet
│   │   │   │   │   ├── JBL/
│   │   │   │   │   │   ├── JBL_1d.parquet
│   │   │   │   │   │   ├── JBL_1min.parquet
│   │   │   │   │   ├── JCI/
│   │   │   │   │   │   ├── JCI_1d.parquet
│   │   │   │   │   ├── JKHY/
│   │   │   │   │   │   ├── JKHY_1d.parquet
│   │   │   │   │   ├── JNJ/
│   │   │   │   │   │   ├── JNJ_1d.parquet
│   │   │   │   │   │   ├── JNJ_1min.parquet
│   │   │   │   │   ├── JNPR/
│   │   │   │   │   │   ├── JNPR_1d.parquet
│   │   │   │   │   ├── JPM/
│   │   │   │   │   │   ├── JPM_1d.parquet
│   │   │   │   │   │   ├── JPM_1min.parquet
│   │   │   │   │   ├── K/
│   │   │   │   │   │   ├── K_1d.parquet
│   │   │   │   │   ├── KDP/
│   │   │   │   │   │   ├── KDP_1d.parquet
│   │   │   │   │   │   ├── KDP_1min.parquet
│   │   │   │   │   ├── KEY/
│   │   │   │   │   │   ├── KEY_1d.parquet
│   │   │   │   │   ├── KEYS/
│   │   │   │   │   │   ├── KEYS_1d.parquet
│   │   │   │   │   ├── KHC/
│   │   │   │   │   │   ├── KHC_1d.parquet
│   │   │   │   │   │   ├── KHC_1min.parquet
│   │   │   │   │   ├── KIM/
│   │   │   │   │   │   ├── KIM_1d.parquet
│   │   │   │   │   ├── KKR/
│   │   │   │   │   │   ├── KKR_1d.parquet
│   │   │   │   │   │   ├── KKR_1min.parquet
│   │   │   │   │   ├── KLAC/
│   │   │   │   │   │   ├── KLAC_1d.parquet
│   │   │   │   │   ├── KMB/
│   │   │   │   │   │   ├── KMB_1d.parquet
│   │   │   │   │   ├── KMI/
│   │   │   │   │   │   ├── KMI_1d.parquet
│   │   │   │   │   │   ├── KMI_1min.parquet
│   │   │   │   │   ├── KMX/
│   │   │   │   │   │   ├── KMX_1d.parquet
│   │   │   │   │   ├── KO/
│   │   │   │   │   │   ├── KO_1d.parquet
│   │   │   │   │   │   ├── KO_1min.parquet
│   │   │   │   │   ├── KR/
│   │   │   │   │   │   ├── KR_1d.parquet
│   │   │   │   │   ├── KVUE/
│   │   │   │   │   │   ├── KVUE_1d.parquet
│   │   │   │   │   │   ├── KVUE_1min.parquet
│   │   │   │   │   ├── L/
│   │   │   │   │   │   ├── L_1d.parquet
│   │   │   │   │   ├── LDOS/
│   │   │   │   │   │   ├── LDOS_1d.parquet
│   │   │   │   │   ├── LEN/
│   │   │   │   │   │   ├── LEN_1d.parquet
│   │   │   │   │   ├── LH/
│   │   │   │   │   │   ├── LH_1d.parquet
│   │   │   │   │   ├── LHX/
│   │   │   │   │   │   ├── LHX_1d.parquet
│   │   │   │   │   ├── LII/
│   │   │   │   │   │   ├── LII_1d.parquet
│   │   │   │   │   ├── LIN/
│   │   │   │   │   │   ├── LIN_1d.parquet
│   │   │   │   │   ├── LKQ/
│   │   │   │   │   │   ├── LKQ_1d.parquet
│   │   │   │   │   │   ├── LKQ_1min.parquet
│   │   │   │   │   ├── LLY/
│   │   │   │   │   │   ├── LLY_1d.parquet
│   │   │   │   │   ├── LMT/
│   │   │   │   │   │   ├── LMT_1d.parquet
│   │   │   │   │   │   ├── LMT_1min.parquet
│   │   │   │   │   ├── LNT/
│   │   │   │   │   │   ├── LNT_1d.parquet
│   │   │   │   │   ├── LOW/
│   │   │   │   │   │   ├── LOW_1d.parquet
│   │   │   │   │   ├── LRCX/
│   │   │   │   │   │   ├── LRCX_1d.parquet
│   │   │   │   │   │   ├── LRCX_1min.parquet
│   │   │   │   │   ├── LULU/
│   │   │   │   │   │   ├── LULU_1d.parquet
│   │   │   │   │   │   ├── LULU_1min.parquet
│   │   │   │   │   ├── LUV/
│   │   │   │   │   │   ├── LUV_1d.parquet
│   │   │   │   │   │   ├── LUV_1min.parquet
│   │   │   │   │   ├── LVS/
│   │   │   │   │   │   ├── LVS_1d.parquet
│   │   │   │   │   ├── LW/
│   │   │   │   │   │   ├── LW_1d.parquet
│   │   │   │   │   ├── LYB/
│   │   │   │   │   │   ├── LYB_1d.parquet
│   │   │   │   │   ├── LYV/
│   │   │   │   │   │   ├── LYV_1d.parquet
│   │   │   │   │   ├── MA/
│   │   │   │   │   │   ├── MA_1d.parquet
│   │   │   │   │   ├── MAA/
│   │   │   │   │   │   ├── MAA_1d.parquet
│   │   │   │   │   ├── MAR/
│   │   │   │   │   │   ├── MAR_1d.parquet
│   │   │   │   │   │   ├── MAR_1min.parquet
│   │   │   │   │   ├── MAS/
│   │   │   │   │   │   ├── MAS_1d.parquet
│   │   │   │   │   │   ├── MAS_1min.parquet
│   │   │   │   │   ├── MBC/
│   │   │   │   │   │   ├── MBC_1d.parquet
│   │   │   │   │   ├── MCD/
│   │   │   │   │   │   ├── MCD_1d.parquet
│   │   │   │   │   │   ├── MCD_1min.parquet
│   │   │   │   │   ├── MCHP/
│   │   │   │   │   │   ├── MCHP_1d.parquet
│   │   │   │   │   ├── MCK/
│   │   │   │   │   │   ├── MCK_1d.parquet
│   │   │   │   │   ├── MCO/
│   │   │   │   │   │   ├── MCO_1d.parquet
│   │   │   │   │   ├── MDLZ/
│   │   │   │   │   │   ├── MDLZ_1d.parquet
│   │   │   │   │   ├── MDT/
│   │   │   │   │   │   ├── MDT_1d.parquet
│   │   │   │   │   │   ├── MDT_1min.parquet
│   │   │   │   │   ├── MET/
│   │   │   │   │   │   ├── MET_1d.parquet
│   │   │   │   │   ├── META/
│   │   │   │   │   │   ├── META_1d.parquet
│   │   │   │   │   ├── MGM/
│   │   │   │   │   │   ├── MGM_1d.parquet
│   │   │   │   │   ├── MHK/
│   │   │   │   │   │   ├── MHK_1d.parquet
│   │   │   │   │   ├── MKC/
│   │   │   │   │   │   ├── MKC_1d.parquet
│   │   │   │   │   ├── MKTX/
│   │   │   │   │   │   ├── MKTX_1d.parquet
│   │   │   │   │   ├── MLM/
│   │   │   │   │   │   ├── MLM_1d.parquet
│   │   │   │   │   ├── MMC/
│   │   │   │   │   │   ├── MMC_1d.parquet
│   │   │   │   │   ├── MMM/
│   │   │   │   │   │   ├── MMM_1d.parquet
│   │   │   │   │   │   ├── MMM_1min.parquet
│   │   │   │   │   ├── MNST/
│   │   │   │   │   │   ├── MNST_1d.parquet
│   │   │   │   │   ├── MO/
│   │   │   │   │   │   ├── MO_1d.parquet
│   │   │   │   │   │   ├── MO_1min.parquet
│   │   │   │   │   ├── MOH/
│   │   │   │   │   │   ├── MOH_1d.parquet
│   │   │   │   │   ├── MOS/
│   │   │   │   │   │   ├── MOS_1d.parquet
│   │   │   │   │   │   ├── MOS_1min.parquet
│   │   │   │   │   ├── MPC/
│   │   │   │   │   │   ├── MPC_1d.parquet
│   │   │   │   │   ├── MPWR/
│   │   │   │   │   │   ├── MPWR_1d.parquet
│   │   │   │   │   ├── MRK/
│   │   │   │   │   │   ├── MRK_1d.parquet
│   │   │   │   │   │   ├── MRK_1min.parquet
│   │   │   │   │   ├── MRNA/
│   │   │   │   │   │   ├── MRNA_15min.parquet
│   │   │   │   │   │   ├── MRNA_1d.parquet
│   │   │   │   │   │   ├── MRNA_1min.parquet
│   │   │   │   │   ├── MS/
│   │   │   │   │   │   ├── MS_1d.parquet
│   │   │   │   │   │   ├── MS_1min.parquet
│   │   │   │   │   ├── MSCI/
│   │   │   │   │   │   ├── MSCI_1d.parquet
│   │   │   │   │   ├── MSFT/
│   │   │   │   │   │   ├── MSFT_1d.parquet
│   │   │   │   │   │   ├── MSFT_1day.parquet
│   │   │   │   │   │   ├── MSFT_1min.parquet
│   │   │   │   │   ├── MSI/
│   │   │   │   │   │   ├── MSI_1d.parquet
│   │   │   │   │   ├── MTB/
│   │   │   │   │   │   ├── MTB_1d.parquet
│   │   │   │   │   ├── MTCH/
│   │   │   │   │   │   ├── MTCH_1d.parquet
│   │   │   │   │   ├── MTD/
│   │   │   │   │   │   ├── MTD_1d.parquet
│   │   │   │   │   ├── MU/
│   │   │   │   │   │   ├── MU_15min.parquet
│   │   │   │   │   │   ├── MU_1d.parquet
│   │   │   │   │   │   ├── MU_1min.parquet
│   │   │   │   │   ├── NCLH/
│   │   │   │   │   │   ├── NCLH_1d.parquet
│   │   │   │   │   │   ├── NCLH_1min.parquet
│   │   │   │   │   ├── NDAQ/
│   │   │   │   │   │   ├── NDAQ_1d.parquet
│   │   │   │   │   │   ├── NDAQ_1min.parquet
│   │   │   │   │   ├── NDSN/
│   │   │   │   │   │   ├── NDSN_1d.parquet
│   │   │   │   │   ├── NEE/
│   │   │   │   │   │   ├── NEE_1d.parquet
│   │   │   │   │   │   ├── NEE_1min.parquet
│   │   │   │   │   ├── NEM/
│   │   │   │   │   │   ├── NEM_1d.parquet
│   │   │   │   │   │   ├── NEM_1min.parquet
│   │   │   │   │   ├── NFLX/
│   │   │   │   │   │   ├── NFLX_1d.parquet
│   │   │   │   │   ├── NI/
│   │   │   │   │   │   ├── NI_1d.parquet
│   │   │   │   │   ├── NKE/
│   │   │   │   │   │   ├── NKE_15min.parquet
│   │   │   │   │   │   ├── NKE_1d.parquet
│   │   │   │   │   │   ├── NKE_1min.parquet
│   │   │   │   │   ├── NOC/
│   │   │   │   │   │   ├── NOC_1d.parquet
│   │   │   │   │   ├── NOW/
│   │   │   │   │   │   ├── NOW_1d.parquet
│   │   │   │   │   ├── NRG/
│   │   │   │   │   │   ├── NRG_1d.parquet
│   │   │   │   │   │   ├── NRG_1min.parquet
│   │   │   │   │   ├── NSC/
│   │   │   │   │   │   ├── NSC_1d.parquet
│   │   │   │   │   ├── NTAP/
│   │   │   │   │   │   ├── NTAP_1d.parquet
│   │   │   │   │   ├── NTRS/
│   │   │   │   │   │   ├── NTRS_1d.parquet
│   │   │   │   │   ├── NUE/
│   │   │   │   │   │   ├── NUE_1d.parquet
│   │   │   │   │   │   ├── NUE_1min.parquet
│   │   │   │   │   ├── NVDA/
│   │   │   │   │   │   ├── NVDA_15min.parquet
│   │   │   │   │   │   ├── NVDA_1d.parquet
│   │   │   │   │   │   ├── NVDA_1day.parquet
│   │   │   │   │   │   ├── NVDA_1hour.parquet
│   │   │   │   │   │   ├── NVDA_1min.parquet
│   │   │   │   │   ├── NVR/
│   │   │   │   │   │   ├── NVR_1d.parquet
│   │   │   │   │   ├── NWS/
│   │   │   │   │   │   ├── NWS_1d.parquet
│   │   │   │   │   ├── NWSA/
│   │   │   │   │   │   ├── NWSA_1d.parquet
│   │   │   │   │   ├── NXPI/
│   │   │   │   │   │   ├── NXPI_1d.parquet
│   │   │   │   │   │   ├── NXPI_1min.parquet
│   │   │   │   │   ├── O/
│   │   │   │   │   │   ├── O_1d.parquet
│   │   │   │   │   │   ├── O_1min.parquet
│   │   │   │   │   ├── ODFL/
│   │   │   │   │   │   ├── ODFL_1d.parquet
│   │   │   │   │   ├── OKE/
│   │   │   │   │   │   ├── OKE_1d.parquet
│   │   │   │   │   ├── OKLO/
│   │   │   │   │   │   ├── OKLO_15min.parquet
│   │   │   │   │   │   ├── OKLO_1day.parquet
│   │   │   │   │   │   ├── OKLO_1hour.parquet
│   │   │   │   │   │   ├── OKLO_1min.parquet
│   │   │   │   │   ├── OMC/
│   │   │   │   │   │   ├── OMC_1d.parquet
│   │   │   │   │   ├── ON/
│   │   │   │   │   │   ├── ON_1d.parquet
│   │   │   │   │   │   ├── ON_1min.parquet
│   │   │   │   │   ├── ORCL/
│   │   │   │   │   │   ├── ORCL_15min.parquet
│   │   │   │   │   │   ├── ORCL_1d.parquet
│   │   │   │   │   │   ├── ORCL_1min.parquet
│   │   │   │   │   ├── ORLY/
│   │   │   │   │   │   ├── ORLY_1d.parquet
│   │   │   │   │   │   ├── ORLY_1min.parquet
│   │   │   │   │   ├── OTIS/
│   │   │   │   │   │   ├── OTIS_1d.parquet
│   │   │   │   │   ├── OXY/
│   │   │   │   │   │   ├── OXY_1d.parquet
│   │   │   │   │   │   ├── OXY_1min.parquet
│   │   │   │   │   ├── PANW/
│   │   │   │   │   │   ├── PANW_1d.parquet
│   │   │   │   │   │   ├── PANW_1min.parquet
│   │   │   │   │   ├── PARA/
│   │   │   │   │   │   ├── PARA_1d.parquet
│   │   │   │   │   ├── PAYC/
│   │   │   │   │   │   ├── PAYC_1d.parquet
│   │   │   │   │   ├── PAYX/
│   │   │   │   │   │   ├── PAYX_1d.parquet
│   │   │   │   │   │   ├── PAYX_1min.parquet
│   │   │   │   │   ├── PCAR/
│   │   │   │   │   │   ├── PCAR_1d.parquet
│   │   │   │   │   ├── PCG/
│   │   │   │   │   │   ├── PCG_1d.parquet
│   │   │   │   │   │   ├── PCG_1min.parquet
│   │   │   │   │   ├── PEG/
│   │   │   │   │   │   ├── PEG_1d.parquet
│   │   │   │   │   ├── PEP/
│   │   │   │   │   │   ├── PEP_1d.parquet
│   │   │   │   │   │   ├── PEP_1min.parquet
│   │   │   │   │   ├── PFE/
│   │   │   │   │   │   ├── PFE_1d.parquet
│   │   │   │   │   │   ├── PFE_1min.parquet
│   │   │   │   │   ├── PFG/
│   │   │   │   │   │   ├── PFG_1d.parquet
│   │   │   │   │   ├── PG/
│   │   │   │   │   │   ├── PG_1d.parquet
│   │   │   │   │   │   ├── PG_1min.parquet
│   │   │   │   │   ├── PGR/
│   │   │   │   │   │   ├── PGR_1d.parquet
│   │   │   │   │   │   ├── PGR_1min.parquet
│   │   │   │   │   ├── PH/
│   │   │   │   │   │   ├── PH_1d.parquet
│   │   │   │   │   ├── PHM/
│   │   │   │   │   │   ├── PHM_1d.parquet
│   │   │   │   │   ├── PKG/
│   │   │   │   │   │   ├── PKG_1d.parquet
│   │   │   │   │   ├── PLD/
│   │   │   │   │   │   ├── PLD_1d.parquet
│   │   │   │   │   ├── PLTR/
│   │   │   │   │   │   ├── PLTR_15min.parquet
│   │   │   │   │   │   ├── PLTR_1d.parquet
│   │   │   │   │   │   ├── PLTR_1day.parquet
│   │   │   │   │   │   ├── PLTR_1hour.parquet
│   │   │   │   │   │   ├── PLTR_1min.parquet
│   │   │   │   │   ├── PM/
│   │   │   │   │   │   ├── PM_1d.parquet
│   │   │   │   │   ├── PNC/
│   │   │   │   │   │   ├── PNC_1d.parquet
│   │   │   │   │   ├── PNR/
│   │   │   │   │   │   ├── PNR_1d.parquet
│   │   │   │   │   ├── PNW/
│   │   │   │   │   │   ├── PNW_1d.parquet
│   │   │   │   │   ├── PODD/
│   │   │   │   │   │   ├── PODD_1d.parquet
│   │   │   │   │   ├── POOL/
│   │   │   │   │   │   ├── POOL_1d.parquet
│   │   │   │   │   ├── PPG/
│   │   │   │   │   │   ├── PPG_1d.parquet
│   │   │   │   │   ├── PPL/
│   │   │   │   │   │   ├── PPL_1d.parquet
│   │   │   │   │   ├── PRU/
│   │   │   │   │   │   ├── PRU_1d.parquet
│   │   │   │   │   ├── PSA/
│   │   │   │   │   │   ├── PSA_1d.parquet
│   │   │   │   │   ├── PSX/
│   │   │   │   │   │   ├── PSX_1d.parquet
│   │   │   │   │   ├── PTC/
│   │   │   │   │   │   ├── PTC_1d.parquet
│   │   │   │   │   ├── PWR/
│   │   │   │   │   │   ├── PWR_1d.parquet
│   │   │   │   │   │   ├── PWR_1min.parquet
│   │   │   │   │   ├── PYPL/
│   │   │   │   │   │   ├── PYPL_1d.parquet
│   │   │   │   │   │   ├── PYPL_1min.parquet
│   │   │   │   │   ├── QCOM/
│   │   │   │   │   │   ├── QCOM_1d.parquet
│   │   │   │   │   │   ├── QCOM_1min.parquet
│   │   │   │   │   ├── QRVO/
│   │   │   │   │   │   ├── QRVO_1d.parquet
│   │   │   │   │   ├── RCL/
│   │   │   │   │   │   ├── RCL_1d.parquet
│   │   │   │   │   │   ├── RCL_1min.parquet
│   │   │   │   │   ├── REG/
│   │   │   │   │   │   ├── REG_1d.parquet
│   │   │   │   │   ├── REGN/
│   │   │   │   │   │   ├── REGN_1d.parquet
│   │   │   │   │   ├── RF/
│   │   │   │   │   │   ├── RF_1d.parquet
│   │   │   │   │   │   ├── RF_1min.parquet
│   │   │   │   │   ├── RHI/
│   │   │   │   │   │   ├── RHI_1d.parquet
│   │   │   │   │   ├── RJF/
│   │   │   │   │   │   ├── RJF_1d.parquet
│   │   │   │   │   ├── RL/
│   │   │   │   │   │   ├── RL_1d.parquet
│   │   │   │   │   ├── RMD/
│   │   │   │   │   │   ├── RMD_1d.parquet
│   │   │   │   │   ├── ROK/
│   │   │   │   │   │   ├── ROK_1d.parquet
│   │   │   │   │   ├── ROL/
│   │   │   │   │   │   ├── ROL_1d.parquet
│   │   │   │   │   ├── ROP/
│   │   │   │   │   │   ├── ROP_1d.parquet
│   │   │   │   │   ├── ROST/
│   │   │   │   │   │   ├── ROST_1d.parquet
│   │   │   │   │   ├── RSG/
│   │   │   │   │   │   ├── RSG_1d.parquet
│   │   │   │   │   ├── RTX/
│   │   │   │   │   │   ├── RTX_1d.parquet
│   │   │   │   │   │   ├── RTX_1min.parquet
│   │   │   │   │   ├── RVTY/
│   │   │   │   │   │   ├── RVTY_1d.parquet
│   │   │   │   │   ├── SBAC/
│   │   │   │   │   │   ├── SBAC_1d.parquet
│   │   │   │   │   ├── SBUX/
│   │   │   │   │   │   ├── SBUX_1d.parquet
│   │   │   │   │   │   ├── SBUX_1min.parquet
│   │   │   │   │   ├── SCHW/
│   │   │   │   │   │   ├── SCHW_1d.parquet
│   │   │   │   │   ├── SHW/
│   │   │   │   │   │   ├── SHW_1d.parquet
│   │   │   │   │   ├── SJM/
│   │   │   │   │   │   ├── SJM_1d.parquet
│   │   │   │   │   ├── SLB/
│   │   │   │   │   │   ├── SLB_1d.parquet
│   │   │   │   │   │   ├── SLB_1min.parquet
│   │   │   │   │   ├── SMCI/
│   │   │   │   │   │   ├── SMCI_15min.parquet
│   │   │   │   │   │   ├── SMCI_1d.parquet
│   │   │   │   │   │   ├── SMCI_1min.parquet
│   │   │   │   │   ├── SNA/
│   │   │   │   │   │   ├── SNA_1d.parquet
│   │   │   │   │   ├── SNPS/
│   │   │   │   │   │   ├── SNPS_1d.parquet
│   │   │   │   │   │   ├── SNPS_1min.parquet
│   │   │   │   │   ├── SO/
│   │   │   │   │   │   ├── SO_1d.parquet
│   │   │   │   │   │   ├── SO_1min.parquet
│   │   │   │   │   ├── SOLV/
│   │   │   │   │   │   ├── SOLV_1d.parquet
│   │   │   │   │   ├── SPG/
│   │   │   │   │   │   ├── SPG_1d.parquet
│   │   │   │   │   ├── SPGI/
│   │   │   │   │   │   ├── SPGI_1d.parquet
│   │   │   │   │   ├── SPY/
│   │   │   │   │   │   ├── SPY_5min.parquet
│   │   │   │   │   ├── SRE/
│   │   │   │   │   │   ├── SRE_1d.parquet
│   │   │   │   │   ├── STE/
│   │   │   │   │   │   ├── STE_1d.parquet
│   │   │   │   │   ├── STLD/
│   │   │   │   │   │   ├── STLD_1d.parquet
│   │   │   │   │   ├── STT/
│   │   │   │   │   │   ├── STT_1d.parquet
│   │   │   │   │   ├── STX/
│   │   │   │   │   │   ├── STX_1d.parquet
│   │   │   │   │   │   ├── STX_1min.parquet
│   │   │   │   │   ├── STZ/
│   │   │   │   │   │   ├── STZ_1d.parquet
│   │   │   │   │   │   ├── STZ_1min.parquet
│   │   │   │   │   ├── SWK/
│   │   │   │   │   │   ├── SWK_1d.parquet
│   │   │   │   │   ├── SWKS/
│   │   │   │   │   │   ├── SWKS_1d.parquet
│   │   │   │   │   ├── SYF/
│   │   │   │   │   │   ├── SYF_1d.parquet
│   │   │   │   │   │   ├── SYF_1min.parquet
│   │   │   │   │   ├── SYK/
│   │   │   │   │   │   ├── SYK_1d.parquet
│   │   │   │   │   ├── SYY/
│   │   │   │   │   │   ├── SYY_1d.parquet
│   │   │   │   │   ├── T/
│   │   │   │   │   │   ├── T_1d.parquet
│   │   │   │   │   │   ├── T_1min.parquet
│   │   │   │   │   ├── TAP/
│   │   │   │   │   │   ├── TAP_1d.parquet
│   │   │   │   │   │   ├── TAP_1min.parquet
│   │   │   │   │   ├── TDG/
│   │   │   │   │   │   ├── TDG_1d.parquet
│   │   │   │   │   ├── TDY/
│   │   │   │   │   │   ├── TDY_1d.parquet
│   │   │   │   │   ├── TECH/
│   │   │   │   │   │   ├── TECH_1d.parquet
│   │   │   │   │   ├── TEL/
│   │   │   │   │   │   ├── TEL_1d.parquet
│   │   │   │   │   ├── TER/
│   │   │   │   │   │   ├── TER_1d.parquet
│   │   │   │   │   ├── TFC/
│   │   │   │   │   │   ├── TFC_1d.parquet
│   │   │   │   │   ├── TFX/
│   │   │   │   │   │   ├── TFX_1d.parquet
│   │   │   │   │   ├── TGT/
│   │   │   │   │   │   ├── TGT_1d.parquet
│   │   │   │   │   │   ├── TGT_1min.parquet
│   │   │   │   │   ├── TJX/
│   │   │   │   │   │   ├── TJX_1d.parquet
│   │   │   │   │   │   ├── TJX_1min.parquet
│   │   │   │   │   ├── TMO/
│   │   │   │   │   │   ├── TMO_1d.parquet
│   │   │   │   │   ├── TMUS/
│   │   │   │   │   │   ├── TMUS_1d.parquet
│   │   │   │   │   │   ├── TMUS_1min.parquet
│   │   │   │   │   ├── TPL/
│   │   │   │   │   │   ├── TPL_1d.parquet
│   │   │   │   │   ├── TPR/
│   │   │   │   │   │   ├── TPR_1d.parquet
│   │   │   │   │   ├── TRGP/
│   │   │   │   │   │   ├── TRGP_1d.parquet
│   │   │   │   │   ├── TRMB/
│   │   │   │   │   │   ├── TRMB_1d.parquet
│   │   │   │   │   │   ├── TRMB_1min.parquet
│   │   │   │   │   ├── TROW/
│   │   │   │   │   │   ├── TROW_1d.parquet
│   │   │   │   │   ├── TRV/
│   │   │   │   │   │   ├── TRV_1d.parquet
│   │   │   │   │   ├── TSCO/
│   │   │   │   │   │   ├── TSCO_1d.parquet
│   │   │   │   │   ├── TSLA/
│   │   │   │   │   │   ├── TSLA_15min.parquet
│   │   │   │   │   │   ├── TSLA_1d.parquet
│   │   │   │   │   │   ├── TSLA_1day.parquet
│   │   │   │   │   │   ├── TSLA_1hour.parquet
│   │   │   │   │   │   ├── TSLA_1min.parquet
│   │   │   │   │   │   ├── TSLA_5min.parquet
│   │   │   │   │   ├── TSN/
│   │   │   │   │   │   ├── TSN_1d.parquet
│   │   │   │   │   │   ├── TSN_1min.parquet
│   │   │   │   │   ├── TT/
│   │   │   │   │   │   ├── TT_1d.parquet
│   │   │   │   │   ├── TTWO/
│   │   │   │   │   │   ├── TTWO_1d.parquet
│   │   │   │   │   ├── TXN/
│   │   │   │   │   │   ├── TXN_1d.parquet
│   │   │   │   │   ├── TXT/
│   │   │   │   │   │   ├── TXT_1d.parquet
│   │   │   │   │   ├── TYL/
│   │   │   │   │   │   ├── TYL_1d.parquet
│   │   │   │   │   ├── UAL/
│   │   │   │   │   │   ├── UAL_1d.parquet
│   │   │   │   │   │   ├── UAL_1min.parquet
│   │   │   │   │   ├── UBER/
│   │   │   │   │   │   ├── UBER_15min.parquet
│   │   │   │   │   │   ├── UBER_1d.parquet
│   │   │   │   │   │   ├── UBER_1min.parquet
│   │   │   │   │   ├── UDR/
│   │   │   │   │   │   ├── UDR_1d.parquet
│   │   │   │   │   ├── UHS/
│   │   │   │   │   │   ├── UHS_1d.parquet
│   │   │   │   │   ├── ULTA/
│   │   │   │   │   │   ├── ULTA_1d.parquet
│   │   │   │   │   ├── UNH/
│   │   │   │   │   │   ├── UNH_1d.parquet
│   │   │   │   │   │   ├── UNH_1min.parquet
│   │   │   │   │   ├── UNP/
│   │   │   │   │   │   ├── UNP_1d.parquet
│   │   │   │   │   ├── UPS/
│   │   │   │   │   │   ├── UPS_1d.parquet
│   │   │   │   │   │   ├── UPS_1min.parquet
│   │   │   │   │   ├── URI/
│   │   │   │   │   │   ├── URI_1d.parquet
│   │   │   │   │   ├── USB/
│   │   │   │   │   │   ├── USB_1d.parquet
│   │   │   │   │   ├── V/
│   │   │   │   │   │   ├── V_1d.parquet
│   │   │   │   │   │   ├── V_1min.parquet
│   │   │   │   │   ├── VICI/
│   │   │   │   │   │   ├── VICI_1d.parquet
│   │   │   │   │   ├── VLO/
│   │   │   │   │   │   ├── VLO_1d.parquet
│   │   │   │   │   ├── VLTO/
│   │   │   │   │   │   ├── VLTO_1d.parquet
│   │   │   │   │   ├── VMC/
│   │   │   │   │   │   ├── VMC_1d.parquet
│   │   │   │   │   ├── VRSK/
│   │   │   │   │   │   ├── VRSK_1d.parquet
│   │   │   │   │   ├── VRSN/
│   │   │   │   │   │   ├── VRSN_1d.parquet
│   │   │   │   │   │   ├── VRSN_1min.parquet
│   │   │   │   │   ├── VRTX/
│   │   │   │   │   │   ├── VRTX_1d.parquet
│   │   │   │   │   ├── VST/
│   │   │   │   │   │   ├── VST_15min.parquet
│   │   │   │   │   │   ├── VST_1d.parquet
│   │   │   │   │   │   ├── VST_1min.parquet
│   │   │   │   │   ├── VTR/
│   │   │   │   │   │   ├── VTR_1d.parquet
│   │   │   │   │   ├── VTRS/
│   │   │   │   │   │   ├── VTRS_1d.parquet
│   │   │   │   │   ├── VZ/
│   │   │   │   │   │   ├── VZ_1d.parquet
│   │   │   │   │   │   ├── VZ_1min.parquet
│   │   │   │   │   ├── WAB/
│   │   │   │   │   │   ├── WAB_1d.parquet
│   │   │   │   │   ├── WAT/
│   │   │   │   │   │   ├── WAT_1d.parquet
│   │   │   │   │   ├── WBA/
│   │   │   │   │   │   ├── WBA_1d.parquet
│   │   │   │   │   │   ├── WBA_1min.parquet
│   │   │   │   │   ├── WBD/
│   │   │   │   │   │   ├── WBD_1d.parquet
│   │   │   │   │   │   ├── WBD_1min.parquet
│   │   │   │   │   ├── WDAY/
│   │   │   │   │   │   ├── WDAY_1d.parquet
│   │   │   │   │   ├── WDC/
│   │   │   │   │   │   ├── WDC_1d.parquet
│   │   │   │   │   │   ├── WDC_1min.parquet
│   │   │   │   │   ├── WEC/
│   │   │   │   │   │   ├── WEC_1d.parquet
│   │   │   │   │   ├── WELL/
│   │   │   │   │   │   ├── WELL_1d.parquet
│   │   │   │   │   ├── WFC/
│   │   │   │   │   │   ├── WFC_1d.parquet
│   │   │   │   │   │   ├── WFC_1min.parquet
│   │   │   │   │   ├── WM/
│   │   │   │   │   │   ├── WM_1d.parquet
│   │   │   │   │   │   ├── WM_1min.parquet
│   │   │   │   │   ├── WMB/
│   │   │   │   │   │   ├── WMB_1d.parquet
│   │   │   │   │   ├── WMT/
│   │   │   │   │   │   ├── WMT_1d.parquet
│   │   │   │   │   │   ├── WMT_1min.parquet
│   │   │   │   │   ├── WRB/
│   │   │   │   │   │   ├── WRB_1d.parquet
│   │   │   │   │   │   ├── WRB_1min.parquet
│   │   │   │   │   ├── WST/
│   │   │   │   │   │   ├── WST_1d.parquet
│   │   │   │   │   ├── WTW/
│   │   │   │   │   │   ├── WTW_1d.parquet
│   │   │   │   │   ├── WY/
│   │   │   │   │   │   ├── WY_1d.parquet
│   │   │   │   │   │   ├── WY_1min.parquet
│   │   │   │   │   ├── WYNN/
│   │   │   │   │   │   ├── WYNN_1d.parquet
│   │   │   │   │   ├── XEL/
│   │   │   │   │   │   ├── XEL_1d.parquet
│   │   │   │   │   ├── XOM/
│   │   │   │   │   │   ├── XOM_1d.parquet
│   │   │   │   │   │   ├── XOM_1min.parquet
│   │   │   │   │   ├── XRAY/
│   │   │   │   │   │   ├── XRAY_1d.parquet
│   │   │   │   │   ├── XYL/
│   │   │   │   │   │   ├── XYL_1d.parquet
│   │   │   │   │   ├── YUM/
│   │   │   │   │   │   ├── YUM_1d.parquet
│   │   │   │   │   ├── ZBH/
│   │   │   │   │   │   ├── ZBH_1d.parquet
│   │   │   │   │   ├── ZBRA/
│   │   │   │   │   │   ├── ZBRA_1d.parquet
│   │   │   │   │   ├── ZION/
│   │   │   │   │   │   ├── ZION_1d.parquet
│   │   │   │   │   ├── ZTS/
│   │   │   │   │   │   ├── ZTS_1d.parquet
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
│   │   │   ├── polygon_server.log
│   │   │   ├── requirements.txt # Python dependencies
│   │   │   ├── server.py
│   │   │   ├── start_server.py
│   │   │   ├── endpoints/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── health.py
│   │   │   │   ├── rest.py
│   │   │   │   ├── websocket.py
│   │   │   ├── polygon_cache/
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
