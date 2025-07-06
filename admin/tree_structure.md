# Project Structure
Last updated: 2025-07-06 20:43:09

```
Sirius/
│   ├── .env
│   ├── README.md # Project documentation
│   ├── fix_imports.py
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
│   ├── archive/
│   │   ├── backtest_old/
│   │   │   ├── backtest_system.py
│   │   │   ├── config.yaml
│   │   │   ├── requirements.txt # Python dependencies
│   │   │   ├── adapters/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py
│   │   │   ├── backtest/
│   │   │   │   ├── results/
│   │   │   │   │   ├── summary.json
│   │   │   │   │   ├── daily/
│   │   │   │   │   │   ├── 20250623/
│   │   │   │   │   │   │   ├── 3e65177c-11e2-4589-89ee-68f5f209accc.json
│   │   │   │   │   │   │   ├── 3e65177c-11e2-4589-89ee-68f5f209accc_bars_storage.parquet
│   │   │   │   │   │   │   ├── 3e65177c-11e2-4589-89ee-68f5f209accc_forward.parquet
│   │   │   │   │   │   │   ├── 3e65177c-11e2-4589-89ee-68f5f209accc_historical.parquet
│   │   │   │   │   │   │   ├── 4358883b-913c-4cfe-8e6b-a2ae4d2d4766.json
│   │   │   │   │   │   │   ├── 4358883b-913c-4cfe-8e6b-a2ae4d2d4766_bars_storage.parquet
│   │   │   │   │   │   │   ├── 4358883b-913c-4cfe-8e6b-a2ae4d2d4766_forward.parquet
│   │   │   │   │   │   │   ├── 4358883b-913c-4cfe-8e6b-a2ae4d2d4766_historical.parquet
│   │   │   │   │   │   │   ├── 5eb7da2e-f2e4-4fe7-8d81-47acfa154907.json
│   │   │   │   │   │   │   ├── 5eb7da2e-f2e4-4fe7-8d81-47acfa154907_bars_storage.parquet
│   │   │   │   │   │   │   ├── 5eb7da2e-f2e4-4fe7-8d81-47acfa154907_forward.parquet
│   │   │   │   │   │   │   ├── 5eb7da2e-f2e4-4fe7-8d81-47acfa154907_historical.parquet
│   │   │   │   │   │   │   ├── 5ffeba60-51f3-4c4c-aaab-391df96f6efa.json
│   │   │   │   │   │   │   ├── 5ffeba60-51f3-4c4c-aaab-391df96f6efa_bars_storage.parquet
│   │   │   │   │   │   │   ├── 5ffeba60-51f3-4c4c-aaab-391df96f6efa_forward.parquet
│   │   │   │   │   │   │   ├── 5ffeba60-51f3-4c4c-aaab-391df96f6efa_historical.parquet
│   │   │   │   │   │   │   ├── da337c86-2a51-47f6-b619-3baae45ac821.json
│   │   │   │   │   │   │   ├── da337c86-2a51-47f6-b619-3baae45ac821_bars_storage.parquet
│   │   │   │   │   │   │   ├── da337c86-2a51-47f6-b619-3baae45ac821_forward.parquet
│   │   │   │   │   │   │   ├── da337c86-2a51-47f6-b619-3baae45ac821_historical.parquet
│   │   │   │   │   │   │   ├── f5a173de-7559-401d-8862-563f70908b2d.json
│   │   │   │   │   │   │   ├── f5a173de-7559-401d-8862-563f70908b2d_bars_storage.parquet
│   │   │   │   │   │   │   ├── f5a173de-7559-401d-8862-563f70908b2d_forward.parquet
│   │   │   │   │   │   │   ├── f5a173de-7559-401d-8862-563f70908b2d_historical.parquet
│   │   │   │   │   │   ├── 20250627/
│   │   │   │   │   │   │   ├── 0a25e18b-5dc1-47d8-9234-cf9c5d6d0cd3.json
│   │   │   │   │   │   │   ├── 0a25e18b-5dc1-47d8-9234-cf9c5d6d0cd3_bars_storage.parquet
│   │   │   │   │   │   │   ├── 0a25e18b-5dc1-47d8-9234-cf9c5d6d0cd3_forward.parquet
│   │   │   │   │   │   │   ├── 0a25e18b-5dc1-47d8-9234-cf9c5d6d0cd3_historical.parquet
│   │   │   │   │   │   │   ├── 4e4572da-2a3d-4266-b5c0-9cab35dcda91.json
│   │   │   │   │   │   │   ├── 4e4572da-2a3d-4266-b5c0-9cab35dcda91_bars_storage.parquet
│   │   │   │   │   │   │   ├── 4e4572da-2a3d-4266-b5c0-9cab35dcda91_forward.parquet
│   │   │   │   │   │   │   ├── 4e4572da-2a3d-4266-b5c0-9cab35dcda91_historical.parquet
│   │   │   │   │   │   │   ├── 50dd2db9-49be-44e5-b561-9760473d8c00.json
│   │   │   │   │   │   │   ├── 50dd2db9-49be-44e5-b561-9760473d8c00_bars_storage.parquet
│   │   │   │   │   │   │   ├── 50dd2db9-49be-44e5-b561-9760473d8c00_forward.parquet
│   │   │   │   │   │   │   ├── 50dd2db9-49be-44e5-b561-9760473d8c00_historical.parquet
│   │   │   │   │   │   │   ├── 6e347956-d246-4b44-ab52-7ff56753ab80.json
│   │   │   │   │   │   │   ├── 6e347956-d246-4b44-ab52-7ff56753ab80_bars_storage.parquet
│   │   │   │   │   │   │   ├── 6e347956-d246-4b44-ab52-7ff56753ab80_forward.parquet
│   │   │   │   │   │   │   ├── 6e347956-d246-4b44-ab52-7ff56753ab80_historical.parquet
│   │   │   │   │   │   │   ├── 7e2f4f3f-c268-4f05-92a7-9ce3ddee19b5.json
│   │   │   │   │   │   │   ├── 7e2f4f3f-c268-4f05-92a7-9ce3ddee19b5_bars_storage.parquet
│   │   │   │   │   │   │   ├── 7e2f4f3f-c268-4f05-92a7-9ce3ddee19b5_forward.parquet
│   │   │   │   │   │   │   ├── 7e2f4f3f-c268-4f05-92a7-9ce3ddee19b5_historical.parquet
│   │   │   │   │   │   │   ├── 8eb9113b-ab18-4b51-b13c-4010693cb66f.json
│   │   │   │   │   │   │   ├── 8eb9113b-ab18-4b51-b13c-4010693cb66f_bars_storage.parquet
│   │   │   │   │   │   │   ├── 8eb9113b-ab18-4b51-b13c-4010693cb66f_forward.parquet
│   │   │   │   │   │   │   ├── 8eb9113b-ab18-4b51-b13c-4010693cb66f_historical.parquet
│   │   │   │   │   │   │   ├── 916284b4-f14d-44e0-a48a-f183f3e36a17.json
│   │   │   │   │   │   │   ├── 916284b4-f14d-44e0-a48a-f183f3e36a17_bars_storage.parquet
│   │   │   │   │   │   │   ├── 916284b4-f14d-44e0-a48a-f183f3e36a17_forward.parquet
│   │   │   │   │   │   │   ├── 916284b4-f14d-44e0-a48a-f183f3e36a17_historical.parquet
│   │   │   │   │   │   │   ├── 96629915-fd7c-4fca-93ca-82c0a87867c3.json
│   │   │   │   │   │   │   ├── 96629915-fd7c-4fca-93ca-82c0a87867c3_bars_storage.parquet
│   │   │   │   │   │   │   ├── 96629915-fd7c-4fca-93ca-82c0a87867c3_forward.parquet
│   │   │   │   │   │   │   ├── 96629915-fd7c-4fca-93ca-82c0a87867c3_historical.parquet
│   │   │   │   │   │   │   ├── 98aa9f87-9848-4f88-b510-21a70fd6b1da.json
│   │   │   │   │   │   │   ├── 98aa9f87-9848-4f88-b510-21a70fd6b1da_bars_storage.parquet
│   │   │   │   │   │   │   ├── 98aa9f87-9848-4f88-b510-21a70fd6b1da_forward.parquet
│   │   │   │   │   │   │   ├── 98aa9f87-9848-4f88-b510-21a70fd6b1da_historical.parquet
│   │   │   │   │   │   │   ├── a10fdbea-a9ff-43cd-800f-62a2abcf929c.json
│   │   │   │   │   │   │   ├── a10fdbea-a9ff-43cd-800f-62a2abcf929c_bars_storage.parquet
│   │   │   │   │   │   │   ├── a10fdbea-a9ff-43cd-800f-62a2abcf929c_forward.parquet
│   │   │   │   │   │   │   ├── a10fdbea-a9ff-43cd-800f-62a2abcf929c_historical.parquet
│   │   │   │   │   │   │   ├── b9b9f1d3-c3c8-49d2-93e7-45aaf2f73c44.json
│   │   │   │   │   │   │   ├── b9b9f1d3-c3c8-49d2-93e7-45aaf2f73c44_bars_storage.parquet
│   │   │   │   │   │   │   ├── b9b9f1d3-c3c8-49d2-93e7-45aaf2f73c44_forward.parquet
│   │   │   │   │   │   │   ├── b9b9f1d3-c3c8-49d2-93e7-45aaf2f73c44_historical.parquet
│   │   │   │   │   │   │   ├── c13ec9f8-6554-46d9-acfc-ff83a1a79675.json
│   │   │   │   │   │   │   ├── c13ec9f8-6554-46d9-acfc-ff83a1a79675_bars_storage.parquet
│   │   │   │   │   │   │   ├── c13ec9f8-6554-46d9-acfc-ff83a1a79675_forward.parquet
│   │   │   │   │   │   │   ├── c13ec9f8-6554-46d9-acfc-ff83a1a79675_historical.parquet
│   │   │   │   │   │   │   ├── cc582d33-afc8-4a08-a7dd-a5cd5c7bfa83.json
│   │   │   │   │   │   │   ├── cc582d33-afc8-4a08-a7dd-a5cd5c7bfa83_bars_storage.parquet
│   │   │   │   │   │   │   ├── cc582d33-afc8-4a08-a7dd-a5cd5c7bfa83_forward.parquet
│   │   │   │   │   │   │   ├── cc582d33-afc8-4a08-a7dd-a5cd5c7bfa83_historical.parquet
│   │   │   │   │   │   │   ├── cdbfdd97-1838-4b0b-8662-023e3ec6e8b0.json
│   │   │   │   │   │   │   ├── cdbfdd97-1838-4b0b-8662-023e3ec6e8b0_bars_storage.parquet
│   │   │   │   │   │   │   ├── cdbfdd97-1838-4b0b-8662-023e3ec6e8b0_forward.parquet
│   │   │   │   │   │   │   ├── cdbfdd97-1838-4b0b-8662-023e3ec6e8b0_historical.parquet
│   │   │   │   │   │   │   ├── d9891abc-008e-4be6-b9bf-271c4cfb739f.json
│   │   │   │   │   │   │   ├── d9891abc-008e-4be6-b9bf-271c4cfb739f_bars_storage.parquet
│   │   │   │   │   │   │   ├── d9891abc-008e-4be6-b9bf-271c4cfb739f_forward.parquet
│   │   │   │   │   │   │   ├── d9891abc-008e-4be6-b9bf-271c4cfb739f_historical.parquet
│   │   │   │   │   │   │   ├── f4e287e4-720c-4786-8427-e424ae9e60f5.json
│   │   │   │   │   │   │   ├── f4e287e4-720c-4786-8427-e424ae9e60f5_bars_storage.parquet
│   │   │   │   │   │   │   ├── f4e287e4-720c-4786-8427-e424ae9e60f5_forward.parquet
│   │   │   │   │   │   │   ├── f4e287e4-720c-4786-8427-e424ae9e60f5_historical.parquet
│   │   │   ├── cache/
│   │   │   │   ├── polygon_data/
│   │   │   │   │   ├── cache_metadata.json
│   │   │   │   │   ├── crcl/
│   │   │   │   │   │   ├── 1min_20250621_15f55fc0.parquet
│   │   │   │   │   │   ├── 1min_20250621_cae34506.parquet
│   │   │   │   │   │   ├── 1min_20250623_b1847ca1.parquet
│   │   │   │   │   │   ├── 1min_20250623_b487ab1e.parquet
│   │   │   │   │   │   ├── 1min_20250623_bd8286c5.parquet
│   │   │   │   │   │   ├── 1min_20250625_8335775f.parquet
│   │   │   │   │   │   ├── 1min_20250625_83958f13.parquet
│   │   │   │   │   │   ├── 1min_20250625_83e28b2e.parquet
│   │   │   │   │   │   ├── 1min_20250625_c1e50394.parquet
│   │   │   │   │   │   ├── 1min_20250625_d7b2788d.parquet
│   │   │   │   │   │   ├── 1min_20250625_e4fce04e.parquet
│   │   │   │   │   │   ├── 1min_20250625_e85c998d.parquet
│   │   │   │   │   │   ├── 1min_20250626_152bb419.parquet
│   │   │   │   │   │   ├── 1min_20250626_675eaa6a.parquet
│   │   │   │   │   │   ├── 1min_20250626_6863f174.parquet
│   │   │   │   │   │   ├── 1min_20250626_a8c3e2f1.parquet
│   │   │   │   │   │   ├── 1min_20250626_b47297a9.parquet
│   │   │   │   │   │   ├── 1min_20250626_d184ce3b.parquet
│   │   │   │   │   │   ├── 1min_20250627_03193d77.parquet
│   │   │   │   │   │   ├── 1min_20250627_04683d04.parquet
│   │   │   │   │   │   ├── 1min_20250627_07e55cde.parquet
│   │   │   │   │   │   ├── 1min_20250627_119eb25d.parquet
│   │   │   │   │   │   ├── 1min_20250627_14a60770.parquet
│   │   │   │   │   │   ├── 1min_20250627_15ec19d3.parquet
│   │   │   │   │   │   ├── 1min_20250627_17e87c6a.parquet
│   │   │   │   │   │   ├── 1min_20250627_20519301.parquet
│   │   │   │   │   │   ├── 1min_20250627_25f6b7eb.parquet
│   │   │   │   │   │   ├── 1min_20250627_293cacd8.parquet
│   │   │   │   │   │   ├── 1min_20250627_30ce2a77.parquet
│   │   │   │   │   │   ├── 1min_20250627_395d5be6.parquet
│   │   │   │   │   │   ├── 1min_20250627_538b723c.parquet
│   │   │   │   │   │   ├── 1min_20250627_5899f6ec.parquet
│   │   │   │   │   │   ├── 1min_20250627_5a4f6503.parquet
│   │   │   │   │   │   ├── 1min_20250627_600385fb.parquet
│   │   │   │   │   │   ├── 1min_20250627_64b3523f.parquet
│   │   │   │   │   │   ├── 1min_20250627_68107e0d.parquet
│   │   │   │   │   │   ├── 1min_20250627_6ab36fd1.parquet
│   │   │   │   │   │   ├── 1min_20250627_77102d44.parquet
│   │   │   │   │   │   ├── 1min_20250627_7a792903.parquet
│   │   │   │   │   │   ├── 1min_20250627_7bd9fb24.parquet
│   │   │   │   │   │   ├── 1min_20250627_80043af4.parquet
│   │   │   │   │   │   ├── 1min_20250627_830d64db.parquet
│   │   │   │   │   │   ├── 1min_20250627_877a5065.parquet
│   │   │   │   │   │   ├── 1min_20250627_8870da07.parquet
│   │   │   │   │   │   ├── 1min_20250627_8e372e57.parquet
│   │   │   │   │   │   ├── 1min_20250627_968e8f99.parquet
│   │   │   │   │   │   ├── 1min_20250627_9ecf0997.parquet
│   │   │   │   │   │   ├── 1min_20250627_9f98d2a2.parquet
│   │   │   │   │   │   ├── 1min_20250627_a6969f66.parquet
│   │   │   │   │   │   ├── 1min_20250627_aaf3192e.parquet
│   │   │   │   │   │   ├── 1min_20250627_af2670ce.parquet
│   │   │   │   │   │   ├── 1min_20250627_badbde6e.parquet
│   │   │   │   │   │   ├── 1min_20250627_c6d24690.parquet
│   │   │   │   │   │   ├── 1min_20250627_c96fd221.parquet
│   │   │   │   │   │   ├── 1min_20250627_d0e6f035.parquet
│   │   │   │   │   │   ├── 1min_20250627_d61ad347.parquet
│   │   │   │   │   │   ├── 1min_20250627_dfc2460e.parquet
│   │   │   │   │   │   ├── 1min_20250627_e86b425d.parquet
│   │   │   │   │   │   ├── 1min_20250627_f1298487.parquet
│   │   │   │   │   │   ├── 1min_20250627_f5019efb.parquet
│   │   │   │   │   │   ├── 1min_20250627_fd1044a6.parquet
│   │   │   │   │   ├── tsla/
│   │   │   │   │   │   ├── 1min_20250623_054a74ec.parquet
│   │   │   │   │   │   ├── 1min_20250623_263ecbcb.parquet
│   │   │   │   │   │   ├── 1min_20250623_27c5ebdd.parquet
│   │   │   │   │   │   ├── 1min_20250623_33474366.parquet
│   │   │   │   │   │   ├── 1min_20250623_519b1a15.parquet
│   │   │   │   │   │   ├── 1min_20250623_80d67753.parquet
│   │   │   ├── chart/
│   │   │   │   ├── __init__.py
│   │   │   ├── core/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── engine.py
│   │   │   │   ├── result_store.py
│   │   │   │   ├── signal_aggregator.py
│   │   │   ├── data/
│   │   │   │   ├── PolygonDataManager.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── polygon_data_manager.py
│   │   │   ├── debug/
│   │   │   │   ├── base_debug.py
│   │   │   │   ├── compare_results.py
│   │   │   │   ├── debug_m1_market_structure.py
│   │   │   │   ├── debug_m5_market_structure.py
│   │   │   │   ├── market_structure_debug.py
│   │   │   ├── plugins/
│   │   │   │   ├── base_plugin.py
│   │   │   │   ├── plugin_loader.py
│   │   │   │   ├── m15_ema/
│   │   │   │   │   ├── README.md # Project documentation
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── adapter.py
│   │   │   │   │   ├── aggregator.py
│   │   │   │   │   ├── plugin.py
│   │   │   │   │   ├── schema.sql
│   │   │   │   │   ├── storage.py
│   │   │   │   ├── m15_market_structure/
│   │   │   │   │   ├── README.md # Project documentation
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── adapter.py
│   │   │   │   │   ├── aggregator.py
│   │   │   │   │   ├── plugin.py
│   │   │   │   │   ├── schema.sql
│   │   │   │   │   ├── storage.py
│   │   │   │   ├── m1_ema/
│   │   │   │   │   ├── README.md # Project documentation
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── adapter.py
│   │   │   │   │   ├── aggregator.py
│   │   │   │   │   ├── plugin.py
│   │   │   │   │   ├── schema.sql
│   │   │   │   │   ├── storage.py
│   │   │   │   ├── m1_market_structure/
│   │   │   │   │   ├── README.md # Project documentation
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── adapter.py
│   │   │   │   │   ├── aggregator.py
│   │   │   │   │   ├── plugin.py
│   │   │   │   │   ├── schema.sql
│   │   │   │   │   ├── storage.py
│   │   │   │   ├── m5_ema/
│   │   │   │   │   ├── README.md # Project documentation
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── adapter.py
│   │   │   │   │   ├── aggregator.py
│   │   │   │   │   ├── plugin.py
│   │   │   │   │   ├── schema.sql
│   │   │   │   │   ├── storage.py
│   │   │   │   ├── m5_market_structure/
│   │   │   │   │   ├── README.md # Project documentation
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── adapter.py
│   │   │   │   │   ├── aggregator.py
│   │   │   │   │   ├── plugin.py
│   │   │   │   │   ├── schema.sql
│   │   │   │   │   ├── storage.py
│   │   │   │   ├── _template_/
│   │   │   │   │   ├── README.md # Project documentation
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── adapter.py
│   │   │   │   │   ├── aggregator.py
│   │   │   │   │   ├── plugin.py
│   │   │   │   │   ├── schema.sql
│   │   │   │   │   ├── storage.py
│   │   │   ├── results/
│   │   │   │   ├── __init__.py
│   │   │   ├── storage/
│   │   │   │   ├── supabase_storage.py
│   │   │   ├── ui/
│   │   │   │   ├── __init__.py
│   │   │   ├── _documentation_/
│   │   │   │   ├── bt_modularization_plan.md
│   │   │   │   ├── data_architecture_bt.md
│   │   ├── plugins_old/
│   │   │   ├── cum_delta/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── plugin.py
│   │   │   │   ├── test.py
│   ├── backtest/
│   │   ├── run.py
│   │   ├── backtest_system/
│   │   │   ├── __init__.py
│   │   │   ├── dashboard.py
│   │   │   ├── main.py # Main application entry point
│   │   │   ├── cache/
│   │   │   │   ├── polygon_data/
│   │   │   │   │   ├── cache_metadata.json
│   │   │   │   │   ├── aapl/
│   │   │   │   │   │   ├── bars/
│   │   │   │   │   │   │   ├── 15min_20250113_0d46ca7e.parquet
│   │   │   │   │   │   │   ├── 15min_20250702_22687f70.parquet
│   │   │   │   │   │   │   ├── 1min_20250115_4c308ac1.parquet
│   │   │   │   │   │   │   ├── 1min_20250115_c4cfddc4.parquet
│   │   │   │   │   │   │   ├── 1min_20250115_eecdf85e.parquet
│   │   │   │   │   │   │   ├── 1min_20250630_ffb4b3ca.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_c7225873.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_e0d7632b.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_064d777f.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_9008f29e.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_964bdfed.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_e6a9df73.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_fc11b256.parquet
│   │   │   │   │   │   │   ├── 5min_20241230_64d6b05b.parquet
│   │   │   │   │   │   │   ├── 5min_20250115_d8b6669b.parquet
│   │   │   │   │   │   │   ├── 5min_20250702_fe5dac33.parquet
│   │   │   │   │   │   ├── quotes/
│   │   │   │   │   │   │   ├── tick_20250115_f6c636d5.parquet
│   │   │   │   │   │   │   ├── tick_20250627_04dd0c13.parquet
│   │   │   │   │   │   │   ├── tick_20250627_26a9c199.parquet
│   │   │   │   │   │   │   ├── tick_20250627_5747c447.parquet
│   │   │   │   │   │   ├── trades/
│   │   │   │   │   │   │   ├── tick_20250115_9462178d.parquet
│   │   │   │   │   │   │   ├── tick_20250627_0e86bb33.parquet
│   │   │   │   │   │   │   ├── tick_20250627_4c6ae0a8.parquet
│   │   │   │   │   │   │   ├── tick_20250627_9724c28f.parquet
│   │   │   │   │   │   │   ├── tick_20250702_2fdb544d.parquet
│   │   │   │   │   │   │   ├── tick_20250702_4aa1f30d.parquet
│   │   │   │   │   │   │   ├── tick_20250702_d1e75879.parquet
│   │   │   │   │   │   │   ├── tick_20250702_e56f3595.parquet
│   │   │   │   │   │   │   ├── tick_20250702_eba382c9.parquet
│   │   │   │   │   ├── crcl/
│   │   │   │   │   │   ├── bars/
│   │   │   │   │   │   │   ├── 15min_20250702_a14667bd.parquet
│   │   │   │   │   │   │   ├── 1min_20250627_cd78b3e5.parquet
│   │   │   │   │   │   │   ├── 1min_20250630_54b9fbcc.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_36e88cbf.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_f0b2d88f.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_0e7aa2df.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_30417220.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_87786dae.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_d652a297.parquet
│   │   │   │   │   │   │   ├── 5min_20250702_d93abafd.parquet
│   │   │   │   │   │   ├── quotes/
│   │   │   │   │   │   │   ├── tick_20250702_6244b843.parquet
│   │   │   │   │   │   │   ├── tick_20250702_8128163c.parquet
│   │   │   │   │   │   │   ├── tick_20250702_eff35cf2.parquet
│   │   │   │   │   │   ├── trades/
│   │   │   │   │   │   │   ├── tick_20250702_240d29f8.parquet
│   │   │   │   │   │   │   ├── tick_20250702_8d17e274.parquet
│   │   │   │   │   │   │   ├── tick_20250702_a123ab38.parquet
│   │   │   │   │   │   │   ├── tick_20250702_da83bfb5.parquet
│   │   │   │   │   ├── orcl/
│   │   │   │   │   │   ├── bars/
│   │   │   │   │   │   │   ├── 15min_20250702_1d97fbc1.parquet
│   │   │   │   │   │   │   ├── 15min_20250702_741f7a69.parquet
│   │   │   │   │   │   │   ├── 15min_20250702_954fedb4.parquet
│   │   │   │   │   │   │   ├── 15min_20250702_9933eaaa.parquet
│   │   │   │   │   │   │   ├── 15min_20250702_b9382e5d.parquet
│   │   │   │   │   │   │   ├── 15min_20250702_bdd79b7b.parquet
│   │   │   │   │   │   │   ├── 1min_20250630_031aa9f5.parquet
│   │   │   │   │   │   │   ├── 1min_20250630_26a18b98.parquet
│   │   │   │   │   │   │   ├── 1min_20250630_3e8f596e.parquet
│   │   │   │   │   │   │   ├── 1min_20250630_7031b4ce.parquet
│   │   │   │   │   │   │   ├── 1min_20250630_c3b99ec6.parquet
│   │   │   │   │   │   │   ├── 1min_20250630_ee4dbe53.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_03c38c55.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_25274a79.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_37a23bc1.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_40c0d70f.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_9b321ba7.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_a4476541.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_b001c8e9.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_b6284a75.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_c2eddfd9.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_d3f605e1.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_dfa51291.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_f079669c.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_0064bc7d.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_044769c1.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_0721104d.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_2703792a.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_2a816d3a.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_324c3063.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_390ac74b.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_45ae9199.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_483cdfe0.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_4cda3a45.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_566bc9af.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_5905db08.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_5afd0ff2.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_61853831.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_647b6f1f.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_8be1fb20.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_8d5161a4.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_b76553c5.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_c8587ccd.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_cb198014.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_d2021c9f.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_de7bc7c6.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_ec0062c3.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_f1e99d37.parquet
│   │   │   │   │   │   │   ├── 1min_20250702_fc1306b2.parquet
│   │   │   │   │   │   │   ├── 5min_20250702_04d20d78.parquet
│   │   │   │   │   │   │   ├── 5min_20250702_374ae6d7.parquet
│   │   │   │   │   │   │   ├── 5min_20250702_47fc63cb.parquet
│   │   │   │   │   │   │   ├── 5min_20250702_6dd2ac80.parquet
│   │   │   │   │   │   │   ├── 5min_20250702_b90e409a.parquet
│   │   │   │   │   │   │   ├── 5min_20250702_e6fa6bb6.parquet
│   │   │   │   │   │   ├── quotes/
│   │   │   │   │   │   │   ├── tick_20250702_25f963a9.parquet
│   │   │   │   │   │   │   ├── tick_20250702_3e4ad719.parquet
│   │   │   │   │   │   │   ├── tick_20250702_444482a5.parquet
│   │   │   │   │   │   │   ├── tick_20250702_63c774a0.parquet
│   │   │   │   │   │   │   ├── tick_20250702_68adc565.parquet
│   │   │   │   │   │   │   ├── tick_20250702_77b6d723.parquet
│   │   │   │   │   │   │   ├── tick_20250702_82bb38e0.parquet
│   │   │   │   │   │   │   ├── tick_20250702_a0aaecb0.parquet
│   │   │   │   │   │   │   ├── tick_20250702_a3cac224.parquet
│   │   │   │   │   │   │   ├── tick_20250702_aef0ee12.parquet
│   │   │   │   │   │   │   ├── tick_20250702_d9423242.parquet
│   │   │   │   │   │   │   ├── tick_20250702_e2106857.parquet
│   │   │   │   │   │   │   ├── tick_20250702_e9b8425c.parquet
│   │   │   │   │   │   ├── trades/
│   │   │   │   │   │   │   ├── tick_20250702_25652d02.parquet
│   │   │   │   │   │   │   ├── tick_20250702_330bfab3.parquet
│   │   │   │   │   │   │   ├── tick_20250702_39681cef.parquet
│   │   │   │   │   │   │   ├── tick_20250702_3dca5d92.parquet
│   │   │   │   │   │   │   ├── tick_20250702_46ad9ef7.parquet
│   │   │   │   │   │   │   ├── tick_20250702_5de8f089.parquet
│   │   │   │   │   │   │   ├── tick_20250702_6f3f0485.parquet
│   │   │   │   │   │   │   ├── tick_20250702_769861e6.parquet
│   │   │   │   │   │   │   ├── tick_20250702_7f807fb2.parquet
│   │   │   │   │   │   │   ├── tick_20250702_80695184.parquet
│   │   │   │   │   │   │   ├── tick_20250702_81ce2ebe.parquet
│   │   │   │   │   │   │   ├── tick_20250702_8cc402ab.parquet
│   │   │   │   │   │   │   ├── tick_20250702_afc2f9d2.parquet
│   │   │   │   │   │   │   ├── tick_20250702_dfa9bf52.parquet
│   │   │   │   │   │   │   ├── tick_20250702_e8f53ca0.parquet
│   │   │   │   │   │   │   ├── tick_20250702_e9732754.parquet
│   │   │   │   │   │   │   ├── tick_20250702_f5d65ee3.parquet
│   │   │   │   │   │   │   ├── tick_20250702_fc072c0a.parquet
│   │   │   │   │   ├── tsla/
│   │   │   │   │   │   ├── bars/
│   │   │   │   │   │   │   ├── 15min_20250627_32594983.parquet
│   │   │   │   │   │   │   ├── 15min_20250627_9d3243a0.parquet
│   │   │   │   │   │   │   ├── 15min_20250627_a66bdc4a.parquet
│   │   │   │   │   │   │   ├── 15min_20250701_07fc408e.parquet
│   │   │   │   │   │   │   ├── 15min_20250701_a548c372.parquet
│   │   │   │   │   │   │   ├── 1min_20250625_334a74d0.parquet
│   │   │   │   │   │   │   ├── 1min_20250625_fb04b785.parquet
│   │   │   │   │   │   │   ├── 1min_20250626_42424fe8.parquet
│   │   │   │   │   │   │   ├── 1min_20250626_436b82a1.parquet
│   │   │   │   │   │   │   ├── 1min_20250626_719cf4bd.parquet
│   │   │   │   │   │   │   ├── 1min_20250626_a1eace4b.parquet
│   │   │   │   │   │   │   ├── 1min_20250626_c82c2eec.parquet
│   │   │   │   │   │   │   ├── 1min_20250626_d376879d.parquet
│   │   │   │   │   │   │   ├── 1min_20250626_ff8ba32a.parquet
│   │   │   │   │   │   │   ├── 1min_20250627_08841cd8.parquet
│   │   │   │   │   │   │   ├── 1min_20250627_320b8368.parquet
│   │   │   │   │   │   │   ├── 1min_20250627_7b2d9a32.parquet
│   │   │   │   │   │   │   ├── 1min_20250627_85f3bcd6.parquet
│   │   │   │   │   │   │   ├── 1min_20250627_8cda84c0.parquet
│   │   │   │   │   │   │   ├── 1min_20250627_93183bf7.parquet
│   │   │   │   │   │   │   ├── 1min_20250627_d1edfcd0.parquet
│   │   │   │   │   │   │   ├── 1min_20250627_de3efb46.parquet
│   │   │   │   │   │   │   ├── 1min_20250627_e572916c.parquet
│   │   │   │   │   │   │   ├── 1min_20250627_e5e82c33.parquet
│   │   │   │   │   │   │   ├── 1min_20250627_ef54901b.parquet
│   │   │   │   │   │   │   ├── 1min_20250627_f2b492f8.parquet
│   │   │   │   │   │   │   ├── 1min_20250629_4e6e6a3e.parquet
│   │   │   │   │   │   │   ├── 1min_20250629_c487e296.parquet
│   │   │   │   │   │   │   ├── 1min_20250630_1b14edc4.parquet
│   │   │   │   │   │   │   ├── 1min_20250630_22e5feec.parquet
│   │   │   │   │   │   │   ├── 1min_20250630_3149965a.parquet
│   │   │   │   │   │   │   ├── 1min_20250630_c2d70639.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_00048804.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_09002e03.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_4cea7c11.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_5ef98e51.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_759ac628.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_a18382fc.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_c3596348.parquet
│   │   │   │   │   │   │   ├── 1min_20250701_e57e1b25.parquet
│   │   │   │   │   │   │   ├── 5min_20250627_4b812760.parquet
│   │   │   │   │   │   │   ├── 5min_20250627_a3615e69.parquet
│   │   │   │   │   │   │   ├── 5min_20250627_f2ebac4c.parquet
│   │   │   │   │   │   │   ├── 5min_20250701_2f2a2872.parquet
│   │   │   │   │   │   │   ├── 5min_20250701_39a290a1.parquet
│   │   │   │   │   │   ├── quotes/
│   │   │   │   │   │   │   ├── tick_20250627_70f77199.parquet
│   │   │   │   │   │   │   ├── tick_20250627_76da4db9.parquet
│   │   │   │   │   │   │   ├── tick_20250627_856ae680.parquet
│   │   │   │   │   │   │   ├── tick_20250627_99b63285.parquet
│   │   │   │   │   │   │   ├── tick_20250627_ba8836e9.parquet
│   │   │   │   │   │   │   ├── tick_20250701_5ec67fc2.parquet
│   │   │   │   │   │   ├── trades/
│   │   │   │   │   │   │   ├── tick_20250627_0b588a2e.parquet
│   │   │   │   │   │   │   ├── tick_20250627_17240d2c.parquet
│   │   │   │   │   │   │   ├── tick_20250627_1c2a656a.parquet
│   │   │   │   │   │   │   ├── tick_20250627_27ea7872.parquet
│   │   │   │   │   │   │   ├── tick_20250627_9086aaf0.parquet
│   │   │   │   │   │   │   ├── tick_20250701_3250a2d8.parquet
│   │   │   │   │   │   │   ├── tick_20250701_e7881148.parquet
│   │   │   │   │   │   │   ├── tick_20250702_85bb52fc.parquet
│   │   │   ├── components/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── multi_result_viewer.py
│   │   │   │   ├── plugin_runner.py
│   │   │   │   ├── result_viewer.py
│   │   ├── cache/
│   │   │   ├── polygon_data/
│   │   │   │   ├── cache_metadata.json
│   │   │   │   ├── aapl/
│   │   │   │   │   ├── bars/
│   │   │   │   │   │   ├── 1min_20240627_092140ab.parquet
│   │   │   │   │   │   ├── 1min_20240627_9eb3abfd.parquet
│   │   │   │   │   │   ├── 1min_20240627_bc77bfe3.parquet
│   │   │   │   │   │   ├── 1min_20250101_a3c05929.parquet
│   │   │   │   │   │   ├── 1min_20250115_9f2f4baf.parquet
│   │   │   │   │   │   ├── 1min_20250619_273c4fd2.parquet
│   │   │   │   │   │   ├── 1min_20250625_ac26028f.parquet
│   │   │   │   │   ├── quotes/
│   │   │   │   │   │   ├── tick_20240627_785ea362.parquet
│   │   │   │   │   │   ├── tick_20240627_b49a07d7.parquet
│   │   │   │   ├── crcl/
│   │   │   │   │   ├── bars/
│   │   │   │   │   │   ├── 15min_20250625_d768cf35.parquet
│   │   │   │   │   │   ├── 15min_20250627_2956c74f.parquet
│   │   │   │   │   │   ├── 15min_20250627_5164d1b3.parquet
│   │   │   │   │   │   ├── 15min_20250627_ab098e5e.parquet
│   │   │   │   │   │   ├── 15min_20250627_d2c248a7.parquet
│   │   │   │   │   │   ├── 1min_20250621_d0ef1473.parquet
│   │   │   │   │   │   ├── 1min_20250623_11c8fc69.parquet
│   │   │   │   │   │   ├── 1min_20250623_1418d1e1.parquet
│   │   │   │   │   │   ├── 1min_20250623_1732bddd.parquet
│   │   │   │   │   │   ├── 1min_20250623_22a72286.parquet
│   │   │   │   │   │   ├── 1min_20250623_6a2eae87.parquet
│   │   │   │   │   │   ├── 1min_20250623_70fcea41.parquet
│   │   │   │   │   │   ├── 1min_20250623_71108066.parquet
│   │   │   │   │   │   ├── 1min_20250623_934f1b46.parquet
│   │   │   │   │   │   ├── 1min_20250623_f1432149.parquet
│   │   │   │   │   │   ├── 1min_20250623_fd2e84fd.parquet
│   │   │   │   │   │   ├── 1min_20250624_199f0728.parquet
│   │   │   │   │   │   ├── 1min_20250624_1f8c1a5e.parquet
│   │   │   │   │   │   ├── 1min_20250624_30922828.parquet
│   │   │   │   │   │   ├── 1min_20250624_3ce615ef.parquet
│   │   │   │   │   │   ├── 1min_20250624_44dfee29.parquet
│   │   │   │   │   │   ├── 1min_20250624_6e7f595a.parquet
│   │   │   │   │   │   ├── 1min_20250624_bd365d35.parquet
│   │   │   │   │   │   ├── 1min_20250624_c55a38a7.parquet
│   │   │   │   │   │   ├── 1min_20250625_06f2a666.parquet
│   │   │   │   │   │   ├── 1min_20250625_10119438.parquet
│   │   │   │   │   │   ├── 1min_20250625_27351a44.parquet
│   │   │   │   │   │   ├── 1min_20250625_35a84801.parquet
│   │   │   │   │   │   ├── 1min_20250625_432a38d7.parquet
│   │   │   │   │   │   ├── 1min_20250625_44f1ca54.parquet
│   │   │   │   │   │   ├── 1min_20250625_4be3906f.parquet
│   │   │   │   │   │   ├── 1min_20250625_4ebd5bf8.parquet
│   │   │   │   │   │   ├── 1min_20250625_5dd69116.parquet
│   │   │   │   │   │   ├── 1min_20250625_6ba14652.parquet
│   │   │   │   │   │   ├── 1min_20250625_7141f1b3.parquet
│   │   │   │   │   │   ├── 1min_20250625_72e442b1.parquet
│   │   │   │   │   │   ├── 1min_20250625_79ae2544.parquet
│   │   │   │   │   │   ├── 1min_20250625_7ae8e567.parquet
│   │   │   │   │   │   ├── 1min_20250625_7c32d730.parquet
│   │   │   │   │   │   ├── 1min_20250625_7f649ea5.parquet
│   │   │   │   │   │   ├── 1min_20250625_8f9f7725.parquet
│   │   │   │   │   │   ├── 1min_20250625_a0c8ffef.parquet
│   │   │   │   │   │   ├── 1min_20250625_a47fabfc.parquet
│   │   │   │   │   │   ├── 1min_20250625_a8d10579.parquet
│   │   │   │   │   │   ├── 1min_20250625_aca1db60.parquet
│   │   │   │   │   │   ├── 1min_20250625_b1df03a9.parquet
│   │   │   │   │   │   ├── 1min_20250625_cd3c5056.parquet
│   │   │   │   │   │   ├── 1min_20250625_cfd122c8.parquet
│   │   │   │   │   │   ├── 1min_20250625_cff1b59e.parquet
│   │   │   │   │   │   ├── 1min_20250625_e7df26a6.parquet
│   │   │   │   │   │   ├── 1min_20250625_f931562e.parquet
│   │   │   │   │   │   ├── 1min_20250625_fb556de3.parquet
│   │   │   │   │   │   ├── 1min_20250626_01607916.parquet
│   │   │   │   │   │   ├── 1min_20250626_034007e6.parquet
│   │   │   │   │   │   ├── 1min_20250626_0a44045e.parquet
│   │   │   │   │   │   ├── 1min_20250626_2551d764.parquet
│   │   │   │   │   │   ├── 1min_20250626_2c7f59b8.parquet
│   │   │   │   │   │   ├── 1min_20250626_2ebe068d.parquet
│   │   │   │   │   │   ├── 1min_20250626_3872a43e.parquet
│   │   │   │   │   │   ├── 1min_20250626_3a0ca4f0.parquet
│   │   │   │   │   │   ├── 1min_20250626_3b370256.parquet
│   │   │   │   │   │   ├── 1min_20250626_50eb23bc.parquet
│   │   │   │   │   │   ├── 1min_20250626_7ca7c978.parquet
│   │   │   │   │   │   ├── 1min_20250626_91b70f05.parquet
│   │   │   │   │   │   ├── 1min_20250626_acdd62ef.parquet
│   │   │   │   │   │   ├── 1min_20250626_c476f944.parquet
│   │   │   │   │   │   ├── 1min_20250626_c80798a1.parquet
│   │   │   │   │   │   ├── 1min_20250626_d2f6e4db.parquet
│   │   │   │   │   │   ├── 1min_20250626_e1deeac9.parquet
│   │   │   │   │   │   ├── 1min_20250626_e6169b58.parquet
│   │   │   │   │   │   ├── 1min_20250627_1bce2976.parquet
│   │   │   │   │   │   ├── 1min_20250627_247c03fb.parquet
│   │   │   │   │   │   ├── 1min_20250627_3ca67960.parquet
│   │   │   │   │   │   ├── 1min_20250627_3f8eaea1.parquet
│   │   │   │   │   │   ├── 1min_20250627_4aaf8ab7.parquet
│   │   │   │   │   │   ├── 1min_20250627_554f3254.parquet
│   │   │   │   │   │   ├── 1min_20250627_7a38b076.parquet
│   │   │   │   │   │   ├── 1min_20250627_7c60cc91.parquet
│   │   │   │   │   │   ├── 1min_20250627_93b4f2ac.parquet
│   │   │   │   │   │   ├── 1min_20250627_94b71a42.parquet
│   │   │   │   │   │   ├── 1min_20250627_b9c9060c.parquet
│   │   │   │   │   │   ├── 1min_20250627_c61b6d20.parquet
│   │   │   │   │   │   ├── 1min_20250627_c6d210cc.parquet
│   │   │   │   │   │   ├── 1min_20250627_c9ad9f40.parquet
│   │   │   │   │   │   ├── 1min_20250627_cd78b3e5.parquet
│   │   │   │   │   │   ├── 1min_20250627_faa77c94.parquet
│   │   │   │   │   │   ├── 5min_20250620_31d10b16.parquet
│   │   │   │   │   │   ├── 5min_20250620_59c556f5.parquet
│   │   │   │   │   │   ├── 5min_20250620_966ed9ef.parquet
│   │   │   │   │   │   ├── 5min_20250620_dccbb40a.parquet
│   │   │   │   │   │   ├── 5min_20250620_e183e9a2.parquet
│   │   │   │   │   │   ├── 5min_20250626_efa66e6c.parquet
│   │   │   │   │   │   ├── 5min_20250627_0964444d.parquet
│   │   │   │   │   ├── quotes/
│   │   │   │   │   │   ├── tick_20250627_02c6ad0a.parquet
│   │   │   │   │   │   ├── tick_20250627_110e955b.parquet
│   │   │   │   │   │   ├── tick_20250627_1c28cb94.parquet
│   │   │   │   │   │   ├── tick_20250627_23ce3212.parquet
│   │   │   │   │   │   ├── tick_20250627_29d2da71.parquet
│   │   │   │   │   │   ├── tick_20250627_2d0be922.parquet
│   │   │   │   │   │   ├── tick_20250627_2f5ed184.parquet
│   │   │   │   │   │   ├── tick_20250627_4c5af8c7.parquet
│   │   │   │   │   │   ├── tick_20250627_689814fe.parquet
│   │   │   │   │   │   ├── tick_20250627_7b9bc489.parquet
│   │   │   │   │   │   ├── tick_20250627_876cd44c.parquet
│   │   │   │   │   │   ├── tick_20250627_920819cd.parquet
│   │   │   │   │   │   ├── tick_20250627_9b38551f.parquet
│   │   │   │   │   │   ├── tick_20250627_b69f501f.parquet
│   │   │   │   │   │   ├── tick_20250627_c3ff4fbf.parquet
│   │   │   │   │   │   ├── tick_20250627_cd91090f.parquet
│   │   │   │   │   │   ├── tick_20250627_cf160e51.parquet
│   │   │   │   │   │   ├── tick_20250627_eff5fdf1.parquet
│   │   │   │   │   │   ├── tick_20250627_f0e0dd3b.parquet
│   │   │   │   │   ├── trades/
│   │   │   │   │   │   ├── tick_20250627_0a17a303.parquet
│   │   │   │   │   │   ├── tick_20250627_1b0b24d1.parquet
│   │   │   │   │   │   ├── tick_20250627_27a61747.parquet
│   │   │   │   │   │   ├── tick_20250627_298d87d8.parquet
│   │   │   │   │   │   ├── tick_20250627_2d45dc20.parquet
│   │   │   │   │   │   ├── tick_20250627_338f65ce.parquet
│   │   │   │   │   │   ├── tick_20250627_62ccb5e5.parquet
│   │   │   │   │   │   ├── tick_20250627_6a8d916c.parquet
│   │   │   │   │   │   ├── tick_20250627_6dc73883.parquet
│   │   │   │   │   │   ├── tick_20250627_6f35ceba.parquet
│   │   │   │   │   │   ├── tick_20250627_7bf1c088.parquet
│   │   │   │   │   │   ├── tick_20250627_84191cb5.parquet
│   │   │   │   │   │   ├── tick_20250627_87c21b7e.parquet
│   │   │   │   │   │   ├── tick_20250627_9661efd3.parquet
│   │   │   │   │   │   ├── tick_20250627_eacb2f94.parquet
│   │   │   │   ├── tsla/
│   │   │   │   │   ├── bars/
│   │   │   │   │   │   ├── 1min_20250627_311941fe.parquet
│   │   │   │   │   │   ├── 1min_20250627_39e98b5e.parquet
│   │   │   │   │   │   ├── 1min_20250627_7919a3bf.parquet
│   │   │   │   │   │   ├── 1min_20250627_e3c6d7ed.parquet
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── circuit_breaker.py
│   │   │   ├── data_coordinator.py
│   │   │   ├── data_validator.py
│   │   │   ├── protected_data_manager.py
│   │   │   ├── request_aggregator.py
│   │   │   ├── trade_quote_aligner.py
│   │   │   ├── unified_data_system.py
│   │   │   ├── debug/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── cli_tester.py
│   │   │   │   ├── test_bar_limits.py
│   │   │   │   ├── test_circuit_breaker.py
│   │   │   │   ├── test_data_validator.py
│   │   │   │   ├── test_data_volumes.py
│   │   │   │   ├── test_extended_hours.py
│   │   │   │   ├── test_integration.py
│   │   │   │   ├── test_request_aggregator.py
│   │   │   │   ├── test_trade_quote_aligner.py
│   │   │   │   ├── test_utils.py
│   │   │   ├── polygon_data_manager/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── api_client.py
│   │   │   │   ├── cache_manager.py
│   │   │   │   ├── data_manager.py
│   │   │   │   ├── file_cache.py
│   │   │   │   ├── memory_cache.py
│   │   │   │   ├── models.py
│   │   │   │   ├── request_tracker.py
│   │   │   ├── temp/
│   │   ├── documentation/
│   │   │   ├── modular_structure_bt.md
│   │   ├── plugins/
│   │   │   ├── base_plugin.py
│   │   │   ├── bid_ask_imbalance/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── investigate_classification.py
│   │   │   │   ├── test.py
│   │   │   ├── buy_sell_ratio/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── chart.py
│   │   │   │   ├── plugin.py
│   │   │   │   ├── test.py
│   │   │   ├── impact_success/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── chart.py
│   │   │   │   ├── plugin.py
│   │   │   │   ├── test.py
│   │   │   ├── large_orders/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── grid.py
│   │   │   │   ├── large_orders_grid_CRCL_20250706_154823.json
│   │   │   │   ├── test.py
│   │   │   ├── m15_ema/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test.py
│   │   │   ├── m15_market_structure/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── plugin.py
│   │   │   │   ├── test.py
│   │   │   ├── m15_statistical_trend/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test.py
│   │   │   ├── m1_bid_ask_analysis/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test.py
│   │   │   ├── m1_ema/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test.py
│   │   │   ├── m1_market_structure/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test.py
│   │   │   ├── m1_statistical_trend/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test.py
│   │   │   ├── m5_ema/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test.py
│   │   │   ├── m5_market_structure/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test.py
│   │   │   ├── m5_statistical_trend/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test.py
│   │   │   ├── net_large_volume/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── plugin.py
│   │   │   │   ├── test.py
│   │   │   ├── order_blocks/
│   │   │   ├── tick_flow/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test.py
│   │   ├── temp/
│   │   │   ├── polygon_data_report_20250701_203701.json
│   │   │   ├── polygon_data_report_20250701_203701_summary.txt
│   │   │   ├── polygon_data_report_20250701_230215.json
│   │   │   ├── polygon_data_report_20250701_230215_summary.txt
│   │   │   ├── polygon_data_report_20250702_120415.json
│   │   │   ├── polygon_data_report_20250702_120415_summary.txt
│   │   │   ├── polygon_data_report_20250702_131318.json
│   │   │   ├── polygon_data_report_20250702_131318_summary.txt
│   │   │   ├── polygon_data_report_20250702_164708.json
│   │   │   ├── polygon_data_report_20250702_164708_summary.txt
│   │   │   ├── polygon_data_report_20250702_164856.json
│   │   │   ├── polygon_data_report_20250702_164856_summary.txt
│   │   │   ├── polygon_data_report_20250702_164929.json
│   │   │   ├── polygon_data_report_20250702_164929_summary.txt
│   │   │   ├── polygon_data_report_20250702_165325.json
│   │   │   ├── polygon_data_report_20250702_165325_summary.txt
│   │   │   ├── polygon_data_report_20250702_165714.json
│   │   │   ├── polygon_data_report_20250702_165714_summary.txt
│   │   │   ├── polygon_data_report_20250703_140912.json
│   │   │   ├── polygon_data_report_20250703_140912_summary.txt
│   │   │   ├── polygon_data_report_20250703_155546.json
│   │   │   ├── polygon_data_report_20250703_155546_summary.txt
│   │   │   ├── polygon_data_report_20250703_171411.json
│   │   │   ├── polygon_data_report_20250703_171411_summary.txt
│   │   │   ├── polygon_data_report_20250703_172740.json
│   │   │   ├── polygon_data_report_20250703_172740_summary.txt
│   │   │   ├── polygon_data_report_20250703_173333.json
│   │   │   ├── polygon_data_report_20250703_173333_summary.txt
│   │   │   ├── polygon_data_report_20250703_194324.json
│   │   │   ├── polygon_data_report_20250703_194324_summary.txt
│   │   │   ├── polygon_data_report_20250703_194338.json
│   │   │   ├── polygon_data_report_20250703_194338_summary.txt
│   │   │   ├── polygon_data_report_20250703_194404.json
│   │   │   ├── polygon_data_report_20250703_194404_summary.txt
│   │   │   ├── polygon_data_report_20250703_194725.json
│   │   │   ├── polygon_data_report_20250703_194725_summary.txt
│   │   │   ├── polygon_data_report_20250703_201903.json
│   │   │   ├── polygon_data_report_20250703_201903_summary.txt
│   │   │   ├── polygon_data_report_20250703_214439.json
│   │   │   ├── polygon_data_report_20250703_214439_summary.txt
│   │   │   ├── polygon_data_report_20250703_232017.json
│   │   │   ├── polygon_data_report_20250703_232017_summary.txt
│   │   │   ├── polygon_data_report_20250703_232616.json
│   │   │   ├── polygon_data_report_20250703_232616_summary.txt
│   │   │   ├── polygon_data_report_20250703_232708.json
│   │   │   ├── polygon_data_report_20250703_232708_summary.txt
│   │   │   ├── polygon_data_report_20250704_125402.json
│   │   │   ├── polygon_data_report_20250704_125402_summary.txt
│   │   │   ├── polygon_data_report_20250704_210339.json
│   │   │   ├── polygon_data_report_20250704_210339_summary.txt
│   │   │   ├── polygon_data_report_20250706_150712.json
│   │   │   ├── polygon_data_report_20250706_150712_summary.txt
│   │   │   ├── polygon_data_report_20250706_151425.json
│   │   │   ├── polygon_data_report_20250706_151425_summary.txt
│   │   │   ├── polygon_data_report_20250706_152841.json
│   │   │   ├── polygon_data_report_20250706_152841_summary.txt
│   │   │   ├── polygon_data_report_20250706_154823.json
│   │   │   ├── polygon_data_report_20250706_154823_summary.txt
│   ├── cache/
│   │   ├── cache_metadata.json
│   │   ├── aapl/
│   │   │   ├── bars/
│   │   │   │   ├── 1min_20250115_9f2f4baf.parquet
│   ├── journal/
│   │   ├── __init__.py
│   │   ├── trades/
│   │   │   ├── __init__.py
│   │   │   ├── database.py
│   │   │   ├── models.py
│   │   │   ├── parser.py
│   │   │   ├── plugin.py
│   │   │   ├── processor.py
│   │   │   ├── test.py
│   │   │   ├── test_simple.py
│   ├── modules/
│   │   ├── calculations/
│   │   │   ├── indicators/
│   │   │   │   ├── m15_ema.py
│   │   │   │   ├── m1_ema.py
│   │   │   │   ├── m5_ema.py
│   │   │   ├── market_structure/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── m15_market_structure.py
│   │   │   │   ├── m1_market_structure.py
│   │   │   │   ├── m5_market_structure.py
│   │   │   ├── order_flow/
│   │   │   │   ├── bid_ask_imbal.py
│   │   │   │   ├── buy_sell_ratio.py
│   │   │   │   ├── cum_delta.py
│   │   │   │   ├── impact_success.py
│   │   │   │   ├── large_orders.py
│   │   │   │   ├── micro_momentum.py
│   │   │   │   ├── net_large_volume.py
│   │   │   │   ├── trade_size_distro.py
│   │   │   ├── trend/
│   │   │   │   ├── statistical_trend_15min.py
│   │   │   │   ├── statistical_trend_1min.py
│   │   │   │   ├── statistical_trend_5min.py
│   │   │   ├── volume/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── cluster_analyzer.py
│   │   │   │   ├── hvn_engine.py
│   │   │   │   ├── m1_bid_ask_analysis.py
│   │   │   │   ├── market_context.py
│   │   │   │   ├── ranking_engine.py
│   │   │   │   ├── session_profile.py
│   │   │   │   ├── tick_flow.py
│   │   │   │   ├── volume_analysis_1min.py
│   │   │   │   ├── volume_profile.py
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
│   │   │   │   │   ├── market_filter_scan_20250630_115438.md
│   │   │   │   │   ├── market_filter_scan_20250630_131902.md
│   │   │   │   │   ├── market_filter_scan_20250630_134040.md
│   │   │   │   │   ├── market_filter_scan_20250701_122612.md
│   │   │   │   │   ├── market_filter_scan_20250702_123649.md
│   │   │   │   │   ├── market_filter_scan_20250703_132010.md
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
│   │   │   │   │   │   ├── A_1min.parquet
│   │   │   │   │   ├── AAL/
│   │   │   │   │   │   ├── AAL_1d.parquet
│   │   │   │   │   │   ├── AAL_1min.parquet
│   │   │   │   │   ├── AAP/
│   │   │   │   │   │   ├── AAP_1d.parquet
│   │   │   │   │   │   ├── AAP_1min.parquet
│   │   │   │   │   ├── AAPL/
│   │   │   │   │   │   ├── AAPL_15min.parquet
│   │   │   │   │   │   ├── AAPL_1d.parquet
│   │   │   │   │   │   ├── AAPL_1day.parquet
│   │   │   │   │   │   ├── AAPL_1hour.parquet
│   │   │   │   │   │   ├── AAPL_1min.parquet
│   │   │   │   │   │   ├── AAPL_5min.parquet
│   │   │   │   │   ├── ABBV/
│   │   │   │   │   │   ├── ABBV_1d.parquet
│   │   │   │   │   │   ├── ABBV_1min.parquet
│   │   │   │   │   ├── ABT/
│   │   │   │   │   │   ├── ABT_1d.parquet
│   │   │   │   │   │   ├── ABT_1min.parquet
│   │   │   │   │   ├── ACGL/
│   │   │   │   │   │   ├── ACGL_1d.parquet
│   │   │   │   │   │   ├── ACGL_1min.parquet
│   │   │   │   │   ├── ACN/
│   │   │   │   │   │   ├── ACN_1d.parquet
│   │   │   │   │   │   ├── ACN_1min.parquet
│   │   │   │   │   ├── ADBE/
│   │   │   │   │   │   ├── ADBE_1d.parquet
│   │   │   │   │   │   ├── ADBE_1min.parquet
│   │   │   │   │   ├── ADI/
│   │   │   │   │   │   ├── ADI_1d.parquet
│   │   │   │   │   │   ├── ADI_1min.parquet
│   │   │   │   │   ├── ADM/
│   │   │   │   │   │   ├── ADM_1d.parquet
│   │   │   │   │   │   ├── ADM_1min.parquet
│   │   │   │   │   ├── ADP/
│   │   │   │   │   │   ├── ADP_1d.parquet
│   │   │   │   │   │   ├── ADP_1min.parquet
│   │   │   │   │   ├── ADSK/
│   │   │   │   │   │   ├── ADSK_1d.parquet
│   │   │   │   │   │   ├── ADSK_1min.parquet
│   │   │   │   │   ├── AEE/
│   │   │   │   │   │   ├── AEE_1d.parquet
│   │   │   │   │   │   ├── AEE_1min.parquet
│   │   │   │   │   ├── AEP/
│   │   │   │   │   │   ├── AEP_1d.parquet
│   │   │   │   │   │   ├── AEP_1min.parquet
│   │   │   │   │   ├── AES/
│   │   │   │   │   │   ├── AES_1d.parquet
│   │   │   │   │   │   ├── AES_1min.parquet
│   │   │   │   │   ├── AFL/
│   │   │   │   │   │   ├── AFL_1d.parquet
│   │   │   │   │   │   ├── AFL_1min.parquet
│   │   │   │   │   ├── AIG/
│   │   │   │   │   │   ├── AIG_1d.parquet
│   │   │   │   │   │   ├── AIG_1min.parquet
│   │   │   │   │   ├── AIZ/
│   │   │   │   │   │   ├── AIZ_1d.parquet
│   │   │   │   │   │   ├── AIZ_1min.parquet
│   │   │   │   │   ├── AJG/
│   │   │   │   │   │   ├── AJG_1d.parquet
│   │   │   │   │   │   ├── AJG_1min.parquet
│   │   │   │   │   ├── AKAM/
│   │   │   │   │   │   ├── AKAM_1d.parquet
│   │   │   │   │   │   ├── AKAM_1min.parquet
│   │   │   │   │   ├── ALB/
│   │   │   │   │   │   ├── ALB_1d.parquet
│   │   │   │   │   │   ├── ALB_1min.parquet
│   │   │   │   │   ├── ALGN/
│   │   │   │   │   │   ├── ALGN_1d.parquet
│   │   │   │   │   │   ├── ALGN_1min.parquet
│   │   │   │   │   ├── ALK/
│   │   │   │   │   │   ├── ALK_1d.parquet
│   │   │   │   │   │   ├── ALK_1min.parquet
│   │   │   │   │   ├── ALL/
│   │   │   │   │   │   ├── ALL_1d.parquet
│   │   │   │   │   │   ├── ALL_1min.parquet
│   │   │   │   │   ├── ALLE/
│   │   │   │   │   │   ├── ALLE_1d.parquet
│   │   │   │   │   │   ├── ALLE_1min.parquet
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
│   │   │   │   │   │   ├── AME_1min.parquet
│   │   │   │   │   ├── AMGN/
│   │   │   │   │   │   ├── AMGN_1d.parquet
│   │   │   │   │   │   ├── AMGN_1min.parquet
│   │   │   │   │   ├── AMP/
│   │   │   │   │   │   ├── AMP_1d.parquet
│   │   │   │   │   ├── AMT/
│   │   │   │   │   │   ├── AMT_1d.parquet
│   │   │   │   │   │   ├── AMT_1min.parquet
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
│   │   │   │   │   │   ├── ANSS_1min.parquet
│   │   │   │   │   ├── AON/
│   │   │   │   │   │   ├── AON_1d.parquet
│   │   │   │   │   │   ├── AON_1min.parquet
│   │   │   │   │   ├── AOS/
│   │   │   │   │   │   ├── AOS_1d.parquet
│   │   │   │   │   │   ├── AOS_1min.parquet
│   │   │   │   │   ├── APA/
│   │   │   │   │   │   ├── APA_1d.parquet
│   │   │   │   │   │   ├── APA_1min.parquet
│   │   │   │   │   ├── APD/
│   │   │   │   │   │   ├── APD_1d.parquet
│   │   │   │   │   │   ├── APD_1min.parquet
│   │   │   │   │   ├── APH/
│   │   │   │   │   │   ├── APH_1d.parquet
│   │   │   │   │   │   ├── APH_1min.parquet
│   │   │   │   │   ├── APO/
│   │   │   │   │   │   ├── APO_1d.parquet
│   │   │   │   │   │   ├── APO_1min.parquet
│   │   │   │   │   ├── APTV/
│   │   │   │   │   │   ├── APTV_1d.parquet
│   │   │   │   │   │   ├── APTV_1min.parquet
│   │   │   │   │   ├── ARE/
│   │   │   │   │   │   ├── ARE_1d.parquet
│   │   │   │   │   │   ├── ARE_1min.parquet
│   │   │   │   │   ├── ATO/
│   │   │   │   │   │   ├── ATO_1d.parquet
│   │   │   │   │   │   ├── ATO_1min.parquet
│   │   │   │   │   ├── AVB/
│   │   │   │   │   │   ├── AVB_1d.parquet
│   │   │   │   │   │   ├── AVB_1min.parquet
│   │   │   │   │   ├── AVGO/
│   │   │   │   │   │   ├── AVGO_15min.parquet
│   │   │   │   │   │   ├── AVGO_1d.parquet
│   │   │   │   │   │   ├── AVGO_1min.parquet
│   │   │   │   │   ├── AVY/
│   │   │   │   │   │   ├── AVY_1d.parquet
│   │   │   │   │   │   ├── AVY_1min.parquet
│   │   │   │   │   ├── AWK/
│   │   │   │   │   │   ├── AWK_1d.parquet
│   │   │   │   │   │   ├── AWK_1min.parquet
│   │   │   │   │   ├── AXON/
│   │   │   │   │   │   ├── AXON_1d.parquet
│   │   │   │   │   ├── AXP/
│   │   │   │   │   │   ├── AXP_1d.parquet
│   │   │   │   │   │   ├── AXP_1min.parquet
│   │   │   │   │   ├── AZO/
│   │   │   │   │   │   ├── AZO_1d.parquet
│   │   │   │   │   ├── BA/
│   │   │   │   │   │   ├── BA_15min.parquet
│   │   │   │   │   │   ├── BA_1d.parquet
│   │   │   │   │   │   ├── BA_1min.parquet
│   │   │   │   │   ├── BAC/
│   │   │   │   │   │   ├── BAC_1d.parquet
│   │   │   │   │   │   ├── BAC_1min.parquet
│   │   │   │   │   ├── BALL/
│   │   │   │   │   │   ├── BALL_1d.parquet
│   │   │   │   │   │   ├── BALL_1min.parquet
│   │   │   │   │   ├── BAX/
│   │   │   │   │   │   ├── BAX_1d.parquet
│   │   │   │   │   │   ├── BAX_1min.parquet
│   │   │   │   │   ├── BBWI/
│   │   │   │   │   │   ├── BBWI_1d.parquet
│   │   │   │   │   │   ├── BBWI_1min.parquet
│   │   │   │   │   ├── BBY/
│   │   │   │   │   │   ├── BBY_1d.parquet
│   │   │   │   │   │   ├── BBY_1min.parquet
│   │   │   │   │   ├── BDX/
│   │   │   │   │   │   ├── BDX_1d.parquet
│   │   │   │   │   │   ├── BDX_1min.parquet
│   │   │   │   │   ├── BEN/
│   │   │   │   │   │   ├── BEN_1d.parquet
│   │   │   │   │   │   ├── BEN_1min.parquet
│   │   │   │   │   ├── BG/
│   │   │   │   │   │   ├── BG_1d.parquet
│   │   │   │   │   │   ├── BG_1min.parquet
│   │   │   │   │   ├── BIIB/
│   │   │   │   │   │   ├── BIIB_1d.parquet
│   │   │   │   │   │   ├── BIIB_1min.parquet
│   │   │   │   │   ├── BIO/
│   │   │   │   │   │   ├── BIO_1d.parquet
│   │   │   │   │   │   ├── BIO_1min.parquet
│   │   │   │   │   ├── BK/
│   │   │   │   │   │   ├── BK_1d.parquet
│   │   │   │   │   │   ├── BK_1min.parquet
│   │   │   │   │   ├── BKNG/
│   │   │   │   │   │   ├── BKNG_1d.parquet
│   │   │   │   │   ├── BKR/
│   │   │   │   │   │   ├── BKR_1d.parquet
│   │   │   │   │   │   ├── BKR_1min.parquet
│   │   │   │   │   ├── BLDR/
│   │   │   │   │   │   ├── BLDR_1d.parquet
│   │   │   │   │   │   ├── BLDR_1min.parquet
│   │   │   │   │   ├── BLK/
│   │   │   │   │   │   ├── BLK_1d.parquet
│   │   │   │   │   ├── BMY/
│   │   │   │   │   │   ├── BMY_1d.parquet
│   │   │   │   │   │   ├── BMY_1min.parquet
│   │   │   │   │   ├── BR/
│   │   │   │   │   │   ├── BR_1d.parquet
│   │   │   │   │   │   ├── BR_1min.parquet
│   │   │   │   │   ├── BRO/
│   │   │   │   │   │   ├── BRO_1d.parquet
│   │   │   │   │   │   ├── BRO_1min.parquet
│   │   │   │   │   ├── BSX/
│   │   │   │   │   │   ├── BSX_1d.parquet
│   │   │   │   │   │   ├── BSX_1min.parquet
│   │   │   │   │   ├── BWA/
│   │   │   │   │   │   ├── BWA_1d.parquet
│   │   │   │   │   │   ├── BWA_1min.parquet
│   │   │   │   │   ├── BX/
│   │   │   │   │   │   ├── BX_1d.parquet
│   │   │   │   │   │   ├── BX_1min.parquet
│   │   │   │   │   ├── BXP/
│   │   │   │   │   │   ├── BXP_1d.parquet
│   │   │   │   │   │   ├── BXP_1min.parquet
│   │   │   │   │   ├── C/
│   │   │   │   │   │   ├── C_1d.parquet
│   │   │   │   │   │   ├── C_1min.parquet
│   │   │   │   │   ├── CAG/
│   │   │   │   │   │   ├── CAG_1d.parquet
│   │   │   │   │   │   ├── CAG_1min.parquet
│   │   │   │   │   ├── CAH/
│   │   │   │   │   │   ├── CAH_1d.parquet
│   │   │   │   │   │   ├── CAH_1min.parquet
│   │   │   │   │   ├── CARR/
│   │   │   │   │   │   ├── CARR_1d.parquet
│   │   │   │   │   │   ├── CARR_1min.parquet
│   │   │   │   │   ├── CAT/
│   │   │   │   │   │   ├── CAT_1d.parquet
│   │   │   │   │   │   ├── CAT_1min.parquet
│   │   │   │   │   ├── CB/
│   │   │   │   │   │   ├── CB_1d.parquet
│   │   │   │   │   │   ├── CB_1min.parquet
│   │   │   │   │   ├── CBOE/
│   │   │   │   │   │   ├── CBOE_1d.parquet
│   │   │   │   │   │   ├── CBOE_1min.parquet
│   │   │   │   │   ├── CBRE/
│   │   │   │   │   │   ├── CBRE_1d.parquet
│   │   │   │   │   │   ├── CBRE_1min.parquet
│   │   │   │   │   ├── CCI/
│   │   │   │   │   │   ├── CCI_1d.parquet
│   │   │   │   │   │   ├── CCI_1min.parquet
│   │   │   │   │   ├── CCL/
│   │   │   │   │   │   ├── CCL_1d.parquet
│   │   │   │   │   │   ├── CCL_1min.parquet
│   │   │   │   │   ├── CDNS/
│   │   │   │   │   │   ├── CDNS_1d.parquet
│   │   │   │   │   │   ├── CDNS_1min.parquet
│   │   │   │   │   ├── CDW/
│   │   │   │   │   │   ├── CDW_1d.parquet
│   │   │   │   │   │   ├── CDW_1min.parquet
│   │   │   │   │   ├── CE/
│   │   │   │   │   │   ├── CE_1d.parquet
│   │   │   │   │   │   ├── CE_1min.parquet
│   │   │   │   │   ├── CEG/
│   │   │   │   │   │   ├── CEG_15min.parquet
│   │   │   │   │   │   ├── CEG_1d.parquet
│   │   │   │   │   │   ├── CEG_1min.parquet
│   │   │   │   │   ├── CF/
│   │   │   │   │   │   ├── CF_1d.parquet
│   │   │   │   │   │   ├── CF_1min.parquet
│   │   │   │   │   ├── CFG/
│   │   │   │   │   │   ├── CFG_1d.parquet
│   │   │   │   │   │   ├── CFG_1min.parquet
│   │   │   │   │   ├── CHD/
│   │   │   │   │   │   ├── CHD_1d.parquet
│   │   │   │   │   │   ├── CHD_1min.parquet
│   │   │   │   │   ├── CHRW/
│   │   │   │   │   │   ├── CHRW_1d.parquet
│   │   │   │   │   │   ├── CHRW_1min.parquet
│   │   │   │   │   ├── CHTR/
│   │   │   │   │   │   ├── CHTR_1d.parquet
│   │   │   │   │   │   ├── CHTR_1min.parquet
│   │   │   │   │   ├── CI/
│   │   │   │   │   │   ├── CI_1d.parquet
│   │   │   │   │   │   ├── CI_1min.parquet
│   │   │   │   │   ├── CINF/
│   │   │   │   │   │   ├── CINF_1d.parquet
│   │   │   │   │   │   ├── CINF_1min.parquet
│   │   │   │   │   ├── CL/
│   │   │   │   │   │   ├── CL_1d.parquet
│   │   │   │   │   │   ├── CL_1min.parquet
│   │   │   │   │   ├── CLX/
│   │   │   │   │   │   ├── CLX_1d.parquet
│   │   │   │   │   │   ├── CLX_1min.parquet
│   │   │   │   │   ├── CMA/
│   │   │   │   │   │   ├── CMA_1d.parquet
│   │   │   │   │   │   ├── CMA_1min.parquet
│   │   │   │   │   ├── CMCSA/
│   │   │   │   │   │   ├── CMCSA_1d.parquet
│   │   │   │   │   │   ├── CMCSA_1min.parquet
│   │   │   │   │   ├── CME/
│   │   │   │   │   │   ├── CME_1d.parquet
│   │   │   │   │   │   ├── CME_1min.parquet
│   │   │   │   │   ├── CMG/
│   │   │   │   │   │   ├── CMG_1d.parquet
│   │   │   │   │   │   ├── CMG_1min.parquet
│   │   │   │   │   ├── CMI/
│   │   │   │   │   │   ├── CMI_1d.parquet
│   │   │   │   │   │   ├── CMI_1min.parquet
│   │   │   │   │   ├── CMS/
│   │   │   │   │   │   ├── CMS_1d.parquet
│   │   │   │   │   │   ├── CMS_1min.parquet
│   │   │   │   │   ├── CNC/
│   │   │   │   │   │   ├── CNC_15min.parquet
│   │   │   │   │   │   ├── CNC_1d.parquet
│   │   │   │   │   │   ├── CNC_1min.parquet
│   │   │   │   │   ├── CNP/
│   │   │   │   │   │   ├── CNP_1d.parquet
│   │   │   │   │   │   ├── CNP_1min.parquet
│   │   │   │   │   ├── COF/
│   │   │   │   │   │   ├── COF_1d.parquet
│   │   │   │   │   │   ├── COF_1min.parquet
│   │   │   │   │   ├── COO/
│   │   │   │   │   │   ├── COO_1d.parquet
│   │   │   │   │   │   ├── COO_1min.parquet
│   │   │   │   │   ├── COP/
│   │   │   │   │   │   ├── COP_1d.parquet
│   │   │   │   │   │   ├── COP_1min.parquet
│   │   │   │   │   ├── COR/
│   │   │   │   │   │   ├── COR_1d.parquet
│   │   │   │   │   │   ├── COR_1min.parquet
│   │   │   │   │   ├── COST/
│   │   │   │   │   │   ├── COST_1d.parquet
│   │   │   │   │   ├── CPAY/
│   │   │   │   │   │   ├── CPAY_1d.parquet
│   │   │   │   │   │   ├── CPAY_1min.parquet
│   │   │   │   │   ├── CPB/
│   │   │   │   │   │   ├── CPB_1d.parquet
│   │   │   │   │   │   ├── CPB_1min.parquet
│   │   │   │   │   ├── CPRT/
│   │   │   │   │   │   ├── CPRT_1d.parquet
│   │   │   │   │   │   ├── CPRT_1min.parquet
│   │   │   │   │   ├── CPT/
│   │   │   │   │   │   ├── CPT_1d.parquet
│   │   │   │   │   │   ├── CPT_1min.parquet
│   │   │   │   │   ├── CRCL/
│   │   │   │   │   │   ├── CRCL_15min.parquet
│   │   │   │   │   │   ├── CRCL_1min.parquet
│   │   │   │   │   ├── CRL/
│   │   │   │   │   │   ├── CRL_1d.parquet
│   │   │   │   │   │   ├── CRL_1min.parquet
│   │   │   │   │   ├── CRM/
│   │   │   │   │   │   ├── CRM_1d.parquet
│   │   │   │   │   │   ├── CRM_1min.parquet
│   │   │   │   │   ├── CRWD/
│   │   │   │   │   │   ├── CRWD_1d.parquet
│   │   │   │   │   │   ├── CRWD_1min.parquet
│   │   │   │   │   ├── CSCO/
│   │   │   │   │   │   ├── CSCO_1d.parquet
│   │   │   │   │   │   ├── CSCO_1min.parquet
│   │   │   │   │   ├── CSGP/
│   │   │   │   │   │   ├── CSGP_1d.parquet
│   │   │   │   │   │   ├── CSGP_1min.parquet
│   │   │   │   │   ├── CSX/
│   │   │   │   │   │   ├── CSX_1d.parquet
│   │   │   │   │   │   ├── CSX_1min.parquet
│   │   │   │   │   ├── CTAS/
│   │   │   │   │   │   ├── CTAS_1d.parquet
│   │   │   │   │   │   ├── CTAS_1min.parquet
│   │   │   │   │   ├── CTRA/
│   │   │   │   │   │   ├── CTRA_1d.parquet
│   │   │   │   │   │   ├── CTRA_1min.parquet
│   │   │   │   │   ├── CTSH/
│   │   │   │   │   │   ├── CTSH_1d.parquet
│   │   │   │   │   │   ├── CTSH_1min.parquet
│   │   │   │   │   ├── CTVA/
│   │   │   │   │   │   ├── CTVA_1d.parquet
│   │   │   │   │   │   ├── CTVA_1min.parquet
│   │   │   │   │   ├── CVS/
│   │   │   │   │   │   ├── CVS_1d.parquet
│   │   │   │   │   │   ├── CVS_1min.parquet
│   │   │   │   │   ├── CVX/
│   │   │   │   │   │   ├── CVX_1d.parquet
│   │   │   │   │   │   ├── CVX_1min.parquet
│   │   │   │   │   ├── CZR/
│   │   │   │   │   │   ├── CZR_1d.parquet
│   │   │   │   │   │   ├── CZR_1min.parquet
│   │   │   │   │   ├── D/
│   │   │   │   │   │   ├── D_1d.parquet
│   │   │   │   │   │   ├── D_1min.parquet
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
│   │   │   │   │   │   ├── DGX_1min.parquet
│   │   │   │   │   ├── DHI/
│   │   │   │   │   │   ├── DHI_1d.parquet
│   │   │   │   │   │   ├── DHI_1min.parquet
│   │   │   │   │   ├── DHR/
│   │   │   │   │   │   ├── DHR_1d.parquet
│   │   │   │   │   │   ├── DHR_1min.parquet
│   │   │   │   │   ├── DIS/
│   │   │   │   │   │   ├── DIS_1d.parquet
│   │   │   │   │   │   ├── DIS_1min.parquet
│   │   │   │   │   ├── DLR/
│   │   │   │   │   │   ├── DLR_1d.parquet
│   │   │   │   │   │   ├── DLR_1min.parquet
│   │   │   │   │   ├── DLTR/
│   │   │   │   │   │   ├── DLTR_1d.parquet
│   │   │   │   │   │   ├── DLTR_1min.parquet
│   │   │   │   │   ├── DOC/
│   │   │   │   │   │   ├── DOC_1d.parquet
│   │   │   │   │   │   ├── DOC_1min.parquet
│   │   │   │   │   ├── DOV/
│   │   │   │   │   │   ├── DOV_1d.parquet
│   │   │   │   │   │   ├── DOV_1min.parquet
│   │   │   │   │   ├── DOW/
│   │   │   │   │   │   ├── DOW_1d.parquet
│   │   │   │   │   │   ├── DOW_1min.parquet
│   │   │   │   │   ├── DPZ/
│   │   │   │   │   │   ├── DPZ_1d.parquet
│   │   │   │   │   │   ├── DPZ_1min.parquet
│   │   │   │   │   ├── DRI/
│   │   │   │   │   │   ├── DRI_1d.parquet
│   │   │   │   │   │   ├── DRI_1min.parquet
│   │   │   │   │   ├── DTE/
│   │   │   │   │   │   ├── DTE_1d.parquet
│   │   │   │   │   │   ├── DTE_1min.parquet
│   │   │   │   │   ├── DUK/
│   │   │   │   │   │   ├── DUK_1d.parquet
│   │   │   │   │   │   ├── DUK_1min.parquet
│   │   │   │   │   ├── DVA/
│   │   │   │   │   │   ├── DVA_1d.parquet
│   │   │   │   │   │   ├── DVA_1min.parquet
│   │   │   │   │   ├── DVN/
│   │   │   │   │   │   ├── DVN_1d.parquet
│   │   │   │   │   │   ├── DVN_1min.parquet
│   │   │   │   │   ├── DXCM/
│   │   │   │   │   │   ├── DXCM_1d.parquet
│   │   │   │   │   │   ├── DXCM_1min.parquet
│   │   │   │   │   ├── EA/
│   │   │   │   │   │   ├── EA_1d.parquet
│   │   │   │   │   │   ├── EA_1min.parquet
│   │   │   │   │   ├── EBAY/
│   │   │   │   │   │   ├── EBAY_1d.parquet
│   │   │   │   │   │   ├── EBAY_1min.parquet
│   │   │   │   │   ├── ECL/
│   │   │   │   │   │   ├── ECL_1d.parquet
│   │   │   │   │   │   ├── ECL_1min.parquet
│   │   │   │   │   ├── ED/
│   │   │   │   │   │   ├── ED_1d.parquet
│   │   │   │   │   │   ├── ED_1min.parquet
│   │   │   │   │   ├── EFX/
│   │   │   │   │   │   ├── EFX_1d.parquet
│   │   │   │   │   │   ├── EFX_1min.parquet
│   │   │   │   │   ├── EG/
│   │   │   │   │   │   ├── EG_1d.parquet
│   │   │   │   │   │   ├── EG_1min.parquet
│   │   │   │   │   ├── EIX/
│   │   │   │   │   │   ├── EIX_1d.parquet
│   │   │   │   │   │   ├── EIX_1min.parquet
│   │   │   │   │   ├── EL/
│   │   │   │   │   │   ├── EL_1d.parquet
│   │   │   │   │   │   ├── EL_1min.parquet
│   │   │   │   │   ├── ELV/
│   │   │   │   │   │   ├── ELV_1d.parquet
│   │   │   │   │   │   ├── ELV_1min.parquet
│   │   │   │   │   ├── EMN/
│   │   │   │   │   │   ├── EMN_1d.parquet
│   │   │   │   │   │   ├── EMN_1min.parquet
│   │   │   │   │   ├── EMR/
│   │   │   │   │   │   ├── EMR_1d.parquet
│   │   │   │   │   │   ├── EMR_1min.parquet
│   │   │   │   │   ├── ENPH/
│   │   │   │   │   │   ├── ENPH_1d.parquet
│   │   │   │   │   │   ├── ENPH_1min.parquet
│   │   │   │   │   ├── EOG/
│   │   │   │   │   │   ├── EOG_1d.parquet
│   │   │   │   │   │   ├── EOG_1min.parquet
│   │   │   │   │   ├── EPAM/
│   │   │   │   │   │   ├── EPAM_1d.parquet
│   │   │   │   │   │   ├── EPAM_1min.parquet
│   │   │   │   │   ├── EQIX/
│   │   │   │   │   │   ├── EQIX_1d.parquet
│   │   │   │   │   ├── EQR/
│   │   │   │   │   │   ├── EQR_1d.parquet
│   │   │   │   │   │   ├── EQR_1min.parquet
│   │   │   │   │   ├── EQT/
│   │   │   │   │   │   ├── EQT_1d.parquet
│   │   │   │   │   │   ├── EQT_1min.parquet
│   │   │   │   │   ├── ERIE/
│   │   │   │   │   │   ├── ERIE_1d.parquet
│   │   │   │   │   │   ├── ERIE_1min.parquet
│   │   │   │   │   ├── ES/
│   │   │   │   │   │   ├── ES_1d.parquet
│   │   │   │   │   │   ├── ES_1min.parquet
│   │   │   │   │   ├── ESS/
│   │   │   │   │   │   ├── ESS_1d.parquet
│   │   │   │   │   │   ├── ESS_1min.parquet
│   │   │   │   │   ├── ETN/
│   │   │   │   │   │   ├── ETN_1d.parquet
│   │   │   │   │   │   ├── ETN_1min.parquet
│   │   │   │   │   ├── ETR/
│   │   │   │   │   │   ├── ETR_1d.parquet
│   │   │   │   │   │   ├── ETR_1min.parquet
│   │   │   │   │   ├── ETSY/
│   │   │   │   │   │   ├── ETSY_1d.parquet
│   │   │   │   │   │   ├── ETSY_1min.parquet
│   │   │   │   │   ├── EVRG/
│   │   │   │   │   │   ├── EVRG_1d.parquet
│   │   │   │   │   │   ├── EVRG_1min.parquet
│   │   │   │   │   ├── EW/
│   │   │   │   │   │   ├── EW_1d.parquet
│   │   │   │   │   │   ├── EW_1min.parquet
│   │   │   │   │   ├── EXC/
│   │   │   │   │   │   ├── EXC_1d.parquet
│   │   │   │   │   │   ├── EXC_1min.parquet
│   │   │   │   │   ├── EXPD/
│   │   │   │   │   │   ├── EXPD_1d.parquet
│   │   │   │   │   │   ├── EXPD_1min.parquet
│   │   │   │   │   ├── EXPE/
│   │   │   │   │   │   ├── EXPE_1d.parquet
│   │   │   │   │   │   ├── EXPE_1min.parquet
│   │   │   │   │   ├── EXR/
│   │   │   │   │   │   ├── EXR_1d.parquet
│   │   │   │   │   │   ├── EXR_1min.parquet
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
│   │   │   │   │   │   ├── FDS_1min.parquet
│   │   │   │   │   ├── FDX/
│   │   │   │   │   │   ├── FDX_1d.parquet
│   │   │   │   │   │   ├── FDX_1min.parquet
│   │   │   │   │   ├── FE/
│   │   │   │   │   │   ├── FE_1d.parquet
│   │   │   │   │   │   ├── FE_1min.parquet
│   │   │   │   │   ├── FFIV/
│   │   │   │   │   │   ├── FFIV_1d.parquet
│   │   │   │   │   │   ├── FFIV_1min.parquet
│   │   │   │   │   ├── FI/
│   │   │   │   │   │   ├── FI_1d.parquet
│   │   │   │   │   │   ├── FI_1min.parquet
│   │   │   │   │   ├── FICO/
│   │   │   │   │   │   ├── FICO_1d.parquet
│   │   │   │   │   ├── FIS/
│   │   │   │   │   │   ├── FIS_1d.parquet
│   │   │   │   │   │   ├── FIS_1min.parquet
│   │   │   │   │   ├── FITB/
│   │   │   │   │   │   ├── FITB_1d.parquet
│   │   │   │   │   │   ├── FITB_1min.parquet
│   │   │   │   │   ├── FMC/
│   │   │   │   │   │   ├── FMC_1d.parquet
│   │   │   │   │   │   ├── FMC_1min.parquet
│   │   │   │   │   ├── FOX/
│   │   │   │   │   │   ├── FOX_1d.parquet
│   │   │   │   │   │   ├── FOX_1min.parquet
│   │   │   │   │   ├── FOXA/
│   │   │   │   │   │   ├── FOXA_1d.parquet
│   │   │   │   │   │   ├── FOXA_1min.parquet
│   │   │   │   │   ├── FRT/
│   │   │   │   │   │   ├── FRT_1d.parquet
│   │   │   │   │   │   ├── FRT_1min.parquet
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
│   │   │   │   │   │   ├── GD_1min.parquet
│   │   │   │   │   ├── GDDY/
│   │   │   │   │   │   ├── GDDY_1d.parquet
│   │   │   │   │   │   ├── GDDY_1min.parquet
│   │   │   │   │   ├── GE/
│   │   │   │   │   │   ├── GE_1d.parquet
│   │   │   │   │   │   ├── GE_1min.parquet
│   │   │   │   │   ├── GEHC/
│   │   │   │   │   │   ├── GEHC_1d.parquet
│   │   │   │   │   │   ├── GEHC_1min.parquet
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
│   │   │   │   │   │   ├── GIS_1min.parquet
│   │   │   │   │   ├── GL/
│   │   │   │   │   │   ├── GL_1d.parquet
│   │   │   │   │   │   ├── GL_1min.parquet
│   │   │   │   │   ├── GLW/
│   │   │   │   │   │   ├── GLW_1d.parquet
│   │   │   │   │   │   ├── GLW_1min.parquet
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
│   │   │   │   │   │   ├── GPC_1min.parquet
│   │   │   │   │   ├── GPN/
│   │   │   │   │   │   ├── GPN_1d.parquet
│   │   │   │   │   │   ├── GPN_1min.parquet
│   │   │   │   │   ├── GRMN/
│   │   │   │   │   │   ├── GRMN_1d.parquet
│   │   │   │   │   │   ├── GRMN_1min.parquet
│   │   │   │   │   ├── GS/
│   │   │   │   │   │   ├── GS_1d.parquet
│   │   │   │   │   ├── GWW/
│   │   │   │   │   │   ├── GWW_1d.parquet
│   │   │   │   │   ├── HAL/
│   │   │   │   │   │   ├── HAL_1d.parquet
│   │   │   │   │   │   ├── HAL_1min.parquet
│   │   │   │   │   ├── HAS/
│   │   │   │   │   │   ├── HAS_1d.parquet
│   │   │   │   │   │   ├── HAS_1min.parquet
│   │   │   │   │   ├── HBAN/
│   │   │   │   │   │   ├── HBAN_1d.parquet
│   │   │   │   │   │   ├── HBAN_1min.parquet
│   │   │   │   │   ├── HCA/
│   │   │   │   │   │   ├── HCA_1d.parquet
│   │   │   │   │   │   ├── HCA_1min.parquet
│   │   │   │   │   ├── HD/
│   │   │   │   │   │   ├── HD_1d.parquet
│   │   │   │   │   │   ├── HD_1min.parquet
│   │   │   │   │   ├── HES/
│   │   │   │   │   │   ├── HES_1d.parquet
│   │   │   │   │   │   ├── HES_1min.parquet
│   │   │   │   │   ├── HIG/
│   │   │   │   │   │   ├── HIG_1d.parquet
│   │   │   │   │   │   ├── HIG_1min.parquet
│   │   │   │   │   ├── HII/
│   │   │   │   │   │   ├── HII_1d.parquet
│   │   │   │   │   │   ├── HII_1min.parquet
│   │   │   │   │   ├── HLT/
│   │   │   │   │   │   ├── HLT_1d.parquet
│   │   │   │   │   │   ├── HLT_1min.parquet
│   │   │   │   │   ├── HOLX/
│   │   │   │   │   │   ├── HOLX_1d.parquet
│   │   │   │   │   │   ├── HOLX_1min.parquet
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
│   │   │   │   │   │   ├── HSIC_1min.parquet
│   │   │   │   │   ├── HST/
│   │   │   │   │   │   ├── HST_1d.parquet
│   │   │   │   │   │   ├── HST_1min.parquet
│   │   │   │   │   ├── HSY/
│   │   │   │   │   │   ├── HSY_1d.parquet
│   │   │   │   │   │   ├── HSY_1min.parquet
│   │   │   │   │   ├── HUBB/
│   │   │   │   │   │   ├── HUBB_1d.parquet
│   │   │   │   │   │   ├── HUBB_1min.parquet
│   │   │   │   │   ├── HUM/
│   │   │   │   │   │   ├── HUM_1d.parquet
│   │   │   │   │   │   ├── HUM_1min.parquet
│   │   │   │   │   ├── HWM/
│   │   │   │   │   │   ├── HWM_1d.parquet
│   │   │   │   │   │   ├── HWM_1min.parquet
│   │   │   │   │   ├── IBM/
│   │   │   │   │   │   ├── IBM_1d.parquet
│   │   │   │   │   │   ├── IBM_1min.parquet
│   │   │   │   │   ├── ICE/
│   │   │   │   │   │   ├── ICE_1d.parquet
│   │   │   │   │   │   ├── ICE_1min.parquet
│   │   │   │   │   ├── IDXX/
│   │   │   │   │   │   ├── IDXX_1d.parquet
│   │   │   │   │   ├── IEX/
│   │   │   │   │   │   ├── IEX_1d.parquet
│   │   │   │   │   │   ├── IEX_1min.parquet
│   │   │   │   │   ├── IFF/
│   │   │   │   │   │   ├── IFF_1d.parquet
│   │   │   │   │   │   ├── IFF_1min.parquet
│   │   │   │   │   ├── ILMN/
│   │   │   │   │   │   ├── ILMN_1d.parquet
│   │   │   │   │   │   ├── ILMN_1min.parquet
│   │   │   │   │   ├── INCY/
│   │   │   │   │   │   ├── INCY_1d.parquet
│   │   │   │   │   │   ├── INCY_1min.parquet
│   │   │   │   │   ├── INTC/
│   │   │   │   │   │   ├── INTC_15min.parquet
│   │   │   │   │   │   ├── INTC_1d.parquet
│   │   │   │   │   │   ├── INTC_1min.parquet
│   │   │   │   │   ├── INTU/
│   │   │   │   │   │   ├── INTU_1d.parquet
│   │   │   │   │   ├── INVH/
│   │   │   │   │   │   ├── INVH_1d.parquet
│   │   │   │   │   │   ├── INVH_1min.parquet
│   │   │   │   │   ├── IP/
│   │   │   │   │   │   ├── IP_1d.parquet
│   │   │   │   │   │   ├── IP_1min.parquet
│   │   │   │   │   ├── IPG/
│   │   │   │   │   │   ├── IPG_1d.parquet
│   │   │   │   │   │   ├── IPG_1min.parquet
│   │   │   │   │   ├── IQV/
│   │   │   │   │   │   ├── IQV_1d.parquet
│   │   │   │   │   │   ├── IQV_1min.parquet
│   │   │   │   │   ├── IR/
│   │   │   │   │   │   ├── IR_1d.parquet
│   │   │   │   │   │   ├── IR_1min.parquet
│   │   │   │   │   ├── IRM/
│   │   │   │   │   │   ├── IRM_1d.parquet
│   │   │   │   │   │   ├── IRM_1min.parquet
│   │   │   │   │   ├── ISRG/
│   │   │   │   │   │   ├── ISRG_1d.parquet
│   │   │   │   │   ├── IT/
│   │   │   │   │   │   ├── IT_1d.parquet
│   │   │   │   │   │   ├── IT_1min.parquet
│   │   │   │   │   ├── ITW/
│   │   │   │   │   │   ├── ITW_1d.parquet
│   │   │   │   │   │   ├── ITW_1min.parquet
│   │   │   │   │   ├── IVZ/
│   │   │   │   │   │   ├── IVZ_1d.parquet
│   │   │   │   │   │   ├── IVZ_1min.parquet
│   │   │   │   │   ├── J/
│   │   │   │   │   │   ├── J_1d.parquet
│   │   │   │   │   │   ├── J_1min.parquet
│   │   │   │   │   ├── JBHT/
│   │   │   │   │   │   ├── JBHT_1d.parquet
│   │   │   │   │   │   ├── JBHT_1min.parquet
│   │   │   │   │   ├── JBL/
│   │   │   │   │   │   ├── JBL_1d.parquet
│   │   │   │   │   │   ├── JBL_1min.parquet
│   │   │   │   │   ├── JCI/
│   │   │   │   │   │   ├── JCI_1d.parquet
│   │   │   │   │   │   ├── JCI_1min.parquet
│   │   │   │   │   ├── JKHY/
│   │   │   │   │   │   ├── JKHY_1d.parquet
│   │   │   │   │   │   ├── JKHY_1min.parquet
│   │   │   │   │   ├── JNJ/
│   │   │   │   │   │   ├── JNJ_1d.parquet
│   │   │   │   │   │   ├── JNJ_1min.parquet
│   │   │   │   │   ├── JNPR/
│   │   │   │   │   │   ├── JNPR_1d.parquet
│   │   │   │   │   │   ├── JNPR_1min.parquet
│   │   │   │   │   ├── JPM/
│   │   │   │   │   │   ├── JPM_1d.parquet
│   │   │   │   │   │   ├── JPM_1min.parquet
│   │   │   │   │   ├── K/
│   │   │   │   │   │   ├── K_1d.parquet
│   │   │   │   │   │   ├── K_1min.parquet
│   │   │   │   │   ├── KDP/
│   │   │   │   │   │   ├── KDP_1d.parquet
│   │   │   │   │   │   ├── KDP_1min.parquet
│   │   │   │   │   ├── KEY/
│   │   │   │   │   │   ├── KEY_1d.parquet
│   │   │   │   │   │   ├── KEY_1min.parquet
│   │   │   │   │   ├── KEYS/
│   │   │   │   │   │   ├── KEYS_1d.parquet
│   │   │   │   │   │   ├── KEYS_1min.parquet
│   │   │   │   │   ├── KHC/
│   │   │   │   │   │   ├── KHC_1d.parquet
│   │   │   │   │   │   ├── KHC_1min.parquet
│   │   │   │   │   ├── KIM/
│   │   │   │   │   │   ├── KIM_1d.parquet
│   │   │   │   │   │   ├── KIM_1min.parquet
│   │   │   │   │   ├── KKR/
│   │   │   │   │   │   ├── KKR_1d.parquet
│   │   │   │   │   │   ├── KKR_1min.parquet
│   │   │   │   │   ├── KLAC/
│   │   │   │   │   │   ├── KLAC_1d.parquet
│   │   │   │   │   ├── KMB/
│   │   │   │   │   │   ├── KMB_1d.parquet
│   │   │   │   │   │   ├── KMB_1min.parquet
│   │   │   │   │   ├── KMI/
│   │   │   │   │   │   ├── KMI_1d.parquet
│   │   │   │   │   │   ├── KMI_1min.parquet
│   │   │   │   │   ├── KMX/
│   │   │   │   │   │   ├── KMX_1d.parquet
│   │   │   │   │   │   ├── KMX_1min.parquet
│   │   │   │   │   ├── KO/
│   │   │   │   │   │   ├── KO_1d.parquet
│   │   │   │   │   │   ├── KO_1min.parquet
│   │   │   │   │   ├── KR/
│   │   │   │   │   │   ├── KR_1d.parquet
│   │   │   │   │   │   ├── KR_1min.parquet
│   │   │   │   │   ├── KVUE/
│   │   │   │   │   │   ├── KVUE_1d.parquet
│   │   │   │   │   │   ├── KVUE_1min.parquet
│   │   │   │   │   ├── L/
│   │   │   │   │   │   ├── L_1d.parquet
│   │   │   │   │   │   ├── L_1min.parquet
│   │   │   │   │   ├── LDOS/
│   │   │   │   │   │   ├── LDOS_1d.parquet
│   │   │   │   │   │   ├── LDOS_1min.parquet
│   │   │   │   │   ├── LEN/
│   │   │   │   │   │   ├── LEN_1d.parquet
│   │   │   │   │   │   ├── LEN_1min.parquet
│   │   │   │   │   ├── LH/
│   │   │   │   │   │   ├── LH_1d.parquet
│   │   │   │   │   │   ├── LH_1min.parquet
│   │   │   │   │   ├── LHX/
│   │   │   │   │   │   ├── LHX_1d.parquet
│   │   │   │   │   │   ├── LHX_1min.parquet
│   │   │   │   │   ├── LII/
│   │   │   │   │   │   ├── LII_1d.parquet
│   │   │   │   │   ├── LIN/
│   │   │   │   │   │   ├── LIN_1d.parquet
│   │   │   │   │   │   ├── LIN_1min.parquet
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
│   │   │   │   │   │   ├── LNT_1min.parquet
│   │   │   │   │   ├── LOW/
│   │   │   │   │   │   ├── LOW_1d.parquet
│   │   │   │   │   │   ├── LOW_1min.parquet
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
│   │   │   │   │   │   ├── LVS_1min.parquet
│   │   │   │   │   ├── LW/
│   │   │   │   │   │   ├── LW_1d.parquet
│   │   │   │   │   │   ├── LW_1min.parquet
│   │   │   │   │   ├── LYB/
│   │   │   │   │   │   ├── LYB_1d.parquet
│   │   │   │   │   │   ├── LYB_1min.parquet
│   │   │   │   │   ├── LYV/
│   │   │   │   │   │   ├── LYV_1d.parquet
│   │   │   │   │   │   ├── LYV_1min.parquet
│   │   │   │   │   ├── MA/
│   │   │   │   │   │   ├── MA_1d.parquet
│   │   │   │   │   ├── MAA/
│   │   │   │   │   │   ├── MAA_1d.parquet
│   │   │   │   │   │   ├── MAA_1min.parquet
│   │   │   │   │   ├── MAR/
│   │   │   │   │   │   ├── MAR_1d.parquet
│   │   │   │   │   │   ├── MAR_1min.parquet
│   │   │   │   │   ├── MAS/
│   │   │   │   │   │   ├── MAS_1d.parquet
│   │   │   │   │   │   ├── MAS_1min.parquet
│   │   │   │   │   ├── MBC/
│   │   │   │   │   │   ├── MBC_1d.parquet
│   │   │   │   │   │   ├── MBC_1min.parquet
│   │   │   │   │   ├── MCD/
│   │   │   │   │   │   ├── MCD_1d.parquet
│   │   │   │   │   │   ├── MCD_1min.parquet
│   │   │   │   │   ├── MCHP/
│   │   │   │   │   │   ├── MCHP_1d.parquet
│   │   │   │   │   │   ├── MCHP_1min.parquet
│   │   │   │   │   ├── MCK/
│   │   │   │   │   │   ├── MCK_1d.parquet
│   │   │   │   │   ├── MCO/
│   │   │   │   │   │   ├── MCO_1d.parquet
│   │   │   │   │   │   ├── MCO_1min.parquet
│   │   │   │   │   ├── MDLZ/
│   │   │   │   │   │   ├── MDLZ_1d.parquet
│   │   │   │   │   │   ├── MDLZ_1min.parquet
│   │   │   │   │   ├── MDT/
│   │   │   │   │   │   ├── MDT_1d.parquet
│   │   │   │   │   │   ├── MDT_1min.parquet
│   │   │   │   │   ├── MET/
│   │   │   │   │   │   ├── MET_1d.parquet
│   │   │   │   │   │   ├── MET_1min.parquet
│   │   │   │   │   ├── META/
│   │   │   │   │   │   ├── META_1d.parquet
│   │   │   │   │   ├── MGM/
│   │   │   │   │   │   ├── MGM_1d.parquet
│   │   │   │   │   │   ├── MGM_1min.parquet
│   │   │   │   │   ├── MHK/
│   │   │   │   │   │   ├── MHK_1d.parquet
│   │   │   │   │   │   ├── MHK_1min.parquet
│   │   │   │   │   ├── MKC/
│   │   │   │   │   │   ├── MKC_1d.parquet
│   │   │   │   │   │   ├── MKC_1min.parquet
│   │   │   │   │   ├── MKTX/
│   │   │   │   │   │   ├── MKTX_1d.parquet
│   │   │   │   │   │   ├── MKTX_1min.parquet
│   │   │   │   │   ├── MLM/
│   │   │   │   │   │   ├── MLM_1d.parquet
│   │   │   │   │   ├── MMC/
│   │   │   │   │   │   ├── MMC_1d.parquet
│   │   │   │   │   │   ├── MMC_1min.parquet
│   │   │   │   │   ├── MMM/
│   │   │   │   │   │   ├── MMM_1d.parquet
│   │   │   │   │   │   ├── MMM_1min.parquet
│   │   │   │   │   ├── MNST/
│   │   │   │   │   │   ├── MNST_1d.parquet
│   │   │   │   │   │   ├── MNST_1min.parquet
│   │   │   │   │   ├── MO/
│   │   │   │   │   │   ├── MO_1d.parquet
│   │   │   │   │   │   ├── MO_1min.parquet
│   │   │   │   │   ├── MOH/
│   │   │   │   │   │   ├── MOH_1d.parquet
│   │   │   │   │   │   ├── MOH_1min.parquet
│   │   │   │   │   ├── MOS/
│   │   │   │   │   │   ├── MOS_1d.parquet
│   │   │   │   │   │   ├── MOS_1min.parquet
│   │   │   │   │   ├── MPC/
│   │   │   │   │   │   ├── MPC_1d.parquet
│   │   │   │   │   │   ├── MPC_1min.parquet
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
│   │   │   │   │   │   ├── MSI_1min.parquet
│   │   │   │   │   ├── MTB/
│   │   │   │   │   │   ├── MTB_1d.parquet
│   │   │   │   │   │   ├── MTB_1min.parquet
│   │   │   │   │   ├── MTCH/
│   │   │   │   │   │   ├── MTCH_1d.parquet
│   │   │   │   │   │   ├── MTCH_1min.parquet
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
│   │   │   │   │   │   ├── NDSN_1min.parquet
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
│   │   │   │   │   │   ├── NI_1min.parquet
│   │   │   │   │   ├── NKE/
│   │   │   │   │   │   ├── NKE_15min.parquet
│   │   │   │   │   │   ├── NKE_1d.parquet
│   │   │   │   │   │   ├── NKE_1min.parquet
│   │   │   │   │   ├── NOC/
│   │   │   │   │   │   ├── NOC_1d.parquet
│   │   │   │   │   │   ├── NOC_1min.parquet
│   │   │   │   │   ├── NOW/
│   │   │   │   │   │   ├── NOW_1d.parquet
│   │   │   │   │   ├── NRG/
│   │   │   │   │   │   ├── NRG_1d.parquet
│   │   │   │   │   │   ├── NRG_1min.parquet
│   │   │   │   │   ├── NSC/
│   │   │   │   │   │   ├── NSC_1d.parquet
│   │   │   │   │   │   ├── NSC_1min.parquet
│   │   │   │   │   ├── NTAP/
│   │   │   │   │   │   ├── NTAP_1d.parquet
│   │   │   │   │   │   ├── NTAP_1min.parquet
│   │   │   │   │   ├── NTRS/
│   │   │   │   │   │   ├── NTRS_1d.parquet
│   │   │   │   │   │   ├── NTRS_1min.parquet
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
│   │   │   │   │   │   ├── NWS_1min.parquet
│   │   │   │   │   ├── NWSA/
│   │   │   │   │   │   ├── NWSA_1d.parquet
│   │   │   │   │   │   ├── NWSA_1min.parquet
│   │   │   │   │   ├── NXPI/
│   │   │   │   │   │   ├── NXPI_1d.parquet
│   │   │   │   │   │   ├── NXPI_1min.parquet
│   │   │   │   │   ├── O/
│   │   │   │   │   │   ├── O_1d.parquet
│   │   │   │   │   │   ├── O_1min.parquet
│   │   │   │   │   ├── ODFL/
│   │   │   │   │   │   ├── ODFL_1d.parquet
│   │   │   │   │   │   ├── ODFL_1min.parquet
│   │   │   │   │   ├── OKE/
│   │   │   │   │   │   ├── OKE_1d.parquet
│   │   │   │   │   │   ├── OKE_1min.parquet
│   │   │   │   │   ├── OKLO/
│   │   │   │   │   │   ├── OKLO_15min.parquet
│   │   │   │   │   │   ├── OKLO_1day.parquet
│   │   │   │   │   │   ├── OKLO_1hour.parquet
│   │   │   │   │   │   ├── OKLO_1min.parquet
│   │   │   │   │   ├── OMC/
│   │   │   │   │   │   ├── OMC_1d.parquet
│   │   │   │   │   │   ├── OMC_1min.parquet
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
│   │   │   │   │   │   ├── OTIS_1min.parquet
│   │   │   │   │   ├── OXY/
│   │   │   │   │   │   ├── OXY_1d.parquet
│   │   │   │   │   │   ├── OXY_1min.parquet
│   │   │   │   │   ├── PANW/
│   │   │   │   │   │   ├── PANW_1d.parquet
│   │   │   │   │   │   ├── PANW_1min.parquet
│   │   │   │   │   ├── PARA/
│   │   │   │   │   │   ├── PARA_1d.parquet
│   │   │   │   │   │   ├── PARA_1min.parquet
│   │   │   │   │   ├── PAYC/
│   │   │   │   │   │   ├── PAYC_1d.parquet
│   │   │   │   │   │   ├── PAYC_1min.parquet
│   │   │   │   │   ├── PAYX/
│   │   │   │   │   │   ├── PAYX_1d.parquet
│   │   │   │   │   │   ├── PAYX_1min.parquet
│   │   │   │   │   ├── PCAR/
│   │   │   │   │   │   ├── PCAR_1d.parquet
│   │   │   │   │   │   ├── PCAR_1min.parquet
│   │   │   │   │   ├── PCG/
│   │   │   │   │   │   ├── PCG_1d.parquet
│   │   │   │   │   │   ├── PCG_1min.parquet
│   │   │   │   │   ├── PEG/
│   │   │   │   │   │   ├── PEG_1d.parquet
│   │   │   │   │   │   ├── PEG_1min.parquet
│   │   │   │   │   ├── PEP/
│   │   │   │   │   │   ├── PEP_1d.parquet
│   │   │   │   │   │   ├── PEP_1min.parquet
│   │   │   │   │   ├── PFE/
│   │   │   │   │   │   ├── PFE_1d.parquet
│   │   │   │   │   │   ├── PFE_1min.parquet
│   │   │   │   │   ├── PFG/
│   │   │   │   │   │   ├── PFG_1d.parquet
│   │   │   │   │   │   ├── PFG_1min.parquet
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
│   │   │   │   │   │   ├── PHM_1min.parquet
│   │   │   │   │   ├── PKG/
│   │   │   │   │   │   ├── PKG_1d.parquet
│   │   │   │   │   │   ├── PKG_1min.parquet
│   │   │   │   │   ├── PLD/
│   │   │   │   │   │   ├── PLD_1d.parquet
│   │   │   │   │   │   ├── PLD_1min.parquet
│   │   │   │   │   ├── PLTR/
│   │   │   │   │   │   ├── PLTR_15min.parquet
│   │   │   │   │   │   ├── PLTR_1d.parquet
│   │   │   │   │   │   ├── PLTR_1day.parquet
│   │   │   │   │   │   ├── PLTR_1hour.parquet
│   │   │   │   │   │   ├── PLTR_1min.parquet
│   │   │   │   │   ├── PM/
│   │   │   │   │   │   ├── PM_1d.parquet
│   │   │   │   │   │   ├── PM_1min.parquet
│   │   │   │   │   ├── PNC/
│   │   │   │   │   │   ├── PNC_1d.parquet
│   │   │   │   │   │   ├── PNC_1min.parquet
│   │   │   │   │   ├── PNR/
│   │   │   │   │   │   ├── PNR_1d.parquet
│   │   │   │   │   │   ├── PNR_1min.parquet
│   │   │   │   │   ├── PNW/
│   │   │   │   │   │   ├── PNW_1d.parquet
│   │   │   │   │   │   ├── PNW_1min.parquet
│   │   │   │   │   ├── PODD/
│   │   │   │   │   │   ├── PODD_1d.parquet
│   │   │   │   │   │   ├── PODD_1min.parquet
│   │   │   │   │   ├── POOL/
│   │   │   │   │   │   ├── POOL_1d.parquet
│   │   │   │   │   │   ├── POOL_1min.parquet
│   │   │   │   │   ├── PPG/
│   │   │   │   │   │   ├── PPG_1d.parquet
│   │   │   │   │   │   ├── PPG_1min.parquet
│   │   │   │   │   ├── PPL/
│   │   │   │   │   │   ├── PPL_1d.parquet
│   │   │   │   │   │   ├── PPL_1min.parquet
│   │   │   │   │   ├── PRU/
│   │   │   │   │   │   ├── PRU_1d.parquet
│   │   │   │   │   │   ├── PRU_1min.parquet
│   │   │   │   │   ├── PSA/
│   │   │   │   │   │   ├── PSA_1d.parquet
│   │   │   │   │   │   ├── PSA_1min.parquet
│   │   │   │   │   ├── PSX/
│   │   │   │   │   │   ├── PSX_1d.parquet
│   │   │   │   │   │   ├── PSX_1min.parquet
│   │   │   │   │   ├── PTC/
│   │   │   │   │   │   ├── PTC_1d.parquet
│   │   │   │   │   │   ├── PTC_1min.parquet
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
│   │   │   │   │   │   ├── QRVO_1min.parquet
│   │   │   │   │   ├── RCL/
│   │   │   │   │   │   ├── RCL_1d.parquet
│   │   │   │   │   │   ├── RCL_1min.parquet
│   │   │   │   │   ├── REG/
│   │   │   │   │   │   ├── REG_1d.parquet
│   │   │   │   │   │   ├── REG_1min.parquet
│   │   │   │   │   ├── REGN/
│   │   │   │   │   │   ├── REGN_1d.parquet
│   │   │   │   │   ├── RF/
│   │   │   │   │   │   ├── RF_1d.parquet
│   │   │   │   │   │   ├── RF_1min.parquet
│   │   │   │   │   ├── RHI/
│   │   │   │   │   │   ├── RHI_1d.parquet
│   │   │   │   │   │   ├── RHI_1min.parquet
│   │   │   │   │   ├── RJF/
│   │   │   │   │   │   ├── RJF_1d.parquet
│   │   │   │   │   │   ├── RJF_1min.parquet
│   │   │   │   │   ├── RL/
│   │   │   │   │   │   ├── RL_1d.parquet
│   │   │   │   │   │   ├── RL_1min.parquet
│   │   │   │   │   ├── RMD/
│   │   │   │   │   │   ├── RMD_1d.parquet
│   │   │   │   │   │   ├── RMD_1min.parquet
│   │   │   │   │   ├── ROK/
│   │   │   │   │   │   ├── ROK_1d.parquet
│   │   │   │   │   │   ├── ROK_1min.parquet
│   │   │   │   │   ├── ROL/
│   │   │   │   │   │   ├── ROL_1d.parquet
│   │   │   │   │   │   ├── ROL_1min.parquet
│   │   │   │   │   ├── ROP/
│   │   │   │   │   │   ├── ROP_1d.parquet
│   │   │   │   │   ├── ROST/
│   │   │   │   │   │   ├── ROST_1d.parquet
│   │   │   │   │   │   ├── ROST_1min.parquet
│   │   │   │   │   ├── RSG/
│   │   │   │   │   │   ├── RSG_1d.parquet
│   │   │   │   │   │   ├── RSG_1min.parquet
│   │   │   │   │   ├── RTX/
│   │   │   │   │   │   ├── RTX_1d.parquet
│   │   │   │   │   │   ├── RTX_1min.parquet
│   │   │   │   │   ├── RVTY/
│   │   │   │   │   │   ├── RVTY_1d.parquet
│   │   │   │   │   │   ├── RVTY_1min.parquet
│   │   │   │   │   ├── SBAC/
│   │   │   │   │   │   ├── SBAC_1d.parquet
│   │   │   │   │   │   ├── SBAC_1min.parquet
│   │   │   │   │   ├── SBUX/
│   │   │   │   │   │   ├── SBUX_1d.parquet
│   │   │   │   │   │   ├── SBUX_1min.parquet
│   │   │   │   │   ├── SCHW/
│   │   │   │   │   │   ├── SCHW_1d.parquet
│   │   │   │   │   │   ├── SCHW_1min.parquet
│   │   │   │   │   ├── SHW/
│   │   │   │   │   │   ├── SHW_1d.parquet
│   │   │   │   │   │   ├── SHW_1min.parquet
│   │   │   │   │   ├── SJM/
│   │   │   │   │   │   ├── SJM_1d.parquet
│   │   │   │   │   │   ├── SJM_1min.parquet
│   │   │   │   │   ├── SLB/
│   │   │   │   │   │   ├── SLB_1d.parquet
│   │   │   │   │   │   ├── SLB_1min.parquet
│   │   │   │   │   ├── SMCI/
│   │   │   │   │   │   ├── SMCI_15min.parquet
│   │   │   │   │   │   ├── SMCI_1d.parquet
│   │   │   │   │   │   ├── SMCI_1min.parquet
│   │   │   │   │   ├── SNA/
│   │   │   │   │   │   ├── SNA_1d.parquet
│   │   │   │   │   │   ├── SNA_1min.parquet
│   │   │   │   │   ├── SNPS/
│   │   │   │   │   │   ├── SNPS_1d.parquet
│   │   │   │   │   │   ├── SNPS_1min.parquet
│   │   │   │   │   ├── SO/
│   │   │   │   │   │   ├── SO_1d.parquet
│   │   │   │   │   │   ├── SO_1min.parquet
│   │   │   │   │   ├── SOLV/
│   │   │   │   │   │   ├── SOLV_1d.parquet
│   │   │   │   │   │   ├── SOLV_1min.parquet
│   │   │   │   │   ├── SPG/
│   │   │   │   │   │   ├── SPG_1d.parquet
│   │   │   │   │   │   ├── SPG_1min.parquet
│   │   │   │   │   ├── SPGI/
│   │   │   │   │   │   ├── SPGI_1d.parquet
│   │   │   │   │   ├── SPY/
│   │   │   │   │   │   ├── SPY_5min.parquet
│   │   │   │   │   ├── SRE/
│   │   │   │   │   │   ├── SRE_1d.parquet
│   │   │   │   │   │   ├── SRE_1min.parquet
│   │   │   │   │   ├── STE/
│   │   │   │   │   │   ├── STE_1d.parquet
│   │   │   │   │   │   ├── STE_1min.parquet
│   │   │   │   │   ├── STLD/
│   │   │   │   │   │   ├── STLD_1d.parquet
│   │   │   │   │   │   ├── STLD_1min.parquet
│   │   │   │   │   ├── STT/
│   │   │   │   │   │   ├── STT_1d.parquet
│   │   │   │   │   │   ├── STT_1min.parquet
│   │   │   │   │   ├── STX/
│   │   │   │   │   │   ├── STX_1d.parquet
│   │   │   │   │   │   ├── STX_1min.parquet
│   │   │   │   │   ├── STZ/
│   │   │   │   │   │   ├── STZ_1d.parquet
│   │   │   │   │   │   ├── STZ_1min.parquet
│   │   │   │   │   ├── SWK/
│   │   │   │   │   │   ├── SWK_1d.parquet
│   │   │   │   │   │   ├── SWK_1min.parquet
│   │   │   │   │   ├── SWKS/
│   │   │   │   │   │   ├── SWKS_1d.parquet
│   │   │   │   │   │   ├── SWKS_1min.parquet
│   │   │   │   │   ├── SYF/
│   │   │   │   │   │   ├── SYF_1d.parquet
│   │   │   │   │   │   ├── SYF_1min.parquet
│   │   │   │   │   ├── SYK/
│   │   │   │   │   │   ├── SYK_1d.parquet
│   │   │   │   │   │   ├── SYK_1min.parquet
│   │   │   │   │   ├── SYY/
│   │   │   │   │   │   ├── SYY_1d.parquet
│   │   │   │   │   │   ├── SYY_1min.parquet
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
│   │   │   │   │   │   ├── TECH_1min.parquet
│   │   │   │   │   ├── TEL/
│   │   │   │   │   │   ├── TEL_1d.parquet
│   │   │   │   │   │   ├── TEL_1min.parquet
│   │   │   │   │   ├── TER/
│   │   │   │   │   │   ├── TER_15min.parquet
│   │   │   │   │   │   ├── TER_1d.parquet
│   │   │   │   │   │   ├── TER_1min.parquet
│   │   │   │   │   ├── TFC/
│   │   │   │   │   │   ├── TFC_1d.parquet
│   │   │   │   │   │   ├── TFC_1min.parquet
│   │   │   │   │   ├── TFX/
│   │   │   │   │   │   ├── TFX_1d.parquet
│   │   │   │   │   │   ├── TFX_1min.parquet
│   │   │   │   │   ├── TGT/
│   │   │   │   │   │   ├── TGT_1d.parquet
│   │   │   │   │   │   ├── TGT_1min.parquet
│   │   │   │   │   ├── TJX/
│   │   │   │   │   │   ├── TJX_1d.parquet
│   │   │   │   │   │   ├── TJX_1min.parquet
│   │   │   │   │   ├── TMO/
│   │   │   │   │   │   ├── TMO_1d.parquet
│   │   │   │   │   │   ├── TMO_1min.parquet
│   │   │   │   │   ├── TMUS/
│   │   │   │   │   │   ├── TMUS_1d.parquet
│   │   │   │   │   │   ├── TMUS_1min.parquet
│   │   │   │   │   ├── TPL/
│   │   │   │   │   │   ├── TPL_1d.parquet
│   │   │   │   │   ├── TPR/
│   │   │   │   │   │   ├── TPR_1d.parquet
│   │   │   │   │   │   ├── TPR_1min.parquet
│   │   │   │   │   ├── TRGP/
│   │   │   │   │   │   ├── TRGP_1d.parquet
│   │   │   │   │   │   ├── TRGP_1min.parquet
│   │   │   │   │   ├── TRMB/
│   │   │   │   │   │   ├── TRMB_1d.parquet
│   │   │   │   │   │   ├── TRMB_1min.parquet
│   │   │   │   │   ├── TROW/
│   │   │   │   │   │   ├── TROW_1d.parquet
│   │   │   │   │   │   ├── TROW_1min.parquet
│   │   │   │   │   ├── TRV/
│   │   │   │   │   │   ├── TRV_1d.parquet
│   │   │   │   │   │   ├── TRV_1min.parquet
│   │   │   │   │   ├── TSCO/
│   │   │   │   │   │   ├── TSCO_1d.parquet
│   │   │   │   │   │   ├── TSCO_1min.parquet
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
│   │   │   │   │   │   ├── TT_1min.parquet
│   │   │   │   │   ├── TTWO/
│   │   │   │   │   │   ├── TTWO_1d.parquet
│   │   │   │   │   │   ├── TTWO_1min.parquet
│   │   │   │   │   ├── TXN/
│   │   │   │   │   │   ├── TXN_1d.parquet
│   │   │   │   │   │   ├── TXN_1min.parquet
│   │   │   │   │   ├── TXT/
│   │   │   │   │   │   ├── TXT_1d.parquet
│   │   │   │   │   │   ├── TXT_1min.parquet
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
│   │   │   │   │   │   ├── UDR_1min.parquet
│   │   │   │   │   ├── UHS/
│   │   │   │   │   │   ├── UHS_1d.parquet
│   │   │   │   │   │   ├── UHS_1min.parquet
│   │   │   │   │   ├── ULTA/
│   │   │   │   │   │   ├── ULTA_1d.parquet
│   │   │   │   │   │   ├── ULTA_1min.parquet
│   │   │   │   │   ├── UNH/
│   │   │   │   │   │   ├── UNH_1d.parquet
│   │   │   │   │   │   ├── UNH_1min.parquet
│   │   │   │   │   ├── UNP/
│   │   │   │   │   │   ├── UNP_1d.parquet
│   │   │   │   │   │   ├── UNP_1min.parquet
│   │   │   │   │   ├── UPS/
│   │   │   │   │   │   ├── UPS_1d.parquet
│   │   │   │   │   │   ├── UPS_1min.parquet
│   │   │   │   │   ├── URI/
│   │   │   │   │   │   ├── URI_1d.parquet
│   │   │   │   │   ├── USB/
│   │   │   │   │   │   ├── USB_1d.parquet
│   │   │   │   │   │   ├── USB_1min.parquet
│   │   │   │   │   ├── V/
│   │   │   │   │   │   ├── V_1d.parquet
│   │   │   │   │   │   ├── V_1min.parquet
│   │   │   │   │   ├── VICI/
│   │   │   │   │   │   ├── VICI_1d.parquet
│   │   │   │   │   │   ├── VICI_1min.parquet
│   │   │   │   │   ├── VLO/
│   │   │   │   │   │   ├── VLO_1d.parquet
│   │   │   │   │   │   ├── VLO_1min.parquet
│   │   │   │   │   ├── VLTO/
│   │   │   │   │   │   ├── VLTO_1d.parquet
│   │   │   │   │   │   ├── VLTO_1min.parquet
│   │   │   │   │   ├── VMC/
│   │   │   │   │   │   ├── VMC_1d.parquet
│   │   │   │   │   │   ├── VMC_1min.parquet
│   │   │   │   │   ├── VRSK/
│   │   │   │   │   │   ├── VRSK_1d.parquet
│   │   │   │   │   │   ├── VRSK_1min.parquet
│   │   │   │   │   ├── VRSN/
│   │   │   │   │   │   ├── VRSN_1d.parquet
│   │   │   │   │   │   ├── VRSN_1min.parquet
│   │   │   │   │   ├── VRTX/
│   │   │   │   │   │   ├── VRTX_1d.parquet
│   │   │   │   │   │   ├── VRTX_1min.parquet
│   │   │   │   │   ├── VST/
│   │   │   │   │   │   ├── VST_15min.parquet
│   │   │   │   │   │   ├── VST_1d.parquet
│   │   │   │   │   │   ├── VST_1min.parquet
│   │   │   │   │   ├── VTR/
│   │   │   │   │   │   ├── VTR_1d.parquet
│   │   │   │   │   │   ├── VTR_1min.parquet
│   │   │   │   │   ├── VTRS/
│   │   │   │   │   │   ├── VTRS_1d.parquet
│   │   │   │   │   │   ├── VTRS_1min.parquet
│   │   │   │   │   ├── VZ/
│   │   │   │   │   │   ├── VZ_1d.parquet
│   │   │   │   │   │   ├── VZ_1min.parquet
│   │   │   │   │   ├── WAB/
│   │   │   │   │   │   ├── WAB_1d.parquet
│   │   │   │   │   │   ├── WAB_1min.parquet
│   │   │   │   │   ├── WAT/
│   │   │   │   │   │   ├── WAT_1d.parquet
│   │   │   │   │   │   ├── WAT_1min.parquet
│   │   │   │   │   ├── WBA/
│   │   │   │   │   │   ├── WBA_1d.parquet
│   │   │   │   │   │   ├── WBA_1min.parquet
│   │   │   │   │   ├── WBD/
│   │   │   │   │   │   ├── WBD_1d.parquet
│   │   │   │   │   │   ├── WBD_1min.parquet
│   │   │   │   │   ├── WDAY/
│   │   │   │   │   │   ├── WDAY_1d.parquet
│   │   │   │   │   │   ├── WDAY_1min.parquet
│   │   │   │   │   ├── WDC/
│   │   │   │   │   │   ├── WDC_1d.parquet
│   │   │   │   │   │   ├── WDC_1min.parquet
│   │   │   │   │   ├── WEC/
│   │   │   │   │   │   ├── WEC_1d.parquet
│   │   │   │   │   │   ├── WEC_1min.parquet
│   │   │   │   │   ├── WELL/
│   │   │   │   │   │   ├── WELL_1d.parquet
│   │   │   │   │   │   ├── WELL_1min.parquet
│   │   │   │   │   ├── WFC/
│   │   │   │   │   │   ├── WFC_1d.parquet
│   │   │   │   │   │   ├── WFC_1min.parquet
│   │   │   │   │   ├── WM/
│   │   │   │   │   │   ├── WM_1d.parquet
│   │   │   │   │   │   ├── WM_1min.parquet
│   │   │   │   │   ├── WMB/
│   │   │   │   │   │   ├── WMB_1d.parquet
│   │   │   │   │   │   ├── WMB_1min.parquet
│   │   │   │   │   ├── WMT/
│   │   │   │   │   │   ├── WMT_1d.parquet
│   │   │   │   │   │   ├── WMT_1min.parquet
│   │   │   │   │   ├── WRB/
│   │   │   │   │   │   ├── WRB_1d.parquet
│   │   │   │   │   │   ├── WRB_1min.parquet
│   │   │   │   │   ├── WST/
│   │   │   │   │   │   ├── WST_1d.parquet
│   │   │   │   │   │   ├── WST_1min.parquet
│   │   │   │   │   ├── WTW/
│   │   │   │   │   │   ├── WTW_1d.parquet
│   │   │   │   │   │   ├── WTW_1min.parquet
│   │   │   │   │   ├── WY/
│   │   │   │   │   │   ├── WY_1d.parquet
│   │   │   │   │   │   ├── WY_1min.parquet
│   │   │   │   │   ├── WYNN/
│   │   │   │   │   │   ├── WYNN_1d.parquet
│   │   │   │   │   │   ├── WYNN_1min.parquet
│   │   │   │   │   ├── XEL/
│   │   │   │   │   │   ├── XEL_1d.parquet
│   │   │   │   │   │   ├── XEL_1min.parquet
│   │   │   │   │   ├── XOM/
│   │   │   │   │   │   ├── XOM_1d.parquet
│   │   │   │   │   │   ├── XOM_1min.parquet
│   │   │   │   │   ├── XRAY/
│   │   │   │   │   │   ├── XRAY_1d.parquet
│   │   │   │   │   │   ├── XRAY_1min.parquet
│   │   │   │   │   ├── XYL/
│   │   │   │   │   │   ├── XYL_1d.parquet
│   │   │   │   │   │   ├── XYL_1min.parquet
│   │   │   │   │   ├── YUM/
│   │   │   │   │   │   ├── YUM_1d.parquet
│   │   │   │   │   │   ├── YUM_1min.parquet
│   │   │   │   │   ├── ZBH/
│   │   │   │   │   │   ├── ZBH_1d.parquet
│   │   │   │   │   │   ├── ZBH_1min.parquet
│   │   │   │   │   ├── ZBRA/
│   │   │   │   │   │   ├── ZBRA_1d.parquet
│   │   │   │   │   │   ├── ZBRA_1min.parquet
│   │   │   │   │   ├── ZION/
│   │   │   │   │   │   ├── ZION_1d.parquet
│   │   │   │   │   │   ├── ZION_1min.parquet
│   │   │   │   │   ├── ZTS/
│   │   │   │   │   │   ├── ZTS_1d.parquet
│   │   │   │   │   │   ├── ZTS_1min.parquet
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
│   ├── test_cache/
│   │   ├── cache_metadata.json
│   │   ├── aapl/
│   │   │   ├── bars/
│   │   │   │   ├── 1min_20250115_518cd58d.parquet
```
