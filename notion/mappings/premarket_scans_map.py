# mappings/premarket_scans_map.py

FIELD_MAPPING = {
    'notion_to_supabase': {
        'Ticker': 'ticker',
        'Interest Score': 'interest_score',
        'Scan Date': 'scan_date',
        'Active': 'active',
        'ATR': 'atr',
        'ATR Percent': 'atr_percent',
        'ATR Percent Score': 'atr_percent_score',
        'Avg Daily Volume': 'avg_daily_volume',
        'Dollar Vol Score': 'dollar_vol_score',
        'Dollar Volume': 'dollar_volume',
        'Market Session': 'market_session',
        'PM Vol Abs Score': 'pm_vol_abs_score',
        'PM Vol Ratio Score': 'pm_vol_ratio_score',
        'Passed Filters': 'passed_filters',
        'Premarket Volume': 'premarket_volume',
        'Price': 'price',
        'Price ATR Bonus': 'price_atr_bonus',
        'Rank': 'rank',
        'Scan Time': 'scan_time',
        'Created At': 'created_at'
    },
    'supabase_to_notion': {
        'ticker': 'Ticker',
        'interest_score': 'Interest Score',
        'scan_date': 'Scan Date',
        'active': 'Active',
        'atr': 'ATR',
        'atr_percent': 'ATR Percent',
        'atr_percent_score': 'ATR Percent Score',
        'avg_daily_volume': 'Avg Daily Volume',
        'dollar_vol_score': 'Dollar Vol Score',
        'dollar_volume': 'Dollar Volume',
        'market_session': 'Market Session',
        'pm_vol_abs_score': 'PM Vol Abs Score',
        'pm_vol_ratio_score': 'PM Vol Ratio Score',
        'passed_filters': 'Passed Filters',
        'premarket_volume': 'Premarket Volume',
        'price': 'Price',
        'price_atr_bonus': 'Price ATR Bonus',
        'rank': 'Rank',
        'scan_time': 'Scan Time',
        'created_at': 'Created At',
        'id': 'supabase_id'  # Maps Supabase ID to the tracking field
    },
    'id_field': 'supabase_id',
    'field_types': {
        'Ticker': 'title'  # Ticker is the title field in Notion
    }
}