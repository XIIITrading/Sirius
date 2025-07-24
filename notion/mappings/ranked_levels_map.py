# mappings/ranked_levels_map.py

FIELD_MAPPING = {
    'notion_to_supabase': {
        'Pre-Market Level ID': 'premarket_level_id',
        'Date': 'date',
        'Ticker': 'ticker',
        'Level Type': 'level_type',
        'Price': 'price',
        'Notes': 'notes',
        'Active': 'active',
        'Rank': 'rank',
        'Confluence Score': 'confluence_score',
        'Zone High': 'zone_high',
        'Zone Low': 'zone_low',
        'TV Variable': 'tv_variable',
        'Current Price': 'current_price',
        'ATR Value': 'atr_value',
        'Strength Score': 'strength_score'
    },
    'supabase_to_notion': {
        'premarket_level_id': 'Pre-Market Level ID',
        'date': 'Date',
        'ticker': 'Ticker',
        'rank': 'Rank',
        'confluence_score': 'Confluence Score',
        'zone_high': 'Zone High',
        'zone_low': 'Zone Low',
        'tv_variable': 'TV Variable',
        'current_price': 'Current Price',
        'atr_value': 'ATR Value',
        'strength_score': 'Strength Score',
        'id': 'Supabase ID'
    },
    'id_field': 'Supabase ID',
    'field_types': {
        'Ticker': 'title',
        'Supabase ID': 'number'
    }
}