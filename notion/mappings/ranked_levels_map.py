# mappings/ranked_levels_map.py

FIELD_MAPPING = {
    'notion_to_supabase': {
        'Pre-Market Level ID': 'premarket_level_id',
        'Date': 'date',
        'Ticker': 'ticker',
        'Level Type': 'level_type',
        'Position': 'position',
        'Price': 'price',
        'Strength Score': 'strength_score',
        'Notes': 'notes',
        'Active': 'active',
        'Rank': 'rank',
        'Confluence Score': 'confluence_score',
        'Zone High': 'zone_high',
        'Zone Low': 'zone_low',
        'TV Variable': 'tv_variable',
        'Current Price': 'current_price',
        'ATR Value': 'atr_value'
    },
    'supabase_to_notion': {
        'premarket_level_id': 'Pre-Market Level ID',
        'date': 'Date',
        'ticker': 'Ticker',
        'level_type': 'Level Type',
        'position': 'Position',
        'price': 'Price',
        'strength_score': 'Strength Score',
        'notes': 'Notes',
        'active': 'Active',
        'rank': 'Rank',
        'confluence_score': 'Confluence Score',
        'zone_high': 'Zone High',
        'zone_low': 'Zone Low',
        'tv_variable': 'TV Variable',
        'current_price': 'Current Price',
        'atr_value': 'ATR Value',
        'created_at': 'Created At',
        'id': 'supabase_id'  # Maps the Supabase ID to your tracking field
    },
    'id_field': 'supabase_id',
    'field_types': {
        'Ticker': 'title'  # This ensures Ticker is treated as the title field
    }
}