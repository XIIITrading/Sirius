# mappings/premarket_levels_map.py

FIELD_MAPPING = {
    'notion_to_supabase': {
        'Date': 'date',
        'Ticker': 'ticker',
        'Level Type': 'level_type',
        'Position': 'position',
        'Price': 'price',
        'Strength Score': 'strength_score',
        'Notes': 'notes',
        'Active': 'active'
    },
    'supabase_to_notion': {
        'date': 'Date',
        'ticker': 'Ticker',
        'level_type': 'Level Type',
        'position': 'Position',
        'price': 'Price',
        'strength_score': 'Strength Score',
        'notes': 'Notes',
        'active': 'Active',
        'created_at': 'Created At',
        'id': 'supabase_id'  # Maps the Supabase ID to your tracking field
    },
    'id_field': 'supabase_id',
    'field_types': {
        'Ticker': 'title'  # This ensures Ticker is treated as the title field
    }
}