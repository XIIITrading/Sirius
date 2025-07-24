# mappings/premarket_levels_map.py

FIELD_MAPPING = {
    'notion_to_supabase': {
        'Date': 'date',
        'Ticker': 'ticker',
        'Level Type': 'level_type',
        'Price': 'price',
        'Notes': 'notes',
        'Active': 'active'
    },
    'supabase_to_notion': {
        'date': 'Date',
        'ticker': 'Ticker',
        'level_type': 'Level Type',
        'price': 'Price',
        'notes': 'Notes',
        'active': 'Active',
        'created_at': 'Created At',
        'id': 'Supabase ID'  # Maps the Supabase ID to your tracking field
    },
    'id_field': 'Supabase ID',
    'field_types': {
        'Ticker': 'title',  # This ensures Ticker is treated as the title field
        'Supabase ID': 'number'
    }
}