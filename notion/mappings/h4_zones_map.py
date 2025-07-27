# mappings/h4_zones_map.py

FIELD_MAPPING = {
    'notion_to_supabase': {
        'Ticker': 'ticker',
        'Date': 'date',
        'H4S4': 'h4s4',
        'H4S3': 'h4s3',
        'H4S2': 'h4s2',
        'H4S1': 'h4s1',
        'H4R1': 'h4r1',
        'H4R2': 'h4r2',
        'H4R3': 'h4r3',
        'H4R4': 'h4r4',
        'Supabase ID': 'supabase_id'
    },
    'supabase_to_notion': {
        'ticker': 'Ticker',
        'date': 'Date',
        'h4s4': 'H4S4',
        'h4s3': 'H4S3',
        'h4s2': 'H4S2',
        'h4s1': 'H4S1',
        'h4r1': 'H4R1',
        'h4r2': 'H4R2',
        'h4r3': 'H4R3',
        'h4r4': 'H4R4',
        'id': 'Supabase ID'  # Maps the Supabase ID to your tracking field
    },
    'id_field': 'Supabase ID',
    'field_types': {
        'Ticker': 'title',  # This ensures Ticker is treated as the title field
        'Supabase ID': 'number'
    }
}