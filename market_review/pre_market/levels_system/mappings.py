# mappings.py

def get_mapping_for_config(config):
    """Return field mappings based on configuration name"""
    
    if 'pre market levels' in config['name'].lower() or 'premarket' in config['name'].lower():
        return {
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
                'created_at': 'Created At'  # Optional
            },
            'id_field': 'supabase_id',
            'field_types': {
                'Ticker': 'title'  # This ensures Ticker is treated as the title field
            }
        }
    
    elif 'ranked levels' in config['name'].lower() or 'ranked' in config['name'].lower():
        return {
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
                # Ranked level specific fields
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
                'created_at': 'Created At'
            },
            'id_field': 'supabase_id',
            'field_types': {
                'Ticker': 'title'
            }
        }
    
    # Default mapping if no match
    return {
        'notion_to_supabase': {},
        'supabase_to_notion': {},
        'id_field': 'supabase_id'
    }