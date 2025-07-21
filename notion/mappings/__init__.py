# mappings/__init__.py
import os
import importlib
import json

def load_mapping(table_name):
    """Dynamically load mapping for a given table name"""
    try:
        # Convert table_name to module name (e.g., 'premarket_scans' -> 'premarket_scans_map')
        module_name = f"{table_name}_map"
        
        # Try to import the module
        module = importlib.import_module(f"mappings.{module_name}")
        
        # Get the FIELD_MAPPING from the module
        if hasattr(module, 'FIELD_MAPPING'):
            print(f"  ✅ Loaded mapping from mappings/{module_name}.py")
            return module.FIELD_MAPPING
        else:
            print(f"  ⚠️ No FIELD_MAPPING found in {module_name}.py")
            return None
            
    except ImportError:
        print(f"  ℹ️ No mapping file found for {table_name}")
        return None

def load_mapping_from_json(config):
    """Try to load mapping from JSON in config first"""
    if config.get('field_mappings') and config['field_mappings'].get('supabase_to_notion'):
        print("  ✅ Using custom mapping from configuration database")
        return config['field_mappings']
    return None

def get_mapping_for_config(config):
    """Get mapping with fallback hierarchy:
    1. Custom JSON mapping from config database
    2. Python mapping file
    3. Empty default
    """
    # First try custom mapping from config
    mapping = load_mapping_from_json(config)
    if mapping:
        return mapping
    
    # Then try loading from Python file
    table_name = config.get('supabase_table_name')
    if table_name:
        mapping = load_mapping(table_name)
        if mapping:
            return mapping
    
    # Finally return empty default
    print("  ⚠️ Using empty default mapping")
    return {
        'notion_to_supabase': {},
        'supabase_to_notion': {},
        'id_field': 'supabase_id'
    }