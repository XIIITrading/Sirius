# notion/validate_sync_config.py

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from notion_client import Client as NotionClient
from typing import Dict, List, Tuple, Any
import traceback

load_dotenv()

class SyncConfigValidator:
    """Validates sync configurations across Supabase, Notion, and mappings"""
    
    def __init__(self):
        self.supabase = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_KEY')
        )
        self.notion = NotionClient(auth=os.getenv('NOTION_TOKEN'))
        self.config_database_id = os.getenv('SYNC_CONFIG_DATABASE_ID')
        
    def get_sync_configurations(self):
        """Fetch all sync configurations (both active and inactive)"""
        configs = []
        
        try:
            response = self.notion.databases.query(
                database_id=self.config_database_id
            )
            
            for page in response['results']:
                props = page['properties']
                
                config = {
                    'config_page_id': page['id'],
                    'name': props['Name']['title'][0]['plain_text'] if props['Name']['title'] else 'Unnamed',
                    'notion_database_id': props['Notion_Database_ID']['rich_text'][0]['plain_text'] if props['Notion_Database_ID']['rich_text'] else None,
                    'supabase_table_name': props['Supabase_Table_Name']['rich_text'][0]['plain_text'] if props['Supabase_Table_Name']['rich_text'] else None,
                    'active': props['Active']['checkbox'] if 'Active' in props else False,
                    'field_mappings': {}
                }
                
                # Parse field mappings
                if props.get('Field_Mappings', {}).get('rich_text'):
                    try:
                        mappings_json = props['Field_Mappings']['rich_text'][0]['plain_text']
                        config['field_mappings'] = json.loads(mappings_json)
                    except:
                        pass
                
                if config['notion_database_id'] and config['supabase_table_name']:
                    configs.append(config)
                    
        except Exception as e:
            print(f"Error fetching sync configurations: {e}")
            
        return configs
    
    def get_supabase_schema(self, table_name: str) -> Dict[str, Dict]:
        """Get schema information for a Supabase table"""
        schema = {}
        
        try:
            # Query the information_schema to get column details
            query = f"""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale
                FROM information_schema.columns
                WHERE table_schema = 'public' 
                AND table_name = '{table_name}'
                ORDER BY ordinal_position;
            """
            
            # Execute raw SQL query
            result = self.supabase.rpc('get_table_schema', {
                'table_name_param': table_name
            }).execute()
            
            # If the RPC doesn't exist, try a different approach
            if not result.data:
                # Fallback: get a sample row to infer schema
                sample = self.supabase.table(table_name).select("*").limit(1).execute()
                if sample.data:
                    for key, value in sample.data[0].items():
                        schema[key] = {
                            'type': type(value).__name__,
                            'nullable': value is None,
                            'inferred': True
                        }
                else:
                    # Try to insert and rollback to get column info from error
                    try:
                        self.supabase.table(table_name).insert({}).execute()
                    except Exception as e:
                        # Parse error message for column information
                        error_str = str(e)
                        if "null value in column" in error_str:
                            # Extract column names from error
                            pass
            else:
                # Parse the schema information
                for col in result.data:
                    schema[col['column_name']] = {
                        'type': col['data_type'],
                        'nullable': col['is_nullable'] == 'YES',
                        'default': col['column_default'],
                        'max_length': col['character_maximum_length'],
                        'precision': col['numeric_precision'],
                        'scale': col['numeric_scale']
                    }
                    
        except Exception as e:
            print(f"Error getting Supabase schema: {e}")
            # Last resort: try to get columns from a select query
            try:
                result = self.supabase.table(table_name).select("*").limit(0).execute()
                # Even with 0 rows, we might get column names from the response
            except:
                pass
                
        return schema
    
    def get_notion_schema(self, database_id: str) -> Dict[str, Dict]:
        """Get schema information for a Notion database"""
        schema = {}
        
        try:
            # Get database details including properties
            database = self.notion.databases.retrieve(database_id)
            
            for prop_name, prop_info in database['properties'].items():
                schema[prop_name] = {
                    'type': prop_info['type'],
                    'id': prop_info['id']
                }
                
                # Add additional type-specific information
                if prop_info['type'] == 'select' and 'select' in prop_info:
                    schema[prop_name]['options'] = [opt['name'] for opt in prop_info['select']['options']]
                elif prop_info['type'] == 'multi_select' and 'multi_select' in prop_info:
                    schema[prop_name]['options'] = [opt['name'] for opt in prop_info['multi_select']['options']]
                elif prop_info['type'] == 'relation' and 'relation' in prop_info:
                    schema[prop_name]['database_id'] = prop_info['relation']['database_id']
                    
        except Exception as e:
            print(f"Error getting Notion schema: {e}")
            
        return schema
    
    def get_field_mapping(self, config):
        """Get field mapping configuration"""
        try:
            from mappings import get_mapping_for_config
            return get_mapping_for_config(config)
        except ImportError:
            if config.get('field_mappings'):
                return config['field_mappings']
            return {
                'notion_to_supabase': {},
                'supabase_to_notion': {},
                'id_field': 'supabase_id'
            }
    
    def map_supabase_to_notion_type(self, supabase_type: str) -> List[str]:
        """Map Supabase data types to compatible Notion property types"""
        type_mapping = {
            # Numeric types
            'integer': ['number'],
            'bigint': ['number'],
            'smallint': ['number'],
            'decimal': ['number'],
            'numeric': ['number'],
            'real': ['number'],
            'double precision': ['number'],
            
            # Text types
            'character varying': ['title', 'rich_text', 'select', 'multi_select'],
            'varchar': ['title', 'rich_text', 'select', 'multi_select'],
            'text': ['title', 'rich_text'],
            'char': ['title', 'rich_text', 'select'],
            
            # Boolean
            'boolean': ['checkbox'],
            'bool': ['checkbox'],
            
            # Date/Time
            'timestamp': ['date'],
            'timestamp with time zone': ['date'],
            'timestamp without time zone': ['date'],
            'date': ['date'],
            'time': ['rich_text'],  # Notion doesn't have time-only
            
            # JSON
            'json': ['rich_text'],
            'jsonb': ['rich_text'],
            
            # Other
            'uuid': ['rich_text'],
            'serial': ['number'],
            'bigserial': ['number']
        }
        
        supabase_type_lower = supabase_type.lower()
        
        # Check for array types
        if '[]' in supabase_type_lower:
            return ['multi_select', 'rich_text']
            
        return type_mapping.get(supabase_type_lower, ['rich_text'])
    
    def validate_single_config(self, config: Dict) -> Dict[str, Any]:
        """Validate a single sync configuration"""
        validation_result = {
            'name': config['name'],
            'active': config['active'],
            'errors': [],
            'warnings': [],
            'info': [],
            'supabase_columns': {},
            'notion_properties': {},
            'mapping_fields': {},
            'missing_in_notion': [],
            'missing_in_supabase': [],
            'missing_in_mapping': [],
            'type_mismatches': []
        }
        
        print(f"\n{'='*60}")
        print(f"🔍 Validating: {config['name']}")
        print(f"   Active: {'✅' if config['active'] else '❌'}")
        
        # 1. Get Supabase schema
        print(f"\n📊 Checking Supabase table: {config['supabase_table_name']}")
        supabase_schema = self.get_supabase_schema(config['supabase_table_name'])
        
        if not supabase_schema:
            validation_result['errors'].append(f"Could not retrieve Supabase schema for table '{config['supabase_table_name']}'")
            print(f"   ❌ Could not retrieve schema")
        else:
            validation_result['supabase_columns'] = supabase_schema
            print(f"   ✅ Found {len(supabase_schema)} columns")
            for col_name, col_info in supabase_schema.items():
                print(f"      - {col_name}: {col_info.get('type', 'unknown')} {'(nullable)' if col_info.get('nullable') else '(required)'}")
        
        # 2. Get Notion schema
        print(f"\n📝 Checking Notion database: {config['notion_database_id']}")
        notion_schema = self.get_notion_schema(config['notion_database_id'])
        
        if not notion_schema:
            validation_result['errors'].append(f"Could not retrieve Notion schema for database '{config['notion_database_id']}'")
            print(f"   ❌ Could not retrieve schema")
        else:
            validation_result['notion_properties'] = notion_schema
            print(f"   ✅ Found {len(notion_schema)} properties")
            for prop_name, prop_info in notion_schema.items():
                print(f"      - {prop_name}: {prop_info['type']}")
        
        # 3. Get field mapping
        print(f"\n🔗 Checking field mappings")
        field_mapping = self.get_field_mapping(config)
        validation_result['mapping_fields'] = field_mapping
        
        notion_to_supabase = field_mapping.get('notion_to_supabase', {})
        supabase_to_notion = field_mapping.get('supabase_to_notion', {})
        
        print(f"   📤 Notion → Supabase: {len(notion_to_supabase)} mappings")
        print(f"   📥 Supabase → Notion: {len(supabase_to_notion)} mappings")
        
        # 4. Validate mappings
        print(f"\n✅ Validating consistency...")
        
        # Check Notion → Supabase mappings
        for notion_field, supabase_field in notion_to_supabase.items():
            # Check if Notion field exists
            if notion_field not in notion_schema:
                validation_result['errors'].append(f"Notion field '{notion_field}' in mapping does not exist in Notion database")
                print(f"   ❌ Notion field '{notion_field}' not found in database")
            
            # Check if Supabase field exists
            if supabase_field not in supabase_schema and supabase_field not in ['id', 'created_at', 'updated_at']:
                validation_result['warnings'].append(f"Supabase field '{supabase_field}' in mapping does not exist in table")
                print(f"   ⚠️  Supabase column '{supabase_field}' not found in table")
            
            # Check type compatibility
            if notion_field in notion_schema and supabase_field in supabase_schema:
                notion_type = notion_schema[notion_field]['type']
                supabase_type = supabase_schema[supabase_field].get('type', 'unknown')
                compatible_types = self.map_supabase_to_notion_type(supabase_type)
                
                if notion_type not in compatible_types:
                    validation_result['type_mismatches'].append({
                        'notion_field': notion_field,
                        'notion_type': notion_type,
                        'supabase_field': supabase_field,
                        'supabase_type': supabase_type,
                        'compatible_types': compatible_types
                    })
                    print(f"   ⚠️  Type mismatch: {notion_field} ({notion_type}) ↔ {supabase_field} ({supabase_type})")
        
        # Check for required Supabase fields not in mapping
        for col_name, col_info in supabase_schema.items():
            if not col_info.get('nullable') and col_info.get('default') is None:
                if col_name not in ['id', 'created_at', 'updated_at'] and col_name not in supabase_to_notion.keys():
                    validation_result['warnings'].append(f"Required Supabase column '{col_name}' is not mapped")
                    print(f"   ⚠️  Required column '{col_name}' is not mapped")
        
        # Check for special fields
        id_field = field_mapping.get('id_field', 'supabase_id')
        if id_field not in notion_schema:
            validation_result['errors'].append(f"ID tracking field '{id_field}' not found in Notion database")
            print(f"   ❌ ID tracking field '{id_field}' not found in Notion")
        
        # Summary
        print(f"\n📊 Validation Summary:")
        print(f"   Errors: {len(validation_result['errors'])}")
        print(f"   Warnings: {len(validation_result['warnings'])}")
        print(f"   Type Mismatches: {len(validation_result['type_mismatches'])}")
        
        return validation_result
    
    def create_rpc_function(self):
        """SQL to create the RPC function for getting table schema"""
        return """
        -- Run this in your Supabase SQL editor to enable schema introspection
        
        CREATE OR REPLACE FUNCTION get_table_schema(table_name_param text)
        RETURNS TABLE (
            column_name text,
            data_type text,
            is_nullable text,
            column_default text,
            character_maximum_length integer,
            numeric_precision integer,
            numeric_scale integer
        )
        LANGUAGE plpgsql
        SECURITY DEFINER
        AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                c.column_name::text,
                c.data_type::text,
                c.is_nullable::text,
                c.column_default::text,
                c.character_maximum_length::integer,
                c.numeric_precision::integer,
                c.numeric_scale::integer
            FROM information_schema.columns c
            WHERE c.table_schema = 'public' 
            AND c.table_name = table_name_param
            ORDER BY c.ordinal_position;
        END;
        $$;
        """
    
    def generate_fix_suggestions(self, validation_results: List[Dict]) -> None:
        """Generate suggestions for fixing validation issues"""
        print(f"\n{'='*60}")
        print("🔧 FIX SUGGESTIONS")
        print(f"{'='*60}")
        
        for result in validation_results:
            if result['errors'] or result['warnings'] or result['type_mismatches']:
                print(f"\n📋 {result['name']}:")
                
                # Type mismatch fixes
                if result['type_mismatches']:
                    print(f"\n   Type Mismatches:")
                    for mismatch in result['type_mismatches']:
                        print(f"   - {mismatch['notion_field']} ({mismatch['notion_type']}) ↔ {mismatch['supabase_field']} ({mismatch['supabase_type']})")
                        print(f"     Suggestion: Change Notion property type to one of: {', '.join(mismatch['compatible_types'])}")
                
                # Missing ID field
                if any("ID tracking field" in err for err in result['errors']):
                    print(f"\n   Missing ID Field:")
                    print(f"   - Add a 'supabase_id' property to your Notion database")
                    print(f"   - Property type: rich_text")
    
    def validate_all(self, specific_name: str = None) -> List[Dict]:
        """Validate all configurations or a specific one"""
        print("\n🔍 Starting Sync Configuration Validation...")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        configs = self.get_sync_configurations()
        
        if not configs:
            print("\n❌ No sync configurations found.")
            return []
        
        # Filter for specific configuration if requested
        if specific_name:
            configs = [c for c in configs if c['name'].lower() == specific_name.lower()]
            if not configs:
                print(f"\n❌ No configuration found with name: {specific_name}")
                return []
        
        print(f"\n📋 Found {len(configs)} configuration(s) to validate")
        
        # First, check if we can query schema
        print(f"\n🔌 Testing Supabase schema access...")
        print("   If schema queries fail, run this SQL in Supabase:")
        print("   " + "-"*50)
        print(self.create_rpc_function())
        print("   " + "-"*50)
        
        validation_results = []
        
        for config in configs:
            try:
                result = self.validate_single_config(config)
                validation_results.append(result)
            except Exception as e:
                print(f"\n❌ Error validating {config['name']}: {e}")
                validation_results.append({
                    'name': config['name'],
                    'active': config['active'],
                    'errors': [f"Validation failed: {str(e)}"],
                    'warnings': [],
                    'info': []
                })
        
        # Generate fix suggestions
        self.generate_fix_suggestions(validation_results)
        
        # Save validation report
        self.save_validation_report(validation_results)
        
        return validation_results
    
    def save_validation_report(self, results: List[Dict]):
        """Save validation results to a JSON file"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_configs': len(results),
            'configs_with_errors': sum(1 for r in results if r['errors']),
            'configs_with_warnings': sum(1 for r in results if r['warnings']),
            'results': results
        }
        
        filename = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Validation report saved to: {filename}")

def main():
    """Main function with CLI options"""
    import sys
    
    validator = SyncConfigValidator()
    
    if len(sys.argv) > 1:
        # Validate specific configuration
        config_name = ' '.join(sys.argv[1:])
        validator.validate_all(specific_name=config_name)
    else:
        # Validate all configurations
        validator.validate_all()

if __name__ == "__main__":
    main()