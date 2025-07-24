import os
import json
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from notion_client import Client as NotionClient
import traceback
import time
import sys

load_dotenv()

class SupabaseToNotionPull:
    """Pull data from Supabase to Notion - Supabase is source of truth"""
    
    def __init__(self):
        self.supabase = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_KEY')
        )
        self.notion = NotionClient(auth=os.getenv('NOTION_TOKEN'))
        self.config_database_id = os.getenv('SYNC_CONFIG_DATABASE_ID')
        
    def get_sync_configurations(self):
        """Fetch all active sync configurations"""
        configs = []
        
        try:
            response = self.notion.databases.query(
                database_id=self.config_database_id,
                filter={
                    "property": "Active",
                    "checkbox": {
                        "equals": True
                    }
                }
            )
            
            for page in response['results']:
                props = page['properties']
                
                config = {
                    'config_page_id': page['id'],
                    'name': props['Name']['title'][0]['plain_text'] if props['Name']['title'] else 'Unnamed',
                    'notion_database_id': props['Notion_Database_ID']['rich_text'][0]['plain_text'] if props['Notion_Database_ID']['rich_text'] else None,
                    'supabase_table_name': props['Supabase_Table_Name']['rich_text'][0]['plain_text'] if props['Supabase_Table_Name']['rich_text'] else None,
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
    
    def pull_database(self, config):
        """Pull data from Supabase to Notion for one database"""
        print(f"\n{'='*50}")
        print(f"📥 Pulling: {config['name']}")
        print(f"   Supabase → Notion")
        
        try:
            field_mapping = self.get_field_mapping(config)
            id_field = field_mapping.get('id_field', 'supabase_id')
            
            # Get all Supabase items
            supabase_items = self.get_supabase_items(config['supabase_table_name'])
            
            # Get all Notion items for comparison
            notion_items = self.get_notion_items(config['notion_database_id'], field_mapping)
            
            print(f"   📊 Found {len(supabase_items)} Supabase items, {len(notion_items)} Notion items")
            
            created = 0
            updated = 0
            deleted = 0
            
            # Build a map of supabase_id to notion_id
            supabase_to_notion_map = {}
            notion_ids_to_keep = set()
            
            for notion_id, notion_data in notion_items.items():
                if notion_data.get('supabase_id'):
                    supabase_to_notion_map[notion_data['supabase_id']] = notion_id
            
            # Process each Supabase item
            for sb_id, sb_data in supabase_items.items():
                if sb_id in supabase_to_notion_map:
                    # Update existing
                    notion_id = supabase_to_notion_map[sb_id]
                    notion_ids_to_keep.add(notion_id)
                    self.update_in_notion(notion_id, sb_data, field_mapping)
                    updated += 1
                else:
                    # Create new
                    new_notion_id = self.create_in_notion(
                        config['notion_database_id'], 
                        sb_id, 
                        sb_data, 
                        field_mapping
                    )
                    if new_notion_id:
                        notion_ids_to_keep.add(new_notion_id)
                        created += 1
            
            # Delete Notion items that don't exist in Supabase
            for notion_id in notion_items.keys():
                if notion_id not in notion_ids_to_keep:
                    self.delete_from_notion(notion_id)
                    deleted += 1
            
            print(f"\n   ✅ Pull Complete:")
            print(f"      Created: {created}")
            print(f"      Updated: {updated}")
            print(f"      Deleted: {deleted}")
            
            # Update sync status
            self.update_sync_status(config['config_page_id'], 'Success', 'Pull')
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(f"   ❌ Pull failed: {e}")
            self.update_sync_status(config['config_page_id'], 'Failed', 'Pull', error_msg)
    
    def get_supabase_items(self, table_name):
        """Get all items from Supabase"""
        items = {}
        
        try:
            response = self.supabase.table(table_name).select("*").execute()
            
            for item in response.data:
                items[str(item['id'])] = item
                
        except Exception as e:
            print(f"Error fetching from Supabase: {e}")
            raise
            
        return items
    
    def get_notion_items(self, database_id, field_mapping):
        """Get all items from Notion"""
        items = {}
        id_field = field_mapping.get('id_field', 'supabase_id')
        field_types = field_mapping.get('field_types', {})
        
        try:
            has_more = True
            start_cursor = None
            
            while has_more:
                if start_cursor:
                    response = self.notion.databases.query(
                        database_id=database_id,
                        start_cursor=start_cursor
                    )
                else:
                    response = self.notion.databases.query(database_id=database_id)
                
                for page in response['results']:
                    props = page['properties']
                    
                    # Get Supabase ID if exists - check field type
                    supabase_id = None
                    if id_field in props:
                        if field_types.get(id_field) == 'number' and props[id_field].get('number') is not None:
                            supabase_id = str(int(props[id_field]['number']))
                        elif props[id_field].get('rich_text') and props[id_field]['rich_text']:
                            supabase_id = props[id_field]['rich_text'][0]['plain_text']
                    
                    items[page['id']] = {
                        'notion_id': page['id'],
                        'supabase_id': supabase_id,
                        'properties': props
                    }
                
                has_more = response.get('has_more', False)
                start_cursor = response.get('next_cursor')
                
        except Exception as e:
            print(f"Error fetching from Notion: {e}")
            raise
            
        return items
    
    def create_notion_property(self, value, notion_field, field_mapping=None):
        """Create Notion property from value"""
        # Check if field has a specific type defined
        if field_mapping and notion_field in field_mapping.get('field_types', {}):
            field_type = field_mapping['field_types'][notion_field]
            if field_type == 'number':
                return {'number': float(value) if value is not None else None}
            elif field_type == 'title':
                return {'title': [{'text': {'content': str(value)}}]}
        
        # Rest of existing logic
        if notion_field in ['Ticker', 'Name', 'Title', 'Symbol']:
            return {'title': [{'text': {'content': str(value)}}]}
        elif notion_field == 'Market Session':
            return {'select': {'name': str(value)}}
        elif notion_field in ['Passed Filters', 'Active']:
            return {'checkbox': bool(value)}
        elif isinstance(value, bool):
            return {'checkbox': value}
        elif isinstance(value, (int, float)):
            return {'number': float(value)}
        elif isinstance(value, str):
            try:
                datetime.fromisoformat(value.replace('Z', '+00:00'))
                return {'date': {'start': value}}
            except:
                return {'rich_text': [{'text': {'content': value}}]}
        elif value is None:
            return None
        
        return {'rich_text': [{'text': {'content': str(value)}}]}
    
    def create_in_notion(self, database_id, supabase_id, data, field_mapping):
        """Create new item in Notion"""
        properties = {}
        
        # Add Supabase ID - check field type
        id_field = field_mapping.get('id_field', 'supabase_id')
        field_types = field_mapping.get('field_types', {})
        
        if field_types.get(id_field) == 'number':
            properties[id_field] = {'number': int(supabase_id)}
        else:
            properties[id_field] = {'rich_text': [{'text': {'content': str(supabase_id)}}]}
        
        # Map fields - pass field_mapping to create_notion_property
        for sb_field, notion_field in field_mapping.get('supabase_to_notion', {}).items():
            # Skip the id field as we already handled it above
            if sb_field == 'id':
                continue
                
            if sb_field in data and data[sb_field] is not None:
                prop = self.create_notion_property(data[sb_field], notion_field, field_mapping)
                if prop:
                    properties[notion_field] = prop
        
        # Ensure we have a title property
        if not any(p.get('title') for p in properties.values()):
            # Find the first title field or create a default
            for field, prop_type in field_mapping.get('field_types', {}).items():
                if prop_type == 'title' and field in properties:
                    # Already has content, just need to ensure it's formatted as title
                    break
                elif prop_type == 'title':
                    properties[field] = {'title': [{'text': {'content': 'Untitled'}}]}
                    break
        
        try:
            response = self.notion.pages.create(
                parent={'database_id': database_id},
                properties=properties
            )
            print(f"      + Created: {data.get('ticker', data.get('symbol', f'ID {supabase_id}'))}")
            return response['id']
        except Exception as e:
            print(f"      ✗ Error creating: {e}")
            return None
    
    def update_in_notion(self, notion_id, data, field_mapping):
        """Update existing Notion page"""
        properties = {}
        
        # Map fields - pass field_mapping to create_notion_property
        for sb_field, notion_field in field_mapping.get('supabase_to_notion', {}).items():
            # Skip the id field as it shouldn't be updated
            if sb_field == 'id':
                continue
                
            if sb_field in data and data[sb_field] is not None:
                prop = self.create_notion_property(data[sb_field], notion_field, field_mapping)
                if prop:
                    properties[notion_field] = prop
        
        if not properties:
            return
            
        try:
            self.notion.pages.update(notion_id, properties=properties)
            print(f"      ↻ Updated: {data.get('ticker', data.get('symbol', 'item'))}")
        except Exception as e:
            print(f"      ✗ Error updating: {e}")
    
    def delete_from_notion(self, notion_id):
        """Delete page from Notion"""
        try:
            self.notion.blocks.delete(block_id=notion_id)
            print(f"      - Deleted: Notion page")
        except Exception as e:
            print(f"      ✗ Error deleting: {e}")
    
    def update_sync_status(self, config_page_id, status, direction, error_msg=None):
        """Update sync status in configuration"""
        properties = {
            'Last_Sync': {'date': {'start': datetime.now().isoformat()}},
            'Sync_Status': {'select': {'name': f'{direction} - {status}'}}
        }
        
        if error_msg:
            properties['Error_Message'] = {'rich_text': [{'text': {'content': str(error_msg)[:2000]}}]}
        else:
            properties['Error_Message'] = {'rich_text': [{'text': {'content': ''}}]}
            
        try:
            self.notion.pages.update(config_page_id, properties=properties)
        except:
            pass
    
    def pull_single(self, table_name):
        """Pull a specific database by name (supports partial matching)"""
        print(f"\n🚀 Starting Pull for '{table_name}'...")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        configs = self.get_sync_configurations()
        
        # Find matching configurations (case-insensitive partial match)
        matching_configs = []
        for config in configs:
            if table_name.lower() in config['name'].lower():
                matching_configs.append(config)
        
        if not matching_configs:
            print(f"\n❌ No configuration found matching '{table_name}'")
            print("\nAvailable configurations:")
            for config in configs:
                print(f"  - {config['name']}")
            return
        
        if len(matching_configs) > 1:
            print(f"\n⚠️  Multiple configurations match '{table_name}':")
            for config in matching_configs:
                print(f"  - {config['name']}")
            print("\nPlease be more specific.")
            return
        
        # Pull the single matching configuration
        matching_config = matching_configs[0]
        print(f"\n📋 Found configuration: {matching_config['name']}")
        
        try:
            self.pull_database(matching_config)
            print(f"\n📊 Pull Complete for '{matching_config['name']}'!")
        except Exception as e:
            print(f"\n❌ Failed to pull {matching_config['name']}: {e}")
            
    def pull_all(self):
        """Pull all configured databases"""
        print("\n🚀 Starting Pull from Supabase...")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        configs = self.get_sync_configurations()
        
        if not configs:
            print("\n❌ No active sync configurations found.")
            return
        
        print(f"\n📋 Found {len(configs)} configuration(s)")
        
        successful = 0
        failed = 0
        
        for config in configs:
            try:
                self.pull_database(config)
                successful += 1
                time.sleep(1)
            except Exception as e:
                failed += 1
                print(f"\n❌ Failed to pull {config['name']}: {e}")
        
        print(f"\n{'='*50}")
        print(f"📊 Pull Complete!")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")

if __name__ == "__main__":
    puller = SupabaseToNotionPull()
    
    if len(sys.argv) > 1:
        # Pull specific table
        table_name = sys.argv[1]
        puller.pull_single(table_name)
    else:
        # Pull all tables
        puller.pull_all()