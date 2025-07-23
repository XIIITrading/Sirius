import os
import json
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from notion_client import Client as NotionClient
import traceback
import time

load_dotenv()

class NotionToSupabasePush:
    """Push data from Notion to Supabase - ONE-WAY OVERWRITE"""
    
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
    
    def push_database(self, config):
        """Push data from Notion to Supabase - OVERWRITES SUPABASE DATA"""
        print(f"\n{'='*50}")
        print(f"📤 Pushing: {config['name']}")
        print(f"   Notion → Supabase (ONE-WAY OVERWRITE)")
        
        try:
            field_mapping = self.get_field_mapping(config)
            id_field = field_mapping.get('id_field', 'supabase_id')
            
            # Get all Notion items
            notion_items = self.get_notion_items(config['notion_database_id'], field_mapping)
            
            # Get all Supabase items for comparison
            supabase_items = self.get_supabase_items(config['supabase_table_name'])
            
            print(f"   📊 Found {len(notion_items)} Notion items, {len(supabase_items)} Supabase items")
            
            created = 0
            updated = 0
            deleted = 0
            
            # Track which Supabase IDs we've seen in Notion
            seen_supabase_ids = set()
            
            # Process each Notion item - ALWAYS OVERWRITES SUPABASE
            for notion_item in notion_items.values():
                supabase_id = notion_item.get('supabase_id')
                data = notion_item['data']
                
                if supabase_id:
                    seen_supabase_ids.add(supabase_id)
                    
                    if supabase_id in supabase_items:
                        # Update existing - ALWAYS OVERWRITE
                        self.update_in_supabase(config['supabase_table_name'], supabase_id, data)
                        updated += 1
                    else:
                        # Supabase ID exists but record doesn't - create it
                        self.create_in_supabase_with_id(config['supabase_table_name'], supabase_id, data)
                        created += 1
                else:
                    # Create new
                    new_id = self.create_in_supabase(config['supabase_table_name'], data)
                    if new_id:
                        # Update Notion with the new ID
                        self.update_notion_id_field(notion_item['notion_id'], new_id, id_field)
                        created += 1
            
            # Delete items that exist in Supabase but not in Notion
            for sb_id in supabase_items.keys():
                if sb_id not in seen_supabase_ids:
                    self.delete_from_supabase(config['supabase_table_name'], sb_id)
                    deleted += 1
            
            print(f"\n   ✅ Push Complete:")
            print(f"      Created: {created}")
            print(f"      Updated (Overwritten): {updated}")
            print(f"      Deleted: {deleted}")
            
            # Update sync status
            self.update_sync_status(config['config_page_id'], 'Success', 'Push')
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(f"   ❌ Push failed: {e}")
            self.update_sync_status(config['config_page_id'], 'Failed', 'Push', error_msg)
    
    def get_notion_items(self, database_id, field_mapping):
        """Get all items from Notion database"""
        items = {}
        id_field = field_mapping.get('id_field', 'supabase_id')
        
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
                    
                    # Get Supabase ID if exists
                    supabase_id = None
                    if props.get(id_field, {}).get('rich_text'):
                        supabase_id = props[id_field]['rich_text'][0]['plain_text']
                    
                    # Extract data based on field mappings
                    data = {}
                    for notion_field, supabase_field in field_mapping.get('notion_to_supabase', {}).items():
                        if notion_field in props:
                            value = self.extract_notion_value(props[notion_field])
                            if value is not None:
                                data[supabase_field] = value
                    
                    items[page['id']] = {
                        'notion_id': page['id'],
                        'supabase_id': supabase_id,
                        'data': data
                    }
                
                has_more = response.get('has_more', False)
                start_cursor = response.get('next_cursor')
                
        except Exception as e:
            print(f"Error fetching from Notion: {e}")
            raise
            
        return items
    
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
    
    def extract_notion_value(self, property_obj):
        """Extract value from Notion property"""
        prop_type = property_obj['type']
        
        if prop_type == 'title' and property_obj.get('title'):
            return property_obj['title'][0]['plain_text'] if property_obj['title'] else None
        elif prop_type == 'rich_text' and property_obj.get('rich_text'):
            return property_obj['rich_text'][0]['plain_text'] if property_obj['rich_text'] else None
        elif prop_type == 'number':
            return property_obj.get('number')
        elif prop_type == 'select' and property_obj.get('select'):
            return property_obj['select']['name']
        elif prop_type == 'date' and property_obj.get('date'):
            return property_obj['date']['start']
        elif prop_type == 'checkbox':
            return property_obj.get('checkbox', False)
        elif prop_type == 'multi_select' and property_obj.get('multi_select'):
            return ','.join([opt['name'] for opt in property_obj['multi_select']])
        
        return None
    
    def create_in_supabase(self, table_name, data):
        """Create new item in Supabase"""
        clean_data = {k: v for k, v in data.items() if v is not None}
        
        if not clean_data:
            return None
            
        try:
            response = self.supabase.table(table_name).insert(clean_data).execute()
            new_id = str(response.data[0]['id'])
            print(f"      + Created: ID {new_id}")
            return new_id
        except Exception as e:
            print(f"      ✗ Error creating: {e}")
            return None
    
    def create_in_supabase_with_id(self, table_name, supabase_id, data):
        """Create item with specific ID"""
        clean_data = {k: v for k, v in data.items() if v is not None}
        clean_data['id'] = int(supabase_id)
        
        try:
            response = self.supabase.table(table_name).insert(clean_data).execute()
            print(f"      + Recreated: ID {supabase_id}")
        except Exception as e:
            print(f"      ✗ Error recreating ID {supabase_id}: {e}")
    
    def update_in_supabase(self, table_name, supabase_id, data):
        """Update existing item in Supabase - ALWAYS OVERWRITES"""
        clean_data = {k: v for k, v in data.items() if v is not None}
        # Remove the automatic updated_at - let Supabase handle it if the column exists
        
        try:
            self.supabase.table(table_name).update(clean_data).eq('id', supabase_id).execute()
            print(f"      ↻ Overwritten: ID {supabase_id}")
        except Exception as e:
            print(f"      ✗ Error updating ID {supabase_id}: {e}")
    
    def delete_from_supabase(self, table_name, supabase_id):
        """Delete item from Supabase"""
        try:
            self.supabase.table(table_name).delete().eq('id', supabase_id).execute()
            print(f"      - Deleted: ID {supabase_id}")
        except Exception as e:
            print(f"      ✗ Error deleting ID {supabase_id}: {e}")
    
    def update_notion_id_field(self, notion_id, supabase_id, id_field):
        """Update Supabase ID in Notion"""
        try:
            self.notion.pages.update(
                notion_id,
                properties={
                    id_field: {'rich_text': [{'text': {'content': str(supabase_id)}}]}
                }
            )
        except Exception as e:
            print(f"      ✗ Error updating Notion ID field: {e}")
    
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
    
    def push_all(self):
        """Push all configured databases - ONE-WAY OVERWRITE"""
        print("\n🚀 Starting Push to Supabase (ONE-WAY OVERWRITE)...")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("⚠️  This will OVERWRITE all data in Supabase with Notion data!")
        
        configs = self.get_sync_configurations()
        
        if not configs:
            print("\n❌ No active sync configurations found.")
            return
        
        print(f"\n📋 Found {len(configs)} configuration(s)")
        
        successful = 0
        failed = 0
        
        for config in configs:
            try:
                self.push_database(config)
                successful += 1
                time.sleep(1)
            except Exception as e:
                failed += 1
                print(f"\n❌ Failed to push {config['name']}: {e}")
        
        print(f"\n{'='*50}")
        print(f"📊 Push Complete!")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")

if __name__ == "__main__":
    pusher = NotionToSupabasePush()
    pusher.push_all()