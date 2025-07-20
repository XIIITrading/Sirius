import os
import json
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from notion_client import Client as NotionClient
import traceback
import time

load_dotenv()

class UniversalSupabaseNotionSync:
    def __init__(self):
        self.supabase = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_KEY')
        )
        self.notion = NotionClient(auth=os.getenv('NOTION_TOKEN'))
        self.config_database_id = os.getenv('SYNC_CONFIG_DATABASE_ID')  # Your config database
        
    def get_sync_configurations(self):
        """Fetch all active sync configurations from the master database"""
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
                
                # Parse field mappings if provided
                if props.get('Field_Mappings', {}).get('rich_text'):
                    try:
                        mappings_json = props['Field_Mappings']['rich_text'][0]['plain_text']
                        config['field_mappings'] = json.loads(mappings_json)
                    except:
                        # Use default mappings if JSON is invalid
                        pass
                
                if config['notion_database_id'] and config['supabase_table_name']:
                    configs.append(config)
                    
        except Exception as e:
            print(f"Error fetching sync configurations: {e}")
            
        return configs
    
    def update_sync_status(self, config_page_id, status, error_message=None):
        """Update the status of a sync configuration"""
        properties = {
            'Last_Sync': {'date': {'start': datetime.now().isoformat()}},
            'Sync_Status': {'select': {'name': status}}
        }
        
        if error_message:
            properties['Error_Message'] = {'rich_text': [{'text': {'content': str(error_message)[:2000]}}]}
        else:
            properties['Error_Message'] = {'rich_text': [{'text': {'content': ''}}]}
            
        try:
            self.notion.pages.update(config_page_id, properties=properties)
        except Exception as e:
            print(f"Error updating sync status: {e}")
    
    def get_field_mapping(self, config):
        """Get field mapping configuration or use defaults"""
        default_mapping = {
            'notion_to_supabase': {
                'Name': 'name',
                'Status': 'status',
                'Amount': 'amount',
                'Description': 'description',
                'Due_Date': 'due_date',
                'Priority': 'priority'
            },
            'supabase_to_notion': {
                'name': 'Name',
                'status': 'Status',
                'amount': 'Amount',
                'description': 'Description',
                'due_date': 'Due_Date',
                'priority': 'Priority'
            },
            'id_field': 'supabase_id'  # The field in Notion that stores Supabase ID
        }
        
        # Merge with custom mappings if provided
        if config.get('field_mappings'):
            default_mapping.update(config['field_mappings'])
            
        return default_mapping
    
    def sync_database_pair(self, config):
        """Sync a single database pair"""
        print(f"\n{'='*50}")
        print(f"Syncing: {config['name']}")
        print(f"Notion DB: {config['notion_database_id']}")
        print(f"Supabase Table: {config['supabase_table_name']}")
        
        try:
            # Update status to running
            self.update_sync_status(config['config_page_id'], 'Running')
            
            # Get field mappings
            field_mapping = self.get_field_mapping(config)
            
            # Fetch data from both sources
            notion_items = self.get_notion_items(config['notion_database_id'], field_mapping)
            supabase_items = self.get_supabase_items(config['supabase_table_name'])
            
            # Sync logic
            synced_count = 0
            
            # 1. Create new items in Notion (from Supabase)
            for sb_id, sb_item in supabase_items.items():
                if sb_id not in notion_items:
                    self.create_in_notion(
                        config['notion_database_id'], 
                        sb_id, 
                        sb_item['data'], 
                        field_mapping
                    )
                    synced_count += 1
            
            # 2. Create new items in Supabase (from Notion)
            for notion_id, notion_item in notion_items.items():
                if notion_id and notion_id not in supabase_items:
                    # This is an item in Notion without a Supabase ID
                    if notion_item.get('supabase_id') is None:
                        new_id = self.create_in_supabase(
                            config['supabase_table_name'], 
                            notion_item['data'], 
                            field_mapping
                        )
                        if new_id:
                            # Update Notion with the new Supabase ID
                            self.update_notion_id_field(
                                notion_item['notion_id'], 
                                new_id, 
                                field_mapping['id_field']
                            )
                            synced_count += 1
            
            # 3. Update existing items based on last modified
            for item_id in set(notion_items.keys()) & set(supabase_items.keys()):
                if item_id:  # Skip empty IDs
                    notion_item = notion_items[item_id]
                    supabase_item = supabase_items[item_id]
                    
                    # Compare timestamps
                    notion_time = datetime.fromisoformat(notion_item['last_modified'].replace('Z', '+00:00'))
                    supabase_time = datetime.fromisoformat(supabase_item['last_modified'].replace('Z', '+00:00'))
                    
                    if notion_time > supabase_time:
                        # Update Supabase
                        self.update_in_supabase(
                            config['supabase_table_name'], 
                            item_id, 
                            notion_item['data'], 
                            field_mapping
                        )
                        synced_count += 1
                    elif supabase_time > notion_time:
                        # Update Notion
                        self.update_in_notion_page(
                            notion_item['notion_id'], 
                            supabase_item['data'], 
                            field_mapping
                        )
                        synced_count += 1
            
            # Update status to success
            self.update_sync_status(config['config_page_id'], 'Success')
            print(f"✅ Sync completed successfully! {synced_count} items synced.")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(f"❌ Sync failed: {error_msg}")
            self.update_sync_status(config['config_page_id'], 'Failed', error_msg)
    
    def get_notion_items(self, database_id, field_mapping):
        """Get all items from a Notion database"""
        items = {}
        id_field = field_mapping.get('id_field', 'supabase_id')
        
        try:
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
                
                key = supabase_id if supabase_id else f"notion_{page['id']}"
                items[key] = {
                    'notion_id': page['id'],
                    'supabase_id': supabase_id,
                    'last_modified': page['last_edited_time'],
                    'data': data
                }
                
        except Exception as e:
            print(f"Error fetching from Notion: {e}")
            raise
            
        return items
    
    def get_supabase_items(self, table_name):
        """Get all items from a Supabase table"""
        items = {}
        
        try:
            response = self.supabase.table(table_name).select("*").execute()
            
            for item in response.data:
                items[str(item['id'])] = {
                    'last_modified': item.get('updated_at', item.get('created_at')),
                    'data': item
                }
                
        except Exception as e:
            print(f"Error fetching from Supabase: {e}")
            raise
            
        return items
    
    def extract_notion_value(self, property_obj):
        """Extract value from Notion property object"""
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
        elif prop_type == 'email' and property_obj.get('email'):
            return property_obj['email']
        elif prop_type == 'url' and property_obj.get('url'):
            return property_obj['url']
        elif prop_type == 'phone_number' and property_obj.get('phone_number'):
            return property_obj['phone_number']
        
        return None
    
    def create_notion_property(self, value, notion_field):
        """Create Notion property object from value"""
        if notion_field == 'Name':  # Title field
            return {'title': [{'text': {'content': str(value)}}]}
        elif isinstance(value, bool):
            return {'checkbox': value}
        elif isinstance(value, (int, float)):
            return {'number': value}
        elif isinstance(value, str):
            # Check if it's a date
            try:
                datetime.fromisoformat(value.replace('Z', '+00:00'))
                return {'date': {'start': value}}
            except:
                # Default to rich_text for strings
                return {'rich_text': [{'text': {'content': value}}]}
        
        return {'rich_text': [{'text': {'content': str(value)}}]}
    
    def create_in_notion(self, database_id, supabase_id, data, field_mapping):
        """Create new item in Notion"""
        properties = {}
        
        # Add Supabase ID
        id_field = field_mapping.get('id_field', 'supabase_id')
        properties[id_field] = {'rich_text': [{'text': {'content': str(supabase_id)}}]}
        
        # Map fields
        for sb_field, notion_field in field_mapping.get('supabase_to_notion', {}).items():
            if sb_field in data and data[sb_field] is not None:
                properties[notion_field] = self.create_notion_property(data[sb_field], notion_field)
        
        # Ensure there's a title
        if 'Name' not in properties:
            properties['Name'] = {'title': [{'text': {'content': 'Untitled'}}]}
        
        try:
            self.notion.pages.create(
                parent={'database_id': database_id},
                properties=properties
            )
            print(f"  ✓ Created in Notion: {supabase_id}")
        except Exception as e:
            print(f"  ✗ Error creating in Notion: {e}")
            raise
    
    def create_in_supabase(self, table_name, data, field_mapping):
        """Create new item in Supabase"""
        # Remove any Notion-specific fields
        clean_data = {k: v for k, v in data.items() if not k.startswith('notion_')}
        
        try:
            response = self.supabase.table(table_name).insert(clean_data).execute()
            new_id = str(response.data[0]['id'])
            print(f"  ✓ Created in Supabase: {new_id}")
            return new_id
        except Exception as e:
            print(f"  ✗ Error creating in Supabase: {e}")
            raise
    
    def update_in_supabase(self, table_name, supabase_id, data, field_mapping):
        """Update item in Supabase"""
        # Remove any Notion-specific fields
        clean_data = {k: v for k, v in data.items() if not k.startswith('notion_')}
        clean_data['updated_at'] = datetime.now().isoformat()
        
        try:
            self.supabase.table(table_name).update(clean_data).eq('id', supabase_id).execute()
            print(f"  ✓ Updated in Supabase: {supabase_id}")
        except Exception as e:
            print(f"  ✗ Error updating in Supabase: {e}")
            raise
    
    def update_in_notion_page(self, notion_id, data, field_mapping):
        """Update existing Notion page"""
        properties = {}
        
        # Map fields
        for sb_field, notion_field in field_mapping.get('supabase_to_notion', {}).items():
            if sb_field in data and data[sb_field] is not None:
                properties[notion_field] = self.create_notion_property(data[sb_field], notion_field)
        
        try:
            self.notion.pages.update(notion_id, properties=properties)
            print(f"  ✓ Updated in Notion: {notion_id}")
        except Exception as e:
            print(f"  ✗ Error updating in Notion: {e}")
            raise
    
    def update_notion_id_field(self, notion_id, supabase_id, id_field):
        """Update just the Supabase ID field in Notion"""
        properties = {
            id_field: {'rich_text': [{'text': {'content': str(supabase_id)}}]}
        }
        
        try:
            self.notion.pages.update(notion_id, properties=properties)
        except Exception as e:
            print(f"  ✗ Error updating ID field in Notion: {e}")
            raise
    
    def run_all_syncs(self):
        """Main function to run all configured syncs"""
        print("Starting Universal Sync Process...")
        print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get all sync configurations
        configs = self.get_sync_configurations()
        
        if not configs:
            print("No active sync configurations found.")
            return
        
        print(f"Found {len(configs)} active sync configurations.")
        
        # Run each sync
        for config in configs:
            try:
                self.sync_database_pair(config)
                time.sleep(1)  # Brief pause between syncs to respect rate limits
            except Exception as e:
                print(f"Failed to sync {config['name']}: {e}")
                continue
        
        print(f"\n{'='*50}")
        print("Universal Sync Process Complete!")

# Additional utility functions for managing configurations

def add_sync_configuration(notion_client, config_db_id, name, notion_db_id, supabase_table, field_mappings=None):
    """Helper function to add a new sync configuration"""
    properties = {
        'Name': {'title': [{'text': {'content': name}}]},
        'Notion_Database_ID': {'rich_text': [{'text': {'content': notion_db_id}}]},
        'Supabase_Table_Name': {'rich_text': [{'text': {'content': supabase_table}}]},
        'Active': {'checkbox': True},
        'Sync_Status': {'select': {'name': 'Never Run'}}
    }
    
    if field_mappings:
        properties['Field_Mappings'] = {'rich_text': [{'text': {'content': json.dumps(field_mappings, indent=2)}}]}
    
    notion_client.pages.create(
        parent={'database_id': config_db_id},
        properties=properties
    )

if __name__ == "__main__":
    syncer = UniversalSupabaseNotionSync()
    syncer.run_all_syncs()