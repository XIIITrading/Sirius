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
        self.config_database_id = os.getenv('SYNC_CONFIG_DATABASE_ID')
        
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
                        print(f"  ⚠️ Warning: Invalid JSON in field mappings for {config['name']}")
                        pass
                
                if config['notion_database_id'] and config['supabase_table_name']:
                    configs.append(config)
                else:
                    print(f"  ⚠️ Skipping {config['name']}: Missing database ID or table name")
                    
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
        # Try to load from mappings module if available
        try:
            from mappings import get_mapping_for_config
            return get_mapping_for_config(config)
        except ImportError:
            # Fallback to config field mappings or defaults
            if config.get('field_mappings') and config['field_mappings'].get('supabase_to_notion'):
                return config['field_mappings']
            
            # Return empty default
            return {
                'notion_to_supabase': {},
                'supabase_to_notion': {},
                'id_field': 'supabase_id'
            }
    
    def sync_database_pair(self, config):
        """Sync a single database pair with soft delete support"""
        print(f"\n{'='*50}")
        print(f"Syncing: {config['name']}")
        print(f"Notion DB: {config['notion_database_id']}")
        print(f"Supabase Table: {config['supabase_table_name']}")
        
        try:
            # Update status to running
            self.update_sync_status(config['config_page_id'], 'Running')
            
            # Get field mappings
            field_mapping = self.get_field_mapping(config)
            
            # Fetch data from both sources (including archived/deleted items)
            notion_items = self.get_notion_items(config['notion_database_id'], field_mapping, include_archived=True)
            supabase_items = self.get_supabase_items(config['supabase_table_name'], include_deleted=True)
            
            print(f"  📊 Found {len(notion_items)} Notion items, {len(supabase_items)} Supabase items")
            
            # Sync counters
            created_in_notion = 0
            created_in_supabase = 0
            updated_in_notion = 0
            updated_in_supabase = 0
            archived_in_notion = 0
            deleted_in_supabase = 0
            
            # 1. Handle deletions - Items in Supabase but not in Notion
            for sb_id, sb_item in supabase_items.items():
                if sb_id not in notion_items and not sb_item['data'].get('deleted', False):
                    # Item was deleted from Notion, soft delete in Supabase
                    self.soft_delete_in_supabase(
                        config['supabase_table_name'], 
                        sb_id
                    )
                    deleted_in_supabase += 1
            
            # 2. Create new items in Notion (from Supabase)
            for sb_id, sb_item in supabase_items.items():
                if sb_id not in notion_items and not sb_item['data'].get('deleted', False):
                    self.create_in_notion(
                        config['notion_database_id'], 
                        sb_id, 
                        sb_item['data'], 
                        field_mapping
                    )
                    created_in_notion += 1
            
            # 3. Create new items in Supabase (from Notion)
            for notion_key, notion_item in notion_items.items():
                if notion_key.startswith('notion_') and not notion_item.get('archived', False):
                    new_id = self.create_in_supabase(
                        config['supabase_table_name'], 
                        notion_item['data'], 
                        field_mapping
                    )
                    if new_id:
                        self.update_notion_id_field(
                            notion_item['notion_id'], 
                            new_id, 
                            field_mapping.get('id_field', 'supabase_id')
                        )
                        created_in_supabase += 1
            
            # 4. Update existing items based on last modified
            for item_id in set(notion_items.keys()) & set(supabase_items.keys()):
                if item_id and not item_id.startswith('notion_'):
                    notion_item = notion_items[item_id]
                    supabase_item = supabase_items[item_id]
                    
                    # Check if either side is deleted/archived
                    notion_archived = notion_item.get('archived', False)
                    supabase_deleted = supabase_item['data'].get('deleted', False)
                    
                    # Sync deletion status
                    if notion_archived and not supabase_deleted:
                        # Archive in Supabase
                        self.soft_delete_in_supabase(
                            config['supabase_table_name'], 
                            item_id
                        )
                        deleted_in_supabase += 1
                    elif supabase_deleted and not notion_archived:
                        # Archive in Notion
                        self.archive_in_notion(
                            notion_item['notion_id'],
                            field_mapping
                        )
                        archived_in_notion += 1
                    elif not notion_archived and not supabase_deleted:
                        # Both active, check timestamps for updates
                        try:
                            notion_time = datetime.fromisoformat(notion_item['last_modified'].replace('Z', '+00:00'))
                            supabase_time = datetime.fromisoformat(supabase_item['last_modified'].replace('Z', '+00:00'))
                            
                            if abs((notion_time - supabase_time).total_seconds()) > 1:
                                if notion_time > supabase_time:
                                    self.update_in_supabase(
                                        config['supabase_table_name'], 
                                        item_id, 
                                        notion_item['data'], 
                                        field_mapping
                                    )
                                    updated_in_supabase += 1
                                else:
                                    self.update_in_notion_page(
                                        notion_item['notion_id'], 
                                        supabase_item['data'], 
                                        field_mapping
                                    )
                                    updated_in_notion += 1
                        except Exception as e:
                            print(f"  ⚠️ Error comparing timestamps for {item_id}: {e}")
            
            # 5. Handle items that exist in Notion but were never in Supabase
            notion_only_items = [item for key, item in notion_items.items() 
                               if not key.startswith('notion_') and key not in supabase_items]
            
            for notion_item in notion_only_items:
                if notion_item.get('supabase_id') and not notion_item.get('archived', False):
                    # This item claims to have a Supabase ID but doesn't exist there
                    # Archive it in Notion as it was likely deleted from Supabase
                    self.archive_in_notion(
                        notion_item['notion_id'],
                        field_mapping
                    )
                    archived_in_notion += 1
            
            # Update status to success
            self.update_sync_status(config['config_page_id'], 'Success')
            
            # Summary
            print(f"\n  ✅ Sync Summary:")
            print(f"     Created in Notion: {created_in_notion}")
            print(f"     Created in Supabase: {created_in_supabase}")
            print(f"     Updated in Notion: {updated_in_notion}")
            print(f"     Updated in Supabase: {updated_in_supabase}")
            print(f"     Archived in Notion: {archived_in_notion}")
            print(f"     Soft deleted in Supabase: {deleted_in_supabase}")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(f"❌ Sync failed: {error_msg}")
            self.update_sync_status(config['config_page_id'], 'Failed', error_msg)
    
    def get_notion_items(self, database_id, field_mapping, include_archived=False):
        """Get all items from a Notion database"""
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
                    
                    # Check if archived (if the property exists)
                    archived = False
                    if 'Archived' in props:
                        archived = props['Archived'].get('checkbox', False)
                    
                    # Skip archived items unless specifically requested
                    if archived and not include_archived:
                        continue
                    
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
                        'data': data,
                        'archived': archived
                    }
                
                has_more = response.get('has_more', False)
                start_cursor = response.get('next_cursor')
                
        except Exception as e:
            print(f"Error fetching from Notion: {e}")
            raise
            
        return items
    
    def get_supabase_items(self, table_name, include_deleted=False):
        """Get all items from a Supabase table"""
        items = {}
        
        try:
            # Build query
            query = self.supabase.table(table_name).select("*")
            
            # Filter out deleted items unless requested
            if not include_deleted:
                # Check if 'deleted' column exists by trying to query
                try:
                    query = query.eq('deleted', False)
                except:
                    # If deleted column doesn't exist, just get all
                    pass
            
            response = query.execute()
            
            for item in response.data:
                items[str(item['id'])] = {
                    'last_modified': item.get('updated_at', item.get('created_at')),
                    'data': item
                }
                
        except Exception as e:
            print(f"Error fetching from Supabase: {e}")
            raise
            
        return items
    
    def soft_delete_in_supabase(self, table_name, supabase_id):
        """Soft delete an item in Supabase"""
        try:
            # Try to update with deleted flag
            self.supabase.table(table_name).update({
                'deleted': True,
                'deleted_at': datetime.now().isoformat()
            }).eq('id', supabase_id).execute()
            print(f"  🗑️ Soft deleted in Supabase: ID {supabase_id}")
        except Exception as e:
            # If deleted column doesn't exist, log but don't fail
            if 'column "deleted" does not exist' in str(e):
                print(f"  ⚠️ No 'deleted' column in {table_name} - skipping soft delete")
            else:
                print(f"  ✗ Error soft deleting in Supabase: {e}")
    
    def archive_in_notion(self, notion_id, field_mapping):
        """Archive an item in Notion"""
        try:
            # Try to update with Archived checkbox
            self.notion.pages.update(
                notion_id,
                properties={
                    'Archived': {'checkbox': True}
                }
            )
            print(f"  🗑️ Archived in Notion: {notion_id}")
        except Exception as e:
            # If Archived property doesn't exist, log but don't fail
            if 'property "Archived" does not exist' in str(e).lower():
                print(f"  ⚠️ No 'Archived' property in Notion - skipping archive")
            else:
                print(f"  ✗ Error archiving in Notion: {e}")
    
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
        elif prop_type == 'multi_select' and property_obj.get('multi_select'):
            return ','.join([opt['name'] for opt in property_obj['multi_select']])
        
        return None
    
    def create_notion_property(self, value, notion_field):
        """Create Notion property object from value"""
        # Special handling for known field types
        if notion_field in ['Ticker', 'Name', 'Title', 'Symbol']:
            return {'title': [{'text': {'content': str(value)}}]}
        elif notion_field == 'Market Session':
            return {'select': {'name': str(value)}}
        elif notion_field in ['Passed Filters', 'Active', 'Archived']:
            return {'checkbox': bool(value)}
        elif notion_field in ['Scan Date', 'Scan Time', 'Created At', 'Updated At', 'Entry Time', 'Exit Time']:
            if isinstance(value, str):
                return {'date': {'start': value}}
        elif isinstance(value, bool):
            return {'checkbox': value}
        elif isinstance(value, (int, float)):
            return {'number': float(value)}
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
                value = data[sb_field]
                
                # Skip the deleted field when creating
                if sb_field in ['deleted', 'deleted_at']:
                    continue
                
                # Handle different property types based on field name
                if notion_field == 'Ticker':
                    properties[notion_field] = {'title': [{'text': {'content': str(value)}}]}
                elif notion_field == 'Market Session':
                    properties[notion_field] = {'select': {'name': str(value)}}
                elif notion_field in ['Passed Filters', 'Active', 'Archived']:
                    properties[notion_field] = {'checkbox': bool(value)}
                elif notion_field in ['Scan Date', 'Scan Time', 'Created At']:
                    if isinstance(value, str):
                        properties[notion_field] = {'date': {'start': value}}
                elif notion_field in ['Price', 'Rank', 'Premarket Volume', 'Avg Daily Volume', 
                                    'Dollar Volume', 'ATR', 'ATR Percent', 'Interest Score',
                                    'PM Vol Ratio Score', 'ATR Percent Score', 'Dollar Vol Score',
                                    'PM Vol Abs Score', 'Price ATR Bonus']:
                    try:
                        properties[notion_field] = {'number': float(value)}
                    except (TypeError, ValueError):
                        print(f"  ⚠️ Warning: Could not convert {sb_field}={value} to number")
                        continue
                else:
                    # Default property creation
                    properties[notion_field] = self.create_notion_property(value, notion_field)
        
        try:
            self.notion.pages.create(
                parent={'database_id': database_id},
                properties=properties
            )
            ticker = data.get('ticker', data.get('symbol', 'Unknown'))
            print(f"  ✓ Created in Notion: {ticker} (ID: {supabase_id})")
        except Exception as e:
            print(f"  ✗ Error creating in Notion: {e}")
            if 'validation_error' in str(e).lower():
                print(f"  Properties attempted: {json.dumps(properties, indent=2)}")
            raise
    
    def create_in_supabase(self, table_name, data, field_mapping):
        """Create new item in Supabase"""
        # Remove any Notion-specific fields and None values
        clean_data = {k: v for k, v in data.items() 
                     if not k.startswith('notion_') and k not in ['archived'] and v is not None}
        
        if not clean_data:
            print(f"  ⚠️ No data to insert into Supabase")
            return None
            
        try:
            response = self.supabase.table(table_name).insert(clean_data).execute()
            new_id = str(response.data[0]['id'])
            print(f"  ✓ Created in Supabase: ID {new_id}")
            return new_id
        except Exception as e:
            print(f"  ✗ Error creating in Supabase: {e}")
            print(f"  Data attempted: {json.dumps(clean_data, indent=2)}")
            raise
    
    def update_in_supabase(self, table_name, supabase_id, data, field_mapping):
        """Update item in Supabase"""
        # Remove any Notion-specific fields and None values
        clean_data = {k: v for k, v in data.items() 
                     if not k.startswith('notion_') and k not in ['archived'] and v is not None}
        
        if not clean_data:
            print(f"  ⚠️ No data to update in Supabase")
            return
            
        clean_data['updated_at'] = datetime.now().isoformat()
        
        try:
            self.supabase.table(table_name).update(clean_data).eq('id', supabase_id).execute()
            print(f"  ✓ Updated in Supabase: ID {supabase_id}")
        except Exception as e:
            print(f"  ✗ Error updating in Supabase: {e}")
            raise
    
    def update_in_notion_page(self, notion_id, data, field_mapping):
        """Update existing Notion page"""
        properties = {}
        
        # Map fields
        for sb_field, notion_field in field_mapping.get('supabase_to_notion', {}).items():
            if sb_field in data and data[sb_field] is not None:
                value = data[sb_field]
                
                # Skip the deleted field when updating
                if sb_field in ['deleted', 'deleted_at']:
                    continue
                
                # Handle different property types based on field name
                if notion_field == 'Ticker':
                    properties[notion_field] = {'title': [{'text': {'content': str(value)}}]}
                elif notion_field == 'Market Session':
                    properties[notion_field] = {'select': {'name': str(value)}}
                elif notion_field in ['Passed Filters', 'Active', 'Archived']:
                    properties[notion_field] = {'checkbox': bool(value)}
                elif notion_field in ['Scan Date', 'Scan Time', 'Created At']:
                    if isinstance(value, str):
                        properties[notion_field] = {'date': {'start': value}}
                elif notion_field in ['Price', 'Rank', 'Premarket Volume', 'Avg Daily Volume', 
                                    'Dollar Volume', 'ATR', 'ATR Percent', 'Interest Score',
                                    'PM Vol Ratio Score', 'ATR Percent Score', 'Dollar Vol Score',
                                    'PM Vol Abs Score', 'Price ATR Bonus']:
                    try:
                        properties[notion_field] = {'number': float(value)}
                    except (TypeError, ValueError):
                        print(f"  ⚠️ Warning: Could not convert {sb_field}={value} to number")
                        continue
                else:
                    # Default property creation
                    properties[notion_field] = self.create_notion_property(value, notion_field)
        
        if not properties:
            print(f"  ⚠️ No properties to update in Notion")
            return
            
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
            print(f"  ✓ Updated Supabase ID in Notion: {supabase_id}")
        except Exception as e:
            print(f"  ✗ Error updating ID field in Notion: {e}")
            raise
    
    def run_all_syncs(self):
        """Main function to run all configured syncs"""
        print("\n🔄 Starting Universal Sync Process...")
        print(f"⏰ Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get all sync configurations
        configs = self.get_sync_configurations()
        
        if not configs:
            print("\n❌ No active sync configurations found.")
            print("   Make sure you have entries in your Sync Configuration database with:")
            print("   - Active checkbox ✓ checked")
            print("   - Valid Notion Database ID")
            print("   - Valid Supabase Table Name")
            return
        
        print(f"\n✅ Found {len(configs)} active sync configuration(s)")
        
        # Run each sync
        successful = 0
        failed = 0
        
        for config in configs:
            try:
                self.sync_database_pair(config)
                successful += 1
                time.sleep(1)  # Brief pause between syncs to respect rate limits
            except Exception as e:
                failed += 1
                print(f"\n❌ Failed to sync {config['name']}: {e}")
                continue
        
        # Final summary
        print(f"\n{'='*50}")
        print("📊 Universal Sync Process Complete!")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏱️  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Helper function for manual sync of specific database
def sync_specific_database(database_name):
    """Manually sync a specific database by name"""
    syncer = UniversalSupabaseNotionSync()
    configs = syncer.get_sync_configurations()
    
    for config in configs:
        if config['name'].lower() == database_name.lower():
            print(f"Found configuration for {database_name}")
            syncer.sync_database_pair(config)
            return
    
    print(f"No configuration found for {database_name}")

# Helper function to prepare databases for soft delete
def prepare_databases_for_soft_delete():
    """Add necessary columns/properties for soft delete functionality"""
    print("📋 To enable soft delete functionality, add these to your databases:\n")
    
    print("SUPABASE - Add these columns to each table:")
    print("```sql")
    print("ALTER TABLE your_table_name ADD COLUMN deleted BOOLEAN DEFAULT FALSE;")
    print("ALTER TABLE your_table_name ADD COLUMN deleted_at TIMESTAMP WITH TIME ZONE;")
    print("```\n")
    
    print("NOTION - Add this property to each database:")
    print("- Property Name: Archived")
    print("- Property Type: Checkbox")
    print("- Default: Unchecked\n")
    
    print("Once added, the sync will automatically handle soft deletes!")

if __name__ == "__main__":
    syncer = UniversalSupabaseNotionSync()
    syncer.run_all_syncs()
    
    # Optionally sync a specific database:
    # sync_specific_database("Premarket Scans")
    
    # Show instructions for enabling soft delete:
    # prepare_databases_for_soft_delete()