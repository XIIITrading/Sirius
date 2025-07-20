import os
import json
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from notion_client import Client as NotionClient

load_dotenv()

class SupabaseNotionSync:
    def __init__(self):
        self.supabase = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_KEY')
        )
        self.notion = NotionClient(auth=os.getenv('NOTION_TOKEN'))
        self.database_id = os.getenv('NOTION_DATABASE_ID')
        self.table_name = os.getenv('SUPABASE_TABLE_NAME')
        
    def get_notion_items(self):
        """Get all items from Notion with their metadata"""
        items = {}
        
        try:
            response = self.notion.databases.query(database_id=self.database_id)
            
            for page in response['results']:
                props = page['properties']
                
                # Extract supabase_id if exists
                supabase_id = None
                if props.get('supabase_id', {}).get('rich_text'):
                    supabase_id = props['supabase_id']['rich_text'][0]['plain_text']
                
                if supabase_id:
                    items[supabase_id] = {
                        'notion_id': page['id'],
                        'last_modified': page['last_edited_time'],
                        'data': self.extract_notion_data(props)
                    }
        except Exception as e:
            print(f"Error fetching from Notion: {e}")
            
        return items
    
    def get_supabase_items(self):
        """Get all items from Supabase"""
        items = {}
        
        try:
            response = self.supabase.table(self.table_name).select("*").execute()
            
            for item in response.data:
                items[str(item['id'])] = {
                    'last_modified': item.get('updated_at', item.get('created_at')),
                    'data': item
                }
        except Exception as e:
            print(f"Error fetching from Supabase: {e}")
            
        return items
    
    def extract_notion_data(self, properties):
        """Extract data from Notion properties"""
        data = {}
        
        # Name
        if properties.get('Name', {}).get('title'):
            data['name'] = properties['Name']['title'][0]['plain_text']
        
        # Status
        if properties.get('Status', {}).get('select'):
            data['status'] = properties['Status']['select']['name']
        
        # Amount
        if properties.get('Amount', {}).get('number') is not None:
            data['amount'] = properties['Amount']['number']
        
        # Add more fields as needed
        
        return data
    
    def create_in_notion(self, supabase_id, data):
        """Create new item in Notion"""
        properties = {
            'Name': {'title': [{'text': {'content': data.get('name', 'Untitled')}}]},
            'supabase_id': {'rich_text': [{'text': {'content': supabase_id}}]},
            'Status': {'select': {'name': data.get('status', 'Active')}},
            'Amount': {'number': data.get('amount', 0)},
            'Last_Updated': {'date': {'start': datetime.now().isoformat()}}
        }
        
        try:
            self.notion.pages.create(
                parent={'database_id': self.database_id},
                properties=properties
            )
            print(f"Created in Notion: {supabase_id}")
        except Exception as e:
            print(f"Error creating in Notion: {e}")
    
    def update_in_notion(self, notion_id, data):
        """Update existing item in Notion"""
        properties = {}
        
        if 'name' in data:
            properties['Name'] = {'title': [{'text': {'content': data['name']}}]}
        if 'status' in data:
            properties['Status'] = {'select': {'name': data['status']}}
        if 'amount' in data:
            properties['Amount'] = {'number': data['amount']}
        
        properties['Last_Updated'] = {'date': {'start': datetime.now().isoformat()}}
        
        try:
            self.notion.pages.update(notion_id, properties=properties)
            print(f"Updated in Notion: {notion_id}")
        except Exception as e:
            print(f"Error updating in Notion: {e}")
    
    def create_in_supabase(self, data):
        """Create new item in Supabase"""
        try:
            response = self.supabase.table(self.table_name).insert(data).execute()
            return str(response.data[0]['id'])
        except Exception as e:
            print(f"Error creating in Supabase: {e}")
            return None
    
    def update_in_supabase(self, supabase_id, data):
        """Update existing item in Supabase"""
        data['updated_at'] = datetime.now().isoformat()
        
        try:
            self.supabase.table(self.table_name).update(data).eq('id', supabase_id).execute()
            print(f"Updated in Supabase: {supabase_id}")
        except Exception as e:
            print(f"Error updating in Supabase: {e}")
    
    def sync(self):
        """Main sync function"""
        print("Starting two-way sync...")
        
        # Get items from both sources
        notion_items = self.get_notion_items()
        supabase_items = self.get_supabase_items()
        
        all_ids = set(notion_items.keys()) | set(supabase_items.keys())
        
        for item_id in all_ids:
            notion_item = notion_items.get(item_id)
            supabase_item = supabase_items.get(item_id)
            
            if notion_item and not supabase_item:
                # Item exists only in Notion - create in Supabase
                self.create_in_supabase(notion_item['data'])
                
            elif supabase_item and not notion_item:
                # Item exists only in Supabase - create in Notion
                self.create_in_notion(item_id, supabase_item['data'])
                
            else:
                # Item exists in both - check which is newer
                notion_time = datetime.fromisoformat(notion_item['last_modified'].replace('Z', '+00:00'))
                supabase_time = datetime.fromisoformat(supabase_item['last_modified'].replace('Z', '+00:00'))
                
                if notion_time > supabase_time:
                    # Notion is newer - update Supabase
                    self.update_in_supabase(item_id, notion_item['data'])
                elif supabase_time > notion_time:
                    # Supabase is newer - update Notion
                    self.update_in_notion(notion_item['notion_id'], supabase_item['data'])
        
        print("Sync complete!")

if __name__ == "__main__":
    syncer = SupabaseNotionSync()
    syncer.sync()