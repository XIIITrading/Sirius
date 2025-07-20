import os
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from notion_client import Client as NotionClient

load_dotenv()

# Initialize clients
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)
notion = NotionClient(auth=os.getenv('NOTION_TOKEN'))

def get_notion_items():
    """Fetch all items from Notion database"""
    database_id = os.getenv('NOTION_DATABASE_ID')
    items = []
    
    try:
        response = notion.databases.query(database_id=database_id)
        
        for page in response['results']:
            props = page['properties']
            
            # Extract data from Notion properties
            item = {
                'notion_id': page['id'],
                'supabase_id': None,
                'name': None,
                'status': None,
                'amount': None,
                'last_modified': page.get('last_edited_time')
            }
            
            # Get supabase_id
            if props.get('supabase_id', {}).get('rich_text'):
                item['supabase_id'] = props['supabase_id']['rich_text'][0]['plain_text']
            
            # Get name
            if props.get('Name', {}).get('title'):
                item['name'] = props['Name']['title'][0]['plain_text']
            
            # Get status
            if props.get('Status', {}).get('select'):
                item['status'] = props['Status']['select']['name']
            
            # Get amount
            if props.get('Amount', {}).get('number') is not None:
                item['amount'] = props['Amount']['number']
            
            # Only include items that have a supabase_id (existing items)
            if item['supabase_id']:
                items.append(item)
                
    except Exception as e:
        print(f"Error fetching from Notion: {e}")
    
    return items

def update_supabase_item(item):
    """Update a single item in Supabase"""
    table_name = os.getenv('SUPABASE_TABLE_NAME')
    
    # Prepare update data (exclude None values)
    update_data = {}
    if item['name'] is not None:
        update_data['name'] = item['name']
    if item['status'] is not None:
        update_data['status'] = item['status']
    if item['amount'] is not None:
        update_data['amount'] = item['amount']
    
    update_data['updated_at'] = datetime.now().isoformat()
    
    try:
        response = supabase.table(table_name).update(update_data).eq('id', item['supabase_id']).execute()
        print(f"Updated Supabase item: {item['supabase_id']}")
    except Exception as e:
        print(f"Error updating Supabase item {item['supabase_id']}: {e}")

def push_updates_to_supabase():
    """Main function to push Notion updates to Supabase"""
    print("Fetching items from Notion...")
    notion_items = get_notion_items()
    
    print(f"Found {len(notion_items)} items to update in Supabase")
    
    for item in notion_items:
        update_supabase_item(item)
    
    print("Push complete!")

if __name__ == "__main__":
    push_updates_to_supabase()