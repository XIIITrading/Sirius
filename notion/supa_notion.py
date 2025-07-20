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

def get_existing_supabase_ids_from_notion():
    """Get all Supabase IDs already in Notion"""
    existing_ids = set()
    database_id = os.getenv('NOTION_DATABASE_ID')
    
    try:
        response = notion.databases.query(database_id=database_id)
        
        for page in response['results']:
            # Assuming you have a 'supabase_id' property in Notion
            supabase_id_prop = page['properties'].get('supabase_id', {})
            if supabase_id_prop.get('rich_text'):
                existing_ids.add(supabase_id_prop['rich_text'][0]['plain_text'])
    except Exception as e:
        print(f"Error fetching from Notion: {e}")
    
    return existing_ids

def create_notion_page(item):
    """Create a new page in Notion from Supabase data"""
    database_id = os.getenv('NOTION_DATABASE_ID')
    
    # Map your Supabase fields to Notion properties
    # Adjust these based on your actual schema
    properties = {
        'Name': {
            'title': [
                {
                    'text': {
                        'content': item.get('name', 'Untitled')
                    }
                }
            ]
        },
        'supabase_id': {
            'rich_text': [
                {
                    'text': {
                        'content': str(item['id'])
                    }
                }
            ]
        },
        'Status': {
            'select': {
                'name': item.get('status', 'Active')
            }
        },
        'Amount': {
            'number': item.get('amount', 0)
        },
        'Last_Updated': {
            'date': {
                'start': item.get('updated_at', datetime.now().isoformat())
            }
        }
        # Add more field mappings as needed
    }
    
    try:
        notion.pages.create(
            parent={'database_id': database_id},
            properties=properties
        )
        print(f"Created Notion page for item: {item['id']}")
    except Exception as e:
        print(f"Error creating Notion page: {e}")

def pull_new_from_supabase():
    """Main function to pull new items from Supabase to Notion"""
    print("Fetching existing Notion items...")
    existing_ids = get_existing_supabase_ids_from_notion()
    
    print("Fetching Supabase data...")
    table_name = os.getenv('SUPABASE_TABLE_NAME')
    response = supabase.table(table_name).select("*").execute()
    
    new_items = [item for item in response.data if str(item['id']) not in existing_ids]
    
    print(f"Found {len(new_items)} new items to add to Notion")
    
    for item in new_items:
        create_notion_page(item)
    
    print("Pull complete!")

if __name__ == "__main__":
    pull_new_from_supabase()