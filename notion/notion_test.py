import os
from dotenv import load_dotenv
from notion_client import Client

load_dotenv()

notion = Client(auth=os.getenv('NOTION_TOKEN'))

# Try with the correct ID (no hyphens)
config_db_id = "23670cfb9d128055ac66ecd1b914b28e"

print(f"Testing database ID: {config_db_id}")

try:
    response = notion.databases.retrieve(database_id=config_db_id)
    print("✅ Success! Database found and accessible!")
    print(f"Database Title: {response['title'][0]['plain_text']}")
except Exception as e:
    print(f"❌ Still getting error: {e}")
    print("\nMake sure you've added your integration to the database!")