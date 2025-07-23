import os
from dotenv import load_dotenv
from notion_client import Client

load_dotenv()
notion = Client(auth=os.getenv('NOTION_TOKEN'))

# Search for your database
response = notion.search(query="Pre Market Levels", filter={"property": "object", "value": "database"})

for db in response['results']:
    if db['object'] == 'database':
        print(f"Database: {db['title'][0]['plain_text']}")
        print(f"ID: {db['id']}")
        print(f"URL: {db['url']}")