# config/database.py

import os
from dotenv import load_dotenv  # Loads environment variables from .env file
from supabase import create_client, Client  # Supabase Python client

# Load environment variables from .env file
# This keeps sensitive data like API keys out of your code
load_dotenv()

class SupabaseConnection:
    """
    A class to manage Supabase database connection.
    This follows the singleton pattern to ensure only one connection exists.
    """
    
    def __init__(self):
        # Retrieve Supabase credentials from environment variables
        # os.getenv() gets the value of an environment variable
        self.url = os.getenv('SUPABASE_URL')
        self.key = os.getenv('SUPABASE_KEY')
        
        # Validate that credentials exist
        if not self.url or not self.key:
            raise ValueError("Missing Supabase credentials in environment variables")
        
        # Create the Supabase client instance
        # This client will be used for all database operations
        self.client: Client = create_client(self.url, self.key)
    
    def get_client(self) -> Client:
        """
        Returns the Supabase client instance.
        
        Returns:
            Client: The Supabase client for database operations
        """
        return self.client

# Create a single instance to be imported by other modules
# This ensures we don't create multiple connections
supabase_connection = SupabaseConnection()