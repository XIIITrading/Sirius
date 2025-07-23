# scripts/setup_database.py

import sys
import os

# Add the parent directory to Python's path so we can import levels_config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from levels_config
from levels_config.database import supabase_connection
from typing import Dict, Any, List

class DatabaseSetup:
    """
    Handles verification and setup instructions for Supabase tables.
    Since we can't create tables directly via the Python client,
    this script will check what exists and provide SQL for manual setup.
    """
    
    def __init__(self):
        self.client = supabase_connection.get_client()
        print("✅ Connected to Supabase")
    
    def check_table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists by attempting to query it.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            bool: True if table exists, False otherwise
        """
        try:
            # Try to select one row from the table
            result = self.client.table(table_name).select("*").limit(1).execute()
            return True
        except Exception as e:
            # If we get an error, table likely doesn't exist
            return False
    
    def generate_sql_scripts(self) -> Dict[str, str]:
        """
        Generate SQL scripts for creating all required tables.
        These can be run in the Supabase SQL Editor.
        
        Returns:
            Dict mapping table names to their CREATE TABLE SQL
        """
        scripts = {}
        
        # SQL for premarket_levels table
        scripts['premarket_levels'] = """
-- Table for storing manually identified price levels
CREATE TABLE IF NOT EXISTS premarket_levels (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    level_type VARCHAR(20) NOT NULL CHECK (level_type IN ('HVN', 'Daily MS', 'H1 OB')),
    position VARCHAR(10) NOT NULL CHECK (position IN ('Above 1', 'Above 2', 'Below 1', 'Below 2')),
    price DECIMAL(10, 2) NOT NULL,
    strength_score INTEGER CHECK (strength_score >= 1 AND strength_score <= 100),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    active BOOLEAN DEFAULT TRUE,
    UNIQUE(date, ticker, level_type, position)
);

-- Create index for faster queries
CREATE INDEX idx_premarket_date_ticker ON premarket_levels(date, ticker);
"""
        
        # SQL for ranked_levels table  
        scripts['ranked_levels'] = """
-- Table for storing processed and ranked levels
CREATE TABLE IF NOT EXISTS ranked_levels (
    id SERIAL PRIMARY KEY,
    premarket_level_id INTEGER REFERENCES premarket_levels(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    rank INTEGER NOT NULL,
    confluence_score DECIMAL(5, 2),
    zone_high DECIMAL(10, 2),
    zone_low DECIMAL(10, 2),
    tv_variable VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(date, ticker, rank)
);

-- Create index for faster queries
CREATE INDEX idx_ranked_date_ticker ON ranked_levels(date, ticker);
"""
        
        # SQL for level_performance table
        scripts['level_performance'] = """
-- Table for tracking level performance during trading session
CREATE TABLE IF NOT EXISTS level_performance (
    id SERIAL PRIMARY KEY,
    ranked_level_id INTEGER REFERENCES ranked_levels(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    touched BOOLEAN DEFAULT FALSE,
    held BOOLEAN DEFAULT FALSE,
    reversal_points DECIMAL(10, 2),
    touches_count INTEGER DEFAULT 0,
    session_high DECIMAL(10, 2),
    session_low DECIMAL(10, 2),
    session_open DECIMAL(10, 2),
    session_close DECIMAL(10, 2),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for faster queries
CREATE INDEX idx_performance_date_ticker ON level_performance(date, ticker);
"""
        
        return scripts
    
    def check_database_status(self) -> Dict[str, bool]:
        """
        Check which tables exist in the database.
        
        Returns:
            Dict mapping table names to existence status
        """
        tables = ['premarket_levels', 'ranked_levels', 'level_performance']
        status = {}
        
        for table in tables:
            exists = self.check_table_exists(table)
            status[table] = exists
            if exists:
                print(f"✅ Table '{table}' exists")
            else:
                print(f"❌ Table '{table}' does not exist")
        
        return status
    
    def test_insert_sample_data(self):
        """
        Test inserting sample data to verify tables are working.
        Only runs if tables exist.
        """
        # Check if premarket_levels table exists
        if not self.check_table_exists('premarket_levels'):
            print("⚠️  Cannot test insert - premarket_levels table doesn't exist")
            return
        
        try:
            # Insert sample data
            sample_data = {
                'date': '2024-01-15',
                'ticker': 'AAPL',
                'level_type': 'HVN',
                'position': 'Above 1',
                'price': 195.50,
                'strength_score': 85,
                'notes': 'Test level - strong volume node',
                'active': True
            }
            
            result = self.client.table('premarket_levels').insert(sample_data).execute()
            print("✅ Successfully inserted test data!")
            
            # Try to read it back
            data = self.client.table('premarket_levels').select("*").eq('ticker', 'AAPL').execute()
            print(f"📊 Found {len(data.data)} records for AAPL")
            
        except Exception as e:
            print(f"❌ Error testing insert: {e}")
    
    def setup_all_tables(self):
        """
        Main method to check tables and provide setup instructions.
        """
        print("\n🔍 Checking database status...")
        status = self.check_database_status()
        
        # Check if any tables are missing
        missing_tables = [table for table, exists in status.items() if not exists]
        
        if missing_tables:
            print(f"\n⚠️  Missing tables: {', '.join(missing_tables)}")
            print("\n📝 To create the missing tables:")
            print("1. Go to your Supabase dashboard")
            print("2. Navigate to the SQL Editor")
            print("3. Run the following SQL scripts:\n")
            
            scripts = self.generate_sql_scripts()
            
            # Save SQL scripts to files
            for table in missing_tables:
                filename = f"create_{table}.sql"
                with open(filename, 'w') as f:
                    f.write(scripts[table])
                print(f"💾 Saved SQL script to: {filename}")
            
            print("\n4. After creating tables, run this script again to verify")
        else:
            print("\n✨ All tables exist! Testing insert functionality...")
            self.test_insert_sample_data()
            print("\n🎉 Database setup complete and verified!")


if __name__ == "__main__":
    setup = DatabaseSetup()
    setup.setup_all_tables()