import os
import sys
from datetime import datetime, date
import pandas as pd
from pathlib import Path
import supabase
from supabase import create_client
import logging
from typing import Dict, List, Optional
import numpy as np
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def find_project_root():
    """Find the project root by looking for .env file."""
    current_path = Path(__file__).resolve()
    
    # Walk up the directory tree looking for .env
    for parent in current_path.parents:
        if (parent / '.env').exists():
            return parent
            
    return None

# Load environment variables from .env file
project_root = find_project_root()
if project_root:
    env_path = project_root / '.env'
    load_dotenv(env_path)
    logging.info(f"Loaded .env from: {env_path}")
else:
    # Try loading from current directory as fallback
    load_dotenv()
    logging.warning("Could not find project root, attempting to load .env from current directory")

class DailyPlaybookIngestion:
    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize the ingestion processor with Supabase credentials."""
        self.supabase = create_client(supabase_url, supabase_key)
        self.today = date.today()
        
    def find_available_images(self, folder_path: Path) -> Dict[str, Path]:
        """Find all ticker PNG images in the folder."""
        images = {}
        for img_file in folder_path.glob("*.png"):
            # Skip files that don't look like ticker names
            ticker = img_file.stem.upper()
            if ticker and len(ticker) <= 10 and ticker.isalpha():
                images[ticker] = img_file
                logging.info(f"Found image for {ticker}")
        return images
        
    def load_excel_data(self, excel_path: Path) -> pd.DataFrame:
        """Load and validate Excel data."""
        if not excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
            
        df = pd.read_excel(excel_path)
        
        # Map the actual column names to expected format
        column_mapping = {
            'symbol': 'Symbol',
            'price_low': 'Price Low',
            'price_high': 'Price High',
            'bullish_target': 'Bullish Target',
            'bearish_target': 'Bearish Target',
            'bullish_statement': 'Bullish Statement',
            'bearish_statement': 'Bearish Statement',
            'rank': 'Rank'
        }
        
        # Check if we have all required columns (in any case/format)
        df_columns_lower = [col.lower().strip() for col in df.columns]
        missing = []
        
        for snake_case, title_case in column_mapping.items():
            if snake_case not in df_columns_lower:
                missing.append(title_case)
        
        if missing:
            print(f"\nExpected columns (snake_case): {list(column_mapping.keys())}")
            print(f"Actual columns: {list(df.columns)}")
            print(f"Missing: {missing}")
            raise ValueError(f"Missing required columns: {missing}")
        
        # Rename columns to match expected format
        df = df.rename(columns=column_mapping)
        
        # Standardize symbol column
        df['Symbol'] = df['Symbol'].str.upper().str.strip()
        
        # Validate rank values
        valid_ranks = ['primary', 'secondary', 'tertiary']
        df['Rank'] = df['Rank'].str.lower().str.strip()
        invalid_ranks = df[~df['Rank'].isin(valid_ranks)]
        if not invalid_ranks.empty:
            logging.warning(f"Invalid rank values found: {invalid_ranks['Rank'].unique()}")
            
        return df
        
    def calculate_risk_reward(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk/reward ratios for each play."""
        # Bullish R:R = (Target - Entry) / (Entry - Stop)
        # Entry = Price High, Stop = Price Low
        df['Bullish_RR'] = (df['Bullish Target'] - df['Price High']) / (df['Price High'] - df['Price Low'])
        
        # Bearish R:R = (Entry - Target) / (Stop - Entry)
        # Entry = Price Low, Stop = Price High
        df['Bearish_RR'] = (df['Price Low'] - df['Bearish Target']) / (df['Price High'] - df['Price Low'])
        
        # Handle infinite or invalid R:R
        df['Bullish_RR'] = df['Bullish_RR'].replace([np.inf, -np.inf], 0)
        df['Bearish_RR'] = df['Bearish_RR'].replace([np.inf, -np.inf], 0)
        
        # Round to 2 decimal places
        df['Bullish_RR'] = df['Bullish_RR'].round(2)
        df['Bearish_RR'] = df['Bearish_RR'].round(2)
        
        # Log any concerning R:R values
        low_rr = df[(df['Bullish_RR'] < 0.5) | (df['Bearish_RR'] < 0.5)]
        if not low_rr.empty:
            logging.warning(f"Low R:R ratios found for: {low_rr['Symbol'].tolist()}")
            
        return df
        
    def check_existing_data(self) -> bool:
        """Check if data already exists for today."""
        try:
            result = self.supabase.table('daily_playbook').select("id").eq(
                'playbook_date', str(self.today)
            ).limit(1).execute()
            
            return len(result.data) > 0
        except Exception as e:
            logging.error(f"Error checking existing data: {e}")
            return False
            
    def clear_today_data(self):
        """Clear existing data for today."""
        try:
            self.supabase.table('daily_playbook').delete().eq(
                'playbook_date', str(self.today)
            ).execute()
            logging.info(f"Cleared existing data for {self.today}")
        except Exception as e:
            logging.error(f"Error clearing today's data: {e}")
            raise
            
    def save_to_supabase(self, df: pd.DataFrame) -> bool:
        """Save data to Supabase."""
        try:
            # Prepare records for insertion
            records = []
            for _, row in df.iterrows():
                record = {
                    'playbook_date': str(self.today),
                    'symbol': row['Symbol'],
                    'price_low': float(row['Price Low']),
                    'price_high': float(row['Price High']),
                    'bullish_target': float(row['Bullish Target']),
                    'bearish_target': float(row['Bearish Target']),
                    'bullish_statement': row['Bullish Statement'],
                    'bearish_statement': row['Bearish Statement'],
                    'rank': row['Rank'],
                    'bullish_rr': float(row['Bullish_RR']),
                    'bearish_rr': float(row['Bearish_RR'])
                }
                records.append(record)
                
            # Insert all records
            result = self.supabase.table('daily_playbook').insert(records).execute()
            
            logging.info(f"Successfully saved {len(records)} plays to Supabase")
            return True
            
        except Exception as e:
            logging.error(f"Error saving to Supabase: {str(e)}")
            return False
            
    def generate_summary(self, df: pd.DataFrame, images: Dict[str, Path]):
        """Generate ingestion summary."""
        print("\n" + "="*60)
        print(f"DAILY PLAYBOOK INGESTION SUMMARY - {self.today}")
        print("="*60)
        
        # Symbol summary
        symbols = df['Symbol'].unique()
        print(f"\nSymbols processed: {', '.join(symbols)}")
        print(f"Total plays: {len(df)}")
        
        # Rank breakdown
        rank_counts = df['Rank'].value_counts()
        print("\nPlays by rank:")
        for rank, count in rank_counts.items():
            print(f"  - {rank.capitalize()}: {count}")
            
        # Image status
        print("\nImage status:")
        for symbol in symbols:
            status = "✓ Found" if symbol in images else "✗ Missing"
            print(f"  - {symbol}: {status}")
            
        # R:R summary
        print("\nRisk/Reward summary:")
        print(f"  - Average Bullish R:R: {df['Bullish_RR'].mean():.2f}")
        print(f"  - Average Bearish R:R: {df['Bearish_RR'].mean():.2f}")
        
        print("="*60 + "\n")
        
    def run(self, dropbox_path: str, force_update: bool = False):
        """Main execution method."""
        try:
            folder_path = Path(dropbox_path)
            if not folder_path.exists():
                raise ValueError(f"Folder path does not exist: {folder_path}")
                
            # Check for existing data
            if self.check_existing_data() and not force_update:
                response = input(f"\nData already exists for {self.today}. Overwrite? (y/n): ")
                if response.lower() != 'y':
                    logging.info("Ingestion cancelled by user")
                    return None
                self.clear_today_data()
                
            # Load Excel file
            excel_path = folder_path / "daily_playbook.xlsx"
            logging.info(f"Loading Excel file: {excel_path}")
            df = self.load_excel_data(excel_path)
            
            # Find available images
            logging.info("Scanning for ticker images...")
            images = self.find_available_images(folder_path)
            
            # Calculate risk/reward
            logging.info("Calculating risk/reward ratios...")
            df = self.calculate_risk_reward(df)
            
            # Save to Supabase
            logging.info("Saving to Supabase...")
            if self.save_to_supabase(df):
                # Generate summary
                self.generate_summary(df, images)
                logging.info("Ingestion completed successfully!")
                
                # Return processed data for PDF generation
                return {
                    'dataframe': df,
                    'images': images,
                    'date': self.today,
                    'folder_path': str(folder_path)
                }
            else:
                raise Exception("Failed to save data to Supabase")
                
        except Exception as e:
            logging.error(f"Ingestion failed: {str(e)}")
            raise

def generate_pdf_report(dropbox_path: str, supabase_url: str, supabase_key: str):
    """Generate PDF report using the playbook_report module."""
    try:
        # Add current directory to Python path to ensure import works
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        # Import the report generator
        from playbook_report import DailyPlaybookReportGenerator
        
        print("\nGenerating PDF report...")
        generator = DailyPlaybookReportGenerator(supabase_url, supabase_key)
        pdf_path = generator.run(dropbox_path)
        return pdf_path
        
    except ImportError as e:
        logging.error(f"Could not import playbook_report.py: {e}")
        print("\nError: Could not import playbook_report.py")
        print(f"Looking in directory: {Path(__file__).parent}")
        print(f"Files in directory: {list(Path(__file__).parent.glob('*.py'))}")
        return None
    except Exception as e:
        logging.error(f"PDF generation failed: {e}")
        print(f"\nError generating PDF: {e}")
        return None

def main():
    """Main entry point."""
    # Get Supabase credentials
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_KEY not found")
        print("\nPlease ensure you have a .env file in your project root with:")
        print("SUPABASE_URL=your-supabase-url")
        print("SUPABASE_KEY=your-supabase-key")
        print(f"\nSearched for .env in: {find_project_root() or 'project root not found'}")
        sys.exit(1)
        
    # Get Dropbox path
    if len(sys.argv) > 1:
        dropbox_path = sys.argv[1]
    else:
        dropbox_path = input("Enter the Dropbox folder path: ").strip()
        
    # Check for force update flag
    force_update = '--force' in sys.argv
    
    # Run ingestion
    processor = DailyPlaybookIngestion(supabase_url, supabase_key)
    result = processor.run(dropbox_path, force_update)
    
    if result:
        print("\nData ingested successfully!")
        
        # Ask if user wants to generate PDF
        generate_pdf = input("\nWould you like to generate the PDF report now? (y/n): ").strip().lower()
        
        if generate_pdf == 'y':
            pdf_path = generate_pdf_report(result['folder_path'], supabase_url, supabase_key)
            if pdf_path:
                print(f"\n✓ PDF report generated successfully!")
                print(f"Location: {pdf_path}")
            else:
                print("\n✗ PDF generation failed. You can run playbook_report.py separately.")
        else:
            print("\nYou can generate the PDF later by running:")
            print(f"python playbook_report.py \"{result['folder_path']}\"")

if __name__ == "__main__":
    main()