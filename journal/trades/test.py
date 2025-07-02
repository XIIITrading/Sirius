# journal/trades/test_trades.py
"""Standalone test for trade processing - run from trades directory."""

import os
import sys
from pathlib import Path
from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional
import json

# Import the modules directly since we're in the same directory
from models import Trade, Execution
from parser import TradeParser
from processor import TradeProcessor
from plugin import TradePlugin


class TradeTestRunner:
    """Test runner for trade processing workflow."""
    
    def __init__(self):
        self.plugin = None
        self.trades = []
        
    def setup_plugin(self, use_database: bool = False):
        """Setup the trade plugin with optional database connection."""
        if use_database:
            supabase_url = os.environ.get('SUPABASE_URL')
            supabase_key = os.environ.get('SUPABASE_KEY')
            
            if not supabase_url or not supabase_key:
                print("\n⚠️  Supabase credentials not found in environment variables.")
                print("   Set SUPABASE_URL and SUPABASE_KEY to enable database saving.")
                return False
                
            self.plugin = TradePlugin(supabase_url, supabase_key)
        else:
            # Create plugin without database
            self.plugin = TradePlugin()
            
        return True
    
    def find_csv_files(self, directory: Path) -> List[Path]:
        """Find all CSV files in the given directory."""
        csv_files = []
        
        if directory.is_file() and directory.suffix.lower() == '.csv':
            return [directory]
            
        if directory.is_dir():
            csv_files = list(directory.glob('*.csv')) + list(directory.glob('*.CSV'))
            
        return sorted(csv_files)
    
    def get_file_path(self) -> Optional[Path]:
        """Prompt user for file path and handle various input formats."""
        print("\n📁 Enter the path to your CSV file or directory:")
        print("   (You can paste the full path from File Explorer/Finder)")
        
        user_input = input("\n→ Path: ").strip()
        
        # Remove quotes if present (common when copying from file explorer)
        user_input = user_input.strip('"').strip("'")
        
        # Handle Windows paths
        if sys.platform == 'win32':
            user_input = user_input.replace('\\', '/')
        
        # Expand user home directory
        user_input = os.path.expanduser(user_input)
        
        path = Path(user_input)
        
        if not path.exists():
            print(f"\n❌ Path not found: {path}")
            return None
            
        return path
    
    def select_file(self, files: List[Path]) -> Optional[Path]:
        """Let user select from multiple CSV files."""
        print(f"\n📋 Found {len(files)} CSV file(s):")
        
        for i, file in enumerate(files, 1):
            print(f"   {i}. {file.name}")
            
        while True:
            try:
                choice = input("\n→ Select file number (or 'q' to quit): ").strip()
                
                if choice.lower() == 'q':
                    return None
                    
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    return files[idx]
                else:
                    print("❌ Invalid selection. Please try again.")
                    
            except ValueError:
                print("❌ Please enter a number or 'q' to quit.")
    
    def get_trade_date(self, file_name: str) -> date:
        """Extract or prompt for trade date."""
        print(f"\n📅 Processing file: {file_name}")
        
        # Try to extract date from filename (e.g., 070125.DAS.TL.csv)
        import re
        date_match = re.search(r'(\d{6})', file_name)
        
        if date_match:
            date_str = date_match.group(1)
            try:
                # Assuming MMDDYY format
                month = int(date_str[:2])
                day = int(date_str[2:4])
                year = 2000 + int(date_str[4:6])
                
                suggested_date = date(year, month, day)
                print(f"   Detected date: {suggested_date.strftime('%B %d, %Y')}")
                
                confirm = input("   Use this date? (Y/n): ").strip().lower()
                if confirm != 'n':
                    return suggested_date
                    
            except ValueError:
                pass
        
        # Manual date entry
        while True:
            date_input = input("   Enter trade date (YYYY-MM-DD): ").strip()
            try:
                return datetime.strptime(date_input, "%Y-%m-%d").date()
            except ValueError:
                print("❌ Invalid date format. Please use YYYY-MM-DD")
    
    def display_trades(self, trades: List[Trade]):
        """Display trades in a formatted table."""
        if not trades:
            print("\n❌ No trades found in the file.")
            return
            
        print(f"\n📊 Processed {len(trades)} trade(s):\n")
        print("-" * 100)
        print(f"{'Symbol':<10} {'Entry Time (UTC)':<20} {'Exit Time (UTC)':<20} "
              f"{'Qty':<8} {'Entry $':<10} {'Exit $':<10} {'P&L':<12}")
        print("-" * 100)
        
        total_pl = Decimal('0')
        
        for trade in trades:
            entry_time = trade.entry_time.strftime("%Y-%m-%d %H:%M:%S")
            exit_time = trade.exit_time.strftime("%Y-%m-%d %H:%M:%S") if trade.exit_time else "OPEN"
            
            entry_price = f"${trade.entry_price:.2f}"
            exit_price = f"${trade.exit_price:.2f}" if trade.exit_price else "---"
            
            if trade.net_pl is not None:
                pl_str = f"${trade.net_pl:,.2f}"
                if trade.net_pl >= 0:
                    pl_str = f"✅ {pl_str}"
                else:
                    pl_str = f"❌ {pl_str}"
                total_pl += trade.net_pl
            else:
                pl_str = "OPEN"
            
            print(f"{trade.symbol:<10} {entry_time:<20} {exit_time:<20} "
                  f"{trade.qty:<8} {entry_price:<10} {exit_price:<10} {pl_str:<12}")
        
        print("-" * 100)
        
        # Summary statistics
        closed_trades = [t for t in trades if t.net_pl is not None]
        if closed_trades:
            winning_trades = [t for t in closed_trades if t.net_pl > 0]
            losing_trades = [t for t in closed_trades if t.net_pl < 0]
            
            print(f"\n📈 Summary:")
            print(f"   Total Trades: {len(trades)} ({len(closed_trades)} closed, {len(trades) - len(closed_trades)} open)")
            print(f"   Winning Trades: {len(winning_trades)}")
            print(f"   Losing Trades: {len(losing_trades)}")
            print(f"   Win Rate: {len(winning_trades)/len(closed_trades)*100:.1f}%" if closed_trades else "N/A")
            
            total_str = f"${total_pl:,.2f}"
            if total_pl >= 0:
                print(f"   Total P&L: ✅ {total_str}")
            else:
                print(f"   Total P&L: ❌ {total_str}")
    
    def save_to_json(self, trades: List[Trade], file_path: Path):
        """Save trades to JSON for inspection."""
        output_file = file_path.stem + "_processed.json"
        
        data = []
        for trade in trades:
            trade_dict = trade.to_dict()
            # Add execution details
            trade_dict['executions'] = [
                {
                    'time': exec.time.strftime("%Y-%m-%d %H:%M:%S"),
                    'side': exec.side,
                    'shares': exec.shares,
                    'price': float(exec.price)
                }
                for exec in trade.executions
            ]
            data.append(trade_dict)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"\n💾 Saved detailed results to: {output_file}")
    
    def run_workflow(self):
        """Run the complete test workflow."""
        print("\n🚀 Trade Processing Test Workflow")
        print("=" * 50)
        
        # Step 1: Get file path
        path = self.get_file_path()
        if not path:
            return
        
        # Step 2: Find CSV files
        csv_files = self.find_csv_files(path)
        if not csv_files:
            print("\n❌ No CSV files found in the specified location.")
            return
        
        # Step 3: Select file if multiple
        if len(csv_files) > 1:
            selected_file = self.select_file(csv_files)
            if not selected_file:
                return
        else:
            selected_file = csv_files[0]
            print(f"\n📋 Found file: {selected_file.name}")
        
        # Step 4: Get trade date
        trade_date = self.get_trade_date(selected_file.name)
        
        # Step 5: Setup plugin
        print("\n⚙️  Setting up trade processor...")
        use_db = input("   Connect to Supabase? (y/N): ").strip().lower() == 'y'
        
        if not self.setup_plugin(use_database=use_db):
            print("   Running in local mode (no database)...")
            self.setup_plugin(use_database=False)
        
        # Step 6: Process trades
        print("\n🔄 Processing trades...")
        try:
            self.trades = self.plugin.process_file(selected_file, trade_date)
            print(f"   ✅ Successfully processed {len(self.trades)} trades")
        except Exception as e:
            print(f"   ❌ Error processing file: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 7: Display results
        self.display_trades(self.trades)
        
        # Step 8: Save options
        print("\n💾 Save Options:")
        print("   1. Save to JSON (local file)")
        if self.plugin.db:
            print("   2. Save to Supabase")
        print("   3. Skip saving")
        
        choice = input("\n→ Choose option (1-3): ").strip()
        
        if choice == '1':
            self.save_to_json(self.trades, selected_file)
        elif choice == '2' and self.plugin.db:
            print("\n🔄 Saving to Supabase...")
            result = self.plugin.save_trades(self.trades)
            if result['success']:
                print(f"   ✅ Successfully saved {result['count']} trades to database")
            else:
                print(f"   ❌ Error saving to database: {result['error']}")
        
        print("\n✨ Test workflow complete!")


def main():
    """Main entry point."""
    runner = TradeTestRunner()
    
    try:
        runner.run_workflow()
        
        # Offer to process another file
        while True:
            again = input("\n\n🔄 Process another file? (y/N): ").strip().lower()
            if again == 'y':
                runner.run_workflow()
            else:
                break
                
    except KeyboardInterrupt:
        print("\n\n👋 Workflow cancelled by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()