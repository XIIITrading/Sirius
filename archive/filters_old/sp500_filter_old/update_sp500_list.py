# modules/filters/sp500_filter/update_sp500_list.py
"""
Utility to help update the S&P 500 ticker list from Wikipedia.
Run this quarterly or when notified of changes.
"""

import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
import sys

def fetch_current_sp500():
    """Fetch current S&P 500 list from Wikipedia."""
    print("Fetching current S&P 500 list from Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    try:
        # Method 1: Try pandas read_html first
        print("Attempting to fetch using pandas...")
        tables = pd.read_html(url)
        df = tables[0]
        
        # Extract tickers
        tickers = sorted(df['Symbol'].str.strip().tolist())
        
        # Handle special cases (. to - conversion for compatibility)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        print(f"Successfully fetched {len(tickers)} tickers using pandas")
        return tickers
        
    except Exception as e:
        print(f"Pandas method failed: {e}")
        print("Attempting alternative method using BeautifulSoup...")
        
        try:
            # Method 2: BeautifulSoup as fallback
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the S&P 500 component table
            table = soup.find('table', {'id': 'constituents'})
            if not table:
                # Try alternative table selection
                tables = soup.find_all('table', {'class': 'wikitable'})
                table = tables[0] if tables else None
            
            if not table:
                raise Exception("Could not find S&P 500 table on Wikipedia")
            
            tickers = []
            rows = table.find_all('tr')[1:]  # Skip header
            
            for row in rows:
                cells = row.find_all('td')
                if cells:
                    # First cell usually contains the ticker
                    ticker = cells[0].text.strip()
                    # Clean up ticker
                    ticker = re.sub(r'\[.*?\]', '', ticker)  # Remove references
                    ticker = ticker.replace('.', '-')  # Convert dots to dashes
                    if ticker and len(ticker) <= 5:  # Basic validation
                        tickers.append(ticker)
            
            tickers = sorted(list(set(tickers)))  # Remove duplicates and sort
            print(f"Successfully fetched {len(tickers)} tickers using BeautifulSoup")
            return tickers
            
        except Exception as e2:
            print(f"BeautifulSoup method also failed: {e2}")
            return None

def generate_python_list(tickers):
    """Format tickers as a Python list for copy-paste."""
    # Format as Python list with 10 tickers per line
    lines = []
    for i in range(0, len(tickers), 10):
        chunk = tickers[i:i+10]
        line = "    " + ", ".join(f"'{t}'" for t in chunk) + ","
        lines.append(line)
    
    # Remove trailing comma from last line
    if lines:
        lines[-1] = lines[-1].rstrip(',')
    
    return "[\n" + "\n".join(lines) + "\n]"

def compare_with_current(new_tickers):
    """Compare new ticker list with current list to show changes."""
    try:
        from sp500_tickers import SP500_TICKERS as current_tickers
        
        current_set = set(current_tickers)
        new_set = set(new_tickers)
        
        added = new_set - current_set
        removed = current_set - new_set
        
        print("\n" + "="*50)
        print("CHANGES DETECTED:")
        print("="*50)
        
        if added:
            print(f"\nAdded ({len(added)} tickers):")
            for ticker in sorted(added):
                print(f"  + {ticker}")
        
        if removed:
            print(f"\nRemoved ({len(removed)} tickers):")
            for ticker in sorted(removed):
                print(f"  - {ticker}")
        
        if not added and not removed:
            print("\nNo changes detected - list is up to date!")
        
        print("="*50 + "\n")
        
    except ImportError:
        print("\nCould not import current ticker list for comparison.")
        print("This might be the first time running this script.\n")

def save_backup():
    """Save a backup of the current ticker list."""
    try:
        from sp500_tickers import SP500_TICKERS, LAST_UPDATED
        
        backup_filename = f"sp500_tickers_backup_{LAST_UPDATED.replace('-', '')}.py"
        with open(backup_filename, 'w') as f:
            f.write(f"# Backup of S&P 500 tickers from {LAST_UPDATED}\n")
            f.write(f"SP500_TICKERS = {generate_python_list(SP500_TICKERS)}\n")
        
        print(f"Backup saved to: {backup_filename}")
        
    except ImportError:
        print("Could not create backup - no existing ticker list found.")

def main():
    """Main update process."""
    print("S&P 500 List Updater")
    print("=" * 50)
    
    # Save backup of current list
    save_backup()
    
    # Fetch current list
    tickers = fetch_current_sp500()
    if not tickers:
        print("\nERROR: Could not fetch S&P 500 list. Please try again later.")
        return
    
    print(f"\nFound {len(tickers)} tickers")
    print(f"Sample: {tickers[:5]}...")
    
    # Compare with current list
    compare_with_current(tickers)
    
    # Generate formatted list
    formatted_list = generate_python_list(tickers)
    
    # Generate update info
    today = datetime.now().strftime("%Y-%m-%d")
    
    print("=" * 50)
    print("INSTRUCTIONS:")
    print("1. Review the changes above")
    print("2. Copy the code below")
    print("3. Replace the content in sp500_tickers.py")
    print("=" * 50 + "\n")
    
    print(f'LAST_UPDATED = "{today}"')
    print(f"\nSP500_TICKERS = {formatted_list}")
    
    # Optionally save to file
    save_to_file = input("\nSave to sp500_tickers_new.py? (y/n): ").lower().strip()
    if save_to_file == 'y':
        with open('sp500_tickers_new.py', 'w') as f:
            f.write('# modules/filters/sp500_filter/sp500_tickers.py\n')
            f.write('"""\n')
            f.write('S&P 500 Ticker List\n')
            f.write('Manually updated from: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies\n')
            f.write('"""\n\n')
            f.write('from datetime import datetime, timedelta\n')
            f.write('import warnings\n\n')
            f.write('# CRITICAL: Update this when you update the ticker list\n')
            f.write(f'LAST_UPDATED = "{today}"  # ISO format YYYY-MM-DD\n')
            f.write('UPDATE_FREQUENCY_DAYS = 90  # Remind to update quarterly\n\n')
            f.write(f'# S&P 500 Tickers as of {datetime.now().strftime("%B %d, %Y")}\n')
            f.write('# Source: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies\n')
            f.write(f'SP500_TICKERS = {formatted_list}\n')
        
        print(f"\n✓ New ticker list saved to: sp500_tickers_new.py")
        print("  Review and rename to sp500_tickers.py when ready.")
    
    print(f"\n✓ Update complete. Found {len(tickers)} tickers.")
    
    # Show summary statistics
    print("\nSummary:")
    print(f"  Total tickers: {len(tickers)}")
    print(f"  Shortest ticker: {min(tickers, key=len)} ({len(min(tickers, key=len))} chars)")
    print(f"  Longest ticker: {max(tickers, key=len)} ({len(max(tickers, key=len))} chars)")
    
    # Check for potential issues
    issues = [t for t in tickers if len(t) > 5 or not t.replace('-', '').isalnum()]
    if issues:
        print(f"\nPotential issues found with {len(issues)} tickers:")
        for ticker in issues:
            print(f"  ? {ticker}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nUpdate cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)