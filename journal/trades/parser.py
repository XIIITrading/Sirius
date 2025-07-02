# journal/trades/parser.py
"""Parse broker statement CSV files with CLOID support."""

import csv
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

# Handle both package and script imports
try:
    from .models import Execution
except ImportError:
    from models import Execution

# Timezone handling
try:
    from zoneinfo import ZoneInfo
except ImportError:
    import pytz
    ZoneInfo = None


class TradeParser:
    """Parse trade executions from broker CSV files."""
    
    def __init__(self, trade_date: Optional[date] = None):
        self.time_format = "%H:%M:%S"
        self.trade_date = trade_date or date.today()
        
        # Setup timezone handling
        if ZoneInfo:
            self.et_timezone = ZoneInfo('America/New_York')
            self.utc_timezone = ZoneInfo('UTC')
        else:
            self.et_timezone = pytz.timezone('America/New_York')
            self.utc_timezone = pytz.UTC
        
    def parse_csv(self, file_path: Path, trade_date: Optional[date] = None) -> List[Execution]:
        """
        Parse CSV file and return list of executions.
        
        Args:
            file_path: Path to the CSV file
            trade_date: Optional date for the trades (if not provided, uses instance default)
        """
        executions = []
        current_trade_date = trade_date or self.trade_date
        
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                # Skip empty rows
                if not row.get('Symbol'):
                    continue
                    
                execution = self._parse_row(row, current_trade_date)
                if execution:
                    executions.append(execution)
                    
        return sorted(executions, key=lambda x: x.time)
    
    def _parse_row(self, row: dict, trade_date: date) -> Optional[Execution]:
        """Parse a single row into an Execution object."""
        try:
            # Map the new format to our Execution model
            # Side: SS (Short Sale), B (Buy)
            side = 'S' if row['Side'] == 'SS' else row['Side']
            
            execution = Execution(
                event='Execute',  # All rows in this format are executions
                side=side,
                symbol=row['Symbol'],
                shares=int(row['Qty']),
                price=Decimal(str(row['Price'])),
                route=row['Route'],
                time=self._parse_time(row['Time'], trade_date),
                account=row['Account'],
                note=f"Type: {row['Type']}, CLOID: {row['Cloid']}"  # Store CLOID in note
            )
            
            # Add CLOID as an attribute for easier access
            execution.cloid = row['Cloid']
            
            return execution
            
        except (KeyError, ValueError) as e:
            print(f"Error parsing row: {e}, Row: {row}")
            return None
    
    def _parse_time(self, time_str: str, trade_date: date) -> datetime:
        """
        Parse time string to datetime and convert from ET to UTC.
        
        Args:
            time_str: Time string in HH:MM:SS format
            trade_date: The date of the trade
        """
        # Parse the time
        time_obj = datetime.strptime(time_str, self.time_format).time()
        
        # Combine with the trade date to create a datetime
        naive_datetime = datetime.combine(trade_date, time_obj)
        
        if ZoneInfo:
            # Using zoneinfo (Python 3.9+)
            et_aware = naive_datetime.replace(tzinfo=self.et_timezone)
            utc_datetime = et_aware.astimezone(self.utc_timezone)
        else:
            # Using pytz
            et_datetime = self.et_timezone.localize(naive_datetime)
            utc_datetime = et_datetime.astimezone(self.utc_timezone)
        
        # Return as timezone-naive UTC
        return utc_datetime.replace(tzinfo=None)