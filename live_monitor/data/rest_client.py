# live_monitor/data/rest_client.py
"""
REST API client for Polygon server
Handles historical data fetching and other REST endpoints
All timestamps in UTC
"""
import logging
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
import time

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class PolygonRESTClient(QObject):
    """
    REST API client for Polygon server
    
    Handles fetching historical bars and other REST operations
    All operations use UTC timestamps
    """
    
    # Signals
    data_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, base_url: str = "http://localhost:8200"):
        super().__init__()
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
    def fetch_bars(self, symbol: str, timespan: str = '1min', 
               multiplier: int = 1, limit: int = 200) -> Optional[List[Dict]]:
        """
        Fetch historical bars from REST API
        
        Args:
            symbol: Stock symbol
            timespan: 1min, 5min, 15min, 30min, 1hour, 4hour, 1day, 1week, 1month
            multiplier: Size of the timespan multiplier
            limit: Number of bars to fetch
            
        Returns:
            List of bar dictionaries or None if error
        """
        try:
            # Build URL for POST endpoint
            url = f"{self.base_url}/api/v1/bars"
            
            # Calculate date range
            end_date = datetime.now(timezone.utc)
            
            # For minute bars, go back enough days to get required bars
            if timespan in ['1min', '5min', '15min', '30min']:
                # ~390 minutes per trading day
                days_needed = max(3, (limit * multiplier) // 390 + 2)
                start_date = end_date - timedelta(days=days_needed)
            else:
                start_date = end_date - timedelta(days=limit * multiplier)
            
            # Create request body
            request_data = {
                "symbol": symbol.upper(),
                "timeframe": timespan,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "limit": limit,
                "use_cache": True,
                "validate": False
            }
            
            logger.info(f"Fetching bars: {url} with data: {request_data}")
            
            # Make POST request
            response = self.session.post(url, json=request_data, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('data'):
                    bars = []
                    # Convert to our standard format
                    for bar in data['data']:
                        # Parse timestamp - it's already in ISO format
                        timestamp = datetime.fromisoformat(
                            bar['timestamp'].replace('Z', '+00:00')
                        )
                        
                        bars.append({
                            'timestamp': timestamp,
                            'open': bar['open'],
                            'high': bar['high'],
                            'low': bar['low'],
                            'close': bar['close'],
                            'volume': bar['volume'],
                            'vwap': bar.get('vwap', 0),
                            'trades': bar.get('transactions', 0)
                        })
                    
                    logger.info(f"Fetched {len(bars)} bars for {symbol}")
                    return bars
                else:
                    logger.warning(f"No data in response for {symbol}")
                    return []
            else:
                logger.error(f"HTTP {response.status_code}: {response.text}")
                self.error_occurred.emit(f"Failed to fetch bars: HTTP {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("Request timeout")
            self.error_occurred.emit("Request timeout - server may be busy")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching bars: {e}")
            self.error_occurred.emit(f"Error fetching bars: {str(e)}")
            return None
    
    def is_market_open(self) -> bool:
        """
        Check if market is currently open (regular hours only)
        All times in UTC
        
        Regular Hours: 14:30 - 21:00 UTC (Mon-Fri)
        """
        now = datetime.now(timezone.utc)
        
        # Check if weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check market hours in UTC
        market_open = now.replace(hour=14, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=21, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def get_market_session(self) -> str:
        """
        Get current market session
        
        Returns: 'pre', 'regular', 'after', or 'closed'
        """
        now = datetime.now(timezone.utc)
        
        # Check if weekend
        if now.weekday() >= 5:
            return 'closed'
        
        # Define sessions in UTC
        pre_market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        regular_start = now.replace(hour=13, minute=30, second=0, microsecond=0)
        regular_end = now.replace(hour=21, minute=0, second=0, microsecond=0)
        after_hours_end = now.replace(hour=1, minute=0, second=0, microsecond=0)
        
        # Handle after-hours crossing midnight
        if now.hour < 2:  # Early morning, could be end of previous day's after-hours
            if now <= after_hours_end:
                return 'after'
        
        if pre_market_start <= now < regular_start:
            return 'pre'
        elif regular_start <= now < regular_end:
            return 'regular'
        elif now >= regular_end:
            return 'after'
        else:
            return 'closed'