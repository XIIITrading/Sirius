"""Direct Polygon API client for market data fetching"""
import os
import time
import logging
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class PolygonAPIClient:
    """Handles direct communication with Polygon API v3"""
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_per_second: float = 65):
        self.api_key = api_key or os.environ.get('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon API key not provided")
            
        self.base_url = "https://api.polygon.io"
        self.last_api_call = 0
        self.min_call_interval = 1.0 / rate_limit_per_second
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {self.api_key}'})
        
    async def fetch_bars(self, symbol: str, start_time: datetime, 
                        end_time: datetime, timeframe: str = '1min') -> Optional[pd.DataFrame]:
        """Fetch bar data from Polygon aggregates endpoint"""
        try:
            # Map timeframe to Polygon format
            timeframe_map = {
                '1min': ('1', 'minute'),
                '5min': ('5', 'minute'),
                '15min': ('15', 'minute'),
                '30min': ('30', 'minute'),
                '1hour': ('1', 'hour'),
                'hour': ('1', 'hour'),
                'day': ('1', 'day')
            }
            
            multiplier, timespan = timeframe_map.get(timeframe, ('1', 'minute'))
            
            # Format timestamps
            from_ts = int(start_time.timestamp() * 1000)
            to_ts = int(end_time.timestamp() * 1000)
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_ts}/{to_ts}"
            
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apiKey': self.api_key,
                'extended': 'true'
            }
            
            all_bars = []
            
            while url:
                self._rate_limit()
                response = self.session.get(url, params=params)
                
                if response.status_code != 200:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    break
                
                data = response.json()
                
                if data.get('status') != 'OK' or 'results' not in data:
                    break
                
                # Process bars
                for bar in data['results']:
                    all_bars.append({
                        'timestamp': pd.Timestamp(bar['t'], unit='ms', tz='UTC'),
                        'open': bar['o'],
                        'high': bar['h'],
                        'low': bar['l'],
                        'close': bar['c'],
                        'volume': bar['v']
                    })
                
                # Check for next page
                url = data.get('next_url')
                params = {'apiKey': self.api_key} if url else None
            
            if all_bars:
                df = pd.DataFrame(all_bars)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                logger.info(f"Fetched {len(df)} bars for {symbol}")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching bars from Polygon: {e}")
            return None
            
    async def fetch_trades(self, symbol: str, start_time: datetime,
                          end_time: datetime) -> Optional[pd.DataFrame]:
        """Fetch trade tick data from Polygon"""
        try:
            all_trades = []
            url = f"{self.base_url}/v3/trades/{symbol}"
            
            params = {
                'apiKey': self.api_key,
                'timestamp.gte': start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'timestamp.lt': end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'limit': 50000,
                'order': 'asc'
            }
            
            while url:
                self._rate_limit()
                response = self.session.get(url, params=params)
                
                if response.status_code != 200:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    break
                
                data = response.json()
                
                if 'results' not in data or not data['results']:
                    break
                
                # Process trades
                for trade in data['results']:
                    all_trades.append({
                        'timestamp': pd.Timestamp(trade['participant_timestamp'], unit='ns', tz='UTC'),
                        'price': trade['price'],
                        'size': trade['size'],
                        'conditions': ','.join(map(str, trade.get('conditions', []))),
                        'exchange': str(trade.get('exchange', ''))
                    })
                
                # Check for next page
                url = data.get('next_url')
                params = {'apiKey': self.api_key} if url else None
            
            if all_trades:
                df = pd.DataFrame(all_trades)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                logger.info(f"Fetched {len(df)} trades for {symbol}")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching trades from Polygon: {e}")
            return None
            
    async def fetch_quotes(self, symbol: str, start_time: datetime,
                          end_time: datetime) -> Optional[pd.DataFrame]:
        """Fetch quote (NBBO) data from Polygon"""
        try:
            all_quotes = []
            url = f"{self.base_url}/v3/quotes/{symbol}"
            
            params = {
                'apiKey': self.api_key,
                'timestamp.gte': start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'timestamp.lt': end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'limit': 50000,
                'order': 'asc'
            }
            
            while url:
                self._rate_limit()
                response = self.session.get(url, params=params)
                
                if response.status_code != 200:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    break
                
                data = response.json()
                
                if 'results' not in data or not data['results']:
                    break
                
                # Process quotes
                for quote in data['results']:
                    all_quotes.append({
                        'timestamp': pd.Timestamp(quote['participant_timestamp'], unit='ns', tz='UTC'),
                        'bid': quote.get('bid_price', 0),
                        'ask': quote.get('ask_price', 0),
                        'bid_size': quote.get('bid_size', 0),
                        'ask_size': quote.get('ask_size', 0),
                        'bid_exchange': quote.get('bid_exchange', 0),
                        'ask_exchange': quote.get('ask_exchange', 0)
                    })
                
                # Check for next page
                url = data.get('next_url')
                params = {'apiKey': self.api_key} if url else None
            
            if all_quotes:
                df = pd.DataFrame(all_quotes)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                logger.info(f"Fetched {len(df)} quotes for {symbol}")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching quotes from Polygon: {e}")
            return None
            
    def _rate_limit(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_call_interval:
            sleep_time = self.min_call_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()