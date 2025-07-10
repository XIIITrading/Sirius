WEBSOCKET_GUIDE

## 4. WEBSOCKET_GUIDE.md

```markdown
# Polygon WebSocket Client Guide

## Overview

The WebSocket client provides real-time streaming data from Polygon.io. It handles authentication, subscriptions, automatic reconnection, and data processing.

## Client Initialization

```python
from polygon.websocket import PolygonWebSocketClient
from polygon.config import get_config

# Initialize client
config = get_config()
client = PolygonWebSocketClient(config)

# Or use context manager
async with PolygonWebSocketClient() as client:
    # Client connects automatically
    await client.subscribe(['AAPL'], ['T', 'Q'], callback)
Connection Management
Manual Connection
python# Connect
await client.connect()

# Check status
status = client.get_status()
print(f"Connected: {status['connected']}")
print(f"Authenticated: {status['authenticated']}")

# Disconnect
await client.disconnect()
Status Information
pythonstatus = client.get_status()
# Returns:
{
    'connected': True,
    'authenticated': True,
    'running': True,
    'subscriptions': {
        'AAPL': ['T', 'Q'],
        'MSFT': ['T']
    },
    'message_count': 1500,
    'last_message_time': '2023-12-01T15:30:45',
    'uptime_seconds': 3600.5,
    'reconnect_attempts': 0
}
Subscriptions
Channel Types

T: Trades
Q: Quotes (bid/ask)
A: Aggregate bars
AM: Aggregate minute bars

Subscribe to Data
python# Define callback function
async def handle_trade(data):
    print(f"Trade: {data['symbol']} @ ${data['price']}")

# Subscribe to single symbol
sub_id = await client.subscribe(
    symbols=['AAPL'],
    channels=['T', 'Q'],
    callback=handle_trade
)

# Subscribe to multiple symbols
sub_id = await client.subscribe(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    channels=['T'],
    callback=handle_trade
)

# Named subscription for easy management
sub_id = await client.subscribe(
    symbols=['SPY'],
    channels=['A'],
    callback=handle_aggregate,
    subscription_id='spy_aggregates'
)
Unsubscribe
python# Unsubscribe specific symbols
await client.unsubscribe(['AAPL'], channels=['T'])

# Unsubscribe all channels for symbols
await client.unsubscribe(['AAPL', 'GOOGL'])

# Unsubscribe specific subscription
await client.unsubscribe(['SPY'], subscription_id='spy_aggregates')
Data Callbacks
Trade Data
pythonasync def handle_trade(data):
    # data format:
    {
        'event_type': 'trade',
        'symbol': 'AAPL',
        'timestamp': 1701439845000,  # milliseconds
        'price': 175.50,
        'size': 100,
        'conditions': [12, 37],
        'exchange': 11,
        'trade_id': '1234567'
    }
Quote Data
pythonasync def handle_quote(data):
    # data format:
    {
        'event_type': 'quote',
        'symbol': 'AAPL',
        'timestamp': 1701439845000,
        'bid_price': 175.45,
        'bid_size': 200,
        'ask_price': 175.50,
        'ask_size': 300,
        'exchange': 11
    }
Aggregate Data
pythonasync def handle_aggregate(data):
    # data format:
    {
        'event_type': 'aggregate',
        'symbol': 'AAPL',
        'timestamp': 1701439800000,  # bar start time
        'open': 175.25,
        'high': 175.75,
        'low': 175.20,
        'close': 175.50,
        'volume': 50000,
        'vwap': 175.48,
        'transactions': 250
    }
Complete Example
pythonimport asyncio
from polygon.websocket import PolygonWebSocketClient

async def main():
    # Initialize client
    client = PolygonWebSocketClient()
    
    # Track prices
    latest_prices = {}
    
    # Define callbacks
    async def on_trade(data):
        symbol = data['symbol']
        price = data['price']
        latest_prices[symbol] = price
        print(f"{symbol}: ${price:.2f} (size: {data['size']})")
    
    async def on_quote(data):
        symbol = data['symbol']
        spread = data['ask_price'] - data['bid_price']
        print(f"{symbol} Quote: Bid ${data['bid_price']:.2f} Ask ${data['ask_price']:.2f} Spread ${spread:.2f}")
    
    try:
        # Connect
        await client.connect()
        print("Connected to Polygon WebSocket")
        
        # Subscribe to trades
        await client.subscribe(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            channels=['T'],
            callback=on_trade
        )
        
        # Subscribe to quotes
        await client.subscribe(
            symbols=['AAPL'],
            channels=['Q'],
            callback=on_quote
        )
        
        # Listen for data
        print("Listening for real-time data...")
        await client.listen()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await client.disconnect()
        print(f"Final prices: {latest_prices}")

# Run
asyncio.run(main())
Error Handling
The client handles several error scenarios automatically:

Connection Errors: Automatic reconnection with exponential backoff
Authentication Errors: Raised as PolygonAuthenticationError
Network Issues: Reconnects automatically up to 5 times
Invalid Subscriptions: Returns error in status message

Reconnection Behavior

Initial delay: 1 second
Max delay: 30 seconds
Max attempts: 5
Exponential backoff: delay * 2^(attempt-1)

Advanced Features
Custom Storage Integration
python# Initialize with storage manager
from polygon.storage import get_storage_manager
storage = get_storage_manager()

client = PolygonWebSocketClient(storage=storage)
# Real-time data automatically saved to cache
Multiple Callbacks per Symbol
python# Different callbacks for different data types
await client.subscribe(['AAPL'], ['T'], handle_trades, 'trades')
await client.subscribe(['AAPL'], ['Q'], handle_quotes, 'quotes')
await client.subscribe(['AAPL'], ['A'], handle_aggregates, 'aggregates')
Performance Monitoring
python# Check message rate
status = client.get_status()
messages_per_second = status['message_count'] / status['uptime_seconds']
print(f"Processing {messages_per_second:.2f} messages/second")
Server WebSocket Endpoint
The REST server also provides a WebSocket endpoint for clients:
Connect
javascriptconst ws = new WebSocket('ws://localhost:8200/ws/client123');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
Protocol
javascript// Subscribe
ws.send(JSON.stringify({
    action: 'subscribe',
    symbols: ['AAPL'],
    channels: ['T', 'Q']
}));

// Unsubscribe
ws.send(JSON.stringify({
    action: 'unsubscribe',
    symbols: ['AAPL']
}));

// Ping
ws.send(JSON.stringify({
    action: 'ping'
}));

## 5. INTEGRATION_GUIDE.md

```markdown
# Polygon Module - Quick Integration Guide

## Installation

```bash
# Clone repository
git clone <repository>

# Install dependencies
pip install -r requirements.txt

# Set API key
export POLYGON_API_KEY=your_polygon_api_key_here
Quick Start Examples
1. Simple Data Fetch
pythonfrom polygon import fetch_data

# Get last 30 days of daily data
df = fetch_data('AAPL', '1day')

# Get specific date range
df = fetch_data('AAPL', '5min', '2023-01-01', '2023-01-31')
2. Multiple Symbols
pythonfrom polygon import PolygonDataManager

manager = PolygonDataManager()

# Fetch multiple symbols
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
data = manager.fetch_multiple_symbols(
    symbols, 
    '1day', 
    '2023-01-01', 
    '2023-12-31'
)

# Access individual DataFrames
aapl_data = data['AAPL']
3. Real-time Streaming
pythonimport asyncio
from polygon import PolygonWebSocketClient

async def handle_price(data):
    print(f"{data['symbol']}: ${data['price']}")

async def stream_prices():
    async with PolygonWebSocketClient() as client:
        await client.subscribe(['AAPL', 'TSLA'], ['T'], handle_price)
        await client.listen()

asyncio.run(stream_prices())
4. Data Validation
pythonfrom polygon import PolygonDataManager

manager = PolygonDataManager()

# Fetch with validation
df = manager.fetch_data('AAPL', '5min', validate=True)

# Get validation report
from polygon.validators import generate_validation_summary
report = generate_validation_summary(df, 'AAPL', '5min')
print(f"Data quality: {report['overall_quality']}")
5. Using the REST API
pythonimport requests

# Start server first: python -m polygon.polygon_server.server

# Fetch bars
response = requests.post('http://localhost:8200/api/v1/bars', json={
    'symbol': 'AAPL',
    'timeframe': '1day',
    'start_date': '2023-01-01',
    'end_date': '2023-12-31'
})
data = response.json()
Common Integration Patterns
Pattern 1: Daily Data Update
pythonfrom polygon import PolygonDataManager
import schedule

manager = PolygonDataManager()

def update_daily_data():
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    for symbol in symbols:
        manager.update_cache(symbol, '1day')
        print(f"Updated {symbol}")

# Schedule daily updates
schedule.every().day.at("18:00").do(update_daily_data)
Pattern 2: Live Trading Integration
pythonfrom polygon import PolygonDataManager, PolygonWebSocketClient
import asyncio

class TradingSystem:
    def __init__(self):
        self.manager = PolygonDataManager()
        self.ws_client = PolygonWebSocketClient()
        self.positions = {}
    
    async def on_trade(self, data):
        symbol = data['symbol']
        price = data['price']
        
        # Get historical context
        historical = self.manager.fetch_latest(symbol, '5min', bars=20)
        
        # Your trading logic here
        signal = self.calculate_signal(historical, price)
        if signal:
            self.execute_trade(symbol, signal, price)
    
    async def start(self):
        await self.ws_client.connect()
        await self.ws_client.subscribe(
            ['AAPL', 'MSFT'], 
            ['T'], 
            self.on_trade
        )
        await self.ws_client.listen()
Pattern 3: Backtesting Data Preparation
pythonfrom polygon import PolygonDataManager

manager = PolygonDataManager()

# Create aligned dataset for backtesting
symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
universe = manager.batch_fetcher.fetch_universe(
    symbols,
    '5min',
    '2023-01-01',
    '2023-12-31',
    aligned=True  # Ensure all symbols have same timestamps
)

# Access data
spy_close = universe[('SPY', 'close')]
qqq_close = universe[('QQQ', 'close')]
Troubleshooting
Common Issues

Authentication Error

python# Check API key
import os
print(f"API Key set: {'POLYGON_API_KEY' in os.environ}")

# Test authentication
from polygon.core import PolygonClient
client = PolygonClient()
try:
    client.get_market_status()
    print("Authentication successful")
except Exception as e:
    print(f"Authentication failed: {e}")

Rate Limiting

pythonfrom polygon import get_rate_limit_status

# Check current usage
status = get_rate_limit_status()
print(f"Minute: {status['minute']['used']}/{status['minute']['limit']}")
print(f"Daily: {status['daily']['used']}/{status['daily']['limit']}")

Missing Data

python# Check data availability
from polygon import PolygonDataManager

manager = PolygonDataManager()
summary = manager.get_data_summary('AAPL', '5min', '2023-01-01', '2023-01-31')
print(f"Cache coverage: {summary['cache_coverage']['percentage']}%")
print(f"Missing ranges: {summary['cache_coverage']['missing_ranges']}")
Performance Tips

Use Caching: Always use use_cache=True (default)
Batch Requests: Use fetch_multiple_symbols() for multiple symbols
Appropriate Timeframes: Use larger timeframes for historical analysis
Update Incrementally: Use update_cache() instead of re-fetching

Getting Help

Check error messages - they include specific details
Enable debug logging:

pythonimport logging
logging.basicConfig(level=logging.DEBUG)

Validate your data:

pythonfrom polygon.validators import validate_symbol_detailed
result = validate_symbol_detailed('YOUR_SYMBOL')
print(result)