# test_hybrid_chart_data.py - Clean Windows Version
"""
Hybrid Chart Data Flow Test - Windows Compatible
"""
import sys
import json
import logging
import requests
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, List, Optional

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

# Add project to path
sys.path.append('.')

from live_monitor.data import PolygonDataManager

# Configure logging with UTF-8 encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hybrid_chart_debug.log', encoding='utf-8')
    ]
)

# Silence noisy loggers
logging.getLogger('live_monitor.data.websocket_client').setLevel(logging.WARNING)
logging.getLogger('live_monitor.data.polygon_data_manager').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class Bar:
    """Simple bar class for testing"""
    def __init__(self, timestamp, open, high, low, close, volume, trades=0):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.trades = trades or 0  # Handle None/null
        
    def __repr__(self):
        return f"Bar({self.timestamp.strftime('%H:%M')}, O={self.open:.2f}, H={self.high:.2f}, L={self.low:.2f}, C={self.close:.2f}, V={self.volume}, T={self.trades})"


class TradeAggregator:
    """Builds minute bars from trades"""
    
    def __init__(self, on_bar_complete):
        self.on_bar_complete = on_bar_complete
        self.current_bars = {}
        self.last_trade_time = None
        
    def process_trade(self, trade_data):
        """Aggregate trade into minute bars"""
        symbol = trade_data.get('symbol', trade_data.get('sym'))
        price = float(trade_data.get('price', trade_data.get('p', 0)))
        size = int(trade_data.get('size', trade_data.get('s', 0)))
        timestamp = trade_data.get('timestamp', trade_data.get('t', 0))
        
        if not symbol or not price:
            return
            
        # Convert timestamp
        trade_time = datetime.fromtimestamp(timestamp / 1000.0)
        self.last_trade_time = trade_time
        
        # Get minute boundary
        minute_time = trade_time.replace(second=0, microsecond=0)
        bar_key = f"{symbol}_{minute_time.timestamp()}"
        
        if bar_key not in self.current_bars:
            # New bar
            self.current_bars[bar_key] = {
                'symbol': symbol,
                'timestamp': minute_time,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': size,
                'trades': 1,
                'first_trade_time': trade_time,
                'last_trade_time': trade_time
            }
            logger.info(f"[TRADE AGG] Started new minute bar for {symbol} at {minute_time.strftime('%H:%M:%S')}")
        else:
            # Update bar
            bar = self.current_bars[bar_key]
            bar['high'] = max(bar['high'], price)
            bar['low'] = min(bar['low'], price)
            bar['close'] = price
            bar['volume'] += size
            bar['trades'] += 1
            bar['last_trade_time'] = trade_time
            
    def check_complete_bars(self):
        """Check and emit completed bars"""
        if not self.last_trade_time:
            return
            
        current_minute = self.last_trade_time.replace(second=0, microsecond=0)
        completed = []
        
        for bar_key, bar_data in self.current_bars.items():
            # If we've moved to a new minute, previous bar is complete
            if bar_data['timestamp'] < current_minute:
                completed.append(bar_key)
                
                # Create bar object
                bar = Bar(
                    timestamp=bar_data['timestamp'],
                    open=bar_data['open'],
                    high=bar_data['high'],
                    low=bar_data['low'],
                    close=bar_data['close'],
                    volume=bar_data['volume'],
                    trades=bar_data['trades']
                )
                
                logger.warning(f"[COMPLETED] Minute bar from TRADES: {bar}")
                self.on_bar_complete(bar_data['symbol'], bar, source='TRADES')
                
        # Remove completed bars
        for key in completed:
            del self.current_bars[key]


class HybridChartDataManager(QObject):
    """Hybrid chart data manager"""
    
    bar_received = pyqtSignal(str, object, str)
    status_update = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        self.symbol = None
        self.bars = defaultdict(lambda: deque(maxlen=500))
        
        # Stats
        self.stats = {
            'trades': 0,
            'quotes': 0,
            'am_events': 0,
            'generated_bars': 0,
            'historical_bars': 0
        }
        
        # Trade aggregator
        self.trade_aggregator = TradeAggregator(self.on_bar_complete)
        
        # Data manager
        self.data_manager = PolygonDataManager()
        self.data_manager.ws_client.data_received.connect(self.on_websocket_data)
        self.data_manager.connection_status_changed.connect(self.on_connection_status)
        
        # Timer to check for completed bars
        self.check_timer = QTimer()
        self.check_timer.timeout.connect(self.trade_aggregator.check_complete_bars)
        self.check_timer.start(1000)
        
        # Debug timer
        self.debug_timer = QTimer()
        self.debug_timer.timeout.connect(self.print_debug_info)
        self.debug_timer.start(15000)
        
    def start(self, symbol: str):
        """Start data collection for symbol"""
        self.symbol = symbol.upper()
        print("\n" + "="*80)
        print(f"Starting Hybrid Data Manager for {self.symbol}")
        print("="*80 + "\n")
        
        # Load historical data
        self.load_historical_data()
        
        # Connect WebSocket
        QTimer.singleShot(1000, self.connect_websocket)
        
    def load_historical_data(self):
        """Load historical minute bars from REST API"""
        logger.info("[HISTORICAL] Loading data from REST API...")
        
        try:
            response = requests.post(
                "http://localhost:8200/api/v1/bars",
                json={
                    "symbol": self.symbol,
                    "timeframe": "1min",
                    "limit": 100
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                bars_data = data.get('data', [])
                
                logger.info(f"[HISTORICAL] Received {len(bars_data)} bars")
                
                # Process bars
                for bar_data in bars_data:
                    bar = Bar(
                        timestamp=datetime.fromisoformat(bar_data['timestamp'].replace('Z', '+00:00')),
                        open=bar_data['open'],
                        high=bar_data['high'],
                        low=bar_data['low'],
                        close=bar_data['close'],
                        volume=bar_data['volume'],
                        trades=bar_data.get('transactions', 0)
                    )
                    
                    self.bars[self.symbol].append(bar)
                    self.stats['historical_bars'] += 1
                    
                if bars_data:
                    latest = self.bars[self.symbol][-1]
                    logger.info(f"[HISTORICAL] Latest bar: {latest}")
                    
            else:
                logger.error(f"[HISTORICAL] API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"[HISTORICAL] Failed to load: {e}")
            
    def connect_websocket(self):
        """Connect to WebSocket for real-time data"""
        logger.info("[WEBSOCKET] Connecting...")
        self.data_manager.connect()
        
        # Set symbol after connection
        QTimer.singleShot(2000, lambda: self.data_manager.change_symbol(self.symbol))
        
    def on_websocket_data(self, data):
        """Process all WebSocket data"""
        event_type = data.get('event_type', data.get('ev'))
        symbol = data.get('symbol', data.get('sym'))
        
        if symbol != self.symbol:
            return
            
        # Count event types
        if event_type in ['trade', 'T']:
            self.stats['trades'] += 1
            self.trade_aggregator.process_trade(data)
            
            # Log milestones
            if self.stats['trades'] in [1, 100, 500, 1000]:
                price = data.get('price', data.get('p'))
                logger.info(f"[TRADE] #{self.stats['trades']}: {symbol} @ ${price}")
                
        elif event_type in ['quote', 'Q']:
            self.stats['quotes'] += 1
            
        elif event_type in ['aggregate', 'A', 'AM']:
            self.stats['am_events'] += 1
            print("\n" + "*"*60)
            print(f"[AM EVENT] Received AM Event #{self.stats['am_events']}")
            print(f"Data: {json.dumps(data, indent=2)}")
            print("*"*60 + "\n")
            
            # Process AM event
            self.process_am_event(data)
            
    def process_am_event(self, data):
        """Process AM aggregate event"""
        try:
            bar = Bar(
                timestamp=datetime.fromtimestamp(data.get('s', data.get('timestamp', 0)) / 1000.0),
                open=data.get('o', data.get('open')),
                high=data.get('h', data.get('high')),
                low=data.get('l', data.get('low')),
                close=data.get('c', data.get('close')),
                volume=data.get('v', data.get('volume')),
                trades=data.get('n', data.get('transactions')) or 0  # Handle null
            )
            
            self.on_bar_complete(self.symbol, bar, source='AM')
            
        except Exception as e:
            logger.error(f"[AM ERROR] Failed to process: {e}")
            
    def on_bar_complete(self, symbol, bar, source):
        """Handle completed bar from any source"""
        if source == 'TRADES':
            self.stats['generated_bars'] += 1
            
        self.bars[symbol].append(bar)
        
        print("\n" + "="*40)
        print(f"[BAR COMPLETE] Source: {source}")
        print(f"[BAR COMPLETE] {bar}")
        print(f"[BAR COMPLETE] Total bars: {len(self.bars[symbol])}")
        print("="*40 + "\n")
        
        self.bar_received.emit(symbol, bar, source)
        
    def on_connection_status(self, connected):
        """Handle connection status changes"""
        status = "Connected" if connected else "Disconnected"
        logger.info(f"[WEBSOCKET] {status}")
        
    def print_debug_info(self):
        """Print debug statistics"""
        print("\n" + "-"*60)
        print("[STATS] Current Statistics:")
        print(f"  Symbol: {self.symbol}")
        print(f"  Historical bars: {self.stats['historical_bars']}")
        print(f"  Trades received: {self.stats['trades']}")
        print(f"  AM events: {self.stats['am_events']}")
        print(f"  Bars from trades: {self.stats['generated_bars']}")
        print(f"  Total bars: {len(self.bars[self.symbol])}")
        
        if self.stats['am_events'] > 0:
            print("  STATUS: Receiving AM events!")
        elif self.stats['trades'] > 0:
            print("  STATUS: Using trade aggregation (no AM events)")
            
        print("-"*60 + "\n")


def main():
    """Run the test"""
    app = QApplication(sys.argv)
    
    # Create manager
    manager = HybridChartDataManager()
    
    # Start with TSLA
    manager.start("TSLA")
    
    # Run for 3 minutes
    QTimer.singleShot(180000, lambda: (
        print("\nTest complete - exiting..."),
        app.quit()
    ))
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()