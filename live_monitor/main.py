# live_monitor/main.py
"""
Live Trading Monitor - Multi-Signal Dashboard
Monitors 4 tickers with unified Bull/Bear signals from multiple calculations
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

# Import all calculation modules
from live_monitor.calculations.market_structure.m1_market_structure import MarketStructureAnalyzer as MS1
from live_monitor.calculations.market_structure.m5_market_structure import M5MarketStructureAnalyzer as MS5
from live_monitor.calculations.market_structure.m15_market_structure import M15MarketStructureAnalyzer as MS15

from live_monitor.calculations.indicators.m1_ema import M1EMACalculator
from live_monitor.calculations.indicators.m5_ema import M5EMACalculator
from live_monitor.calculations.indicators.m15_ema import M15EMACalculator

from live_monitor.calculations.volume.tick_flow import TickFlowAnalyzer
from live_monitor.calculations.volume.volume_analysis_1min import VolumeAnalysis1Min
from live_monitor.calculations.volume.market_context import MarketContext
from live_monitor.calculations.volume.m1_bid_ask_analysis import M1VolumeAnalyzer

from live_monitor.calculations.order_flow.bid_ask_imbal import BidAskImbalance
from live_monitor.calculations.order_flow.buy_sell_ratio import BuySellRatioCalculator
from live_monitor.calculations.order_flow.large_orders import LargeOrderDetector
from live_monitor.calculations.order_flow.micro_momentum import MicrostructureMomentum
from live_monitor.calculations.order_flow.trade_size_distro import TradeSizeDistribution

from live_monitor.calculations.trend.statistical_trend_1min import StatisticalTrend1MinSimplified
from live_monitor.calculations.trend.statistical_trend_5min import StatisticalTrend5Min
from live_monitor.calculations.trend.statistical_trend_15min import StatisticalTrend15MinSimplified

# Import Polygon infrastructure
from polygon import PolygonWebSocketClient, DataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UnifiedSignal:
    """Standardized signal format across all modules"""
    timestamp: datetime
    symbol: str
    module: str
    timeframe: str
    signal: str  # 'BULL', 'BEAR', 'NEUTRAL'
    strength: float  # 0-100
    confidence: float  # 0-100
    reason: str
    raw_data: Dict = field(default_factory=dict)


class SignalAggregator:
    """Aggregates signals from multiple modules into consensus"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'market_structure': 1.5,
            'ema': 1.0,
            'volume': 1.2,
            'order_flow': 1.3,
            'trend': 1.0
        }
        
    def aggregate_signals(self, signals: List[UnifiedSignal]) -> Dict:
        """Create consensus signal from multiple sources"""
        if not signals:
            return {
                'consensus': 'NEUTRAL',
                'bull_score': 0,
                'bear_score': 0,
                'confidence': 0,
                'signal_count': 0
            }
        
        bull_weighted = 0
        bear_weighted = 0
        total_weight = 0
        
        for signal in signals:
            # Get module category
            category = self._get_category(signal.module)
            weight = self.weights.get(category, 1.0)
            
            # Apply signal
            if signal.signal == 'BULL':
                bull_weighted += signal.strength * weight
                total_weight += weight
            elif signal.signal == 'BEAR':
                bear_weighted += signal.strength * weight
                total_weight += weight
            # NEUTRAL doesn't contribute to scores
        
        # Calculate consensus
        if total_weight > 0:
            bull_score = bull_weighted / total_weight
            bear_score = bear_weighted / total_weight
            
            if bull_score > bear_score and bull_score > 50:
                consensus = 'BULL'
            elif bear_score > bull_score and bear_score > 50:
                consensus = 'BEAR'
            else:
                consensus = 'NEUTRAL'
                
            confidence = max(bull_score, bear_score)
        else:
            bull_score = bear_score = confidence = 0
            consensus = 'NEUTRAL'
        
        return {
            'consensus': consensus,
            'bull_score': bull_score,
            'bear_score': bear_score,
            'confidence': confidence,
            'signal_count': len(signals),
            'signals': signals
        }
    
    def _get_category(self, module_name: str) -> str:
        """Map module to category"""
        if 'market_structure' in module_name:
            return 'market_structure'
        elif 'ema' in module_name:
            return 'ema'
        elif 'volume' in module_name or 'tick_flow' in module_name:
            return 'volume'
        elif 'order_flow' in module_name or 'bid_ask' in module_name:
            return 'order_flow'
        elif 'trend' in module_name:
            return 'trend'
        return 'other'


class CalculationManager:
    """Manages all calculation modules for a symbol"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.data_fetcher = DataFetcher()
        
        # Initialize all modules
        self.modules = {
            # Market Structure
            'ms_1min': MS1(bars_needed=200),
            'ms_5min': MS5(bars_needed=100),
            'ms_15min': MS15(bars_needed=60),
            
            # EMA
            'ema_1min': M1EMACalculator(),
            'ema_5min': M5EMACalculator(),
            'ema_15min': M15EMACalculator(),
            
            # Volume
            'tick_flow': TickFlowAnalyzer(buffer_size=200),
            'volume_1min': VolumeAnalysis1Min(lookback_bars=14),
            'market_context': MarketContext(lookback_bars=10),
            'bid_ask_1min': M1VolumeAnalyzer(lookback_bars=14),
            
            # Order Flow
            'bid_ask_imbalance': BidAskImbalance(imbalance_lookback=1000),
            'buy_sell_ratio': BuySellRatioCalculator(window_minutes=30),
            'large_orders': LargeOrderDetector(stats_window_minutes=15),
            'micro_momentum': MicrostructureMomentum(),
            'trade_size': TradeSizeDistribution(buffer_size=500),
            
            # Trend
            'trend_1min': StatisticalTrend1MinSimplified(lookback_periods=10),
            'trend_5min': StatisticalTrend5Min(lookback_periods=10),
            'trend_15min': StatisticalTrend15MinSimplified(lookback_periods=10)
        }
        
        # Track data readiness
        self.data_ready = {module: False for module in self.modules}
        self.warmup_complete = False
        
        # Store latest signals
        self.latest_signals: Dict[str, UnifiedSignal] = {}
        
        # Data buffers
        self.bars_1min = pd.DataFrame()
        self.bars_5min = pd.DataFrame()
        self.bars_15min = pd.DataFrame()
        self.trades_buffer = deque(maxlen=1000)
        
    async def initialize(self):
        """Load historical data for warmup"""
        logger.info(f"Initializing {self.symbol} with historical data...")
        
        # Fetch historical data
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=2)  # 2 days for safety
        
        # Fetch different timeframes
        self.bars_1min = self.data_fetcher.fetch_data(
            self.symbol, '1min', start_date, end_date
        )
        self.bars_5min = self.data_fetcher.fetch_data(
            self.symbol, '5min', start_date, end_date
        )
        self.bars_15min = self.data_fetcher.fetch_data(
            self.symbol, '15min', start_date, end_date
        )
        
        # Warm up modules that need bar data
        await self._warmup_bar_modules()
        
        logger.info(f"{self.symbol} initialization complete")
        self.warmup_complete = True
        
    async def _warmup_bar_modules(self):
        """Warm up modules that process bar data"""
        # Market Structure modules
        if len(self.bars_1min) >= 200:
            self.modules['ms_1min'].process_bars_dataframe(self.symbol, self.bars_1min.tail(200))
            self.data_ready['ms_1min'] = True
            
        if len(self.bars_5min) >= 100:
            self.modules['ms_5min'].process_bars_dataframe(self.symbol, self.bars_5min.tail(100))
            self.data_ready['ms_5min'] = True
            
        if len(self.bars_15min) >= 60:
            self.modules['ms_15min'].process_bars_dataframe(self.symbol, self.bars_15min.tail(60))
            self.data_ready['ms_15min'] = True
        
        # EMA modules (convert bars to format they expect)
        if len(self.bars_1min) >= 26:
            bars_dict = self.bars_1min.tail(26).to_dict('records')
            result = self.modules['ema_1min'].calculate(bars_dict)
            if result:
                self.data_ready['ema_1min'] = True
                
        # Similar for 5min and 15min EMAs...
        
    def process_trade(self, trade_data: Dict) -> List[UnifiedSignal]:
        """Process incoming trade and return signals"""
        signals = []
        
        # Update trade-based modules
        if 'tick_flow' in self.modules:
            result = self.modules['tick_flow'].process_trade(self.symbol, trade_data)
            if result:
                signal = self._convert_to_unified(result, 'tick_flow', 'tick')
                signals.append(signal)
                self.latest_signals['tick_flow'] = signal
        
        # Update other trade-based modules...
        
        return signals
    
    def process_quote(self, quote_data: Dict) -> List[UnifiedSignal]:
        """Process incoming quote and return signals"""
        signals = []
        
        # Update quote-based modules
        if 'micro_momentum' in self.modules:
            result = self.modules['micro_momentum'].process_quote(quote_data)
            if result:
                signal = self._convert_to_unified(result, 'micro_momentum', 'tick')
                signals.append(signal)
                self.latest_signals['micro_momentum'] = signal
        
        return signals
    
    def _convert_to_unified(self, module_signal: Any, module_name: str, 
                           timeframe: str) -> UnifiedSignal:
        """Convert module-specific signal to unified format"""
        # Map different signal formats to unified BULL/BEAR/NEUTRAL
        signal_map = {
            'BULLISH': 'BULL',
            'BEARISH': 'BEAR',
            'STRONG BUY': 'BULL',
            'STRONG SELL': 'BEAR',
            'BUY': 'BULL',
            'SELL': 'BEAR'
        }
        
        # Extract common fields
        if hasattr(module_signal, 'signal'):
            raw_signal = module_signal.signal
        elif hasattr(module_signal, 'signal_type'):
            raw_signal = module_signal.signal_type
        else:
            raw_signal = 'NEUTRAL'
            
        mapped_signal = signal_map.get(raw_signal, raw_signal)
        if mapped_signal not in ['BULL', 'BEAR', 'NEUTRAL']:
            mapped_signal = 'NEUTRAL'
        
        # Extract strength/confidence
        strength = getattr(module_signal, 'strength', 50)
        confidence = getattr(module_signal, 'confidence', 
                           getattr(module_signal, 'signal_confidence', 50))
        
        return UnifiedSignal(
            timestamp=datetime.now(timezone.utc),
            symbol=self.symbol,
            module=module_name,
            timeframe=timeframe,
            signal=mapped_signal,
            strength=float(strength),
            confidence=float(confidence),
            reason=getattr(module_signal, 'reason', ''),
            raw_data=module_signal.__dict__ if hasattr(module_signal, '__dict__') else {}
        )


class LiveTradingMonitor:
    """Main monitoring system for multiple symbols"""
    
    def __init__(self, symbols: List[str], display_interval: int = 1):
        self.symbols = symbols
        self.display_interval = display_interval
        
        # Always use server at fixed address
        self.server_url = "ws://localhost:8200"
        
        # Initialize managers
        self.managers = {symbol: CalculationManager(symbol) for symbol in symbols}
        self.aggregator = SignalAggregator()
        
        # WebSocket client will be set when connecting to server
        self.ws_client = None
        self.session = None  # aiohttp session
        
        # Display state
        self.consensus_signals = {symbol: {} for symbol in symbols}
        self.last_display_time = datetime.now(timezone.utc)
        
    async def start(self):
        """Start the monitoring system"""
        logger.info("Starting Live Trading Monitor...")
        
        # Initialize all symbols
        await asyncio.gather(*[
            manager.initialize() for manager in self.managers.values()
        ])
        
        # Connect to local server WebSocket
        import aiohttp
        import uuid
        
        self.session = aiohttp.ClientSession()
        client_id = str(uuid.uuid4())
        
        try:
            # Connect to server
            ws_url = f"{self.server_url}/ws/{client_id}"
            async with self.session.ws_connect(ws_url) as ws:
                self.ws_client = ws
                logger.info(f"Connected to server at {ws_url}")
                
                # Subscribe to symbols
                await ws.send_json({
                    "action": "subscribe",
                    "symbols": self.symbols,
                    "channels": ["T", "Q"]  # Trades and Quotes
                })
                
                # Start display loop
                display_task = asyncio.create_task(self._display_loop())
                
                # Listen for messages
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = msg.json()
                        if data.get("type") == "market_data":
                            await self._handle_market_data(data.get("data", {}))
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {ws.exception()}")
                        break
                        
        finally:
            if self.session:
                await self.session.close()
        
    async def _handle_market_data(self, data: Dict):
        """Handle incoming market data"""
        event_type = data.get('ev', data.get('event_type'))
        symbol = data.get('sym', data.get('symbol'))
        
        if symbol not in self.managers:
            return
            
        manager = self.managers[symbol]
        
        if event_type == 'T':  # Trade
            signals = manager.process_trade(data)
            if signals:
                self._update_consensus(symbol, signals)
                
        elif event_type == 'Q':  # Quote
            signals = manager.process_quote(data)
            if signals:
                self._update_consensus(symbol, signals)
    
    def _update_consensus(self, symbol: str, new_signals: List[UnifiedSignal]):
        """Update consensus for symbol"""
        # Get all latest signals for symbol
        all_signals = list(self.managers[symbol].latest_signals.values())
        
        # Calculate consensus
        self.consensus_signals[symbol] = self.aggregator.aggregate_signals(all_signals)
        
    async def _display_loop(self):
        """Periodic display update"""
        while True:
            await asyncio.sleep(self.display_interval)
            self._display_dashboard()
            
    def _display_dashboard(self):
        """Display current state"""
        # Clear screen (platform dependent)
        print("\033[2J\033[H")  # ANSI clear screen
        
        print("=" * 120)
        print(f"LIVE TRADING MONITOR - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 120)
        
        # Display each symbol
        for symbol in self.symbols:
            consensus = self.consensus_signals.get(symbol, {})
            manager = self.managers[symbol]
            
            # Symbol header
            signal = consensus.get('consensus', 'NEUTRAL')
            confidence = consensus.get('confidence', 0)
            
            # Color coding
            if signal == 'BULL':
                color = '\033[92m'  # Green
                arrow = '↑'
            elif signal == 'BEAR':
                color = '\033[91m'  # Red
                arrow = '↓'
            else:
                color = '\033[93m'  # Yellow
                arrow = '→'
                
            print(f"\n{color}{'─' * 40} {symbol} {arrow} {'─' * 40}\033[0m")
            print(f"Consensus: {signal} ({confidence:.0f}%) | "
                  f"Bull: {consensus.get('bull_score', 0):.0f} | "
                  f"Bear: {consensus.get('bear_score', 0):.0f} | "
                  f"Signals: {consensus.get('signal_count', 0)}")
            
            # Show individual signals
            if manager.latest_signals:
                print("\nActive Signals:")
                for module, signal in list(manager.latest_signals.items())[-5:]:  # Last 5
                    sig_char = '●' if signal.signal == 'BULL' else '○' if signal.signal == 'BEAR' else '◐'
                    print(f"  {sig_char} {module:<20} {signal.signal:<7} "
                          f"S:{signal.strength:>3.0f} C:{signal.confidence:>3.0f} "
                          f"{signal.reason[:50]}")
            
            # Warmup status
            if not manager.warmup_complete:
                ready_count = sum(manager.data_ready.values())
                total_count = len(manager.modules)
                print(f"\n⏳ Warming up... ({ready_count}/{total_count} modules ready)")
        
        print("\n" + "=" * 120)


async def main():
    """Main entry point"""
    # This is only used when running main.py directly
    # Normally you should use run_live_monitor.py
    SYMBOLS = ['AAPL', 'TSLA', 'SPY', 'QQQ']
    
    monitor = LiveTradingMonitor(symbols=SYMBOLS, display_interval=1)
    
    try:
        await monitor.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        

if __name__ == "__main__":
    asyncio.run(main())