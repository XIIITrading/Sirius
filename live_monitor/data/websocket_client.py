# live_monitor/data/websocket_client.py
"""
Qt-compatible WebSocket client for Polygon server connection
"""
import asyncio
import json
import logging
from typing import Dict, Optional, Callable
from datetime import datetime
import websockets

from PyQt6.QtCore import QObject, QThread, pyqtSignal

logger = logging.getLogger(__name__)


class WebSocketThread(QThread):
    """Thread to run asyncio event loop for WebSocket"""
    
    def __init__(self):
        super().__init__()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
    def run(self):
        """Run event loop in thread"""
        logger.info("Starting WebSocket thread event loop")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        logger.info("WebSocket thread event loop stopped")
        
    def stop(self):
        """Stop the event loop"""
        if self.loop:
            logger.info("Stopping WebSocket thread")
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.wait()


class PolygonWebSocketClient(QObject):
    """
    Qt-compatible WebSocket client for Polygon server
    
    Manages connection to the Polygon WebSocket server and emits
    Qt signals for data updates.
    """
    
    # Signals
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    data_received = pyqtSignal(dict)  # Raw data from WebSocket
    error_occurred = pyqtSignal(str)
    connection_status = pyqtSignal(bool)  # True=connected, False=disconnected
    
    def __init__(self, server_url: str = "ws://localhost:8200/ws/{client_id}", 
             client_id: str = "live_monitor"):
        super().__init__()
        
        self.server_url = server_url.format(client_id=client_id)
        self.client_id = client_id
        self.ws_connection = None
        self.is_running = False
        
        logger.info(f"WebSocket client initialized with URL: {self.server_url}")
        
        # Async thread setup
        self.ws_thread = WebSocketThread()
        self.ws_thread.start()
        
        # Wait for thread to start
        import time
        time.sleep(0.1)
        
        # Current subscriptions
        self.current_symbols: set = set()
        self.current_channels = ["T", "Q", "AM"]  # Trades, Quotes, Minute Aggregates
        
        # Track current symbol for easy access
        self.current_symbol: Optional[str] = None
        
        # Reconnection settings
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 30.0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
    def connect_to_server(self):
        """Initiate connection to WebSocket server"""
        logger.info("connect_to_server() called")
        if self.ws_thread.loop:
            logger.info("Scheduling connection coroutine")
            future = asyncio.run_coroutine_threadsafe(
                self._connect(), 
                self.ws_thread.loop
            )
            # Add a callback to log any exceptions
            future.add_done_callback(self._connection_done_callback)
        else:
            logger.error("WebSocket thread loop not available!")
            self.error_occurred.emit("WebSocket thread not initialized")
    
    def _connection_done_callback(self, future):
        """Callback to handle connection completion"""
        try:
            future.result()
        except Exception as e:
            logger.error(f"Connection future failed: {e}")
            self.error_occurred.emit(str(e))
    
    async def _connect(self):
        """Async connection handler"""
        try:
            logger.info(f"Attempting connection to {self.server_url}")
            self.ws_connection = await websockets.connect(self.server_url)
            logger.info("WebSocket connection established")
            
            self.is_running = True
            self.reconnect_attempts = 0
            
            # Emit connection signal
            self.connected.emit()
            self.connection_status.emit(True)
            
            logger.info("Starting listen loop")
            # Start listening
            await self._listen()
            
        except Exception as e:
            logger.error(f"Connection failed: {type(e).__name__}: {e}")
            self.error_occurred.emit(f"Connection failed: {str(e)}")
            self.connection_status.emit(False)
            
            # Schedule reconnection
            await self._schedule_reconnect()
    
    async def _listen(self):
        """Main listening loop"""
        logger.info("Listen loop started")
        try:
            while self.is_running and self.ws_connection:
                try:
                    message = await self.ws_connection.recv()
                    logger.debug(f"Received message: {message[:100]}...")
                    data = json.loads(message)
                    
                    # Handle different message types
                    msg_type = data.get('type')
                    logger.info(f"Message type: {msg_type}")
                    
                    if msg_type == 'connected':
                        logger.info("Server confirmed connection")
                        # Resubscribe if we have symbols
                        if self.current_symbols:
                            await self._subscribe(list(self.current_symbols))
                            
                    elif msg_type == 'market_data':
                        # Emit raw market data
                        self.data_received.emit(data.get('data', {}))
                        
                    elif msg_type == 'error':
                        error_msg = data.get('message', 'Unknown error')
                        logger.error(f"Server error: {error_msg}")
                        self.error_occurred.emit(error_msg)
                        
                    elif msg_type in ['subscribed', 'unsubscribed']:
                        logger.info(f"{msg_type}: {data.get('symbols', [])}")
                        
                except asyncio.TimeoutError:
                    logger.debug("Receive timeout, continuing...")
                    continue
                        
        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.disconnected.emit()
            self.connection_status.emit(False)
            await self._schedule_reconnect()
            
        except Exception as e:
            logger.error(f"Listen error: {type(e).__name__}: {e}")
            self.error_occurred.emit(str(e))
            self.is_running = False
        
        logger.info("Listen loop ended")
    
    async def _subscribe(self, symbols: list):
        """Send subscription message"""
        if self.ws_connection:
            sub_msg = {
                "action": "subscribe",
                "symbols": symbols,
                "channels": self.current_channels
            }
            logger.info(f"Sending subscription: {sub_msg}")
            await self.ws_connection.send(json.dumps(sub_msg))
            logger.info(f"Subscription sent for {symbols}")
        else:
            logger.warning("Cannot subscribe - no WebSocket connection")
    
    async def _unsubscribe(self, symbols: list):
        """Send unsubscribe message"""
        if self.ws_connection:
            unsub_msg = {
                "action": "unsubscribe",
                "symbols": symbols
            }
            logger.info(f"Sending unsubscribe message: {unsub_msg}")
            await self.ws_connection.send(json.dumps(unsub_msg))
            logger.info(f"Unsubscribed from {symbols}")
    
    async def _schedule_reconnect(self):
        """Schedule reconnection with exponential backoff"""
        if not self.is_running:
            return
            
        self.reconnect_attempts += 1
        delay = min(
            self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)),
            self.max_reconnect_delay
        )
        
        logger.info(f"Reconnecting in {delay} seconds (attempt {self.reconnect_attempts})")
        await asyncio.sleep(delay)
        
        if self.is_running:
            await self._connect()
    
    def subscribe(self, symbols: list):
        """Subscribe to symbols (thread-safe)"""
        logger.info(f"subscribe() called with symbols: {symbols}")
        logger.info(f"Current symbols before change: {self.current_symbols}")
        
        # Update tracking
        old_symbols = self.current_symbols.copy()
        self.current_symbols = set(symbols)
        
        # Determine changes
        to_unsub = list(old_symbols - self.current_symbols)
        to_sub = list(self.current_symbols - old_symbols)
        
        logger.info(f"Subscription changes - Unsubscribe: {to_unsub}, Subscribe: {to_sub}")
        
        if self.ws_thread.loop and self.ws_connection:
            # Unsubscribe old
            if to_unsub:
                logger.info(f"Unsubscribing from: {to_unsub}")
                asyncio.run_coroutine_threadsafe(
                    self._unsubscribe(to_unsub),
                    self.ws_thread.loop
                )
            
            # Subscribe new
            if to_sub:
                logger.info(f"Subscribing to: {to_sub}")
                asyncio.run_coroutine_threadsafe(
                    self._subscribe(to_sub),
                    self.ws_thread.loop
                )
        else:
            logger.warning(f"Cannot subscribe - WebSocket not connected")
            
        logger.info(f"Current symbols after change: {self.current_symbols}")
    
    def change_symbol(self, symbol: str):
        """Convenience method to change to single symbol"""
        logger.info(f"change_symbol() called with: {symbol}")
        if symbol:
            self.current_symbol = symbol  # Track current symbol
            self.subscribe([symbol.upper()])
    
    def unsubscribe_all(self):
        """Unsubscribe from all symbols"""
        logger.info(f"Unsubscribing from all symbols: {self.current_symbols}")
        if self.current_symbols:
            symbols_to_unsub = list(self.current_symbols)
            self.current_symbols.clear()
            self.current_symbol = None  # Clear current symbol
            
            if self.ws_thread.loop and self.ws_connection:
                asyncio.run_coroutine_threadsafe(
                    self._unsubscribe(symbols_to_unsub),
                    self.ws_thread.loop
                )
            else:
                logger.warning("Cannot unsubscribe - WebSocket not connected")
    
    def disconnect(self):
        """Disconnect and cleanup"""
        logger.info("disconnect() called")
        self.is_running = False
        
        if self.ws_thread.loop and self.ws_connection:
            asyncio.run_coroutine_threadsafe(
                self._disconnect(),
                self.ws_thread.loop
            )
        
        # Stop thread
        self.ws_thread.stop()
    
    async def _disconnect(self):
        """Async disconnect"""
        logger.info("Disconnecting WebSocket")
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
        self.disconnected.emit()
        self.connection_status.emit(False)