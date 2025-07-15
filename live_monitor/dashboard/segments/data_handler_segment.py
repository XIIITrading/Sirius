# live_monitor/dashboard/segments/data_handler_segment.py
"""
Data Handler Segment - Handles all data update callbacks
"""

import logging
from PyQt6.QtCore import QTimer

logger = logging.getLogger(__name__)


class DataHandlerSegment:
    """Dashboard segment for handling data updates from the data manager"""
    
    def _on_ticker_changed(self, ticker):
        """Handle ticker symbol changes"""
        if ticker:
            logger.info(f"Ticker changed to: {ticker}")
            self.current_symbol = ticker
            self.symbol_label.setText(f"Symbol: {ticker}")
            self.status_bar.showMessage(f"Subscribing to {ticker}...", 2000)
            
            # Clear existing data
            self.ticker_calculations.clear_calculations()
            self.point_call_entry.clear_signals()
            self.point_call_exit.clear_signals()
            self.accumulated_data.clear()
            
            # Clear tables
            self.hvn_table.clear_zones()
            self.supply_demand_table.clear_zones()
            self.order_blocks_table.clear_blocks()
            
            # Reset all signal labels
            self.m1_signal_label.setText("M1: --")
            self.m1_signal_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
            self.m5_signal_label.setText("M5: --")
            self.m5_signal_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
            self.m15_signal_label.setText("M15: --")
            self.m15_signal_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
            self.stat_signal_label.setText("STAT: --")
            self.stat_signal_label.setStyleSheet("QLabel { font-weight: bold; margin-left: 10px; }")
            
            # Change symbol in data manager
            self.data_manager.change_symbol(ticker)
            
            # Run calculations after a delay to allow data to load
            QTimer.singleShot(5000, self.run_calculations)
    
    def _on_market_data_updated(self, data):
        """Handle market data updates"""
        # Update ticker calculations
        self.ticker_calculations.update_calculations(data)
        
        # Update last update time
        if 'last_update' in data:
            self.update_time_label.setText(f"Last Update: {data['last_update']}")
        
        # Show in status bar
        if 'last_price' in data:
            self.status_bar.showMessage(
                f"Last: ${data['last_price']:.2f} | "
                f"Bid: ${data.get('bid', 0):.2f} | "
                f"Ask: ${data.get('ask', 0):.2f}", 
                2000
            )
    
    def _on_chart_data_updated(self, data: dict):
        """Handle chart data updates - accumulate for calculations"""
        if data.get('bars'):
            # Add bars to accumulated data
            self.accumulated_data.extend(data['bars'])
            
            # Keep only recent data (e.g., last 2000 bars)
            if len(self.accumulated_data) > 2000:
                self.accumulated_data = self.accumulated_data[-2000:]
            
            logger.info(f"Accumulated {len(self.accumulated_data)} bars for {data['symbol']}")
    
    def _on_entry_signal(self, signal_data):
        """Handle new entry signal"""
        logger.info(f"Entry signal received: {signal_data}")
        
        # Use the new add_or_update_signal method that handles duplicates
        self.point_call_entry.add_or_update_signal(signal_data)
    
    def _on_exit_signal(self, signal_data):
        """Handle new exit signal"""
        logger.info(f"Exit signal received: {signal_data}")
        self.point_call_exit.add_exit_signal(
            time=signal_data.get('time', ''),
            exit_type=signal_data.get('exit_type', 'TARGET'),
            price=signal_data.get('price', ''),
            pnl=signal_data.get('pnl', ''),
            signal=signal_data.get('signal', ''),
            urgency=signal_data.get('urgency', 'Normal')
        )
    
    def _on_connection_status_changed(self, is_connected):
        """Handle connection status changes"""
        # Update top-right indicator
        self.server_status_widget.set_connected(is_connected)
        
        # Update status bar indicator
        if is_connected:
            self.connection_label.setText("● Connected")
            self.connection_label.setStyleSheet("QLabel { color: #44ff44; font-weight: bold; }")
            self.setWindowTitle("Live Monitor - Trading Dashboard [Connected]")
            self.status_bar.showMessage("Connected to Polygon server", 2000)
        else:
            self.connection_label.setText("● Disconnected")
            self.connection_label.setStyleSheet("QLabel { color: #ff4444; font-weight: bold; }")
            self.setWindowTitle("Live Monitor - Trading Dashboard [Disconnected]")
            self.status_bar.showMessage("Disconnected from server", 2000)
    
    def _on_data_error(self, error_msg):
        """Handle data errors"""
        logger.error(f"Data error: {error_msg}")
        self.status_bar.showMessage(f"Error: {error_msg}", 5000)
    
    def _on_calculation_complete(self, results):
        """Handle calculation completion"""
        logger.info(f"Calculation complete: {results}")
    
    def _on_entry_selected(self, entry_data):
        """Handle entry signal selection"""
        logger.info(f"Entry selected: {entry_data}")
        if 'price' in entry_data:
            try:
                price_str = entry_data['price'].replace('$', '').replace(',', '')
                price = float(price_str)
                self.entry_calculations.update_entry_price(price)
            except ValueError:
                logger.error(f"Could not parse price: {entry_data['price']}")
    
    def _on_exit_selected(self, exit_data):
        """Handle exit signal selection"""
        logger.info(f"Exit selected: {exit_data}")
    
    def _on_add_zone_requested(self):
        """Handle request to add new supply/demand zone"""
        logger.info("Add zone requested - would open Supabase dialog")
        self.status_bar.showMessage("Add zone feature coming soon...", 2000)
    
    def _on_refresh_zones_requested(self):
        """Handle request to refresh supply/demand zones"""
        logger.info("Refresh zones requested - would query Supabase")
        self.status_bar.showMessage("Refreshing zones from database...", 2000)
        self.run_calculations()