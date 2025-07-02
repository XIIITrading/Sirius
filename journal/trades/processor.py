# journal/trades/processor.py
"""Process executions into complete trades using CLOID."""

from collections import defaultdict
from decimal import Decimal
from typing import List, Dict, Tuple

# Handle both package and script imports
try:
    from .models import Execution, Trade
except ImportError:
    from models import Execution, Trade


class TradeProcessor:
    """Process executions into complete trades with P&L calculations."""
    
    def process_executions(self, executions: List[Execution]) -> List[Trade]:
        """Group executions into trades and calculate metrics."""
        # First, group by CLOID to get order-level aggregation
        orders = self._group_by_cloid(executions)
        
        # Then process into trades by symbol
        symbol_groups = self._group_orders_by_symbol(orders)
        
        trades = []
        for symbol, symbol_orders in symbol_groups.items():
            symbol_trades = self._process_symbol_trades(symbol, symbol_orders)
            trades.extend(symbol_trades)
            
        return sorted(trades, key=lambda t: t.entry_time)
    
    def _group_by_cloid(self, executions: List[Execution]) -> Dict[str, Dict]:
        """Group executions by CLOID to create aggregated orders."""
        cloid_groups = defaultdict(list)
        
        for execution in executions:
            if hasattr(execution, 'cloid'):
                cloid_groups[execution.cloid].append(execution)
            else:
                # Fallback if CLOID not available
                # Create a fake CLOID based on time and symbol
                fake_cloid = f"{execution.time.strftime('%H%M%S')}_{execution.symbol}_{execution.side}"
                cloid_groups[fake_cloid].append(execution)
        
        # Aggregate each CLOID group into a single order
        orders = {}
        for cloid, execs in cloid_groups.items():
            if execs:
                order = self._aggregate_executions(execs)
                orders[cloid] = order
                
        return orders
    
    def _aggregate_executions(self, executions: List[Execution]) -> Dict:
        """Aggregate multiple executions from the same order."""
        total_qty = sum(e.shares for e in executions)
        total_value = sum(e.shares * e.price for e in executions)
        avg_price = total_value / total_qty if total_qty > 0 else Decimal('0')
        
        return {
            'symbol': executions[0].symbol,
            'side': executions[0].side,
            'qty': total_qty,
            'price': avg_price,
            'time': executions[0].time,  # First fill time
            'executions': executions
        }
    
    def _group_orders_by_symbol(self, orders: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """Group orders by symbol."""
        symbol_groups = defaultdict(list)
        
        # Sort orders by time
        sorted_orders = sorted(orders.values(), key=lambda x: x['time'])
        
        for order in sorted_orders:
            symbol_groups[order['symbol']].append(order)
            
        return symbol_groups
    
    def _process_symbol_trades(self, symbol: str, orders: List[Dict]) -> List[Trade]:
        """Process all trades for a single symbol."""
        trades = []
        position = 0
        entry_orders = []
        
        for order in orders:
            if order['side'] == 'B':  # Buy
                if position <= 0:  # New long or closing short
                    if position < 0:  # Closing short
                        # Calculate how much closes the position
                        close_qty = min(order['qty'], abs(position))
                        
                        # Create trade for closed position
                        trade = self._create_trade_from_orders(
                            symbol, entry_orders, [order], close_qty
                        )
                        if trade:
                            trades.append(trade)
                        
                        # Update position
                        position += order['qty']
                        
                        # If we have leftover, it's a new entry
                        if position > 0:
                            entry_orders = [order]
                        else:
                            entry_orders = [] if position == 0 else entry_orders
                    else:  # New long position
                        entry_orders = [order]
                        position = order['qty']
                else:  # Adding to long
                    entry_orders.append(order)
                    position += order['qty']
                    
            else:  # Sell/Short
                if position >= 0:  # New short or closing long
                    if position > 0:  # Closing long
                        # Calculate how much closes the position
                        close_qty = min(order['qty'], position)
                        
                        # Create trade for closed position
                        trade = self._create_trade_from_orders(
                            symbol, entry_orders, [order], close_qty
                        )
                        if trade:
                            trades.append(trade)
                        
                        # Update position
                        position -= order['qty']
                        
                        # If we have leftover, it's a new entry
                        if position < 0:
                            entry_orders = [order]
                        else:
                            entry_orders = [] if position == 0 else entry_orders
                    else:  # New short position
                        entry_orders = [order]
                        position = -order['qty']
                else:  # Adding to short
                    entry_orders.append(order)
                    position -= order['qty']
        
        # Handle any remaining open position
        if entry_orders and position != 0:
            trade = self._create_open_trade(symbol, entry_orders, abs(position))
            if trade:
                trades.append(trade)
                
        return trades
    
    def _create_trade_from_orders(self, symbol: str, entry_orders: List[Dict], 
                                  exit_orders: List[Dict], qty: int) -> Trade:
        """Create a closed trade from entry and exit orders."""
        if not entry_orders or not exit_orders:
            return None
        
        # Calculate entry price (weighted average)
        entry_value = sum(o['price'] * o['qty'] for o in entry_orders)
        entry_qty = sum(o['qty'] for o in entry_orders)
        entry_price = entry_value / entry_qty if entry_qty > 0 else Decimal('0')
        
        # Calculate exit price
        exit_price = exit_orders[0]['price']  # Since it's usually one order
        
        # Determine if long or short trade
        is_long = entry_orders[0]['side'] == 'B'
        
        # Calculate P&L
        if is_long:
            net_pl = (exit_price - entry_price) * qty
        else:
            net_pl = (entry_price - exit_price) * qty
        
        # Collect all executions
        all_executions = []
        for order in entry_orders:
            all_executions.extend(order['executions'])
        for order in exit_orders:
            all_executions.extend(order['executions'])
        
        return Trade(
            symbol=symbol,
            entry_time=entry_orders[0]['time'],
            exit_time=exit_orders[0]['time'],
            entry_price=entry_price,
            exit_price=exit_price,
            qty=qty,
            net_pl=net_pl,
            executions=all_executions
        )
    
    def _create_open_trade(self, symbol: str, entry_orders: List[Dict], qty: int) -> Trade:
        """Create an open trade (no exit yet)."""
        if not entry_orders:
            return None
        
        # Calculate entry price (weighted average)
        entry_value = sum(o['price'] * o['qty'] for o in entry_orders)
        entry_qty = sum(o['qty'] for o in entry_orders)
        entry_price = entry_value / entry_qty if entry_qty > 0 else Decimal('0')
        
        # Collect all executions
        all_executions = []
        for order in entry_orders:
            all_executions.extend(order['executions'])
        
        return Trade(
            symbol=symbol,
            entry_time=entry_orders[0]['time'],
            exit_time=None,
            entry_price=entry_price,
            exit_price=None,
            qty=qty,
            net_pl=None,
            executions=all_executions
        )
    