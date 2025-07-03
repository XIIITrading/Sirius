"""
Test module for Net Large Volume plugin
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest

from ..plugin import NetLargeVolumePlugin


def generate_test_data(symbol: str = 'TEST', 
                      hours: int = 2,
                      avg_size: int = 100,
                      large_order_freq: float = 0.05) -> tuple:
    """Generate test trade and quote data with large orders"""
    
    np.random.seed(42)
    
    # Time range
    start_time = datetime.now() - timedelta(hours=hours)
    
    # Generate trades
    num_trades = hours * 3600  # ~1 trade per second
    timestamps = pd.date_range(start=start_time, periods=num_trades, freq='1s')
    
    trades = []
    quotes = []
    
    base_price = 100.0
    spread = 0.02
    
    for i, ts in enumerate(timestamps):
        # Random walk for price
        base_price += np.random.normal(0, 0.01)
        
        # Quote
        bid = base_price - spread/2
        ask = base_price + spread/2
        
        quotes.append({
            'symbol': symbol,
            'timestamp': ts,
            'bid': bid,
            'ask': ask
        })
        
        # Trade
        is_large = np.random.random() < large_order_freq
        
        if is_large:
            # Large order
            size = int(avg_size * np.random.uniform(1.5, 3.0))
            # Tend to trade at bid/ask
            if np.random.random() > 0.5:
                price = ask  # Buy
            else:
                price = bid  # Sell
        else:
            # Normal order
            size = int(np.random.lognormal(np.log(avg_size), 0.5))
            price = np.random.uniform(bid, ask)
        
        trades.append({
            'symbol': symbol,
            'timestamp': ts,
            'price': price,
            'size': size,
            'bid': bid,
            'ask': ask
        })
    
    trades_df = pd.DataFrame(trades)
    quotes_df = pd.DataFrame(quotes)
    
    return trades_df, quotes_df


def test_plugin_initialization():
    """Test plugin initialization"""
    config = {
        'large_order_config': {
            'stats_window_minutes': 15,
            'min_trades_for_std': 50
        },
        'volume_tracker_config': {
            'history_points': 100,
            'session_reset_hour': 9
        }
    }
    
    plugin = NetLargeVolumePlugin(config)
    assert plugin.large_order_detector is not None
    assert plugin.volume_tracker is not None


def test_process_historical_data():
    """Test processing historical data"""
    # Generate test data
    trades_df, quotes_df = generate_test_data()
    
    # Initialize plugin
    plugin = NetLargeVolumePlugin({})
    
    # Process data
    results = plugin.process_historical_data(trades_df, quotes_df)
    
    # Check results
    assert 'TEST' in results
    assert len(results['TEST']) > 0
    assert 'net_volume' in results['TEST'].columns
    
    # Verify cumulative nature
    net_volumes = results['TEST']['net_volume'].values
    assert len(net_volumes) > 1  # Should have multiple points
    
    # Check that buy/sell volumes are tracked
    assert all(results['TEST']['buy_volume'] >= 0)
    assert all(results['TEST']['sell_volume'] >= 0)


def test_accumulation_detection():
    """Test detection of accumulation periods"""
    # Generate test data with bias
    trades_df, quotes_df = generate_test_data()
    
    # Add artificial buy bias
    buy_mask = trades_df['price'] >= (trades_df['bid'] + trades_df['ask']) / 2
    trades_df.loc[buy_mask, 'size'] = trades_df.loc[buy_mask, 'size'] * 1.5
    
    # Process
    plugin = NetLargeVolumePlugin({})
    results = plugin.process_historical_data(trades_df, quotes_df)
    
    # Get final stats
    stats = plugin.get_summary_stats()
    
    # Should show net positive volume
    assert stats['TEST']['current']['net_volume'] > 0


if __name__ == '__main__':
    test_plugin_initialization()
    test_process_historical_data()
    test_accumulation_detection()
    print("All tests passed!")