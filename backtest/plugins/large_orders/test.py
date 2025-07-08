"""
Test module for Large Orders Grid plugin using real Polygon data
Tests successful large order detection and grid display
"""

import sys
import os
import argparse
from datetime import datetime, timezone
import asyncio
import json

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backtest.plugins.large_orders import run_analysis_with_data_manager, PLUGIN_NAME
from backtest.data.polygon_data_manager import PolygonDataManager
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut


async def test_with_real_data(symbol: str, timestamp_str: str, direction: str, show_grid: bool = False):
    """Test with real market data from Polygon"""
    print(f"\nTesting Large Orders Grid for {symbol} at {timestamp_str} ({direction})")
    
    # Parse timestamp
    test_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    if test_time.tzinfo is None:
        test_time = test_time.replace(tzinfo=timezone.utc)
    
    # Initialize data manager
    data_manager = PolygonDataManager(
        file_cache_hours=24
    )
    data_manager.set_current_plugin(f"{PLUGIN_NAME} Test")
    
    # Progress callback for testing
    def progress_callback(pct: int, msg: str):
        print(f"[{pct:3d}%] {msg}")
    
    try:
        # Run the plugin analysis
        print("\nRunning Large Orders Grid analysis...")
        result = await run_analysis_with_data_manager(
            symbol=symbol,
            entry_time=test_time,
            direction=direction,
            data_manager=data_manager,
            progress_callback=progress_callback
        )
        
        # Check for errors
        if 'error' in result:
            print(f"\nERROR: {result['error']}")
            return
        
        # Display signal information
        signal = result.get('signal', {})
        print(f"\n{'='*60}")
        print(f"SIGNAL ANALYSIS")
        print(f"{'='*60}")
        print(f"Direction: {signal.get('direction', 'N/A')}")
        print(f"Strength: {signal.get('strength', 0):.1f}%")
        print(f"Confidence: {signal.get('confidence', 0):.1f}%")
        print(f"Reason: {signal.get('reason', 'N/A')}")
        print(f"Aligned with {direction}: {'YES' if signal.get('aligned') else 'NO'}")
        
        # Display statistics
        large_orders_count = result.get('large_orders_count', 0)
        print(f"\n{'='*60}")
        print(f"LARGE ORDER STATISTICS")
        print(f"{'='*60}")
        print(f"Successful Large Orders Found: {large_orders_count}")
        
        # Display diagnostics
        diagnostics = result.get('diagnostics', {})
        if diagnostics:
            print(f"\nTotal Large Orders Detected: {diagnostics.get('total_large_orders', 0)}")
            print(f"Status Breakdown:")
            status_breakdown = diagnostics.get('status_breakdown', {})
            for status, count in status_breakdown.items():
                print(f"  {status}: {count}")
        
        # Display table data
        display_data = result.get('display_data', {})
        table_data = display_data.get('table_data', [])
        if table_data:
            print(f"\n{'='*60}")
            print(f"SUMMARY STATISTICS")
            print(f"{'='*60}")
            for metric, value in table_data:
                print(f"{metric:<25} {value:>20}")
        
        # Show grid if requested
        if show_grid:
            grid_config = display_data.get('grid_widget', {})
            grid_data = grid_config.get('data', [])
            
            if grid_data:
                print(f"\n{'='*60}")
                print(f"LARGE ORDERS PREVIEW (first 10)")
                print(f"{'='*60}")
                print(f"{'Time':<12} {'Price':>8} {'Size':>10} {'Side':<6} {'Impact':>8}")
                print(f"{'-'*52}")
                
                for order in grid_data[:10]:
                    timestamp = order['timestamp']
                    if isinstance(timestamp, datetime):
                        time_str = timestamp.strftime('%H:%M:%S')
                    else:
                        time_str = str(timestamp)[:8]
                    
                    print(f"{time_str:<12} "
                          f"${order['price']:>7.2f} "
                          f"{order['size']:>10,} "
                          f"{order['side']:<6} "
                          f"{order['impact_magnitude']:>8.2f}")
                
                print("\nLaunching Large Orders Grid...")
                
                # Import grid
                from backtest.plugins.large_orders.grid import LargeOrdersGrid
                
                app = QApplication.instance()
                if app is None:
                    app = QApplication(sys.argv)
                
                # Create main window
                window = QMainWindow()
                window.setWindowTitle(f"Large Orders Grid - {symbol}")
                window.resize(1000, 600)
                
                # Create central widget
                central_widget = QWidget()
                layout = QVBoxLayout(central_widget)
                
                # Create grid with configuration
                grid = LargeOrdersGrid(grid_config.get('config', {}))
                
                # Update grid with data
                grid.update_from_data(grid_data)
                
                # Connect selection signal
                def on_order_selected(order):
                    print(f"\nSelected Order:")
                    print(f"  Time: {order['timestamp']}")
                    print(f"  Price: ${order['price']:.2f}")
                    print(f"  Size: {order['size']:,}")
                    print(f"  Side: {order['side']}")
                    print(f"  Impact: {order['impact_magnitude']:.2f} spreads")
                
                grid.order_selected.connect(on_order_selected)
                
                layout.addWidget(grid)
                window.setCentralWidget(central_widget)
                window.show()
                
                # Add keyboard shortcuts
                escape = QShortcut(Qt.Key.Key_Escape, window)
                escape.activated.connect(window.close)
                
                print("\nGrid window opened:")
                print("- Click on rows to see order details")
                print("- Click column headers to sort")
                print("- Press ESC to close")
                app.exec()
            else:
                print("\nNo large orders found to display.")
        
        # Save result to file for inspection
        output_file = f"large_orders_grid_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            def json_serial(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")
            
            json.dump(result, f, indent=2, default=json_serial)
        print(f"\nFull result saved to: {output_file}")
        
    except Exception as e:
        print(f"\nERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate data manager report
    print("\nGenerating data manager report...")
    json_file, summary_file = data_manager.generate_data_report()
    print(f"Data reports saved to:\n  {json_file}\n  {summary_file}")


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description='Test Large Orders Grid Plugin')
    parser.add_argument('-s', '--symbol', type=str, required=True, help='Symbol to test')
    parser.add_argument('-t', '--timestamp', type=str, required=True, 
                       help='Timestamp for test (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('-d', '--direction', type=str, choices=['LONG', 'SHORT'], 
                       default='LONG', help='Direction bias for analysis')
    parser.add_argument('-g', '--grid', action='store_true', 
                       help='Show grid visualization')
    
    args = parser.parse_args()
    
    # Run async test
    asyncio.run(test_with_real_data(args.symbol, args.timestamp, args.direction, args.grid))


if __name__ == '__main__':
    main()