"""
Test module for Impact Success plugin using real Polygon data
Tests large order pressure tracking (buy/sell volume imbalance)
"""

import sys
import os
import argparse
from datetime import datetime, timedelta, timezone
import pandas as pd
import asyncio
import json

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backtest.plugins.impact_success.plugin import Plugin
from backtest.data.polygon_data_manager import PolygonDataManager
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


async def test_with_real_data(symbol: str, timestamp_str: str, direction: str, show_chart: bool = False):
    """Test with real market data from Polygon"""
    print(f"\nTesting Large Order Pressure Analysis for {symbol} at {timestamp_str} ({direction})")
    
    # Parse timestamp
    test_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    if test_time.tzinfo is None:
        test_time = test_time.replace(tzinfo=timezone.utc)
    
    # Initialize data manager
    data_manager = PolygonDataManager(
        disable_polygon_cache=True,  # Ensure fresh data
        file_cache_hours=24
    )
    data_manager.set_current_plugin("Impact Success Test")
    
    # Initialize plugin
    plugin = Plugin()
    
    # Progress callback for testing
    def progress_callback(pct: int, msg: str):
        print(f"[{pct:3d}%] {msg}")
    
    try:
        # Run the plugin analysis
        print("\nRunning Large Order Pressure analysis...")
        result = await plugin.run(
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
        
        # Display pressure statistics
        stats = result.get('stats', {})
        if stats:
            print(f"\n{'='*60}")
            print(f"PRESSURE STATISTICS (Last 30 minutes)")
            print(f"{'='*60}")
            print(f"Total Buy Volume:  {stats.get('total_buy_volume', 0):>10,} shares")
            print(f"Total Sell Volume: {stats.get('total_sell_volume', 0):>10,} shares")
            print(f"Net Pressure:      {stats.get('net_pressure', 0):>+10,} shares")
            print(f"Buy Orders:        {stats.get('buy_order_count', 0):>10} large orders")
            print(f"Sell Orders:       {stats.get('sell_order_count', 0):>10} large orders")
            print(f"Cumulative Total:  {stats.get('current_cumulative', 0):>+10,} shares")
            print(f"\nInterpretation: {stats.get('interpretation', 'No data')}")
        
        # Display table data
        display_data = result.get('display_data', {})
        table_data = display_data.get('table_data', [])
        if table_data:
            print(f"\n{'='*60}")
            print(f"DETAILED METRICS")
            print(f"{'='*60}")
            for metric, value in table_data:
                print(f"{metric:<25} {value:>20}")
        
        # Show chart if requested
        if show_chart:
            chart_config = display_data.get('chart_widget', {})
            chart_data = chart_config.get('data', [])
            
            if chart_data:
                print(f"\n{'='*60}")
                print(f"PRESSURE DATA PREVIEW (last 10 points)")
                print(f"{'='*60}")
                print(f"{'Time':<12} {'Net':>10} {'Cumulative':>12} {'Buy Vol':>10} {'Sell Vol':>10}")
                print(f"{'-'*60}")
                
                for point in chart_data[-10:]:
                    timestamp = datetime.fromisoformat(point['timestamp'])
                    print(f"{timestamp.strftime('%H:%M:%S'):<12} "
                          f"{point['net_pressure']:>+10,} "
                          f"{point['cumulative_pressure']:>+12,} "
                          f"{point['buy_volume']:>10,} "
                          f"{point['sell_volume']:>10,}")
                
                print("\nLaunching pressure chart...")
                
                # Import chart
                from backtest.plugins.impact_success.chart import ImpactSuccessChart
                
                app = QtWidgets.QApplication.instance()
                if app is None:
                    app = QtWidgets.QApplication(sys.argv)
                
                # Create main window
                window = QtWidgets.QMainWindow()
                window.setWindowTitle(f"Large Order Pressure Chart - {symbol}")
                window.resize(1200, 600)
                
                # Create chart (show cumulative by default)
                chart = ImpactSuccessChart(window_minutes=30, show_cumulative=True)
                
                # Update chart with data
                chart.update_from_data(chart_data)
                
                # Add entry marker if available
                if chart_config.get('entry_time'):
                    # Calculate minutes from start
                    if chart_data:
                        start_time = datetime.fromisoformat(chart_data[0]['timestamp'])
                        entry_time = datetime.fromisoformat(chart_config['entry_time'])
                        minutes_from_start = (entry_time - start_time).total_seconds() / 60
                        chart.add_marker(minutes_from_start, "Entry", "#ffff00")
                
                # Add toggle button for cumulative/period view
                toolbar = window.addToolBar("Chart Options")
                toggle_action = toolbar.addAction("Toggle Cumulative/Period")
                toggle_action.triggered.connect(chart.toggle_cumulative)
                
                # Set as central widget
                window.setCentralWidget(chart)
                window.show()
                
                # Add keyboard shortcuts
                escape = QtGui.QShortcut(QtCore.Qt.Key.Key_Escape, window)
                escape.activated.connect(window.close)
                
                toggle = QtGui.QShortcut(QtCore.Qt.Key.Key_Space, window)
                toggle.activated.connect(chart.toggle_cumulative)
                
                print("\nChart window opened:")
                print("- Press ESC to close")
                print("- Press SPACE to toggle between cumulative and period views")
                app.exec()
            else:
                print("\nNo pressure data available to display.")
        
        # Show detection statistics
        summary_stats = plugin.tracker.get_summary_stats(symbol)
        if summary_stats:
            detection_stats = summary_stats.get('current_stats', {})
            thresholds = summary_stats.get('detection_thresholds', {})
            
            print(f"\n{'='*60}")
            print(f"LARGE ORDER DETECTION STATS")
            print(f"{'='*60}")
            print(f"Average Trade Size: {detection_stats.get('mean_size', 0):.0f}")
            print(f"Std Deviation:      {detection_stats.get('std_size', 0):.0f}")
            print(f"Total Trades:       {detection_stats.get('trade_count', 0)}")
            print(f"\nDetection Thresholds:")
            print(f"- Ratio (1.5x avg): {thresholds.get('ratio_threshold', 0):.0f} shares")
            if thresholds.get('stdev_threshold'):
                print(f"- StDev (1.25σ):    {thresholds.get('stdev_threshold', 0):.0f} shares")
        
        # Save result to file for inspection
        output_file = f"pressure_test_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nFull result saved to: {output_file}")
        
    except Exception as e:
        print(f"\nERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate data manager report
    print("\nGenerating data manager report...")
    data_manager.generate_data_report()


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description='Test Large Order Pressure Analysis')
    parser.add_argument('-s', '--symbol', type=str, required=True, help='Symbol to test')
    parser.add_argument('-t', '--timestamp', type=str, required=True, 
                       help='Timestamp for test (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('-d', '--direction', type=str, choices=['LONG', 'SHORT'], 
                       default='LONG', help='Direction bias for analysis')
    parser.add_argument('-c', '--chart', action='store_true', 
                       help='Show pressure chart visualization')
    
    args = parser.parse_args()
    
    # Run async test
    asyncio.run(test_with_real_data(args.symbol, args.timestamp, args.direction, args.chart))


if __name__ == '__main__':
    main()