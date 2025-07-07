"""
Dashboard Simulation Test for Impact Success Plugin
This test simulates how the dashboard uses the plugin and chart module
"""

import asyncio
import argparse
import importlib
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any

# Add project root to path (like dashboard does)
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import like dashboard would
from backtest.plugins.impact_success import run_analysis, set_data_manager
from backtest.data.polygon_data_manager import PolygonDataManager

# For PyQt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardSimulator:
    """Simulates the dashboard's behavior for testing"""
    
    def __init__(self):
        # Initialize data manager like dashboard does
        self.data_manager = PolygonDataManager()
        
        # Set the data manager for the plugin (like dashboard would do)
        set_data_manager(self.data_manager)
        
    async def run_plugin(self, symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
        """Run plugin exactly like dashboard would"""
        print(f"\n{'='*80}")
        print(f"DASHBOARD SIMULATION TEST")
        print(f"Running plugin: Impact Success")
        print(f"Symbol: {symbol}")
        print(f"Entry Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Direction: {direction}")
        print(f"{'='*80}\n")
        
        # Run the plugin through its standard interface
        # Note: NOT passing data_manager as argument - it was already set via set_data_manager()
        print("Calling plugin's run_analysis function...")
        result = await run_analysis(symbol, entry_time, direction)
        
        return result
    
    def display_result(self, result: Dict[str, Any]):
        """Display result like dashboard would"""
        print("\n" + "="*60)
        print("PLUGIN RESULT")
        print("="*60)
        
        # Check for error
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            return
        
        # Display signal
        signal = result.get('signal', {})
        print(f"\nSignal Analysis:")
        print(f"  Direction: {signal.get('direction', 'N/A')}")
        print(f"  Strength: {signal.get('strength', 0):.1f}%")
        print(f"  Confidence: {signal.get('confidence', 0):.1f}%")
        print(f"  Reason: {signal.get('reason', 'N/A')}")
        print(f"  Aligned: {signal.get('aligned', False)}")
        
        # Display summary data
        display_data = result.get('display_data', {})
        print(f"\nSummary: {display_data.get('summary', 'N/A')}")
        print(f"\nDescription:\n{display_data.get('description', 'N/A')}")
        
        # Display table data
        table_data = display_data.get('table_data', [])
        if table_data:
            print("\nMetrics:")
            for metric, value in table_data:
                print(f"  {metric}: {value}")
        
        # Display stats if available
        stats = result.get('stats', {})
        if stats:
            print(f"\nInterpretation: {stats.get('interpretation', 'N/A')}")
        
        # Check for chart widget
        chart_config = display_data.get('chart_widget')
        if chart_config:
            print(f"\nChart Configuration Found:")
            print(f"  Module: {chart_config.get('module')}")
            print(f"  Type: {chart_config.get('type')}")
            print(f"  Data points: {len(chart_config.get('data', []))}")
            print(f"  Entry time: {chart_config.get('entry_time')}")
        else:
            print("\nNo chart configuration in result")
    
    def create_chart_from_result(self, result: Dict[str, Any], show_chart: bool = True):
        """Create chart exactly like dashboard would"""
        display_data = result.get('display_data', {})
        chart_config = display_data.get('chart_widget')
        
        if not chart_config:
            print("No chart widget configuration found")
            return
        
        print(f"\n{'='*60}")
        print("CREATING CHART (Dashboard Simulation)")
        print(f"{'='*60}")
        
        try:
            # Import the chart module exactly like dashboard does
            module_name = chart_config.get('module')
            chart_class_name = chart_config.get('type')
            
            print(f"Importing module: {module_name}")
            module = importlib.import_module(module_name)
            
            print(f"Getting class: {chart_class_name}")
            ChartClass = getattr(module, chart_class_name)
            
            print(f"Creating chart instance...")
            chart = ChartClass()
            
            # Update chart with data
            chart_data = chart_config.get('data', [])
            print(f"Updating chart with {len(chart_data)} data points...")
            
            if hasattr(chart, 'update_from_data'):
                chart.update_from_data(chart_data)
                print("Called update_from_data successfully")
            else:
                print("ERROR: Chart has no update_from_data method!")
                return
            
            # Add entry marker if supported
            if chart_config.get('entry_time') and hasattr(chart, 'add_marker'):
                chart.add_marker(30, "Entry", "#ff0000")
                print("Added entry marker")
            
            print("✅ Chart created successfully!")
            
            # Display the chart if requested
            if show_chart:
                self._show_chart_window(chart, result)
                
        except Exception as e:
            print(f"❌ Error creating chart: {e}")
            import traceback
            traceback.print_exc()
    
    def _show_chart_window(self, chart, result: Dict[str, Any]):
        """Display chart in a window like dashboard would"""
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Create window
        window = QMainWindow()
        window.setWindowTitle(f"Dashboard Simulation - {result.get('plugin_name', 'Plugin')} Chart")
        window.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add info label
        signal = result.get('signal', {})
        info_text = (
            f"Plugin: {result.get('plugin_name', 'Unknown')}\n"
            f"Symbol: {result.get('symbol', 'N/A')}\n"
            f"Direction: {result.get('direction', 'N/A')}\n"
            f"Signal: {signal.get('direction', 'N/A')} "
            f"(Strength: {signal.get('strength', 0):.1f}%, "
            f"Confidence: {signal.get('confidence', 0):.1f}%)"
        )
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(info_label)
        
        # Add the chart
        layout.addWidget(chart)
        
        # Apply styling
        window.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel {
                background-color: #1e1e1e;
                color: #ffffff;
            }
        """)
        
        window.show()
        app.exec()


async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(
        description='Dashboard Simulation Test - Tests Impact Success plugin and chart integration'
    )
    
    parser.add_argument('-s', '--symbol', type=str, required=True,
                       help='Stock symbol (e.g., AAPL, TSLA, SPY)')
    parser.add_argument('-t', '--time', type=str, default=None,
                       help='Entry time in format "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument('-d', '--direction', type=str, choices=['LONG', 'SHORT'],
                       default='LONG', help='Trade direction')
    parser.add_argument('-c', '--chart', action='store_true',
                       help='Display chart window')
    
    args = parser.parse_args()
    
    # Parse datetime
    if args.time:
        try:
            entry_time = datetime.strptime(args.time, "%Y-%m-%d %H:%M:%S")
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"ERROR: Invalid datetime format: {args.time}")
            return
    else:
        entry_time = datetime.now(timezone.utc)
        print(f"Using current time: {entry_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Create simulator
    simulator = DashboardSimulator()
    
    try:
        # Step 1: Run the plugin
        result = await simulator.run_plugin(
            symbol=args.symbol.upper(),
            entry_time=entry_time,
            direction=args.direction
        )
        
        # Step 2: Display the result
        simulator.display_result(result)
        
        # Step 3: Create and display chart (if requested)
        if args.chart:
            simulator.create_chart_from_result(result, show_chart=True)
        else:
            # Still test chart creation without showing
            simulator.create_chart_from_result(result, show_chart=False)
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())