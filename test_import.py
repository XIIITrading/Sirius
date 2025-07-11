# Create a test file: test_import.py in the Sirius directory

import sys
import traceback

print("Testing imports...")

try:
    import live_monitor.dashboard
    print("✓ live_monitor.dashboard imported successfully")
except Exception as e:
    print(f"✗ Failed to import live_monitor.dashboard: {e}")
    traceback.print_exc()

try:
    from live_monitor.dashboard import main_dashboard
    print("✓ main_dashboard module imported successfully")
except Exception as e:
    print(f"✗ Failed to import main_dashboard module: {e}")
    traceback.print_exc()

try:
    # Try to import each dependency separately
    from live_monitor.styles import BaseStyles
    print("✓ BaseStyles imported successfully")
except Exception as e:
    print(f"✗ Failed to import BaseStyles: {e}")

try:
    from live_monitor.dashboard.components import (TickerEntry, TickerCalculations, 
                                                  EntryCalculations, PointCallEntry, 
                                                  PointCallExit, ChartWidget)
    print("✓ Components imported successfully")
except Exception as e:
    print(f"✗ Failed to import components: {e}")

try:
    from live_monitor.data import PolygonDataManager
    print("✓ PolygonDataManager imported successfully")
except Exception as e:
    print(f"✗ Failed to import PolygonDataManager: {e}")

try:
    from live_monitor.dashboard.components.chart.data.aggregate_data_handler import AggregateDataHandler
    print("✓ AggregateDataHandler imported successfully")
except Exception as e:
    print(f"✗ Failed to import AggregateDataHandler: {e}")

try:
    from live_monitor.dashboard.components.chart.zone_calculator import ZoneCalculator
    print("✓ ZoneCalculator imported successfully")
except Exception as e:
    print(f"✗ Failed to import ZoneCalculator: {e}")

print("\nChecking if main_dashboard.py exists...")
import os
path = os.path.join("live_monitor", "dashboard", "main_dashboard.py")
if os.path.exists(path):
    print(f"✓ File exists at {path}")
    print(f"  File size: {os.path.getsize(path)} bytes")
else:
    print(f"✗ File not found at {path}")