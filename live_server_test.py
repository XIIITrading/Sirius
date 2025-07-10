# live_server_test.py
"""
Test script for Polygon WebSocket connection
"""
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PyQt6.QtCore import QCoreApplication, QTimer
from live_monitor.data.polygon_data_manager import PolygonDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Test the Polygon data manager connection"""
    app = QCoreApplication(sys.argv)
    
    # Create data manager
    print("Creating Polygon Data Manager...")
    manager = PolygonDataManager()
    
    # Connect signals to print output
    manager.market_data_updated.connect(
        lambda data: print(f"\n📊 Market Update: {data}")
    )
    
    manager.connection_status_changed.connect(
        lambda status: print(f"\n🔌 Connection Status: {'Connected' if status else 'Disconnected'}")
    )
    
    manager.error_occurred.connect(
        lambda error: print(f"\n❌ Error: {error}")
    )
    
    # Connect to server
    print("\nConnecting to Polygon server...")
    manager.connect()
    
    # Set a test symbol after a short delay to ensure connection
    def set_test_symbol():
        symbol = "AAPL"  # Change this to test different symbols
        print(f"\nSubscribing to {symbol}...")
        manager.change_symbol(symbol)
    
    QTimer.singleShot(1000, set_test_symbol)  # Wait 1 second then subscribe
    
    # Optional: Stop after some time for testing
    # QTimer.singleShot(30000, app.quit)  # Stop after 30 seconds
    
    print("\nTest running. Press Ctrl+C to stop.\n")
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\n\nStopping test...")
        manager.disconnect()
        app.quit()

if __name__ == "__main__":
    main()