"""Main entry point for backtest system"""

import sys
import asyncio
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QEventLoop, QThread
import qasync

from .dashboard import BacktestDashboard


def main():
    """Run the backtest system"""
    # Create Qt application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set up async event loop
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    # Create and show dashboard
    dashboard = BacktestDashboard()
    dashboard.show()
    
    # Run event loop
    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()