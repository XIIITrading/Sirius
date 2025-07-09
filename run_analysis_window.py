# run_analysis_window.py
"""
Stock Analysis Window Launcher
Purpose: Launch the comprehensive stock analysis tool with all indicators
Usage: 
    python run_analysis_window.py [TICKER]
    python run_analysis_window.py TSLA
    python run_analysis_window.py  # Defaults to AAPL
"""

import sys
import os
import logging
from datetime import datetime

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
log_dir = os.path.join(project_root, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f'analysis_window_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def check_environment():
    """Check if all required packages are installed"""
    required_packages = {
        'PyQt6': 'PyQt6',
        'pyqtgraph': 'pyqtgraph',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'polygon': 'polygon-api-client'  # If using Polygon API
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(pip_name)
            logger.error(f"✗ {package} is NOT installed")
    
    if missing_packages:
        logger.error("\nMissing required packages!")
        logger.error("Install them using:")
        logger.error(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def initialize_data_manager():
    """Initialize the data manager for supply/demand calculations"""
    try:
        from market_review.dashboards.data_manager import DataManager
        from market_review.calculations.zones.supply_demand import set_data_manager
        
        # Get data manager instance
        data_manager = DataManager.get_instance()
        
        # Set data manager for supply/demand module
        set_data_manager(data_manager)
        
        logger.info("✓ Data manager initialized successfully")
        return data_manager
        
    except Exception as e:
        logger.warning(f"Could not initialize data manager: {e}")
        logger.warning("Supply/Demand analysis may not work properly")
        
        # Try to use PolygonDataManager as fallback
        try:
            from market_review.dashboards.components.supply_demand_chart import PolygonDataManager, initialize_data_manager
            
            polygon_dm = PolygonDataManager()
            initialize_data_manager(polygon_dm)
            logger.info("✓ Using PolygonDataManager as fallback")
            return polygon_dm
            
        except Exception as e2:
            logger.error(f"Failed to initialize any data manager: {e2}")
            return None


def main():
    """Main launcher function"""
    logger.info("=" * 70)
    logger.info("STOCK ANALYSIS WINDOW LAUNCHER")
    logger.info("=" * 70)
    
    # Parse command line arguments
    default_ticker = "AAPL"
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else default_ticker
    
    logger.info(f"Starting analysis window for ticker: {ticker}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Project root: {project_root}")
    
    # Check environment
    if not check_environment():
        return 1
    
    try:
        # Import PyQt6 and check version
        from PyQt6.QtWidgets import QApplication, QStyleFactory
        from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
        from PyQt6.QtGui import QPalette, QColor
        
        logger.info(f"Qt version: {QT_VERSION_STR}")
        logger.info(f"PyQt version: {PYQT_VERSION_STR}")
        
    except ImportError as e:
        logger.error(f"Failed to import PyQt6: {e}")
        return 1
    
    try:
        # Import pyqtgraph and configure
        import pyqtgraph as pg
        
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('background', '#1a1a1a')
        pg.setConfigOption('foreground', '#ffffff')
        
        logger.info("✓ PyQtGraph configured successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import pyqtgraph: {e}")
        return 1
    
    try:
        # Import required modules
        logger.info("\nImporting market_review modules...")
        
        from market_review.dashboards.stock_analysis_window import StockAnalysisWindow
        
        logger.info("✓ Successfully imported StockAnalysisWindow")
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure you're running from the project root directory")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create Qt Application
    app = QApplication(sys.argv)
    app.setApplicationName("Stock Analysis Tool")
    app.setOrganizationName("MarketReview")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Optional: Set dark palette for entire application
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(26, 26, 26))
    dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(42, 42, 42))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(66, 66, 66))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 0, 0))
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(42, 42, 42))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(dark_palette)
    
    # Initialize data manager
    logger.info("\nInitializing data manager...")
    data_manager = initialize_data_manager()
    
    # Create and show main window
    try:
        logger.info(f"\nCreating analysis window for {ticker}...")
        
        window = StockAnalysisWindow(ticker)
        
        # Set window properties
        window.setWindowTitle(f"Stock Analysis - {ticker}")
        window.setGeometry(150, 150, 1400, 900)
        window.setMinimumSize(1200, 700)
        
        # Connect window close signal
        def on_window_closed(closed_ticker):
            logger.info(f"Analysis window closed for {closed_ticker}")
            app.quit()
        
        window.window_closed.connect(on_window_closed)
        
        # Show window
        window.show()
        
        logger.info("\n" + "=" * 70)
        logger.info("STOCK ANALYSIS WINDOW LAUNCHED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info("\nAvailable features:")
        logger.info("  📊 HVN Analysis - Volume profile with high volume nodes")
        logger.info("  📈 Supply/Demand - Order blocks and breaker blocks")
        logger.info("  📉 Camarilla Pivots - Support and resistance levels")
        logger.info("  📋 Summary - Integrated view with zone aggregation")
        logger.info("  💼 Trading Plan - (Coming soon)")
        logger.info("\nControls:")
        logger.info("  - Change ticker using the input field in header")
        logger.info("  - Toggle indicators on/off in Summary tab")
        logger.info("  - Click zones for details")
        logger.info("  - Use mouse wheel to zoom charts")
        logger.info("\n" + "=" * 70)
        
    except Exception as e:
        logger.error(f"Failed to create analysis window: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run the application
    return app.exec()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)