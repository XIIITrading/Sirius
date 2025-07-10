#!/usr/bin/env python3
"""
Live Monitor Dashboard Launcher - Enhanced Version

This script launches the Live Monitor trading dashboard with additional options.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_logging(debug=False):
    """Set up logging configuration"""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"live_monitor_{timestamp}.log"
    
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Launch the Live Monitor Trading Dashboard"
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--ticker', '-t',
        type=str,
        help='Initial ticker symbol to load'
    )
    
    parser.add_argument(
        '--style', '-s',
        choices=['dark', 'light'],
        default='dark',
        help='UI theme (default: dark)'
    )
    
    parser.add_argument(
        '--maximize', '-m',
        action='store_true',
        help='Start with maximized window'
    )
    
    return parser.parse_args()


def check_dependencies():
    """Check if required dependencies are installed"""
    required = {
        'PyQt6': 'PyQt6',
        'pyqtgraph': 'pyqtgraph',  # If you plan to use it for charts
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Error: Missing required dependencies:")
        for package in missing:
            print(f"  - {package}")
        print("\nInstall them using:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


def main():
    """Main launcher function"""
    # Parse arguments
    args = parse_arguments()
    
    # Check dependencies
    check_dependencies()
    
    # Set up logging
    log_file = setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("Live Monitor Dashboard Starting")
    logger.info("=" * 50)
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Project Root: {project_root}")
    logger.info(f"Log File: {log_file}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info("=" * 50)
    
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QT_VERSION_STR
        from live_monitor.dashboard.main_dashboard import LiveMonitorDashboard
        
        logger.info(f"PyQt6 Version: {QT_VERSION_STR}")
        
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("Live Monitor")
        app.setOrganizationName("Trading Tools")
        
        # Create dashboard
        dashboard = LiveMonitorDashboard()
        
        # Apply initial settings from arguments
        if args.ticker:
            logger.info(f"Setting initial ticker: {args.ticker}")
            dashboard.components.ticker_entry.set_ticker(args.ticker)
        
        if args.maximize:
            dashboard.showMaximized()
        else:
            dashboard.show()
        
        logger.info("Dashboard launched successfully")
        
        # Run the application
        exit_code = app.exec()
        
        logger.info(f"Application exited with code: {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()