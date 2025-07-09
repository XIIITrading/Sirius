#!/usr/bin/env python3
"""
Run Live Trading Monitor
Execute from root directory: python run_live_monitor.py
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
import argparse

# Add current directory to path so we can import live_monitor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the monitor
from live_monitor.main import LiveTradingMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Live Trading Monitor - Multi-Signal Dashboard'
    )
    
    parser.add_argument(
        'symbols',
        nargs='*',
        default=['AAPL', 'TSLA', 'SPY', 'QQQ'],
        help='Symbols to monitor (default: AAPL TSLA SPY QQQ)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='Display update interval in seconds (default: 1)'
    )
    
    parser.add_argument(
        '--use-server',
        action='store_true',
        help='Connect through local Polygon server instead of direct connection'
    )
    
    parser.add_argument(
        '--server-url',
        default='ws://localhost:8200',
        help='Polygon server WebSocket URL (default: ws://localhost:8200)'
    )
    
    parser.add_argument(
        '--modules',
        nargs='*',
        help='Specific modules to enable (default: all)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate symbols
    symbols = [s.upper() for s in args.symbols]
    if not symbols:
        logger.error("No symbols specified")
        return
    
    logger.info("=" * 60)
    logger.info("LIVE TRADING MONITOR")
    logger.info("=" * 60)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Update Interval: {args.interval}s")
    logger.info(f"Connection: {'Server' if args.use_server else 'Direct'}")
    if args.modules:
        logger.info(f"Modules: {', '.join(args.modules)}")
    logger.info("=" * 60)
    
    # Create monitor with configuration
    monitor_config = {
        'symbols': symbols,
        'display_interval': args.interval,
        'use_server': args.use_server,
        'server_url': args.server_url,
        'enabled_modules': args.modules  # None means all modules
    }
    
    # Initialize monitor
    try:
        monitor = LiveTradingMonitor(**monitor_config)
        
        logger.info("Starting monitor... (Press Ctrl+C to stop)")
        await monitor.start()
        
    except KeyboardInterrupt:
        logger.info("\nShutting down monitor...")
    except Exception as e:
        logger.error(f"Error running monitor: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run():
    """Run the async main function"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C
        print("\nMonitor stopped.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()