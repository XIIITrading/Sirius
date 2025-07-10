# run_server.py
"""
Run the Polygon WebSocket/REST server

Usage:
    python run_server.py [--host HOST] [--port PORT] [--reload]
"""
import sys
import os
import argparse
import logging
from pathlib import Path

# Add the current directory to Python path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from polygon.polygon_server.server import app
from polygon.polygon_server.config import config


def setup_logging(log_level: str = "INFO"):
    """Configure logging for the server"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main entry point for running the server"""
    parser = argparse.ArgumentParser(description="Run the Polygon Data Server")
    parser.add_argument(
        "--host", 
        type=str, 
        default=config.host,
        help=f"Host to bind to (default: {config.host})"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=config.port,
        help=f"Port to bind to (default: {config.port})"
    )
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Check for API key
    if not config.polygon_api_key:
        logger.error("POLYGON_API_KEY environment variable not set!")
        logger.error("Please set: export POLYGON_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Log startup info
    logger.info("=" * 60)
    logger.info("Starting Polygon Data Server")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"API Documentation: http://{args.host}:{args.port}/docs")
    logger.info(f"WebSocket Endpoint: ws://{args.host}:{args.port}/ws/{{client_id}}")
    logger.info("=" * 60)
    
    # Configure uvicorn
    uvicorn_config = {
        "app": "polygon.polygon_server.server:app",
        "host": args.host,
        "port": args.port,
        "reload": args.reload,
        "log_level": args.log_level.lower(),
        "access_log": args.log_level == "DEBUG",
    }
    
    # Add workers only if not in reload mode
    if not args.reload and args.workers > 1:
        uvicorn_config["workers"] = args.workers
    
    try:
        # Run the server
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()