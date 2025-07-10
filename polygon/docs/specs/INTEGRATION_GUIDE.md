REST_SERVER
## 3. REST_SERVER.md

```markdown
# Polygon REST Server Documentation

## Overview

The Polygon REST server provides HTTP endpoints for accessing polygon data. Built with FastAPI, it offers automatic API documentation, WebSocket support, and high-performance data serving.

## Starting the Server

```bash
# Set environment variables
export POLYGON_API_KEY=your_key_here
export SERVER_PORT=8200

# Start server
python -m polygon.polygon_server.server

# Or with uvicorn directly
uvicorn polygon.polygon_server.server:app --host 0.0.0.0 --port 8200
Configuration
Environment variables:

SERVER_HOST: Host to bind (default: "0.0.0.0")
SERVER_PORT: Port number (default: 8200)
SERVER_WORKERS: Number of workers (default: 1)
CORS_ORIGINS: Comma-separated origins (default: "*")
CACHE_DIR: Cache directory (default: "./polygon_cache")
CACHE_TTL: Cache TTL in seconds (default: 3600)
LOG_LEVEL: Logging level (default: "INFO")

REST Endpoints
Health & Status
GET /health
Basic health check.
Response:
json{
    "status": "healthy",
    "timestamp": "2023-12-01T10:30:00"
}
GET /status
Detailed server status.
Response:
json{
    "status": "healthy",
    "version": "1.0.0",
    "uptime_seconds": 3600,
    "polygon_connected": true,
    "websocket_clients": 5,
    "cache_stats": {...},
    "rate_limit_status": {...},
    "system_metrics": {
        "memory_mb": 256.5,
        "cpu_percent": 12.5,
        "threads": 4
    }
}
Market Data Endpoints
POST /api/v1/bars
Get historical OHLCV bars.
Request Body:
json{
    "symbol": "AAPL",
    "timeframe": "5min",
    "start_date": "2023-01-01",
    "end_date": "2023-01-31",
    "limit": 1000,
    "use_cache": true,
    "validate": true
}
Response:
json{
    "symbol": "AAPL",
    "timeframe": "5min",
    "start_date": "2023-01-01",
    "end_date": "2023-01-31",
    "bar_count": 1000,
    "data": [
        {
            "timestamp": "2023-01-01T09:30:00Z",
            "open": 150.25,
            "high": 150.50,
            "low": 150.00,
            "close": 150.45,
            "volume": 100000,
            "vwap": 150.35,
            "transactions": 500
        }
    ],
    "cached": true,
    "validation": {...}
}
POST /api/v1/bars/multiple
Get bars for multiple symbols.
Request Body:
json{
    "symbols": ["AAPL", "GOOGL", "MSFT"],
    "timeframe": "1day",
    "start_date": "2023-01-01",
    "end_date": "2023-01-31",
    "parallel": true
}
Response:
json{
    "AAPL": {
        "success": true,
        "bar_count": 20,
        "first_bar": "2023-01-03T00:00:00Z",
        "last_bar": "2023-01-31T00:00:00Z"
    },
    "GOOGL": {...},
    "MSFT": {...}
}
GET /api/v1/latest/{symbol}
Get latest price for a symbol.
Response:
json{
    "symbol": "AAPL",
    "price": 175.50,
    "timestamp": "2023-12-01T15:30:00Z"
}
POST /api/v1/validate
Validate symbols.
Request Body:
json{
    "symbols": ["AAPL", "INVALID", "GOOGL"],
    "detailed": true
}
Response:
json{
    "AAPL": {
        "valid": true,
        "normalized": "AAPL",
        "type": "equity",
        "characteristics": [],
        "warnings": []
    },
    "INVALID": {
        "valid": false,
        "error": "Invalid symbol format"
    },
    "GOOGL": {...}
}
GET /api/v1/search
Search for symbols.
Query Parameters:

query: Search string
active_only: Only active symbols (default: true)

Response:
json{
    "query": "apple",
    "count": 3,
    "results": [
        {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "market": "stocks",
            "locale": "us",
            "primary_exchange": "XNAS",
            "type": "CS",
            "active": true,
            "currency_name": "usd"
        }
    ]
}
Cache Management
GET /api/v1/cache/stats
Get cache statistics.
Response:
json{
    "total_entries": 150,
    "total_size_mb": 125.5,
    "top_symbols": [...],
    "by_timeframe": [...],
    "recent_access": [...]
}
DELETE /api/v1/cache
Clear cache data.
Query Parameters:

symbol: Clear specific symbol (optional)
older_than_days: Clear data older than N days (optional)

Response:
json{
    "files_removed": 10,
    "space_freed_mb": 50.5,
    "entries_removed": 10
}
GET /api/v1/rate-limit
Get current rate limit status.
Response:
json{
    "minute": {
        "used": 45,
        "limit": 100,
        "remaining": 55,
        "usage_pct": 45.0
    },
    "daily": {...},
    "queue_size": 0,
    "tier": "advanced"
}
Request Models
TimeframeEnum
Valid values:

"1min", "5min", "15min", "30min"
"1hour", "4hour"
"1day", "1week", "1month"

BarsRequest

symbol (required): Stock symbol
timeframe: Bar timeframe (default: "1day")
start_date: Start date YYYY-MM-DD
end_date: End date YYYY-MM-DD
limit: Maximum bars to return
use_cache: Use cached data (default: true)
validate: Validate data quality (default: true)

Error Responses
All errors follow this format:
json{
    "error": "error_type",
    "message": "Human readable message",
    "details": {...},
    "timestamp": "2023-12-01T10:30:00Z"
}
Common HTTP status codes:

400: Bad Request (invalid parameters)
401: Unauthorized (invalid API key)
404: Not Found (symbol not found)
429: Too Many Requests (rate limited)
500: Internal Server Error

Performance Features

ORJSON: Fast JSON serialization
Response Caching: Configurable TTL
Parallel Processing: Multiple symbol fetching
Streaming: Large datasets streamed efficiently
Compression: Gzip response compression

API Documentation
When server is running:

Swagger UI: http://localhost:8200/docs
ReDoc: http://localhost:8200/redoc
OpenAPI JSON: http://localhost:8200/openapi.json

CORS Configuration
Default allows all origins. For production:
bashexport CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
Deployment
Docker
dockerfileFROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8200
CMD ["uvicorn", "polygon.polygon_server.server:app", "--host", "0.0.0.0", "--port", "8200"]
Production Settings
bash# Use multiple workers
export SERVER_WORKERS=4

# Enable production logging
export LOG_LEVEL=WARNING

# Restrict CORS
export CORS_ORIGINS="https://yourdomain.com"

# Start with gunicorn
gunicorn polygon.polygon_server.server:app -w 4 -k uvicorn.workers.UvicornWorker