Modular Backtest System - Technical Specifications
System Overview
The Modular Backtest System is a sophisticated Python-based backtesting framework designed for intraday trading analysis. It features a plugin-based architecture that allows for extensible calculation modules, unified data management, and comprehensive result storage. The system supports both GUI and CLI interfaces, making it suitable for interactive analysis and automated workflows.
Core Architecture Principles

Plugin-Based Extensibility: Calculations are implemented as self-contained plugins that can be dynamically loaded and executed
Shared Data Cache: Efficient data management through a centralized cache that prevents redundant API calls
Signal Aggregation: Multiple calculation signals are aggregated using a point & call consensus system
Comprehensive Storage: Results are stored both locally and in Supabase with unique identifiers (UIDs)
Real-time and Historical Analysis: Supports both live signal generation and historical backtesting

System Components
1. Main Entry Point (backtest_system.py)
The main entry point provides both GUI and CLI interfaces for running backtests.
Key Features:

PyQt6-based GUI dashboard for interactive backtesting
Command-line interface for automated/scripted execution
Real-time progress tracking and result visualization
Automatic plugin discovery and loading
Cache management and statistics

GUI Components:

Symbol input with direction selection (LONG/SHORT)
DateTime picker for entry time (UTC)
Calculation selector (individual or all calculations)
Results display with summary metrics and signal details
Supabase storage integration with auto-store option
Cache statistics viewer

Worker Threads:

BacktestWorker: Runs backtests asynchronously without blocking the UI
PushWorker: Handles Supabase storage operations in the background

2. Backtest Engine (core/engine.py)
The core orchestration component that manages the complete backtest lifecycle.
Primary Responsibilities:

Plugin registry management
Data requirement aggregation from all adapters
Shared data cache coordination
Signal collection and aggregation
Forward movement analysis
Result compilation and storage

Configuration (BacktestConfig):
python@dataclass
class BacktestConfig:
    symbol: str                    # Stock ticker
    entry_time: datetime          # UTC timestamp for entry
    direction: str                # 'LONG' or 'SHORT'
    historical_lookback_hours: int = 2
    forward_bars: int = 60        # 1-minute bars to track
    enabled_calculations: List[str] = []  # Empty = run all
    store_to_supabase: bool = False
Backtest Process Flow:

Collect data requirements from enabled adapters
Calculate required data window
Fetch all data once into shared cache
Initialize adapters with shared data
Collect entry signals from all calculations
Aggregate signals using point & call system
Analyze forward price movement (60 bars)
Store results locally and optionally to Supabase

3. Plugin System
Plugins are self-contained modules that implement specific calculation strategies.
Plugin Structure:
plugins/
├── plugin_loader.py      # Dynamic plugin discovery and loading
├── base_plugin.py        # Base class all plugins inherit from
└── m1_ema/              # Example plugin
    ├── __init__.py
    ├── plugin.py        # Plugin definition
    ├── adapter.py       # Calculation adapter
    ├── aggregator.py    # Data aggregation logic
    ├── storage.py       # Supabase storage handler
    └── schema.sql       # Database schema
Plugin Requirements:
Each plugin must provide:

Metadata: Name, version, adapter name
Adapter Class: Implements calculation logic
Data Requirements: Specifies needed data (bars, trades, quotes)
Storage Mapping: Converts signals to storage format
Storage Handler: Manages Supabase storage operations

Base Plugin Interface:
pythonclass BacktestPlugin:
    @property
    def name(self) -> str: ...
    
    @property
    def adapter_class(self) -> Type: ...
    
    @property
    def storage_table(self) -> str: ...
    
    def get_adapter_config(self) -> Dict[str, Any]: ...
    
    def get_storage_mapping(self, signal_data: Dict) -> Dict[str, Any]: ...
    
    async def store_results(self, supabase_client, uid: str, signal_data: Dict) -> bool: ...
4. Calculation Adapters
Adapters implement the actual calculation logic and conform to a standard interface.
Standard Adapter Interface:
pythonclass CalculationAdapter:
    def get_data_requirements(self) -> Dict: ...
    def set_data_cache(self, cache: Dict): ...
    def initialize(self, symbol: str): ...
    def feed_historical_data(self, data: pd.DataFrame, symbol: str): ...
    def get_signal_at_time(self, timestamp: datetime) -> StandardSignal: ...
StandardSignal Format:
python@dataclass
class StandardSignal:
    name: str                # Calculation name
    timestamp: datetime      # Signal timestamp
    direction: str          # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: float         # 0-100 scale
    confidence: float       # 0-100 scale
    metadata: Dict[str, Any]  # Calculation-specific data
5. Data Management
Shared Data Cache Structure:
python{
    'symbol': str,
    'data_start': datetime,
    'data_end': datetime,
    'entry_time': datetime,
    '1min_bars': pd.DataFrame,  # Always provided
    'trades': List[Dict],        # If requested
    'quotes': List[Dict]         # If requested
}
Data Flow:

Engine collects requirements from all adapters
Maximum lookback period is determined
Data is fetched once from the data source
Shared cache is distributed to all adapters
Adapters use cache for calculations

6. Signal Aggregation
The system uses a point & call consensus mechanism to aggregate multiple signals.
Aggregation Output:
python{
    'consensus_direction': str,      # 'BULLISH', 'BEARISH', 'NEUTRAL'
    'agreement_score': float,        # 0-100% agreement
    'vote_breakdown': Dict[str, int], # Direction counts
    'average_strength': float,
    'average_confidence': float,
    'participating_calculations': int,
    'total_calculations': int
}
7. Storage System
UID Format: TICKER.MMDDYY.HHMM.SIDE

Example: AAPL.062724.1335.L (Apple, June 27 2024, 13:35 UTC, Long)

Supabase Tables:

bt_index: Master record for each backtest
bt_bars: 75 bars of OHLCV data (-15 to +59 from entry)
bt_aggregated: Consensus results and forward analysis
Plugin-specific tables (e.g., bt_m1_ema)

Storage Process:

Generate UID from configuration
Store/update master record in bt_index
Store exactly 75 bars (15 before, 60 after entry)
Delegate calculation results to plugin storage handlers
Store aggregated results and forward analysis

Plugin Development Guide
Creating a New Plugin

Create plugin directory under plugins/
Implement plugin.py inheriting from BacktestPlugin
Create adapter.py implementing calculation logic
Add storage.py for Supabase storage handling
Define schema.sql for database table
Optional: Add aggregator.py for data preprocessing

Example Plugin Structure (M1 EMA):
python# plugin.py
class M1EMAPlugin(BacktestPlugin):
    @property
    def name(self) -> str:
        return "1-Min EMA Crossover"
    
    @property
    def adapter_class(self) -> Type:
        return M1EMABackAdapter
    
    def get_storage_mapping(self, signal_data: Dict) -> Dict[str, Any]:
        metadata = signal_data.get('metadata', {})
        return {
            'signal_direction': signal_data.get('direction'),
            'ema_9': float(metadata.get('ema_9', 0)),
            'ema_21': float(metadata.get('ema_21', 0)),
            # ... other fields
        }

# adapter.py
class M1EMABackAdapter(CalculationAdapter):
    def get_data_requirements(self) -> Dict:
        return {
            'bars': {
                'timeframe': '1min',
                'lookback_minutes': 120
            }
        }
    
    def feed_historical_data(self, data: pd.DataFrame, symbol: str):
        # Use self.data_cache to access shared data
        bars = self.data_cache.get('1min_bars')
        # Process bars and generate signals
Usage Examples
GUI Usage

Launch the dashboard:

bashpython backtest/backtest_system.py --gui

Enter parameters:

Symbol: AAPL
Entry Time: Select date/time in UTC
Direction: LONG or SHORT
Calculation: Choose specific or "All Calculations"


Click "Run Backtest" to execute
Review results:

Summary metrics (P&L, max moves)
Individual calculation signals
Aggregated consensus (if running all)
Forward analysis


Optionally push to Supabase for permanent storage

CLI Usage
bashpython backtest/backtest_system.py --cli \
    --symbol AAPL \
    --entry-time "2024-06-27 13:35:00" \
    --direction LONG \
    --calculation all \
    --store-supabase \
    --output results.json
Programmatic Usage
pythonfrom core.engine import BacktestEngine, BacktestConfig
from data.polygon_data_manager import PolygonDataManager
from plugins.plugin_loader import PluginLoader

# Initialize components
plugin_loader = PluginLoader()
plugins = plugin_loader.load_all_plugins()
data_manager = PolygonDataManager()

# Create engine
engine = BacktestEngine(
    data_manager=data_manager,
    plugin_registry=plugin_loader.get_registry()
)

# Register adapters
for name, config in plugin_loader.get_adapter_configs().items():
    adapter = config['adapter_class'](**config['adapter_config'])
    engine.register_adapter(name, adapter)

# Run backtest
config = BacktestConfig(
    symbol="AAPL",
    entry_time=datetime(2024, 6, 27, 13, 35, tzinfo=timezone.utc),
    direction="LONG"
)

result = await engine.run_backtest(config)
Performance Considerations

Data Caching: The shared cache prevents redundant API calls
Concurrent Processing: Adapters can be initialized concurrently
Batch Storage: Bars are stored in batches to Supabase
Memory Management: Historical data windows are limited
Async Operations: All I/O operations are asynchronous

Error Handling

Adapter Failures: Individual adapter failures don't crash the backtest
Storage Failures: Local storage succeeds even if Supabase fails
Data Validation: Invalid bars are filtered out
Type Conversion: NumPy types are converted for JSON serialization

Best Practices

Plugin Independence: Plugins should not depend on other plugins
Data Requirements: Declare minimal necessary data requirements
Signal Metadata: Include relevant calculation details in metadata
Storage Efficiency: Store only essential data in Supabase
Error Messages: Provide clear error messages for debugging

Future Extensions
The modular architecture supports:

Additional data sources beyond Polygon
New calculation strategies as plugins
Alternative storage backends
Real-time signal generation
Multi-symbol backtesting
Advanced visualization plugins
Performance analytics modules

This system provides a robust foundation for systematic backtesting with the flexibility to adapt to changing requirements through its plugin architecture.