# Modular Backtesting System - Comprehensive Technical Documentation

## System Overview

This is a sophisticated modular backtesting system designed for quantitative trading analysis. The system orchestrates historical market data fetching, signal calculation, consensus aggregation, forward performance analysis, and result storage in both local and cloud (Supabase) databases. It uses a **plugin-based architecture** for maximum extensibility and maintainability.

### Core Architecture

```

Sirius/
├── backtest/
│   ├── backtest_system.py          # Main entry point (GUI/CLI)
│   ├── core/
│   │   ├── engine.py               # Orchestration engine
│   │   ├── data_manager.py         # Supabase data interface
│   │   ├── result_store.py         # Local result storage
│   │   └── signal_aggregator.py    # Point & Call consensus
│   ├── data/
│   │   └── polygon_data_manager.py # Polygon API + caching
│   ├── adapters/
│   │   └── base.py                 # Abstract adapter interface
│   ├── plugins/                    # Plugin-based calculations
│   │   ├── base_plugin.py          # Base plugin interface
│   │   ├── plugin_loader.py        # Dynamic plugin discovery
│   │   ├── m1_ema/                 # Example plugin
│   │   │   ├── plugin.py           # Plugin definition
│   │   │   ├── adapter.py          # Adapter implementation
│   │   │   └── schema.sql          # Supabase table schema
│   │   └── m5_ema/                 # Another plugin
│   │       ├── plugin.py
│   │       ├── adapter.py
│   │       └── schema.sql
│   ├── storage/
│   │   └── supabase_storage.py     # Cloud storage module
│   ├── cache/                       # Local cache directory
│   └── results/                     # Local results storage

```

## Key Components

### 1. BacktestEngine (`core/engine.py`)

The central orchestrator that manages the complete backtest lifecycle:

```python

python
class BacktestEngine:
    def __init__(self, data_manager=None, enable_supabase_storage=True):
# Initializes data manager (Polygon or Supabase)# Sets up local result store# Initializes signal aggregator# Configures Supabase storage if available

```

**Workflow:**

1. Collect data requirements from all plugins
2. Load historical data (based on max requirements)
3. Create shared data cache
4. Distribute data to plugin adapters
5. Collect entry signals from all calculations
6. Aggregate signals using Point & Call voting
7. Analyze forward price movement
8. Store results locally and optionally to Supabase

### 2. Plugin System

### Base Plugin Interface (`plugins/base_plugin.py`)

```python

python
class BacktestPlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for the calculation"""

    @property
    @abstractmethod
    def adapter_name(self) -> str:
        """Unique identifier for the adapter"""

    @property
    @abstractmethod
    def adapter_class(self) -> Type[CalculationAdapter]:
        """The adapter class for this calculation"""

    @property
    @abstractmethod
    def storage_table(self) -> str:
        """Supabase table name for storing results"""

    @abstractmethod
    def get_adapter_config(self) -> Dict[str, Any]:
        """Configuration for the adapter instance"""

    @abstractmethod
    def get_storage_mapping(self, signal_data: Dict) -> Dict[str, Any]:
        """Convert signal data to storage format"""

```

### Plugin Loader (`plugins/plugin_loader.py`)

- Auto-discovers all plugins in the plugins directory
- Registers adapters with the engine
- Provides storage mappings to the storage module
- Handles plugin initialization and error handling

### 3. Data Management Layer

### PolygonDataManager (`data/polygon_data_manager.py`)

- Two-tier caching: Memory (LRU) + File (Parquet)
- Extended window fetching (±500 bars) for efficiency
- Automatic timezone handling (UTC internally)
- Smart cache hit detection for adjacent queries
- **Enhanced Methods:**
    - `load_bars()` - Bar data for any timeframe
    - `load_trades()` - Tick-level trade data
    - `load_quotes()` - NBBO quote data

### Shared Data Cache

The engine creates a shared data cache that all plugins access:

```python

python
data_cache = {
    '1min_bars': pd.DataFrame,# Always fetched as base
    '5min_bars': pd.DataFrame,# Aggregated from 1min
    '15min_bars': pd.DataFrame,# Aggregated from 1min
    'trades': List[Dict],# Chronologically sorted
    'quotes': List[Dict],# Chronologically sorted
    'symbol': 'AAPL',
    'data_start': datetime,
    'data_end': datetime,
    'entry_time': datetime
}

```

### 4. Adapter System (`adapters/base.py`)

Abstract base class providing standardized interface:

```python

python
class CalculationAdapter(ABC):
    def get_data_requirements(self) -> Dict:
        """Declare data requirements for this calculation"""

    def set_data_cache(self, cache: Dict):
        """Receive shared data cache from engine"""

    @abstractmethod
    def feed_historical_data(self, data: pd.DataFrame, symbol: str) -> None:
        """Warm up calculation with historical data"""

    def get_signal_at_time(self, timestamp: datetime) -> Optional[StandardSignal]:
        """Get signal at specific time (entry point)"""

```

**StandardSignal Format:**

```python

python
StandardSignal(
    name="5-Min EMA Crossover",
    timestamp=datetime,
    direction="BULLISH|BEARISH|NEUTRAL",
    strength=0-100,
    confidence=0-100,
    metadata={...}# Calculation-specific data
)

```

### 5. Storage System

### Plugin-Aware Storage (`storage/supabase_storage.py`)

- Dynamically maps signals to tables using plugin system
- No hardcoded table mappings
- Each plugin defines its own storage schema
- Automatic numpy type conversion

## Database Schema

### Core Tables (Required)

```sql

sql
-- Master index for all backtests
CREATE TABLE bt_index (
    uid VARCHAR(50) PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    time TIME NOT NULL,
    side CHAR(1) NOT NULL CHECK (side IN ('L', 'S')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    calculations_run TEXT[] DEFAULT '{}'
);

-- Bar data (75 bars per backtest)
CREATE TABLE bt_bars (
    uid VARCHAR(50) NOT NULL,
    bar_index INTEGER NOT NULL CHECK (bar_index >= -15 AND bar_index <= 59),
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(10,4) NOT NULL,
    high DECIMAL(10,4) NOT NULL,
    low DECIMAL(10,4) NOT NULL,
    close DECIMAL(10,4) NOT NULL,
    volume BIGINT NOT NULL,
    PRIMARY KEY (uid, bar_index),
    FOREIGN KEY (uid) REFERENCES bt_index(uid) ON DELETE CASCADE
);

-- Aggregated results (consensus)
CREATE TABLE bt_aggregated (
    uid VARCHAR(50) PRIMARY KEY,
    consensus_direction VARCHAR(20),
    agreement_score DECIMAL(5,2),
    average_strength DECIMAL(5,2),
    average_confidence DECIMAL(5,2),
    participating_calculations INTEGER,
    total_calculations INTEGER,
    entry_price DECIMAL(10,4),
    exit_price DECIMAL(10,4),
    final_pnl DECIMAL(8,4),
    max_favorable_move DECIMAL(8,4),
    max_adverse_move DECIMAL(8,4),
    signal_matched_user BOOLEAN,
    trade_profitable BOOLEAN,
    signal_aligned_outcome BOOLEAN,
    FOREIGN KEY (uid) REFERENCES bt_index(uid) ON DELETE CASCADE
);

```

### Calculation-Specific Tables

Each plugin includes its own `schema.sql` file defining its storage table.

## Adding New Calculations - Plugin-Based Approach

### 1. Create Plugin Directory Structure

```

backtest/plugins/your_calculation/
├── __init__.py
├── plugin.py       # Plugin definition
├── adapter.py      # Adapter implementation
└── schema.sql      # Supabase table schema

```

### 2. Implement the Plugin (`plugin.py`)

```python

python
from plugins.base_plugin import BacktestPlugin
from .adapter import YourCalculationAdapter

class YourCalculationPlugin(BacktestPlugin):
    @property
    def name(self) -> str:
        return "Your Calculation Name"

    @property
    def adapter_name(self) -> str:
        return "your_calculation"

    @property
    def adapter_class(self) -> Type:
        return YourCalculationAdapter

    @property
    def storage_table(self) -> str:
        return "bt_your_calculation"

    def get_adapter_config(self) -> Dict[str, Any]:
        return {
            'param1': value1,
            'param2': value2
        }

    def get_storage_mapping(self, signal_data: Dict) -> Dict[str, Any]:
        """Convert signal to storage format"""
        metadata = signal_data.get('metadata', {})
        return {
            'signal_direction': signal_data.get('direction'),
            'signal_strength': float(signal_data.get('strength', 0)),
            'signal_confidence': float(signal_data.get('confidence', 0)),
# Add your specific fields here
            'your_metric': metadata.get('your_metric', 0)
        }

# Export plugin instance
plugin = YourCalculationPlugin()

```

### 3. Create the Adapter (`adapter.py`)

```python

python
from adapters.base import CalculationAdapter, StandardSignal
from modules.calculations.your_module import YourCalculator

class YourCalculationAdapter(CalculationAdapter):
    def __init__(self, **config):
        super().__init__(
            calculation_class=YourCalculator,
            config=config,
            name="Your Calculation Name"
        )
        self.data_cache = None

    def get_data_requirements(self) -> Dict:
        return {
            'bars': {'timeframe': '1min', 'lookback_minutes': 120},
            'trades': {'lookback_minutes': 30},# If needed
            'quotes': None# If not needed
        }

    def set_data_cache(self, cache: Dict):
        self.data_cache = cache

    def feed_historical_data(self, data: pd.DataFrame, symbol: str) -> None:
# Use data from cache based on your requirements
        bars = self.data_cache.get('1min_bars')
        trades = self.data_cache.get('trades', [])

# Process historical data up to entry time# Generate signals and store last one

```

### 4. Define Schema (`schema.sql`)

```

```

---