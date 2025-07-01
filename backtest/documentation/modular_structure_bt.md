Technical Specification: Modular Backtest System Architecture
Table of Contents

Executive Summary
Architecture Overview
Design Principles
System Components
Plugin Architecture
Implementation Guide
Data Flow
API Reference
Migration Guide
Best Practices
Troubleshooting


1. Executive Summary
The Modular Backtest System represents a paradigm shift in backtesting architecture, moving from a complex multi-layered system to a streamlined, plugin-based architecture. This document provides comprehensive technical specifications for understanding, implementing, and extending the system.
Key Achievements:

Reduced complexity from 7+ files per calculation to 2 files
Eliminated middleware layers (adapters, bridges, aggregators)
Direct plugin-to-UI communication with no transformation layers
Self-contained plugins that handle their own data fetching and processing
Parallel execution capability for multiple calculations
Clean separation between calculation logic and presentation

System Philosophy:
"Each plugin is a complete, independent unit that receives simple inputs (symbol, time, direction) and returns formatted results ready for display."

2. Architecture Overview
2.1 High-Level Architecture
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Controls  │  │ Multi-Result │  │  Single Result  │   │
│  │   (Input)   │  │   Viewer     │  │     Viewer      │   │
│  └──────┬──────┘  └──────────────┘  └─────────────────┘   │
│         │                                                    │
└─────────┼────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                      PLUGIN RUNNER                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Plugin    │  │   Plugin    │  │   Plugin    │  ...   │
│  │ Discovery   │  │ Execution   │  │   Results   │        │
│  └─────────────┘  └──────┬──────┘  └─────────────┘        │
│                          │                                   │
└──────────────────────────┼───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                        PLUGINS                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │M1 Market     │  │M5 Market     │  │Your Custom   │     │
│  │Structure     │  │Structure     │  │Plugin        │ ... │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
2.2 Old vs New Architecture Comparison
Old Architecture (Complex, Multi-layered):
User Input → BacktestEngine → DataCache → Adapter → 
Aggregator → Calculation → Adapter → SignalAggregator → 
ResultStore → Transformer → UI
New Architecture (Simple, Direct):
User Input → Plugin → UI

3. Design Principles
3.1 Single Responsibility
Each plugin is responsible for one calculation type and handles everything related to that calculation internally.
3.2 Self-Containment
Plugins are completely self-contained units that:

Fetch their own data
Process calculations
Format results for display
Handle errors gracefully

3.3 No Shared State
No global state or shared caches. Each plugin manages its own data and state.
3.4 Direct Communication
Plugins communicate directly with the UI without intermediate transformation layers.
3.5 Standardized Interface
All plugins implement the same simple interface: run_analysis(symbol, entry_time, direction)

4. System Components
4.1 Directory Structure
backtest/
├── plugins/                      # Plugin directory
│   ├── base_plugin.py           # Base interface all plugins implement
│   ├── m1_market_structure/     # Example plugin
│   │   ├── __init__.py         # Plugin entry point
│   │   └── plugin.py           # Plugin implementation
│   └── [your_plugin]/          # Your custom plugins
│
├── backtest_system/             # UI application
│   ├── __init__.py
│   ├── main.py                 # Application entry point
│   ├── dashboard.py            # Main window
│   └── components/
│       ├── plugin_runner.py    # Plugin discovery and execution
│       ├── result_viewer.py    # Single result display
│       └── multi_result_viewer.py # Multiple results table
│
├── storage/                     # Future: Storage components
├── aggregator/                  # Future: Results aggregation
└── run.py                      # Launch script
4.2 External Dependencies
modules/                         # Existing calculation modules
    calculations/
        market_structure/
            m1_market_structure.py
            m5_market_structure.py

polygon/                         # Data fetching
    data_manager.py

5. Plugin Architecture
5.1 Plugin Interface
Every plugin must implement the following interface:
python# plugins/base_plugin.py
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any

class BacktestPlugin(ABC):
    """Base class for all backtest plugins"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable plugin name"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass
    
    @abstractmethod
    async def run_analysis(
        self, 
        symbol: str, 
        entry_time: datetime, 
        direction: str
    ) -> Dict[str, Any]:
        """
        Run the analysis.
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            entry_time: Entry time in UTC
            direction: 'LONG' or 'SHORT'
            
        Returns:
            Standardized result dictionary
        """
        pass
5.2 Plugin Result Format
All plugins must return results in this standardized format:
python{
    'plugin_name': str,           # Name of the plugin
    'timestamp': datetime,        # Entry time
    'signal': {
        'direction': str,         # 'BULLISH', 'BEARISH', or 'NEUTRAL'
        'strength': float,        # 0-100
        'confidence': float       # 0-100
    },
    'details': {                  # Plugin-specific details
        # Any plugin-specific data
    },
    'display_data': {            # Pre-formatted for UI
        'summary': str,          # One-line summary
        'description': str,      # Detailed description
        'table_data': [          # Data for display table
            [key, value],        # List of [label, value] pairs
            ...
        ],
        'chart_markers': [       # Optional chart annotations
            {...}
        ]
    }
}

6. Implementation Guide
6.1 Creating a New Plugin: Step-by-Step
Let's implement the M1 Market Structure plugin as a complete example:
Step 1: Create Plugin Directory
bashmkdir -p backtest/plugins/m1_market_structure
Step 2: Create Entry Point (__init__.py)
python# backtest/plugins/m1_market_structure/__init__.py
"""
M1 Market Structure Plugin
Complete self-contained plugin for 1-minute fractal-based market structure analysis.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from .plugin import M1MarketStructurePlugin

logger = logging.getLogger(__name__)

# Create plugin instance
_plugin = M1MarketStructurePlugin()

# Export the main interface
async def run_analysis(symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
    """
    Run M1 Market Structure analysis.
    
    This is the single entry point for the plugin.
    External systems only need to call this function.
    """
    try:
        # Validate inputs
        if not _plugin.validate_inputs(symbol, entry_time, direction):
            raise ValueError("Invalid input parameters")
        
        # Run analysis
        result = await _plugin.run_analysis(symbol, entry_time, direction)
        
        logger.info(f"M1 Market Structure analysis complete for {symbol} at {entry_time}")
        return result
        
    except Exception as e:
        logger.error(f"Error in M1 Market Structure analysis: {e}")
        # Return error result in standard format
        return {
            'plugin_name': _plugin.name,
            'timestamp': entry_time,
            'error': str(e),
            'signal': {
                'direction': 'NEUTRAL',
                'strength': 0,
                'confidence': 0
            }
        }

# Export metadata for plugin discovery
PLUGIN_NAME = _plugin.name
PLUGIN_VERSION = _plugin.version
Step 3: Implement Plugin Logic (plugin.py)
python# backtest/plugins/m1_market_structure/plugin.py
"""
M1 Market Structure Plugin Implementation

This file contains the complete implementation of the M1 Market Structure plugin.
It is self-contained and handles:
1. Data fetching
2. Market structure analysis
3. Result formatting
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import pandas as pd
import logging

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backtest.plugins.base_plugin import BacktestPlugin
from modules.calculations.market_structure.m1_market_structure import (
    MarketStructureAnalyzer, 
    MarketStructureSignal
)
from data.polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)


class M1MarketStructurePlugin(BacktestPlugin):
    """
    Plugin for 1-minute market structure analysis using fractals.
    
    This plugin:
    1. Fetches 1-minute bar data
    2. Detects fractals (swing highs/lows)
    3. Identifies structure breaks (BOS/CHoCH)
    4. Returns formatted signals for display
    """
    
    def __init__(self):
        """Initialize plugin with configuration"""
        self.data_manager = PolygonDataManager()
        self.config = {
            'fractal_length': 5,          # Bars on each side for fractal
            'buffer_size': 200,           # Number of candles to analyze
            'min_candles_required': 21,   # Minimum for valid signal
            'lookback_minutes': 200       # Historical data to fetch
        }
        
    @property
    def name(self) -> str:
        return "1-Min Market Structure"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def get_config(self) -> Dict[str, Any]:
        """Get plugin configuration for UI settings"""
        return self.config.copy()
    
    def validate_inputs(self, symbol: str, entry_time: datetime, direction: str) -> bool:
        """
        Validate input parameters.
        
        Args:
            symbol: Must be non-empty string
            entry_time: Must be datetime object
            direction: Must be 'LONG' or 'SHORT'
            
        Returns:
            True if all inputs valid, False otherwise
        """
        if not symbol or not isinstance(symbol, str):
            logger.error(f"Invalid symbol: {symbol}")
            return False
        
        if not isinstance(entry_time, datetime):
            logger.error(f"Invalid entry_time: {entry_time}")
            return False
            
        if direction not in ['LONG', 'SHORT']:
            logger.error(f"Invalid direction: {direction}")
            return False
            
        return True
    
    async def run_analysis(self, symbol: str, entry_time: datetime, direction: str) -> Dict[str, Any]:
        """
        Run the complete M1 market structure analysis.
        
        This is the main method that orchestrates:
        1. Data fetching
        2. Analysis
        3. Result formatting
        """
        try:
            # 1. Fetch required data
            logger.info(f"Fetching data for {symbol}")
            bars = await self._fetch_data(symbol, entry_time)
            
            if bars.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # 2. Run market structure analysis
            logger.info(f"Running market structure analysis")
            analyzer = MarketStructureAnalyzer(
                fractal_length=self.config['fractal_length'],
                buffer_size=self.config['buffer_size'],
                min_candles_required=self.config['min_candles_required']
            )
            
            # Process bars up to entry time
            signal = self._process_bars(analyzer, symbol, bars, entry_time)
            
            # 3. Format and return results
            logger.info(f"Formatting results")
            return self._format_results(signal, entry_time, direction)
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise
    
    async def _fetch_data(self, symbol: str, entry_time: datetime) -> pd.DataFrame:
        """
        Fetch required bar data from data source.
        
        This method handles all data fetching logic internally.
        No external data management needed.
        """
        # Calculate time range
        start_time = entry_time - timedelta(minutes=self.config['lookback_minutes'])
        
        logger.info(f"Fetching {symbol} data from {start_time} to {entry_time}")
        
        # Use PolygonDataManager to fetch data
        bars = await self.data_manager.load_bars(
            symbol=symbol,
            start_time=start_time,
            end_time=entry_time,
            timeframe='1min',
            use_cache=True  # Use caching for efficiency
        )
        
        logger.info(f"Fetched {len(bars)} bars")
        return bars
    
    def _process_bars(self, analyzer: MarketStructureAnalyzer, 
                     symbol: str, bars: pd.DataFrame, 
                     entry_time: datetime) -> Optional[MarketStructureSignal]:
        """
        Process bars through the market structure analyzer.
        
        Converts DataFrame to format expected by analyzer and
        processes all bars up to entry time.
        """
        candles = []
        
        # Convert bars to candle format expected by analyzer
        for timestamp, row in bars.iterrows():
            # Stop at entry time to prevent look-ahead bias
            if timestamp >= entry_time:
                break
                
            candle_dict = {
                't': timestamp,
                'o': float(row['open']),
                'h': float(row['high']),
                'l': float(row['low']),
                'c': float(row['close']),
                'v': float(row['volume'])
            }
            candles.append(candle_dict)
        
        logger.info(f"Processing {len(candles)} candles")
        
        # Process all historical candles
        signal = analyzer.process_historical_candles(symbol, candles)
        
        # If no signal from historical processing, get current state
        if not signal:
            signal = analyzer.get_current_analysis(symbol)
            
        return signal
    
    def _format_results(self, signal: Optional[MarketStructureSignal], 
                       entry_time: datetime, direction: str) -> Dict[str, Any]:
        """
        Format analysis results into standardized output format.
        
        This method transforms the raw calculation output into
        the standardized format expected by the UI.
        """
        if not signal:
            # No signal - return neutral result
            return self._create_neutral_result(entry_time)
        
        # Map signal direction to standard format
        direction_map = {
            'BULL': 'BULLISH',
            'BEAR': 'BEARISH',
            'NEUTRAL': 'NEUTRAL'
        }
        
        signal_direction = direction_map.get(signal.signal, 'NEUTRAL')
        metrics = signal.metrics
        
        # Build display table rows
        display_rows = []
        
        # Main signal information
        display_rows.append(['Structure Type', signal.structure_type or 'None'])
        display_rows.append(['Current Trend', metrics.get('current_trend', 'NEUTRAL')])
        display_rows.append(['Signal Strength', f"{signal.strength:.0f}%"])
        
        # Fractal information
        if metrics.get('last_high_fractal'):
            display_rows.append(['Last High Fractal', f"${metrics['last_high_fractal']:.2f}"])
        if metrics.get('last_low_fractal'):
            display_rows.append(['Last Low Fractal', f"${metrics['last_low_fractal']:.2f}"])
        
        # Structure break information
        if metrics.get('last_break_type'):
            display_rows.append(['Last Break', metrics['last_break_type']])
            if metrics.get('last_break_price'):
                display_rows.append(['Break Price', f"${metrics['last_break_price']:.2f}"])
        
        # Statistics
        display_rows.append(['Total Fractals', str(metrics.get('fractal_count', 0))])
        display_rows.append(['Structure Breaks', str(metrics.get('structure_breaks', 0))])
        display_rows.append(['Trend Changes', str(metrics.get('trend_changes', 0))])
        
        # Create summary text
        summary = f"{signal.structure_type or 'No Signal'}"
        if signal.structure_type:
            summary += f" - {signal_direction}"
            if signal.structure_type == 'CHoCH':
                summary += " (Trend Reversal)"
            elif signal.structure_type == 'BOS':
                summary += " (Trend Continuation)"
        
        # Return standardized format
        return {
            'plugin_name': self.name,
            'timestamp': entry_time,
            'signal': {
                'direction': signal_direction,
                'strength': float(signal.strength),
                'confidence': float(signal.strength)  # Using strength as confidence
            },
            'details': {
                'structure_type': signal.structure_type,
                'current_trend': metrics.get('current_trend'),
                'last_high_fractal': metrics.get('last_high_fractal'),
                'last_low_fractal': metrics.get('last_low_fractal'),
                'last_break_type': metrics.get('last_break_type'),
                'last_break_price': metrics.get('last_break_price'),
                'fractal_count': metrics.get('fractal_count', 0),
                'structure_breaks': metrics.get('structure_breaks', 0),
                'trend_changes': metrics.get('trend_changes', 0),
                'candles_processed': metrics.get('candles_processed', 0),
                'reason': signal.reason
            },
            'display_data': {
                'summary': summary,
                'description': signal.reason,
                'table_data': display_rows,
                'chart_markers': self._get_chart_markers(metrics)
            }
        }
    
    def _create_neutral_result(self, entry_time: datetime) -> Dict[str, Any]:
        """Create a neutral result when no signal is found"""
        return {
            'plugin_name': self.name,
            'timestamp': entry_time,
            'signal': {
                'direction': 'NEUTRAL',
                'strength': 50.0,
                'confidence': 50.0
            },
            'details': {
                'structure_type': None,
                'current_trend': 'NEUTRAL',
                'reason': 'No clear market structure signal'
            },
            'display_data': {
                'summary': 'No Signal',
                'description': 'No clear market structure detected',
                'table_data': [
                    ['Structure Type', 'None'],
                    ['Current Trend', 'NEUTRAL'],
                    ['Signal Strength', '50%']
                ]
            }
        }
    
    def _get_chart_markers(self, metrics: Dict) -> List[Dict]:
        """Generate chart markers for visualization"""
        markers = []
        
        # Add fractal markers
        if metrics.get('last_high_fractal'):
            markers.append({
                'type': 'high_fractal',
                'price': metrics['last_high_fractal'],
                'label': 'H'
            })
            
        if metrics.get('last_low_fractal'):
            markers.append({
                'type': 'low_fractal', 
                'price': metrics['last_low_fractal'],
                'label': 'L'
            })
            
        return markers
6.2 Key Implementation Patterns
Pattern 1: Self-Contained Data Fetching
pythonasync def _fetch_data(self, symbol: str, entry_time: datetime) -> pd.DataFrame:
    """Each plugin fetches its own data"""
    self.data_manager = PolygonDataManager()  # Own data manager
    return await self.data_manager.load_bars(...)  # Direct fetch
Pattern 2: Direct Calculation Usage
pythondef _process_bars(self, analyzer, symbol, bars, entry_time):
    """Direct use of calculation modules"""
    analyzer = MarketStructureAnalyzer(...)  # Create analyzer
    signal = analyzer.process_historical_candles(...)  # Direct processing
    return signal  # Return raw signal
Pattern 3: Clean Result Formatting
pythondef _format_results(self, signal, entry_time, direction):
    """Transform calculation output to standard format"""
    return {
        'plugin_name': self.name,
        'signal': {...},  # Standardized signal
        'details': {...},  # Raw details preserved
        'display_data': {...}  # Pre-formatted for UI
    }

7. Data Flow
7.1 Complete Data Flow Example
1. User Input
   - Symbol: "AAPL"
   - Entry Time: 2024-01-15 10:30:00 UTC
   - Direction: "LONG"

2. Plugin Execution
   dashboard.py → plugin_runner.py → m1_market_structure.__init__.py

3. Plugin Processing
   a. Validate inputs
   b. Fetch data (200 1-minute bars)
   c. Process through MarketStructureAnalyzer
   d. Format results

4. Result Return
   {
     'plugin_name': '1-Min Market Structure',
     'signal': {
       'direction': 'BULLISH',
       'strength': 75.0,
       'confidence': 75.0
     },
     'details': {
       'structure_type': 'BOS',
       'current_trend': 'BULL',
       ...
     },
     'display_data': {
       'summary': 'BOS - BULLISH (Trend Continuation)',
       'table_data': [...],
       ...
     }
   }

5. UI Display
   - Multi-result viewer shows summary row
   - Click row → detailed view shows full data
7.2 Error Handling Flow
pythontry:
    result = await plugin.run_analysis(...)
except Exception as e:
    # Return standardized error result
    return {
        'plugin_name': plugin_name,
        'timestamp': entry_time,
        'error': str(e),
        'signal': {
            'direction': 'NEUTRAL',
            'strength': 0,
            'confidence': 0
        }
    }

8. API Reference
8.1 Plugin Runner API
pythonclass PluginRunner:
    def get_available_plugins() -> List[str]:
        """Get list of discovered plugin names"""
        
    async def run_single_plugin(
        plugin_name: str, 
        symbol: str, 
        entry_time: datetime, 
        direction: str
    ) -> Dict[str, Any]:
        """Run a single plugin"""
        
    async def run_multiple_plugins(
        plugin_names: List[str], 
        symbol: str,
        entry_time: datetime, 
        direction: str
    ) -> List[Dict[str, Any]]:
        """Run multiple plugins in parallel"""
8.2 Result Viewer API
pythonclass MultiResultViewer:
    def add_result(result: Dict[str, Any]) -> None:
        """Add a plugin result to the table"""
        
    def clear_results() -> None:
        """Clear all results"""
        
    # Signal
    result_selected = pyqtSignal(dict)  # Emitted on row selection

9. Migration Guide
9.1 Converting Old Adapter-Based Calculations
Step 1: Identify Core Calculation Logic
Find the actual calculation class in your modules:
python# modules/calculations/your_calculation.py
class YourCalculation:
    def process_data(self, ...):
        # Core logic here
Step 2: Create Plugin Structure
bashmkdir -p backtest/plugins/your_calculation
touch backtest/plugins/your_calculation/__init__.py
touch backtest/plugins/your_calculation/plugin.py
Step 3: Move Logic to Plugin

Data fetching - Move from adapter to _fetch_data()
Processing - Call calculation directly in _process_bars()
Formatting - Move from adapter to _format_results()

Step 4: Remove Dependencies

No shared cache
No adapter base class
No signal aggregator
No storage in main flow

9.2 Common Migration Patterns
Old: Complex Data Flow
python# adapter.py
def feed_historical_data(self, data, symbol):
    self.data_cache = shared_cache
    aggregated = self.aggregator.aggregate(data)
    transformed = self.transform_data(aggregated)
    self.calculation.process(transformed)
New: Direct Processing
python# plugin.py
async def run_analysis(self, symbol, entry_time, direction):
    bars = await self._fetch_data(symbol, entry_time)
    signal = self._process_bars(analyzer, symbol, bars, entry_time)
    return self._format_results(signal, entry_time, direction)

10. Best Practices
10.1 Plugin Development

Keep It Simple

One plugin = one calculation type
Direct data flow
No unnecessary abstractions


Handle Errors Gracefully

Always return standard format
Include error in result
Log errors for debugging


Optimize Data Fetching

Use caching when available
Fetch only required data
Consider data limits


Format for Display

Pre-format in plugin
Include summary text
Provide table data



10.2 Code Organization
pythonclass YourPlugin(BacktestPlugin):
    def __init__(self):
        # Configuration only
        
    async def run_analysis(self, ...):
        # Main orchestration
        
    async def _fetch_data(self, ...):
        # Data fetching logic
        
    def _process_data(self, ...):
        # Calculation logic
        
    def _format_results(self, ...):
        # Result formatting
10.3 Testing Plugins
python# Test plugin independently
import asyncio
from datetime import datetime
from backtest.plugins.m1_market_structure import run_analysis

async def test():
    result = await run_analysis(
        'AAPL', 
        datetime.now(), 
        'LONG'
    )
    print(result)

asyncio.run(test())

11. Troubleshooting
11.1 Common Issues
Plugin Not Discovered

Check __init__.py exports run_analysis
Verify directory structure
Check for import errors

Import Errors
python# Add to top of plugin.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
Data Not Available

Check data source connection
Verify time ranges
Check symbol validity

11.2 Debug Mode
Add logging to trace execution:
pythonlogger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.debug(f"Fetching data for {symbol}")
logger.info(f"Processing {len(bars)} bars")
logger.error(f"Error: {e}")
11.3 Performance Optimization

Use Async Properly
python# Run plugins in parallel
results = await asyncio.gather(*tasks)

Cache Data
pythonbars = await self.data_manager.load_bars(
    use_cache=True  # Enable caching
)

Limit Data Window
pythonself.config = {
    'lookback_minutes': 200  # Only what's needed
}



Conclusion
The Modular Backtest System represents a fundamental shift in architecture design, prioritizing:

Simplicity over complexity
Independence over coupling
Direct communication over transformation layers
Self-containment over shared state

By following this specification, you can create clean, maintainable, and efficient backtesting plugins that are easy to understand, test, and extend. Each plugin is a complete unit that handles its own data, processing, and formatting - making the entire system more robust and easier to maintain.
The M1 Market Structure plugin serves as a complete reference implementation, demonstrating all the patterns and practices needed to build additional plugins for your backtesting needs.