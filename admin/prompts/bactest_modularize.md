Backtesting Dashboard Modularization Guide
Current State

Single file: modules/ui/dashboards/backtest_dashboard.py (1400+ lines)
Contains: UI components, data fetching, worker threads, chart widgets, calculations
Imports calculations from modules/calculations/ directory
Has 7 more metric calculations to be added
Working successfully but becoming unwieldy

Target Architecture
root/
├── backtest/                      # NEW LOCATION (root level)
│   ├── __init__.py
│   ├── dashboard.py              # Main UI (~300 lines)
│   ├── worker.py                 # Calculation worker (~500 lines)
│   ├── data_fetcher.py          # Enhanced Polygon fetcher (~300 lines)
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── chart.py             # CandlestickChart (~150 lines)
│   │   └── order_flow.py        # BacktestOrderFlowWidget (~100 lines)
│   └── utils/
│       ├── __init__.py
│       └── validators.py        # Market hours validation (~50 lines)
└── modules/
    ├── calculations/             # Existing calculation modules
    └── ui/
        └── dashboards/
            └── entry_dashboard.py # Keep as-is (imported by backtest)
Extraction Plan
1. Extract EnhancedDataFetcher → backtest/data_fetcher.py
Lines to extract: ~67-390 from original
python# Move class EnhancedDataFetcher(DataFetcher) and all its methods:
- __init__
- __aenter__ / __aexit__
- _fetch_trade_chunk
- _fetch_quote_chunk  
- fetch_trades_concurrent
- fetch_quotes_concurrent
- fetch_trades

# Update imports at top:
import sys
import os
# Add path setup to find modules/
vega_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if vega_root not in sys.path:
    sys.path.insert(0, vega_root)

from polygon import DataFetcher  # Now finds it from root
2. Extract CandlestickChart → backtest/widgets/chart.py
Lines to extract: ~392-530 from original
python# Move entire class CandlestickChart(pg.GraphicsLayoutWidget)
# No path modifications needed as pyqtgraph is external
3. Extract BacktestOrderFlowWidget → backtest/widgets/order_flow.py
Lines to extract: New class definition (shown in conversation)
python# Import setup needs path adjustment:
import sys
import os
vega_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if vega_root not in sys.path:
    sys.path.insert(0, vega_root)

from modules.ui.dashboards.entry_dashboard import OrderFlowWidget
4. Extract BacktestWorker → backtest/worker.py
Lines to extract: ~532-920 from original
python# Key modifications:
# 1. Import paths need adjustment for root-level backtest/
# 2. Import data_fetcher locally:
from .data_fetcher import EnhancedDataFetcher

# 3. Add extensible calculator initialization:
def init_calculators(self):
    """Initialize all calculation modules"""
    # Existing calculators
    self.calc_1min = StatisticalTrend1Min()
    self.calc_5min = StatisticalTrend5Min()
    self.calc_15min = StatisticalTrend15Min()
    self.trade_size_calc = TradeSizeDistribution()
    self.bid_ask_calc = BidAskImbalance()
    self.tick_flow_calc = TickFlowAnalyzer()
    self.volume_1min_calc = VolumeAnalysis1Min()
    self.market_context_calc = MarketContext()
    
    # Placeholder for 7 new calculators
    # self.rsi_divergence_calc = RSIDivergenceCalculator()
    # self.vwap_calc = VWAPCalculator()
    # self.delta_calc = DeltaCalculator()
    # self.footprint_calc = FootprintCalculator()
    # self.cum_delta_calc = CumulativeDeltaCalculator()
    # self.dom_calc = DOMImbalanceCalculator()
    # self.sweep_calc = SweepDetector()
5. Extract validators → backtest/utils/validators.py
New file with market hours validation logic from lines ~1150-1170
6. Main dashboard → backtest/dashboard.py
What remains: ~922-1400 minus extracted components
python# Key changes:
# 1. Update all imports to local modules:
from .worker import BacktestWorker
from .widgets.chart import CandlestickChart
from .widgets.order_flow import BacktestOrderFlowWidget
from .utils.validators import validate_market_hours

# 2. Path setup for modules/:
import sys
import os
vega_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if vega_root not in sys.path:
    sys.path.insert(0, vega_root)

# 3. Simplify init_ui() by removing validator logic
Implementation Steps

Create directory structure
bashmkdir -p backtest/widgets backtest/utils
touch backtest/__init__.py backtest/widgets/__init__.py backtest/utils/__init__.py

Extract in order (to maintain dependencies):

data_fetcher.py first
widgets/* next
validators.py
worker.py
dashboard.py last


Update imports in each file:

Root-level imports need path setup to find modules/
Local imports use relative imports (.module_name)


Test progressively:

Test data_fetcher standalone first
Test worker with mock UI signals
Test full integration


Update main entry point:
python# In main script or wherever dashboard is launched:
from backtest.dashboard import BacktestingDashboard


Key Patterns to Preserve

Signal/Slot connections - Keep all pyqtSignal definitions with their classes
Async patterns - Maintain async/await structure in data fetcher
Progress callbacks - Keep progress reporting through signals
Error handling - Preserve try/except blocks and error signals

Adding New Calculations
In backtest/worker.py, extend init_calculators() and run_backtest():
python# In init_calculators():
self.new_calc = NewCalculator()

# In run_backtest(), after existing calculations:
results['new_metric'] = self.new_calc.calculate(data)

# In dashboard.py, add widget:
self.widget_new = BacktestOrderFlowWidget("New Metric", "new_metric")

# In display_results():
if results.get('new_metric'):
    self.widget_new.update_new_metric_signal(results['new_metric'])
File Size Targets

dashboard.py: ~300 lines (UI setup only)
worker.py: ~500 lines (will grow with new calcs)
data_fetcher.py: ~300 lines
chart.py: ~150 lines
order_flow.py: ~100 lines (will grow with new update methods)

Notes

Preserve all functionality, just reorganize
Keep consistent import patterns throughout
Maintain PyQt6 signal/slot architecture
All timestamps remain in UTC
Keep performance optimizations (concurrent fetching, batch processing)