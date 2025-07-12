# Live Monitor Dashboard - Complete Technical Reference

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Classes](#core-classes)
4. [Component System](#component-system)
5. [Data Flow](#data-flow)
6. [Signal System](#signal-system)
7. [Calculation Integration](#calculation-integration)
8. [Key Methods Reference](#key-methods-reference)
9. [Extension Guide](#extension-guide)
10. [Technical Details](#technical-details)

## System Overview

The Live Monitor Dashboard is a sophisticated PyQt6-based trading application that provides:
- Real-time market data visualization via Polygon WebSocket
- Technical analysis with HVN zones, Order Blocks, and EMA calculations
- Signal generation and interpretation for entry/exit decisions
- Position sizing and risk management tools
- Multi-timeframe analysis (M1, M5, M15)

### Key Features
- **Real-time Data**: WebSocket connection to Polygon for live market data
- **Zone Analysis**: HVN (High Volume Nodes), Supply/Demand zones, Order Blocks
- **Signal System**: Automated entry/exit signal generation with confidence levels
- **Risk Management**: Position sizing calculator with account/risk parameters
- **Modular Design**: Segment-based architecture for maintainability

## Architecture
live_monitor/dashboard/
├── main_dashboard.py          # Main application window
├── segments/                  # Functional segments
│   ├── calculations_segment.py    # All calculation logic
│   ├── data_handler_segment.py    # Data update callbacks
│   ├── ui_builder_segment.py      # UI construction
│   └── signal_display_segment.py  # Signal visualization
├── components/                # Reusable UI widgets
│   ├── ticker_entry.py       # Symbol input
│   ├── ticker_calculations.py # Market data display
│   ├── entry_calculations.py  # Position sizing
│   ├── point_call_entry.py   # Entry signals grid
│   ├── point_call_exit.py    # Exit signals grid
│   ├── hvn_table.py          # HVN zones table
│   ├── supply_demand_table.py # S/D zones table
│   ├── order_blocks_table.py  # Order blocks table
│   └── zone_aggregator.py    # Zone overlap utility
└── widgets/                  # Custom widgets
└── server_status.py      # Connection indicator

## Core Classes

### LiveMonitorDashboard

**Location**: `main_dashboard.py`

The main application window that orchestrates all components and data flow.

```python
class LiveMonitorDashboard(QMainWindow, UIBuilderSegment, DataHandlerSegment, 
                          CalculationsSegment, SignalDisplaySegment):
Key Attributes:

data_manager: PolygonDataManager instance for WebSocket data
signal_interpreter: Processes raw signals into standardized format
hvn_engine: HVN calculation engine
order_block_analyzer: Order block detection
m1_ema_calculator, m5_ema_calculator, m15_ema_calculator: EMA calculators
accumulated_data: List storing recent market bars (max 2000)
current_symbol: Currently monitored ticker symbol

Key Methods:

__init__(): Initializes all engines, UI, and connections
connect_signals(): Connects UI component signals
setup_data_connections(): Links data manager signals
setup_signal_connections(): Links signal interpreter
connect_to_polygon(): Establishes WebSocket connection
closeEvent(): Cleanup on window close

Initialization Flow:

Create calculation engines
Build UI (via UIBuilderSegment)
Connect all signals
Start WebSocket connection
Begin calculation timer (30-second intervals)

Segment Classes
UIBuilderSegment
Purpose: Constructs the entire UI layout
Key Methods:

init_ui(): Main UI initialization
_create_header(): Top header with server status
_create_content(): Main content area with splitters
_create_left_column(): Ticker entry, calculations, position sizing
_create_middle_right_section(): Signal grids and zone tables
setup_status_bar(): Status bar with connection and signal indicators

Layout Structure:
┌─────────────────────────────────────────────────────┐
│                  Header (40px)                       │
├─────────┬───────────────────────────────────────────┤
│         │          Point & Call Grids               │
│  Left   ├───────────────────────────────────────────┤
│  25%    │     HVN │ Supply/Demand │ Order Blocks   │
│         │              (3 tables)                    │
└─────────┴───────────────────────────────────────────┘
DataHandlerSegment
Purpose: Handles all data update callbacks from PolygonDataManager
Key Callback Methods:

_on_ticker_changed(ticker): Symbol change handling
_on_market_data_updated(data): Live price/volume updates
_on_chart_data_updated(data): Bar data accumulation
_on_entry_signal(signal_data): Entry signal processing
_on_exit_signal(signal_data): Exit signal processing
_on_connection_status_changed(is_connected): Connection state
_on_data_error(error_msg): Error handling

Data Accumulation Logic:

Stores up to 2000 bars in accumulated_data
Clears old data when switching symbols
Triggers calculations after 5-second delay on symbol change

CalculationsSegment
Purpose: Executes all technical analysis calculations
Key Methods:

run_calculations(): Main calculation orchestrator
_process_ema_calculations(df): Runs M1/M5/M15 EMA analysis
_process_zone_calculations(df, current_price, m15_atr): HVN & Order Blocks
calculate_m15_atr(df, period=14): Calculates 15-minute ATR

Calculation Flow:

Convert accumulated data to DataFrame
Calculate M15 ATR for proximity detection
Process EMA signals (M1, M5, M15)
Run HVN analysis (top 5 clusters)
Detect Order Blocks (3 bullish, 3 bearish)
Update all display tables

SignalDisplaySegment
Purpose: Updates signal displays in status bar
Key Method:

update_signal_display(signal_value, category, timeframe): Updates M1/M5/M15 labels

Signal Categories & Colors:

Strong Bullish (≥25): Dark Green (#26a69a)
Weak Bullish (>0): Light Green (#66bb6a)
Weak Bearish (>-25): Light Red (#ef5350)
Strong Bearish (≤-25): Dark Red (#d32f2f)

Component System
Core Components
TickerEntry
Purpose: Symbol input and submission
Signals:
pythonticker_changed = pyqtSignal(str)  # Emitted on valid symbol
Key Methods:

submit_ticker(): Validates and emits symbol
get_current_ticker(): Returns current symbol
set_ticker(ticker): Programmatic symbol setting

TickerCalculations
Purpose: Real-time market data display
Display Sections:

Price Info: Last, Change, Bid/Ask, Spread
Volume Info: Current, Average, Ratio
Range Info: Day range, ATR, Position %

Key Method:
pythonupdate_calculations(data: dict)  # Updates all displays
EntryCalculations
Purpose: Position sizing and risk management
Signals:
pythoncalculation_complete = pyqtSignal(dict)  # Emits calculation results
Calculations:

Position Size = (Account Size × Risk%) / (Entry - Stop Loss)
Risk Amount = Account Size × Risk%
Shares = Risk Amount / (Entry - Stop Loss)

PointCallEntry & PointCallExit
Purpose: Signal management grids
Signals:
pythonentry_selected = pyqtSignal(dict)  # Row selection
exit_selected = pyqtSignal(dict)   # Row selection
Entry Grid Columns: Time, Type, Price, Signal, Strength, Notes
Exit Grid Columns: Time, Type, Price, P&L, Signal, Urgency
Zone Display Tables
HVNTableWidget
Purpose: High Volume Node zones from technical analysis
Key Method:
pythonupdate_hvn_zones(zones: List[Dict], current_price: float, m15_atr: float)
Zone Format:
python{
    'price_high': 151.50,
    'price_low': 150.75,
    'center_price': 151.125,
    'strength': 85.5,
    'type': 'hvn'
}
SupplyDemandTableWidget
Purpose: User-defined supply/demand zones
Signals:
pythonzone_clicked = pyqtSignal(dict)
add_zone_requested = pyqtSignal()
refresh_requested = pyqtSignal()
OrderBlocksTableWidget
Purpose: Smart money order blocks
Block Format:
python{
    'block_type': 'bullish',  # or 'bearish'
    'top': 152.00,
    'bottom': 151.50,
    'center': 151.75,
    'is_breaker': False,
    'time': datetime
}
Data Flow
1. Symbol Change Flow
User Input → TickerEntry → ticker_changed signal
    ↓
LiveMonitorDashboard._on_ticker_changed()
    ↓
Clear existing data → data_manager.change_symbol()
    ↓
WebSocket subscription → Bar accumulation
    ↓
5-second delay → run_calculations()
2. Real-time Data Flow
Polygon WebSocket → PolygonDataManager
    ↓
market_data_updated signal → _on_market_data_updated()
    ↓
TickerCalculations.update_calculations()
    ↓
Status bar update
3. Calculation Flow
30-second timer → run_calculations()
    ↓
DataFrame conversion → EMA calculations
    ↓
HVN analysis → Order Block detection
    ↓
Table updates → Signal display updates
Signal System
Signal Interpreter Integration
The dashboard integrates with SignalInterpreter to process raw calculation results into standardized signals:
python# M1 EMA Processing
m1_result = self.m1_ema_calculator.calculate(df)
standard_signal = self.signal_interpreter.process_m1_ema(m1_result)
self.update_signal_display(
    standard_signal.value,
    standard_signal.category.value,
    'M1'
)
Signal Flow

Calculator produces raw result
SignalInterpreter standardizes the signal
Dashboard updates display with category and confidence
Entry/Exit signals generated based on thresholds

Calculation Integration
HVN Engine Configuration
pythonself.hvn_engine = HVNEngine(
    levels=100,                    # Price levels for volume analysis
    percentile_threshold=80.0,     # Minimum strength threshold
    proximity_atr_minutes=30       # ATR period for proximity
)
Order Block Analyzer
pythonself.order_block_analyzer = OrderBlockAnalyzer(
    swing_length=7,    # Bars for swing high/low detection
    show_bullish=3,    # Maximum bullish blocks to display
    show_bearish=3     # Maximum bearish blocks to display
)
EMA Calculators

M1 EMA: 1-minute exponential moving average analysis
M5 EMA: 5-minute EMA with resampling
M15 EMA: 15-minute EMA with confidence levels

Key Methods Reference
Data Management
Accumulating Bar Data
pythondef _on_chart_data_updated(self, data: dict):
    """Accumulates bars, maintains 2000 bar limit"""
    self.accumulated_data.extend(data['bars'])
    if len(self.accumulated_data) > 2000:
        self.accumulated_data = self.accumulated_data[-2000:]
Clearing on Symbol Change
pythondef _on_ticker_changed(self, ticker):
    """Clears all data and resets displays"""
    self.accumulated_data.clear()
    # Clear all tables and calculations
    # Resubscribe to new symbol
Calculation Execution
Main Calculation Method
pythondef run_calculations(self):
    """Orchestrates all calculations"""
    # Requires minimum 100 bars
    # Converts to DataFrame
    # Runs EMA and zone calculations
    # Updates all displays
ATR Calculation
pythondef calculate_m15_atr(self, df: pd.DataFrame, period: int = 14) -> float:
    """Resamples to 15-min bars and calculates ATR"""
    # Resample 1-min to 15-min
    # Calculate True Range
    # Return average of last 'period' bars
Signal Updates
Processing Entry Signals
pythondef _on_entry_signal(self, signal_data):
    """Adds entry signal to grid"""
    self.point_call_entry.add_entry_signal(
        time=signal_data.get('time'),
        signal_type=signal_data.get('signal_type'),
        price=signal_data.get('price'),
        signal=signal_data.get('signal'),
        strength=signal_data.get('strength'),
        notes=signal_data.get('notes')
    )
Extension Guide
Adding New Components

Create Component Class:

python# In components/new_component.py
class NewComponent(QWidget):
    # Define signals
    component_event = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.apply_styles()

Add to UI:

python# In ui_builder_segment.py
self.new_component = NewComponent()
layout.addWidget(self.new_component)

Connect Signals:

python# In main_dashboard.py
self.new_component.component_event.connect(self._on_component_event)
Adding New Calculations

Create Calculator:

python# In calculations/
class NewCalculator:
    def calculate(self, df: pd.DataFrame) -> CalculationResult:
        # Implement calculation logic
        pass

Initialize in Dashboard:

python# In main_dashboard.__init__()
self.new_calculator = NewCalculator()

Add to Calculation Flow:

python# In calculations_segment.py
def run_calculations(self):
    # Existing calculations...
    new_result = self.new_calculator.calculate(df)
    self._process_new_result(new_result)
Adding New Signals

Define Signal Data Format:

pythonsignal_data = {
    'time': '10:30:15',
    'type': 'NEW_SIGNAL',
    'price': 150.25,
    'confidence': 0.85,
    'metadata': {...}
}

Create Display Method:

pythondef display_new_signal(self, signal_data):
    # Update appropriate UI component
    pass

Connect to Data Flow:

python# Connect to appropriate signal source
self.data_manager.new_signal.connect(self.display_new_signal)
Technical Details
Threading Model

All UI updates occur on the main thread
WebSocket runs on separate thread in PolygonDataManager
Calculations triggered by QTimer on main thread

Memory Management

Bar data limited to 2000 entries
Tables clear on symbol change
Automatic garbage collection for old data

Error Handling

Try/except blocks in all calculation methods
Error messages displayed in status bar
Graceful degradation when data unavailable

Performance Considerations

30-second calculation interval prevents overload
Batch table updates for efficiency
Conditional widget updates based on data changes

Dependencies

PyQt6: UI framework
pandas: Data manipulation
numpy: Numerical calculations
PolygonDataManager: WebSocket data source
Calculation engines: Technical analysis

Configuration

Window size: 1600x900 pixels
Left column: 25% width
Top section: 40% height
Calculation interval: 30 seconds
Data retention: 2000 bars

Best Practices

Always validate data before display
Use signals for component communication
Maintain separation of concerns (UI/Logic/Data)
Clear old data when switching contexts
Handle None/invalid values gracefully
Log important events and errors
Test with market closed/open scenarios
Document new features and methods

Future Enhancements

Chart widget integration
Trade execution capabilities
Multi-symbol monitoring
Alert system
Performance analytics
Historical backtesting
Custom indicator framework
Keyboard shortcuts
Layout persistence
Theme customization


This comprehensive README provides a complete technical reference for your Live Monitor Dashboard, covering all aspects of the system architecture, key classes, methods, data flow, and extension guidelines. It should give the OPUS model in another conversation everything needed to understand and continue building on this tool.