I'll create a comprehensive README.md for your dashboard/ directory that covers the entire trading dashboard system.
markdown# Live Monitor Dashboard Module

## Overview

The `live_monitor/dashboard/` module is a sophisticated PyQt6-based trading dashboard that provides real-time market monitoring, technical analysis visualization, and trade management tools. Built with a modular architecture, it serves as a pure UI layer that receives data from the Polygon WebSocket connection and presents it through specialized widgets.

## Architecture
dashboard/
├── init.py                    # Module exports
├── main_dashboard.py              # Main window orchestrator
├── README.md                      # This documentation
└── components/                    # UI components
├── init.py               # Component exports
├── ticker_entry.py           # Symbol input widget
├── ticker_calculations.py    # Market data display
├── entry_calculations.py     # Position sizing calculator
├── point_call_entry.py       # Entry signals grid
├── point_call_exit.py        # Exit signals grid
├── hvn_table.py              # HVN zones table
├── supply_demand_table.py    # S/D zones table
├── order_blocks_table.py     # Order blocks table
└── zone_aggregator.py        # Zone overlap detection

## Main Dashboard

### LiveMonitorDashboard (`main_dashboard.py`)

The central orchestrator that manages all components and data flow.

**Key Features:**
- 3-column responsive layout with QSplitter
- Real-time Polygon data integration
- Automatic calculation scheduling (30-second intervals)
- Server connection status indicators
- Comprehensive error handling

**Layout Structure:**
┌─────────────────────────────────────────────────────────┐
│                    Header with Status                    │
├─────────────┬───────────────────────────────────────────┤
│             │        Point & Call Entry/Exit            │
│   Ticker    ├───────────────────────────────────────────┤
│   Entry     │                                           │
│   Calcs     │    HVN Table │ S/D Table │ Order Blocks  │
│             │                                           │
└─────────────┴───────────────────────────────────────────┘

**Data Flow:**
1. Ticker change → Data subscription → Market data updates
2. Bar accumulation → Calculation engine → Table updates
3. Signal generation → Entry/Exit grids → Position calculator

## Component Reference

### 1. TickerEntry (`ticker_entry.py`)

Symbol input and management widget.

**Features:**
- Auto-uppercase conversion
- Enter key submission
- Visual feedback on submit
- Current symbol display

**Signals:**
```python
ticker_changed = pyqtSignal(str)  # Emitted on symbol change
Usage:
pythonticker_entry = TickerEntry()
ticker_entry.ticker_changed.connect(on_symbol_change)
ticker_entry.set_ticker("AAPL")  # Programmatic setting
current = ticker_entry.get_current_ticker()
2. TickerCalculations (ticker_calculations.py)
Real-time market data display widget.
Display Sections:

Price Info: Last, Change, Bid/Ask, Spread
Volume Info: Current, Average, Ratio
Range Info: Day range, ATR, Position in range

Data Format:
python{
    'last_price': 150.25,
    'bid': 150.20,
    'ask': 150.30,
    'change': 2.50,
    'change_pct': 1.69,
    'volume': 1234567,
    'avg_volume': 1000000,
    'day_high': 151.00,
    'day_low': 149.00,
    'atr': 2.15
}
Dynamic Styling:

Green text for positive changes
Red text for negative changes
Automatic number formatting

3. EntryCalculations (entry_calculations.py)
Position sizing and risk management calculator.
Input Fields:

Account Size
Risk Percentage
Entry Price
Stop Loss

Calculated Outputs:

Position Size (dollar amount)
Shares/Contracts
Risk Amount
Risk/Reward ratio

Signals:
pythoncalculation_complete = pyqtSignal(dict)  # Results dictionary
Risk Level Indicators:

Low (≤1%): Green
Medium (1-2%): Orange
High (>2%): Red

4. PointCallEntry (point_call_entry.py)
Entry signal management grid.
Table Columns:

Time
Type (LONG/SHORT)
Price
Signal description
Strength (Strong/Medium/Weak)
Notes

Features:

Row selection emits signal details
Color-coded signal types
Strength indicators

Adding Signals:
pythonpoint_call_entry.add_entry_signal(
    time="09:30:15",
    signal_type="LONG",
    price="150.25",
    signal="HVN Break",
    strength="Strong",
    notes="Volume confirmation"
)
5. PointCallExit (point_call_exit.py)
Exit signal management grid.
Table Columns:

Time
Type (TARGET/STOP/TRAIL)
Price
P&L percentage
Signal description
Urgency (Urgent/Warning/Normal)

Visual Indicators:

P&L coloring (green/red)
Urgency highlighting
Exit type badges

6. HVNTableWidget (hvn_table.py)
High Volume Node zones display.
Features:

Real-time zone updates from HVN engine
Distance to current price calculation
M15 ATR proximity highlighting
Strength percentage indicators

Table Columns:

Zone #
Price High/Low
Strength %
Within M15 ATR

Zone Highlighting:

Strong (≥80%): Green
Medium (50-80%): Orange
Weak (<50%): Red

7. SupplyDemandTableWidget (supply_demand_table.py)
User-defined supply and demand zones.
Features:

Add/Refresh zone buttons
Supply (red) and Demand (green) coloring
ATR proximity detection
Database integration ready

Signals:
pythonzone_clicked = pyqtSignal(dict)
add_zone_requested = pyqtSignal()
refresh_requested = pyqtSignal()
8. OrderBlocksTableWidget (order_blocks_table.py)
Smart money order blocks display.
Features:

Bullish/Bearish block detection
Breaker block identification
Status tracking (Valid/Broken)
Row highlighting for nearby blocks

Visual Coding:

Bullish blocks: Green
Bearish blocks: Red
Broken blocks: Orange

9. ZoneAggregator (zone_aggregator.py)
Intelligent zone overlap detection and merging.
Features:

Configurable overlap threshold
Multi-source zone combination
Unified zone representation
Priority-based merging

Usage:
pythonaggregator = ZoneAggregator(overlap_threshold=0.1)
unified_zones = aggregator.aggregate_zones(
    hvn_result=hvn_data,
    supply_demand_result=sd_data
)
nearby = aggregator.get_zones_near_price(zones, current_price, 0.03)
Integration Examples
Basic Setup
pythonfrom live_monitor.dashboard import LiveMonitorDashboard
from PyQt6.QtWidgets import QApplication

app = QApplication([])
dashboard = LiveMonitorDashboard()
dashboard.show()
app.exec()
Custom Data Integration
pythonclass CustomDashboard(LiveMonitorDashboard):
    def __init__(self):
        super().__init__()
        # Add custom data source
        self.custom_data_source = MyDataSource()
        self.custom_data_source.data_ready.connect(self.process_custom_data)
    
    def process_custom_data(self, data):
        # Update specific widgets
        self.ticker_calculations.update_calculations({
            'last_price': data['price'],
            'volume': data['volume']
        })
Adding Custom Signals
python# Generate entry signal
def generate_entry_signal(self, indicator_data):
    if indicator_data['rsi'] < 30:
        self.point_call_entry.add_entry_signal(
            time=datetime.now().strftime("%H:%M:%S"),
            signal_type="LONG",
            price=str(indicator_data['price']),
            signal="RSI Oversold",
            strength="Strong",
            notes=f"RSI: {indicator_data['rsi']:.1f}"
        )
Zone Management
python# Update all zone tables with calculated data
def update_zones(self):
    current_price = self.get_current_price()
    m15_atr = self.calculate_m15_atr()
    
    # Update HVN zones
    hvn_zones = self.hvn_engine.get_zones()
    self.hvn_table.update_hvn_zones(hvn_zones, current_price, m15_atr)
    
    # Update order blocks
    order_blocks = self.order_block_analyzer.get_blocks()
    self.order_blocks_table.update_order_blocks(
        order_blocks, current_price, m15_atr
    )
Signal Flow Diagram
User Input (Ticker) → LiveMonitorDashboard
                            ↓
                    PolygonDataManager
                            ↓
                    WebSocket Data Stream
                            ↓
                    Bar Accumulation
                            ↓
                 ┌──────────┴──────────┐
                 ↓                     ↓
          Market Data Updates    Calculation Timer
                 ↓                     ↓
          TickerCalculations    HVN/OrderBlock Analysis
                                      ↓
                               Zone Table Updates
                                      ↓
                              Signal Generation
                                      ↓
                          Entry/Exit Signal Grids
                                      ↓
                          Position Calculator
Calculation Integration
HVN Engine Integration
python# In main_dashboard.py
self.hvn_engine = HVNEngine(
    levels=100,                    # Price levels for analysis
    percentile_threshold=80.0,     # Strength threshold
    proximity_atr_minutes=30       # ATR period for proximity
)

# Analysis execution
hvn_result = self.hvn_engine.analyze(df, include_pre=True, include_post=True)
Order Block Analysis
pythonself.order_block_analyzer = OrderBlockAnalyzer(
    swing_length=7,    # Bars for swing detection
    show_bullish=3,    # Max bullish blocks
    show_bearish=3     # Max bearish blocks
)

# Analysis execution
order_blocks = self.order_block_analyzer.analyze_zones(df)
Status Management
Connection Status
The dashboard provides multiple connection status indicators:

Header Widget: Visual server connection indicator
Status Bar: Text-based connection status
Window Title: Connection state in title

Error Handling
pythondef _on_data_error(self, error_msg):
    """Comprehensive error handling"""
    logger.error(f"Data error: {error_msg}")
    self.status_bar.showMessage(f"Error: {error_msg}", 5000)
    # Additional error recovery logic
Performance Optimization
Data Management

Bar accumulation limited to 2000 bars
30-second calculation intervals
Efficient DataFrame operations

UI Updates

Conditional widget updates
Cached style sheets
Minimal redraws

Extending the Dashboard
Adding New Components
python# Create new component
class CustomIndicatorWidget(QWidget):
    indicator_triggered = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def update_indicator(self, data):
        # Process and display
        pass

# Add to dashboard
self.custom_indicator = CustomIndicatorWidget()
left_layout.addWidget(self.custom_indicator)
Custom Calculations
pythonclass ExtendedDashboard(LiveMonitorDashboard):
    def run_calculations(self):
        # Call parent calculations
        super().run_calculations()
        
        # Add custom calculations
        if self.accumulated_data:
            df = pd.DataFrame(self.accumulated_data)
            custom_result = self.custom_analyzer.analyze(df)
            self.update_custom_display(custom_result)
Configuration
Layout Proportions
python# Horizontal split (25% left, 75% right)
main_splitter.setSizes([400, 1200])

# Vertical split (40% top, 60% bottom)
vertical_splitter.setSizes([360, 540])
Update Intervals
python# Calculation timer (milliseconds)
self.calculation_timer.start(30000)  # 30 seconds

# Data accumulation limits
MAX_BARS = 2000
Troubleshooting
Common Issues

No Data Display

Check WebSocket connection
Verify symbol is valid
Ensure sufficient data accumulated


Calculation Errors

Check data integrity
Verify DataFrame index
Review calculation logs


UI Freezing

Check calculation complexity
Verify timer intervals
Monitor memory usage



Debug Mode
python# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

# Add debug outputs
logger.debug(f"Accumulated bars: {len(self.accumulated_data)}")
logger.debug(f"Current price: {current_price}")
Best Practices

Signal Handling: Always use Qt signals for component communication
Data Validation: Validate data before display updates
Error Recovery: Implement graceful error handling
Memory Management: Clear old data regularly
UI Responsiveness: Use timers for heavy calculations

Future Enhancements

Chart widget integration (matplotlib/plotly)
Trade execution integration
Performance analytics
Multi-symbol monitoring
Alert system
Theme customization
Layout persistence


This comprehensive README provides complete documentation for the dashboard module, including architecture, component details, integration examples, and best practices for extending and maintaining the trading dashboard.