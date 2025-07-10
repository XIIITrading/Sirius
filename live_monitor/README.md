# Live Monitor Dashboard - Technical Reference

## Overview

PyQt6-based trading dashboard with modular architecture. Pure UI layer with no business logic - designed for data passthrough only.

## Directory Structure

```
live_monitor/
├── styles/                   # Styling modules (dark theme)
│   ├── base_styles.py       # Core theme and color palette
│   ├── ticker_calcs.py      # Ticker calculation display styles
│   ├── entry_calcs.py       # Entry/position sizing styles
│   ├── point_call_entry.py  # Entry signal grid styles
│   ├── point_call_exit.py   # Exit signal grid styles
│   └── chart.py             # Chart component styles
└── dashboard/
    ├── main_dashboard.py    # Main window orchestrator
    └── components/          # UI components
        ├── ticker_entry.py           # Symbol input widget
        ├── ticker_calculations.py    # Market data display
        ├── entry_calculations.py     # Position sizing calculator
        ├── point_call_entry.py       # Entry signals grid
        ├── point_call_exit.py        # Exit signals grid
        └── chart_widget.py           # Chart container
```

## Layout Architecture

### Visual Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    Live Monitor Dashboard                     │
├───────────────┬─────────────────────────────────────────────┤
│               │  Point & Call Entry  │  Point & Call Exit    │
│ Ticker Entry  ├─────────────────────┴─────────────────────┤
│               │                                               │
├───────────────┤                                               │
│               │                                               │
│    Ticker     │                 Chart Widget                  │
│ Calculations  │            (Spans full width)                 │
│               │                                               │
├───────────────┤                                               │
│               │                                               │
│ Entry / Size  │                                               │
│ Calculations  │                                               │
│               │                                               │
└───────────────┴─────────────────────────────────────────────┘
```

### Layout Structure

- **Main Horizontal Split**: 25% (left) / 75% (middle-right)
- **Left Column**: Fixed at 25% width
    - Ticker Entry
    - Ticker Calculations
    - Entry/Size Calculations
- **Middle-Right Section**: 75% width with vertical split
    - **Top Section** (40% height): Point & Call Entry | Point & Call Exit
    - **Bottom Section** (60% height): Chart (full width)

## Component Architecture

### Main Dashboard (`main_dashboard.py`)

- **Class**: `LiveMonitorDashboard(QMainWindow)`
- **Layout**: Nested splitters (horizontal main, vertical for middle-right)
- **Responsibilities**: Component instantiation, signal routing only

### Components Overview

#### 1. TickerEntry (`ticker_entry.py`)

python

```python
# Signals
ticker_changed = pyqtSignal(str)  # Emits on symbol change

# Key Methods
get_ticker() -> str
set_ticker(ticker: str)
```

#### 2. TickerCalculations (`ticker_calculations.py`)

python

```python
# Methods
update_calculations(data: dict)  # Updates display fields
clear_calculations()

# Expected data keys:
# last_price, bid, ask, change, change_percent, volume, etc.
```

#### 3. EntryCalculations (`entry_calculations.py`)

python

```python
# Signals
calculation_complete = pyqtSignal(dict)  # Emits calculation results

# Methods
calculate_position()  # Triggered by button
update_entry_price(price: float)
clear_results()

# Input fields: account_size, risk_percent, entry_price, stop_loss
# Output data: position_size, shares, risk_amount, risk_reward
```

#### 4. PointCallEntry (`point_call_entry.py`)

python

```python
# Signals
entry_selected = pyqtSignal(dict)  # Emits when row selected

# Methods
add_entry_signal(time, signal_type, price, signal, strength, notes)
clear_signals()
update_data(data)  # Placeholder

# Table columns: Time, Type, Price, Signal, Strength, Notes
```

#### 5. PointCallExit (`point_call_exit.py`)

python

```python
# Signals
exit_selected = pyqtSignal(dict)  # Emits when row selected

# Methods
add_exit_signal(time, exit_type, price, pnl, signal, urgency)
clear_signals()
update_data(data)  # Placeholder

# Table columns: Time, Type, Price, P&L, Signal, Urgency
```

#### 6. ChartWidget (`chart_widget.py`)

python

```python
# Signals
timeframe_changed = pyqtSignal(str)
indicator_toggled = pyqtSignal(str, bool)

# Methods
update_chart_data(data: dict)  # Placeholder
add_entry_marker(price, time)  # Placeholder
add_exit_marker(price, time)   # Placeholder

# Controls: Timeframe selector, HVN/Order Blocks/Camarilla toggles
```

## Signal Flow

```
TickerEntry.ticker_changed → Dashboard._on_ticker_changed() → [Your data fetch]
                                                                      ↓
TickerCalculations.update_calculations() ← [Market data]
                                                                      ↓
PointCallEntry.add_entry_signal() ← [Entry signals]
PointCallExit.add_exit_signal() ← [Exit signals]
                                                                      ↓
PointCallEntry.entry_selected → EntryCalculations.update_entry_price()
                                                                      ↓
EntryCalculations.calculation_complete → [Position size data]
```

## Integration Points

### Data Input Methods

python

```python
# Update ticker calculations
dashboard.ticker_calculations.update_calculations({
    'last_price': 150.25,
    'bid': 150.20,
    'ask': 150.30,
    'change': 2.50,
    'change_percent': 1.69,
    'volume': 1234567
})

# Add entry signal
dashboard.point_call_entry.add_entry_signal(
    time="09:30:15",
    signal_type="LONG",  # or "SHORT"
    price="150.25",
    signal="HVN Break",
    strength="Strong",   # Strong/Medium/Weak
    notes="Volume confirmation"
)

# Add exit signal
dashboard.point_call_exit.add_exit_signal(
    time="10:30:15",
    exit_type="TARGET",  # TARGET/STOP/TRAIL
    price="153.50",
    pnl="+2.50%",
    signal="Target 1 Reached",
    urgency="Normal"     # Urgent/Warning/Normal
)
```

### Layout Access

python

```python
# Access splitters for dynamic resizing
main_splitter = dashboard.centralWidget().layout().itemAt(0).widget()
vertical_splitter = main_splitter.widget(1).layout().itemAt(0).widget()

# Adjust layout proportions
main_splitter.setSizes([300, 1300])  # 300px left, 1300px right
vertical_splitter.setSizes([300, 600])  # 300px top, 600px bottom
```

### Style System

- Color palette defined in `base_styles.py`
- Each component has dedicated style module
- Apply custom styling via `apply_styles()` method
- Dark theme by default

### Key Design Decisions

1. **No Business Logic**: Dashboard is pure UI
2. **Signal-Based Communication**: Components don't directly reference each other
3. **Modular Styles**: Each component's styling is isolated
4. **Placeholder Methods**: Ready for data integration
5. **PyQt6 Signals**: Type-safe inter-component communication
6. **Flexible Layout**: Nested splitters allow user-adjustable sizing

## Quick Start

python

```python
from live_monitor.dashboard import LiveMonitorDashboard
from PyQt6.QtWidgets import QApplication

app = QApplication([])
dashboard = LiveMonitorDashboard()
dashboard.show()
app.exec()
```

## Extending Components

Each component inherits from `QWidget` and can be extended:

python

```python
class CustomTickerCalcs(TickerCalculations):
    def update_calculations(self, data):
        super().update_calculations(data)
        # Add custom logic here
```

## Notes for Integration

- All components accept data via methods, not constructors
- Use PyQt signals for loose coupling
- Style changes should go in respective style modules
- Chart widget is a placeholder - replace with actual charting library
- All monetary values displayed with "$" prefix
- Tables use alternating row colors for readability
- Point & Call systems are displayed side-by-side for easy comparison
- Chart spans full width to maximize visual analysis space