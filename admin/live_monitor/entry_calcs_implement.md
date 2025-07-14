Core System Files (Required)
1. Signal Processing Core
live_monitor/signals/signal_interpreter.py

Shows the 4-tier signal mapping system
Contains all process_X methods patterns
Defines StandardSignal dataclass

2. Dashboard Orchestration
live_monitor/dashboard/main_dashboard.py

Shows calculator initialization patterns
Active entry sources configuration
Signal connections setup

3. Calculation Execution
live_monitor/dashboard/segments/calculations_segment.py

Contains all _process_X method patterns
Shows data preparation (resampling, etc.)
Historical data handling

4. UI Components
live_monitor/dashboard/segments/ui_builder_segment.py
live_monitor/dashboard/components/point_call_entry.py

Status bar label setup
Entry signal display logic

5. Data Models
live_monitor/data/models/signals.py

EntrySignal and ExitSignal type definitions

Example Implementation Files (M5 Statistical Trend)
Include these files to show a complete integration example:
1. The Calculation Module
live_monitor/calculations/trend/statistical_trend_5min.py

Shows PositionSignal5Min dataclass
Demonstrates analyze() method pattern
4-tier signal generation

2. Historical Data Fetcher (if applicable)
live_monitor/data/hist_request/trend_fetchers.py

Shows data fetching configuration
Bar requirements

3. Integration Points
Show the specific sections in:

signal_interpreter.py - the process_m5_statistical_trend method
calculations_segment.py - the _process_m5_statistical_trend method
main_dashboard.py - initialization of statistical_trend_5min
ui_builder_segment.py - the M5 TREND label addition

Optional but Helpful Files
For Understanding Data Flow:
live_monitor/data/polygon_data_manager.py
live_monitor/dashboard/segments/data_handler_segment.py
For Understanding Styling:
live_monitor/styles/base_styles.py
Initial GitHub Message Template
markdown# New Calculation Integration Request

I need to integrate a new calculation into the Live Monitor Dashboard. Here are the relevant system files:

## Core System Files:
- [signal_interpreter.py](link) - Signal processing and standardization
- [main_dashboard.py](link) - Main dashboard initialization
- [calculations_segment.py](link) - Calculation execution patterns
- [ui_builder_segment.py](link) - UI component setup
- [point_call_entry.py](link) - Entry signal display
- [signals.py](link) - Data models

## Example Implementation (M5 Statistical Trend):
- [statistical_trend_5min.py](link) - The calculation module
- Integration in signal_interpreter.py: lines 497-579 (process_m5_statistical_trend)
- Integration in calculations_segment.py: lines 418-454 (_process_m5_statistical_trend)
- Dashboard initialization: main_dashboard.py lines 65, 127

## New Calculation to Integrate:
- [my_new_calculation.py](link) - The calculation I want to integrate

The calculation currently produces [describe current outputs]. 
I would like it to display as "[desired display name]" in the dashboard.

Please review and provide the complete integration following the 3-step process.
This gives the AI everything needed to understand the system architecture and provide a proper integration while maintaining consistency with existing patterns.

----

Comprehensive Integration Guide for Live Monitor Dashboard Calculations
System Architecture Overview (~1,500 tokens)
The Live Monitor Dashboard is a sophisticated real-time trading analysis system built on PyQt6 that integrates multiple technical analysis calculations into a unified signal framework. The system follows a strict architectural pattern designed for modularity, maintainability, and consistent signal generation.
Core System Components
1. Data Layer (live_monitor/data/)

PolygonDataManager: Central hub for WebSocket market data, managing real-time price/volume updates and bar accumulation
HistoricalFetchCoordinator: Orchestrates parallel fetching of historical data across multiple timeframes (M1, M5, M15)
Maintains up to 2000 bars of accumulated data for calculations

2. Calculation Layer (live_monitor/calculations/)

Indicators: EMA calculations for M1, M5, M15 timeframes
Volume: HVN (High Volume Node) analysis for identifying price levels with significant trading activity
Zones: Order Block detection for supply/demand zone identification
Trend: Statistical trend analyzers using volatility-adjusted strength metrics

Each calculator follows a standard pattern:

Accepts a pandas DataFrame with OHLCV data
Returns a typed result dataclass with standardized fields
Produces signals in one of four categories: BUY, WEAK BUY, SELL, WEAK SELL

3. Signal Interpretation Layer (live_monitor/signals/)

SignalInterpreter: The critical translation layer that converts raw calculation outputs to standardized signals
Maps all signals to a -50 to +50 scale with four categories:

Bullish (25 to 50): Strong positive signals (BUY)
Weak Bullish (0 to 24): Weak positive signals (WEAK BUY)
Weak Bearish (-24 to 0): Weak negative signals (WEAK SELL)
Bearish (-50 to -25): Strong negative signals (SELL)


Manages entry/exit signal generation based on signal strength and changes
Tracks previous signals to detect reversals and weakening trends

4. Dashboard Layer (live_monitor/dashboard/)

Main Dashboard: Central orchestrator inheriting from multiple segments
Segments:

UIBuilderSegment: Constructs the UI layout and components
CalculationsSegment: Manages all calculation execution and timing
DataHandlerSegment: Handles data callbacks from PolygonDataManager
SignalDisplaySegment: Updates signal displays in the status bar


Runs calculations every 30 seconds via QTimer
Manages symbol changes and data clearing

5. UI Components (live_monitor/dashboard/components/)

TickerEntry: Symbol input widget
TickerCalculations: Real-time market data display
EntryCalculations: Position sizing calculator
PointCallEntry: Entry signal grid showing all active signals
PointCallExit: Exit signal grid for position management
Zone tables (HVN, Supply/Demand, Order Blocks)

Data Flow Architecture

Market Data Flow:
Polygon WebSocket → PolygonDataManager → accumulated_data[] → DataFrame conversion

Calculation Flow:
DataFrame → Calculator.analyze() → Result Dataclass → SignalInterpreter.process_X() → StandardSignal

Signal Display Flow:
StandardSignal → update_signal_display() → Status Bar Labels
StandardSignal → _generate_entry_signal() → Point & Call Entry Grid

Historical Data Flow:
Symbol Change → HistoricalFetchCoordinator → Parallel fetches → Calculator-specific processing


Signal Standardization Protocol
All calculations must produce signals that fit the four-tier system:

BUY: Strong bullish signal (volatility-adjusted strength typically > 1.0)
WEAK BUY: Weak bullish signal (volatility-adjusted strength 0.4-1.0)
WEAK SELL: Weak bearish signal (volatility-adjusted strength 0.4-1.0)
SELL: Strong bearish signal (volatility-adjusted strength typically > 1.0)

The SignalInterpreter maps these to numerical values:

BUY → +25 to +50
WEAK BUY → 0 to +24
WEAK SELL → -24 to 0
SELL → -50 to -25

Confidence scores (0-100%) modify signal generation thresholds and display properties.
Key Integration Points

Active Entry Sources: Dashboard maintains a dictionary controlling which calculations generate entry signals
Signal Metadata: Each signal carries extensive metadata for debugging and advanced filtering
Source Identification: Each calculation has a unique SOURCE_KEY (e.g., 'M1_EMA', 'STATISTICAL_TREND_5M')
Display Consistency: All signals show in status bar with consistent coloring based on signal strength

Integration Process
Step 1: Calculation Compatibility Review
When integrating a new calculation, the AI must:

Request the full calculation module with the prompt:
"Please provide the complete calculation module code that you want to integrate into the Live Monitor Dashboard."

Analyze the output structure for compatibility:

Does it produce BUY/WEAK BUY/SELL/WEAK SELL signals?
If not, can the output be mapped to this structure?
What metrics determine signal strength?


If incompatible, provide specific modifications:
python# Example modification pattern
def _generate_signal(self, metric_value: float, threshold_strong: float, threshold_weak: float) -> str:
    """Generate 4-tier signal from calculation metric"""
    if abs(metric_value) >= threshold_strong:
        return 'BUY' if metric_value > 0 else 'SELL'
    elif abs(metric_value) >= threshold_weak:
        return 'WEAK BUY' if metric_value > 0 else 'WEAK SELL'
    else:
        # Handle edge cases - always assign to weak category
        return 'WEAK BUY' if metric_value > 0 else 'WEAK SELL'


Step 2: Integration Implementation
The AI must provide complete, copy-paste ready code for each component:

Calculation Modifications (if needed)
Main Dashboard Updates (imports, initialization, active sources)
Signal Interpreter Method (full process_X method)
Calculations Segment Method (full _process_X method)
UI Updates (status bar label if needed)
Point & Call Entry Updates (source identification)

Each code block must be complete with:

All necessary imports
Full method implementations
Proper error handling
Appropriate logging

Step 3: Review and Validation
The AI must validate:

Signal Mapping Accuracy:

Verify the -50 to +50 mapping preserves calculation intent
Ensure confidence calculations make sense
Check that metadata includes all relevant information


Integration Completeness:

List all files modified with line numbers
Confirm SOURCE_KEY consistency across all files
Verify display labels match UI conventions


Error Scenarios:

Insufficient data handling
Invalid calculation results
Symbol change behavior


Testing Checklist:
□ Calculation runs without errors
□ Signal appears in status bar
□ Entry signals appear in Point & Call grid
□ Historical data processing works
□ Active source toggle functions correctly
□ Logging provides useful debugging info


Example Integration Request
When requesting a new integration:
"I want to integrate the [CalculationName] calculation into the Live Monitor Dashboard.

The calculation is located at: [path/to/calculation.py]
It currently produces signals: [list current signal types]
I want it to display as: [display name in UI]

Please review the calculation and provide the complete integration code."
The AI will then follow the three-step process, ensuring the calculation fits the standardized signal framework while maintaining its analytical integrity.