# backtest/plugins/__init__.py
"""
Unified plugin system with module-level functions
All plugins share a single data manager instance with validation checks
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Callable, List, Tuple
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import required modules
from backtest.data import PolygonDataManager, ProtectedDataManager, CircuitBreaker

logger = logging.getLogger(__name__)

# ============================================================================
# SHARED DATA MANAGER SETUP
# ============================================================================

# Initialize single shared data manager instance
_data_manager = None
_use_circuit_breaker = True

def initialize_data_manager(
    api_key: Optional[str] = None,
    use_circuit_breaker: bool = True,
    cache_dir: str = './cache',
    memory_cache_size: int = 200,
    file_cache_hours: int = 24,
    extend_window_bars: int = 2000
) -> None:
    """Initialize the shared data manager instance"""
    global _data_manager, _use_circuit_breaker
    
    # Get API key
    if not api_key:
        api_key = os.environ.get('POLYGON_API_KEY')
    
    if not api_key:
        raise ValueError("Polygon API key not provided")
    
    # Create base Polygon manager
    polygon_manager = PolygonDataManager(
        api_key=api_key,
        cache_dir=cache_dir,
        memory_cache_size=memory_cache_size,
        file_cache_hours=file_cache_hours,
        extend_window_bars=extend_window_bars
    )
    
    # Wrap with circuit breaker if requested
    if use_circuit_breaker:
        _data_manager = ProtectedDataManager(
            polygon_manager,
            circuit_breaker_config={
                'failure_threshold': 0.5,
                'consecutive_failures': 5,
                'recovery_timeout': 60,
                'rate_limits': {
                    'bars': {'per_minute': 100, 'burst': 10},
                    'trades': {'per_minute': 50, 'burst': 5},
                    'quotes': {'per_minute': 50, 'burst': 5}
                }
            }
        )
    else:
        _data_manager = polygon_manager
    
    _use_circuit_breaker = use_circuit_breaker
    logger.info("Shared data manager initialized")

def get_data_manager() -> Any:
    """Get the shared data manager instance"""
    if _data_manager is None:
        initialize_data_manager()
    return _data_manager

# ============================================================================
# DATA VALIDATION AND CHECKING
# ============================================================================

async def validate_data_availability(
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    required_data_types: List[str]
) -> Dict[str, Any]:
    """
    Check data availability before making requests
    This is the check module that runs before calculations
    """
    validation_results = {
        'symbol': symbol,
        'start_time': start_time,
        'end_time': end_time,
        'data_available': {},
        'warnings': [],
        'errors': [],
        'can_proceed': True
    }
    
    data_manager = get_data_manager()
    
    # Check each required data type
    for data_type in required_data_types:
        try:
            if data_type == 'bars_1min':
                # Check 1-minute bars availability
                sample_bars = await data_manager.load_bars(
                    symbol, start_time, start_time + timedelta(minutes=5), '1min'
                )
                available = not sample_bars.empty
                validation_results['data_available']['bars_1min'] = available
                
            elif data_type == 'bars_5min':
                # Check 5-minute bars availability
                sample_bars = await data_manager.load_bars(
                    symbol, start_time, start_time + timedelta(minutes=15), '5min'
                )
                available = not sample_bars.empty
                validation_results['data_available']['bars_5min'] = available
                
            elif data_type == 'bars_15min':
                # Check 15-minute bars availability
                sample_bars = await data_manager.load_bars(
                    symbol, start_time, start_time + timedelta(minutes=30), '15min'
                )
                available = not sample_bars.empty
                validation_results['data_available']['bars_15min'] = available
                
            elif data_type == 'trades':
                # Check trades availability
                sample_trades = await data_manager.load_trades(
                    symbol, start_time, start_time + timedelta(minutes=5)
                )
                available = not sample_trades.empty
                validation_results['data_available']['trades'] = available
                
            elif data_type == 'quotes':
                # Check quotes availability
                sample_quotes = await data_manager.load_quotes(
                    symbol, start_time, start_time + timedelta(minutes=5)
                )
                available = not sample_quotes.empty
                validation_results['data_available']['quotes'] = available
                
        except Exception as e:
            validation_results['errors'].append(f"Error checking {data_type}: {str(e)}")
            validation_results['data_available'][data_type] = False
            validation_results['can_proceed'] = False
    
    # Generate warnings
    for data_type, available in validation_results['data_available'].items():
        if not available:
            validation_results['warnings'].append(f"No {data_type} data available")
    
    return validation_results

# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================

def create_error_signal(plugin_name: str, entry_time: datetime, 
                       error_msg: str) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        'plugin_name': plugin_name,
        'timestamp': entry_time,
        'error': error_msg,
        'signal': {
            'direction': 'NEUTRAL',
            'strength': 0,
            'confidence': 0
        },
        'details': {'error': error_msg},
        'display_data': {}
    }

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ============================================================================
# PLUGIN: 1-Min EMA Crossover
# ============================================================================

def get_1min_ema_crossover_config() -> Dict[str, Any]:
    """Get configuration for 1-Min EMA Crossover plugin"""
    return {
        'ema_short': 9,
        'ema_long': 21,
        'lookback_minutes': 120,
        'min_bars_required': 50
    }

def validate_1min_ema_crossover_inputs(symbol: str, entry_time: datetime, 
                                      direction: str) -> bool:
    """Validate inputs for 1-Min EMA Crossover plugin"""
    if not symbol or not isinstance(symbol, str):
        return False
    if not isinstance(entry_time, datetime):
        return False
    if direction not in ['LONG', 'SHORT']:
        return False
    return True

async def run_1min_ema_crossover_analysis(
    symbol: str,
    entry_time: datetime,
    direction: str
) -> Dict[str, Any]:
    """Run 1-Min EMA Crossover analysis"""
    plugin_name = "1-Min EMA Crossover"
    
    # Validate inputs
    if not validate_1min_ema_crossover_inputs(symbol, entry_time, direction):
        return create_error_signal(plugin_name, entry_time, "Invalid inputs")
    
    try:
        # Get configuration
        config = get_1min_ema_crossover_config()
        
        # Set current plugin on data manager
        data_manager = get_data_manager()
        data_manager.set_current_plugin(plugin_name)
        
        # Check data availability
        validation = await validate_data_availability(
            symbol, 
            entry_time - timedelta(minutes=config['lookback_minutes']),
            entry_time,
            ['bars_1min']
        )
        
        if not validation['can_proceed']:
            return create_error_signal(plugin_name, entry_time, 
                                     "Data validation failed: " + ", ".join(validation['errors']))
        
        # Fetch data
        start_time = entry_time - timedelta(minutes=config['lookback_minutes'])
        bars = await data_manager.load_bars(symbol, start_time, entry_time, '1min')
        
        if bars.empty or len(bars) < config['min_bars_required']:
            return create_error_signal(plugin_name, entry_time, 
                                     f"Insufficient data: {len(bars)} bars")
        
        # Calculate EMAs
        ema_short = calculate_ema(bars['close'], config['ema_short'])
        ema_long = calculate_ema(bars['close'], config['ema_long'])
        
        # Get current and previous values
        current_short = ema_short.iloc[-1]
        current_long = ema_long.iloc[-1]
        prev_short = ema_short.iloc[-2]
        prev_long = ema_long.iloc[-2]
        
        # Detect crossover
        bullish_cross = prev_short <= prev_long and current_short > current_long
        bearish_cross = prev_short >= prev_long and current_short < current_long
        
        # Calculate signal strength
        ema_diff = current_short - current_long
        ema_diff_pct = (ema_diff / current_long) * 100
        strength = min(100, abs(ema_diff_pct) * 10)
        
        # Determine signal direction
        if direction == 'LONG':
            if bullish_cross:
                signal_direction = 'BULLISH'
                confidence = 80
            elif current_short > current_long:
                signal_direction = 'BULLISH'
                confidence = 60
            else:
                signal_direction = 'NEUTRAL'
                confidence = 30
        else:  # SHORT
            if bearish_cross:
                signal_direction = 'BEARISH'
                confidence = 80
            elif current_short < current_long:
                signal_direction = 'BEARISH'
                confidence = 60
            else:
                signal_direction = 'NEUTRAL'
                confidence = 30
        
        # Build response
        return {
            'plugin_name': plugin_name,
            'timestamp': entry_time,
            'signal': {
                'direction': signal_direction,
                'strength': strength,
                'confidence': confidence
            },
            'details': {
                'ema_short': float(current_short),
                'ema_long': float(current_long),
                'ema_diff': float(ema_diff),
                'ema_diff_pct': float(ema_diff_pct),
                'bullish_cross': bullish_cross,
                'bearish_cross': bearish_cross,
                'bars_analyzed': len(bars)
            },
            'display_data': {
                'chart_data': {
                    'timestamps': bars.index.tolist()[-50:],
                    'prices': bars['close'].tolist()[-50:],
                    'ema_short': ema_short.tolist()[-50:],
                    'ema_long': ema_long.tolist()[-50:]
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error in {plugin_name}: {str(e)}")
        return create_error_signal(plugin_name, entry_time, str(e))

async def run_1min_ema_crossover_analysis_with_progress(
    symbol: str,
    entry_time: datetime,
    direction: str,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """Run 1-Min EMA Crossover analysis with progress tracking"""
    if progress_callback:
        progress_callback(0, "Starting 1-Min EMA Crossover analysis...")
    
    if progress_callback:
        progress_callback(25, "Validating data availability...")
    
    result = await run_1min_ema_crossover_analysis(symbol, entry_time, direction)
    
    if progress_callback:
        progress_callback(100, "Analysis complete")
    
    return result

# ============================================================================
# PLUGIN: 5-Min EMA Crossover
# ============================================================================

def get_5min_ema_crossover_config() -> Dict[str, Any]:
    """Get configuration for 5-Min EMA Crossover plugin"""
    return {
        'ema_short': 9,
        'ema_long': 21,
        'lookback_minutes': 300,
        'min_bars_required': 30
    }

def validate_5min_ema_crossover_inputs(symbol: str, entry_time: datetime, 
                                      direction: str) -> bool:
    """Validate inputs for 5-Min EMA Crossover plugin"""
    return validate_1min_ema_crossover_inputs(symbol, entry_time, direction)

async def run_5min_ema_crossover_analysis(
    symbol: str,
    entry_time: datetime,
    direction: str
) -> Dict[str, Any]:
    """Run 5-Min EMA Crossover analysis"""
    plugin_name = "5-Min EMA Crossover"
    
    if not validate_5min_ema_crossover_inputs(symbol, entry_time, direction):
        return create_error_signal(plugin_name, entry_time, "Invalid inputs")
    
    try:
        config = get_5min_ema_crossover_config()
        data_manager = get_data_manager()
        data_manager.set_current_plugin(plugin_name)
        
        # Check data availability
        validation = await validate_data_availability(
            symbol,
            entry_time - timedelta(minutes=config['lookback_minutes']),
            entry_time,
            ['bars_5min']
        )
        
        if not validation['can_proceed']:
            return create_error_signal(plugin_name, entry_time,
                                     "Data validation failed: " + ", ".join(validation['errors']))
        
        # Fetch data
        start_time = entry_time - timedelta(minutes=config['lookback_minutes'])
        bars = await data_manager.load_bars(symbol, start_time, entry_time, '5min')
        
        if bars.empty or len(bars) < config['min_bars_required']:
            return create_error_signal(plugin_name, entry_time,
                                     f"Insufficient data: {len(bars)} bars")
        
        # Calculate EMAs
        ema_short = calculate_ema(bars['close'], config['ema_short'])
        ema_long = calculate_ema(bars['close'], config['ema_long'])
        
        # Analysis logic (similar to 1-min but for 5-min timeframe)
        current_short = ema_short.iloc[-1]
        current_long = ema_long.iloc[-1]
        prev_short = ema_short.iloc[-2]
        prev_long = ema_long.iloc[-2]
        
        bullish_cross = prev_short <= prev_long and current_short > current_long
        bearish_cross = prev_short >= prev_long and current_short < current_long
        
        ema_diff = current_short - current_long
        ema_diff_pct = (ema_diff / current_long) * 100
        strength = min(100, abs(ema_diff_pct) * 8)  # Slightly lower multiplier for 5-min
        
        # Signal determination
        if direction == 'LONG':
            if bullish_cross:
                signal_direction = 'BULLISH'
                confidence = 85
            elif current_short > current_long:
                signal_direction = 'BULLISH'
                confidence = 65
            else:
                signal_direction = 'NEUTRAL'
                confidence = 35
        else:
            if bearish_cross:
                signal_direction = 'BEARISH'
                confidence = 85
            elif current_short < current_long:
                signal_direction = 'BEARISH'
                confidence = 65
            else:
                signal_direction = 'NEUTRAL'
                confidence = 35
        
        return {
            'plugin_name': plugin_name,
            'timestamp': entry_time,
            'signal': {
                'direction': signal_direction,
                'strength': strength,
                'confidence': confidence
            },
            'details': {
                'ema_short': float(current_short),
                'ema_long': float(current_long),
                'ema_diff': float(ema_diff),
                'ema_diff_pct': float(ema_diff_pct),
                'bullish_cross': bullish_cross,
                'bearish_cross': bearish_cross,
                'bars_analyzed': len(bars)
            },
            'display_data': {
                'chart_data': {
                    'timestamps': bars.index.tolist()[-30:],
                    'prices': bars['close'].tolist()[-30:],
                    'ema_short': ema_short.tolist()[-30:],
                    'ema_long': ema_long.tolist()[-30:]
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error in {plugin_name}: {str(e)}")
        return create_error_signal(plugin_name, entry_time, str(e))

async def run_5min_ema_crossover_analysis_with_progress(
    symbol: str,
    entry_time: datetime,
    direction: str,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """Run 5-Min EMA Crossover analysis with progress tracking"""
    if progress_callback:
        progress_callback(0, "Starting 5-Min EMA Crossover analysis...")
    
    if progress_callback:
        progress_callback(25, "Validating data availability...")
    
    result = await run_5min_ema_crossover_analysis(symbol, entry_time, direction)
    
    if progress_callback:
        progress_callback(100, "Analysis complete")
    
    return result

# ============================================================================
# PLUGIN REGISTRY
# ============================================================================

# Map plugin names to their functions
PLUGIN_REGISTRY = {
    "1-Min EMA Crossover": {
        "run_analysis": run_1min_ema_crossover_analysis,
        "run_analysis_with_progress": run_1min_ema_crossover_analysis_with_progress,
        "get_config": get_1min_ema_crossover_config,
        "validate_inputs": validate_1min_ema_crossover_inputs
    },
    "5-Min EMA Crossover": {
        "run_analysis": run_5min_ema_crossover_analysis,
        "run_analysis_with_progress": run_5min_ema_crossover_analysis_with_progress,
        "get_config": get_5min_ema_crossover_config,
        "validate_inputs": validate_5min_ema_crossover_inputs
    }
}

def get_available_plugins() -> List[str]:
    """Get list of available plugin names"""
    return list(PLUGIN_REGISTRY.keys())

def get_plugin_functions(plugin_name: str) -> Dict[str, Callable]:
    """Get functions for a specific plugin"""
    if plugin_name not in PLUGIN_REGISTRY:
        raise ValueError(f"Unknown plugin: {plugin_name}")
    return PLUGIN_REGISTRY[plugin_name]

# ============================================================================
# BACKWARDS COMPATIBILITY WRAPPER (Optional)
# ============================================================================

class PluginWrapper:
    """
    Wrapper class for backwards compatibility with class-based plugin interface
    """
    def __init__(self, plugin_name: str):
        if plugin_name not in PLUGIN_REGISTRY:
            raise ValueError(f"Unknown plugin: {plugin_name}")
        
        self._plugin_name = plugin_name
        self._functions = PLUGIN_REGISTRY[plugin_name]
    
    @property
    def name(self) -> str:
        return self._plugin_name
    
    @property
    def version(self) -> str:
        return "2.0.0"  # Module-based version
    
    async def run_analysis(self, symbol: str, entry_time: datetime, 
                          direction: str) -> Dict[str, Any]:
        return await self._functions["run_analysis"](symbol, entry_time, direction)
    
    async def run_analysis_with_progress(self, symbol: str, entry_time: datetime,
                                       direction: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        return await self._functions["run_analysis_with_progress"](
            symbol, entry_time, direction, progress_callback
        )
    
    def get_config(self) -> Dict[str, Any]:
        return self._functions["get_config"]()
    
    def validate_inputs(self, symbol: str, entry_time: datetime, 
                       direction: str) -> bool:
        return self._functions["validate_inputs"](symbol, entry_time, direction)

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Initialize data manager on module import if API key is available
if os.environ.get('POLYGON_API_KEY'):
    try:
        initialize_data_manager()
        logger.info("Backtest plugins module initialized with shared data manager")
    except Exception as e:
        logger.warning(f"Could not initialize data manager on import: {e}")
else:
    logger.info("Backtest plugins module loaded (data manager not initialized - no API key)")

# Export main functions and constants
__all__ = [
    # Data manager functions
    'initialize_data_manager',
    'get_data_manager',
    'validate_data_availability',
    
    # Plugin functions
    'run_1min_ema_crossover_analysis',
    'run_1min_ema_crossover_analysis_with_progress',
    'run_5min_ema_crossover_analysis', 
    'run_5min_ema_crossover_analysis_with_progress',
    
    # Registry functions
    'get_available_plugins',
    'get_plugin_functions',
    
    # Utility functions
    'create_error_signal',
    
    # Constants
    'PLUGIN_REGISTRY',
    
    # Compatibility wrapper
    'PluginWrapper'
]