# backtest/plugins/m1_statistical_trend/plugin.py
"""
1-Minute Statistical Trend Plugin - Pure Passthrough Implementation
No calculations here - just data flow management
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any
import pandas as pd
import logging

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backtest.plugins.base_plugin import BacktestPlugin
from modules.calculations.trend.statistical_trend_1min import StatisticalTrend1Min, StatisticalSignal
from backtest.data.polygon_data_manager import PolygonDataManager

logger = logging.getLogger(__name__)


class M1StatisticalTrendPlugin(BacktestPlugin):
    """
    Plugin acts as pure interface between dashboard and calculation module.
    No calculations performed here.
    """
    
    def __init__(self):
        self.data_manager = PolygonDataManager()
        self.calculator = StatisticalTrend1Min(
            lookback_periods=10,  # Default 10 bars
            min_confidence=25.0   # Updated for new thresholds
        )
        # Configuration for the plugin
        self.config = {
            'lookback_periods': 10,
            'min_confidence': 25.0,  # Updated
            'data_minutes': 15  # How many minutes of data to fetch
        }
        
    @property
    def name(self) -> str:
        return "1-Min Statistical Trend"
    
    @property
    def version(self) -> str:
        return "2.0.0"
    
    def get_config(self) -> Dict[str, Any]:
        """Return plugin configuration (required by base class)"""
        return self.config.copy()
    
    def validate_inputs(self, symbol: str, entry_time: datetime, direction: str) -> bool:
        """Validate input parameters"""
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
        Pure passthrough:
        1. Fetch data
        2. Send to calculator
        3. Format response for dashboard
        """
        try:
            logger.info(f"Starting analysis for {symbol} at {entry_time}")
            
            # Use get_bars which is the main interface method
            # We need data BEFORE the entry time for analysis
            bars_df = self.data_manager.get_bars(
                symbol=symbol,
                entry_time=entry_time,
                lookback_hours=0.25,  # 15 minutes = 0.25 hours
                forward_bars=0,       # No forward bars needed for analysis
                timeframe='1min'
            )
            
            if bars_df is None or bars_df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            logger.info(f"Fetched {len(bars_df)} bars from {bars_df.index.min()} to {bars_df.index.max()}")
            
            # 2. Send to calculator - NO CALCULATIONS HERE
            signal = self.calculator.analyze(
                symbol=symbol,
                bars_df=bars_df,
                entry_time=entry_time
            )
            
            logger.info(f"Received signal: {signal.signal} with {signal.confidence:.1f}% confidence")
            
            # 3. Format for dashboard - just restructuring data
            return self._format_dashboard_response(signal, direction, entry_time)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            # Return error response
            return {
                'plugin_name': self.name,
                'timestamp': entry_time,
                'error': str(e),
                'signal': {
                    'direction': 'NEUTRAL',
                    'strength': 0,
                    'confidence': 0
                }
            }
    
    def _format_dashboard_response(self, signal: StatisticalSignal, 
                                  direction: str, entry_time: datetime) -> Dict[str, Any]:
        """
        Format calculator output for dashboard display.
        No calculations - just data restructuring.
        """
        # Check alignment - now with multiple levels
        alignment_map = {
            'LONG': ['STRONG BUY', 'BUY', 'WEAK BUY'],
            'SHORT': ['STRONG SELL', 'SELL', 'WEAK SELL']
        }
        aligned = signal.signal in alignment_map.get(direction, [])
        
        # Map signal to dashboard format with descriptive names
        signal_descriptions = {
            'STRONG BUY': 'STRONG BULLISH (60%+ confidence)',
            'BUY': 'VERY BULLISH (50%+ confidence)', 
            'WEAK BUY': 'BULLISH (25%+ confidence)',
            'STRONG SELL': 'STRONG BEARISH (60%+ confidence)',
            'SELL': 'VERY BEARISH (50%+ confidence)',
            'WEAK SELL': 'BEARISH (25%+ confidence)',
            'NEUTRAL': 'NEUTRAL (<25% confidence)'
        }
        
        # For dashboard compatibility
        signal_map = {
            'STRONG BUY': 'BULLISH',
            'BUY': 'BULLISH',
            'WEAK BUY': 'BULLISH',
            'STRONG SELL': 'BEARISH',
            'SELL': 'BEARISH',
            'WEAK SELL': 'BEARISH',
            'NEUTRAL': 'NEUTRAL'
        }
        
        # Build response structure
        return {
            'plugin_name': self.name,
            'timestamp': entry_time,
            'signal': {
                'direction': signal_map[signal.signal],
                'strength': float(signal.trend_strength * 100),  # Convert to percentage
                'confidence': float(signal.confidence)
            },
            'details': {
                'raw_signal': signal.signal,
                'signal_description': signal_descriptions.get(signal.signal, signal.signal),
                'p_value': float(signal.p_value),
                'volatility': float(signal.volatility),
                'aligned': aligned,
                'statistical_metrics': signal.metrics
            },
            'display_data': {
                'summary': f"{signal.signal} - {signal.confidence:.0f}% confidence",
                'description': self._build_description(signal, aligned, direction),
                'table_data': self._build_table_data(signal)
            }
        }
    
    def _build_description(self, signal: StatisticalSignal, aligned: bool, direction: str) -> str:
        """Build description string - no calculations"""
        # Get signal description
        signal_descriptions = {
            'STRONG BUY': 'Strong Bullish',
            'BUY': 'Very Bullish',
            'WEAK BUY': 'Bullish',
            'STRONG SELL': 'Strong Bearish',
            'SELL': 'Very Bearish',
            'WEAK SELL': 'Bearish',
            'NEUTRAL': 'Neutral'
        }
        
        desc = f"{signal_descriptions.get(signal.signal, signal.signal)} signal"
        
        if signal.p_value < 0.05:
            desc += " (statistically significant)"
        
        if aligned:
            desc += f" ✅ Aligned with {direction}"
        else:
            desc += f" ⚠️ Not aligned with {direction}"
            
        return desc
    
    def _build_table_data(self, signal: StatisticalSignal) -> list:
        """Build table data - just formatting existing data"""
        # Get signal description
        signal_descriptions = {
            'STRONG BUY': 'Strong Bullish',
            'BUY': 'Very Bullish',
            'WEAK BUY': 'Bullish',
            'STRONG SELL': 'Strong Bearish',
            'SELL': 'Very Bearish',
            'WEAK SELL': 'Bearish',
            'NEUTRAL': 'Neutral'
        }
        
        rows = [
            ['Signal', f"{signal.signal} ({signal_descriptions.get(signal.signal, '')})"],
            ['Confidence', f"{signal.confidence:.1f}%"],
            ['P-Value', f"{signal.p_value:.4f}"],
            ['Trend Strength', f"{signal.trend_strength:.2f}%"],
            ['Volatility', f"{signal.volatility:.2f}%"]
        ]
        
        # Add statistical test results
        if signal.metrics:
            if 'linear_regression' in signal.metrics:
                lr = signal.metrics['linear_regression']
                rows.append(['Linear Trend', f"{lr['slope']:.3f}%/min (R²={lr['r_squared']:.2f})"])
            
            if 'mann_kendall' in signal.metrics:
                mk = signal.metrics['mann_kendall']
                rows.append(['Mann-Kendall', f"{mk['trend']} (p={mk['p_value']:.3f})"])
            
            if 'momentum' in signal.metrics:
                mom = signal.metrics['momentum']
                rows.append(['Momentum', f"{mom['simple_momentum']:.2f}%"])
        
        return rows