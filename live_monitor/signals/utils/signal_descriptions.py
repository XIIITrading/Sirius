# live_monitor/signals/utils/signal_descriptions.py
"""
Utility functions for generating signal descriptions
"""

from typing import Optional


def generate_ema_description(source: str, direction: str, crossover_type: Optional[str] = None,
                           is_crossover: bool = False, signal_type: Optional[str] = None) -> str:
    """Generate EMA signal description"""
    timeframe = source.replace('_EMA', '')
    
    if signal_type == 'NEUTRAL':
        return f"{timeframe} EMA NEUTRAL (Price vs Trend Conflict)"
    elif is_crossover and crossover_type:
        return f"{timeframe} EMA {crossover_type.title()} Crossover"
    else:
        return f"{timeframe} EMA {direction} Signal"


def generate_trend_description(source: str, original_signal: str, vol_adj: float,
                             volume_confirmation: bool = False,
                             regime: Optional[str] = None, daily_bias: Optional[str] = None) -> str:
    """Generate statistical trend signal description"""
    if source == 'STATISTICAL_TREND':
        # M1 Trend
        if vol_adj >= 2.0:
            desc = f"M1 Trend STRONG {original_signal}"
        elif vol_adj >= 1.0:
            desc = f"M1 Trend {original_signal}"
        else:
            desc = f"M1 Trend WEAK {original_signal}"
        
        if volume_confirmation:
            desc += " (Vol Confirm)"
            
    elif source == 'STATISTICAL_TREND_5M':
        # M5 Trend
        desc = f"M5 Trend: {original_signal}"
        desc += f" (Vol-Adj: {vol_adj:.2f})"
        
        if volume_confirmation:
            desc += " +Vol"  # Changed from "✓Vol" to "+Vol"
            
    elif source == 'STATISTICAL_TREND_15M':
        # M15 Trend
        desc = f"M15 Trend: {original_signal}"
        
        if regime and regime in ['BULL MARKET', 'BEAR MARKET']:
            desc += f" ({regime})"
        
        if daily_bias and daily_bias in ['LONG ONLY', 'SHORT ONLY', 'STAY OUT']:
            desc += f" [{daily_bias}]"
    else:
        desc = f"{source} {original_signal}"
    
    return desc

def generate_market_structure_description(source: str, original_signal: str, 
                                        structure_type: str, direction: str) -> str:
    """Generate market structure signal description"""
    if source == 'M1_MARKET_STRUCTURE':
        desc = f"M1 MS: {structure_type}"
        if structure_type == 'CHoCH':
            desc += f" {direction} Reversal"
        else:  # BOS
            desc += f" {direction} Continuation"
        desc += f" ({original_signal})"
    elif source == 'M5_MARKET_STRUCTURE':  # ADD THIS
        desc = f"M5 MS: {structure_type}"
        if structure_type == 'CHoCH':
            desc += f" {direction} Reversal (5m)"
        else:  # BOS
            desc += f" {direction} Continuation (5m)"
        desc += f" ({original_signal})"
    elif source == 'M15_MARKET_STRUCTURE':  # ADD THIS
        desc = f"M15 MS: {structure_type}"
        if structure_type == 'CHoCH':
            desc += f" MAJOR {direction} Reversal"
        else:  # BOS
            desc += f" Strong {direction} Continuation"
        desc += f" ({original_signal})"
    else:
        desc = f"{source} {original_signal}"
    
    return desc

def get_source_identifier(signal: str) -> Optional[str]:
    """Identify the source from the signal description"""
    # Check for EMA signals
    if "M1 EMA" in signal:
        return "M1_EMA"
    elif "M5 EMA" in signal:
        return "M5_EMA"
    elif "M15 EMA" in signal:
        return "M15_EMA"
    
    # Check for Statistical Trend signals
    elif "M1 Trend" in signal:
        return "STATISTICAL_TREND"
    elif "M5 Trend" in signal:
        return "STATISTICAL_TREND_5M"
    elif "M15 Trend" in signal:
        return "STATISTICAL_TREND_15M"
    
    # Check for Market Structure signals
    elif "M1 MS:" in signal:
        return "M1_MARKET_STRUCTURE"
    elif "M5 MS:" in signal:
        return "M5_MARKET_STRUCTURE"
    elif "M15 MS:" in signal:
        return "M15_MARKET_STRUCTURE"
    
    return None