# modules/calculations/trend/statistical_trend_1min.py
"""
Single-timeframe Statistical Trend Analyzer
Uses statistical methods on 1-minute bars to determine trend confidence
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class StatisticalSignal:
    """Statistical trend signal with confidence metrics"""
    symbol: str
    timestamp: datetime
    price: float
    signal: str  # 'STRONG BUY', 'BUY', 'WEAK BUY', 'NEUTRAL', 'WEAK SELL', 'SELL', 'STRONG SELL'
    confidence: float  # 0-100, based on statistical tests
    p_value: float  # Statistical significance
    trend_strength: float  # Effect size
    volatility: float  # Recent volatility
    metrics: Dict  # Detailed statistical metrics


class StatisticalTrend1Min:
    """
    Statistical trend analyzer using single timeframe with multiple signal levels
    
    Signal Levels:
    - STRONG BUY/SELL: >= 60% confidence
    - BUY/SELL: >= 50% confidence  
    - WEAK BUY/SELL: >= 25% confidence
    - NEUTRAL: < 25% confidence
    """
    
    def __init__(self, lookback_periods: int = 10, min_confidence: float = 25.0):
        """
        Args:
            lookback_periods: Number of 1-min bars to analyze (default 10)
            min_confidence: Minimum confidence for signals (not used with new thresholds)
        """
        self.lookback_periods = lookback_periods
        
        # Confidence thresholds for different signal levels
        self.thresholds = {
            'strong': 60.0,    # Strong Buy/Sell
            'normal': 50.0,    # Buy/Sell (Very Bullish/Bearish)
            'weak': 25.0       # Weak Buy/Sell (Bullish/Bearish)
        }
        
    def analyze(self, symbol: str, bars_df: pd.DataFrame, 
                entry_time: datetime) -> StatisticalSignal:
        """
        Perform statistical analysis on recent bars
        
        Args:
            symbol: Stock symbol
            bars_df: DataFrame with OHLCV data
            entry_time: Time point for analysis
            
        Returns:
            StatisticalSignal with confidence metrics
        """
        # Get recent bars up to entry time
        recent_bars = bars_df[bars_df.index <= entry_time].tail(self.lookback_periods)
        
        if len(recent_bars) < self.lookback_periods:
            raise ValueError(f"Insufficient data: need {self.lookback_periods} bars")
        
        prices = recent_bars['close'].values
        volumes = recent_bars['volume'].values
        current_price = prices[-1]
        
        # Run statistical tests
        metrics = {}
        
        # 1. Linear Regression Analysis
        lr_results = self._linear_regression_test(prices)
        metrics['linear_regression'] = lr_results
        
        # 2. Mann-Kendall Trend Test (non-parametric)
        mk_results = self._mann_kendall_test(prices)
        metrics['mann_kendall'] = mk_results
        
        # 3. Price Momentum with Statistical Significance
        momentum_results = self._momentum_analysis(prices, volumes)
        metrics['momentum'] = momentum_results
        
        # 4. Volatility Analysis
        volatility_results = self._volatility_analysis(prices)
        metrics['volatility'] = volatility_results
        
        # 5. Volume-Price Correlation
        vp_results = self._volume_price_correlation(prices, volumes)
        metrics['volume_price'] = vp_results
        
        # Combine all metrics into final signal
        signal = self._generate_statistical_signal(metrics, current_price)
        signal.symbol = symbol
        signal.timestamp = entry_time
        signal.metrics = metrics
        
        return signal
    
    def _linear_regression_test(self, prices: np.ndarray) -> Dict:
        """
        Perform linear regression with statistical tests
        
        Returns:
            - slope: Normalized slope (% per minute)
            - r_squared: Goodness of fit
            - p_value: Significance of slope
            - confidence: 0-100 based on R² and p-value
        """
        n = len(prices)
        x = np.arange(n)
        
        # Perform regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        r_squared = r_value ** 2
        
        # Normalize slope to % per minute
        normalized_slope = (slope / prices[0]) * 100
        
        # Calculate confidence based on R² and p-value
        # High R² means good fit, low p-value means significant slope
        if p_value < 0.05 and r_squared > 0.5:
            confidence = min(100, r_squared * 100)
        elif p_value < 0.1 and r_squared > 0.3:
            confidence = min(80, r_squared * 80)
        else:
            confidence = r_squared * 50
        
        return {
            'slope': normalized_slope,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_error': std_err,
            'confidence': confidence,
            'direction': 'bullish' if slope > 0 else 'bearish'
        }
    
    def _mann_kendall_test(self, prices: np.ndarray) -> Dict:
        """
        Mann-Kendall trend test (non-parametric)
        
        Good for detecting monotonic trends without assuming linearity
        """
        n = len(prices)
        s = 0
        
        # Calculate S statistic
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(prices[j] - prices[i])
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate z-score
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Confidence based on p-value
        if p_value < 0.01:
            confidence = 95
        elif p_value < 0.05:
            confidence = 80
        elif p_value < 0.1:
            confidence = 60
        else:
            confidence = 30
        
        return {
            'z_score': z,
            'p_value': p_value,
            'trend': 'bullish' if z > 0 else 'bearish' if z < 0 else 'neutral',
            'confidence': confidence
        }
    
    def _momentum_analysis(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """
        Analyze momentum with volume weighting
        """
        # Simple momentum
        momentum = (prices[-1] - prices[0]) / prices[0] * 100
        
        # Volume-weighted momentum (emphasize high-volume moves)
        weights = volumes / volumes.sum()
        weighted_prices = prices * weights
        weighted_momentum = (weighted_prices[-1] - weighted_prices[0]) / weighted_prices[0] * 100
        
        # Test if momentum is statistically significant
        # Using returns to test if mean return is different from zero
        returns = np.diff(prices) / prices[:-1]
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        
        # Confidence based on momentum magnitude and significance
        if abs(momentum) > 0.5 and p_value < 0.05:
            confidence = min(100, abs(momentum) * 100)
        else:
            confidence = abs(momentum) * 50
        
        return {
            'simple_momentum': momentum,
            'volume_weighted_momentum': weighted_momentum,
            'returns_t_stat': t_stat,
            'returns_p_value': p_value,
            'confidence': confidence,
            'direction': 'bullish' if momentum > 0 else 'bearish'
        }
    
    def _volatility_analysis(self, prices: np.ndarray) -> Dict:
        """
        Analyze price volatility to adjust confidence
        """
        returns = np.diff(prices) / prices[:-1]
        volatility = returns.std() * 100  # Percentage volatility
        
        # Calculate trend-to-volatility ratio
        price_change = abs(prices[-1] - prices[0]) / prices[0] * 100
        signal_to_noise = price_change / (volatility + 0.0001)  # Avoid division by zero
        
        # High signal-to-noise means clearer trend
        if signal_to_noise > 2:
            volatility_confidence = 90
        elif signal_to_noise > 1:
            volatility_confidence = 70
        else:
            volatility_confidence = 40
        
        return {
            'volatility': volatility,
            'signal_to_noise': signal_to_noise,
            'confidence': volatility_confidence
        }
    
    def _volume_price_correlation(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """
        Test correlation between volume and price changes
        Strong correlation can indicate trend strength
        """
        price_changes = np.diff(prices)
        volume_changes = volumes[1:]  # Align with price changes
        
        # Spearman correlation (non-parametric)
        correlation, p_value = stats.spearmanr(volume_changes, price_changes)
        
        # Positive correlation means volume supports price movement
        confidence = 0
        if abs(correlation) > 0.5 and p_value < 0.05:
            confidence = 80
        elif abs(correlation) > 0.3 and p_value < 0.1:
            confidence = 60
        else:
            confidence = 30
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'confidence': confidence,
            'supports_trend': correlation > 0
        }
    
    def _generate_statistical_signal(self, metrics: Dict, current_price: float) -> StatisticalSignal:
        """
        Combine all statistical tests into final signal with multiple levels
        """
        # Extract individual confidences and directions
        tests = {
            'linear': metrics['linear_regression'],
            'mann_kendall': metrics['mann_kendall'],
            'momentum': metrics['momentum'],
            'volatility': metrics['volatility'],
            'volume': metrics['volume_price']
        }
        
        # Count bullish/bearish signals
        bullish_count = sum(1 for t in tests.values() 
                           if t.get('direction') == 'bullish' or t.get('trend') == 'bullish')
        bearish_count = sum(1 for t in tests.values() 
                           if t.get('direction') == 'bearish' or t.get('trend') == 'bearish')
        
        # Weight confidences by test importance
        weights = {
            'linear': 0.25,
            'mann_kendall': 0.25,
            'momentum': 0.20,
            'volatility': 0.20,
            'volume': 0.10
        }
        
        # Calculate weighted confidence
        weighted_confidence = sum(
            tests[test]['confidence'] * weight 
            for test, weight in weights.items()
        )
        
        # Determine signal direction and strength
        if bullish_count > bearish_count:
            # Bullish signals
            if weighted_confidence >= self.thresholds['strong']:
                signal = 'STRONG BUY'
                # Boost confidence if strong agreement
                if bullish_count >= 4:
                    weighted_confidence = min(100, weighted_confidence * 1.2)
            elif weighted_confidence >= self.thresholds['normal']:
                signal = 'BUY'
            elif weighted_confidence >= self.thresholds['weak']:
                signal = 'WEAK BUY'
            else:
                signal = 'NEUTRAL'
                
        elif bearish_count > bullish_count:
            # Bearish signals
            if weighted_confidence >= self.thresholds['strong']:
                signal = 'STRONG SELL'
                if bearish_count >= 4:
                    weighted_confidence = min(100, weighted_confidence * 1.2)
            elif weighted_confidence >= self.thresholds['normal']:
                signal = 'SELL'
            elif weighted_confidence >= self.thresholds['weak']:
                signal = 'WEAK SELL'
            else:
                signal = 'NEUTRAL'
        else:
            # No clear direction
            signal = 'NEUTRAL'
            weighted_confidence = weighted_confidence * 0.5  # Reduce confidence for neutral
        
        # Calculate average p-value for overall significance
        p_values = [
            tests['linear']['p_value'],
            tests['mann_kendall']['p_value'],
            tests['momentum']['returns_p_value'],
            tests['volume']['p_value']
        ]
        avg_p_value = np.mean(p_values)
        
        # Trend strength based on effect sizes
        trend_strength = abs(tests['momentum']['simple_momentum'])
        
        return StatisticalSignal(
            symbol='',  # Set by caller
            timestamp=datetime.now(timezone.utc),
            price=current_price,
            signal=signal,
            confidence=weighted_confidence,
            p_value=avg_p_value,
            trend_strength=trend_strength,
            volatility=tests['volatility']['volatility'],
            metrics={}  # Set by caller
        )

    # Add methods for live trading (WebSocket support)
    async def start_live_monitoring(self, symbols: list, callback=None):
        """Placeholder for live monitoring - implement based on your needs"""
        logger.info(f"Live monitoring would start for {symbols}")
        # Implementation would go here for WebSocket connection
        pass