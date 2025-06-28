# modules/utils/result_formatter.py
"""
Module: Result Formatter for Claude Integration
Purpose: Format backtest results for optimal Claude analysis
Features: Structured formatting, metric extraction, context preservation
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FormattedResult:
    """Formatted result for Claude analysis"""
    summary: str
    detailed_metrics: Dict[str, Any]
    trading_signals: Dict[str, Any]
    recommendations: List[str]
    context: Dict[str, Any]


class BacktestResultFormatter:
    """Format backtest results for Claude analysis"""
    
    @staticmethod
    def format_results(results: Dict[str, Any], context: Dict[str, Any] = None) -> FormattedResult:
        """
        Format backtest results into structured format for Claude.
        
        Args:
            results: Raw backtest results
            context: Additional context (symbol, timeframe, etc.)
            
        Returns:
            FormattedResult object
        """
        # Extract summary
        summary = BacktestResultFormatter._create_summary(results, context)
        
        # Extract detailed metrics
        detailed_metrics = BacktestResultFormatter._extract_metrics(results)
        
        # Extract trading signals
        trading_signals = BacktestResultFormatter._extract_signals(results)
        
        # Generate initial recommendations
        recommendations = BacktestResultFormatter._generate_recommendations(results)
        
        return FormattedResult(
            summary=summary,
            detailed_metrics=detailed_metrics,
            trading_signals=trading_signals,
            recommendations=recommendations,
            context=context or {}
        )
    
    @staticmethod
    def _create_summary(results: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create executive summary"""
        summary_parts = []
        
        # Context
        if context:
            if 'symbol' in context:
                summary_parts.append(f"Symbol: {context['symbol']}")
            if 'entry_time' in context:
                summary_parts.append(f"Entry Time: {context['entry_time']}")
        
        # Primary signal
        if 'trend_1min' in results and results['trend_1min']:
            trend = results['trend_1min']
            if hasattr(trend, 'signal'):
                summary_parts.append(f"Primary Signal: {trend.signal}")
                summary_parts.append(f"Confidence: {trend.confidence:.1f}%")
        
        # Trend alignment
        trends = []
        for timeframe in ['1min', '5min', '15min']:
            key = f'trend_{timeframe}'
            if key in results and results[key]:
                trend = results[key]
                if hasattr(trend, 'direction'):
                    trends.append(f"{timeframe}: {trend.direction}")
                elif hasattr(trend, 'regime_state'):
                    trends.append(f"{timeframe}: {trend.regime_state}")
        
        if trends:
            summary_parts.append("Trend Alignment: " + ", ".join(trends))
        
        return " | ".join(summary_parts)
    
    @staticmethod
    def _extract_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all numerical metrics"""
        metrics = {}
        
        # Trend metrics
        for timeframe in ['1min', '5min', '15min']:
            key = f'trend_{timeframe}'
            if key in results and results[key]:
                trend = results[key]
                timeframe_metrics = {}
                
                # Extract attributes
                for attr in ['strength', 'confidence', 'composite_trend']:
                    if hasattr(trend, attr):
                        timeframe_metrics[attr] = getattr(trend, attr)
                
                # Extract components if available
                if hasattr(trend, 'components'):
                    components = trend.components
                    for comp_attr in ['linear_slope', 'linear_r_squared', 
                                     'mann_kendall_z_score', 'price_momentum']:
                        if hasattr(components, comp_attr):
                            timeframe_metrics[comp_attr] = getattr(components, comp_attr)
                
                metrics[key] = timeframe_metrics
        
        # Order flow metrics
        for flow_key in ['trade_size', 'tick_flow', 'volume_1min', 'market_context']:
            if flow_key in results and results[flow_key]:
                data = results[flow_key]
                if isinstance(data, dict):
                    metrics[flow_key] = {k: v for k, v in data.items() 
                                       if not k.startswith('_') and 
                                       isinstance(v, (int, float))}
        
        return metrics
    
    @staticmethod
    def _extract_signals(results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trading signals"""
        signals = {}
        
        # Primary scalper signal
        if 'trend_1min' in results and results['trend_1min']:
            trend = results['trend_1min']
            if hasattr(trend, 'signal'):
                signals['primary'] = {
                    'signal': trend.signal,
                    'confidence': getattr(trend, 'confidence', 0),
                    'target_hold': getattr(trend, 'target_hold', 'N/A'),
                    'reason': getattr(trend, 'reason', '')
                }
        
        # Timeframe directions
        for timeframe in ['1min', '5min', '15min']:
            key = f'trend_{timeframe}'
            if key in results and results[key]:
                trend = results[key]
                if hasattr(trend, 'direction'):
                    signals[f'{timeframe}_direction'] = trend.direction
                elif hasattr(trend, 'regime_state'):
                    signals[f'{timeframe}_regime'] = trend.regime_state
        
        return signals
    
    @staticmethod
    def _generate_recommendations(results: Dict[str, Any]) -> List[str]:
        """Generate initial recommendations based on results"""
        recommendations = []
        
        # Check trend alignment
        directions = []
        for timeframe in ['1min', '5min', '15min']:
            key = f'trend_{timeframe}'
            if key in results and results[key]:
                trend = results[key]
                if hasattr(trend, 'direction'):
                    directions.append(trend.direction)
        
        if len(set(directions)) > 1:
            recommendations.append("Consider waiting for better trend alignment across timeframes")
        
        # Check confidence levels
        confidences = []
        for timeframe in ['1min', '5min', '15min']:
            key = f'trend_{timeframe}'
            if key in results and results[key]:
                trend = results[key]
                if hasattr(trend, 'confidence'):
                    confidences.append(trend.confidence)
        
        if confidences and min(confidences) < 60:
            recommendations.append("Low confidence detected - consider additional confirmation signals")
        
        # Check order flow
        if 'trade_size' in results and results['trade_size']:
            if isinstance(results['trade_size'], dict):
                if results['trade_size'].get('large_trade_ratio', 0) > 0.7:
                    recommendations.append("High institutional activity detected - monitor for continuation")
        
        return recommendations
    
    @staticmethod
    def format_for_export(formatted_result: FormattedResult) -> str:
        """Format result for text export"""
        output = []
        output.append("BACKTEST ANALYSIS RESULTS")
        output.append("=" * 50)
        output.append(f"\nSUMMARY: {formatted_result.summary}\n")
        
        output.append("\nDETAILED METRICS:")
        output.append("-" * 30)
        for category, metrics in formatted_result.detailed_metrics.items():
            output.append(f"\n{category.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    output.append(f"  {metric}: {value:.3f}")
                else:
                    output.append(f"  {metric}: {value}")
        
        output.append("\n\nTRADING SIGNALS:")
        output.append("-" * 30)
        for signal_type, signal_data in formatted_result.trading_signals.items():
            if isinstance(signal_data, dict):
                output.append(f"\n{signal_type.upper()}:")
                for key, value in signal_data.items():
                    output.append(f"  {key}: {value}")
            else:
                output.append(f"{signal_type}: {signal_data}")
        
        if formatted_result.recommendations:
            output.append("\n\nINITIAL RECOMMENDATIONS:")
            output.append("-" * 30)
            for i, rec in enumerate(formatted_result.recommendations, 1):
                output.append(f"{i}. {rec}")
        
        return "\n".join(output)


# ============= STANDALONE TEST =============
if __name__ == "__main__":
    print("=== Testing Result Formatter ===\n")
    
    # Test data
    test_results = {
        'trend_1min': type('obj', (object,), {
            'signal': 'BUY',
            'confidence': 75.5,
            'strength': 68.2,
            'target_hold': '15-30 min',
            'direction': 'bullish',
            'components': type('obj', (object,), {
                'linear_slope': 0.023,
                'linear_r_squared': 0.85,
                'price_momentum': 0.45
            })()
        })(),
        'trend_5min': type('obj', (object,), {
            'direction': 'bullish',
            'strength': 72.3,
            'confidence': 80.1,
            'composite_trend': 0.65
        })(),
        'trend_15min': type('obj', (object,), {
            'regime_state': 'trending_up',
            'strength': 65.4,
            'confidence': 77.8
        })(),
        'trade_size': {
            'large_trade_ratio': 0.65,
            'buy_sell_imbalance': 0.23,
            'average_trade_size': 487
        }
    }
    
    test_context = {
        'symbol': 'AAPL',
        'entry_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        'timeframe': '1min'
    }
    
    # Test formatting
    formatter = BacktestResultFormatter()
    formatted = formatter.format_results(test_results, test_context)
    
    print("Formatted Result:")
    print("-" * 50)
    print(f"Summary: {formatted.summary}")
    print(f"\nMetrics: {json.dumps(formatted.detailed_metrics, indent=2)}")
    print(f"\nSignals: {json.dumps(formatted.trading_signals, indent=2)}")
    print(f"\nRecommendations: {formatted.recommendations}")
    
    print("\n\nText Export Format:")
    print("-" * 50)
    print(formatter.format_for_export(formatted))
    
    print("\n✅ Formatter test complete!")