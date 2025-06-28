# modules/integrations/claude_integration.py
"""
Module: Claude API Integration
Purpose: Modular integration with Anthropic's Claude API for trading analysis
Features: Conversation management, retry logic, rate limiting, context preservation
Performance: Async support, response caching, error handling
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import time
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message
import aiohttp
from functools import lru_cache

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s'
)
logging.Formatter.converter = time.gmtime
logger = logging.getLogger(__name__)


@dataclass
class ClaudeMessage:
    """Single message in conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaudeConversation:
    """Conversation session with Claude"""
    id: str
    messages: List[ClaudeMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = field(default_factory=dict)
    model: str = "claude-3-opus-20240229"
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add message to conversation"""
        self.messages.append(ClaudeMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        ))
    
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Format messages for Claude API"""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]
    
    def to_dict(self) -> Dict:
        """Convert conversation to dictionary for export"""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'model': self.model,
            'context': self.context,
            'messages': [
                {
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat(),
                    'metadata': msg.metadata
                }
                for msg in self.messages
            ]
        }


class ClaudeIntegration:
    """
    Modular Claude API integration for trading analysis.
    Handles conversations, rate limiting, and error recovery.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
        """
        Initialize Claude integration.
        
        Args:
            api_key: Claude API key (defaults to env variable)
            model: Claude model to use
        """
        self.api_key = api_key or os.getenv('CLAUDE_API_KEY')
        if not self.api_key:
            raise ValueError("Claude API key not found. Set CLAUDE_API_KEY in .env file")
        
        self.model = model
        self.client = Anthropic(api_key=self.api_key)
        self.async_client = AsyncAnthropic(api_key=self.api_key)
        
        # Conversation management
        self.conversations: Dict[str, ClaudeConversation] = {}
        self.active_conversation_id: Optional[str] = None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # seconds between requests
        
        # Configuration
        self.max_tokens = 4096
        self.temperature = 0.7
        self.system_prompt = """You are an expert quantitative trading analyst and strategy optimizer. 
You analyze backtesting results and provide actionable insights for improving trading strategies.
Focus on:
1. Signal quality and timing analysis
2. Risk/reward optimization
3. Market regime considerations
4. Practical implementation improvements
5. Statistical significance of patterns

Provide specific, actionable recommendations backed by the data provided.
Always consider transaction costs, slippage, and real-world execution challenges."""
        
        logger.info(f"Claude integration initialized with model: {self.model}")
    
    def create_conversation(self, context: Dict[str, Any] = None) -> str:
        """
        Create new conversation session.
        
        Args:
            context: Initial context (symbol, timeframe, etc.)
            
        Returns:
            Conversation ID
        """
        conv_id = f"conv_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        conversation = ClaudeConversation(
            id=conv_id,
            context=context or {},
            model=self.model
        )
        self.conversations[conv_id] = conversation
        self.active_conversation_id = conv_id
        
        logger.info(f"Created new conversation: {conv_id}")
        return conv_id
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def send_message(self, content: str, conversation_id: Optional[str] = None,
                    metadata: Dict = None) -> Tuple[str, ClaudeConversation]:
        """
        Send message to Claude and get response.
        
        Args:
            content: Message content
            conversation_id: Specific conversation (uses active if None)
            metadata: Additional metadata to store
            
        Returns:
            Tuple of (response_text, conversation)
        """
        # Get or create conversation
        if conversation_id:
            if conversation_id not in self.conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
            conversation = self.conversations[conversation_id]
        elif self.active_conversation_id:
            conversation = self.conversations[self.active_conversation_id]
        else:
            # Create new conversation
            conv_id = self.create_conversation()
            conversation = self.conversations[conv_id]
        
        # Add user message
        conversation.add_message('user', content, metadata)
        
        # Enforce rate limit
        self._enforce_rate_limit()
        
        try:
            # Prepare messages for API
            messages = conversation.get_messages_for_api()
            
            # Send to Claude
            logger.info(f"Sending message to Claude (conversation: {conversation.id})")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=messages
            )
            
            # Extract response
            response_text = response.content[0].text
            
            # Add assistant response to conversation
            conversation.add_message('assistant', response_text, {
                'model': self.model,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens
                }
            })
            
            logger.info(f"Received response: {len(response_text)} chars")
            return response_text, conversation
            
        except Exception as e:
            logger.error(f"Error sending message to Claude: {e}")
            raise
    
    async def send_message_async(self, content: str, conversation_id: Optional[str] = None,
                               metadata: Dict = None) -> Tuple[str, ClaudeConversation]:
        """Async version of send_message"""
        # Get or create conversation
        if conversation_id:
            if conversation_id not in self.conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
            conversation = self.conversations[conversation_id]
        elif self.active_conversation_id:
            conversation = self.conversations[self.active_conversation_id]
        else:
            conv_id = self.create_conversation()
            conversation = self.conversations[conv_id]
        
        # Add user message
        conversation.add_message('user', content, metadata)
        
        # Enforce rate limit
        await asyncio.sleep(max(0, self.min_request_interval - (time.time() - self.last_request_time)))
        self.last_request_time = time.time()
        
        try:
            messages = conversation.get_messages_for_api()
            
            logger.info(f"Sending async message to Claude (conversation: {conversation.id})")
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=messages
            )
            
            response_text = response.content[0].text
            
            conversation.add_message('assistant', response_text, {
                'model': self.model,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens
                }
            })
            
            return response_text, conversation
            
        except Exception as e:
            logger.error(f"Error in async message: {e}")
            raise
    
    def analyze_backtest_results(self, results: Dict[str, Any], 
                               context: Dict[str, Any] = None) -> Tuple[str, str]:
        """
        Analyze backtest results and get recommendations.
        
        Args:
            results: Backtest results dictionary
            context: Additional context (symbol, entry time, etc.)
            
        Returns:
            Tuple of (analysis, conversation_id)
        """
        # Create new conversation with context
        conv_id = self.create_conversation(context or {})
        
        # Format the analysis request
        prompt = self._format_backtest_prompt(results, context)
        
        # Send to Claude
        analysis, _ = self.send_message(prompt, conv_id)
        
        return analysis, conv_id
    
    def _format_backtest_prompt(self, results: Dict[str, Any], 
                               context: Dict[str, Any] = None) -> str:
        """Format backtest results into Claude prompt"""
        prompt = "Please analyze these backtesting results and provide optimization recommendations:\n\n"
        
        # Add context
        if context:
            prompt += "**Context:**\n"
            if 'symbol' in context:
                prompt += f"- Symbol: {context['symbol']}\n"
            if 'entry_time' in context:
                prompt += f"- Entry Time: {context['entry_time']}\n"
            if 'timeframe' in context:
                prompt += f"- Timeframe: {context['timeframe']}\n"
            prompt += "\n"
        
        # Add trend analysis results
        prompt += "**Trend Analysis Results:**\n"
        
        # 1-minute trend
        if 'trend_1min' in results and results['trend_1min']:
            trend = results['trend_1min']
            prompt += f"\n1-Minute Trend (Scalper):\n"
            prompt += f"- Signal: {trend.signal}\n"
            prompt += f"- Confidence: {trend.confidence:.1f}%\n"
            prompt += f"- Strength: {trend.strength:.1f}%\n"
            prompt += f"- Target Hold: {trend.target_hold}\n"
            if hasattr(trend, 'micro_trend') and trend.micro_trend:
                prompt += f"- Micro Trend: {trend.micro_trend.get('direction', 'N/A')}\n"
        
        # 5-minute trend
        if 'trend_5min' in results and results['trend_5min']:
            trend = results['trend_5min']
            prompt += f"\n5-Minute Trend (Position):\n"
            prompt += f"- Direction: {trend.direction}\n"
            prompt += f"- Strength: {trend.strength:.1f}%\n"
            prompt += f"- Confidence: {trend.confidence:.1f}%\n"
        
        # 15-minute trend
        if 'trend_15min' in results and results['trend_15min']:
            trend = results['trend_15min']
            prompt += f"\n15-Minute Trend (Regime):\n"
            prompt += f"- State: {trend.regime_state}\n"
            prompt += f"- Strength: {trend.strength:.1f}%\n"
            prompt += f"- Confidence: {trend.confidence:.1f}%\n"
        
        # Order flow analysis
        prompt += "\n**Order Flow Analysis:**\n"
        
        # Add order flow metrics
        for key in ['trade_size', 'tick_flow', 'volume_1min', 'market_context']:
            if key in results and results[key]:
                prompt += f"\n{key.replace('_', ' ').title()}:\n"
                data = results[key]
                if isinstance(data, dict):
                    for k, v in data.items():
                        if not k.startswith('_'):  # Skip private keys
                            prompt += f"- {k}: {v}\n"
                else:
                    prompt += f"- Data: {str(data)[:200]}...\n"
        
        prompt += "\n**Questions:**\n"
        prompt += "1. What are the key strengths and weaknesses of this setup?\n"
        prompt += "2. How could the entry timing be improved?\n"
        prompt += "3. What market conditions would make this setup more reliable?\n"
        prompt += "4. Suggest specific parameter adjustments for the calculations.\n"
        prompt += "5. What additional indicators or filters would improve the strategy?\n"
        
        return prompt
    
    def export_conversation(self, conversation_id: str, filepath: str):
        """Export conversation to JSON file"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conversation = self.conversations[conversation_id]
        
        with open(filepath, 'w') as f:
            json.dump(conversation.to_dict(), f, indent=2)
        
        logger.info(f"Exported conversation to: {filepath}")
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get summary of conversation"""
        if conversation_id not in self.conversations:
            return {}
        
        conv = self.conversations[conversation_id]
        
        return {
            'id': conv.id,
            'created_at': conv.created_at.isoformat(),
            'message_count': len(conv.messages),
            'model': conv.model,
            'context': conv.context,
            'total_input_tokens': sum(
                msg.metadata.get('usage', {}).get('input_tokens', 0)
                for msg in conv.messages if msg.role == 'assistant'
            ),
            'total_output_tokens': sum(
                msg.metadata.get('usage', {}).get('output_tokens', 0)
                for msg in conv.messages if msg.role == 'assistant'
            )
        }


# ============= STANDALONE TEST =============
if __name__ == "__main__":
    print("=== Testing Claude Integration Module ===\n")
    
    # Test data simulating backtest results
    test_results = {
        'trend_1min': type('obj', (object,), {
            'signal': 'BUY',
            'confidence': 75.5,
            'strength': 68.2,
            'target_hold': '15-30 min',
            'micro_trend': {'direction': 'bullish'}
        })(),
        'trend_5min': type('obj', (object,), {
            'direction': 'bullish',
            'strength': 72.3,
            'confidence': 80.1
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
        'symbol': 'TSLA',
        'entry_time': datetime.now(timezone.utc).isoformat(),
        'timeframe': '1min'
    }
    
    try:
        # Initialize integration
        claude = ClaudeIntegration()
        print("✓ Claude integration initialized\n")
        
        # Test 1: Create conversation
        conv_id = claude.create_conversation(test_context)
        print(f"✓ Created conversation: {conv_id}\n")
        
        # Test 2: Analyze backtest results
        print("Analyzing backtest results...")
        analysis, conv_id = claude.analyze_backtest_results(test_results, test_context)
        print(f"\nClaude's Analysis:\n{'-'*50}\n{analysis}\n{'-'*50}\n")
        
        # Test 3: Follow-up question
        print("Sending follow-up question...")
        follow_up = "Based on the momentum indicators, should I tighten the stop loss for this setup?"
        response, _ = claude.send_message(follow_up, conv_id)
        print(f"\nFollow-up Response:\n{'-'*50}\n{response}\n{'-'*50}\n")
        
        # Test 4: Get conversation summary
        summary = claude.get_conversation_summary(conv_id)
        print(f"Conversation Summary:")
        print(f"- Messages: {summary['message_count']}")
        print(f"- Input tokens: {summary['total_input_tokens']}")
        print(f"- Output tokens: {summary['total_output_tokens']}")
        
        # Test 5: Export conversation
        export_path = f"claude_conversation_{conv_id}.json"
        claude.export_conversation(conv_id, export_path)
        print(f"\n✓ Exported conversation to: {export_path}")
        
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()