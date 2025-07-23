# src/services/ranking_engine.py

from typing import List, Dict, Tuple, Optional
from datetime import date
from dataclasses import dataclass
import math

from src.models.level_models import PremarketLevel, RankedLevel, RankedLevelWithDetails
from src.services.level_service import LevelService

@dataclass
class MarketData:
    """Market data needed for ranking calculations."""
    ticker: str
    current_price: float
    atr: float
    
class RankingEngine:
    """
    Core ranking algorithm for pre-market levels.
    
    Implements the confluence-based scoring system:
    - Type Weight (40%): HVN > Daily MS > H1 OB
    - Manual Strength (20%): From analyst input
    - Confluence Factor (25%): Nearby levels boost score
    - Distance Score (15%): Closer to current price scores higher
    """
    
    def __init__(self):
        self.level_service = LevelService()
        
        # Type weights (normalized to 40 points max)
        self.type_weights = {
            'HVN': 40.0,        # 100% of type weight
            'Daily MS': 32.0,   # 80% of type weight  
            'H1 OB': 24.0       # 60% of type weight
        }
        
        # Position multipliers (closer positions get higher priority)
        self.position_multipliers = {
            'Above 1': 1.2,
            'Above 2': 1.0,
            'Below 1': 1.2,
            'Below 2': 1.0
        }
        
        # ATR multiplier for zone calculation
        self.atr_multiplier = 1.0
        
        # Confluence detection range (in ATR units)
        self.confluence_range = 1.5
    
    def rank_levels(self, ticker: str, target_date: date, market_data: MarketData) -> List[RankedLevel]:
        """
        Main ranking method - processes all levels for a ticker.
        
        Args:
            ticker: Stock symbol
            target_date: Date to process
            market_data: Current price and ATR data
            
        Returns:
            List of RankedLevel objects sorted by rank
        """
        print(f"\n🔄 Starting ranking process for {ticker}")
        print(f"   Current Price: ${market_data.current_price:.2f}")
        print(f"   ATR: ${market_data.atr:.2f}")
        
        # Clear any existing rankings for this ticker/date
        self.level_service.clear_existing_rankings(ticker, target_date)
        
        # Get all premarket levels
        levels = self.level_service.get_premarket_levels_for_analysis(ticker, target_date)
        
        if not levels:
            print(f"⚠️  No levels found for {ticker} on {target_date}")
            return []
        
        # Calculate scores for each level
        scored_levels = []
        for level in levels:
            score_components = self._calculate_level_score(level, levels, market_data)
            total_score = sum(score_components.values())
            
            # Apply position multiplier
            total_score *= self.position_multipliers.get(level.position, 1.0)
            
            scored_levels.append({
                'level': level,
                'score': total_score,
                'components': score_components
            })
        
        # Sort by score (highest first)
        scored_levels.sort(key=lambda x: x['score'], reverse=True)
        
        # Create ranked levels
        ranked_levels = []
        for rank, scored in enumerate(scored_levels, 1):
            level = scored['level']
            
            # Calculate zone boundaries using ATR
            zone_width = market_data.atr * self.atr_multiplier
            zone_high = level.price + zone_width
            zone_low = level.price - zone_width
            
            # Generate TradingView variable name
            tv_variable = self._generate_tv_variable(level.level_type, level.position)
            
            # Create RankedLevel object
            ranked_level = RankedLevel(
                premarket_level_id=level.id,
                date=target_date,
                ticker=ticker,
                rank=rank,
                confluence_score=round(scored['score'], 2),
                zone_high=round(zone_high, 2),
                zone_low=round(zone_low, 2),
                tv_variable=tv_variable,
                current_price=market_data.current_price,
                atr_value=market_data.atr
            )
            
            ranked_levels.append(ranked_level)
            
            # Print ranking details
            print(f"\n   Rank #{rank}: {level.level_type} {level.position} @ ${level.price:.2f}")
            print(f"   Score: {scored['score']:.2f}")
            print(f"   Components: {scored['components']}")
            print(f"   Zone: ${zone_low:.2f} - ${zone_high:.2f}")
        
        # Save to database
        saved_levels = self.level_service.save_ranked_levels(ranked_levels)
        
        print(f"\n✅ Ranking complete! Processed {len(saved_levels)} levels")
        return saved_levels
    
    def _calculate_level_score(self, level: PremarketLevel, all_levels: List[PremarketLevel], 
                              market_data: MarketData) -> Dict[str, float]:
        """
        Calculate individual score components for a level.
        
        Returns dict with score breakdown:
        - type_weight: Score from level type (40% max)
        - strength_score: Manual strength normalized (20% max)
        - confluence_score: Points for nearby levels (25% max)
        - distance_score: Proximity to current price (15% max)
        """
        components = {}
        
        # 1. Type Weight (40% of total)
        components['type_weight'] = self.type_weights.get(level.level_type, 0)
        
        # 2. Manual Strength Score (20% of total)
        # Normalize 1-100 to 0-20
        components['strength_score'] = (level.strength_score / 100) * 20
        
        # 3. Confluence Factor (25% of total)
        components['confluence_score'] = self._calculate_confluence(level, all_levels, market_data.atr)
        
        # 4. Distance Score (15% of total)
        components['distance_score'] = self._calculate_distance_score(level.price, market_data.current_price)
        
        return components
    
    def _calculate_confluence(self, target_level: PremarketLevel, all_levels: List[PremarketLevel], 
                             atr: float) -> float:
        """
        Calculate confluence score based on nearby levels.
        
        Confluence rules:
        - Levels within 1.5 ATR are considered confluent
        - Different level types in confluence score higher
        - Maximum 25 points
        """
        confluence_distance = atr * self.confluence_range
        confluence_count = 0
        different_type_bonus = 0
        
        for level in all_levels:
            if level.id == target_level.id:
                continue
                
            # Check if within confluence range
            distance = abs(level.price - target_level.price)
            if distance <= confluence_distance:
                confluence_count += 1
                
                # Bonus for different type confluence
                if level.level_type != target_level.level_type:
                    different_type_bonus += 2
        
        # Calculate score (max 25)
        base_score = min(confluence_count * 3, 15)  # Max 15 from count
        bonus_score = min(different_type_bonus, 10)  # Max 10 from type diversity
        
        return min(base_score + bonus_score, 25)
    
    def _calculate_distance_score(self, level_price: float, current_price: float) -> float:
        """
        Calculate distance score based on proximity to current price.
        
        Closer levels get higher scores (max 15 points).
        Uses exponential decay based on percentage distance.
        """
        # Calculate percentage distance
        pct_distance = abs(level_price - current_price) / current_price * 100
        
        # Exponential decay scoring
        # At 0% distance: 15 points
        # At 5% distance: ~7.5 points  
        # At 10% distance: ~3.75 points
        score = 15 * math.exp(-pct_distance / 5)
        
        return round(score, 2)
    
    def _generate_tv_variable(self, level_type: str, position: str) -> str:
        """Generate TradingView variable name."""
        type_map = {
            'HVN': 'hvn',
            'Daily MS': 'ds',
            'H1 OB': 'ob'
        }
        
        pos_map = {
            'Above 1': 'a1',
            'Above 2': 'a2', 
            'Below 1': 'b1',
            'Below 2': 'b2'
        }
        
        type_code = type_map.get(level_type, 'unk')
        pos_code = pos_map.get(position, 'unk')
        
        return f"{type_code}_{pos_code}"
    
    def process_multiple_tickers(self, tickers: List[str], target_date: date, 
                                market_data_dict: Dict[str, MarketData]) -> Dict[str, List[RankedLevel]]:
        """
        Process multiple tickers in batch.
        
        Args:
            tickers: List of ticker symbols
            target_date: Date to process
            market_data_dict: Dict of ticker -> MarketData
            
        Returns:
            Dict of ticker -> List[RankedLevel]
        """
        results = {}
        
        for ticker in tickers:
            if ticker not in market_data_dict:
                print(f"⚠️  Skipping {ticker} - no market data provided")
                continue
                
            market_data = market_data_dict[ticker]
            ranked_levels = self.rank_levels(ticker, target_date, market_data)
            results[ticker] = ranked_levels
        
        return results