# src/models/level_models.py

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, List

@dataclass
class PremarketLevel:
    """Model for premarket level data."""
    id: Optional[int] = None
    date: Optional[date] = None
    ticker: Optional[str] = None
    level_type: Optional[str] = None
    price: Optional[float] = None
    strength_score: Optional[int] = None
    notes: Optional[str] = None
    active: Optional[bool] = True
    created_at: Optional[datetime] = None
    
    def validate(self) -> List[str]:
        """Validate the level data."""
        errors = []
        
        # Validate level_type
        valid_types = ['HVN', 'Daily MS', 'H1 OB']
        if self.level_type not in valid_types:
            errors.append(f"Invalid level_type: {self.level_type}. Must be one of {valid_types}")
        
        # Validate strength_score
        if self.strength_score is not None and not 1 <= self.strength_score <= 100:
            errors.append(f"Invalid strength_score: {self.strength_score}. Must be between 1 and 100")
        
        # Validate price
        if self.price is not None and self.price <= 0:
            errors.append(f"Invalid price: {self.price}. Must be positive")
        
        return errors


@dataclass
class RankedLevel:
    """
    Data model for a processed/ranked level.
    
    This is created by the Python analyzer after processing PremarketLevels.
    """
    premarket_level_id: int
    date: date
    ticker: str
    rank: int  # 1-12 (3 types × 4 positions)
    confluence_score: float
    zone_high: float
    zone_low: float
    tv_variable: str  # e.g., 'hvn_a1', 'ds_b2'
    current_price: Optional[float] = None  # Price when ranking was done
    atr_value: Optional[float] = None      # ATR used for zone calculation
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class RankedLevelWithDetails:
    """
    Enriched ranked level that includes the original premarket level data.
    Used for display and TradingView code generation.
    """
    ranked_level: RankedLevel
    premarket_level: PremarketLevel
    
    @property
    def level_type(self) -> str:
        return self.premarket_level.level_type
    
    @property
    def price(self) -> float:
        return self.premarket_level.price
    
    @property
    def strength_score(self) -> int:
        return self.premarket_level.strength_score