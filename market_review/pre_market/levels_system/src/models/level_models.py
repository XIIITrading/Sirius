# src/models/level_models.py

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, List
from decimal import Decimal

@dataclass
class PremarketLevel:
    """
    Data model for a manually identified price level from Notion.
    
    This represents the raw data entered in Notion and synced to Supabase.
    """
    date: date
    ticker: str
    level_type: str  # 'HVN', 'Daily MS', 'H1 OB'
    position: str    # 'Above 1', 'Above 2', 'Below 1', 'Below 2'
    price: float
    strength_score: int  # 1-100
    notes: Optional[str] = None
    active: bool = True
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    
    def validate(self) -> List[str]:
        """Validate the level data."""
        errors = []
        
        # Validate level_type
        valid_types = ['HVN', 'Daily MS', 'H1 OB']
        if self.level_type not in valid_types:
            errors.append(f"Invalid level_type: {self.level_type}. Must be one of {valid_types}")
        
        # Validate position
        valid_positions = ['Above 1', 'Above 2', 'Below 1', 'Below 2']
        if self.position not in valid_positions:
            errors.append(f"Invalid position: {self.position}. Must be one of {valid_positions}")
        
        # Validate strength_score
        if not 1 <= self.strength_score <= 100:
            errors.append(f"Invalid strength_score: {self.strength_score}. Must be between 1 and 100")
        
        # Validate price
        if self.price <= 0:
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
    
    def __post_init__(self):
        """Generate TradingView variable name if not provided."""
        if not self.tv_variable and hasattr(self, 'level_type') and hasattr(self, 'position'):
            # Generate variable name based on level type and position
            type_map = {'HVN': 'hvn', 'Daily MS': 'ds', 'H1 OB': 'ob'}
            pos_map = {'Above 1': 'a1', 'Above 2': 'a2', 'Below 1': 'b1', 'Below 2': 'b2'}
            
            type_code = type_map.get(self.level_type, 'unk')
            pos_code = pos_map.get(self.position, 'unk')
            self.tv_variable = f"{type_code}_{pos_code}"


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
    def position(self) -> str:
        return self.premarket_level.position
    
    @property
    def price(self) -> float:
        return self.premarket_level.price
    
    @property
    def strength_score(self) -> int:
        return self.premarket_level.strength_score