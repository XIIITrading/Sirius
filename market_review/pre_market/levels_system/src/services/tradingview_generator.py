# src/services/tradingview_generator.py

from typing import List, Dict
from datetime import date
from src.models.level_models import RankedLevelWithDetails
from src.services.level_service import LevelService

class TradingViewGenerator:
    """
    Generates symbol level data for TradingView indicator.
    Only creates the data section that can be plugged into existing Pine Script template.
    """
    
    def __init__(self):
        self.level_service = LevelService()
    
    def generate_symbol_data(self, ticker: str, target_date: date) -> str:
        """
        Generate symbol data block for a ticker's ranked levels.
        
        Args:
            ticker: Stock symbol
            target_date: Date of levels
            
        Returns:
            Pine Script code snippet for symbol levels
        """
        # Get ranked levels with details
        levels = self.level_service.get_ranked_levels_with_details(ticker, target_date)
        
        if not levels:
            return f"// No ranked levels found for {ticker} on {target_date}"
        
        # Organize levels by type and position
        level_map = self._organize_levels(levels)
        
        # Generate the symbol data block
        return self._generate_symbol_block(ticker, level_map, target_date)
    
    def _organize_levels(self, levels: List[RankedLevelWithDetails]) -> Dict[str, float]:
        """
        Organize levels into a map by their TradingView variable names.
        
        Returns dict like: {'ds_a1': 195.50, 'hvn_b1': 192.00, ...}
        """
        level_map = {}
        
        for detail in levels:
            ranked = detail.ranked_level
            premarket = detail.premarket_level
            
            # Map level type to variable prefix
            type_prefix = {
                'Daily MS': 'ds',
                'HVN': 'hvn',
                'H1 OB': 'ob'
            }
            
            # Get the variable name (e.g., 'ds_a1', 'hvn_b2')
            var_name = ranked.tv_variable
            
            # Store the price
            level_map[var_name] = premarket.price
        
        return level_map
    
    def _generate_symbol_block(self, ticker: str, level_map: Dict[str, float], target_date: date) -> str:
        """
        Generate the Pine Script code block for a symbol.
        """
        lines = []
        
        # Header comment
        lines.append(f"    // {ticker} levels - Generated {target_date}")
        lines.append(f"    if current_symbol == \"{ticker}\"")
        
        # Define all possible variables with defaults
        all_vars = [
            'ds_a1', 'ds_a2', 'ds_b1', 'ds_b2',
            'hvn_a1', 'hvn_a2', 'hvn_b1', 'hvn_b2',
            'ob_a1', 'ob_a2', 'ob_b1', 'ob_b2'
        ]
        
        # Generate assignments
        for var in all_vars:
            if var in level_map:
                price = level_map[var]
                lines.append(f"        {var} := {price:.2f}")
            else:
                # Set to 0 if no level exists for this position
                lines.append(f"        {var} := 0.0")
        
        return '\n'.join(lines)
    
    def generate_multiple_symbols(self, target_date: date) -> str:
        """
        Generate data blocks for all symbols with levels on the given date.
        """
        # Get all tickers for the date
        tickers = self.level_service.get_all_tickers_for_date(target_date)
        
        if not tickers:
            return f"// No symbols found with levels for {target_date}"
        
        blocks = []
        
        # Generate header
        blocks.append(f"// ==================== SYMBOL LEVELS ====================")
        blocks.append(f"// Generated: {target_date}")
        blocks.append(f"// Symbols: {', '.join(tickers)}")
        blocks.append("")
        
        # Generate each symbol block
        for i, ticker in enumerate(tickers):
            block = self.generate_symbol_data(ticker, target_date)
            
            # Add 'else' for subsequent symbols
            if i > 0:
                block = block.replace("    if current_symbol", "    else if current_symbol")
            
            blocks.append(block)
            blocks.append("")  # Empty line between symbols
        
        return '\n'.join(blocks)
    
    def save_symbol_data(self, target_date: date, filename: str = None) -> str:
        """
        Generate and save symbol data to a file.
        
        Args:
            target_date: Date of levels to export
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        import os
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            filename = f"symbol_levels_{target_date.strftime('%Y%m%d')}.txt"
        
        filepath = os.path.join("output", filename)
        
        # Generate the data
        data = self.generate_multiple_symbols(target_date)
        
        # Save to file
        with open(filepath, 'w') as f:
            f.write(data)
        
        print(f"✅ Symbol data saved to: {filepath}")
        return filepath