# src/test_data_access.py

from datetime import date
from src.services.level_service import LevelService
from src.models.level_models import PremarketLevel, RankedLevel

def test_workflow():
    """Test the complete data access workflow."""
    service = LevelService()
    
    # Get today's date
    today = date.today()
    
    # 1. Check what tickers have levels for today
    print("\n=== Checking Available Tickers ===")
    tickers = service.get_all_tickers_for_date(today)
    print(f"Tickers with levels: {tickers}")
    
    # 2. Get analysis summary
    print("\n=== Analysis Summary ===")
    summary = service.get_analysis_summary(today)
    print(f"Total entered: {summary['total_entered']}")
    print(f"Total analyzed: {summary['total_analyzed']}")
    print(f"Pending analysis: {summary['tickers_pending']}")
    
    # 3. If we have tickers, get levels for the first one
    if tickers:
        ticker = tickers[0]
        print(f"\n=== Getting Levels for {ticker} ===")
        levels = service.get_premarket_levels_for_analysis(ticker, today)
        
        for level in levels[:3]:  # Show first 3
            print(f"  {level.level_type} {level.position}: ${level.price} (Score: {level.strength_score})")

if __name__ == "__main__":
    test_workflow()