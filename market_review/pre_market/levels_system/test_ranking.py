# test_ranking.py

from datetime import date
from src.services.ranking_engine import RankingEngine, MarketData
from src.services.level_service import LevelService

def test_ranking_single_ticker():
    """Test ranking algorithm with AAPL."""
    engine = RankingEngine()
    
    # Mock market data (in real implementation, this would come from market data API)
    market_data = MarketData(
        ticker="AAPL",
        current_price=194.00,  # Current price between some levels
        atr=2.50              # Average True Range
    )
    
    # Run ranking
    today = date.today()
    ranked_levels = engine.rank_levels("AAPL", today, market_data)
    
    print(f"\n📊 Ranking Results Summary:")
    print(f"Total levels ranked: {len(ranked_levels)}")
    
    # Display top 5
    print("\n🏆 Top 5 Levels:")
    for level in ranked_levels[:5]:
        print(f"  #{level.rank}: {level.tv_variable} - Score: {level.confluence_score}")

def test_ranking_multiple_tickers():
    """Test batch processing of multiple tickers."""
    engine = RankingEngine()
    service = LevelService()
    
    # Get all tickers for today
    today = date.today()
    tickers = service.get_all_tickers_for_date(today)
    
    # Mock market data for each ticker
    market_data_dict = {
        "AAPL": MarketData(ticker="AAPL", current_price=194.00, atr=2.50),
        "SPY": MarketData(ticker="SPY", current_price=447.50, atr=4.20)
    }
    
    # Process all tickers
    results = engine.process_multiple_tickers(tickers, today, market_data_dict)
    
    print(f"\n📊 Batch Processing Results:")
    for ticker, levels in results.items():
        print(f"\n{ticker}: {len(levels)} levels ranked")
        if levels:
            print(f"  Top level: {levels[0].tv_variable} (Score: {levels[0].confluence_score})")

def display_detailed_results():
    """Display detailed results with all information."""
    service = LevelService()
    today = date.today()
    
    # Get ranked levels with full details
    detailed_levels = service.get_ranked_levels_with_details("AAPL", today)
    
    if detailed_levels:
        print(f"\n📊 Detailed Ranking Results for AAPL:")
        print(f"{'Rank':<6} {'Type':<10} {'Position':<10} {'Price':<8} {'Score':<8} {'Zone Low':<10} {'Zone High':<10} {'TV Var':<10}")
        print("-" * 90)
        
        for detail in detailed_levels[:10]:  # Show top 10
            ranked = detail.ranked_level
            premarket = detail.premarket_level
            
            print(f"{ranked.rank:<6} {premarket.level_type:<10} {premarket.position:<10} "
                  f"${premarket.price:<7.2f} {ranked.confluence_score:<8.2f} "
                  f"${ranked.zone_low:<9.2f} ${ranked.zone_high:<9.2f} {ranked.tv_variable:<10}")

if __name__ == "__main__":
    print("🚀 Testing Ranking Engine\n")
    
    # Test single ticker ranking
    test_ranking_single_ticker()
    
    # Test batch processing
    print("\n" + "="*50 + "\n")
    test_ranking_multiple_tickers()
    
    # Show detailed results
    print("\n" + "="*50)
    display_detailed_results()