# test_tradingview.py

from datetime import date
from src.services.tradingview_generator import TradingViewGenerator

def test_generate_script():
    """Test TradingView script generation."""
    generator = TradingViewGenerator()
    
    # Generate script for AAPL
    today = date.today()
    script = generator.generate_script("AAPL", today)
    
    # Print first 50 lines
    lines = script.split('\n')
    print("📊 Generated TradingView Script Preview:")
    print("=" * 60)
    for i, line in enumerate(lines[:50]):
        print(line)
    print(f"\n... ({len(lines)} total lines)")
    
    # Save to file
    filepath = generator.save_to_file("AAPL", today)
    
    # Also generate for SPY if it has levels
    try:
        spy_script = generator.generate_script("SPY", today)
        if "No ranked levels" not in spy_script:
            spy_path = generator.save_to_file("SPY", today)
    except:
        pass

if __name__ == "__main__":
    test_generate_script()