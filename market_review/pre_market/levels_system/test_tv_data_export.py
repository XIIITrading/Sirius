# test_tv_data_export.py

from datetime import date
from src.services.tradingview_generator import TradingViewGenerator

def test_symbol_data_generation():
    """Test generating symbol data for TradingView."""
    generator = TradingViewGenerator()
    
    # Generate for today
    today = date.today()
    
    # Generate data for all symbols
    print("📊 Generating TradingView Symbol Data")
    print("=" * 60)
    
    data = generator.generate_multiple_symbols(today)
    print(data)
    
    # Save to file
    filepath = generator.save_symbol_data(today)
    
    print("\n" + "=" * 60)
    print(f"✅ Data saved to: {filepath}")
    print("\n📋 To use in TradingView:")
    print("1. Open your Pre-Market Levels indicator in Pine Editor")
    print("2. Find the get_symbol_levels() function")
    print("3. Replace or add the symbol blocks from the generated file")
    print("4. Save and add to chart")

if __name__ == "__main__":
    test_symbol_data_generation()