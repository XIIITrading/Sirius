# create_test_data.py

from datetime import date
from src.services.level_service import LevelService
from src.models.level_models import PremarketLevel

def create_sample_levels():
    """Create sample levels for testing the ranking algorithm."""
    service = LevelService()
    today = date.today()
    
    # Sample levels for AAPL - complete set (3 types × 4 positions = 12 levels)
    test_levels = [
        # HVN levels
        PremarketLevel(date=today, ticker="AAPL", level_type="HVN", position="Above 1", 
                      price=195.50, strength_score=85, notes="Major volume node"),
        PremarketLevel(date=today, ticker="AAPL", level_type="HVN", position="Above 2", 
                      price=198.25, strength_score=70, notes="Secondary volume shelf"),
        PremarketLevel(date=today, ticker="AAPL", level_type="HVN", position="Below 1", 
                      price=192.00, strength_score=90, notes="Strong volume support"),
        PremarketLevel(date=today, ticker="AAPL", level_type="HVN", position="Below 2", 
                      price=189.75, strength_score=65, notes="Lower volume node"),
        
        # Daily MS levels
        PremarketLevel(date=today, ticker="AAPL", level_type="Daily MS", position="Above 1", 
                      price=196.00, strength_score=75, notes="Previous day high"),
        PremarketLevel(date=today, ticker="AAPL", level_type="Daily MS", position="Above 2", 
                      price=199.50, strength_score=80, notes="Weekly high"),
        PremarketLevel(date=today, ticker="AAPL", level_type="Daily MS", position="Below 1", 
                      price=191.50, strength_score=85, notes="Previous day low"),
        PremarketLevel(date=today, ticker="AAPL", level_type="Daily MS", position="Below 2", 
                      price=188.00, strength_score=70, notes="Weekly low"),
        
        # H1 OB levels
        PremarketLevel(date=today, ticker="AAPL", level_type="H1 OB", position="Above 1", 
                      price=194.75, strength_score=60, notes="Bullish OB from yesterday"),
        PremarketLevel(date=today, ticker="AAPL", level_type="H1 OB", position="Above 2", 
                      price=197.00, strength_score=55, notes="Older bullish OB"),
        PremarketLevel(date=today, ticker="AAPL", level_type="H1 OB", position="Below 1", 
                      price=192.50, strength_score=65, notes="Bearish OB support"),
        PremarketLevel(date=today, ticker="AAPL", level_type="H1 OB", position="Below 2", 
                      price=190.25, strength_score=50, notes="Deeper bearish OB"),
    ]
    
    # Insert all levels
    print(f"Creating {len(test_levels)} test levels for AAPL...")
    
    for level in test_levels:
        try:
            # First check if this exact level already exists
            existing = service.client.table('premarket_levels')\
                .select("*")\
                .eq('date', level.date.isoformat())\
                .eq('ticker', level.ticker)\
                .eq('level_type', level.level_type)\
                .eq('position', level.position)\
                .execute()
            
            if existing.data:
                print(f"⚠️  Level already exists: {level.level_type} {level.position}")
                continue
            
            # Insert the level
            data = {
                'date': level.date.isoformat(),
                'ticker': level.ticker,
                'level_type': level.level_type,
                'position': level.position,
                'price': level.price,
                'strength_score': level.strength_score,
                'notes': level.notes,
                'active': level.active
            }
            
            result = service.client.table('premarket_levels').insert(data).execute()
            print(f"✅ Created: {level.level_type} {level.position} @ ${level.price}")
            
        except Exception as e:
            print(f"❌ Error creating level: {e}")
    
    # Also create some levels for another ticker (SPY)
    spy_levels = [
        PremarketLevel(date=today, ticker="SPY", level_type="HVN", position="Above 1", 
                      price=450.50, strength_score=80, notes="Major SPY volume"),
        PremarketLevel(date=today, ticker="SPY", level_type="HVN", position="Below 1", 
                      price=445.00, strength_score=85, notes="SPY support volume"),
    ]
    
    print(f"\nCreating {len(spy_levels)} test levels for SPY...")
    for level in spy_levels:
        try:
            data = {
                'date': level.date.isoformat(),
                'ticker': level.ticker,
                'level_type': level.level_type,
                'position': level.position,
                'price': level.price,
                'strength_score': level.strength_score,
                'notes': level.notes,
                'active': level.active
            }
            
            result = service.client.table('premarket_levels').insert(data).execute()
            print(f"✅ Created: {level.ticker} {level.level_type} {level.position}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    create_sample_levels()
    print("\n✨ Test data creation complete!")