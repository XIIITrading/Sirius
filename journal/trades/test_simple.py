"""Simple test to verify imports are working."""

print("Testing imports...")

try:
    from models import Trade, Execution
    print("✅ models imported")
except Exception as e:
    print(f"❌ models import failed: {e}")

try:
    from parser import TradeParser
    print("✅ parser imported")
except Exception as e:
    print(f"❌ parser import failed: {e}")

try:
    from processor import TradeProcessor
    print("✅ processor imported")
except Exception as e:
    print(f"❌ processor import failed: {e}")

try:
    from plugin import TradePlugin
    print("✅ plugin imported")
except Exception as e:
    print(f"❌ plugin import failed: {e}")

print("\nAll imports successful! You can now run test.py")