# diagnose_imports.py
import os
import sys
import importlib.util

print(f"Python Version: {sys.version}")
print(f"Current Directory: {os.getcwd()}")
print(f"Python Path[0]: {sys.path[0]}\n")

# Check if modules exist
modules_to_check = [
    "market_review",
    "market_review.pre_market",
    "market_review.pre_market.sp500_filter",
    "market_review.pre_market.sp500_filter.market_filter"
]

for module_name in modules_to_check:
    try:
        spec = importlib.util.find_spec(module_name)
        if spec:
            print(f"✅ Found module: {module_name}")
            print(f"   Location: {spec.origin}")
        else:
            print(f"❌ Module not found: {module_name}")
    except Exception as e:
        print(f"❌ Error finding {module_name}: {e}")

print("\n" + "="*50 + "\n")

# Try to import step by step
try:
    print("Step 1: Import market_review")
    import market_review
    print("✅ Success\n")
    
    print("Step 2: Import market_review.pre_market")
    import market_review.pre_market
    print("✅ Success\n")
    
    print("Step 3: Import market_review.pre_market.sp500_filter")
    import market_review.pre_market.sp500_filter
    print("✅ Success\n")
    
    print("Step 4: Import market_filter module directly")
    import market_review.pre_market.sp500_filter.market_filter
    print("✅ Success\n")
    
    print("Step 5: Import MarketFilter class")
    from market_review.pre_market.sp500_filter.market_filter import MarketFilter
    print("✅ Success - MarketFilter imported!\n")
    
except Exception as e:
    print(f"❌ Failed at: {e}")
    import traceback
    traceback.print_exc()