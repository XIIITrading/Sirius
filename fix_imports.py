# fix_imports.py
import os
import re

def fix_imports_in_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix patterns
    replacements = [
        # Fix relative imports to absolute
        (r'from data\.debug\.', 'from backtest.data.debug.'),
        (r'from data\.', 'from backtest.data.'),
        (r'from \.', 'from backtest.data.'),
        
        # Fix specific issues
        (r'from polygon_data_manager import', 'from backtest.data.polygon_data_manager import'),
        (r'from backtest.data.polygon_data_manager import PolygonDataManager\nfrom backtest.data.polygon_data_manager import PolygonDataManager', 
         'from backtest.data.polygon_data_manager import PolygonDataManager'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed imports in {filepath}")

# Fix all test files
test_files = [
    'backtest/data/debug/cli_tester.py',
    'backtest/data/debug/test_circuit_breaker.py',
    'backtest/data/debug/test_data_validator.py',
    'backtest/data/debug/test_integration.py',
    'backtest/data/debug/test_request_aggregator.py',
    'backtest/data/debug/test_trade_quote_aligner.py',
    'backtest/data/debug/test_utils.py',
    'backtest/data/data_coordinator.py',
]

for filepath in test_files:
    if os.path.exists(filepath):
        fix_imports_in_file(filepath)