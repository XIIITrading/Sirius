# market_review/market_filter_run.py
"""
Simple runner for the market filter
"""

import sys
import os

# Add parent directory to path when running directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_review.pre_market.sp500_filter.market_filter import example_usage

if __name__ == "__main__":
    example_usage()