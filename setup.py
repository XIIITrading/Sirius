# C:\XIIITrading\Sirius\setup.py
from setuptools import setup, find_packages

setup(
    name="market_review",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "pyqtgraph",
        "PyQt6",
        "python-dotenv",
        "supabase",
        # Add other dependencies
    ],
)