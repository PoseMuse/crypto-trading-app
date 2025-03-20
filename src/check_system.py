#!/usr/bin/env python
"""
System Compatibility Check Script for Cryptocurrency Trading Bot.

This script checks if the system has all the required components and dependencies
for running backtesting and paper trading simulations.
"""

import os
import sys
import argparse
import importlib
import subprocess
from pathlib import Path


class Compatibility:
    """Class to check system compatibility."""
    
    def __init__(self):
        self.checks = []
        self.warnings = []
        self.errors = []
    
    def add_check(self, message, success):
        """Add a check result."""
        self.checks.append((message, success))
        if not success:
            self.errors.append(message)
    
    def add_warning(self, message):
        """Add a warning message."""
        self.warnings.append(message)
    
    def print_report(self):
        """Print the compatibility report."""
        print("\n=== Compatibility Report ===\n")
        
        for message, success in self.checks:
            result = "✅ PASS" if success else "❌ FAIL"
            print(f"{result}: {message}")
        
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"⚠️  {warning}")
        
        print(f"\nOverall result: {'✅ READY' if not self.errors else '❌ NOT READY'}")
        
        if self.errors:
            print("\nPlease fix the above issues before running the trading bot.")
        
        return len(self.errors) == 0


def check_python_version(compatibility):
    """Check if the Python version is compatible."""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    success = current_version >= min_version
    compatibility.add_check(
        f"Python version {'.'.join(map(str, current_version))} (minimum {'.'.join(map(str, min_version))})",
        success
    )
    
    if not success:
        print(f"Current Python version: {'.'.join(map(str, current_version))}")
        print(f"Required Python version: {'.'.join(map(str, min_version))} or higher")


def check_package(compatibility, package_name, min_version=None, import_name=None):
    """Check if a Python package is installed and at the required version."""
    import_name = import_name or package_name
    
    try:
        module = importlib.import_module(import_name)
        
        if min_version:
            # Try to get version in different ways
            version = None
            for attr in ['__version__', 'VERSION', 'version']:
                if hasattr(module, attr):
                    version = getattr(module, attr)
                    if callable(version):
                        version = version()
                    break
            
            if version:
                success = version >= min_version
                compatibility.add_check(
                    f"{package_name} version {version} (minimum {min_version})",
                    success
                )
            else:
                compatibility.add_check(
                    f"{package_name} version check (minimum {min_version})",
                    True
                )
                compatibility.add_warning(f"Could not determine version of {package_name}")
        else:
            compatibility.add_check(f"{package_name} is installed", True)
        
    except ImportError:
        compatibility.add_check(f"{package_name} is installed", False)


def check_directory_structure(compatibility):
    """Check if the expected directory structure exists."""
    expected_dirs = [
        "src/backtesting",
        "src/paper_trading",
        "output/backtests",
        "output/paper_trading",
        "output/evaluation"
    ]
    
    for dir_path in expected_dirs:
        exists = os.path.isdir(dir_path)
        if not exists:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                compatibility.add_check(f"Directory {dir_path} created", True)
            except Exception as e:
                compatibility.add_check(f"Directory {dir_path} exists", False)
                print(f"Error creating directory {dir_path}: {e}")
        else:
            compatibility.add_check(f"Directory {dir_path} exists", True)


def check_api_access(compatibility):
    """Check if API access is properly configured."""
    # Check if .env file exists
    if os.path.isfile(".env"):
        compatibility.add_check(".env file exists", True)
        
        # Check if API keys are configured
        with open(".env", "r") as f:
            content = f.read().lower()
            
        binance_keys = "binance_api_key" in content
        coinbase_keys = "coinbase_api_key" in content
        
        if binance_keys or coinbase_keys:
            compatibility.add_check("Exchange API keys are configured", True)
        else:
            compatibility.add_check("Exchange API keys are configured", False)
            print("Please add your exchange API keys to the .env file")
    else:
        compatibility.add_check(".env file exists", False)
        
        # Create sample .env file
        try:
            with open(".env.example", "r") as example:
                example_content = example.read()
            
            with open(".env", "w") as f:
                f.write(example_content)
            
            compatibility.add_warning(".env file created from example. Please fill in your API keys")
        except Exception as e:
            print(f"Error creating .env file: {e}")


def check_ccxt_exchange_access(compatibility, exchange="binance"):
    """Check if CCXT can access the specified exchange."""
    try:
        import ccxt
        
        # Try to create an exchange instance (without API keys)
        exchange_class = getattr(ccxt, exchange)
        exchange_instance = exchange_class()
        
        # Try to fetch a simple public endpoint
        exchange_instance.load_markets()
        
        compatibility.add_check(f"CCXT can access {exchange.capitalize()} public API", True)
    except Exception as e:
        compatibility.add_check(f"CCXT can access {exchange.capitalize()} public API", False)
        print(f"Error accessing {exchange} API: {e}")


def check_backtrader(compatibility):
    """Check if Backtrader is properly installed and working."""
    try:
        import backtrader as bt
        
        # Create a simple Cerebro instance
        cerebro = bt.Cerebro()
        
        compatibility.add_check("Backtrader is working properly", True)
    except Exception as e:
        compatibility.add_check("Backtrader is working properly", False)
        print(f"Error with Backtrader: {e}")


def check_sentiment_models(compatibility):
    """Check if sentiment analysis models are available."""
    try:
        # Check for NLTK data
        import nltk
        
        try:
            nltk.data.find('tokenizers/punkt')
            compatibility.add_check("NLTK data is installed", True)
        except LookupError:
            compatibility.add_check("NLTK data is installed", False)
            print("Please install NLTK data using nltk.download()")
        
        # Check for VADER
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            sid = SentimentIntensityAnalyzer()
            compatibility.add_check("VADER sentiment analyzer is available", True)
        except Exception as e:
            compatibility.add_check("VADER sentiment analyzer is available", False)
            print(f"Error with VADER: {e}")
        
    except ImportError:
        compatibility.add_check("NLTK is installed", False)


def main():
    """Run the compatibility check."""
    parser = argparse.ArgumentParser(description='Check system compatibility for cryptocurrency trading bot')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix issues')
    args = parser.parse_args()
    
    compatibility = Compatibility()
    
    print("Checking system compatibility...")
    
    # Check Python version
    check_python_version(compatibility)
    
    # Check required packages
    print("\nChecking required packages...")
    packages = [
        ("ccxt", "2.0.0"),
        ("backtrader", "1.9.0"),
        ("pandas", "1.3.0"),
        ("numpy", "1.20.0"),
        ("matplotlib", "3.4.0"),
        ("pyfolio", "0.9.0"),
        ("scipy", "1.7.0"),
        ("scikit-learn", "1.0.0"),
        ("nltk", "3.6.0"),
        ("vaderSentiment", None, "vaderSentiment"),
        ("python-dotenv", "0.19.0", "dotenv")
    ]
    
    for package in packages:
        if len(package) == 2:
            check_package(compatibility, package[0], package[1])
        else:
            check_package(compatibility, package[0], package[1], package[2])
    
    # Check directory structure
    print("\nChecking directory structure...")
    check_directory_structure(compatibility)
    
    # Check API access
    print("\nChecking API access...")
    check_api_access(compatibility)
    
    # Check CCXT exchange access
    print("\nChecking CCXT exchange access...")
    check_ccxt_exchange_access(compatibility, "binance")
    
    # Check Backtrader
    print("\nChecking Backtrader...")
    check_backtrader(compatibility)
    
    # Check sentiment models
    print("\nChecking sentiment models...")
    check_sentiment_models(compatibility)
    
    # Print the compatibility report
    is_compatible = compatibility.print_report()
    
    # Attempt to fix issues if requested
    if args.fix and not is_compatible:
        print("\nAttempting to fix issues...")
        
        # Try to install required packages
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Installed required packages from requirements.txt")
        except Exception as e:
            print(f"❌ Error installing packages: {e}")
        
        # Try to download NLTK data
        try:
            import nltk
            nltk.download('punkt')
            nltk.download('vader_lexicon')
            print("✅ Downloaded NLTK data")
        except Exception as e:
            print(f"❌ Error downloading NLTK data: {e}")
        
        print("\nFixed issues. Please run this script again to verify.")
    
    return 0 if is_compatible else 1


if __name__ == "__main__":
    sys.exit(main()) 