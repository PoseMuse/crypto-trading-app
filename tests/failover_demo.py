import sys
import os
import time
import ccxt
from unittest import mock

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchange_manager import ExchangeManager

def main():
    """Demo the failover functionality with real exchanges."""
    
    print("Initializing Exchange Manager...")
    manager = ExchangeManager("binance", test_mode=True)
    
    # Test 1: Normal operation
    print("\n--- Test 1: Normal API Call ---")
    try:
        ticker = manager.fetch_ticker("BTC/USDT")
        print(f"Successfully fetched BTC/USDT ticker from Binance")
        print(f"Price: ${ticker['last']:.2f}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Simulated primary exchange failure with mock
    print("\n--- Test 2: Failover Test (Simulated) ---")
    
    class CustomException(Exception):
        pass
    
    # Create a mock for binance exchange that will fail
    with mock.patch('ccxt.binance') as mock_binance:
        # Setup the mock to throw an exception when fetch_ticker is called
        mock_exchange = mock.MagicMock()
        mock_exchange.fetch_ticker.side_effect = CustomException("Simulated exchange failure")
        mock_binance.return_value = mock_exchange
        
        # Test failover
        print("Simulating primary exchange (Binance) failure...")
        
        try:
            result = manager.attempt_failover("binance", "kucoin", "fetch_ticker", "BTC/USDT")
            print("Failover successful!")
            print(f"BTC/USDT price from backup exchange: ${result['last']:.2f}")
        except Exception as e:
            print(f"Failover failed: {e}")
    
    # Test 3: Real rate limit handling
    print("\n--- Test 3: Rate Limit Handling ---")
    print("Making multiple rapid requests to trigger rate limiting...")
    
    start_time = time.time()
    
    for i in range(10):
        try:
            ticker = manager.fetch_ticker("BTC/USDT")
            print(f"Request {i+1}: Success - BTC/USDT @ ${ticker['last']:.2f}")
        except ccxt.RateLimitExceeded:
            print(f"Request {i+1}: Rate limit hit")
        except Exception as e:
            print(f"Request {i+1}: Error - {e}")
        
        # No delay between requests to try to trigger rate limiting
    
    duration = time.time() - start_time
    print(f"Completed 10 requests in {duration:.2f} seconds")

if __name__ == "__main__":
    main() 