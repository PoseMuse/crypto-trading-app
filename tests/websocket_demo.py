import sys
import os
import time

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchange_manager import ExchangeManager

def main():
    """Demo the WebSocket functionality with a real exchange."""
    
    print("Initializing Exchange Manager for Binance...")
    manager = ExchangeManager("binance", test_mode=True)
    
    # Data received counter
    counter = 0
    
    def ticker_callback(data):
        nonlocal counter
        counter += 1
        print(f"\rReceived ticker {counter}: BTC/USDT @ {data['last']:.2f} USD", end="")
    
    try:
        print("Subscribing to BTC/USDT ticker stream...")
        connection_id = manager.subscribe_ticker_stream("BTC/USDT", ticker_callback)
        
        print("WebSocket connection established.")
        print("Listening for 30 seconds. Press Ctrl+C to stop early.")
        
        # Listen for 30 seconds
        start_time = time.time()
        while time.time() - start_time < 30:
            time.sleep(0.1)
            
        print("\n\nTest complete.")
        print(f"Received {counter} ticker updates in 30 seconds.")
        
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Clean up
        print("Unsubscribing from ticker stream...")
        if 'connection_id' in locals():
            manager.unsubscribe_ticker_stream(connection_id)
        print("Done.")

if __name__ == "__main__":
    main() 