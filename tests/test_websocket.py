import sys
import os
import time
import pytest
import threading
from unittest import mock
from queue import Queue

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchange_manager import ExchangeManager
import ccxt

class TestWebSocket:
    """Test case for WebSocket functionality in the ExchangeManager class."""
    
    def test_subscribe_ticker_stream(self):
        """Test subscribing to a ticker stream."""
        with mock.patch('websocket.WebSocketApp') as mock_websocket:
            # Set up the mock
            mock_ws = mock.MagicMock()
            mock_websocket.return_value = mock_ws
            
            # Set up a mock thread
            with mock.patch('threading.Thread') as mock_thread:
                mock_thread_instance = mock.MagicMock()
                mock_thread.return_value = mock_thread_instance
                
                # Create manager and subscribe to ticker
                manager = ExchangeManager("binance", test_mode=True)
                
                # Create a callback function
                callback_data = []
                def callback(data):
                    callback_data.append(data)
                
                # Subscribe to ticker
                connection_id = manager.subscribe_ticker_stream("BTC/USDT", callback)
                
                # Check if WebSocketApp was created with the correct URL
                assert mock_websocket.call_count == 1
                call_args = mock_websocket.call_args[0]
                assert "btcusdt@ticker" in call_args[0]
                
                # Check if thread was started
                assert mock_thread_instance.start.call_count == 1
                
                # Check if connection was stored
                assert connection_id in manager.websocket_connections
                assert manager.websocket_connections[connection_id]['symbol'] == "BTC/USDT"
                assert manager.websocket_connections[connection_id]['active'] is True
    
    def test_unsubscribe_ticker_stream(self):
        """Test unsubscribing from a ticker stream."""
        with mock.patch('websocket.WebSocketApp') as mock_websocket:
            # Set up the mock
            mock_ws = mock.MagicMock()
            mock_websocket.return_value = mock_ws
            
            # Mock the socket connection status
            mock_ws.sock = mock.MagicMock()
            mock_ws.sock.connected = True
            
            # Create manager and subscribe to ticker
            manager = ExchangeManager("binance", test_mode=True)
            
            # Create a callback function
            def callback(data):
                pass
            
            # Subscribe to ticker
            with mock.patch('threading.Thread'):
                connection_id = manager.subscribe_ticker_stream("BTC/USDT", callback)
            
            # Check if connection was stored
            assert connection_id in manager.websocket_connections
            
            # Unsubscribe
            manager.unsubscribe_ticker_stream(connection_id)
            
            # Check if close was called
            assert mock_ws.close.call_count == 1
            
            # Check if connection was removed
            assert connection_id not in manager.websocket_connections
    
    def test_websocket_binance_message_handling(self):
        """Test handling of Binance WebSocket messages."""
        with mock.patch('websocket.WebSocketApp') as mock_websocket:
            # Create the manager
            manager = ExchangeManager("binance", test_mode=True)
            
            # Set up a callback queue to collect data
            data_queue = Queue()
            def callback(data):
                data_queue.put(data)
            
            # Set up a mock WebSocket
            mock_ws = mock.MagicMock()
            mock_websocket.return_value = mock_ws
            
            # Subscribe to ticker
            with mock.patch('threading.Thread'):
                connection_id = manager.subscribe_ticker_stream("BTC/USDT", callback)
            
            # Get the on_message handler
            on_message = mock_websocket.call_args[1]['on_message']
            
            # Simulate a message
            test_message = '''{
                "e": "24hrTicker",
                "E": 123456789,
                "s": "BTCUSDT",
                "p": "500.0",
                "P": "5.0",
                "w": "40000.0",
                "c": "40500.0",
                "Q": "1.0",
                "o": "40000.0",
                "h": "41000.0",
                "l": "39500.0",
                "v": "1000.0",
                "q": "40000000.0",
                "O": 0,
                "C": 86400000,
                "F": 0,
                "L": 100,
                "n": 100,
                "b": "40490.0",
                "a": "40510.0"
            }'''
            
            # Call the on_message handler
            on_message(mock_ws, test_message)
            
            # Check if callback was called with standardized data
            assert not data_queue.empty()
            data = data_queue.get(timeout=1)
            
            # Verify standardized data
            assert data['symbol'] == "BTC/USDT"
            assert data['last'] == 40500.0
            assert data['bid'] == 40490.0
            assert data['ask'] == 40510.0
            assert data['high'] == 41000.0
            assert data['low'] == 39500.0
            assert data['volume'] == 1000.0
            assert data['change'] == 500.0
            assert data['percentage'] == 5.0
    
    def test_websocket_reconnection(self):
        """Test WebSocket reconnection on error."""
        with mock.patch('websocket.WebSocketApp') as mock_websocket:
            # Create the manager
            manager = ExchangeManager("binance", test_mode=True)
            
            # Create a callback function
            def callback(data):
                pass
            
            # Subscribe to ticker
            with mock.patch('threading.Thread'):
                connection_id = manager.subscribe_ticker_stream("BTC/USDT", callback)
            
            # Get the on_error handler
            on_error = mock_websocket.call_args[1]['on_error']
            
            # Mock the reconnect method
            with mock.patch.object(manager, '_reconnect_websocket') as mock_reconnect:
                # Simulate an error
                on_error(mock_websocket.return_value, Exception("Test error"))
                
                # Check if reconnect was called
                assert mock_reconnect.call_count == 1
                assert mock_reconnect.call_args[0][0] == connection_id
    
    def test_websocket_close_handling(self):
        """Test handling of WebSocket close events."""
        with mock.patch('websocket.WebSocketApp') as mock_websocket:
            # Create the manager
            manager = ExchangeManager("binance", test_mode=True)
            
            # Create a callback function
            def callback(data):
                pass
            
            # Subscribe to ticker
            with mock.patch('threading.Thread'):
                connection_id = manager.subscribe_ticker_stream("BTC/USDT", callback)
            
            # Get the on_close handler
            on_close = mock_websocket.call_args[1]['on_close']
            
            # Mock the reconnect method
            with mock.patch.object(manager, '_reconnect_websocket') as mock_reconnect:
                # Simulate a close event
                on_close(mock_websocket.return_value, 1000, "Normal closure")
                
                # Check if reconnect was called
                assert mock_reconnect.call_count == 1
                assert mock_reconnect.call_args[0][0] == connection_id
    
    @pytest.mark.skip(reason="This test requires integration with the real API")
    def test_real_websocket_connection(self):
        """Test connecting to a real WebSocket (integration test)."""
        # Create the manager
        manager = ExchangeManager("binance", test_mode=True)
        
        # Set up a data queue
        data_queue = Queue()
        def callback(data):
            data_queue.put(data)
            print(f"Received data: {data}")
        
        print("Subscribing to BTC/USDT ticker stream...")
        connection_id = manager.subscribe_ticker_stream("BTC/USDT", callback)
        
        # Wait for 10 seconds to receive some data
        max_wait = 10
        start_time = time.time()
        
        try:
            while time.time() - start_time < max_wait:
                if not data_queue.empty():
                    data = data_queue.get(timeout=1)
                    print(f"Received ticker: {data}")
                    break
                time.sleep(0.5)
            
            assert not data_queue.empty(), "No data received from WebSocket"
        finally:
            # Clean up
            manager.unsubscribe_ticker_stream(connection_id)

class TestRateLimitHandling:
    """Test case for rate limit handling in the ExchangeManager class."""
    
    def test_rate_limit_detection(self):
        """Test detection of rate limit errors."""
        with mock.patch('ccxt.binance') as mock_binance:
            # Setup mock
            mock_exchange = mock.MagicMock()
            mock_exchange.fetch_ticker.side_effect = ccxt.RateLimitExceeded("Rate limit exceeded")
            mock_binance.return_value = mock_exchange
            
            # Create manager
            manager = ExchangeManager("binance", test_mode=True)
            
            # Mock the _handle_rate_limit_error method
            with mock.patch.object(manager, '_handle_rate_limit_error') as mock_handler:
                # Call method that should raise rate limit error
                with pytest.raises(ccxt.RateLimitExceeded):
                    manager.fetch_ticker("BTC/USDT")
                
                # Check if handler was called
                assert mock_handler.call_count == 1
    
    def test_rate_limit_backoff(self):
        """Test exponential backoff on rate limit errors."""
        # Create manager
        manager = ExchangeManager("binance", test_mode=True)
        
        # Initial backoff should be 1 second
        assert manager.rate_limit_tracker['backoff_time'] == 1
        
        # Mock time.time to return increasing values for each call
        with mock.patch('time.time') as mock_time, mock.patch('time.sleep') as mock_sleep:
            # First call to time.time - initial time
            mock_time.return_value = 1000
            
            # First rate limit error
            manager._handle_rate_limit_error()
            
            # Check if sleep was called with correct backoff time
            assert mock_sleep.call_count == 1
            assert mock_sleep.call_args[0][0] == 1
            assert manager.rate_limit_tracker['backoff_time'] == 2  # Should be doubled for next time
            
            # Second call to time.time - 2 seconds later (above the 1-second threshold)
            mock_time.return_value = 1002
            
            # Second rate limit error
            manager._handle_rate_limit_error()
            
            # Check if sleep was called with the new backoff time
            assert mock_sleep.call_count == 2
            assert mock_sleep.call_args[0][0] == 2
            assert manager.rate_limit_tracker['backoff_time'] == 4  # Should be doubled again
            
            # Third call to time.time - 2 seconds later
            mock_time.return_value = 1004
            
            # Third rate limit error
            manager._handle_rate_limit_error()
            
            # Check if sleep was called with the new backoff time
            assert mock_sleep.call_count == 3
            assert mock_sleep.call_args[0][0] == 4
            assert manager.rate_limit_tracker['backoff_time'] == 8  # Should be doubled again

class TestFailover:
    """Test case for failover functionality in the ExchangeManager class."""
    
    def test_failover_mechanism(self):
        """Test failover from primary to backup exchange."""
        # Create a mock for the primary exchange that fails
        with mock.patch('ccxt.binance') as mock_binance:
            mock_primary = mock.MagicMock()
            mock_primary.fetch_ticker.side_effect = Exception("Primary exchange failed")
            mock_binance.return_value = mock_primary
            
            # Create a mock for the backup exchange that works
            with mock.patch('ccxt.kucoin') as mock_kucoin:
                mock_backup = mock.MagicMock()
                mock_backup.fetch_ticker.return_value = {
                    'symbol': 'BTC/USDT',
                    'last': 40000.0
                }
                mock_kucoin.return_value = mock_backup
                
                # Create a manager to test failover
                manager = ExchangeManager("binance", test_mode=True)
                
                # Test the failover
                result = manager.attempt_failover("binance", "kucoin", "fetch_ticker", "BTC/USDT")
                
                # Check if the backup exchange was used
                assert result['symbol'] == 'BTC/USDT'
                assert result['last'] == 40000.0
                assert mock_primary.fetch_ticker.call_count == 1
                assert mock_backup.fetch_ticker.call_count == 1
    
    def test_both_exchanges_fail(self):
        """Test behavior when both primary and backup exchanges fail."""
        # Create mocks for both exchanges that fail
        with mock.patch('ccxt.binance') as mock_binance:
            mock_primary = mock.MagicMock()
            mock_primary.fetch_ticker.side_effect = Exception("Primary exchange failed")
            mock_binance.return_value = mock_primary
            
            with mock.patch('ccxt.kucoin') as mock_kucoin:
                mock_backup = mock.MagicMock()
                mock_backup.fetch_ticker.side_effect = Exception("Backup exchange failed")
                mock_kucoin.return_value = mock_backup
                
                # Create a manager to test failover
                manager = ExchangeManager("binance", test_mode=True)
                
                # Test the failover
                with pytest.raises(Exception) as excinfo:
                    manager.attempt_failover("binance", "kucoin", "fetch_ticker", "BTC/USDT")
                
                # Check the exception message
                assert "Both exchanges failed" in str(excinfo.value)
                assert "Primary error" in str(excinfo.value)
                assert "Backup error" in str(excinfo.value)


if __name__ == "__main__":
    # Run integration test manually
    test_ws = TestWebSocket()
    test_ws.test_real_websocket_connection() 