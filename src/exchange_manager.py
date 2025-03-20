import ccxt
import time
import threading
import json
import websocket
import logging
from typing import Dict, Optional, Any, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('exchange_manager')

class ExchangeManager:
    """
    Manages interactions with cryptocurrency exchanges using ccxt.
    
    Provides a unified interface for common exchange operations like fetching
    market data and executing trades across different exchanges.
    """
    
    def __init__(
        self, 
        exchange_name: str, 
        api_key: str = "YOUR_API_KEY", 
        api_secret: str = "YOUR_API_SECRET",
        test_mode: bool = False
    ):
        """
        Initialize the exchange manager with given credentials.
        
        Args:
            exchange_name: Name of the exchange (e.g., 'binance', 'bybit')
            api_key: API key for the exchange (default: placeholder)
            api_secret: API secret for the exchange (default: placeholder)
            test_mode: Whether to use test/sandbox mode (default: False)
        """
        # Check if exchange is supported by ccxt
        if not hasattr(ccxt, exchange_name):
            raise ValueError(f"Exchange '{exchange_name}' is not supported by ccxt")
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        
        # Set test mode if requested
        if test_mode:
            try:
                self.exchange.set_sandbox_mode(True)
                logger.info(f"Sandbox mode enabled for {exchange_name}")
            except Exception as e:
                logger.warning(f"Failed to set sandbox mode for {exchange_name}: {e}")
                logger.warning("Continuing without sandbox mode")
        
        self.exchange_name = exchange_name
        self.websocket_connections = {}
        self.rate_limit_tracker = {
            'last_rate_limit_error': 0,
            'backoff_time': 1,  # Initial backoff of 1 second
            'next_backoff_time': 2  # The next backoff time after a rate limit hit
        }
    
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker information for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary containing ticker information
            
        Raises:
            Various exceptions from ccxt if the request fails
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except ccxt.RateLimitExceeded as e:
            self._handle_rate_limit_error()
            logger.error(f"Rate limit exceeded when fetching ticker for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise
    
    def fetch_balance(self) -> Dict[str, Any]:
        """
        Fetch account balance information.
        
        Returns:
            Dictionary containing balance information
            
        Raises:
            Various exceptions from ccxt if the request fails
        """
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except ccxt.RateLimitExceeded as e:
            self._handle_rate_limit_error()
            logger.error(f"Rate limit exceeded when fetching balance: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            raise
    
    def create_order(
        self, 
        symbol: str, 
        order_type: str, 
        side: str, 
        amount: float, 
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Create a new order on the exchange.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            order_type: Type of order ('limit', 'market')
            side: Order side ('buy', 'sell')
            amount: Order amount in base currency
            price: Order price (required for limit orders)
            
        Returns:
            Dictionary containing order information
            
        Raises:
            Various exceptions from ccxt if the request fails
        """
        try:
            order = self.exchange.create_order(symbol, order_type, side, amount, price)
            return order
        except ccxt.RateLimitExceeded as e:
            self._handle_rate_limit_error()
            logger.error(f"Rate limit exceeded when creating {side} {order_type} order for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating {side} {order_type} order for {symbol}: {e}")
            raise
    
    def get_supported_symbols(self) -> list:
        """
        Get list of trading pairs supported by the exchange.
        
        Returns:
            List of symbol strings
        """
        try:
            markets = self.exchange.load_markets()
            return list(markets.keys())
        except ccxt.RateLimitExceeded as e:
            self._handle_rate_limit_error()
            logger.error(f"Rate limit exceeded when loading markets: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading markets: {e}")
            return []
    
    def _handle_rate_limit_error(self):
        """Handle rate limit errors with exponential backoff."""
        now = time.time()
        
        # Get current backoff time
        backoff_time = self.rate_limit_tracker['backoff_time']
        
        # Update backoff time if this is a new rate limit error (not within 1 second of the last one)
        if now - self.rate_limit_tracker['last_rate_limit_error'] > 1:
            # For next time, use the next backoff time value
            self.rate_limit_tracker['backoff_time'] = self.rate_limit_tracker['next_backoff_time']
            # Calculate the next-next backoff time using exponential growth, capped at 60 seconds
            self.rate_limit_tracker['next_backoff_time'] = min(
                self.rate_limit_tracker['backoff_time'] * 2,
                60  # Cap at 60 seconds
            )
        
        self.rate_limit_tracker['last_rate_limit_error'] = now
        
        # Sleep to respect the rate limit
        logger.warning(f"Rate limit hit. Backing off for {backoff_time} seconds")
        time.sleep(backoff_time)
    
    # WebSocket methods
    
    def subscribe_ticker_stream(self, symbol: str, callback: Callable[[Dict[str, Any]], None]) -> str:
        """
        Subscribe to a real-time ticker stream for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            callback: Function to call with ticker data when received
            
        Returns:
            Connection ID string that can be used to unsubscribe
        """
        if self.exchange_name.lower() == 'binance':
            return self._subscribe_binance_ticker(symbol, callback)
        elif self.exchange_name.lower() == 'bybit':
            return self._subscribe_bybit_ticker(symbol, callback)
        else:
            raise NotImplementedError(f"WebSocket support for {self.exchange_name} not implemented")
    
    def unsubscribe_ticker_stream(self, connection_id: str):
        """
        Unsubscribe from a ticker stream.
        
        Args:
            connection_id: The connection ID returned by subscribe_ticker_stream
        """
        if connection_id in self.websocket_connections:
            ws = self.websocket_connections[connection_id]['ws']
            if ws and ws.sock and ws.sock.connected:
                ws.close()
            del self.websocket_connections[connection_id]
            logger.info(f"Unsubscribed from ticker stream: {connection_id}")
    
    def _subscribe_binance_ticker(self, symbol: str, callback: Callable[[Dict[str, Any]], None]) -> str:
        """
        Subscribe to Binance ticker WebSocket.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            callback: Function to call with ticker data when received
            
        Returns:
            Connection ID string
        """
        # Convert CCXT symbol format to Binance format (lowercase and remove '/')
        binance_symbol = symbol.lower().replace('/', '')
        
        # Determine WebSocket URL based on test mode
        base_url = "wss://testnet.binance.vision/ws/" if hasattr(self.exchange, 'urls') and 'test' in self.exchange.urls else "wss://stream.binance.com:9443/ws/"
        
        ws_url = f"{base_url}{binance_symbol}@ticker"
        connection_id = f"binance_ticker_{binance_symbol}_{int(time.time())}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                # Convert Binance format to a standardized format similar to CCXT ticker
                standardized_data = {
                    'symbol': symbol,
                    'last': float(data.get('c', 0)),
                    'bid': float(data.get('b', 0)),
                    'ask': float(data.get('a', 0)),
                    'volume': float(data.get('v', 0)),
                    'timestamp': int(data.get('E', 0)),
                    'high': float(data.get('h', 0)),
                    'low': float(data.get('l', 0)),
                    'open': float(data.get('o', 0)),
                    'close': float(data.get('c', 0)),
                    'change': float(data.get('p', 0)),
                    'percentage': float(data.get('P', 0)),
                    'raw': data  # Include the raw data
                }
                callback(standardized_data)
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            # Try to reconnect
            self._reconnect_websocket(connection_id)
        
        def on_close(ws, close_status_code, close_reason):
            logger.info(f"WebSocket closed: {close_status_code} - {close_reason}")
            # Try to reconnect if not intentionally closed
            if connection_id in self.websocket_connections and self.websocket_connections[connection_id]['active']:
                self._reconnect_websocket(connection_id)
        
        def on_open(ws):
            logger.info(f"WebSocket connection opened: {ws_url}")
        
        # Create and store the WebSocket connection
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Store connection details
        self.websocket_connections[connection_id] = {
            'ws': ws,
            'url': ws_url,
            'symbol': symbol,
            'callback': callback,
            'active': True,
            'type': 'binance_ticker'
        }
        
        # Start WebSocket connection in a separate thread
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
        
        return connection_id
    
    def _subscribe_bybit_ticker(self, symbol: str, callback: Callable[[Dict[str, Any]], None]) -> str:
        """
        Subscribe to Bybit ticker WebSocket.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            callback: Function to call with ticker data when received
            
        Returns:
            Connection ID string
        """
        # Convert CCXT symbol format to Bybit format (remove '/')
        bybit_symbol = symbol.replace('/', '')
        
        # Determine WebSocket URL based on test mode
        base_url = "wss://stream-testnet.bybit.com/v5/public/spot" if hasattr(self.exchange, 'urls') and 'test' in self.exchange.urls else "wss://stream.bybit.com/v5/public/spot"
        
        ws_url = base_url
        connection_id = f"bybit_ticker_{bybit_symbol}_{int(time.time())}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                
                # Check if it's a ticker message for our symbol
                if 'topic' in data and data['topic'] == f"tickers.{bybit_symbol}":
                    ticker_data = data.get('data', {})
                    # Convert Bybit format to a standardized format similar to CCXT ticker
                    standardized_data = {
                        'symbol': symbol,
                        'last': float(ticker_data.get('lastPrice', 0)),
                        'bid': float(ticker_data.get('bid1Price', 0)),
                        'ask': float(ticker_data.get('ask1Price', 0)),
                        'volume': float(ticker_data.get('volume24h', 0)),
                        'timestamp': int(ticker_data.get('timestamp', 0)),
                        'high': float(ticker_data.get('highPrice24h', 0)),
                        'low': float(ticker_data.get('lowPrice24h', 0)),
                        'change': 0,  # Bybit doesn't provide this directly
                        'percentage': float(ticker_data.get('price24hPcnt', 0)) * 100,
                        'raw': data  # Include the raw data
                    }
                    callback(standardized_data)
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            # Try to reconnect
            self._reconnect_websocket(connection_id)
        
        def on_close(ws, close_status_code, close_reason):
            logger.info(f"WebSocket closed: {close_status_code} - {close_reason}")
            # Try to reconnect if not intentionally closed
            if connection_id in self.websocket_connections and self.websocket_connections[connection_id]['active']:
                self._reconnect_websocket(connection_id)
        
        def on_open(ws):
            logger.info(f"WebSocket connection opened: {ws_url}")
            # Subscribe to ticker topic
            subscribe_msg = {
                "op": "subscribe",
                "args": [f"tickers.{bybit_symbol}"]
            }
            ws.send(json.dumps(subscribe_msg))
        
        # Create and store the WebSocket connection
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Store connection details
        self.websocket_connections[connection_id] = {
            'ws': ws,
            'url': ws_url,
            'symbol': symbol,
            'callback': callback,
            'active': True,
            'type': 'bybit_ticker'
        }
        
        # Start WebSocket connection in a separate thread
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
        
        return connection_id
    
    def _reconnect_websocket(self, connection_id: str, max_retries: int = 5, backoff_factor: float = 1.5):
        """
        Attempt to reconnect a WebSocket connection with exponential backoff.
        
        Args:
            connection_id: The connection ID to reconnect
            max_retries: Maximum number of reconnection attempts
            backoff_factor: Multiplier for backoff time between retries
        """
        if connection_id not in self.websocket_connections:
            logger.error(f"Cannot reconnect unknown connection: {connection_id}")
            return
        
        connection = self.websocket_connections[connection_id]
        
        # Set retries counter if it doesn't exist
        if 'retries' not in connection:
            connection['retries'] = 0
        
        # Check if we've exceeded max retries
        if connection['retries'] >= max_retries:
            logger.error(f"Maximum reconnection attempts reached for {connection_id}. Giving up.")
            connection['active'] = False
            return
        
        # Calculate backoff time
        backoff_time = (backoff_factor ** connection['retries'])
        connection['retries'] += 1
        
        logger.info(f"Attempting to reconnect {connection_id} in {backoff_time:.2f} seconds (attempt {connection['retries']})")
        
        # Schedule reconnection
        time.sleep(backoff_time)
        
        # Recreate the connection based on its type
        if connection['type'] == 'binance_ticker':
            self._subscribe_binance_ticker(connection['symbol'], connection['callback'])
        elif connection['type'] == 'bybit_ticker':
            self._subscribe_bybit_ticker(connection['symbol'], connection['callback'])
        else:
            logger.warning(f"Unknown WebSocket connection type: {connection['type']}")
    
    def attempt_failover(self, primary_exchange_name: str, backup_exchange_name: str, method_name: str, *args, **kwargs):
        """
        Attempt to execute a method on a primary exchange, falling back to a backup exchange if it fails.
        
        Args:
            primary_exchange_name: Name of primary exchange
            backup_exchange_name: Name of backup exchange
            method_name: Name of the method to call on the exchange manager
            *args, **kwargs: Arguments to pass to the method
            
        Returns:
            Result of the method call
            
        Raises:
            Exception if both exchanges fail
        """
        # Create exchange managers
        primary_manager = ExchangeManager(primary_exchange_name, test_mode=True)
        backup_manager = ExchangeManager(backup_exchange_name, test_mode=True)
        
        try:
            # Try primary exchange
            logger.info(f"Attempting method {method_name} on primary exchange {primary_exchange_name}")
            method = getattr(primary_manager, method_name)
            return method(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary exchange {primary_exchange_name} failed: {e}")
            
            try:
                # Try backup exchange
                logger.info(f"Attempting method {method_name} on backup exchange {backup_exchange_name}")
                method = getattr(backup_manager, method_name)
                return method(*args, **kwargs)
            except Exception as backup_e:
                logger.error(f"Backup exchange {backup_exchange_name} also failed: {backup_e}")
                raise Exception(f"Both exchanges failed. Primary error: {e}, Backup error: {backup_e}") 