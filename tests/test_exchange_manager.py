import sys
import os
from unittest import mock
import pytest

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchange_manager import ExchangeManager

def test_exchange_manager_initialization():
    """Test that the ExchangeManager initializes correctly."""
    # Test with valid exchange
    manager = ExchangeManager("binance", test_mode=True)
    assert manager.exchange_name == "binance"
    assert manager.exchange is not None
    
    # Test with invalid exchange
    with pytest.raises(ValueError):
        ExchangeManager("not_a_real_exchange")

@mock.patch('ccxt.binance')
def test_fetch_ticker(mock_binance):
    """Test fetch_ticker method with mocked exchange."""
    # Setup mock
    mock_exchange = mock.MagicMock()
    mock_exchange.fetch_ticker.return_value = {
        'symbol': 'BTC/USDT',
        'last': 40000.0,
        'bid': 39999.0,
        'ask': 40001.0,
        'volume': 1000.0
    }
    mock_binance.return_value = mock_exchange
    
    # Create manager and test
    manager = ExchangeManager("binance", test_mode=True)
    ticker = manager.fetch_ticker("BTC/USDT")
    
    # Assert
    assert ticker['symbol'] == 'BTC/USDT'
    assert ticker['last'] == 40000.0
    mock_exchange.fetch_ticker.assert_called_once_with("BTC/USDT")

@mock.patch('ccxt.binance')
def test_fetch_balance(mock_binance):
    """Test fetch_balance method with mocked exchange."""
    # Setup mock
    mock_exchange = mock.MagicMock()
    mock_exchange.fetch_balance.return_value = {
        'free': {'BTC': 1.0, 'USDT': 50000.0},
        'used': {'BTC': 0.1, 'USDT': 5000.0},
        'total': {'BTC': 1.1, 'USDT': 55000.0}
    }
    mock_binance.return_value = mock_exchange
    
    # Create manager and test
    manager = ExchangeManager("binance", test_mode=True)
    balance = manager.fetch_balance()
    
    # Assert
    assert balance['free']['BTC'] == 1.0
    assert balance['total']['USDT'] == 55000.0
    mock_exchange.fetch_balance.assert_called_once()

@mock.patch('ccxt.binance')
def test_create_order(mock_binance):
    """Test create_order method with mocked exchange."""
    # Setup mock
    mock_exchange = mock.MagicMock()
    mock_exchange.create_order.return_value = {
        'id': '123456',
        'symbol': 'BTC/USDT',
        'type': 'limit',
        'side': 'buy',
        'price': 40000.0,
        'amount': 0.1,
        'status': 'open'
    }
    mock_binance.return_value = mock_exchange
    
    # Create manager and test
    manager = ExchangeManager("binance", test_mode=True)
    order = manager.create_order(
        symbol="BTC/USDT", 
        order_type="limit", 
        side="buy", 
        amount=0.1, 
        price=40000.0
    )
    
    # Assert
    assert order['id'] == '123456'
    assert order['status'] == 'open'
    mock_exchange.create_order.assert_called_once_with(
        "BTC/USDT", "limit", "buy", 0.1, 40000.0
    )

def test_real_ticker_fetch():
    """
    Test fetching a real ticker from Binance (testnet).
    This test actually connects to the exchange.
    """
    manager = ExchangeManager("binance", test_mode=True)
    
    try:
        ticker = manager.fetch_ticker("BTC/USDT")
        print(f"\nBTC/USDT Ticker: {ticker}")
        # Basic validation that we received a ticker
        assert 'symbol' in ticker
        assert ticker['symbol'] == 'BTC/USDT'
        assert 'last' in ticker
    except Exception as e:
        pytest.skip(f"Skipping real API test due to error: {e}")

if __name__ == "__main__":
    # For manual testing
    test_real_ticker_fetch() 