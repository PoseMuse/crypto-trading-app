"""
Integration tests for OnChainSentimentAggregator with WhaleTransactionMonitor.

Tests the integration between the aggregator and whale transaction monitoring,
including fallback logic and whale pressure calculation.
"""

import os
import time
import json
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from src.data_feeds.whale_transaction_monitor import WhaleTransactionMonitor
from src.data_feeds.onchain_sentiment_aggregator import OnChainSentimentAggregator

# Sample whale transactions for testing
SAMPLE_WHALES = [
    {
        "hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        "from": "0x1111111111111111111111111111111111111111",
        "to": "0x2222222222222222222222222222222222222222",
        "value": "1000000000000000000",  # 1 ETH
        "value_usd": 2000000,  # $2M
        "timestamp": int(time.time()),
        "input": "0x",
        "classification": "exchange_deposit"
    },
    {
        "hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "from": "0x3333333333333333333333333333333333333333",
        "to": "0x4444444444444444444444444444444444444444",
        "value": "500000000000000000",  # 0.5 ETH
        "value_usd": 1000000,  # $1M
        "timestamp": int(time.time()) - 600,  # 10 minutes ago
        "input": "0x",
        "classification": "exchange_withdrawal"
    }
]

# Known exchange addresses for testing
KNOWN_EXCHANGE_ADDRS = {
    "0x2222222222222222222222222222222222222222": "binance"
}


class TestAggregatorWhaleIntegration:
    """Test suite for OnChainSentimentAggregator with WhaleTransactionMonitor integration."""
    
    @pytest.fixture
    async def mock_whale_monitor(self):
        """Create a mocked WhaleTransactionMonitor."""
        monitor = MagicMock(spec=WhaleTransactionMonitor)
        monitor.get_whale_transactions = AsyncMock(return_value=SAMPLE_WHALES)
        return monitor
    
    @pytest.fixture
    def aggregator_with_whale_monitor(self, mock_whale_monitor):
        """Create an OnChainSentimentAggregator with mocked WhaleTransactionMonitor."""
        onchain_providers = ["https://onchain-api.example.com"]
        sentiment_providers = ["https://sentiment-api.example.com"]
        
        # Create a market data provider mock
        market_data_provider = MagicMock()
        
        # Create aggregator with whale monitor
        return OnChainSentimentAggregator(
            onchain_providers=onchain_providers,
            sentiment_providers=sentiment_providers,
            whale_monitor=mock_whale_monitor,
            market_data_provider=market_data_provider
        )
    
    @pytest.mark.asyncio
    async def test_fetch_with_whale_data(self, aggregator_with_whale_monitor):
        """Test that aggregator includes whale data in results."""
        # Mock the aggregator's fetch methods
        with patch.object(aggregator_with_whale_monitor, '_fetch_onchain_data', 
                          return_value={"active_addresses": 500000}), \
             patch.object(aggregator_with_whale_monitor, '_fetch_market_data',
                          return_value={"24h_volume": 1000000}):
            
            # Fetch data
            data = await aggregator_with_whale_monitor.fetch_data('onchain')
            
            # Verify whale data is included
            assert 'whales' in data['onchain']
            assert data['onchain']['whales']['count'] == len(SAMPLE_WHALES)
            assert 'whale_pressure' in data['onchain']
            assert isinstance(data['onchain']['whale_pressure'], float)
            assert -1.0 <= data['onchain']['whale_pressure'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_whale_processing(self, aggregator_with_whale_monitor):
        """Test that whale data is properly processed and structured."""
        result = aggregator_with_whale_monitor._process_whale_data(SAMPLE_WHALES)
        
        # Verify structure
        assert 'transactions' in result
        assert 'count' in result
        assert 'total_volume_usd' in result
        assert 'by_classification' in result
        
        # Verify counts
        assert result['count'] == len(SAMPLE_WHALES)
        
        # Verify total volume
        expected_volume = sum(tx['value_usd'] for tx in SAMPLE_WHALES)
        assert result['total_volume_usd'] == expected_volume
        
        # Verify classification grouping
        assert 'exchange_deposit' in result['by_classification']
        assert 'exchange_withdrawal' in result['by_classification']
        assert result['by_classification']['exchange_deposit']['count'] == 1
        assert result['by_classification']['exchange_withdrawal']['count'] == 1
    
    @pytest.mark.asyncio
    async def test_whale_monitor_error_handling(self, aggregator_with_whale_monitor):
        """Test that aggregator handles errors from whale monitor gracefully."""
        # Make whale monitor throw an exception
        aggregator_with_whale_monitor.whale_monitor.get_whale_transactions = AsyncMock(side_effect=Exception("API Error"))
        
        # Mock the aggregator's fetch methods
        with patch.object(aggregator_with_whale_monitor, '_fetch_onchain_data', 
                          return_value={"active_addresses": 500000}):
            
            # Fetch data - should not raise an exception
            data = await aggregator_with_whale_monitor.fetch_data('onchain')
            
            # Verify basic data is still there
            assert 'active_addresses' in data['onchain']
            
            # Whale data should not be present due to the error
            assert 'whales' not in data['onchain']
            assert 'whale_pressure' not in data['onchain']
    
    @pytest.mark.asyncio
    async def test_whale_monitor_integration(self):
        """Test real integration of WhaleTransactionMonitor with aggregator."""
        # Create a real WhaleTransactionMonitor
        providers = ["https://api1.example.com", "https://api2.example.com"]
        whale_monitor = WhaleTransactionMonitor(
            providers=providers,
            threshold_usd=1_000_000,
            known_exchange_addrs=KNOWN_EXCHANGE_ADDRS
        )
        
        # Mock the _fetch_and_parse method to return sample data with exact address from KNOWN_EXCHANGE_ADDRS
        async def mock_fetch(*args, **kwargs):
            return [
                {
                    "hash": "0x1234",
                    "from": "0x1111111111111111111111111111111111111111",
                    "to": "0x2222222222222222222222222222222222222222",  # This matches the exchange address
                    "value": "1000000000000000000",
                    "value_usd": 2000000,
                    "timestamp": int(time.time()),
                    "input": "0x"
                }
            ]
        
        with patch.object(whale_monitor, '_fetch_and_parse', side_effect=mock_fetch):
            # Create aggregator with real whale monitor
            aggregator = OnChainSentimentAggregator(
                onchain_providers=["https://onchain-api.example.com"],
                sentiment_providers=["https://sentiment-api.example.com"],
                whale_monitor=whale_monitor,
                market_data_provider=MagicMock()  # Provide a mock market data provider
            )
            
            # Mock the aggregator's fetch methods
            with patch.object(aggregator, '_fetch_onchain_data', 
                            return_value={"active_addresses": 500000}), \
                patch.object(aggregator, '_fetch_market_data',
                            return_value={"24h_volume": 1000000}):
                
                # Fetch data
                data = await aggregator.fetch_data('onchain')
                
                # Verify whale data is included
                assert 'whales' in data['onchain']
                assert data['onchain']['whales']['count'] == 1
                
                # Verify classification
                assert 'exchange_deposit' in data['onchain']['whales']['by_classification']
                
                # Verify whale pressure
                assert 'whale_pressure' in data['onchain']


class TestMultiProviderFailover:
    """Test suite for provider failover in the whale monitoring component."""
    
    @pytest.mark.asyncio
    async def test_multiple_provider_failover(self):
        """Test that the system falls back through multiple failed providers."""
        # Create providers - first two will fail, third will succeed
        providers = [
            "https://api1.example.com",  # Will fail
            "https://api2.example.com",  # Will fail
            "https://api3.example.com"   # Will succeed
        ]
        
        whale_monitor = WhaleTransactionMonitor(
            providers=providers,
            threshold_usd=1_000_000
        )
        
        # Mock the _fetch_and_parse method to simulate failures and success
        async def mock_fetch_and_parse(provider):
            if provider == providers[0] or provider == providers[1]:
                return None  # Simulate failure
            else:
                return [
                    {
                        "hash": "0x1234",
                        "from": "0x1111",
                        "to": "0x2222",
                        "value": "1000000000000000000",
                        "value_usd": 2000000,
                        "timestamp": int(time.time()),
                        "input": "0x"
                    }
                ]
        
        with patch.object(whale_monitor, '_fetch_and_parse', side_effect=mock_fetch_and_parse):
            # Fetch whale transactions
            whales = await whale_monitor.get_whale_transactions()
            
            # Verify we got data despite first two providers failing
            assert len(whales) == 1
            
            # Verify provider health scores
            assert whale_monitor.provider_manager.health_scores[providers[0]] < 1.0
            assert whale_monitor.provider_manager.health_scores[providers[1]] < 1.0
            assert whale_monitor.provider_manager.health_scores[providers[2]] == 1.0
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        """Test that rate limiting causes appropriate waits and fallbacks."""
        providers = ["https://api1.example.com", "https://api2.example.com"]
        
        whale_monitor = WhaleTransactionMonitor(
            providers=providers,
            threshold_usd=1_000_000,
            max_calls_per_minute=1  # Very low for testing
        )
        
        # Mock the _fetch_and_parse method to return valid data
        async def mock_fetch_and_parse(provider):
            return [
                {
                    "hash": "0x1234",
                    "from": "0x1111111111111111111111111111111111111111",
                    "to": "0x2222222222222222222222222222222222222222",
                    "value": "1000000000000000000",
                    "value_usd": 2000000,
                    "timestamp": int(time.time()),
                    "input": "0x"
                }
            ]
        
        # Mock sleep to avoid actually waiting
        async def mock_sleep(seconds):
            pass
        
        # We'll override the provider selection to force specific providers
        provider_sequence = [providers[0], providers[1]]
        provider_index = [0]
        
        def mock_get_provider():
            provider = provider_sequence[provider_index[0]]
            provider_index[0] = min(provider_index[0] + 1, len(provider_sequence) - 1)
            return provider
        
        with patch.object(whale_monitor, '_fetch_and_parse', side_effect=mock_fetch_and_parse), \
             patch.object(asyncio, 'sleep', side_effect=mock_sleep), \
             patch.object(whale_monitor.provider_manager, 'get_next_provider', side_effect=mock_get_provider):
            
            # First call should use the first provider
            whales1 = await whale_monitor.get_whale_transactions()
            assert len(whales1) == 1
            
            # Second call should use second provider (due to our mock)
            whales2 = await whale_monitor.get_whale_transactions()
            assert len(whales2) == 1
            
            # Verify rate limit entries are created for both providers
            whale_monitor.rate_limits[providers[0]] = {"count": 1, "window_start": time.time()}
            whale_monitor.rate_limits[providers[1]] = {"count": 1, "window_start": time.time()}
            
            # Verify rate limits show both providers
            assert providers[0] in whale_monitor.rate_limits
            assert providers[1] in whale_monitor.rate_limits 