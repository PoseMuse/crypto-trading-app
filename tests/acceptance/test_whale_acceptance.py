"""
Acceptance tests for whale transaction monitoring integration.

These tests validate that:
1. The WhaleTransactionMonitor successfully retrieves and classifies transactions
2. The OnChainSentimentAggregator correctly integrates whale data
3. The system handles multiple provider failures with graceful fallback
4. Rate limiting works properly under high load
5. Whale pressure value is correctly computed and reported
"""

import os
import time
import json
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from src.data_feeds.whale_transaction_monitor import WhaleTransactionMonitor, compute_whale_influence
from src.data_feeds.onchain_sentiment_aggregator import OnChainSentimentAggregator

# Dictionary of known exchange addresses
EXCHANGE_ADDRESSES = {
    # Binance deposit addresses
    "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE": "binance",
    "0xD551234Ae421e3BCBA99A0Da6d736074f22192FF": "binance",
    # Coinbase addresses
    "0x71660c4005BA85c37CCec55d0C4493E66Fe775d3": "coinbase",
    # Kraken addresses
    "0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2": "kraken"
}

# Dictionary of known DeFi protocol addresses
DEFI_ADDRESSES = {
    # Uniswap v2 Router
    "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D": "uniswap_v2",
    # Aave Lending Pool
    "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9": "aave_v2",
    # Compound
    "0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B": "compound"
}

# Sample whale transaction data
WHALE_TRANSACTIONS = [
    {
        "hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        "from": "0x1111111111111111111111111111111111111111",
        "to": "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",  # Binance deposit
        "value": "1000000000000000000",  # 1 ETH
        "value_usd": 2000000,  # $2M
        "timestamp": int(time.time()),
        "input": "0x",
    },
    {
        "hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "from": "0x71660c4005BA85c37CCec55d0C4493E66Fe775d3",  # Coinbase withdrawal
        "to": "0x2222222222222222222222222222222222222222",
        "value": "500000000000000000",  # 0.5 ETH
        "value_usd": 1000000,  # $1M
        "timestamp": int(time.time()) - 600,  # 10 minutes ago
        "input": "0x",
    },
    {
        "hash": "0x2345678901abcdef2345678901abcdef2345678901abcdef2345678901abcdef",
        "from": "0x3333333333333333333333333333333333333333",
        "to": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",  # Uniswap
        "value": "0",
        "value_usd": 3000000,  # $3M
        "timestamp": int(time.time()) - 1200,  # 20 minutes ago
        "input": "0xfb3bdb41000000000000000000000000000000000000000000000001158e460913d00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003333333333333333333333333333333333333333000000000000000000000000000000000000000000000000000000006394644b",  # Uniswap swap
    },
    {
        "hash": "0x3456789012abcdef3456789012abcdef3456789012abcdef3456789012abcdef",
        "from": "0x4444444444444444444444444444444444444444",
        "to": "",  # Contract creation
        "value": "0",
        "value_usd": 5000000,  # $5M
        "timestamp": int(time.time()) - 1800,  # 30 minutes ago
        "input": "0x608060405234801561001057600080fd5b506040516107...",  # Contract bytecode
    }
]


@pytest.mark.acceptance
class TestWhaleMonitoringAcceptance:
    """Acceptance test suite for whale transaction monitoring."""
    
    @pytest.fixture
    def whale_monitor(self):
        """Create a WhaleTransactionMonitor with test configuration."""
        providers = [
            "https://api1.example.com",
            "https://api2.example.com",
            "https://api3.example.com"
        ]
        
        return WhaleTransactionMonitor(
            providers=providers,
            threshold_usd=1_000_000,
            cache_ttl=60,
            max_calls_per_minute=30,
            known_exchange_addrs=EXCHANGE_ADDRESSES,
            known_defi_addrs=DEFI_ADDRESSES
        )
    
    @pytest.fixture
    def aggregator(self, whale_monitor):
        """Create an OnChainSentimentAggregator with the whale monitor."""
        return OnChainSentimentAggregator(
            onchain_providers=["https://onchain-api.example.com"],
            sentiment_providers=["https://sentiment-api.example.com"],
            cache_ttl=60,
            whale_monitor=whale_monitor,
            market_data_provider=MagicMock()
        )
    
    @pytest.mark.asyncio
    async def test_transaction_classification(self, whale_monitor):
        """Test that transactions are correctly classified."""
        # Mock the whale monitor to return our test data
        with patch.object(whale_monitor, '_fetch_and_parse', return_value=WHALE_TRANSACTIONS):
            # Get whale transactions
            whales = await whale_monitor.get_whale_transactions()
            
            # Verify all transactions were processed
            assert len(whales) == len(WHALE_TRANSACTIONS)
            
            # Verify classifications
            classifications = [tx['classification'] for tx in whales]
            assert 'exchange_deposit' in classifications
            assert 'exchange_withdrawal' in classifications
            assert 'defi_interaction' in classifications
            assert 'contract_creation' in classifications
    
    @pytest.mark.asyncio
    async def test_multi_provider_failover(self, whale_monitor):
        """Test that the system falls back through multiple providers."""
        # First two providers fail, third succeeds
        async def mock_fetch_and_parse(provider):
            if provider == whale_monitor.provider_manager.providers[0] or \
               provider == whale_monitor.provider_manager.providers[1]:
                return None
            else:
                return WHALE_TRANSACTIONS
        
        with patch.object(whale_monitor, '_fetch_and_parse', side_effect=mock_fetch_and_parse):
            # Get whale transactions
            whales = await whale_monitor.get_whale_transactions()
            
            # Verify we got data from the third provider
            assert len(whales) == len(WHALE_TRANSACTIONS)
            
            # Verify provider health scores were updated
            providers = whale_monitor.provider_manager.providers
            assert whale_monitor.provider_manager.health_scores[providers[0]] < 1.0
            assert whale_monitor.provider_manager.health_scores[providers[1]] < 1.0
            assert whale_monitor.provider_manager.health_scores[providers[2]] == 1.0
    
    @pytest.mark.asyncio
    async def test_rate_limit_functionality(self, whale_monitor):
        """Test that rate limiting works under load."""
        # Set a very low rate limit for testing
        whale_monitor.max_calls_per_minute = 2
        
        # Mock successful API response
        with patch.object(whale_monitor, '_fetch_and_parse', return_value=WHALE_TRANSACTIONS), \
             patch.object(asyncio, 'sleep', AsyncMock()):  # Mock sleep to avoid waiting
            
            # Make multiple calls
            for _ in range(5):
                whales = await whale_monitor.get_whale_transactions()
                assert len(whales) > 0
            
            # Verify rate limits were enforced
            for provider in whale_monitor.provider_manager.providers:
                if provider in whale_monitor.rate_limits:
                    # Rate should not exceed the limit
                    assert whale_monitor.rate_limits[provider]['count'] <= whale_monitor.max_calls_per_minute
    
    @pytest.mark.asyncio
    async def test_aggregator_whale_integration(self, aggregator):
        """Test that the aggregator successfully integrates whale data."""
        # Mock the fetch methods
        with patch.object(aggregator, '_fetch_onchain_data', return_value={"active_addresses": 500000}), \
             patch.object(aggregator, '_fetch_market_data', return_value={"24h_volume": 100000000}), \
             patch.object(aggregator.whale_monitor, '_fetch_and_parse', return_value=WHALE_TRANSACTIONS):
            
            # Fetch data
            data = await aggregator.fetch_data('onchain')
            
            # Verify whale data is included
            assert 'whales' in data['onchain']
            assert data['onchain']['whales']['count'] == len(WHALE_TRANSACTIONS)
            
            # Verify whale pressure
            assert 'whale_pressure' in data['onchain']
            assert isinstance(data['onchain']['whale_pressure'], float)
            assert -1.0 <= data['onchain']['whale_pressure'] <= 1.0
            
            # Verify classification breakdown
            by_classification = data['onchain']['whales']['by_classification']
            assert 'exchange_deposit' in by_classification
            assert 'exchange_withdrawal' in by_classification
            assert 'defi_interaction' in by_classification
            assert 'contract_creation' in by_classification
    
    @pytest.mark.asyncio
    async def test_whale_pressure_calculation(self):
        """Test that whale pressure is correctly calculated based on transaction patterns."""
        # Test case 1: More deposits than withdrawals (selling pressure)
        deposit_heavy = [
            {"classification": "exchange_deposit", "value_usd": 5000000, "timestamp": int(time.time())},
            {"classification": "exchange_deposit", "value_usd": 3000000, "timestamp": int(time.time()) - 300},
            {"classification": "exchange_withdrawal", "value_usd": 1000000, "timestamp": int(time.time()) - 600}
        ]
        
        # Test case 2: More withdrawals than deposits (buying pressure)
        withdrawal_heavy = [
            {"classification": "exchange_withdrawal", "value_usd": 5000000, "timestamp": int(time.time())},
            {"classification": "exchange_withdrawal", "value_usd": 3000000, "timestamp": int(time.time()) - 300},
            {"classification": "exchange_deposit", "value_usd": 1000000, "timestamp": int(time.time()) - 600}
        ]
        
        market_data = {"24h_volume": 100000000}  # $100M daily volume
        
        # Calculate pressure
        deposit_pressure = compute_whale_influence(deposit_heavy, market_data)
        withdrawal_pressure = compute_whale_influence(withdrawal_heavy, market_data)
        
        # Verify deposit-heavy scenario shows positive pressure (selling)
        assert deposit_pressure > 0
        
        # Verify withdrawal-heavy scenario shows negative pressure (buying)
        assert withdrawal_pressure < 0
        
        # Verify relative magnitudes make sense
        assert abs(deposit_pressure) > 0.1  # Should have significant magnitude
        assert abs(withdrawal_pressure) > 0.1  # Should have significant magnitude
    
    @pytest.mark.asyncio
    async def test_full_acceptance_flow(self, aggregator):
        """
        Full acceptance test of the whale monitoring and aggregator integration.
        Tests the end-to-end flow from data retrieval to final aggregated output.
        """
        # Mock the required methods
        with patch.object(aggregator, '_fetch_onchain_data', return_value={
                "active_addresses": 500000,
                "transaction_count": 250000,
                "transaction_volume": 10000,
                "avg_transaction_value": 0.04,
                "nvt_ratio": 50,
                "realized_price": 49000
            }), \
             patch.object(aggregator, '_fetch_sentiment_data', return_value={
                "social_sentiment": 0.3,
                "news_sentiment": 0.5,
                "social_volume": 8000,
                "sentiment_score_reddit": 0.2,
                "sentiment_score_twitter": 0.4
            }), \
             patch.object(aggregator, '_fetch_market_data', return_value={
                "24h_volume": 100000000
            }), \
             patch.object(aggregator.whale_monitor, '_fetch_and_parse', return_value=WHALE_TRANSACTIONS):
            
            # Fetch all data
            data = await aggregator.fetch_data('all')
            
            # Verify all data components are present
            assert 'onchain' in data
            assert 'sentiment' in data
            assert 'whales' in data['onchain']
            assert 'whale_pressure' in data['onchain']
            
            # Verify onchain metrics
            assert data['onchain']['active_addresses'] == 500000
            assert data['onchain']['transaction_count'] == 250000
            
            # Verify sentiment metrics
            assert data['sentiment']['social_sentiment'] == 0.3
            assert data['sentiment']['news_sentiment'] == 0.5
            
            # Verify whale data
            assert data['onchain']['whales']['count'] == len(WHALE_TRANSACTIONS)
            
            # Verify by_classification includes all transaction types
            by_class = data['onchain']['whales']['by_classification']
            expected_types = set(['exchange_deposit', 'exchange_withdrawal', 
                                 'defi_interaction', 'contract_creation'])
            actual_types = set(by_class.keys())
            assert expected_types.issubset(actual_types)
            
            # Verify total volume matches sum of transactions
            expected_volume = sum(tx['value_usd'] for tx in WHALE_TRANSACTIONS)
            assert data['onchain']['whales']['total_volume_usd'] == expected_volume
            
            # Verify transaction details are preserved
            assert len(data['onchain']['whales']['transactions']) == len(WHALE_TRANSACTIONS)
            
            # Log the whale pressure for visibility
            whale_pressure = data['onchain']['whale_pressure']
            assert -1.0 <= whale_pressure <= 1.0
            print(f"\nWhale pressure: {whale_pressure:.4f}")
            
            pressure_desc = "bullish" if whale_pressure < 0 else "bearish"
            magnitude = abs(whale_pressure)
            strength = "strong" if magnitude > 0.5 else "moderate" if magnitude > 0.2 else "weak"
            print(f"Interpretation: {strength} {pressure_desc} pressure") 