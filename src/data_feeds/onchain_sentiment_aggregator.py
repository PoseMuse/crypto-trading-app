"""
On-Chain and Sentiment Data Aggregator

This module provides functionality to aggregate data from multiple sources,
including on-chain metrics, sentiment data, and whale transactions.
It implements fallback mechanisms for reliability.
"""

import os
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.data_feeds.whale_transaction_monitor import WhaleTransactionMonitor, compute_whale_influence

logger = logging.getLogger('onchain_sentiment_aggregator')

class OnChainSentimentAggregator:
    """
    Aggregates on-chain and sentiment data from multiple sources with fallback mechanisms.
    
    Features:
    - Combines data from multiple providers
    - Implements fallback for reliability
    - Caches results to reduce API calls
    - Integrates whale transaction monitoring
    """
    
    def __init__(self, 
                 onchain_providers: List[str] = None,
                 sentiment_providers: List[str] = None,
                 cache_ttl: int = 60,
                 whale_monitor: WhaleTransactionMonitor = None,
                 market_data_provider = None):
        """
        Initialize the aggregator.
        
        Args:
            onchain_providers: List of on-chain data provider URLs
            sentiment_providers: List of sentiment data provider URLs
            cache_ttl: Cache time-to-live in seconds
            whale_monitor: WhaleTransactionMonitor instance
            market_data_provider: Provider for market data metrics
        """
        self.onchain_providers = onchain_providers or []
        self.sentiment_providers = sentiment_providers or []
        self.cache_ttl = cache_ttl
        self.whale_monitor = whale_monitor
        self.market_data_provider = market_data_provider
        
        # Cache
        self.cache = {
            'onchain': None,
            'sentiment': None,
            'last_update': {
                'onchain': 0,
                'sentiment': 0
            }
        }
        
        logger.info(f"OnChainSentimentAggregator initialized with {len(self.onchain_providers)} on-chain providers, "
                  f"{len(self.sentiment_providers)} sentiment providers")
        if whale_monitor:
            logger.info("Whale transaction monitoring enabled")
    
    async def fetch_data(self, data_type: str = 'all') -> Dict[str, Any]:
        """
        Fetch data from all configured sources.
        
        Args:
            data_type: Type of data to fetch ('all', 'onchain', 'sentiment')
            
        Returns:
            Dictionary with aggregated data
        """
        result = {'timestamp': datetime.now().isoformat()}
        data_types_to_fetch = ['onchain', 'sentiment'] if data_type == 'all' else [data_type]
        
        for dt in data_types_to_fetch:
            # Check cache
            if self.cache[dt] and (time.time() - self.cache['last_update'][dt] < self.cache_ttl):
                result[dt] = self.cache[dt]
                logger.debug(f"Using cached {dt} data")
                continue
            
            # Fetch fresh data
            if dt == 'onchain':
                data = await self._fetch_onchain_data()
            elif dt == 'sentiment':
                data = await self._fetch_sentiment_data()
            else:
                logger.warning(f"Unknown data type: {dt}")
                continue
            
            # Update cache
            self.cache[dt] = data
            self.cache['last_update'][dt] = time.time()
            
            # Add to result
            result[dt] = data
        
        # Add whale transaction data if enabled
        if self.whale_monitor and ('onchain' in data_types_to_fetch):
            # Get whale transactions
            try:
                whales = await self.whale_monitor.get_whale_transactions()
                result['onchain']['whales'] = self._process_whale_data(whales)
                
                # Compute whale pressure if market data is available
                if self.market_data_provider:
                    market_data = await self._fetch_market_data()
                    result['onchain']['whale_pressure'] = compute_whale_influence(whales, market_data)
                    logger.info(f"Whale pressure: {result['onchain']['whale_pressure']:.2f}")
            except Exception as e:
                logger.error(f"Error processing whale transactions: {str(e)}")
        
        return result
    
    async def _fetch_onchain_data(self) -> Dict[str, Any]:
        """
        Fetch on-chain data from providers with fallback.
        
        Returns:
            Dictionary with on-chain data
        """
        # This is a simplified example. In a real implementation, 
        # you would fetch from each provider and implement fallback logic.
        result = {
            'active_addresses': 0,
            'transaction_count': 0,
            'transaction_volume': 0,
            'avg_transaction_value': 0,
            'nvt_ratio': 0,
            'realized_price': 0
        }
        
        for provider in self.onchain_providers:
            try:
                # Fetch data from provider
                # For this example, we'll just use placeholder data
                data = await self._fetch_from_provider(provider, 'onchain')
                if data:
                    # Update result with fetched data
                    result.update(data)
                    logger.debug(f"Successfully fetched on-chain data from {provider}")
                    break
            except Exception as e:
                logger.warning(f"Error fetching on-chain data from {provider}: {str(e)}")
                continue
        
        return result
    
    async def _fetch_sentiment_data(self) -> Dict[str, Any]:
        """
        Fetch sentiment data from providers with fallback.
        
        Returns:
            Dictionary with sentiment data
        """
        # Similar to _fetch_onchain_data
        result = {
            'social_sentiment': 0,
            'news_sentiment': 0,
            'social_volume': 0,
            'sentiment_score_reddit': 0,
            'sentiment_score_twitter': 0
        }
        
        for provider in self.sentiment_providers:
            try:
                data = await self._fetch_from_provider(provider, 'sentiment')
                if data:
                    result.update(data)
                    logger.debug(f"Successfully fetched sentiment data from {provider}")
                    break
            except Exception as e:
                logger.warning(f"Error fetching sentiment data from {provider}: {str(e)}")
                continue
        
        return result
    
    async def _fetch_market_data(self) -> Dict[str, Any]:
        """
        Fetch market data for whale pressure calculation.
        
        Returns:
            Dictionary with market data
        """
        if not self.market_data_provider:
            return {'24h_volume': 1000000}  # Default fallback
        
        try:
            # In a real implementation, you would fetch actual market data
            # This is a placeholder
            return {'24h_volume': 1000000}
        except Exception as e:
            logger.warning(f"Error fetching market data: {str(e)}")
            return {'24h_volume': 1000000}  # Default fallback
    
    async def _fetch_from_provider(self, provider: str, data_type: str) -> Optional[Dict[str, Any]]:
        """
        Fetch data from a specific provider.
        
        Args:
            provider: Provider URL
            data_type: Type of data to fetch
            
        Returns:
            Dictionary with fetched data or None if failed
        """
        # In a real implementation, you would make an HTTP request to the provider
        # This is a placeholder that returns synthetic data
        await asyncio.sleep(0.1)  # Simulate network delay
        
        if data_type == 'onchain':
            return {
                'active_addresses': 500000,
                'transaction_count': 250000,
                'transaction_volume': 10000,
                'avg_transaction_value': 0.04,
                'nvt_ratio': 50,
                'realized_price': 49000
            }
        elif data_type == 'sentiment':
            return {
                'social_sentiment': 0.3,
                'news_sentiment': 0.5,
                'social_volume': 8000,
                'sentiment_score_reddit': 0.2,
                'sentiment_score_twitter': 0.4
            }
        
        return None
    
    def _process_whale_data(self, whales: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process and structure whale data for the aggregator.
        
        Args:
            whales: List of whale transactions
            
        Returns:
            Structured whale data
        """
        result = {
            'transactions': whales,
            'count': len(whales),
            'total_volume_usd': sum(tx.get('value_usd', 0) for tx in whales),
            'last_update': datetime.now().isoformat(),
            'by_classification': {}
        }
        
        # Group by classification
        for tx in whales:
            classification = tx.get('classification', 'other')
            if classification not in result['by_classification']:
                result['by_classification'][classification] = {
                    'count': 0,
                    'volume_usd': 0
                }
            
            result['by_classification'][classification]['count'] += 1
            result['by_classification'][classification]['volume_usd'] += tx.get('value_usd', 0)
        
        return result 