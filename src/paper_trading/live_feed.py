"""
Live Feed module for paper trading.

This module provides functionality to create live data feeds for paper trading
using Backtrader's live feed capabilities.
"""

import os
import pandas as pd
import numpy as np
import backtrader as bt
import ccxt
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

# Constants
MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR


class CCXTStore(object):
    """
    CCXT Store for Backtrader.
    
    This store connects to cryptocurrency exchanges through the CCXT library
    and provides data to Backtrader.
    """
    
    def __init__(self, exchange, config=None, retries=5):
        """
        Initialize the CCXT store.
        
        Args:
            exchange: Name of the exchange (e.g., 'binance')
            config: Configuration dictionary for the exchange
            retries: Number of retries for API calls
        """
        self.exchange_name = exchange
        self.retries = retries
        
        # Process config
        self.config = config or {}
        self.api_key = self.config.get('apiKey')
        self.secret = self.config.get('secret')
        
        # Initialize exchange
        if self.exchange_name not in ccxt.exchanges:
            raise ValueError(f"Exchange {self.exchange_name} not found in CCXT")
        
        exchange_class = getattr(ccxt, self.exchange_name)
        self.exchange = exchange_class(self.config)
        
        # Set rate limit parameters
        self.exchange.enableRateLimit = True
        
    def get_datafeed(self, dataname, timeframe='1m', from_date=None, compression=1, ohlcv_limit=None):
        """
        Get a live data feed for Backtrader.
        
        Args:
            dataname: Symbol to trade (e.g., 'BTC/USDT')
            timeframe: Data timeframe (default: '1m')
            from_date: Start date for data
            compression: Data compression
            ohlcv_limit: Maximum number of OHLCV candles to fetch
            
        Returns:
            A Backtrader data feed
        """
        return CCXTLiveFeed(
            store=self,
            dataname=dataname,
            timeframe=timeframe,
            from_date=from_date,
            compression=compression,
            ohlcv_limit=ohlcv_limit,
        )


class CCXTLiveFeed(bt.feeds.DataBase):
    """
    Live Feed implementation for CCXT.
    
    This class provides a live data feed for Backtrader using CCXT.
    """
    
    params = (
        ('timeframe', '1m'),  # Default timeframe
        ('compression', 1),   # Default compression
        ('ohlcv_limit', 100), # Number of historical candles to load
        ('from_date', None),  # Start date
        ('historical', True), # Fetch historical data first
    )
    
    # Map Backtrader timeframes to CCXT timeframes
    _timeframes = {
        (bt.TimeFrame.Minutes, 1): '1m',
        (bt.TimeFrame.Minutes, 3): '3m',
        (bt.TimeFrame.Minutes, 5): '5m',
        (bt.TimeFrame.Minutes, 15): '15m',
        (bt.TimeFrame.Minutes, 30): '30m',
        (bt.TimeFrame.Minutes, 60): '1h',
        (bt.TimeFrame.Minutes, 120): '2h',
        (bt.TimeFrame.Minutes, 240): '4h',
        (bt.TimeFrame.Minutes, 360): '6h',
        (bt.TimeFrame.Minutes, 480): '8h',
        (bt.TimeFrame.Minutes, 720): '12h',
        (bt.TimeFrame.Days, 1): '1d',
        (bt.TimeFrame.Days, 3): '3d',
        (bt.TimeFrame.Weeks, 1): '1w',
        (bt.TimeFrame.Months, 1): '1M',
    }
    
    # Add sentiment line
    lines = ('sentiment',)
    
    def __init__(self, store, **kwargs):
        """
        Initialize the live feed.
        
        Args:
            store: CCXT store for exchange connectivity
            **kwargs: Additional parameters
        """
        self.store = store
        self.exchange = store.exchange
        
        # Get timeframe from params
        self.timeframe = self.p.timeframe
        
        # Set fetch interval based on timeframe
        interval = self._get_interval_seconds(self.timeframe)
        self._interval = interval
        
        # Set last candle timestamp
        self._last_candle_ts = None
        
        # Queue to store historical data
        self._historical_data = []
        
        # Flag to check if we're still loading historical data
        self._historical_loading = self.p.historical
        
        super(CCXTLiveFeed, self).__init__(**kwargs)
        
    def _get_interval_seconds(self, timeframe):
        """Get the interval in seconds for a timeframe."""
        if timeframe.endswith('m'):
            return int(timeframe[:-1]) * MINUTE
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * HOUR
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * DAY
        elif timeframe.endswith('w'):
            return int(timeframe[:-1]) * 7 * DAY
        elif timeframe.endswith('M'):
            return int(timeframe[:-1]) * 30 * DAY
        else:
            return MINUTE  # Default to 1 minute
    
    def start(self):
        """Start the data feed."""
        super(CCXTLiveFeed, self).start()
        
        # Load historical data if requested
        if self.p.historical:
            self._load_historical()
        
    def _load_historical(self):
        """Load historical data from the exchange."""
        since = None
        if self.p.from_date:
            since = int(self.p.from_date.timestamp() * 1000)  # Convert to milliseconds
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.p.dataname,
                timeframe=self.timeframe,
                since=since,
                limit=self.p.ohlcv_limit
            )
            
            # Sort by timestamp and reverse for FIFO
            if ohlcv:
                ohlcv.sort(key=lambda x: x[0])
                self._historical_data = list(reversed(ohlcv))
                self._last_candle_ts = ohlcv[-1][0] / 1000  # Convert from ms to seconds
        except Exception as e:
            print(f"Error loading historical data: {e}")
    
    def _load_new_candle(self):
        """Load new candle data from the exchange."""
        now = time.time()
        
        # Only fetch if enough time has passed since last candle
        if self._last_candle_ts and now - self._last_candle_ts < self._interval:
            return False
        
        # Fetch one new candle
        try:
            since = None
            if self._last_candle_ts:
                # Add 1ms to avoid getting the same candle
                since = int((self._last_candle_ts + 0.001) * 1000)
                
            ohlcv = self.exchange.fetch_ohlcv(
                self.p.dataname,
                timeframe=self.timeframe,
                since=since,
                limit=1
            )
            
            if ohlcv and len(ohlcv) > 0:
                self._historical_data.append(ohlcv[0])
                self._last_candle_ts = ohlcv[0][0] / 1000
                return True
        except Exception as e:
            print(f"Error fetching new candle: {e}")
            
        return False
    
    def _get_sentiment(self, timestamp):
        """
        Get the sentiment value for a timestamp.
        
        In a real implementation, this would fetch sentiment from the sentiment analysis
        component. For now, we'll return a random value for demonstration.
        """
        # TODO: Implement real sentiment fetching
        # This would connect to the sentiment pipeline
        return np.random.normal(0, 0.1)  # Random value around 0
    
    def preload(self):
        """Preload data as needed."""
        if self._historical_loading and self._historical_data:
            return True
        
        return False
        
    def _load(self):
        """Load data for backtrader."""
        if self._historical_loading and self._historical_data:
            # Load from historical data
            candle = self._historical_data.pop()
            self._historical_loading = len(self._historical_data) > 0
            
            timestamp, open_, high, low, close, volume = candle
            timestamp_dt = datetime.fromtimestamp(timestamp / 1000.0)
            
            # Get sentiment
            sentiment = self._get_sentiment(timestamp_dt)
            
            self.lines.datetime[0] = bt.date2num(timestamp_dt)
            self.lines.open[0] = open_
            self.lines.high[0] = high
            self.lines.low[0] = low
            self.lines.close[0] = close
            self.lines.volume[0] = volume
            self.lines.sentiment[0] = sentiment
            
            return True
            
        # No more historical data, try to fetch new candle
        if self._load_new_candle():
            candle = self._historical_data.pop()
            
            timestamp, open_, high, low, close, volume = candle
            timestamp_dt = datetime.fromtimestamp(timestamp / 1000.0)
            
            # Get sentiment
            sentiment = self._get_sentiment(timestamp_dt)
            
            self.lines.datetime[0] = bt.date2num(timestamp_dt)
            self.lines.open[0] = open_
            self.lines.high[0] = high
            self.lines.low[0] = low
            self.lines.close[0] = close
            self.lines.volume[0] = volume
            self.lines.sentiment[0] = sentiment
            
            return True
            
        return False 