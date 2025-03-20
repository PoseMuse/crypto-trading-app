"""
Trading Strategies module for backtesting.

This module defines various trading strategies that can be used
with Backtrader for backtesting and paper trading.
"""

import os
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

# Import our AI model and sentiment components
from ..ai_models.sentiment_model import SentimentEnhancedModel


class BasicSMAStrategy(bt.Strategy):
    """
    Basic SMA crossover strategy for comparison purposes.
    
    This strategy uses two Simple Moving Averages (SMAs) and generates:
    - Buy signal when fast SMA crosses above slow SMA
    - Sell signal when fast SMA crosses below slow SMA
    """
    
    params = (
        ('fast_period', 10),  # Fast SMA period
        ('slow_period', 30),  # Slow SMA period
        ('pct_size', 0.95),   # Percentage of available cash to use for positions
    )
    
    def __init__(self):
        """Initialize the strategy."""
        # Initialize indicators
        self.fast_sma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_sma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        
        # Create a CrossOver signal
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)
        
        # Keep track of open orders
        self.order = None
        
        # Initialize logs
        self.log(f"Strategy initialized with fast_period={self.params.fast_period}, slow_period={self.params.slow_period}")
    
    def log(self, txt, dt=None):
        """Log strategy information."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}: {txt}")
    
    def notify_order(self, order):
        """Process order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            # Order has been submitted/accepted - no action required
            return
        
        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
            else:
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Canceled/Margin/Rejected: {order.status}")
        
        # Reset order
        self.order = None
    
    def notify_trade(self, trade):
        """Process trade notifications."""
        if not trade.isclosed:
            return
        
        self.log(f"TRADE CLOSED, Profit: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}")
    
    def next(self):
        """Define what will be done in each iteration."""
        # Check if an order is pending
        if self.order:
            return
        
        # Check if we are in the market
        if not self.position:
            # We are not in the market, check if we should buy
            if self.crossover > 0:  # Fast SMA crossed above slow SMA
                size = self.broker.getcash() * self.params.pct_size / self.data.close[0]
                self.log(f"BUY CREATE, {self.data.close[0]:.2f}")
                self.order = self.buy(size=size)
        
        else:
            # We are in the market, check if we should sell
            if self.crossover < 0:  # Fast SMA crossed below slow SMA
                self.log(f"SELL CREATE, {self.data.close[0]:.2f}")
                self.order = self.sell(size=self.position.size)


class SentimentStrategy(bt.Strategy):
    """
    Trading strategy that uses sentiment data from social media.
    
    This strategy combines technical indicators with sentiment data:
    - Buy signal when sentiment is positive and technical indicator confirms
    - Sell signal when sentiment turns negative or technical indicator signals a sell
    """
    
    params = (
        ('fast_period', 10),       # Fast SMA period
        ('slow_period', 30),       # Slow SMA period
        ('sentiment_threshold', 0.2),  # Minimum sentiment score to consider positive
        ('pct_size', 0.95),        # Percentage of available cash to use for positions
    )
    
    def __init__(self):
        """Initialize the strategy."""
        # Price-based indicators
        self.fast_sma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_sma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)
        
        # RSI for additional confirmation
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        
        # Access the sentiment data line
        self.sentiment = self.data.sentiment
        
        # Keep track of open orders
        self.order = None
        
        # Initialize logs
        self.log(f"Sentiment Strategy initialized with threshold={self.params.sentiment_threshold}")
    
    def log(self, txt, dt=None):
        """Log strategy information."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}: {txt}")
    
    def notify_order(self, order):
        """Process order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            # Order has been submitted/accepted - no action required
            return
        
        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
            else:
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Canceled/Margin/Rejected: {order.status}")
        
        # Reset order
        self.order = None
    
    def notify_trade(self, trade):
        """Process trade notifications."""
        if not trade.isclosed:
            return
        
        self.log(f"TRADE CLOSED, Profit: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}")
    
    def next(self):
        """Define what will be done in each iteration."""
        # Check if an order is pending
        if self.order:
            return
        
        # Get current sentiment score
        current_sentiment = self.sentiment[0]
        self.log(f"Sentiment: {current_sentiment:.2f}, RSI: {self.rsi[0]:.2f}")
        
        # Check if we are in the market
        if not self.position:
            # We are not in the market, check if we should buy
            
            # Buy conditions:
            # 1. Sentiment is above threshold
            # 2. Fast SMA crosses above slow SMA
            # 3. RSI is not overbought (< 70)
            if (current_sentiment > self.params.sentiment_threshold and 
                self.crossover > 0 and 
                self.rsi[0] < 70):
                
                size = self.broker.getcash() * self.params.pct_size / self.data.close[0]
                self.log(f"BUY CREATE (Sentiment: {current_sentiment:.2f}), Price: {self.data.close[0]:.2f}")
                self.order = self.buy(size=size)
        
        else:
            # We are in the market, check if we should sell
            
            # Sell conditions:
            # 1. Sentiment turns negative (below negative threshold)
            # 2. Fast SMA crosses below slow SMA
            # 3. RSI is overbought (> 70)
            if (current_sentiment < -self.params.sentiment_threshold or 
                self.crossover < 0 or 
                self.rsi[0] > 70):
                
                self.log(f"SELL CREATE (Sentiment: {current_sentiment:.2f}), Price: {self.data.close[0]:.2f}")
                self.order = self.sell(size=self.position.size)


class AIModelStrategy(bt.Strategy):
    """
    Trading strategy that uses predictions from our AI model.
    
    This strategy uses predictions from a pre-trained LightGBM model:
    - Buy signal when the model predicts positive returns above a threshold
    - Sell signal when the model predicts negative returns below a threshold
    """
    
    params = (
        ('model_path', None),       # Path to the saved model file
        ('positive_threshold', 0.2), # Threshold for positive predictions (percentage)
        ('negative_threshold', -0.2), # Threshold for negative predictions (percentage)
        ('pct_size', 0.95),         # Percentage of available cash to use for positions
        ('symbol', 'BTC/USDT'),     # Symbol to trade
        ('use_sentiment', True),    # Whether to use sentiment-enhanced model
    )
    
    def __init__(self):
        """Initialize the strategy."""
        # Initialize the AI model
        self.model = None
        self.load_model()
        
        # Basic technical indicators for comparison
        self.sma50 = bt.indicators.SMA(self.data.close, period=50)
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        
        # Create a prediction array
        self.predictions = []
        
        # Keep track of open orders
        self.order = None
        
        # Log initialization
        self.log("AI Model Strategy initialized")
    
    def load_model(self):
        """Load the pre-trained model."""
        try:
            # If model_path is provided, load from file
            if self.params.model_path and os.path.exists(self.params.model_path):
                import joblib
                self.model = joblib.load(self.params.model_path)
                self.log(f"Loaded model from {self.params.model_path}")
                return
            
            # Otherwise, use the SentimentEnhancedModel class
            if self.params.use_sentiment:
                from ..ai_models.sentiment_model import SentimentEnhancedModel
                self.sentiment_model = SentimentEnhancedModel(
                    symbol=self.params.symbol,
                    sentiment_lookback_days=7,
                    reddit_use_mock=True
                )
                
                # Fetch data
                self.sentiment_model.fetch_price_data(days=365)
                self.sentiment_model.fetch_sentiment_data()
                
                # Prepare features and train model
                self.sentiment_model.prepare_features()
                self.model, _ = self.sentiment_model.train(use_walk_forward=False)
                
                self.log("Trained sentiment-enhanced model")
            else:
                from ..ai_models.model_pipeline import fetch_historical_data, prepare_features, train_lightgbm
                
                # Fetch historical data
                start_date = datetime.now() - timedelta(days=365)
                end_date = datetime.now()
                df = fetch_historical_data(self.params.symbol, start_date=start_date, end_date=end_date)
                
                # Prepare features
                features, target = prepare_features(df)
                
                # Train model
                self.model, _ = train_lightgbm(features, target)
                
                self.log("Trained basic model without sentiment")
                
        except Exception as e:
            self.log(f"Error loading model: {e}")
            self.model = None
    
    def log(self, txt, dt=None):
        """Log strategy information."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}: {txt}")
    
    def notify_order(self, order):
        """Process order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            # Order has been submitted/accepted - no action required
            return
        
        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
            else:
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}")
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Canceled/Margin/Rejected: {order.status}")
        
        # Reset order
        self.order = None
    
    def notify_trade(self, trade):
        """Process trade notifications."""
        if not trade.isclosed:
            return
        
        self.log(f"TRADE CLOSED, Profit: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}")
    
    def get_features(self):
        """Extract features for model prediction from current data."""
        try:
            # This is a simplified feature extraction
            # In a real implementation, you would compute the same features used in training
            
            # Get historical prices for feature calculation
            prices = np.array([self.data.close.get(i, 0) for i in range(-100, 1)])
            
            # If we don't have enough price history, return None
            if len(prices) < 100:
                return None
            
            # Basic features
            returns = np.diff(prices) / prices[:-1]
            
            # Moving averages
            ma5 = np.mean(prices[-5:])
            ma10 = np.mean(prices[-10:])
            ma20 = np.mean(prices[-20:])
            ma50 = np.mean(prices[-50:])
            
            # Volatility
            vol_5 = np.std(returns[-5:])
            vol_10 = np.std(returns[-10:])
            
            # Current sentiment if available
            sentiment = self.data.sentiment[0] if hasattr(self.data, 'sentiment') else 0.0
            
            # Combine features into a DataFrame
            import pandas as pd
            features = pd.DataFrame({
                'returns': [returns[-1]],
                'log_returns': [np.log(prices[-1]/prices[-2])],
                'ma5': [ma5 / prices[-1]],
                'ma10': [ma10 / prices[-1]],
                'ma20': [ma20 / prices[-1]],
                'ma50': [ma50 / prices[-1]],
                'volatility_5': [vol_5],
                'volatility_10': [vol_10],
                'sentiment_score': [sentiment],
                'weighted_sentiment': [sentiment],
                'positive_ratio': [max(0, sentiment)],
                'negative_ratio': [max(0, -sentiment)]
            })
            
            return features
            
        except Exception as e:
            self.log(f"Error getting features: {e}")
            return None
    
    def next(self):
        """Define what will be done in each iteration."""
        # Check if an order is pending
        if self.order:
            return
        
        # Skip if model is not loaded
        if self.model is None:
            return
        
        # Get features for prediction
        features = self.get_features()
        if features is None:
            return
        
        # Make prediction
        try:
            prediction = self.model.predict(features)[0]
            self.predictions.append(prediction)
            
            # Convert to percentage
            prediction_pct = prediction * 100
            
            self.log(f"Prediction: {prediction_pct:.2f}%")
            
            # Check if we are in the market
            if not self.position:
                # We are not in the market, check if we should buy
                if prediction_pct > self.params.positive_threshold:
                    size = self.broker.getcash() * self.params.pct_size / self.data.close[0]
                    self.log(f"BUY CREATE (Prediction: {prediction_pct:.2f}%), Price: {self.data.close[0]:.2f}")
                    self.order = self.buy(size=size)
            
            else:
                # We are in the market, check if we should sell
                if prediction_pct < self.params.negative_threshold:
                    self.log(f"SELL CREATE (Prediction: {prediction_pct:.2f}%), Price: {self.data.close[0]:.2f}")
                    self.order = self.sell(size=self.position.size)
        
        except Exception as e:
            self.log(f"Error making prediction: {e}")


class CombinedStrategy(bt.Strategy):
    """
    Combined strategy that uses both AI model predictions and sentiment data.
    
    This strategy:
    - Uses the predictions from the AI model to determine trade direction
    - Uses sentiment as a confidence factor to adjust position size
    - Uses technical indicators as risk management filters
    """
    
    params = (
        ('model_path', None),        # Path to the saved model file
        ('prediction_threshold', 0.2), # Threshold for model predictions (percentage)
        ('sentiment_boost', 0.5),    # How much to boost position size when sentiment agrees
        ('pct_size', 0.9),           # Base percentage of available cash to use
        ('symbol', 'BTC/USDT'),      # Symbol to trade
    )
    
    def __init__(self):
        """Initialize the strategy."""
        # Initialize the AI model
        self.model = None
        self.load_model()
        
        # Technical indicators for risk management
        self.sma50 = bt.indicators.SMA(self.data.close, period=50)
        self.sma200 = bt.indicators.SMA(self.data.close, period=200)
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.atr = bt.indicators.ATR(self.data, period=14)
        
        # Get sentiment data
        self.sentiment = self.data.sentiment
        
        # Keep track of open orders
        self.order = None
        
        # Log initialization
        self.log("Combined Strategy initialized")
    
    def load_model(self):
        """Load the pre-trained model."""
        try:
            # If model_path is provided, load from file
            if self.params.model_path and os.path.exists(self.params.model_path):
                import joblib
                self.model = joblib.load(self.params.model_path)
                self.log(f"Loaded model from {self.params.model_path}")
                return
            
            # Otherwise, use the SentimentEnhancedModel class
            from ..ai_models.sentiment_model import SentimentEnhancedModel
            self.sentiment_model = SentimentEnhancedModel(
                symbol=self.params.symbol,
                sentiment_lookback_days=7,
                reddit_use_mock=True
            )
            
            # Fetch data
            self.sentiment_model.fetch_price_data(days=365)
            self.sentiment_model.fetch_sentiment_data()
            
            # Prepare features and train model
            self.sentiment_model.prepare_features()
            self.model, _ = self.sentiment_model.train(use_walk_forward=False)
            
            self.log("Trained sentiment-enhanced model")
                
        except Exception as e:
            self.log(f"Error loading model: {e}")
            self.model = None
    
    def log(self, txt, dt=None):
        """Log strategy information."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}: {txt}")
    
    def notify_order(self, order):
        """Process order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            # Order has been submitted/accepted - no action required
            return
        
        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size:.6f}, Value: {order.executed.value:.2f}")
            else:
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size:.6f}, Value: {order.executed.value:.2f}")
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Canceled/Margin/Rejected: {order.status}")
        
        # Reset order
        self.order = None
    
    def notify_trade(self, trade):
        """Process trade notifications."""
        if not trade.isclosed:
            return
        
        self.log(f"TRADE CLOSED, Profit: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}")
    
    def get_features(self):
        """Extract features for model prediction from current data."""
        try:
            # This is a simplified feature extraction
            # In a real implementation, you would compute the same features used in training
            
            # Get historical prices for feature calculation
            prices = np.array([self.data.close.get(i, 0) for i in range(-100, 1)])
            
            # If we don't have enough price history, return None
            if len(prices) < 100:
                return None
            
            # Basic features
            returns = np.diff(prices) / prices[:-1]
            
            # Moving averages
            ma5 = np.mean(prices[-5:])
            ma10 = np.mean(prices[-10:])
            ma20 = np.mean(prices[-20:])
            ma50 = np.mean(prices[-50:])
            
            # Volatility
            vol_5 = np.std(returns[-5:])
            vol_10 = np.std(returns[-10:])
            
            # Current sentiment
            sentiment = self.sentiment[0]
            
            # Combine features into a DataFrame
            import pandas as pd
            features = pd.DataFrame({
                'returns': [returns[-1]],
                'log_returns': [np.log(prices[-1]/prices[-2])],
                'ma5': [ma5 / prices[-1]],
                'ma10': [ma10 / prices[-1]],
                'ma20': [ma20 / prices[-1]],
                'ma50': [ma50 / prices[-1]],
                'volatility_5': [vol_5],
                'volatility_10': [vol_10],
                'sentiment_score': [sentiment],
                'weighted_sentiment': [sentiment],
                'positive_ratio': [max(0, sentiment)],
                'negative_ratio': [max(0, -sentiment)]
            })
            
            return features
            
        except Exception as e:
            self.log(f"Error getting features: {e}")
            return None
    
    def next(self):
        """Define what will be done in each iteration."""
        # Check if an order is pending
        if self.order:
            return
        
        # Skip if model is not loaded
        if self.model is None:
            return
        
        # Get features for prediction
        features = self.get_features()
        if features is None:
            return
        
        # Make prediction
        try:
            prediction = self.model.predict(features)[0]
            
            # Convert to percentage
            prediction_pct = prediction * 100
            
            # Get current sentiment
            current_sentiment = self.sentiment[0]
            
            # Determine if sentiment agrees with prediction (both positive or both negative)
            sentiment_agrees = (prediction > 0 and current_sentiment > 0) or (prediction < 0 and current_sentiment < 0)
            
            # Adjust position size based on sentiment agreement
            position_size_factor = self.params.pct_size
            if sentiment_agrees:
                position_size_factor += self.params.sentiment_boost
            
            self.log(f"Prediction: {prediction_pct:.2f}%, Sentiment: {current_sentiment:.2f}, RSI: {self.rsi[0]:.2f}")
            
            # Risk management filters
            # 1. Check if price is above 200-day SMA (bullish trend)
            trend_bullish = self.data.close[0] > self.sma200[0]
            # 2. Check if RSI is not extreme
            rsi_ok = 30 < self.rsi[0] < 70
            
            # Check if we are in the market
            if not self.position:
                # We are not in the market, check if we should buy
                if (prediction_pct > self.params.prediction_threshold and 
                    trend_bullish and 
                    rsi_ok):
                    
                    # Calculate position size
                    size = self.broker.getcash() * position_size_factor / self.data.close[0]
                    
                    self.log(f"BUY CREATE (Pred: {prediction_pct:.2f}%, Sent: {current_sentiment:.2f}), Size: {size:.6f}")
                    self.order = self.buy(size=size)
            
            else:
                # We are in the market, check if we should sell
                if (prediction_pct < -self.params.prediction_threshold or
                    not trend_bullish or
                    self.rsi[0] > 70):  # RSI overbought
                    
                    self.log(f"SELL CREATE (Pred: {prediction_pct:.2f}%, Sent: {current_sentiment:.2f}), Size: {self.position.size}")
                    self.order = self.sell(size=self.position.size)
        
        except Exception as e:
            self.log(f"Error in next method: {e}")
            import traceback
            traceback.print_exc() 