"""
Sentiment-enhanced trading model implementation.

This module extends the standard model pipeline to incorporate social media sentiment.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List, Any, Union

from .model_pipeline import (
    fetch_historical_data,
    prepare_features,
    train_lightgbm,
    evaluate_model,
    walk_forward_validation,
    save_model
)

from ..sentiment_analysis.sentiment_pipeline import (
    fetch_reddit_posts,
    aggregate_sentiment,
    load_sentiment_data
)


class SentimentEnhancedModel:
    """
    Trading model that incorporates social media sentiment data.
    
    This class extends the base model pipeline by adding sentiment features
    from social media platforms like Reddit.
    """
    
    def __init__(
        self,
        symbol: str,
        sentiment_lookback_days: int = 7,
        sentiment_data_dir: str = 'data/sentiment',
        model_dir: str = 'models/sentiment',
        reddit_use_mock: bool = True
    ):
        """
        Initialize the sentiment-enhanced model.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            sentiment_lookback_days: Days to apply sentiment data retroactively
            sentiment_data_dir: Directory to store sentiment data
            model_dir: Directory to save trained models
            reddit_use_mock: Whether to use mock Reddit data
        """
        self.symbol = symbol
        self.sentiment_lookback_days = sentiment_lookback_days
        self.sentiment_data_dir = sentiment_data_dir
        self.model_dir = model_dir
        self.reddit_use_mock = reddit_use_mock
        
        # Extract base currency for sentiment analysis
        self.base_currency = symbol.split('/')[0].lower()
        
        # Create directories
        os.makedirs(sentiment_data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize other properties
        self.model = None
        self.sentiment_data = None
        self.price_data = None
        self.features = None
        self.target = None
        
    def map_currency_to_subreddits(self) -> List[str]:
        """
        Map cryptocurrency symbol to relevant subreddits.
        
        Returns:
            List of subreddit names
        """
        # Map currencies to relevant subreddits
        subreddit_mapping = {
            'btc': ['bitcoin', 'CryptoCurrency'],
            'eth': ['ethereum', 'CryptoCurrency'],
            'sol': ['solana', 'CryptoCurrency'],
            'ada': ['cardano', 'CryptoCurrency'],
            'xrp': ['ripple', 'CryptoCurrency'],
            'doge': ['dogecoin', 'CryptoCurrency'],
            'dot': ['polkadot', 'CryptoCurrency'],
            'bnb': ['binance', 'CryptoCurrency'],
            # Add more mappings as needed
        }
        
        # Default to cryptocurrency if no specific mapping
        return subreddit_mapping.get(
            self.base_currency.lower(), 
            ['CryptoCurrency']
        )
    
    def fetch_sentiment_data(self, force_refresh: bool = False) -> Dict:
        """
        Fetch or load sentiment data for the specified symbol.
        
        Args:
            force_refresh: Whether to force refresh even if cached data exists
            
        Returns:
            Dictionary containing sentiment metrics
        """
        # Path to store sentiment data
        sentiment_file = os.path.join(
            self.sentiment_data_dir, 
            f"{self.base_currency}_sentiment.json"
        )
        
        # Try to load existing sentiment data if not forcing refresh
        if not force_refresh and os.path.exists(sentiment_file):
            try:
                sentiment_data = load_sentiment_data(sentiment_file)
                # Check if data is fresh (less than 24 hours old)
                if 'timestamp' in sentiment_data:
                    timestamp = datetime.fromisoformat(sentiment_data['timestamp'])
                    if datetime.now() - timestamp < timedelta(hours=24):
                        print(f"Using cached sentiment data from {timestamp}")
                        self.sentiment_data = sentiment_data
                        return sentiment_data
            except Exception as e:
                print(f"Error loading sentiment data: {e}")
        
        # Get relevant subreddits
        subreddits = self.map_currency_to_subreddits()
        print(f"Fetching fresh sentiment data from {subreddits}")
        
        # Fetch Reddit posts
        posts = fetch_reddit_posts(
            subreddits=subreddits,
            limit=100,
            time_filter="day",
            use_mock=self.reddit_use_mock
        )
        
        # Aggregate sentiment
        sentiment_data = aggregate_sentiment(posts)
        
        # Add metadata
        sentiment_data['symbol'] = self.symbol
        sentiment_data['base_currency'] = self.base_currency
        sentiment_data['subreddits'] = subreddits
        sentiment_data['timestamp'] = datetime.now().isoformat()
        
        # Save to file
        with open(sentiment_file, 'w') as f:
            json.dump(sentiment_data, f, indent=2)
        
        self.sentiment_data = sentiment_data
        return sentiment_data
    
    def fetch_price_data(
        self,
        days: int = 365,
        exchange_manager: Any = None
    ) -> pd.DataFrame:
        """
        Fetch historical price data.
        
        Args:
            days: Number of days of historical data to fetch
            exchange_manager: Exchange manager instance or None
            
        Returns:
            DataFrame with price data
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch price data
        price_data = fetch_historical_data(
            symbol=self.symbol,
            exchange_manager=exchange_manager,
            start_date=start_date,
            end_date=end_date
        )
        
        self.price_data = price_data
        return price_data
    
    def combine_price_and_sentiment(self) -> pd.DataFrame:
        """
        Combine price data with sentiment data.
        
        Returns:
            DataFrame with combined price and sentiment data
        """
        if self.price_data is None:
            raise ValueError("Price data must be fetched before combining")
            
        if self.sentiment_data is None:
            raise ValueError("Sentiment data must be fetched before combining")
        
        # Clone the price data
        df = self.price_data.copy()
        
        # Extract sentiment values
        overall_sentiment = self.sentiment_data.get('overall_sentiment', 0)
        weighted_sentiment = self.sentiment_data.get('weighted_sentiment', 0)
        positive_ratio = self.sentiment_data.get('positive_ratio', 0)
        negative_ratio = self.sentiment_data.get('negative_ratio', 0)
        
        # Get the last 'sentiment_lookback_days' entries (or all if fewer)
        lookback_range = min(self.sentiment_lookback_days, len(df))
        
        # Add sentiment data as new columns to the most recent entries
        df['sentiment_score'] = 0.0
        df['weighted_sentiment'] = 0.0
        df['positive_ratio'] = 0.0
        df['negative_ratio'] = 0.0
        
        # Apply sentiment to recent days
        if lookback_range > 0:
            df.iloc[-lookback_range:, df.columns.get_loc('sentiment_score')] = overall_sentiment
            df.iloc[-lookback_range:, df.columns.get_loc('weighted_sentiment')] = weighted_sentiment
            df.iloc[-lookback_range:, df.columns.get_loc('positive_ratio')] = positive_ratio
            df.iloc[-lookback_range:, df.columns.get_loc('negative_ratio')] = negative_ratio
        
        return df
    
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features from combined price and sentiment data.
        
        Returns:
            Tuple of (features_df, target_series)
        """
        # Combine price and sentiment data
        combined_data = self.combine_price_and_sentiment()
        
        # Prepare features using standard pipeline
        features, target = prepare_features(combined_data)
        
        self.features = features
        self.target = target
        
        return features, target
    
    def train(
        self,
        use_walk_forward: bool = True,
        n_splits: int = 5,
        initial_train_size: float = 0.6,
        params: Optional[Dict] = None
    ) -> Tuple[Any, Dict]:
        """
        Train the model.
        
        Args:
            use_walk_forward: Whether to use walk-forward validation
            n_splits: Number of splits for walk-forward validation
            initial_train_size: Initial training size for walk-forward validation
            params: Model parameters
            
        Returns:
            Tuple of (trained_model, metrics)
        """
        if self.features is None or self.target is None:
            self.prepare_features()
        
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'verbose': -1
            }
        
        if use_walk_forward:
            # Perform walk-forward validation
            avg_metrics, fold_results = walk_forward_validation(
                features=self.features,
                target=self.target,
                params=params,
                n_splits=n_splits,
                initial_train_size=initial_train_size,
                verbose=True
            )
            
            # Save validation results
            results_file = os.path.join(
                self.model_dir,
                f"{self.base_currency}_walk_forward_results.json"
            )
            with open(results_file, 'w') as f:
                json.dump({
                    'avg_metrics': avg_metrics,
                    'fold_results': fold_results
                }, f, indent=2)
            
            # Train final model on all data
            model, training_info = train_lightgbm(
                features=self.features,
                target=self.target,
                params=params
            )
            
            self.model = model
            return model, avg_metrics
        else:
            # Train using standard training method
            model, training_info = train_lightgbm(
                features=self.features,
                target=self.target,
                params=params
            )
            
            # Evaluate
            from sklearn.model_selection import train_test_split
            _, X_test, _, y_test = train_test_split(
                self.features, self.target, test_size=0.2, random_state=42
            )
            metrics = evaluate_model(model, X_test, y_test)
            
            self.model = model
            return model, metrics
    
    def save(self, filename: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Args:
            filename: Custom filename or None for auto-generated
            
        Returns:
            Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        if filename is None:
            filename = f"sentiment_model_{self.symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.txt"
        
        model_path = os.path.join(self.model_dir, filename)
        save_model(self.model, model_path)
        
        return model_path
    
    def predict(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            data: DataFrame to predict on, or None to use the training features
            
        Returns:
            NumPy array of predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before predicting")
        
        # Use provided data or fall back to training features
        features_to_predict = data if data is not None else self.features
        
        return self.model.predict(features_to_predict)
    
    def get_sentiment_importance(self) -> pd.DataFrame:
        """
        Get the importance of sentiment features in the model.
        
        Returns:
            DataFrame with feature importance for sentiment features
        """
        if self.model is None:
            raise ValueError("Model must be trained to get feature importance")
        
        # Get all feature importances
        importance = self.model.feature_importance()
        importance_df = pd.DataFrame({
            'Feature': self.features.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Filter for sentiment features
        sentiment_features = [
            'sentiment_score', 'weighted_sentiment', 
            'positive_ratio', 'negative_ratio'
        ]
        sentiment_importance = importance_df[
            importance_df['Feature'].isin(sentiment_features)
        ]
        
        return sentiment_importance 