#!/usr/bin/env python3
"""
Sentiment Integration Script

This script:
1. Fetches cryptocurrency market data
2. Retrieves sentiment data
3. Combines them to create enhanced features for model training
4. Trains a model using the combined data

Usage:
    python src/sentiment_integration.py
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Import our model pipeline and sentiment analysis
from ai_models.model_pipeline import (
    fetch_historical_data,
    prepare_features,
    train_lightgbm,
    evaluate_model,
    save_model
)

from sentiment_analysis.sentiment_pipeline import (
    fetch_reddit_posts,
    aggregate_sentiment,
    load_sentiment_data
)

def get_or_fetch_sentiment(symbol: str, data_dir: str = 'data/sentiment'):
    """
    Get sentiment data for a cryptocurrency, fetching it if not available.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        data_dir: Directory to store/look for sentiment data
        
    Returns:
        DataFrame with sentiment data
    """
    # Extract the base currency (e.g., 'BTC' from 'BTC/USDT')
    base_currency = symbol.split('/')[0].lower()
    
    # Map currencies to relevant subreddits
    subreddit_mapping = {
        'btc': ['bitcoin', 'CryptoCurrency'],
        'eth': ['ethereum', 'CryptoCurrency'],
        'sol': ['solana', 'CryptoCurrency'],
        'ada': ['cardano', 'CryptoCurrency'],
        'xrp': ['ripple', 'CryptoCurrency'],
        # Add more mappings as needed
    }
    
    # Default to cryptocurrency if no specific mapping
    subreddits = subreddit_mapping.get(
        base_currency.lower(), 
        ['CryptoCurrency']
    )
    
    # Path to store sentiment data
    sentiment_file = os.path.join(data_dir, f"{base_currency}_sentiment.json")
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Try to load existing sentiment data
    if os.path.exists(sentiment_file):
        try:
            sentiment_data = load_sentiment_data(sentiment_file)
            # Check if data is fresh (less than 24 hours old)
            if 'timestamp' in sentiment_data:
                timestamp = datetime.fromisoformat(sentiment_data['timestamp'])
                if datetime.now() - timestamp < timedelta(hours=24):
                    print(f"Using cached sentiment data for {base_currency} from {timestamp}")
                    return sentiment_data
        except Exception as e:
            print(f"Error loading sentiment data: {e}")
    
    # If we get here, we need to fetch new sentiment data
    print(f"Fetching fresh sentiment data for {base_currency} from {subreddits}")
    
    # Fetch Reddit posts with mock data for now (replace with real data in production)
    posts = fetch_reddit_posts(subreddits, limit=100, use_mock=True)
    
    # Aggregate sentiment
    sentiment_data = aggregate_sentiment(posts)
    
    # Add metadata
    sentiment_data['symbol'] = symbol
    sentiment_data['base_currency'] = base_currency
    sentiment_data['subreddits'] = subreddits
    sentiment_data['timestamp'] = datetime.now().isoformat()
    
    # Save to file
    with open(sentiment_file, 'w') as f:
        json.dump(sentiment_data, f, indent=2)
    
    print(f"Saved fresh sentiment data to {sentiment_file}")
    return sentiment_data

def combine_price_and_sentiment(
    price_data: pd.DataFrame,
    sentiment_data: dict,
    lookback_days: int = 7
) -> pd.DataFrame:
    """
    Combine price data with sentiment data to create enhanced features.
    
    Args:
        price_data: DataFrame with OHLCV data
        sentiment_data: Dictionary with sentiment metrics
        lookback_days: Number of days to apply sentiment data to past prices
        
    Returns:
        DataFrame with combined data
    """
    # Clone the price data
    df = price_data.copy()
    
    # Extract sentiment values
    overall_sentiment = sentiment_data.get('overall_sentiment', 0)
    weighted_sentiment = sentiment_data.get('weighted_sentiment', 0)
    positive_ratio = sentiment_data.get('positive_ratio', 0)
    negative_ratio = sentiment_data.get('negative_ratio', 0)
    
    # Get the last 'lookback_days' entries (or all if fewer)
    lookback_range = min(lookback_days, len(df))
    
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

def main():
    """Main function to demonstrate sentiment integration."""
    print("Starting sentiment integration demo...")
    
    # Configuration
    symbol = "BTC/USDT"
    days = 365
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Directories for data and models
    data_dir = 'data/sentiment'
    model_dir = 'models/sentiment_enhanced'
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Fetch historical price data
    print(f"Fetching historical data for {symbol}...")
    price_data = fetch_historical_data(symbol, start_date=start_date, end_date=end_date)
    
    if price_data.empty:
        print("Error: No price data available. Exiting.")
        return
    
    print(f"Fetched {len(price_data)} price data points")
    
    # 2. Get sentiment data
    print(f"Getting sentiment data for {symbol}...")
    sentiment_data = get_or_fetch_sentiment(symbol, data_dir)
    
    # 3. Combine price and sentiment data
    print("Combining price and sentiment data...")
    combined_data = combine_price_and_sentiment(price_data, sentiment_data)
    
    # 4. Prepare features
    print("Preparing features...")
    features, target = prepare_features(combined_data)
    
    # Print feature list to see the sentiment features
    print("\nFeatures including sentiment:")
    print(features.columns.tolist())
    
    # 5. Train a model with the enhanced features
    print("\nTraining model with sentiment-enhanced features...")
    model, training_info = train_lightgbm(features, target)
    
    # 6. Evaluate the model
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("Model evaluation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 7. Save the model
    model_path = os.path.join(
        model_dir, 
        f"sentiment_enhanced_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.txt"
    )
    save_model(model, model_path)
    
    # Print feature importance
    print("\nFeature importance:")
    importance_dict = training_info['feature_importance']
    importance_df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    }).sort_values('Importance', ascending=False)
    
    print(importance_df.head(10))
    
    # Check if sentiment features are important
    sentiment_features = ['sentiment_score', 'weighted_sentiment', 'positive_ratio', 'negative_ratio']
    sentiment_importance = importance_df[importance_df['Feature'].isin(sentiment_features)]
    
    print("\nSentiment feature importance:")
    print(sentiment_importance)
    
    print("\nSentiment integration demo complete!")
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main() 