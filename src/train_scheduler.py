#!/usr/bin/env python3
"""
Model Training Scheduler

This script implements a scheduled retraining system for the cryptocurrency trading models.
It can be run manually or scheduled via cron to ensure models are regularly updated
with the latest market data.

Example usage:
    # Retrain with current date
    python src/train_scheduler.py

    # Retrain with specific date
    python src/train_scheduler.py --date 2023-10-25
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import lightgbm as lgb

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai_models.model_pipeline import (
    fetch_historical_data,
    prepare_features,
    train_lightgbm,
    evaluate_model,
    save_model,
    export_to_onnx
)
from src.sentiment_analysis.sentiment_pipeline import (
    fetch_reddit_posts,
    analyze_sentiment,
    aggregate_sentiment,
    aggregate_multisource_sentiment
)
from src.sentiment_analysis.telegram_pipeline import (
    fetch_telegram_messages,
    fetch_telegram_sentiment
)
from src.sentiment_analysis.twitter_pipeline import (
    fetch_twitter_tweets,
    fetch_twitter_sentiment
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Ensure required directories exist."""
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/lightgbm', exist_ok=True)
    os.makedirs('models/onnx', exist_ok=True)
    os.makedirs('data/sentiment', exist_ok=True)

def fetch_sentiment_data(symbols, days_back=7):
    """
    Fetch and analyze sentiment data for specified symbols from multiple sources.
    
    Args:
        symbols: List of symbol names (e.g., ['BTC', 'ETH'])
        days_back: Number of days of historical sentiment to analyze
        
    Returns:
        DataFrame with sentiment data
    """
    all_sentiment = {}
    
    for symbol in symbols:
        # Reddit data
        subreddits = [f"crypto", f"{symbol}", f"{symbol}currency"]
        
        logger.info(f"Fetching Reddit posts for {symbol} from {subreddits}")
        reddit_posts = fetch_reddit_posts(
            subreddits=subreddits,
            limit=100,
            time_filter="week"
        )
        
        # Telegram data
        telegram_channels = [f"crypto_{symbol.lower()}", "cryptosignals"]
        
        logger.info(f"Fetching Telegram messages for {symbol} from {telegram_channels}")
        telegram_messages = []
        for channel in telegram_channels:
            channel_messages = fetch_telegram_messages(
                channel_username=channel,
                limit=100
            )
            telegram_messages.extend(channel_messages)
        
        # Twitter data
        twitter_query = f"{symbol} crypto"
        
        logger.info(f"Fetching Twitter tweets for query: {twitter_query}")
        twitter_tweets = fetch_twitter_tweets(
            query=twitter_query,
            limit=100,
            days_back=days_back
        )
        
        # Aggregate sentiment from all sources
        if reddit_posts or telegram_messages or twitter_tweets:
            # Source weights - can be adjusted based on reliability
            source_weights = {
                'reddit': 1.0,    # Full weight for Reddit
                'telegram': 0.5,  # Half weight for Telegram (could be less reliable)
                'twitter': 1.0    # Full weight for Twitter
            }
            
            sentiment_data = aggregate_multisource_sentiment(
                reddit_posts=reddit_posts,
                telegram_messages=telegram_messages,
                twitter_tweets=twitter_tweets,
                source_weights=source_weights
            )
            
            all_sentiment[symbol] = sentiment_data.get('compound_score', 0)
            
            # Log the breakdown by source
            source_breakdown = sentiment_data.get('source_breakdown', {})
            logger.info(f"Sentiment for {symbol}: {all_sentiment[symbol]} (overall)")
            logger.info(f"  Reddit: {source_breakdown.get('reddit', {}).get('compound_score', 0)} ({len(reddit_posts)} posts)")
            logger.info(f"  Telegram: {source_breakdown.get('telegram', {}).get('compound_score', 0)} ({len(telegram_messages)} messages)")
            logger.info(f"  Twitter: {source_breakdown.get('twitter', {}).get('compound_score', 0)} ({len(twitter_tweets)} tweets)")
        else:
            all_sentiment[symbol] = 0
            logger.warning(f"No data found for {symbol} from any source")
    
    # Create a simple dataframe with sentiment scores
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days_back)]
    sentiment_df = pd.DataFrame({
        'date': dates
    })
    
    # Add sentiment for each symbol
    for symbol in symbols:
        # In a real-world scenario, we would have historical sentiment
        # Here we're just using the same sentiment for all days as a placeholder
        sentiment_df[f'{symbol}_sentiment'] = all_sentiment.get(symbol, 0)
    
    return sentiment_df

def merge_sentiment_with_market_data(market_data, sentiment_data):
    """
    Merge market data with sentiment data.
    
    Args:
        market_data: DataFrame with market data
        sentiment_data: DataFrame with sentiment data
        
    Returns:
        Merged DataFrame
    """
    # Convert date formats to be compatible
    market_data = market_data.reset_index()
    market_data['date'] = market_data['timestamp'].dt.strftime('%Y-%m-%d')
    
    # Merge on date
    merged_data = pd.merge(
        market_data,
        sentiment_data,
        on='date',
        how='left'
    )
    
    # Fill missing sentiment with neutral (0)
    for col in merged_data.columns:
        if '_sentiment' in col:
            merged_data[col] = merged_data[col].fillna(0)
    
    # Set index back to timestamp
    merged_data = merged_data.set_index('timestamp')
    
    return merged_data

def train_models(training_date=None):
    """
    Train models for all configured trading pairs.
    
    Args:
        training_date: Optional specific date for training
        
    Returns:
        Dictionary with training results
    """
    # Configure trading pairs
    trading_pairs = [
        ('BTC', 'USDT'),
        ('ETH', 'USDT'),
        ('SOL', 'USDT')
    ]
    
    results = {}
    
    # Ensure directories exist
    setup_directories()
    
    for base, quote in trading_pairs:
        symbol = f"{base}/{quote}"
        symbol_name = f"{base}_{quote}"
        
        logger.info(f"Training model for {symbol}")
        
        try:
            # 1. Fetch historical market data
            days_back = 180  # Use 6 months of data
            end_date = datetime.now() if training_date is None else training_date
            start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"Fetching historical data from {start_date} to {end_date}")
            market_data = fetch_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if market_data.empty:
                logger.error(f"Failed to fetch market data for {symbol}")
                continue
            
            logger.info(f"Fetched {len(market_data)} data points for {symbol}")
            
            # 2. Fetch sentiment data
            sentiment_data = fetch_sentiment_data([base], days_back=7)
            
            # 3. Merge sentiment with market data
            merged_data = merge_sentiment_with_market_data(market_data, sentiment_data)
            
            # 4. Prepare features and target
            features, target = prepare_features(merged_data)
            
            # Add sentiment as a feature if available
            sentiment_col = f"{base}_sentiment"
            if sentiment_col in merged_data.columns:
                features[sentiment_col] = merged_data[sentiment_col]
            
            logger.info(f"Prepared {len(features)} features with {features.shape[1]} columns")
            
            # 5. Train LightGBM model
            model, training_info = train_lightgbm(features, target)
            
            # 6. Evaluate model
            eval_metrics = evaluate_model(model, features, target)
            logger.info(f"Model evaluation: {eval_metrics}")
            
            # 7. Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            lightgbm_path = f"models/lightgbm/{symbol_name}_{timestamp}.txt"
            save_model(model, lightgbm_path)
            
            # 8. Export to ONNX
            onnx_path = f"models/onnx/{symbol_name}_{timestamp}.onnx"
            export_to_onnx(model, list(features.columns), onnx_path)
            
            # 9. Create symlinks to latest models
            latest_lightgbm = f"models/lightgbm/{symbol_name}_latest.txt"
            latest_onnx = f"models/onnx/{symbol_name}_latest.onnx"
            
            if os.path.exists(latest_lightgbm):
                os.remove(latest_lightgbm)
            if os.path.exists(latest_onnx):
                os.remove(latest_onnx)
                
            os.symlink(os.path.basename(lightgbm_path), latest_lightgbm)
            os.symlink(os.path.basename(onnx_path), latest_onnx)
            
            logger.info(f"Model training completed for {symbol}")
            
            # Store results
            results[symbol] = {
                'training_info': training_info,
                'eval_metrics': eval_metrics,
                'lightgbm_path': lightgbm_path,
                'onnx_path': onnx_path
            }
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            results[symbol] = {'error': str(e)}
    
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train cryptocurrency trading models')
    parser.add_argument('--date', type=str, help='Training date in YYYY-MM-DD format')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Parse training date if provided
    training_date = None
    if args.date:
        try:
            training_date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD.")
            sys.exit(1)
    
    # Run the training
    logger.info(f"Starting model training for {training_date or 'today'}")
    results = train_models(training_date)
    
    # Log overall results
    success_count = sum(1 for r in results.values() if 'error' not in r)
    logger.info(f"Training completed. Successful: {success_count}/{len(results)}")
    
    for symbol, result in results.items():
        if 'error' in result:
            logger.error(f"Failed for {symbol}: {result['error']}")
        else:
            logger.info(f"Success for {symbol}: RMSE={result['eval_metrics']['rmse']:.4f}, "
                       f"Directional Accuracy={result['eval_metrics']['directional_accuracy']:.4f}") 