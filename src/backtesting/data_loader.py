"""
Data Loader module for backtesting.

This module provides functionality to load and prepare data for backtesting
trading strategies using Backtrader.
"""

import os
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List, Any, Union

from ..ai_models.model_pipeline import fetch_historical_data
from ..sentiment_analysis.sentiment_pipeline import (
    fetch_reddit_posts, 
    aggregate_sentiment,
    aggregate_multisource_sentiment
)
from ..sentiment_analysis.telegram_pipeline import fetch_telegram_messages
from ..sentiment_analysis.twitter_pipeline import fetch_twitter_tweets


class CCXTCsvData(bt.feeds.GenericCSVData):
    """
    Data feed for CCXT data saved as CSV files.
    
    This class allows Backtrader to use OHLCV data from cryptocurrency exchanges.
    """
    # Add a sentiment line
    lines = ('sentiment',)
    
    # Add sentiment to the list of columns
    params = (
        ('nullvalue', 0.0),
        ('dtformat', '%Y-%m-%d %H:%M:%S'),
        ('datetime', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('sentiment', 6),
    )


def fetch_and_prepare_data(
    symbol: str,
    timeframe: str = '1d',
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    include_sentiment: bool = True,
    use_multi_source: bool = True,
    sentiment_lookback: int = 7,
    output_dir: str = 'data/backtest',
    use_cached: bool = True,
    exchange_manager: Any = None
) -> str:
    """
    Fetch historical price data and prepare it for backtesting.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        timeframe: Timeframe for data ('1d', '1h', etc.)
        start_date: Start date for historical data
        end_date: End date for historical data
        include_sentiment: Whether to include sentiment data
        use_multi_source: Whether to use multi-source sentiment (Reddit, Telegram, Twitter)
        sentiment_lookback: Days to apply sentiment data retrospectively
        output_dir: Directory to save prepared data
        use_cached: Whether to use cached data if available
        exchange_manager: Exchange manager instance or None
        
    Returns:
        Path to the prepared CSV file
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365)  # 1 year of data
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    base_currency = symbol.split('/')[0].lower()
    quote_currency = symbol.split('/')[1].lower()
    date_str = datetime.now().strftime('%Y%m%d')
    sentiment_suffix = ""
    if include_sentiment:
        sentiment_suffix = "_with_sentiment"
        if use_multi_source:
            sentiment_suffix += "_multi"
    
    filename = f"{base_currency}_{quote_currency}_{timeframe}_{date_str}{sentiment_suffix}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Check if file already exists and we can use cached data
    if use_cached and os.path.exists(filepath):
        print(f"Using cached data from {filepath}")
        return filepath
    
    # Fetch historical price data
    print(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
    df = fetch_historical_data(
        symbol=symbol,
        exchange_manager=exchange_manager,
        start_date=start_date,
        end_date=end_date
    )
    
    if df.empty:
        raise ValueError(f"No data available for {symbol}")
    
    # Reset index to have datetime as a column
    df = df.reset_index()
    
    # Add sentiment data if requested
    if include_sentiment:
        print(f"Adding {'multi-source' if use_multi_source else 'standard'} sentiment data...")
        sentiment_data = get_historical_sentiment(
            symbol=symbol,
            dates=df['timestamp'].tolist(),
            lookback_days=sentiment_lookback,
            output_dir=os.path.join(output_dir, '../sentiment'),
            use_multi_source=use_multi_source
        )
        
        # Initialize sentiment columns
        df['sentiment'] = 0.0
        if use_multi_source:
            df['reddit_sentiment'] = 0.0
            df['telegram_sentiment'] = 0.0
            df['twitter_sentiment'] = 0.0
        
        # Apply sentiment data
        for date_str, sentiment_value in sentiment_data.items():
            try:
                # Parse the date string
                date = datetime.fromisoformat(date_str)
                
                # Find matching rows in the dataframe
                mask = (df['timestamp'] >= date - timedelta(days=sentiment_lookback)) & (df['timestamp'] <= date)
                
                # Apply sentiment to matching rows
                df.loc[mask, 'sentiment'] = sentiment_value
            except Exception as e:
                print(f"Error applying sentiment for date {date_str}: {e}")
    else:
        # Add zeros columns for sentiment
        df['sentiment'] = 0.0
        
    # Ensure proper column order for backtrader - add additional columns as needed
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'sentiment']
    if include_sentiment and use_multi_source:
        columns.extend(['reddit_sentiment', 'telegram_sentiment', 'twitter_sentiment'])
    
    # Select only columns that exist in the dataframe
    existing_columns = [col for col in columns if col in df.columns]
    df = df[existing_columns]
    
    # Save to CSV
    df.to_csv(filepath, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f"Data saved to {filepath}")
    
    return filepath


def get_historical_sentiment(
    symbol: str,
    dates: List[datetime],
    lookback_days: int = 7,
    output_dir: str = 'data/sentiment',
    use_multi_source: bool = True
) -> Dict[str, float]:
    """
    Generate or retrieve historical sentiment data for a series of dates.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        dates: List of dates to get sentiment for
        lookback_days: Days to look back for sentiment
        output_dir: Directory to store sentiment data
        use_multi_source: Whether to use multi-source sentiment (Reddit, Telegram, Twitter)
        
    Returns:
        Dictionary mapping date strings to sentiment scores
    """
    # Extract base currency for sentiment analysis
    base_currency = symbol.split('/')[0].lower()
    
    # Path to store sentiment data
    os.makedirs(output_dir, exist_ok=True)
    sentiment_suffix = "_multi" if use_multi_source else ""
    sentiment_file = os.path.join(output_dir, f"{base_currency}_historical_sentiment{sentiment_suffix}.json")
    
    # Check if historical sentiment data already exists
    if os.path.exists(sentiment_file):
        try:
            import json
            with open(sentiment_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading sentiment data: {e}")
    
    # Data doesn't exist, fetch real sentiment data
    print(f"Fetching historical sentiment data for {base_currency}")
    sentiment_data = {}
    
    # Map currency to relevant subreddits and search terms
    mapping = {
        'btc': {
            'subreddits': ['bitcoin', 'cryptocurrency'],
            'search_terms': ['bitcoin', 'btc', 'crypto'],
            'telegram_channels': ['bitcoinsignals', 'crypto_trading'],
        },
        'eth': {
            'subreddits': ['ethereum', 'cryptocurrency'],
            'search_terms': ['ethereum', 'eth', 'crypto'],
            'telegram_channels': ['ethereumsignals', 'crypto_trading'],
        },
        'sol': {
            'subreddits': ['solana', 'cryptocurrency'],
            'search_terms': ['solana', 'sol', 'crypto'],
            'telegram_channels': ['solanasignals', 'crypto_trading'],
        },
        'ada': {
            'subreddits': ['cardano', 'cryptocurrency'],
            'search_terms': ['cardano', 'ada', 'crypto'],
            'telegram_channels': ['cardanosignals', 'crypto_trading'],
        },
        'xrp': {
            'subreddits': ['ripple', 'cryptocurrency'],
            'search_terms': ['ripple', 'xrp', 'crypto'],
            'telegram_channels': ['xrpsignals', 'crypto_trading'],
        },
        'doge': {
            'subreddits': ['dogecoin', 'cryptocurrency'], 
            'search_terms': ['dogecoin', 'doge', 'crypto'],
            'telegram_channels': ['dogesignals', 'crypto_trading'],
        },
        # Default case
        'default': {
            'subreddits': ['cryptocurrency'],
            'search_terms': ['crypto', 'cryptocurrency'],
            'telegram_channels': ['crypto_trading'],
        }
    }
    
    # Get the relevant search parameters for this currency
    currency_info = mapping.get(base_currency.lower(), mapping['default'])
    subreddits = currency_info['subreddits']
    search_terms = currency_info['search_terms']
    telegram_channels = currency_info['telegram_channels']
    
    # Group dates by year-month to reduce API calls
    date_groups = {}
    for date in dates:
        year_month = date.strftime('%Y-%m')
        if year_month not in date_groups:
            date_groups[year_month] = []
        date_groups[year_month].append(date)
    
    # Process each month
    for year_month, month_dates in date_groups.items():
        # Use the middle date of the month for sentiment analysis
        reference_date = month_dates[len(month_dates) // 2]
        
        # Fetch sentiment data for each source
        if use_multi_source:
            # Reddit
            reddit_posts = fetch_reddit_posts(
                subreddits=subreddits,
                limit=100,
                time_filter='month'
            )
            
            # Telegram
            telegram_messages = []
            for channel in telegram_channels:
                channel_messages = fetch_telegram_messages(
                    channel_username=channel,
                    limit=100
                )
                telegram_messages.extend(channel_messages)
            
            # Twitter
            twitter_tweets = []
            for term in search_terms:
                tweets = fetch_twitter_tweets(
                    query=term,
                    limit=100,
                    days_back=lookback_days
                )
                twitter_tweets.extend(tweets)
            
            # Aggregate sentiment
            aggregated = aggregate_multisource_sentiment(
                reddit_posts=reddit_posts,
                telegram_messages=telegram_messages,
                twitter_tweets=twitter_tweets
            )
            
            sentiment_score = aggregated.get('compound_score', 0.0)
        else:
            # Reddit only
            reddit_posts = fetch_reddit_posts(
                subreddits=subreddits,
                limit=100,
                time_filter='month'
            )
            
            aggregated = aggregate_sentiment(reddit_posts)
            sentiment_score = aggregated.get('compound_score', 0.0)
        
        # Apply the sentiment score to all dates in this month
        for date in month_dates:
            sentiment_data[date.isoformat()] = sentiment_score
    
    # Save the sentiment data for future use
    try:
        import json
        with open(sentiment_file, 'w') as f:
            json.dump(sentiment_data, f)
    except Exception as e:
        print(f"Error saving sentiment data: {e}")
    
    return sentiment_data

def load_backtest_data(
    filepath: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> bt.feeds.DataBase:
    """
    Load data from CSV file for backtesting.
    
    Args:
        filepath: Path to the CSV file
        start_date: Start date for backtesting
        end_date: End date for backtesting
        
    Returns:
        Backtrader data feed
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Create data feed
    data = CCXTCsvData(
        dataname=filepath,
        fromdate=start_date,
        todate=end_date,
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        sentiment=6,
        openinterest=-1  # Not used
    )
    
    return data 