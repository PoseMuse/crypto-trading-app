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
from ..sentiment_analysis.sentiment_pipeline import fetch_reddit_posts, aggregate_sentiment


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
    sentiment_suffix = '_with_sentiment' if include_sentiment else ''
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
        print("Adding sentiment data...")
        sentiment_data = get_historical_sentiment(
            symbol=symbol,
            dates=df['timestamp'].tolist(),
            lookback_days=sentiment_lookback,
            output_dir=output_dir
        )
        
        # Initialize sentiment column with zeros
        df['sentiment'] = 0.0
        
        # Apply sentiment data
        for date_str, sentiment in sentiment_data.items():
            try:
                # Parse the date string
                date = datetime.fromisoformat(date_str)
                
                # Find matching rows in the dataframe
                mask = (df['timestamp'] >= date - timedelta(days=sentiment_lookback)) & (df['timestamp'] <= date)
                
                # Apply sentiment to matching rows
                df.loc[mask, 'sentiment'] = sentiment
            except Exception as e:
                print(f"Error applying sentiment for date {date_str}: {e}")
    else:
        # Add a zeros column for sentiment
        df['sentiment'] = 0.0
    
    # Ensure proper column order for backtrader
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'sentiment']]
    
    # Save to CSV
    df.to_csv(filepath, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f"Data saved to {filepath}")
    
    return filepath


def get_historical_sentiment(
    symbol: str,
    dates: List[datetime],
    lookback_days: int = 7,
    output_dir: str = 'data/sentiment'
) -> Dict[str, float]:
    """
    Generate or retrieve historical sentiment data for a series of dates.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        dates: List of dates to get sentiment for
        lookback_days: Days to look back for sentiment
        output_dir: Directory to store sentiment data
        
    Returns:
        Dictionary mapping date strings to sentiment scores
    """
    # Extract base currency for sentiment analysis
    base_currency = symbol.split('/')[0].lower()
    
    # Path to store sentiment data
    os.makedirs(output_dir, exist_ok=True)
    sentiment_file = os.path.join(output_dir, f"{base_currency}_historical_sentiment.json")
    
    # Check if historical sentiment data already exists
    if os.path.exists(sentiment_file):
        try:
            import json
            with open(sentiment_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading sentiment data: {e}")
    
    # If not, we'll generate simulated historical sentiment
    # This is a mock implementation for backtesting purposes
    # In a real-world scenario, you would retrieve actual historical sentiment data
    print(f"Generating mock historical sentiment data for {symbol}")
    
    sentiment_data = {}
    
    # Map currency to relevant subreddits
    subreddit_mapping = {
        'btc': ['bitcoin', 'cryptocurrency'],
        'eth': ['ethereum', 'cryptocurrency'],
        'sol': ['solana', 'cryptocurrency'],
        'ada': ['cardano', 'cryptocurrency'],
        'xrp': ['ripple', 'cryptocurrency'],
        'doge': ['dogecoin', 'cryptocurrency'],
        # Add more as needed
    }
    
    # Get relevant subreddits
    subreddits = subreddit_mapping.get(base_currency, ['cryptocurrency'])
    
    # Generate sentiment for each month in the date range
    unique_months = set()
    for date in dates:
        # Group by month to reduce the number of data points
        month_key = date.strftime('%Y-%m-01')
        unique_months.add(month_key)
    
    # For each month, generate a sentiment value
    import random
    from math import sin, pi
    
    # Sort months for deterministic generation
    sorted_months = sorted(list(unique_months))
    
    # Generate sentiment with some cyclical patterns plus noise
    for i, month in enumerate(sorted_months):
        # Base cyclical pattern (sine wave with 6-month period)
        cycle_component = 0.3 * sin(2 * pi * i / 6)
        
        # Trend component (slight upward trend)
        trend_component = 0.001 * i
        
        # Random component
        random_component = random.uniform(-0.2, 0.2)
        
        # Combine components
        sentiment = cycle_component + trend_component + random_component
        
        # Ensure within bounds [-1, 1]
        sentiment = max(-1.0, min(1.0, sentiment))
        
        # Store in dictionary
        sentiment_data[month] = sentiment
    
    # Save to file for future use
    try:
        import json
        with open(sentiment_file, 'w') as f:
            json.dump(sentiment_data, f, indent=2)
        print(f"Saved historical sentiment data to {sentiment_file}")
    except Exception as e:
        print(f"Error saving sentiment data: {e}")
    
    return sentiment_data


def load_backtest_data(
    filepath: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> bt.feeds.DataBase:
    """
    Load data from a CSV file for use with Backtrader.
    
    Args:
        filepath: Path to the CSV file
        start_date: Start date for the backtest
        end_date: End date for the backtest
        
    Returns:
        Backtrader data feed
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Create a data feed
    data = CCXTCsvData(
        dataname=filepath,
        fromdate=start_date,
        todate=end_date,
        timeframe=bt.TimeFrame.Days,  # Adjust based on your data
        nullvalue=0.0
    )
    
    return data 