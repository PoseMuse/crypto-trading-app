#!/usr/bin/env python
"""
Example script showing how to use the backtesting pipeline.

This script demonstrates how to load data, prepare it for backtesting, 
run a backtest with the CryptoStrategy, and visualize the results.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from backtesting.backtesting_pipeline import (
    CryptoStrategy,
    prepare_backtest_data,
    load_data_from_csv,
    run_backtest,
    plot_backtest_results
)


def generate_sample_data(days=100, save_path=None):
    """
    Generate sample price, AI signal, and sentiment data for demonstration.
    
    Args:
        days: Number of days of data to generate
        save_path: Path to save the data (if None, don't save)
        
    Returns:
        Tuple of (price_data, ai_signals, sentiment_data)
    """
    # Create date range
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create price data with a realistic pattern
    np.random.seed(42)  # For reproducibility
    
    # Start with a base price
    base_price = 50000
    
    # Generate daily returns with a slight upward bias and volatility
    daily_returns = np.random.normal(0.001, 0.02, size=len(dates))
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + daily_returns)
    
    # Calculate close prices
    close_prices = base_price * cumulative_returns
    
    # Create price DataFrame
    price_data = pd.DataFrame(index=dates)
    price_data['close'] = close_prices
    
    # Calculate realistic open, high, low prices
    daily_volatility = 0.015
    price_data['open'] = price_data['close'].shift(1).fillna(price_data['close'][0])
    price_data['high'] = price_data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, daily_volatility, size=len(dates)))
    price_data['low'] = price_data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, daily_volatility, size=len(dates)))
    
    # Generate trading volume (higher on volatile days)
    returns_abs = np.abs(daily_returns)
    normalized_returns = returns_abs / returns_abs.max()
    base_volume = 1000
    variable_volume = 4000
    price_data['volume'] = base_volume + variable_volume * normalized_returns * np.random.uniform(0.5, 1.5, size=len(dates))
    
    # Create AI signals data
    ai_data = pd.DataFrame(index=dates)
    
    # Generate AI signals with a combination of trend and oscillations
    t = np.linspace(0, 2 * np.pi, len(dates))
    trend = np.linspace(-0.3, 0.3, len(dates))  # Underlying trend
    oscillation = np.sin(t * 3) * 0.3  # Oscillating pattern
    noise = np.random.normal(0, 0.1, size=len(dates))  # Random noise
    
    ai_data['ai_signal'] = trend + oscillation + noise
    
    # Clip to range [-1, 1]
    ai_data['ai_signal'] = np.clip(ai_data['ai_signal'], -1, 1)
    
    # Create sentiment data
    sentiment_data = pd.DataFrame(index=dates)
    
    # Generate sentiment data with a different pattern
    t = np.linspace(0, 4 * np.pi, len(dates))
    trend = np.linspace(0.2, -0.2, len(dates))  # Underlying trend (opposite to AI)
    oscillation = np.cos(t * 2) * 0.25  # Different oscillating pattern
    noise = np.random.normal(0, 0.15, size=len(dates))  # More random noise
    
    sentiment_data['sentiment'] = trend + oscillation + noise
    
    # Clip to range [-1, 1]
    sentiment_data['sentiment'] = np.clip(sentiment_data['sentiment'], -1, 1)
    
    # Save data if requested
    if save_path:
        # Create directory if it doesn't exist
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save price data
        price_data.to_csv(os.path.join(save_path, 'price_data.csv'))
        
        # Save AI signals
        ai_data.to_csv(os.path.join(save_path, 'ai_signals.csv'))
        
        # Save sentiment data
        sentiment_data.to_csv(os.path.join(save_path, 'sentiment_data.csv'))
        
        print(f"Sample data saved to {save_path}")
    
    return price_data, ai_data, sentiment_data


def run_example_backtest():
    """
    Run an example backtest using generated sample data.
    """
    # Create directories for data and results
    data_dir = "data/sample"
    results_dir = "output/example_backtest"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate sample data
    print("Generating sample data...")
    price_data, ai_data, sentiment_data = generate_sample_data(days=365, save_path=data_dir)
    
    # Prepare data for backtesting
    print("Preparing backtest data...")
    data = prepare_backtest_data(
        price_data=price_data,
        ai_signals=ai_data,
        sentiment_data=sentiment_data
    )
    
    # Define strategy parameters
    strategy_params = {
        'ai_threshold': 0.4,        # Threshold for AI signals
        'sentiment_threshold': 0.3,  # Threshold for sentiment signals
        'ai_weight': 0.65,           # Weight for AI signals vs sentiment
        'stop_loss': 0.05,           # Stop loss percentage
        'take_profit': 0.15,         # Take profit percentage
        'position_size': 0.95        # Position size as percentage of portfolio
    }
    
    # Run backtest
    print("Running backtest...")
    metrics = run_backtest(
        data=data,
        strategy=CryptoStrategy,
        strategy_params=strategy_params,
        initial_cash=10000,
        commission=0.001
    )
    
    # Plot and save results
    print("Generating plot...")
    plot_file = os.path.join(results_dir, "backtest_results.png")
    plot_backtest_results(
        data=data,
        metrics=metrics,
        save_path=plot_file
    )
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(results_dir, "backtest_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    
    print(f"\nBacktest results saved to {results_dir}")
    print(f"- Plot: {plot_file}")
    print(f"- Metrics: {metrics_file}")


if __name__ == "__main__":
    run_example_backtest() 