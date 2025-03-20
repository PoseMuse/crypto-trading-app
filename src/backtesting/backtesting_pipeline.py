"""
Backtesting pipeline for cryptocurrency trading strategies.

This module provides a framework for backtesting trading strategies on historical data,
with support for technical indicators, machine learning signals, and sentiment analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os


class CryptoStrategy:
    """
    Base strategy for cryptocurrency trading with AI signals and sentiment analysis.
    
    This strategy combines AI predictions and sentiment analysis to make trading decisions.
    """
    
    def __init__(self, ai_threshold=0.5, sentiment_threshold=0.3, ai_weight=0.7,
                 stop_loss=0.05, take_profit=0.1, position_size=1.0):
        """
        Initialize the strategy with parameters.
        
        Args:
            ai_threshold: Threshold for AI signals (-1 to 1)
            sentiment_threshold: Threshold for sentiment signals (-1 to 1)
            ai_weight: Weight for AI vs sentiment (0.0 to 1.0)
            stop_loss: Stop loss percentage (0.0 to 1.0)
            take_profit: Take profit percentage (0.0 to 1.0)
            position_size: Size of position relative to portfolio (0.0 to 1.0)
        """
        self.ai_threshold = ai_threshold
        self.sentiment_threshold = sentiment_threshold
        self.ai_weight = ai_weight
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        
        # Trading state
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.entry_price = 0.0
        self.cash = 0.0
        self.holdings = 0.0
        self.equity = []
        self.trades = []
    
    def initialize(self, initial_cash):
        """Initialize the strategy with starting cash."""
        self.cash = initial_cash
        self.equity = [initial_cash]
        self.trades = []
    
    def calculate_position_signal(self, ai_signal, sentiment):
        """
        Calculate the combined position signal from AI and sentiment.
        
        Returns:
            int: -1 for short, 0 for neutral, 1 for long
        """
        # Combine signals according to weights
        combined_signal = (ai_signal * self.ai_weight) + (sentiment * (1 - self.ai_weight))
        
        # Determine position based on thresholds
        if combined_signal > self.ai_threshold:
            return 1  # Long
        elif combined_signal < -self.ai_threshold:
            return -1  # Short
        else:
            return 0  # Neutral
    
    def on_data(self, date, price, ai_signal, sentiment, commission=0.001):
        """
        Process a new data point and make trading decisions.
        
        Args:
            date: Date/time of the data point
            price: Current price
            ai_signal: AI prediction signal (-1 to 1)
            sentiment: Sentiment signal (-1 to 1)
            commission: Trading commission as a percentage
        """
        # Calculate the position signal
        signal = self.calculate_position_signal(ai_signal, sentiment)
        
        # Check stop loss and take profit if in a position
        if self.position != 0:
            pnl_pct = (price / self.entry_price - 1) * self.position
            
            # Exit on stop loss
            if pnl_pct <= -self.stop_loss:
                # Record trade
                self.trades.append({
                    'entry_date': self.entry_date,
                    'entry_price': self.entry_price,
                    'exit_date': date,
                    'exit_price': price,
                    'position': self.position,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'stop_loss'
                })
                
                # Close position
                self.cash = self.cash + (self.holdings * price * (1 - commission))
                self.holdings = 0
                self.position = 0
            
            # Exit on take profit
            elif pnl_pct >= self.take_profit:
                # Record trade
                self.trades.append({
                    'entry_date': self.entry_date,
                    'entry_price': self.entry_price,
                    'exit_date': date,
                    'exit_price': price,
                    'position': self.position,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'take_profit'
                })
                
                # Close position
                self.cash = self.cash + (self.holdings * price * (1 - commission))
                self.holdings = 0
                self.position = 0
        
        # Execute new position if signal differs from current position
        if signal != 0 and signal != self.position:
            # Close existing position if any
            if self.position != 0:
                pnl_pct = (price / self.entry_price - 1) * self.position
                
                # Record trade
                self.trades.append({
                    'entry_date': self.entry_date,
                    'entry_price': self.entry_price,
                    'exit_date': date,
                    'exit_price': price,
                    'position': self.position,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'signal_change'
                })
                
                # Close position
                self.cash = self.cash + (self.holdings * price * (1 - commission))
                self.holdings = 0
            
            # Enter new position
            position_value = self.cash * self.position_size
            self.holdings = position_value / price * signal
            self.cash = self.cash - (position_value * (1 + commission))
            self.position = signal
            self.entry_price = price
            self.entry_date = date
        
        # Update equity
        current_equity = self.cash + (self.holdings * price)
        self.equity.append(current_equity)
        
        return current_equity

def load_data_from_csv(price_file, ai_signals_file=None, sentiment_file=None):
    """
    Load CSV data for backtesting.
    
    Args:
        price_file: Path to price data CSV (must have 'close' column)
        ai_signals_file: Path to AI signals CSV (optional)
        sentiment_file: Path to sentiment data CSV (optional)
        
    Returns:
        tuple: (price_data, ai_signals, sentiment_data)
    """
    # Load price data
    price_data = pd.read_csv(price_file, index_col=0, parse_dates=True)
    
    # Load AI signals if provided
    ai_signals = None
    if ai_signals_file:
        ai_signals = pd.read_csv(ai_signals_file, index_col=0, parse_dates=True)
    
    # Load sentiment data if provided
    sentiment_data = None
    if sentiment_file:
        sentiment_data = pd.read_csv(sentiment_file, index_col=0, parse_dates=True)
    
    return price_data, ai_signals, sentiment_data

def prepare_backtest_data(price_data, ai_signals=None, sentiment_data=None):
    """
    Prepare data for backtesting by aligning all datasets.
    
    Args:
        price_data: DataFrame with price data
        ai_signals: DataFrame with AI signals (optional)
        sentiment_data: DataFrame with sentiment data (optional)
        
    Returns:
        DataFrame: Combined data for backtesting
    """
    # Create a combined DataFrame with price data
    data = price_data.copy()
    
    # Add AI signals if provided
    if ai_signals is not None:
        # Merge on index (date)
        data = data.join(ai_signals, how='left')
    else:
        # Add a neutral signal
        data['ai_signal'] = 0.0
    
    # Add sentiment data if provided
    if sentiment_data is not None:
        data = data.join(sentiment_data, how='left')
    else:
        # Add a neutral sentiment
        data['sentiment'] = 0.0
    
    # Forward fill missing values
    data = data.fillna(method='ffill')
    
    # Drop rows with any remaining NaN values
    data = data.dropna()
    
    return data

def run_backtest(data, strategy, strategy_params, initial_cash=10000, commission=0.001):
    """
    Run a backtest on the provided data with the specified strategy.
    
    Args:
        data: DataFrame with price, AI, and sentiment data
        strategy: Strategy class to use
        strategy_params: Dictionary of parameters for the strategy
        initial_cash: Initial cash for the backtest
        commission: Trading commission as a percentage
        
    Returns:
        dict: Metrics from the backtest
    """
    # Initialize the strategy
    strategy_instance = strategy(**strategy_params)
    strategy_instance.initialize(initial_cash)
    
    # Run the backtest
    for index, row in data.iterrows():
        strategy_instance.on_data(
            date=index,
            price=row['close'],
            ai_signal=row['ai_signal'],
            sentiment=row['sentiment'],
            commission=commission
        )
    
    # Calculate metrics
    equity = np.array(strategy_instance.equity)
    returns = np.diff(equity) / equity[:-1]
    
    # Collect trades
    trades = pd.DataFrame(strategy_instance.trades)
    
    # Calculate basic metrics
    total_return = (equity[-1] / equity[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    # Win rate
    if len(trades) > 0:
        win_rate = len(trades[trades['pnl_pct'] > 0]) / len(trades)
    else:
        win_rate = 0
    
    # Maximum drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Final metrics
    metrics = {
        'initial_equity': equity[0],
        'final_equity': equity[-1],
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': len(trades),
    }
    
    return metrics

def plot_backtest_results(data, metrics, save_path=None):
    """
    Plot the results of a backtest.
    
    Args:
        data: The data used for the backtest
        metrics: Dictionary of backtest metrics
        save_path: Path to save the plot (optional)
        
    Returns:
        None
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot 1: Price
    axes[0].plot(data.index, data['close'], label='Price')
    axes[0].set_title('Backtest Results')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: AI Signal
    axes[1].plot(data.index, data['ai_signal'], label='AI Signal', color='green')
    axes[1].set_ylabel('AI Signal')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Sentiment
    axes[2].plot(data.index, data['sentiment'], label='Sentiment', color='purple')
    axes[2].set_ylabel('Sentiment')
    axes[2].set_xlabel('Date')
    axes[2].legend()
    axes[2].grid(True)
    
    # Add metrics as text
    metrics_text = (
        f"Total Return: {metrics['total_return']:.2%}\n"
        f"Annualized Return: {metrics['annualized_return']:.2%}\n"
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
        f"Win Rate: {metrics['win_rate']:.2%}\n"
        f"Number of Trades: {metrics['num_trades']}"
    )
    
    axes[0].annotate(metrics_text, xy=(0.02, 0.02), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig 