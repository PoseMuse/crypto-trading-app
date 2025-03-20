"""
Backtesting package for cryptocurrency trading strategies.

This package provides tools and utilities for backtesting trading strategies
using historical price data, AI signals, and sentiment analysis.
"""

from src.backtesting.backtesting_pipeline import (
    CryptoStrategy,
    prepare_backtest_data,
    load_data_from_csv,
    run_backtest,
    plot_backtest_results
)

__all__ = [
    'CryptoStrategy',
    'prepare_backtest_data',
    'load_data_from_csv',
    'run_backtest',
    'plot_backtest_results'
] 