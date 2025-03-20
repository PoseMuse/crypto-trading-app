"""
Tests for backtesting functionality.

This module contains tests for the backtesting pipeline and paper trading functionality.
"""

import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

# Import the components to test
from src.backtesting.backtesting_pipeline import (
    CryptoStrategy,
    SignalData,
    prepare_backtest_data,
    run_backtest,
    plot_backtest_results
)


class TestBacktestingPipeline(unittest.TestCase):
    """
    Test cases for the backtesting pipeline.
    """
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Generate test data
        self.price_data = self._generate_test_price_data()
        self.ai_data = self._generate_test_ai_data()
        self.sentiment_data = self._generate_test_sentiment_data()
    
    def tearDown(self):
        """Clean up after the tests."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_dir)
    
    def _generate_test_price_data(self) -> pd.DataFrame:
        """
        Generate test price data.
        
        Returns:
            DataFrame with test price data
        """
        # Create date range for test data
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate price data with a simple trend and volatility
        np.random.seed(42)  # For reproducibility
        close_prices = np.linspace(40000, 45000, len(dates))
        close_prices += np.random.normal(0, 500, size=len(dates))  # Add noise
        
        # Create DataFrame
        df = pd.DataFrame(index=dates)
        df['open'] = close_prices * (1 - np.random.uniform(0, 0.02, size=len(dates)))
        df['high'] = close_prices * (1 + np.random.uniform(0, 0.02, size=len(dates)))
        df['low'] = close_prices * (1 - np.random.uniform(0, 0.02, size=len(dates)))
        df['close'] = close_prices
        df['volume'] = np.random.uniform(1000, 5000, size=len(dates))
        
        return df
    
    def _generate_test_ai_data(self) -> pd.DataFrame:
        """
        Generate test AI signal data.
        
        Returns:
            DataFrame with test AI signal data
        """
        # Use the same date range as price data
        df = pd.DataFrame(index=self.price_data.index)
        
        # Generate AI signals with a sine pattern
        t = np.linspace(0, 4 * np.pi, len(df))
        df['ai_signal'] = np.sin(t) * 0.6
        
        return df
    
    def _generate_test_sentiment_data(self) -> pd.DataFrame:
        """
        Generate test sentiment data.
        
        Returns:
            DataFrame with test sentiment data
        """
        # Use the same date range as price data
        df = pd.DataFrame(index=self.price_data.index)
        
        # Generate sentiment data with a cosine pattern
        t = np.linspace(0, 4 * np.pi, len(df))
        df['sentiment'] = np.cos(t) * 0.4
        
        return df
    
    def test_prepare_backtest_data(self):
        """Test the prepare_backtest_data function."""
        # Prepare backtest data
        data = prepare_backtest_data(
            price_data=self.price_data,
            ai_signals=self.ai_data,
            sentiment_data=self.sentiment_data
        )
        
        # Check that the data has the expected columns
        self.assertIn('open', data.columns)
        self.assertIn('high', data.columns)
        self.assertIn('low', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)
        self.assertIn('ai_signal', data.columns)
        self.assertIn('sentiment', data.columns)
        
        # Check that the data has the expected length
        self.assertEqual(len(data), len(self.price_data))
        
        # Check that the data has the expected values
        np.testing.assert_array_equal(data['close'].values, self.price_data['close'].values)
        np.testing.assert_array_equal(data['ai_signal'].values, self.ai_data['ai_signal'].values)
        np.testing.assert_array_equal(data['sentiment'].values, self.sentiment_data['sentiment'].values)
    
    def test_prepare_backtest_data_missing_signals(self):
        """Test the prepare_backtest_data function with missing signal data."""
        # Prepare backtest data with missing AI signals
        data = prepare_backtest_data(
            price_data=self.price_data,
            sentiment_data=self.sentiment_data
        )
        
        # Check that the data has default AI signals
        self.assertIn('ai_signal', data.columns)
        self.assertEqual(data['ai_signal'].iloc[0], 0.0)
        
        # Prepare backtest data with missing sentiment data
        data = prepare_backtest_data(
            price_data=self.price_data,
            ai_signals=self.ai_data
        )
        
        # Check that the data has default sentiment signals
        self.assertIn('sentiment', data.columns)
        self.assertEqual(data['sentiment'].iloc[0], 0.0)
    
    def test_run_backtest(self):
        """Test the run_backtest function."""
        # Prepare backtest data
        data = prepare_backtest_data(
            price_data=self.price_data,
            ai_signals=self.ai_data,
            sentiment_data=self.sentiment_data
        )
        
        # Run backtest
        metrics = run_backtest(
            data=data,
            strategy=CryptoStrategy,
            strategy_params={
                'ai_threshold': 0.5,
                'sentiment_threshold': 0.3,
                'ai_weight': 0.7,
                'stop_loss': 0.05,
                'take_profit': 0.1
            },
            initial_cash=10000,
            commission=0.001
        )
        
        # Check that the metrics have the expected keys
        self.assertIn('initial_cash', metrics)
        self.assertIn('final_value', metrics)
        self.assertIn('pnl', metrics)
        self.assertIn('returns_pct', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown_pct', metrics)
        self.assertIn('annual_return_pct', metrics)
        self.assertIn('total_trades', metrics)
        self.assertIn('win_rate_pct', metrics)
        
        # Check that the metrics have reasonable values
        self.assertEqual(metrics['initial_cash'], 10000)
        self.assertGreater(metrics['final_value'], 0)
        self.assertIsInstance(metrics['pnl'], float)
        self.assertIsInstance(metrics['returns_pct'], float)
        self.assertIsInstance(metrics['sharpe_ratio'], float)
        self.assertGreaterEqual(metrics['max_drawdown_pct'], 0)
        self.assertLessEqual(metrics['max_drawdown_pct'], 100)
        self.assertIsInstance(metrics['annual_return_pct'], float)
        self.assertGreaterEqual(metrics['total_trades'], 0)
        self.assertGreaterEqual(metrics['win_rate_pct'], 0)
        self.assertLessEqual(metrics['win_rate_pct'], 100)
    
    def test_plot_backtest_results(self):
        """Test the plot_backtest_results function."""
        # Prepare backtest data
        data = prepare_backtest_data(
            price_data=self.price_data,
            ai_signals=self.ai_data,
            sentiment_data=self.sentiment_data
        )
        
        # Run backtest
        metrics = run_backtest(
            data=data,
            strategy=CryptoStrategy,
            initial_cash=10000,
            commission=0.001
        )
        
        # Generate plot file path
        plot_file = os.path.join(self.test_dir, "backtest_plot.png")
        
        # Plot backtest results
        plot_backtest_results(
            data=data,
            metrics=metrics,
            save_path=plot_file
        )
        
        # Check that the plot file exists
        self.assertTrue(os.path.exists(plot_file))
        self.assertGreater(os.path.getsize(plot_file), 0)


if __name__ == '__main__':
    unittest.main() 