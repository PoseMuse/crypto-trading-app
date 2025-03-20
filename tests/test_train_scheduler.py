"""
Unit tests for the model training scheduler
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Import the functions to test
from src.train_scheduler import (
    setup_directories,
    fetch_sentiment_data,
    merge_sentiment_with_market_data,
    train_models
)

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    # Create a synthetic daily price dataset
    np.random.seed(42)  # For reproducibility
    
    n_days = 20
    base_price = 10000.0
    
    # Generate timestamps
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate synthetic prices with some randomness and trend
    prices = []
    price = base_price
    for i in range(n_days):
        change = np.random.normal(0, 0.02)  # Random price change
        trend = 0.0005 * i  # Small upward trend
        price = price * (1 + change + trend)
        prices.append(price)
    
    # Create DataFrame with OHLCV data
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.03)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
        'volume': [np.random.uniform(100, 1000) for _ in range(n_days)]
    })
    
    return df.set_index('timestamp')

@pytest.fixture
def sample_sentiment_data():
    """Create sample sentiment data for testing."""
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    
    # Create DataFrame with sentiment data
    df = pd.DataFrame({
        'date': dates,
        'BTC_sentiment': [np.random.uniform(-1, 1) for _ in range(7)],
        'ETH_sentiment': [np.random.uniform(-1, 1) for _ in range(7)]
    })
    
    return df

def test_setup_directories(tmpdir):
    """Test directory setup functionality."""
    # Change to a temporary directory
    original_dir = os.getcwd()
    os.chdir(tmpdir)
    
    try:
        # Call the function
        setup_directories()
        
        # Check that all directories were created
        assert os.path.exists('logs')
        assert os.path.exists('models')
        assert os.path.exists('models/lightgbm')
        assert os.path.exists('models/onnx')
        assert os.path.exists('data/sentiment')
    finally:
        # Change back to the original directory
        os.chdir(original_dir)

@patch('src.sentiment_analysis.sentiment_pipeline.fetch_reddit_posts')
@patch('src.sentiment_analysis.sentiment_pipeline.aggregate_sentiment')
def test_fetch_sentiment_data(mock_aggregate, mock_fetch):
    """Test fetching sentiment data."""
    # Mock the fetch_reddit_posts function
    mock_fetch.return_value = [{'id': '1', 'title': 'Test', 'body': 'Test post'}]
    
    # Mock the aggregate_sentiment function
    mock_aggregate.return_value = {'compound_score': 0.5}
    
    # Call the function
    result = fetch_sentiment_data(['BTC', 'ETH'], days_back=3)
    
    # Verify the result
    assert isinstance(result, pd.DataFrame)
    assert 'date' in result.columns
    assert 'BTC_sentiment' in result.columns
    assert 'ETH_sentiment' in result.columns
    assert len(result) == 3  # days_back=3

def test_merge_sentiment_with_market_data(sample_market_data, sample_sentiment_data):
    """Test merging market data with sentiment data."""
    # Call the function
    result = merge_sentiment_with_market_data(sample_market_data, sample_sentiment_data)
    
    # Verify the result
    assert isinstance(result, pd.DataFrame)
    assert 'open' in result.columns
    assert 'close' in result.columns
    assert 'BTC_sentiment' in result.columns
    assert result.index.name == 'timestamp'

@patch('src.train_scheduler.fetch_historical_data')
@patch('src.train_scheduler.fetch_sentiment_data')
@patch('src.train_scheduler.merge_sentiment_with_market_data')
@patch('src.train_scheduler.prepare_features')
@patch('src.train_scheduler.train_lightgbm')
@patch('src.train_scheduler.evaluate_model')
@patch('src.train_scheduler.save_model')
@patch('src.train_scheduler.export_to_onnx')
@patch('os.symlink')
def test_train_models(mock_symlink, mock_export, mock_save, mock_evaluate, 
                     mock_train, mock_prepare, mock_merge, mock_fetch_sentiment,
                     mock_fetch_historical, sample_market_data, sample_sentiment_data,
                     tmpdir):
    """Test the full model training process with mocks."""
    # Change to temporary directory
    original_dir = os.getcwd()
    os.chdir(tmpdir)
    
    try:
        # Create the necessary directories
        setup_directories()
        
        # Mock fetch_historical_data
        mock_fetch_historical.return_value = sample_market_data
        
        # Mock fetch_sentiment_data
        mock_fetch_sentiment.return_value = sample_sentiment_data
        
        # Mock merge_sentiment_with_market_data
        merged_data = sample_market_data.copy()
        merged_data['BTC_sentiment'] = 0.5
        mock_merge.return_value = merged_data
        
        # Mock prepare_features
        features = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 10),
            'feature2': np.random.normal(0, 1, 10)
        })
        target = pd.Series(np.random.normal(0, 1, 10))
        mock_prepare.return_value = (features, target)
        
        # Mock train_lightgbm
        mock_model = MagicMock()
        training_info = {'best_iteration': 10}
        mock_train.return_value = (mock_model, training_info)
        
        # Mock evaluate_model
        eval_metrics = {'rmse': 0.1, 'directional_accuracy': 0.6}
        mock_evaluate.return_value = eval_metrics
        
        # Mock save_model and export_to_onnx
        mock_save.return_value = None
        mock_export.return_value = 'models/onnx/test.onnx'
        
        # Mock os.symlink
        mock_symlink.return_value = None
        
        # Call the function
        results = train_models()
        
        # Verify the results
        assert isinstance(results, dict)
        assert 'BTC/USDT' in results
        assert 'training_info' in results['BTC/USDT']
        assert 'eval_metrics' in results['BTC/USDT']
        assert 'lightgbm_path' in results['BTC/USDT']
        assert 'onnx_path' in results['BTC/USDT']
        
        # Verify that the mock functions were called
        mock_fetch_historical.assert_called()
        mock_fetch_sentiment.assert_called()
        mock_merge.assert_called()
        mock_prepare.assert_called()
        mock_train.assert_called()
        mock_evaluate.assert_called()
        mock_save.assert_called()
        mock_export.assert_called()
    finally:
        # Restore the original directory
        os.chdir(original_dir) 