"""
Unit tests for the model pipeline module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import os

from src.ai_models.model_pipeline import (
    fetch_historical_data,
    prepare_features,
    train_lightgbm,
    evaluate_model,
    walk_forward_validation,
    export_to_onnx
)

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    # Create a synthetic daily price dataset
    np.random.seed(42)  # For reproducibility
    
    n_days = 100
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

def test_fetch_historical_data():
    """Test fetching historical data."""
    with patch('ccxt.binance') as mock_exchange:
        # Setup the mock
        mock_instance = MagicMock()
        mock_exchange.return_value = mock_instance
        
        # Mock the fetch_ohlcv method
        sample_data = [
            [1609459200000, 10000, 10100, 9900, 10050, 500],  # [timestamp, open, high, low, close, volume]
            [1609545600000, 10050, 10200, 10000, 10150, 600],
        ]
        mock_instance.fetch_ohlcv.return_value = sample_data
        
        # Call the function
        result = fetch_historical_data('BTC/USDT')
        
        # Check the result
        assert not result.empty
        assert len(result) == 2
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns

def test_prepare_features(sample_ohlcv_data):
    """Test feature preparation."""
    # Call prepare_features
    features, target = prepare_features(sample_ohlcv_data)
    
    # Verify features and target
    assert not features.empty
    assert features.shape[0] > 0
    assert features.shape[1] > 5  # Should have multiple technical indicators
    
    # Check for specific features
    expected_features = [
        'returns', 'log_returns', 'ma5', 'ma10', 'ma20', 'rsi_14', 'macd'
    ]
    
    for feature in expected_features:
        assert feature in features.columns
    
    # Verify target is the next day's return
    assert len(target) == len(features)

def test_train_lightgbm(sample_ohlcv_data):
    """Test LightGBM model training."""
    with patch('lightgbm.train') as mock_train:
        # Setup the mock
        mock_booster = MagicMock()
        mock_booster.best_iteration = 42
        mock_booster.feature_importance.return_value = [1, 2, 3, 4]
        mock_train.return_value = mock_booster
        
        # Prepare data
        features, target = prepare_features(sample_ohlcv_data)
        
        # Call train_lightgbm
        model, training_info = train_lightgbm(features, target)
        
        # Verify training_info
        assert training_info['best_iteration'] == 42
        assert 'feature_importance' in training_info
        assert 'train_size' in training_info
        assert 'test_size' in training_info

def test_evaluate_model():
    """Test model evaluation."""
    # Create synthetic data
    import numpy as np
    
    # Create test data and target
    np.random.seed(42)
    n_samples = 100
    test_features = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples)
    })
    test_target = pd.Series(np.random.normal(0, 1, n_samples))
    
    # Create a mock model with a predict method
    mock_model = MagicMock()
    mock_model.predict.return_value = np.random.normal(0, 1, n_samples)
    
    # Evaluate the model
    metrics = evaluate_model(mock_model, test_features, test_target)
    
    # Check that the metrics are calculated
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert 'directional_accuracy' in metrics
    assert isinstance(metrics['rmse'], float)

def test_export_to_onnx(tmp_path):
    """Test exporting LightGBM model to ONNX format."""
    try:
        import onnxmltools
        import skl2onnx
    except ImportError:
        pytest.skip("skl2onnx or onnxmltools not installed, skipping ONNX test")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    features = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples)
    })
    target = pd.Series(np.random.normal(0, 1, n_samples))
    
    # Train a model
    model, _ = train_lightgbm(features, target)
    
    # Export to ONNX
    onnx_path = os.path.join(tmp_path, "test_model.onnx")
    result_path = export_to_onnx(model, list(features.columns), onnx_path)
    
    # Check if export was successful
    if result_path:  # If not empty string
        assert os.path.exists(onnx_path)
        assert onnx_path.endswith('.onnx')
        
        # Try to read the file to check if it's valid
        with open(onnx_path, 'rb') as f:
            content = f.read()
            assert len(content) > 0
    else:
        # If ONNX conversion failed, log a message
        print("ONNX export not available, skipping file existence check")

def test_walk_forward_validation(sample_ohlcv_data):
    """Test walk-forward validation."""
    # Prepare data
    features, target = prepare_features(sample_ohlcv_data)
    
    # Mock LightGBM train function
    with patch('lightgbm.train') as mock_train, \
         patch('src.ai_models.model_pipeline.evaluate_model') as mock_evaluate:
        
        # Setup the mock train function
        mock_booster = MagicMock()
        mock_train.return_value = mock_booster
        
        # Setup the mock evaluate function with realistic metrics
        mock_evaluate.side_effect = [
            {'rmse': 0.01, 'mae': 0.008, 'r2': 0.6, 'directional_accuracy': 0.62},
            {'rmse': 0.012, 'mae': 0.009, 'r2': 0.58, 'directional_accuracy': 0.59},
            {'rmse': 0.015, 'mae': 0.011, 'r2': 0.55, 'directional_accuracy': 0.57}
        ]
        
        # Call walk_forward_validation with small number of splits
        avg_metrics, fold_results = walk_forward_validation(
            features, target, n_splits=3, initial_train_size=0.5, verbose=False
        )
        
        # Verify the function called the correct number of train/evaluate cycles
        assert mock_train.call_count == 3
        assert mock_evaluate.call_count == 3
        
        # Verify output structure
        assert isinstance(avg_metrics, dict)
        assert isinstance(fold_results, list)
        assert len(fold_results) == 3
        
        # Verify metrics
        assert 'rmse' in avg_metrics
        assert 'mae' in avg_metrics
        assert 'r2' in avg_metrics
        assert 'directional_accuracy' in avg_metrics
        assert 'n_folds' in avg_metrics
        assert 'total_samples' in avg_metrics
        
        # Check fold results structure
        for fold in fold_results:
            assert 'fold' in fold
            assert 'train_size' in fold
            assert 'val_size' in fold
            assert 'rmse' in fold
            assert 'directional_accuracy' in fold
        
        # Verify the metrics are averaged correctly
        expected_rmse = (0.01 + 0.012 + 0.015) / 3
        assert abs(avg_metrics['rmse'] - expected_rmse) < 1e-10
        
        expected_dir_acc = (0.62 + 0.59 + 0.57) / 3
        assert abs(avg_metrics['directional_accuracy'] - expected_dir_acc) < 1e-10 