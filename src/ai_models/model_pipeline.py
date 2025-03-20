"""
Model Pipeline for cryptocurrency trading predictions.

This module contains functions to:
1. Fetch historical trading data
2. Prepare features for model training
3. Train LightGBM models
4. Evaluate model performance
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List, Any, Union

def fetch_historical_data(
    symbol: str,
    exchange_manager: Any = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given symbol.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        exchange_manager: Exchange manager instance or None
        start_date: Start date for historical data
        end_date: End date for historical data
        
    Returns:
        DataFrame with columns: [timestamp, open, high, low, close, volume]
    
    Note:
        If exchange_manager is None, this will attempt direct CCXT calls.
        In a production environment, proper error handling should be added.
    """
    try:
        if exchange_manager is not None:
            # Use the exchange manager if provided
            ohlcv_data = exchange_manager.fetch_historical_ohlcv(
                symbol=symbol,
                timeframe='1d',  # Daily data
                since=int(start_date.timestamp() * 1000) if start_date else None,
                limit=500  # Adjust as needed
            )
        else:
            # Direct CCXT implementation (placeholder)
            import ccxt
            exchange = ccxt.binance()
            ohlcv_data = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe='1d',
                since=int(start_date.timestamp() * 1000) if start_date else None,
                limit=500
            )
            
        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        # Return empty DataFrame with correct columns as fallback
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for model training from OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Tuple of (features_df, target_series)
        
    Note:
        This is a starting point with basic technical indicators.
        More sophisticated feature engineering can be added later.
    """
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Basic price features
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close']).diff()
    
    # Volume features
    data['volume_change'] = data['volume'].pct_change()
    data['volume_ma5'] = data['volume'].rolling(5).mean()
    
    # Moving averages
    data['ma5'] = data['close'].rolling(window=5).mean()
    data['ma10'] = data['close'].rolling(window=10).mean()
    data['ma20'] = data['close'].rolling(window=20).mean()
    data['ma50'] = data['close'].rolling(window=50).mean()
    
    # Price relative to moving averages
    data['close_ma5_ratio'] = data['close'] / data['ma5']
    data['close_ma10_ratio'] = data['close'] / data['ma10']
    
    # Volatility features
    data['volatility_5'] = data['returns'].rolling(window=5).std()
    data['volatility_10'] = data['returns'].rolling(window=10).std()
    
    # Relative Strength Index (RSI)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    ema12 = data['close'].ewm(span=12, adjust=False).mean()
    ema26 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema12 - ema26
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_diff'] = data['macd'] - data['macd_signal']
    
    # Target: Next day's return (can be modified to different horizons)
    data['target'] = data['returns'].shift(-1)
    
    # Drop rows with NaN values
    data.dropna(inplace=True)
    
    # Split features and target
    features = data.drop(['target', 'open', 'high', 'low', 'close', 'volume'], axis=1)
    target = data['target']
    
    return features, target


def train_lightgbm(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    params: Optional[Dict] = None
) -> Tuple[lgb.Booster, Dict]:
    """
    Train a LightGBM model for cryptocurrency price prediction.
    
    Args:
        features: DataFrame of prepared features
        target: Series with target values
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        params: LightGBM parameters (or None for defaults)
        
    Returns:
        Tuple of (trained_model, training_info)
        
    Note:
        For production, consider using cross-validation or walk-forward testing
        which is more appropriate for time series data.
    """
    from sklearn.model_selection import train_test_split
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    
    # Default parameters if none provided
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1
        }
    
    # Create dataset for LightGBM
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    # Train model
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )
    
    # Record training information
    training_info = {
        'feature_importance': dict(zip(features.columns, model.feature_importance())),
        'best_iteration': model.best_iteration,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'num_features': features.shape[1]
    }
    
    return model, training_info


def evaluate_model(
    model: lgb.Booster,
    test_features: pd.DataFrame,
    test_target: pd.Series
) -> Dict:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained LightGBM model
        test_features: Test features DataFrame
        test_target: Test target Series
        
    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Make predictions
    predictions = model.predict(test_features)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test_target, predictions))
    mae = mean_absolute_error(test_target, predictions)
    r2 = r2_score(test_target, predictions)
    
    # Calculate directional accuracy (up/down prediction)
    direction_actual = (test_target > 0).astype(int)
    direction_pred = (predictions > 0).astype(int)
    directional_accuracy = (direction_actual == direction_pred).mean()
    
    # Return metrics
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'directional_accuracy': directional_accuracy
    }


def save_model(model: lgb.Booster, filepath: str) -> None:
    """
    Save LightGBM model to file.
    
    Args:
        model: Trained LightGBM model
        filepath: Path to save the model
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save the model
        model.save_model(filepath)
        print(f"Model saved to {filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")


def export_to_onnx(model: lgb.Booster, feature_names: List[str], filepath: str) -> str:
    """
    Export a trained LightGBM model to ONNX format.
    
    Args:
        model: Trained LightGBM model
        feature_names: List of feature names used in the model
        filepath: Path to save the ONNX model
        
    Returns:
        Path to saved ONNX model
        
    Notes:
        Requires skl2onnx and onnxmltools packages to be installed.
        Install with: pip install skl2onnx onnxmltools
    """
    try:
        # Import required packages
        import skl2onnx
        from skl2onnx.common.data_types import FloatTensorType
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType as ONNXFloatTensorType
        
        # Set the filepath extension if not provided
        if not filepath.endswith('.onnx'):
            filepath += '.onnx'
            
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Get number of features
        num_features = len(feature_names)
        
        # Initial types for the model
        initial_types = [('input', ONNXFloatTensorType([None, num_features]))]
        
        # Convert the model to ONNX
        onnx_model = convert_lightgbm(model, initial_types=initial_types, target_opset=12)
        
        # Save the ONNX model
        with open(filepath, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        print(f"ONNX model exported to {filepath}")
        return filepath
    except ImportError as e:
        print(f"Error importing ONNX libraries: {e}")
        print("Please install required packages with: pip install skl2onnx onnxmltools")
        return ""
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        return ""


def load_model(filepath: str) -> lgb.Booster:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded LightGBM model
    """
    return lgb.Booster(model_file=filepath)


def walk_forward_validation(
    features: pd.DataFrame,
    target: pd.Series,
    params: Optional[Dict] = None,
    n_splits: int = 3,
    initial_train_size: float = 0.5,
    verbose: bool = True
) -> Tuple[Dict, List[Dict]]:
    """
    Perform walk-forward (rolling window) validation for time series forecasting.
    
    Args:
        features: DataFrame of features, chronologically ordered
        target: Series of target values, chronologically ordered
        params: LightGBM parameters (or None for defaults)
        n_splits: Number of validation folds
        initial_train_size: Initial proportion of data to use for first training
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (average_metrics, list_of_fold_results)
        
    Note:
        This is a more appropriate validation strategy for time series than
        random train-test splits, as it respects the temporal ordering of data.
    """
    if features.shape[0] != target.shape[0]:
        raise ValueError("Features and target must have the same number of samples")
    
    # Default parameters if none provided
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1
        }
    
    # Calculate initial training size
    n_samples = features.shape[0]
    initial_train_end = int(n_samples * initial_train_size)
    
    # Calculate size of each validation fold
    fold_size = (n_samples - initial_train_end) // n_splits
    
    # Create folds
    folds = []
    for i in range(n_splits):
        val_start = initial_train_end + i * fold_size
        val_end = val_start + fold_size if i < n_splits - 1 else n_samples
        folds.append((val_start, val_end))
    
    # Run walk-forward validation
    all_metrics = []
    all_fold_info = []
    
    for fold_idx, (val_start, val_end) in enumerate(folds):
        if verbose:
            print(f"Fold {fold_idx+1}/{n_splits}: Training on data up to index {val_start}, "
                  f"validating on indices {val_start} to {val_end-1}")
        
        # Split data
        X_train = features.iloc[:val_start]
        y_train = target.iloc[:val_start]
        X_val = features.iloc[val_start:val_end]
        y_val = target.iloc[val_start:val_end]
        
        # Create datasets
        lgb_train = lgb.Dataset(X_train, y_train)
        
        # Train model
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            verbose_eval=False
        )
        
        # Evaluate on validation fold
        metrics = evaluate_model(model, X_val, y_val)
        
        if verbose:
            print(f"  Fold {fold_idx+1} metrics: RMSE={metrics['rmse']:.4f}, "
                  f"Directional Accuracy={metrics['directional_accuracy']:.4f}")
        
        # Store metrics and fold information
        all_metrics.append(metrics)
        fold_info = {
            'fold': fold_idx + 1,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'train_end_idx': val_start,
            'val_start_idx': val_start,
            'val_end_idx': val_end,
            **metrics
        }
        all_fold_info.append(fold_info)
    
    # Calculate average metrics
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        avg_metrics[metric] = sum(fold[metric] for fold in all_metrics) / len(all_metrics)
    
    avg_metrics['n_folds'] = n_splits
    avg_metrics['total_samples'] = n_samples
    
    if verbose:
        print("\nWalk-forward validation complete.")
        print(f"Average metrics over {n_splits} folds:")
        for k, v in avg_metrics.items():
            if k not in ['n_folds', 'total_samples']:
                print(f"  {k}: {v:.4f}")
    
    return avg_metrics, all_fold_info


def prepare_lstm_data(
    features: pd.DataFrame,
    target: pd.Series,
    lookback: int = 10,
    validation_split: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for LSTM models by creating sequences.
    
    Args:
        features: Features DataFrame
        target: Target Series
        lookback: Number of timesteps to look back
        validation_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
        
    Note:
        This reshapes the data into 3D arrays required by LSTM: 
        (samples, timesteps, features)
    """
    from sklearn.preprocessing import StandardScaler
    
    # Scale features for better LSTM performance
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(len(features_scaled) - lookback):
        X.append(features_scaled[i:i+lookback])
        y.append(target.iloc[i+lookback])
    
    X, y = np.array(X), np.array(y)
    
    # Split into train and validation sets (respecting time order)
    train_size = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, X_val, y_train, y_val


def train_lstm(
    features: pd.DataFrame,
    target: pd.Series,
    lookback: int = 10,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.2,
    verbose: int = 1
) -> Tuple[Any, Dict]:
    """
    Train an LSTM model for cryptocurrency price prediction.
    
    Args:
        features: DataFrame of prepared features
        target: Series with target values
        lookback: Number of past time steps to use
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        verbose: Verbosity level for training
        
    Returns:
        Tuple of (trained_model, training_info)
        
    Note:
        This is a simple LSTM implementation that serves as a starting point.
        More sophisticated architectures can be implemented as needed.
    """
    try:
        # Import TensorFlow and Keras
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        
        # Prepare sequences for LSTM
        X_train, X_val, y_train, y_val = prepare_lstm_data(
            features, target, lookback, validation_split
        )
        
        # Get input shape
        n_features = X_train.shape[2]
        
        # Build model
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, 
                      input_shape=(lookback, n_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(30, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mse')
        
        # Add early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        # Calculate metrics
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        best_epoch = np.argmin(val_loss) + 1
        
        # Make predictions
        train_preds = model.predict(X_train).flatten()
        val_preds = model.predict(X_val).flatten()
        
        # Directional accuracy
        train_dir_acc = np.mean(((y_train > 0) == (train_preds > 0)).astype(float))
        val_dir_acc = np.mean(((y_val > 0) == (val_preds > 0)).astype(float))
        
        # Create training info
        training_info = {
            'model_type': 'LSTM',
            'best_epoch': best_epoch,
            'final_train_loss': train_loss[-1],
            'final_val_loss': val_loss[-1],
            'best_val_loss': val_loss[best_epoch - 1],
            'train_dir_accuracy': train_dir_acc,
            'val_dir_accuracy': val_dir_acc,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'lookback': lookback,
            'num_features': n_features
        }
        
        return model, training_info
    
    except ImportError:
        print("TensorFlow/Keras not available. Install with: pip install tensorflow")
        return None, {'error': 'TensorFlow/Keras not available'}


def save_keras_model(model: Any, filepath: str) -> None:
    """
    Save a Keras model to disk.
    
    Args:
        model: Trained Keras model
        filepath: Path to save the model
        
    Returns:
        None
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        model.save(filepath)
        print(f"Keras model saved to {filepath}")
    except Exception as e:
        print(f"Error saving Keras model: {e}")


def load_keras_model(filepath: str) -> Any:
    """
    Load a Keras model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded Keras model
    """
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(filepath)
    except ImportError:
        print("TensorFlow/Keras not available. Install with: pip install tensorflow")
        return None
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return None 