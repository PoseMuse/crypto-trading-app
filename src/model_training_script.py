#!/usr/bin/env python3
"""
Cryptocurrency Trading Bot - Model Training Script

This script:
1. Fetches historical data for a specified cryptocurrency
2. Prepares features
3. Trains a LightGBM model
4. Evaluates the model
5. Saves the model to disk

Usage:
    python src/model_training_script.py
"""

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Import our model pipeline
from ai_models.model_pipeline import (
    fetch_historical_data,
    prepare_features,
    train_lightgbm,
    evaluate_model,
    save_model,
    walk_forward_validation
)

# Load environment variables
load_dotenv()

def main():
    """Main function to train and save the model."""
    print("Starting model training process...")
    
    # Configuration
    symbol = "BTC/USDT"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Get one year of data
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Fetch data
    print(f"Fetching historical data for {symbol} from {start_date.date()} to {end_date.date()}")
    data = fetch_historical_data(symbol, start_date=start_date, end_date=end_date)
    
    if data.empty:
        print("Error: No data fetched. Exiting.")
        return
    
    print(f"Fetched {len(data)} data points")
    
    # Prepare features
    print("Preparing features...")
    features, target = prepare_features(data)
    print(f"Prepared {len(features)} samples with {features.shape[1]} features")
    
    # Save feature names for later use
    feature_names = features.columns.tolist()
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.joblib"))
    
    # Display feature information
    print("Feature preview:")
    print(features.describe().T[['count', 'mean', 'min', 'max']])
    
    # === Walk-forward validation ===
    print("\nPerforming walk-forward validation...")
    
    # LightGBM parameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    
    # Perform walk-forward validation with 5 folds
    avg_metrics, fold_results = walk_forward_validation(
        features, target, 
        params=lgb_params,
        n_splits=5,
        initial_train_size=0.6,
        verbose=True
    )
    
    # Save validation results
    validation_results = pd.DataFrame(fold_results)
    validation_results.to_csv(os.path.join(model_dir, "walk_forward_results.csv"), index=False)
    
    # Plot validation metrics across folds
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(validation_results['fold'], validation_results['rmse'], 'o-', label='RMSE')
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.title('RMSE across validation folds')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(validation_results['fold'], validation_results['directional_accuracy'], 'o-', label='Dir. Accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Directional Accuracy')
    plt.title('Directional Accuracy across validation folds')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "validation_metrics.png"))
    
    # === Final model training ===
    print("\nTraining final model on all data...")
    model, training_info = train_lightgbm(features, target, test_size=0.2, params=lgb_params)
    
    # Print training information
    print(f"Model trained with {training_info['train_size']} samples")
    print(f"Best iteration: {training_info['best_iteration']}")
    
    # Print feature importance
    print("\nFeature importance:")
    importance_dict = training_info['feature_importance']
    importance_df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    }).sort_values('Importance', ascending=False)
    
    print(importance_df.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "feature_importance.png"))
    
    # Evaluate the model
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("Model evaluation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Compare with walk-forward validation metrics
    print("\nComparison with walk-forward validation:")
    print(f"  Random split RMSE: {metrics['rmse']:.4f} vs Walk-forward RMSE: {avg_metrics['rmse']:.4f}")
    print(f"  Random split Dir. Accuracy: {metrics['directional_accuracy']:.4f} vs "
          f"Walk-forward Dir. Accuracy: {avg_metrics['directional_accuracy']:.4f}")
    
    # Save evaluation metrics
    all_metrics = {
        'random_split': metrics,
        'walk_forward': avg_metrics
    }
    
    # Save metrics as CSV
    pd.DataFrame([{f"{k}_{m}": v for k, metrics_dict in all_metrics.items() 
                  for m, v in metrics_dict.items() if isinstance(v, (int, float))}]).to_csv(
        os.path.join(model_dir, "model_metrics.csv"), index=False
    )
    
    # Save the model
    model_path = os.path.join(model_dir, f"lightgbm_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.txt")
    save_model(model, model_path)
    
    print("Model training complete!")
    print(f"Model saved to: {model_path}")
    print(f"Feature importance plot saved to: {os.path.join(model_dir, 'feature_importance.png')}")
    print(f"Validation metrics plot saved to: {os.path.join(model_dir, 'validation_metrics.png')}")

if __name__ == "__main__":
    main() 