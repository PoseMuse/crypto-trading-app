# AI Pipeline Documentation

## Overview

The AI pipeline in the crypto trading bot processes historical market data, integrates sentiment analysis, trains machine learning models, and exports them for inference in different environments. This document outlines the key components and processes of this pipeline.

## Pipeline Components

### 1. Data Collection

- **Historical Market Data**: Fetched from exchanges via the `fetch_historical_data()` function, providing OHLCV (Open, High, Low, Close, Volume) data for specified trading pairs.
- **Sentiment Data**: Collected from social media sources (primarily Reddit) to gauge market sentiment about specific cryptocurrencies.

### 2. Feature Engineering

The `prepare_features()` function creates a rich set of features including:

- Price-based features (returns, logarithmic returns)
- Volume indicators 
- Moving averages (5, 10, 20, 50 day)
- Volatility measures
- Technical indicators (RSI, MACD)
- Sentiment scores (integrated from sentiment analysis)

### 3. Model Training

LightGBM is the primary model used for price prediction:

```python
from src.ai_models.model_pipeline import train_lightgbm

# Train the model
model, training_info = train_lightgbm(features, target)
```

The `train_lightgbm()` function handles:
- Train/test splitting
- Parameter configuration
- Early stopping
- Feature importance tracking

### 4. Model Evaluation

The `evaluate_model()` function provides comprehensive metrics:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R-squared score
- Directional accuracy (percentage of correct up/down predictions)

### 5. ONNX Model Export

The pipeline now supports exporting trained LightGBM models to ONNX format:

```python
from src.ai_models.model_pipeline import export_to_onnx

# Export the model to ONNX
onnx_path = export_to_onnx(model, feature_names, "models/model.onnx")
```

#### Benefits of ONNX Export

- **Portability**: Run the same model across different platforms and hardware
- **Performance**: Potential speed improvements via optimized runtime
- **Integration**: Easier to integrate with web services and mobile apps
- **Deployment**: Simplified deployment in production environments

#### Requirements

To use the ONNX export functionality:
1. Install required packages:
   ```
   pip install skl2onnx onnxmltools
   ```
2. Ensure you provide the correct feature names list when exporting

## Sentiment Integration

Sentiment analysis is integrated with the AI pipeline to enhance prediction quality:

1. **Collection**: Reddit posts are analyzed from crypto-related subreddits
2. **Analysis**: VADER sentiment analyzer scores each post
3. **Aggregation**: Scores are combined using post popularity as weights
4. **Integration**: The `merge_sentiment_with_market_data()` function adds sentiment as features
5. **Training**: Models can then learn from both price action and market sentiment

Example of sentiment integration:

```python
# Fetch and analyze sentiment
sentiment_data = fetch_sentiment_data(['BTC', 'ETH'])

# Merge with market data
merged_data = merge_sentiment_with_market_data(market_data, sentiment_data)

# Prepare features (sentiment columns are automatically included)
features, target = prepare_features(merged_data)
```

## Periodic Retraining

The `train_scheduler.py` script enables automated, periodic model retraining:

```bash
# Retrain models with current date
python src/train_scheduler.py

# Retrain models with a specific date
python src/train_scheduler.py --date 2023-10-15
```

### Retraining Process

1. Fetches the latest historical data for configured trading pairs
2. Retrieves current sentiment data from social media sources
3. Merges market and sentiment data
4. Trains new LightGBM models with the latest data
5. Evaluates model performance
6. Exports models to both native LightGBM and ONNX formats
7. Logs the process and results to `logs/training.log`

### Scheduling Retraining

To set up automatic retraining, add the script to your crontab:

```bash
# Example: Retrain models every day at 2:00 AM
0 2 * * * cd /path/to/project && python src/train_scheduler.py >> logs/cron.log 2>&1
```

## Walk-Forward Validation

For backtesting purposes, the pipeline includes walk-forward validation:

```python
from src.ai_models.model_pipeline import walk_forward_validation

# Perform walk-forward validation
results, fold_metrics = walk_forward_validation(features, target)
```

This technique more accurately simulates real-world trading by:
1. Training on a historical period
2. Testing on subsequent unseen data
3. Moving the window forward
4. Repeating for multiple time periods

## Best Practices

1. **Regular Retraining**: Markets evolve, so models should be retrained regularly
2. **Feature Selection**: Periodically evaluate feature importance and remove noise
3. **Parameter Tuning**: Use `walk_forward_validation` to tune hyperparameters
4. **Monitoring**: Track model degradation over time to detect concept drift
5. **Versioning**: Keep track of model versions and their performance

## Future Enhancements

- Ensemble methods combining multiple model types
- Deep learning models for complex pattern recognition
- Reinforcement learning for optimal trading strategies
- Automated feature selection
- Distributed training for larger datasets

## Troubleshooting

- **Missing Data**: If errors occur due to missing data, check exchange API limits
- **ONNX Conversion**: If export fails, verify that all features are numeric
- **Memory Issues**: For large datasets, consider batch processing
- **Sentiment API Limits**: Reddit API has rate limits; use mock mode for development 