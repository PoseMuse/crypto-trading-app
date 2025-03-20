#!/usr/bin/env python3
"""
Sentiment-Enhanced Model Demo

This script demonstrates how to use the SentimentEnhancedModel class to train
and evaluate a trading model that incorporates social media sentiment.

Usage:
    python src/sentiment_model_demo.py
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from ai_models.sentiment_model import SentimentEnhancedModel

def main():
    """Main function to demonstrate the sentiment-enhanced model."""
    print("===============================================")
    print("  Sentiment-Enhanced Trading Model Demo")
    print("===============================================")
    
    # Create a SentimentEnhancedModel instance for Bitcoin
    model = SentimentEnhancedModel(
        symbol="BTC/USDT",
        sentiment_lookback_days=7,
        sentiment_data_dir="data/sentiment",
        model_dir="models/sentiment",
        reddit_use_mock=True  # Use mock data for demo
    )
    
    # 1. Fetch sentiment data
    print("\n1. Fetching sentiment data...")
    sentiment_data = model.fetch_sentiment_data()
    
    print(f"Sentiment data summary:")
    print(f"  Overall sentiment: {sentiment_data['overall_sentiment']:.4f} (-1 to +1)")
    print(f"  Weighted sentiment: {sentiment_data['weighted_sentiment']:.4f} (-1 to +1)")
    print(f"  Positive ratio: {sentiment_data['positive_ratio']*100:.1f}%")
    print(f"  Negative ratio: {sentiment_data['negative_ratio']*100:.1f}%")
    print(f"  Neutral ratio: {sentiment_data['neutral_ratio']*100:.1f}%")
    
    # 2. Fetch price data
    print("\n2. Fetching historical price data...")
    price_data = model.fetch_price_data(days=365)
    
    print(f"Fetched {len(price_data)} days of price data")
    print(f"Price data summary:")
    print(price_data.describe())
    
    # 3. Prepare features
    print("\n3. Preparing features with sentiment data...")
    features, target = model.prepare_features()
    
    print(f"Prepared {len(features)} samples with {features.shape[1]} features")
    print("Features include:")
    for feature in features.columns:
        print(f"  - {feature}")
    
    # 4. Train the model with walk-forward validation
    print("\n4. Training model with walk-forward validation...")
    _, metrics = model.train(
        use_walk_forward=True,
        n_splits=5,
        initial_train_size=0.6
    )
    
    print("Model training complete!")
    print("Validation metrics:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
    
    # 5. Save the model
    print("\n5. Saving the model...")
    model_path = model.save()
    print(f"Model saved to: {model_path}")
    
    # 6. Analyze sentiment feature importance
    print("\n6. Analyzing sentiment feature importance...")
    sentiment_importance = model.get_sentiment_importance()
    
    print("Sentiment feature importance:")
    print(sentiment_importance)
    
    # 7. Visualize results
    print("\n7. Generating visualizations...")
    
    # Create output directory
    output_dir = "output/sentiment_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot sentiment feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(sentiment_importance['Feature'], sentiment_importance['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Sentiment Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sentiment_importance.png"))
    
    # Plot recent price with sentiment overlay
    plt.figure(figsize=(12, 8))
    
    # Price subplot
    ax1 = plt.subplot(2, 1, 1)
    price_data['close'][-30:].plot(ax=ax1, color='blue')
    ax1.set_ylabel('Price')
    ax1.set_title('Recent Price Movement')
    ax1.grid(True, alpha=0.3)
    
    # Sentiment subplot
    ax2 = plt.subplot(2, 1, 2)
    combined_data = model.combine_price_and_sentiment()
    sentiment_cols = ['sentiment_score', 'weighted_sentiment']
    combined_data[sentiment_cols][-30:].plot(ax=ax2)
    ax2.set_ylabel('Sentiment (-1 to +1)')
    ax2.set_title('Sentiment Indicators')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "price_with_sentiment.png"))
    
    print(f"Visualizations saved to {output_dir}")
    
    # 8. Make a prediction for tomorrow
    print("\n8. Making prediction for next day...")
    
    # Get the latest features
    latest_features = features.iloc[-1].values.reshape(1, -1)
    prediction = model.predict(pd.DataFrame([latest_features], columns=features.columns))
    
    # Convert prediction to percentage
    next_day_return = prediction[0] * 100
    
    print(f"Predicted next day return: {next_day_return:.2f}%")
    
    # Trading signal based on prediction and sentiment
    weighted_sentiment = sentiment_data['weighted_sentiment']
    
    if next_day_return > 0.5 and weighted_sentiment > 0.1:
        signal = "STRONG BUY"
    elif next_day_return > 0.2 or weighted_sentiment > 0.2:
        signal = "BUY"
    elif next_day_return < -0.5 and weighted_sentiment < -0.1:
        signal = "STRONG SELL"
    elif next_day_return < -0.2 or weighted_sentiment < -0.2:
        signal = "SELL"
    else:
        signal = "HOLD"
    
    print(f"Trading signal: {signal}")
    
    print("\nDemo complete!")

if __name__ == "__main__":
    main() 