#!/usr/bin/env python3
"""
Sentiment Analysis Script for Cryptocurrency Trading Bot

This script:
1. Fetches posts from Reddit for specified cryptocurrencies
2. Analyzes sentiment in those posts
3. Saves results to CSV/JSON for later use in trading decisions

Usage:
    python src/sentiment_script.py --subreddits bitcoin ethereum --limit 50 --output data/sentiment.json
"""

import os
import json
import argparse
import datetime
from typing import List, Dict, Optional
from pathlib import Path

# Import our sentiment analysis pipeline
from sentiment_analysis.sentiment_pipeline import (
    fetch_reddit_posts,
    analyze_sentiment,
    aggregate_sentiment,
    save_sentiment_data
)

# Define default crypto subreddits
DEFAULT_SUBREDDITS = [
    "bitcoin",
    "cryptocurrency",
    "ethereum",
    "CryptoMarkets"
]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Reddit Cryptocurrency Sentiment Analysis')
    
    parser.add_argument('--subreddits', '-s', nargs='+', default=DEFAULT_SUBREDDITS,
                        help='List of subreddits to analyze (without r/)')
    
    parser.add_argument('--limit', '-l', type=int, default=100,
                        help='Maximum number of posts to fetch per subreddit')
    
    parser.add_argument('--time-filter', '-t', choices=['hour', 'day', 'week', 'month', 'year', 'all'],
                        default='day', help='Time filter for Reddit posts')
    
    parser.add_argument('--output', '-o', default='data/sentiment/latest_sentiment.json',
                        help='Output file path for sentiment data')
                        
    parser.add_argument('--weight-by', '-w', choices=['score', 'num_comments', 'none'],
                        default='score', help='Weight posts by this attribute')
    
    parser.add_argument('--use-mock', action='store_true',
                        help='Use mock data instead of calling Reddit API')
    
    parser.add_argument('--credentials', '-c', default='.env',
                        help='Path to credentials file (.env format)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed information')
    
    return parser.parse_args()


def load_credentials(credentials_path: str) -> Dict[str, str]:
    """
    Load Reddit API credentials from a file.
    
    Args:
        credentials_path: Path to credentials file (.env format)
        
    Returns:
        Dictionary of credentials
    """
    credentials = {}
    
    try:
        # Check if the file exists
        if os.path.exists(credentials_path):
            with open(credentials_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        credentials[key.strip()] = value.strip().strip('"\'')
    except Exception as e:
        print(f"Error loading credentials: {e}")
    
    return credentials


def main():
    """Main function to run sentiment analysis."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Load credentials if using real API
    credentials = {}
    if not args.use_mock and os.path.exists(args.credentials):
        credentials = load_credentials(args.credentials)
    
    # Print configuration
    if args.verbose:
        print(f"Configuration:")
        print(f"  Subreddits: {args.subreddits}")
        print(f"  Post limit: {args.limit}")
        print(f"  Time filter: {args.time_filter}")
        print(f"  Output: {args.output}")
        print(f"  Using mock data: {args.use_mock}")
    
    print(f"Fetching posts from {len(args.subreddits)} subreddits...")
    
    # Fetch Reddit posts
    posts = fetch_reddit_posts(
        subreddits=args.subreddits,
        limit=args.limit,
        time_filter=args.time_filter,
        reddit_client_id=credentials.get('REDDIT_CLIENT_ID', 'YOUR_CLIENT_ID'),
        reddit_client_secret=credentials.get('REDDIT_CLIENT_SECRET', 'YOUR_CLIENT_SECRET'),
        reddit_user_agent=credentials.get('REDDIT_USER_AGENT', 'YOUR_USER_AGENT'),
        use_mock=args.use_mock
    )
    
    print(f"Fetched {len(posts)} posts.")
    
    # Calculate sentiment
    print("Analyzing sentiment...")
    weight_key = None if args.weight_by == 'none' else args.weight_by
    sentiment_data = aggregate_sentiment(posts, weight_key=weight_key)
    
    # Add metadata
    sentiment_data['subreddits'] = args.subreddits
    sentiment_data['time_filter'] = args.time_filter
    sentiment_data['fetch_time'] = datetime.datetime.now().isoformat()
    
    # Print results
    print("\nSentiment Analysis Results:")
    print(f"  Overall sentiment: {sentiment_data['overall_sentiment']:.4f} (-1 to +1)")
    print(f"  Weighted sentiment: {sentiment_data['weighted_sentiment']:.4f} (-1 to +1)")
    print(f"  Positive posts: {sentiment_data['positive_ratio']*100:.1f}%")
    print(f"  Negative posts: {sentiment_data['negative_ratio']*100:.1f}%")
    print(f"  Neutral posts: {sentiment_data['neutral_ratio']*100:.1f}%")
    print(f"  Posts analyzed: {sentiment_data['post_count']}")
    
    # Save results
    save_sentiment_data(sentiment_data, args.output)
    
    # Also save to CSV for time series tracking
    csv_path = os.path.join(
        os.path.dirname(args.output),
        "sentiment_history.csv"
    )
    
    # Flatten the dictionary for CSV
    csv_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'overall_sentiment': sentiment_data['overall_sentiment'],
        'weighted_sentiment': sentiment_data['weighted_sentiment'],
        'positive_ratio': sentiment_data['positive_ratio'],
        'negative_ratio': sentiment_data['negative_ratio'],
        'neutral_ratio': sentiment_data['neutral_ratio'],
        'post_count': sentiment_data['post_count'],
        'subreddits': ','.join(args.subreddits),
        'time_filter': args.time_filter
    }
    
    # Create or append to CSV
    import pandas as pd
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df.append(csv_data, ignore_index=True)
        else:
            df = pd.DataFrame([csv_data])
            
        df.to_csv(csv_path, index=False)
        print(f"Sentiment history saved to {csv_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")
    
    print(f"Sentiment data saved to {args.output}")
    
    # Recommendation based on sentiment
    if sentiment_data['weighted_sentiment'] > 0.2:
        print("\nSentiment recommendation: BULLISH 📈")
    elif sentiment_data['weighted_sentiment'] < -0.2:
        print("\nSentiment recommendation: BEARISH 📉")
    else:
        print("\nSentiment recommendation: NEUTRAL ↔️")


if __name__ == "__main__":
    main() 