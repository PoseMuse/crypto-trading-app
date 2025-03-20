"""
Sentiment Analysis Pipeline for cryptocurrency trading.

This module contains functions to:
1. Fetch posts from Reddit
2. Analyze sentiment of text
3. Aggregate sentiment scores
"""

import os
import json
import datetime
from typing import List, Dict, Union, Tuple, Optional, Any
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure NLTK resources are downloaded (uncomment first time)
# nltk.download('vader_lexicon')


def fetch_reddit_posts(
    subreddits: List[str],
    limit: int = 100,
    time_filter: str = "day",
    reddit_client_id: str = "YOUR_CLIENT_ID",
    reddit_client_secret: str = "YOUR_CLIENT_SECRET",
    reddit_user_agent: str = "YOUR_USER_AGENT",
    use_mock: bool = False
) -> List[Dict]:
    """
    Fetch recent posts from specified subreddits.
    
    Args:
        subreddits: List of subreddit names to fetch from (without 'r/')
        limit: Maximum number of posts to fetch per subreddit
        time_filter: One of 'hour', 'day', 'week', 'month', 'year', 'all'
        reddit_client_id: Reddit API client ID
        reddit_client_secret: Reddit API client secret
        reddit_user_agent: Reddit API user agent string
        use_mock: If True, return mock data instead of calling Reddit API
        
    Returns:
        List of dictionaries containing post data
    
    Note:
        Requires Reddit API credentials. Set use_mock=True for testing without credentials.
    """
    if use_mock:
        return _mock_reddit_posts(subreddits, limit)
    
    try:
        import praw
    except ImportError:
        print("Error: PRAW library not installed. Run 'pip install praw' or use use_mock=True")
        return _mock_reddit_posts(subreddits, limit)
    
    # Initialize the Reddit API client
    try:
        reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )
    except Exception as e:
        print(f"Error initializing Reddit client: {e}")
        print("Falling back to mock data.")
        return _mock_reddit_posts(subreddits, limit)
    
    all_posts = []
    
    # Fetch posts from each subreddit
    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            
            # Get top posts from the specified time period
            for post in subreddit.top(time_filter=time_filter, limit=limit):
                post_data = {
                    'id': post.id,
                    'subreddit': subreddit_name,
                    'title': post.title,
                    'selftext': post.selftext,
                    'created_utc': datetime.datetime.fromtimestamp(post.created_utc).isoformat(),
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'author': str(post.author),
                    'url': post.url,
                    'fulltext': f"{post.title} {post.selftext}"
                }
                all_posts.append(post_data)
                
        except Exception as e:
            print(f"Error fetching from r/{subreddit_name}: {e}")
    
    return all_posts


def _mock_reddit_posts(subreddits: List[str], limit: int = 10) -> List[Dict]:
    """
    Generate mock Reddit post data for testing.
    
    Args:
        subreddits: List of subreddit names
        limit: Approximate number of posts to generate
        
    Returns:
        List of dictionaries with mock post data
    """
    import random
    from datetime import datetime, timedelta
    
    # Sample post titles and sentiments for different crypto subreddits
    sentiment_templates = {
        'positive': [
            "Just bought more {coin}! To the moon!",
            "Bullish on {coin}, here's why...",
            "{coin} breaking resistance, looking good!",
            "Why {coin} is undervalued right now",
            "I'm hodling {coin} for the next 5 years",
            "Great news for {coin} - new partnership announced!",
            "{coin} outperforming the market this week",
            "Technical analysis: {coin} setting up for a breakout",
        ],
        'negative': [
            "Is {coin} dead? Thoughts?",
            "Getting tired of waiting for {coin} to recover",
            "The problem with {coin} that nobody talks about",
            "Why I sold all my {coin} yesterday",
            "Warning: {coin} might drop further",
            "Bearish on {coin} for these 3 reasons",
            "{coin} support level broken, be careful",
            "Should I cut my losses with {coin}?",
        ],
        'neutral': [
            "Discussion thread: {coin} price action",
            "Weekly {coin} thread",
            "Question about {coin} staking",
            "How to store {coin} safely?",
            "What's your average cost basis for {coin}?",
            "How much {coin} are you holding?",
            "New to {coin}, what should I know?",
            "Historical analysis of {coin} cycles",
        ]
    }
    
    # Map subreddits to coin names
    coin_mapping = {
        'bitcoin': 'BTC',
        'cryptocurrency': 'crypto',
        'ethereum': 'ETH',
        'cardano': 'ADA',
        'solana': 'SOL',
        'dogecoin': 'DOGE',
        'binance': 'BNB',
    }
    
    # Default to the subreddit name if not in the mapping
    for sub in subreddits:
        if sub not in coin_mapping:
            coin_mapping[sub] = sub.upper()
    
    # Generate mock posts
    mock_posts = []
    now = datetime.now()
    
    for subreddit in subreddits:
        coin = coin_mapping.get(subreddit.lower(), subreddit.upper())
        
        # Generate posts with a mix of sentiments
        for _ in range(limit // len(subreddits) + 1):
            # Select sentiment with weighted probabilities
            sentiment_type = random.choices(
                ['positive', 'negative', 'neutral'],
                weights=[0.4, 0.3, 0.3],
                k=1
            )[0]
            
            templates = sentiment_templates[sentiment_type]
            title = random.choice(templates).format(coin=coin)
            
            # Generate mock post content based on the title
            content_length = random.randint(0, 500)
            if content_length > 0:
                selftext = f"This is a mock post about {coin}. " * (content_length // 30 + 1)
            else:
                selftext = ""
            
            # Random time within the last week
            created_time = now - timedelta(
                days=random.randint(0, 6),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            post_data = {
                'id': f"mock_{subreddit}_{len(mock_posts)}",
                'subreddit': subreddit,
                'title': title,
                'selftext': selftext,
                'created_utc': created_time.isoformat(),
                'score': random.randint(1, 1000),
                'num_comments': random.randint(0, 200),
                'author': f"user_{random.randint(1000, 9999)}",
                'url': f"https://reddit.com/r/{subreddit}/mock_{len(mock_posts)}",
                'fulltext': f"{title} {selftext}"
            }
            
            mock_posts.append(post_data)
            
            # Stop if we've reached the requested limit
            if len(mock_posts) >= limit:
                break
    
    return mock_posts[:limit]


def analyze_sentiment(text: str) -> float:
    """
    Analyze the sentiment of a text using VADER sentiment analyzer.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Sentiment score from -1 (negative) to +1 (positive)
    
    Note:
        Uses NLTK's VADER sentiment analyzer, which is specifically attuned to
        social media content and can handle emojis, slang, etc.
    """
    if not text or text.strip() == "":
        return 0.0
    
    try:
        # Initialize the sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        
        # Get sentiment scores
        sentiment_scores = analyzer.polarity_scores(text)
        
        # Return the compound score normalized from -1 to 1
        return sentiment_scores['compound']
    
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 0.0


def aggregate_sentiment(
    posts: List[Dict],
    text_key: str = 'fulltext',
    weight_key: Optional[str] = 'score'
) -> Dict:
    """
    Analyze and aggregate sentiment across multiple posts.
    
    Args:
        posts: List of post dictionaries
        text_key: Key in post dict containing the text to analyze
        weight_key: Key to use for weighting (e.g., 'score', 'num_comments')
                   If None, all posts are weighted equally
    
    Returns:
        Dictionary with aggregated sentiment metrics
    
    Note:
        The weighted_score is weighted by the specified weight_key if provided.
        Otherwise, it's a simple average.
    """
    if not posts:
        return {
            'overall_sentiment': 0.0,
            'weighted_sentiment': 0.0,
            'positive_ratio': 0.0,
            'negative_ratio': 0.0,
            'neutral_ratio': 0.0,
            'post_count': 0
        }
    
    # Analyze sentiment for each post
    sentiments = []
    weights = []
    sentiment_categories = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for post in posts:
        # Skip posts without the required text or weight
        if text_key not in post:
            continue
            
        text = post[text_key]
        sentiment = analyze_sentiment(text)
        sentiments.append(sentiment)
        
        # Categorize sentiment
        if sentiment > 0.05:
            sentiment_categories['positive'] += 1
        elif sentiment < -0.05:
            sentiment_categories['negative'] += 1
        else:
            sentiment_categories['neutral'] += 1
        
        # Get weight if applicable
        if weight_key and weight_key in post:
            weight = float(post[weight_key])
            weights.append(max(0.1, weight))  # Ensure minimum weight
        else:
            weights.append(1.0)  # Equal weight
    
    # Calculate ratios
    total = len(sentiments)
    positive_ratio = sentiment_categories['positive'] / total if total > 0 else 0
    negative_ratio = sentiment_categories['negative'] / total if total > 0 else 0
    neutral_ratio = sentiment_categories['neutral'] / total if total > 0 else 0
    
    # Calculate simple and weighted averages
    if sentiments:
        overall_sentiment = sum(sentiments) / len(sentiments)
        
        # Weighted average
        weighted_sum = sum(s * w for s, w in zip(sentiments, weights))
        total_weight = sum(weights)
        weighted_sentiment = weighted_sum / total_weight if total_weight > 0 else 0
    else:
        overall_sentiment = 0.0
        weighted_sentiment = 0.0
    
    return {
        'overall_sentiment': overall_sentiment,
        'weighted_sentiment': weighted_sentiment,
        'positive_ratio': positive_ratio,
        'negative_ratio': negative_ratio,
        'neutral_ratio': neutral_ratio,
        'post_count': len(sentiments)
    }


def save_sentiment_data(sentiment_data: Dict, filepath: str) -> None:
    """
    Save sentiment data to a JSON file.
    
    Args:
        sentiment_data: Dictionary of sentiment metrics
        filepath: Path to save the data
        
    Returns:
        None
    """
    # Add timestamp
    sentiment_data['timestamp'] = datetime.datetime.now().isoformat()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(sentiment_data, f, indent=2)
    
    print(f"Sentiment data saved to {filepath}")


def load_sentiment_data(filepath: str) -> Dict:
    """
    Load sentiment data from a JSON file.
    
    Args:
        filepath: Path to the sentiment data file
        
    Returns:
        Dictionary of sentiment metrics
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading sentiment data: {e}")
        return {} 