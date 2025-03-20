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
    reddit_client_id: str = None,
    reddit_client_secret: str = None,
    reddit_user_agent: str = None
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
        
    Returns:
        List of dictionaries containing post data
    
    Note:
        Requires Reddit API credentials from environment variables if not provided.
    """
    try:
        import praw
    except ImportError:
        print("Error: PRAW library not installed. Run 'pip install praw'")
        return []
    
    # Get credentials from environment variables if not provided
    reddit_client_id = reddit_client_id or os.getenv("REDDIT_CLIENT_ID")
    reddit_client_secret = reddit_client_secret or os.getenv("REDDIT_CLIENT_SECRET")
    reddit_user_agent = reddit_user_agent or os.getenv("REDDIT_USER_AGENT", "CryptoBot/1.0")
    
    if not reddit_client_id or not reddit_client_secret:
        print("Error: Reddit API credentials not provided. Set environment variables or pass as parameters.")
        return []
    
    # Initialize the Reddit API client
    try:
        reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )
    except Exception as e:
        print(f"Error initializing Reddit client: {e}")
        return []
    
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


def aggregate_multisource_sentiment(
    reddit_posts: List[Dict] = None,
    telegram_messages: List[Dict] = None,
    twitter_tweets: List[Dict] = None,
    source_weights: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Aggregate sentiment from multiple social media sources with optional weighting.
    
    Args:
        reddit_posts: List of Reddit post dictionaries
        telegram_messages: List of Telegram message dictionaries
        twitter_tweets: List of Twitter tweet dictionaries
        source_weights: Dictionary of weights for each source, e.g., {'reddit': 1.0, 'telegram': 0.5, 'twitter': 1.0}
                       If not provided, all sources are weighted equally (1.0)
    
    Returns:
        Dictionary with aggregated sentiment analysis results
    """
    # Initialize default weights if not provided
    if source_weights is None:
        source_weights = {
            'reddit': 1.0,
            'telegram': 1.0,
            'twitter': 1.0
        }
    
    # Ensure lists are not None
    reddit_posts = reddit_posts or []
    telegram_messages = telegram_messages or []
    twitter_tweets = twitter_tweets or []
    
    # Calculate sentiment for each source
    reddit_sentiment = None
    telegram_sentiment = None
    twitter_sentiment = None
    
    if reddit_posts:
        reddit_sentiment = aggregate_sentiment(reddit_posts)
    
    if telegram_messages:
        from .telegram_pipeline import analyze_telegram_sentiment
        telegram_sentiment = analyze_telegram_sentiment(telegram_messages)
    
    if twitter_tweets:
        from .twitter_pipeline import analyze_twitter_sentiment
        twitter_sentiment = analyze_twitter_sentiment(twitter_tweets)
    
    # Count total items across all sources
    total_items = len(reddit_posts) + len(telegram_messages) + len(twitter_tweets)
    
    if total_items == 0:
        return {
            'count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'compound_score': 0.0,
            'sentiment': 'neutral',
            'source_breakdown': {
                'reddit': {'count': 0, 'weight': source_weights.get('reddit', 1.0)},
                'telegram': {'count': 0, 'weight': source_weights.get('telegram', 1.0)},
                'twitter': {'count': 0, 'weight': source_weights.get('twitter', 1.0)}
            }
        }
    
    # Calculate weighted sentiment scores
    weighted_scores = 0.0
    total_weight = 0.0
    
    # Track counts by sentiment category
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    # Reddit sentiment
    if reddit_sentiment:
        weight = source_weights.get('reddit', 1.0) * len(reddit_posts)
        weighted_scores += reddit_sentiment.get('compound_score', 0.0) * weight
        total_weight += weight
        
        positive_count += reddit_sentiment.get('positive_count', 0)
        negative_count += reddit_sentiment.get('negative_count', 0)
        neutral_count += reddit_sentiment.get('neutral_count', 0)
    
    # Telegram sentiment
    if telegram_sentiment:
        weight = source_weights.get('telegram', 1.0) * len(telegram_messages)
        weighted_scores += telegram_sentiment.get('compound_score', 0.0) * weight
        total_weight += weight
        
        positive_count += telegram_sentiment.get('positive_count', 0)
        negative_count += telegram_sentiment.get('negative_count', 0)
        neutral_count += telegram_sentiment.get('neutral_count', 0)
    
    # Twitter sentiment
    if twitter_sentiment:
        weight = source_weights.get('twitter', 1.0) * len(twitter_tweets)
        weighted_scores += twitter_sentiment.get('compound_score', 0.0) * weight
        total_weight += weight
        
        positive_count += twitter_sentiment.get('positive_count', 0)
        negative_count += twitter_sentiment.get('negative_count', 0)
        neutral_count += twitter_sentiment.get('neutral_count', 0)
    
    # Calculate overall sentiment score
    compound_score = weighted_scores / total_weight if total_weight > 0 else 0.0
    
    # Determine overall sentiment
    if compound_score > 0.05:
        sentiment = 'positive'
    elif compound_score < -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    # Create source breakdown for the results
    source_breakdown = {
        'reddit': {
            'count': len(reddit_posts),
            'weight': source_weights.get('reddit', 1.0),
            'compound_score': reddit_sentiment.get('compound_score', 0.0) if reddit_sentiment else 0.0
        },
        'telegram': {
            'count': len(telegram_messages),
            'weight': source_weights.get('telegram', 1.0),
            'compound_score': telegram_sentiment.get('compound_score', 0.0) if telegram_sentiment else 0.0
        },
        'twitter': {
            'count': len(twitter_tweets),
            'weight': source_weights.get('twitter', 1.0),
            'compound_score': twitter_sentiment.get('compound_score', 0.0) if twitter_sentiment else 0.0
        }
    }
    
    return {
        'count': total_items,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'compound_score': compound_score,
        'sentiment': sentiment,
        'source_breakdown': source_breakdown
    } 