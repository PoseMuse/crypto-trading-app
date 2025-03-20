"""
Twitter Scraping and Sentiment Analysis for cryptocurrency trading.

This module contains functions to:
1. Fetch tweets using snscrape
2. Filter tweets by cryptocurrency keywords
3. Analyze sentiment of tweets
"""

import os
import logging
import subprocess
import json
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# Import from our sentiment pipeline
from .sentiment_pipeline import analyze_sentiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import snscrape, gracefully handle if not installed
try:
    import snscrape.modules.twitter as sntwitter
    SNSCRAPE_AVAILABLE = True
except ImportError:
    logger.warning("snscrape not installed. Run 'pip install snscrape' to use Twitter scraping.")
    SNSCRAPE_AVAILABLE = False

def fetch_twitter_tweets(
    query: str,
    limit: int = 100,
    days_back: int = 7
) -> List[Dict[str, Any]]:
    """
    Fetch tweets matching the query using snscrape.
    
    Args:
        query: Search query for tweets (e.g., 'bitcoin')
        limit: Maximum number of tweets to retrieve
        days_back: Number of days to look back for tweets
        
    Returns:
        List of dictionaries containing tweet data
    """
    if not SNSCRAPE_AVAILABLE:
        logger.warning("snscrape not installed, please run 'pip install snscrape'")
        return []
    
    tweets = []
    
    try:
        # Add date filter if specified
        if days_back > 0:
            since_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            if "since:" not in query:
                query += f" since:{since_date}"
        
        # Fetch tweets
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= limit:
                break
            
            tweet_data = {
                'id': str(tweet.id),
                'date': tweet.date.isoformat() if tweet.date else None,
                'content': tweet.content,
                'username': tweet.user.username if tweet.user else None,
                'retweet_count': getattr(tweet, 'retweetCount', 0),
                'like_count': getattr(tweet, 'likeCount', 0),
                'reply_count': getattr(tweet, 'replyCount', 0),
                'quote_count': getattr(tweet, 'quoteCount', 0),
                'lang': getattr(tweet, 'lang', 'en')
            }
            tweets.append(tweet_data)
        
        return tweets
    
    except Exception as e:
        logger.error(f"Error fetching tweets for query '{query}': {e}")
        return []

def _mock_twitter_tweets(query: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Generate mock Twitter data for testing.
    
    Args:
        query: Search query to simulate
        limit: Number of mock tweets to generate
        
    Returns:
        List of dictionaries with mock tweet data
    """
    import random
    from datetime import datetime, timedelta
    
    # Extract topic from query
    topic = query.split()[0].lower() if query else "crypto"
    
    # Sample crypto topics
    crypto_terms = [
        "BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "SHIB", 
        "Bitcoin", "Ethereum", "crypto", "altcoin", "DeFi", "NFT"
    ]
    
    # If topic is not a crypto term, use a random one
    if topic not in [term.lower() for term in crypto_terms]:
        topic = random.choice(crypto_terms).lower()
    
    # Sample tweet templates with different sentiments
    tweet_templates = {
        'positive': [
            "Just bought more {coin}! Best decision I've made this year. #crypto #bullish",
            "The future of {coin} looks bright! Technical analysis shows strong support. #crypto",
            "{coin} is definitely undervalued right now. Great buying opportunity! 🚀",
            "Bullish on {coin} after today's announcement. This could be huge for adoption.",
            "My {coin} investment is finally paying off! Diamond hands win again. 💎🙌",
            "Just read an amazing analysis on {coin}. The upside potential is enormous! #investing",
            "{coin} breaking resistance levels today. Next stop: the moon! 🌕 #ToTheMoon",
        ],
        'negative': [
            "Selling my {coin} bags. This project is going nowhere. #crypto",
            "{coin} dropping like a stone. Get out while you still can! 📉",
            "Lost faith in {coin} after latest developer update. Moving on to better projects.",
            "The {coin} chart looks terrible. Clear head and shoulders pattern forming. #bearish",
            "Disappointed by {coin}'s performance this quarter. Expected much better.",
            "Regulation fears hitting {coin} hard today. Might be time to reconsider positions.",
            "{coin} team failing to deliver on promises again. Red flags everywhere! 🚩",
        ],
        'neutral': [
            "Interesting analysis of {coin} market cycles. What do you think? #crypto",
            "Anyone else following the {coin} developments this week? Looking for opinions.",
            "Just transferred my {coin} to cold storage for long-term hodling.",
            "Comparing {coin} to other Layer 1 solutions. Pros and cons thread 🧵",
            "Weekly {coin} price discussion. Share your thoughts below! #crypto",
            "New to {coin} - what wallets do you recommend for secure storage?",
            "Trying to understand the tokenomics of {coin}. Any good resources out there?",
        ]
    }
    
    # Generate mock tweets
    mock_tweets = []
    now = datetime.now()
    
    for i in range(limit):
        # Select sentiment with weighted probabilities
        sentiment_type = random.choices(
            ['positive', 'negative', 'neutral'],
            weights=[0.4, 0.3, 0.3],
            k=1
        )[0]
        
        # Random time within the last week
        tweet_time = now - timedelta(
            days=random.randint(0, 6),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Use query topic or choose a random crypto term
        coin = topic.upper() if topic in [term.lower() for term in crypto_terms] else random.choice(crypto_terms)
        
        # Generate tweet text
        templates = tweet_templates[sentiment_type]
        content = random.choice(templates).format(coin=coin)
        
        # Calculate engagement metrics with some randomness
        retweet_count = int(random.expovariate(1/5))  # Exponential distribution for retweets
        like_count = int(retweet_count * random.uniform(1.5, 5))  # Likes usually exceed retweets
        reply_count = int(retweet_count * random.uniform(0.3, 1.2))  # Some fraction of retweets
        quote_count = int(retweet_count * random.uniform(0.1, 0.5))  # Usually fewer than retweets
        
        # Create tweet data
        tweet_data = {
            'id': f"mock_{i}_{int(datetime.now().timestamp())}",
            'date': tweet_time.isoformat(),
            'content': content,
            'username': f"user_{random.randint(1000, 9999)}",
            'retweet_count': retweet_count,
            'like_count': like_count,
            'reply_count': reply_count,
            'quote_count': quote_count,
            'lang': 'en'
        }
        
        mock_tweets.append(tweet_data)
    
    return mock_tweets

def filter_crypto_tweets(tweets: List[Dict[str, Any]], crypto_keywords: List[str] = None) -> List[Dict[str, Any]]:
    """
    Filter tweets containing specific cryptocurrency keywords.
    
    Args:
        tweets: List of tweet dictionaries
        crypto_keywords: List of keywords to filter by (default: common crypto terms)
        
    Returns:
        Filtered list of tweets
    """
    if not crypto_keywords:
        crypto_keywords = [
            "BTC", "Bitcoin", "ETH", "Ethereum", "crypto", "altcoin",
            "SOL", "Solana", "ADA", "Cardano", "XRP", "Ripple",
            "DOT", "Polkadot", "AVAX", "Avalanche", "MATIC", "Polygon",
            "DOGE", "Dogecoin", "SHIB", "Shiba", "NFT", "DeFi"
        ]
    
    filtered_tweets = []
    
    for tweet in tweets:
        content = tweet.get('content', '').upper()
        if any(keyword.upper() in content for keyword in crypto_keywords):
            filtered_tweets.append(tweet)
    
    return filtered_tweets

def analyze_twitter_sentiment(tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze sentiment of tweets.
    
    Args:
        tweets: List of tweet dictionaries
        
    Returns:
        Dictionary with sentiment analysis results
    """
    if not tweets:
        return {
            'count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'compound_score': 0.0,
            'average_score': 0.0,
            'weighted_score': 0.0,
            'sentiment': 'neutral'
        }
    
    # Analyze sentiment for each tweet
    total_score = 0.0
    weighted_total = 0.0
    weights_sum = 0
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    for tweet in tweets:
        content = tweet.get('content', '')
        score = analyze_sentiment(content)
        
        # Count by sentiment category
        if score > 0.05:
            positive_count += 1
        elif score < -0.05:
            negative_count += 1
        else:
            neutral_count += 1
        
        total_score += score
        
        # Use engagement metrics as weight
        weight = (
            tweet.get('retweet_count', 0) * 3 +  # Retweets are most valuable
            tweet.get('like_count', 0) * 1 +     # Likes are common
            tweet.get('reply_count', 0) * 2 +    # Replies show engagement
            tweet.get('quote_count', 0) * 3      # Quotes are high engagement
        )
        # Ensure minimum weight of 1
        weight = max(1, weight)
        
        weighted_total += score * weight
        weights_sum += weight
    
    # Calculate averages
    count = len(tweets)
    average_score = total_score / count if count > 0 else 0.0
    weighted_score = weighted_total / weights_sum if weights_sum > 0 else average_score
    
    # Determine overall sentiment
    if weighted_score > 0.05:
        sentiment = 'positive'
    elif weighted_score < -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return {
        'count': count,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'compound_score': weighted_score,  # For consistency with other sentiment sources
        'average_score': average_score,
        'weighted_score': weighted_score,
        'sentiment': sentiment
    }

def fetch_twitter_sentiment(
    query: str,
    limit: int = 100,
    days_back: int = 7,
    crypto_keywords: List[str] = None,
    use_mock: bool = False
) -> float:
    """
    Fetch and analyze sentiment from Twitter for a specific query.
    
    Args:
        query: Search query for tweets (e.g., 'bitcoin')
        limit: Maximum number of tweets to analyze
        days_back: Number of days to look back
        crypto_keywords: List of keywords to filter by
        use_mock: If True, use mock data
        
    Returns:
        Sentiment score as a float between -1 and 1
    """
    # Fetch tweets
    tweets = fetch_twitter_tweets(
        query=query,
        limit=limit,
        days_back=days_back
    )
    
    # Filter tweets by crypto keywords if needed
    if crypto_keywords:
        tweets = filter_crypto_tweets(tweets, crypto_keywords)
    
    # Analyze sentiment
    sentiment_data = analyze_twitter_sentiment(tweets)
    
    # Return the compound score
    return sentiment_data.get('compound_score', 0.0) 