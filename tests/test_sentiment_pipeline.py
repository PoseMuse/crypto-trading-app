"""
Unit tests for the sentiment analysis pipeline.
"""

import pytest
import pandas as pd
import json
from unittest.mock import patch, MagicMock

from src.sentiment_analysis.sentiment_pipeline import (
    fetch_reddit_posts,
    analyze_sentiment,
    aggregate_sentiment,
    _mock_reddit_posts
)

@pytest.fixture
def sample_posts():
    """Create a sample set of Reddit posts for testing."""
    return [
        {
            'id': 'post1',
            'subreddit': 'bitcoin',
            'title': 'Bitcoin is going to the moon!',
            'selftext': 'I am very bullish on Bitcoin and think it will reach new highs soon.',
            'score': 100,
            'num_comments': 25,
            'fulltext': 'Bitcoin is going to the moon! I am very bullish on Bitcoin and think it will reach new highs soon.'
        },
        {
            'id': 'post2',
            'subreddit': 'cryptocurrency',
            'title': 'Market analysis shows bearish signals',
            'selftext': 'We might see a significant drop in prices. Be careful and consider taking profits.',
            'score': 50,
            'num_comments': 15,
            'fulltext': 'Market analysis shows bearish signals. We might see a significant drop in prices. Be careful and consider taking profits.'
        },
        {
            'id': 'post3',
            'subreddit': 'ethereum',
            'title': 'Ethereum 2.0 update discussion',
            'selftext': 'Let\'s talk about the technical aspects of the upcoming Ethereum update.',
            'score': 75,
            'num_comments': 30,
            'fulltext': 'Ethereum 2.0 update discussion. Let\'s talk about the technical aspects of the upcoming Ethereum update.'
        }
    ]

def test_mock_reddit_posts():
    """Test the mock Reddit posts generator."""
    subreddits = ['bitcoin', 'ethereum']
    limit = 10
    
    posts = _mock_reddit_posts(subreddits, limit)
    
    # Check that we got the right number of posts
    assert len(posts) <= limit
    
    # Check that the posts have the expected structure
    for post in posts:
        assert 'id' in post
        assert 'subreddit' in post
        assert 'title' in post
        assert 'selftext' in post
        assert 'created_utc' in post
        assert 'score' in post
        assert 'num_comments' in post
        assert 'author' in post
        assert 'url' in post
        assert 'fulltext' in post
        
        # Check that the subreddit is one of the requested ones
        assert post['subreddit'] in subreddits

def test_fetch_reddit_posts_mock():
    """Test fetching Reddit posts with mock data."""
    subreddits = ['bitcoin', 'ethereum']
    limit = 5
    
    posts = fetch_reddit_posts(subreddits, limit, use_mock=True)
    
    # Check that we got the right number of posts
    assert len(posts) <= limit
    
    # Check that each post has the necessary fields
    for post in posts:
        assert 'id' in post
        assert 'subreddit' in post
        assert 'title' in post
        assert 'fulltext' in post

def test_analyze_sentiment_positive():
    """Test analyzing sentiment for positive text."""
    positive_text = "I'm very excited about this project! It's going to be amazing and bring great returns!"
    sentiment = analyze_sentiment(positive_text)
    
    # Check that the sentiment is positive
    assert sentiment > 0
    assert -1 <= sentiment <= 1  # Check range

def test_analyze_sentiment_negative():
    """Test analyzing sentiment for negative text."""
    negative_text = "This project is disappointing. The price keeps falling and I'm losing money."
    sentiment = analyze_sentiment(negative_text)
    
    # Check that the sentiment is negative
    assert sentiment < 0
    assert -1 <= sentiment <= 1  # Check range

def test_analyze_sentiment_neutral():
    """Test analyzing sentiment for neutral text."""
    neutral_text = "Here is the latest update on the project. The team is working on implementation."
    sentiment = analyze_sentiment(neutral_text)
    
    # Check that the sentiment is near neutral
    assert -0.3 <= sentiment <= 0.3
    assert -1 <= sentiment <= 1  # Check range

def test_analyze_sentiment_empty():
    """Test analyzing sentiment for empty text."""
    sentiment = analyze_sentiment("")
    
    # Empty text should return a neutral sentiment (0.0)
    assert sentiment == 0.0

def test_aggregate_sentiment(sample_posts):
    """Test aggregating sentiment across multiple posts."""
    result = aggregate_sentiment(sample_posts)
    
    # Check that the result has the expected structure
    assert 'overall_sentiment' in result
    assert 'weighted_sentiment' in result
    assert 'positive_ratio' in result
    assert 'negative_ratio' in result
    assert 'neutral_ratio' in result
    assert 'post_count' in result
    
    # Check that the values are in the expected ranges
    assert -1 <= result['overall_sentiment'] <= 1
    assert -1 <= result['weighted_sentiment'] <= 1
    assert 0 <= result['positive_ratio'] <= 1
    assert 0 <= result['negative_ratio'] <= 1
    assert 0 <= result['neutral_ratio'] <= 1
    assert result['post_count'] == len(sample_posts)
    
    # Check that the ratios sum to 1
    assert abs(result['positive_ratio'] + result['negative_ratio'] + result['neutral_ratio'] - 1.0) < 0.001

def test_aggregate_sentiment_with_weights(sample_posts):
    """Test aggregating sentiment with different weighting strategies."""
    # Test with score weighting
    result_score = aggregate_sentiment(sample_posts, weight_key='score')
    
    # Test with comment weighting
    result_comments = aggregate_sentiment(sample_posts, weight_key='num_comments')
    
    # Test with no weighting
    result_none = aggregate_sentiment(sample_posts, weight_key=None)
    
    # All results should be valid
    assert -1 <= result_score['weighted_sentiment'] <= 1
    assert -1 <= result_comments['weighted_sentiment'] <= 1
    assert -1 <= result_none['weighted_sentiment'] <= 1
    
    # The unweighted overall sentiment should be the same in all cases
    assert result_score['overall_sentiment'] == result_comments['overall_sentiment']
    assert result_score['overall_sentiment'] == result_none['overall_sentiment']
    
    # For this specific test data, the weighted results might differ
    # This isn't a strict requirement, but it's likely for realistic data
    # that different weighting schemes produce different results.

def test_aggregate_sentiment_empty():
    """Test aggregating sentiment with no posts."""
    result = aggregate_sentiment([])
    
    # Check that default values are returned
    assert result['overall_sentiment'] == 0.0
    assert result['weighted_sentiment'] == 0.0
    assert result['positive_ratio'] == 0.0
    assert result['negative_ratio'] == 0.0
    assert result['neutral_ratio'] == 0.0
    assert result['post_count'] == 0 