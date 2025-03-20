"""
Telegram Scraping and Sentiment Analysis for cryptocurrency trading.

This module contains functions to:
1. Connect to Telegram using telethon
2. Fetch messages from Telegram channels or groups
3. Filter and analyze sentiment of messages
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

# Import from our sentiment pipeline
from .sentiment_pipeline import analyze_sentiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import telethon, gracefully handle if not installed
try:
    from telethon import TelegramClient, events
    from telethon.errors import SessionPasswordNeededError
    TELETHON_AVAILABLE = True
except ImportError:
    logger.warning("Telethon not installed. Run 'pip install telethon' to use Telegram scraping.")
    TELETHON_AVAILABLE = False

def create_telegram_client(
    api_id: str = None,
    api_hash: str = None,
    session_name: str = "crypto_telegram_session"
) -> Any:
    """
    Create and return a Telegram client.
    
    Args:
        api_id: Telegram API ID from https://my.telegram.org/
        api_hash: Telegram API hash from https://my.telegram.org/
        session_name: Name for the session file
        
    Returns:
        TelegramClient instance or None if telethon is not installed
    """
    if not TELETHON_AVAILABLE:
        logger.error("Cannot create Telegram client: telethon is not installed")
        return None
    
    # If not provided, try to get API credentials from environment variables
    api_id = api_id or os.environ.get("TELEGRAM_API_ID")
    api_hash = api_hash or os.environ.get("TELEGRAM_API_HASH")
    
    if not api_id or not api_hash:
        logger.error(
            "Telegram API credentials not provided. Set TELEGRAM_API_ID and "
            "TELEGRAM_API_HASH environment variables or pass them as arguments."
        )
        return None
    
    try:
        client = TelegramClient(session_name, api_id, api_hash)
        return client
    except Exception as e:
        logger.error(f"Error creating Telegram client: {e}")
        return None

async def _fetch_telegram_messages_async(
    client: Any,
    channel_username: str,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Async function to fetch messages from a Telegram channel or group.
    
    Args:
        client: TelegramClient instance
        channel_username: Username/name of the channel (with or without '@')
        limit: Maximum number of messages to fetch
        
    Returns:
        List of dictionaries containing message data
    """
    if not client:
        return []
    
    messages = []
    
    try:
        # Ensure channel username has the right format
        if channel_username.startswith('@'):
            channel_username = channel_username[1:]
        
        # Get the entity
        entity = await client.get_entity(channel_username)
        
        # Iterate through messages
        async for msg in client.iter_messages(entity, limit=limit):
            if msg.text:  # Only include messages with text
                message_data = {
                    'id': msg.id,
                    'channel': channel_username,
                    'text': msg.text,
                    'date': msg.date.isoformat() if msg.date else None,
                    'views': getattr(msg, 'views', 0),
                    'forwards': getattr(msg, 'forwards', 0),
                    'from_id': str(msg.from_id) if msg.from_id else None
                }
                messages.append(message_data)
        
    except Exception as e:
        logger.error(f"Error fetching messages from {channel_username}: {e}")
    
    return messages

def fetch_telegram_messages(
    channel_username: str,
    limit: int = 100,
    api_id: str = None,
    api_hash: str = None,
    session_name: str = "crypto_telegram_session"
) -> List[Dict[str, Any]]:
    """
    Fetch messages from a Telegram channel or group.
    
    Args:
        channel_username: Username of the channel (with or without '@')
        limit: Maximum number of messages to fetch
        api_id: Telegram API ID
        api_hash: Telegram API hash
        session_name: Name for the session file
        
    Returns:
        List of dictionaries containing message data
    """
    if not TELETHON_AVAILABLE:
        logger.warning("Telethon not installed, please run 'pip install telethon'")
        return []
    
    client = create_telegram_client(api_id, api_hash, session_name)
    if not client:
        logger.warning("Failed to create Telegram client, please check API credentials")
        return []
    
    try:
        # Run the async function to fetch messages
        loop = asyncio.get_event_loop()
        messages = loop.run_until_complete(_fetch_telegram_messages_async(client, channel_username, limit))
        return messages
    
    except Exception as e:
        logger.error(f"Error in fetch_telegram_messages: {e}")
        return []

def _mock_telegram_messages(channel_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Generate mock Telegram message data for testing.
    
    Args:
        channel_name: Name of the channel
        limit: Number of messages to generate
        
    Returns:
        List of dictionaries with mock message data
    """
    import random
    from datetime import datetime, timedelta
    
    crypto_terms = [
        "BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "SHIB", 
        "bull market", "bear market", "breakout", "support", "resistance",
        "Bitcoin", "Ethereum", "altcoin season", "DeFi", "NFT", "staking"
    ]
    
    # Sample message templates with different sentiments
    message_templates = {
        'positive': [
            "Just bought more {coin}! Feeling bullish 🚀",
            "Market analysis shows {coin} could reach new ATH soon",
            "Strong signal for {coin}, big green candles ahead 📈",
            "{coin}'s recent development is impressive",
            "Accumulating {coin} at these prices is a no-brainer",
            "{coin} looking strong despite market conditions",
            "Technical analysis suggests {coin} is ready for a pump 🔥",
        ],
        'negative': [
            "Dumping my {coin} bags, too much uncertainty",
            "{coin} breaking down key support levels 📉",
            "Not looking good for {coin} holders right now",
            "Bearish pattern forming on {coin} chart, be careful",
            "Getting out of {coin} before it drops further",
            "Warning: {coin} might see more downside in coming days",
            "Sentiment around {coin} has turned extremely negative",
        ],
        'neutral': [
            "Daily update on {coin}: price stabilizing around current levels",
            "Market analysis: {coin} vs altcoins performance",
            "What's your take on {coin} this week?",
            "Interesting developments in the {coin} ecosystem",
            "Comparing {coin} to other cryptocurrencies",
            "Latest news for {coin}: partnership announcements and updates",
            "Weekly overview: {coin} market statistics and volume",
        ]
    }
    
    # Generate mock messages
    mock_messages = []
    now = datetime.now()
    
    for i in range(limit):
        # Select sentiment with weighted probabilities
        sentiment_type = random.choices(
            ['positive', 'negative', 'neutral'],
            weights=[0.4, 0.3, 0.3],
            k=1
        )[0]
        
        # Random time within the last week
        message_time = now - timedelta(
            days=random.randint(0, 6),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Choose a random crypto term
        coin = random.choice(crypto_terms)
        
        # Generate message text
        templates = message_templates[sentiment_type]
        text = random.choice(templates).format(coin=coin)
        
        # Add some randomness to message length
        if random.random() > 0.7:
            extra_sentences = [
                f"This is not financial advice.",
                f"Do your own research!",
                f"What do you think?",
                f"Looking for opinions.",
                f"Chart analysis available on TradingView."
            ]
            text += f" {random.choice(extra_sentences)}"
        
        # Create message data
        message_data = {
            'id': i + 1,
            'channel': channel_name,
            'text': text,
            'date': message_time.isoformat(),
            'views': random.randint(100, 10000),
            'forwards': random.randint(0, 1000),
            'from_id': f"user_{random.randint(10000, 99999)}"
        }
        
        mock_messages.append(message_data)
    
    return mock_messages

def filter_crypto_messages(messages: List[Dict[str, Any]], crypto_keywords: List[str] = None) -> List[Dict[str, Any]]:
    """
    Filter messages containing specific cryptocurrency keywords.
    
    Args:
        messages: List of message dictionaries
        crypto_keywords: List of keywords to filter by (default: common crypto terms)
        
    Returns:
        Filtered list of messages
    """
    if not crypto_keywords:
        crypto_keywords = [
            "BTC", "Bitcoin", "ETH", "Ethereum", "crypto", "altcoin",
            "SOL", "Solana", "ADA", "Cardano", "XRP", "Ripple",
            "DOT", "Polkadot", "AVAX", "Avalanche", "MATIC", "Polygon",
            "DOGE", "Dogecoin", "SHIB", "Shiba", "NFT", "DeFi"
        ]
    
    filtered_messages = []
    
    for message in messages:
        text = message.get('text', '').upper()
        if any(keyword.upper() in text for keyword in crypto_keywords):
            filtered_messages.append(message)
    
    return filtered_messages

def analyze_telegram_sentiment(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze sentiment of Telegram messages.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Dictionary with sentiment analysis results
    """
    if not messages:
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
    
    # Analyze sentiment for each message
    total_score = 0.0
    weighted_total = 0.0
    weights_sum = 0
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    for message in messages:
        text = message.get('text', '')
        score = analyze_sentiment(text)
        
        # Count by sentiment category
        if score > 0.05:
            positive_count += 1
        elif score < -0.05:
            negative_count += 1
        else:
            neutral_count += 1
        
        total_score += score
        
        # Weight by views or forwards if available
        weight = message.get('views', 1) + message.get('forwards', 0) * 2
        weighted_total += score * weight
        weights_sum += weight
    
    # Calculate averages
    count = len(messages)
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
        'compound_score': weighted_score,  # For consistency with Reddit sentiment
        'average_score': average_score,
        'weighted_score': weighted_score,
        'sentiment': sentiment
    }

def fetch_telegram_sentiment(
    channel_username: str,
    days: int = 3,
    limit: int = 100,
    api_id: str = None,
    api_hash: str = None,
    crypto_keywords: List[str] = None,
    use_mock: bool = False
) -> float:
    """
    Fetch and analyze sentiment from a Telegram channel or group.
    
    Args:
        channel_username: Username of the channel (with or without '@')
        days: Number of days to consider
        limit: Maximum number of messages to fetch
        api_id: Telegram API ID
        api_hash: Telegram API hash
        crypto_keywords: List of keywords to filter by
        use_mock: If True, use mock data
        
    Returns:
        Sentiment score as a float between -1 and 1
    """
    # Fetch messages
    messages = fetch_telegram_messages(
        channel_username=channel_username,
        limit=limit,
        api_id=api_id,
        api_hash=api_hash
    )
    
    # Filter messages by date (last n days)
    if days > 0:
        cutoff_date = (datetime.now() - timedelta(days=days))
        filtered_by_date = []
        for msg in messages:
            msg_date = msg.get('date')
            if msg_date:
                try:
                    # Handle both string ISO format and datetime objects
                    if isinstance(msg_date, str):
                        msg_datetime = datetime.fromisoformat(msg_date)
                    else:
                        msg_datetime = msg_date
                    
                    if msg_datetime >= cutoff_date:
                        filtered_by_date.append(msg)
                except (ValueError, TypeError):
                    # If date parsing fails, include the message anyway
                    filtered_by_date.append(msg)
            else:
                # If there's no date, include the message
                filtered_by_date.append(msg)
        
        messages = filtered_by_date
    
    # Filter messages by crypto keywords
    if crypto_keywords:
        messages = filter_crypto_messages(messages, crypto_keywords)
    
    # Analyze sentiment
    sentiment_data = analyze_telegram_sentiment(messages)
    
    # Return the compound score
    return sentiment_data.get('compound_score', 0.0) 