# Running with Real Sentiment Data

Now that all mocking has been removed, you can run the following commands to execute the system with real sentiment data:

## Backtesting with Real Sentiment Data

```bash
python src/backtesting/run_backtest.py --symbol BTC/USDT --timeframe 1d --use-sentiment --multi-source
```

## Paper Trading with Real Sentiment Data

```bash
python src/paper_trading_script.py --symbol BTC/USDT --enable-sentiment
```

## Environment Variables

Make sure to set the following environment variables for accessing the APIs:

```
# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent

# Telegram API
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash

# Exchange API (for paper trading)
BINANCE_API_KEY=your_api_key
BINANCE_SECRET=your_api_secret
```
