#!/usr/bin/env python
"""
Paper Trading Script for Cryptocurrency Trading Bot.

This script implements paper trading using real-time data from cryptocurrency
exchanges via CCXT, applying the same strategy logic used in backtesting.
"""

import os
import sys
import time
import json
import ccxt
import argparse
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Import our backtesting components
from backtesting.backtesting_pipeline import CryptoStrategy, SignalData


class CCXTStore:
    """
    Store for communicating with CCXT exchange API.
    """
    
    def __init__(self, exchange_id: str, config: Optional[Dict] = None):
        """
        Initialize the CCXT store.
        
        Args:
            exchange_id: ID of the exchange (e.g., 'binance')
            config: Exchange configuration
        """
        self.exchange_id = exchange_id
        self.config = config or {}
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class(self.config)
        
        # Enable rate limiting
        self.exchange.enableRateLimit = True
        
        print(f"Initialized {exchange_id} exchange connection")


class CCXTData(bt.feeds.PandasData):
    """
    Data feed for CCXT real-time data.
    
    This class extends the PandasData feed to include AI and sentiment signals.
    """
    
    lines = ('ai_signal', 'sentiment',)
    params = (
        ('datetime', None),
        ('open', -1),
        ('high', -1),
        ('low', -1),
        ('close', -1),
        ('volume', -1),
        ('openinterest', None),
        ('ai_signal', -1),
        ('sentiment', -1),
    )


class PaperTrader:
    """
    Paper trader for cryptocurrency trading strategies.
    
    This class manages the paper trading environment, connecting to exchanges,
    fetching real-time data, and executing strategies.
    """
    
    def __init__(
        self,
        exchange_id: str = 'binance',
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        strategy: Any = CryptoStrategy,
        strategy_params: Optional[Dict] = None,
        initial_cash: float = 10000,
        commission: float = 0.001,
        output_dir: str = 'output/paper_trading',
        enable_sentiment: bool = False
    ):
        """
        Initialize the paper trader.
        
        Args:
            exchange_id: ID of the exchange to use
            symbol: Symbol to trade (e.g., 'BTC/USDT')
            timeframe: Data timeframe (e.g., '1h', '15m')
            strategy: Strategy class to use
            strategy_params: Parameters for the strategy
            initial_cash: Initial cash amount
            commission: Commission rate
            output_dir: Directory for output files
            enable_sentiment: Whether to enable sentiment analysis
        """
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy_class = strategy
        self.strategy_params = strategy_params or {}
        self.initial_cash = initial_cash
        self.commission = commission
        self.output_dir = output_dir
        self.enable_sentiment = enable_sentiment
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize variables
        self.store = None
        self.cerebro = bt.Cerebro()
        self.data_feed = None
        self.historical_data = None
        self.running = False
        self.start_time = None
        self.end_time = None
        self.last_processed_time = None
        self.trade_history = []
        
        # Load environment variables
        load_dotenv()
        
        # Configure cerebro
        self._setup_cerebro()
    
    def _setup_cerebro(self):
        """Set up the Backtrader cerebro instance."""
        # Add the strategy
        if self.strategy_params:
            self.cerebro.addstrategy(self.strategy_class, **self.strategy_params)
        else:
            self.cerebro.addstrategy(self.strategy_class)
        
        # Set initial cash
        self.cerebro.broker.setcash(self.initial_cash)
        
        # Set commission
        self.cerebro.broker.setcommission(commission=self.commission)
        
        # Add analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        print(f"Cerebro initialized with {self.strategy_class.__name__} strategy")
    
    def _get_api_credentials(self) -> Dict:
        """
        Get API credentials from environment variables.
        
        Returns:
            Dictionary with API credentials
        """
        api_key = os.getenv(f"{self.exchange_id.upper()}_API_KEY", "")
        api_secret = os.getenv(f"{self.exchange_id.upper()}_SECRET", "")
        
        return {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        }
    
    def connect(self) -> bool:
        """
        Connect to the exchange.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get API credentials
            config = self._get_api_credentials()
            
            # Initialize store
            self.store = CCXTStore(self.exchange_id, config)
            
            # Test connection
            self.store.exchange.fetch_ticker(self.symbol)
            
            print(f"Connected to {self.exchange_id} exchange")
            return True
        except Exception as e:
            print(f"Error connecting to exchange: {e}")
            return False
    
    def _fetch_historical_data(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch historical data from the exchange.
        
        Args:
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.store.exchange.fetch_ohlcv(
                self.symbol,
                timeframe=self.timeframe,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add dummy AI and sentiment signals for now
            # In a real implementation, these would come from the respective modules
            df['ai_signal'] = 0.0
            df['sentiment'] = 0.0
            
            return df
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def _fetch_ai_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch AI signals for the given data.
        
        In a real implementation, this would call the AI model to generate predictions.
        For now, we'll use a simple moving average crossover as a placeholder.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with AI signals
        """
        # Create a copy of the data
        df = data.copy()
        
        # Calculate fast and slow moving averages
        df['fast_ma'] = df['close'].rolling(window=10).mean()
        df['slow_ma'] = df['close'].rolling(window=30).mean()
        
        # Calculate the signal (-1 to 1)
        df['ai_signal'] = (df['fast_ma'] - df['slow_ma']) / df['slow_ma']
        
        # Normalize the signal to between -1 and 1
        max_signal = df['ai_signal'].abs().max()
        if max_signal > 0:
            df['ai_signal'] = df['ai_signal'] / max_signal
        
        # Drop the intermediate columns
        df.drop(['fast_ma', 'slow_ma'], axis=1, inplace=True)
        
        return df
    
    def _fetch_sentiment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch sentiment data for the given data.
        
        Uses the sentiment analysis module to get real sentiment data for the current symbol.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with sentiment data
        """
        # Create a copy of the data
        df = data.copy()
        
        try:
            # Extract the base currency for sentiment analysis
            base_currency = self.symbol.split('/')[0]
            
            # Import sentiment modules here to avoid circular imports
            from sentiment_analysis.sentiment_pipeline import (
                fetch_reddit_posts, 
                aggregate_sentiment,
                aggregate_multisource_sentiment
            )
            from sentiment_analysis.telegram_pipeline import fetch_telegram_messages
            from sentiment_analysis.twitter_pipeline import fetch_twitter_tweets
            
            # Define sources to search
            subreddits = [f"crypto", f"{base_currency.lower()}", "cryptocurrency"]
            telegram_channels = [f"crypto_{base_currency.lower()}", "cryptosignals"]
            twitter_query = f"{base_currency} crypto"
            
            # Fetch sentiment data from multiple sources
            print(f"Fetching sentiment data for {base_currency} from multiple sources...")
            
            # Reddit
            reddit_posts = fetch_reddit_posts(
                subreddits=subreddits,
                limit=100,
                time_filter="day"
            )
            
            # Telegram
            telegram_messages = []
            for channel in telegram_channels:
                channel_messages = fetch_telegram_messages(
                    channel_username=channel,
                    limit=100
                )
                telegram_messages.extend(channel_messages)
            
            # Twitter
            twitter_tweets = fetch_twitter_tweets(
                query=twitter_query,
                limit=100,
                days_back=1
            )
            
            # Aggregate sentiment from all sources
            if reddit_posts or telegram_messages or twitter_tweets:
                # Source weights - can be adjusted based on reliability
                source_weights = {
                    'reddit': 1.0,    # Full weight for Reddit
                    'telegram': 0.5,  # Half weight for Telegram
                    'twitter': 1.0    # Full weight for Twitter
                }
                
                sentiment_data = aggregate_multisource_sentiment(
                    reddit_posts=reddit_posts,
                    telegram_messages=telegram_messages,
                    twitter_tweets=twitter_tweets,
                    source_weights=source_weights
                )
                
                # Set the sentiment value for all rows
                sentiment_score = sentiment_data.get('compound_score', 0)
                df['sentiment'] = sentiment_score
                
                print(f"Sentiment for {base_currency}: {sentiment_score}")
                print(f"  Reddit: {len(reddit_posts)} posts")
                print(f"  Telegram: {len(telegram_messages)} messages")
                print(f"  Twitter: {len(twitter_tweets)} tweets")
            else:
                print(f"No sentiment data found for {base_currency}")
                df['sentiment'] = 0.0
        except Exception as e:
            print(f"Error fetching sentiment data: {e}")
            df['sentiment'] = 0.0
        
        return df
    
    def _prepare_data(self) -> pd.DataFrame:
        """
        Prepare data for paper trading.
        
        This fetches historical data and adds AI and sentiment signals.
        
        Returns:
            DataFrame with prepared data
        """
        # Fetch historical data
        data = self._fetch_historical_data(limit=100)
        if data.empty:
            print("Error: Failed to fetch historical data")
            return pd.DataFrame()
        
        # Add AI signals
        data = self._fetch_ai_signals(data)
        
        # Add sentiment data if enabled
        if self.enable_sentiment:
            print("Fetching sentiment data...")
            data = self._fetch_sentiment_data(data)
        else:
            print("Sentiment analysis disabled")
            data['sentiment'] = 0.0
        
        # Save historical data
        self.historical_data = data.copy()
        self.last_processed_time = data.index[-1]
        
        return data
    
    def _create_data_feed(self, data: pd.DataFrame) -> None:
        """
        Create a data feed from the prepared data.
        
        Args:
            data: DataFrame with prepared data
        """
        # Create data feed
        self.data_feed = CCXTData(
            dataname=data,
            datetime=None,  # Index is the datetime
            open=data['open'].values,
            high=data['high'].values,
            low=data['low'].values,
            close=data['close'].values,
            volume=data['volume'].values,
            ai_signal=data['ai_signal'].values,
            sentiment=data['sentiment'].values
        )
        
        # Add data feed to cerebro
        self.cerebro.adddata(self.data_feed)
        
        print("Data feed created and added to cerebro")
    
    def _fetch_new_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch new data since the last processed time.
        
        Returns:
            DataFrame with new data, or None if no new data
        """
        try:
            # Calculate since time
            since = int(self.last_processed_time.timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = self.store.exchange.fetch_ohlcv(
                self.symbol,
                timeframe=self.timeframe,
                since=since
            )
            
            # If no new data, return None
            if not ohlcv or len(ohlcv) <= 1:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filter to only include new data
            df = df[df.index > self.last_processed_time]
            
            # If no new data after filtering, return None
            if df.empty:
                return None
            
            # Add AI signals
            df = self._fetch_ai_signals(df)
            
            # Add sentiment data
            if self.enable_sentiment:
                df = self._fetch_sentiment_data(df)
            
            # Update last processed time
            self.last_processed_time = df.index[-1]
            
            return df
        except Exception as e:
            print(f"Error fetching new data: {e}")
            return None
    
    def _update_data_feed(self, new_data: pd.DataFrame) -> None:
        """
        Update the data feed with new data.
        
        Args:
            new_data: New data to add to the feed
        """
        # Add AI signals
        df = self._fetch_ai_signals(new_data)
        
        # Add sentiment data if enabled
        if self.enable_sentiment:
            print("Updating sentiment data...")
            df = self._fetch_sentiment_data(df)
        
        # Update the data feed
        print(f"Updating data feed with candle at {df.index[-1]}")
        self.data_feed.add_data_row(df.iloc[-1])
        
        # Update the last processed time
        self.last_processed_time = df.index[-1]
    
    def start(self, duration_minutes: Optional[int] = None) -> Dict:
        """
        Start paper trading.
        
        Args:
            duration_minutes: Duration to run in minutes (None for indefinite)
            
        Returns:
            Dictionary with performance metrics
        """
        if not self.store:
            if not self.connect():
                print("Error: Failed to connect to exchange")
                return {}
        
        # Prepare data
        data = self._prepare_data()
        if data.empty:
            print("Error: Failed to prepare data")
            return {}
        
        # Create data feed
        self._create_data_feed(data)
        
        # Set start and end times
        self.start_time = datetime.now()
        self.end_time = None if duration_minutes is None else (self.start_time + timedelta(minutes=duration_minutes))
        
        # Start running
        self.running = True
        
        print(f"Starting paper trading for {self.symbol} on {self.exchange_id}")
        print(f"Initial cash: ${self.initial_cash:.2f}")
        if self.end_time:
            print(f"Running until: {self.end_time.isoformat()}")
        else:
            print("Running indefinitely (press Ctrl+C to stop)")
        
        try:
            # Main loop
            while self.running:
                # Check if duration has elapsed
                if self.end_time and datetime.now() >= self.end_time:
                    print(f"Duration elapsed, stopping at {self.end_time.isoformat()}")
                    self.running = False
                    break
                
                # Fetch new data
                new_data = self._fetch_new_data()
                if new_data is not None and not new_data.empty:
                    # Update data feed
                    self._update_data_feed(new_data)
                    
                    # Run one step of the strategy
                    self.cerebro.run(runonce=False, preload=False)
                    
                    # Get current portfolio value
                    portfolio_value = self.cerebro.broker.getvalue()
                    
                    # Display current status
                    returns_pct = (portfolio_value / self.initial_cash - 1) * 100
                    print(f"{datetime.now().isoformat()} - Portfolio: ${portfolio_value:.2f} ({returns_pct:+.2f}%)")
                
                # Sleep before next update
                time_to_sleep = self._get_sleep_time()
                time.sleep(time_to_sleep)
            
            # Run the final step
            self.cerebro.run(runonce=False, preload=False)
            
            # Get results
            results = self.cerebro.runstrats[0][0]
            
            # Save results
            self._save_results(results)
            
            # Return performance metrics
            return self._get_performance_metrics(results)
            
        except KeyboardInterrupt:
            print("Paper trading interrupted by user")
            
            # Run the final step
            self.cerebro.run(runonce=False, preload=False)
            
            # Get results
            results = self.cerebro.runstrats[0][0]
            
            # Save results
            self._save_results(results)
            
            # Return performance metrics
            return self._get_performance_metrics(results)
    
    def _get_sleep_time(self) -> float:
        """
        Calculate the time to sleep between updates.
        
        Returns:
            Time to sleep in seconds
        """
        # Calculate sleep time based on timeframe
        if self.timeframe.endswith('m'):
            minutes = int(self.timeframe[:-1])
            return max(1, minutes * 60 / 10)  # Check 10 times per period
        elif self.timeframe.endswith('h'):
            hours = int(self.timeframe[:-1])
            return max(1, hours * 60 * 60 / 20)  # Check 20 times per period
        else:
            return 60  # Default to 1 minute
    
    def _save_results(self, results: Any) -> None:
        """
        Save the results of the paper trading.
        
        Args:
            results: Results from cerebro run
        """
        try:
            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save trade history
            if hasattr(results, 'trades') and results.trades:
                # Convert trades to serializable format
                trade_list = []
                for trade in results.trades:
                    trade_dict = {
                        'entry_price': float(trade['entry_price']),
                        'exit_price': float(trade['exit_price']),
                        'entry_time': trade['entry_time'].isoformat(),
                        'exit_time': trade['exit_time'].isoformat(),
                        'result_pct': float(trade['result_pct']),
                        'result_cash': float(trade['result_cash'])
                    }
                    trade_list.append(trade_dict)
                
                # Save trades to JSON
                trades_file = os.path.join(self.output_dir, f"trades_{timestamp}.json")
                with open(trades_file, 'w') as f:
                    json.dump(trade_list, f, indent=4)
                    
                print(f"Trade history saved to {trades_file}")
            
            # Save performance metrics
            metrics = self._get_performance_metrics(results)
            metrics_file = os.path.join(self.output_dir, f"metrics_{timestamp}.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
                
            print(f"Performance metrics saved to {metrics_file}")
            
            # Save historical data
            if self.historical_data is not None:
                data_file = os.path.join(self.output_dir, f"data_{timestamp}.csv")
                self.historical_data.to_csv(data_file)
                
                print(f"Historical data saved to {data_file}")
        
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def _get_performance_metrics(self, results: Any) -> Dict:
        """
        Get performance metrics from the results.
        
        Args:
            results: Results from cerebro run
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Get the final portfolio value
            final_value = self.cerebro.broker.getvalue()
            
            # Calculate performance metrics
            pnl = final_value - self.initial_cash
            returns_pct = (final_value / self.initial_cash - 1) * 100
            
            # Extract analyzer results
            sharpe_ratio = results.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
            max_drawdown = results.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0.0)
            drawdown_length = results.analyzers.drawdown.get_analysis().get('max', {}).get('len', 0)
            annual_return = results.analyzers.returns.get_analysis().get('rnorm100', 0.0)
            
            # Extract trade metrics
            trade_analysis = results.analyzers.trades.get_analysis()
            total_trades = trade_analysis.get('total', {}).get('total', 0)
            won_trades = trade_analysis.get('won', {}).get('total', 0)
            lost_trades = trade_analysis.get('lost', {}).get('total', 0)
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0.0
            
            # Create metrics dictionary
            metrics = {
                'symbol': self.symbol,
                'exchange': self.exchange_id,
                'timeframe': self.timeframe,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': datetime.now().isoformat(),
                'initial_cash': self.initial_cash,
                'final_value': float(final_value),
                'pnl': float(pnl),
                'returns_pct': float(returns_pct),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown_pct': float(max_drawdown * 100),
                'drawdown_length': drawdown_length,
                'annual_return_pct': float(annual_return),
                'total_trades': total_trades,
                'won_trades': won_trades,
                'lost_trades': lost_trades,
                'win_rate_pct': float(win_rate),
                'strategy': self.strategy_class.__name__,
                'strategy_params': self.strategy_params
            }
            
            # Print summary
            print("\nPaper Trading Summary:")
            print(f"Symbol: {self.symbol} on {self.exchange_id}")
            print(f"Timeframe: {self.timeframe}")
            print(f"Initial Cash: ${self.initial_cash:.2f}")
            print(f"Final Value: ${final_value:.2f}")
            print(f"P&L: ${pnl:.2f} ({returns_pct:.2f}%)")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
            print(f"Annual Return: {annual_return:.2f}%")
            print(f"Total Trades: {total_trades}, Won: {won_trades}, Lost: {lost_trades}")
            print(f"Win Rate: {win_rate:.2f}%")
            
            return metrics
            
        except Exception as e:
            print(f"Error getting performance metrics: {e}")
            return {}


def main():
    """Run the paper trading script."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run paper trading for cryptocurrency')
    parser.add_argument('--exchange', type=str, default='binance',
                       help='Exchange to use (default: binance)')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                       help='Symbol to trade (default: BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default='1h',
                       help='Timeframe to use (default: 1h)')
    parser.add_argument('--cash', type=float, default=10000.0,
                       help='Initial cash amount (default: 10000.0)')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Commission rate (default: 0.001 or 0.1%)')
    parser.add_argument('--duration', type=int, default=None,
                       help='Duration to run in minutes (default: indefinite)')
    parser.add_argument('--output-dir', type=str, default='output/paper_trading',
                       help='Output directory for results (default: output/paper_trading)')
    parser.add_argument('--enable-sentiment', action='store_true',
                       help='Enable real-time sentiment analysis (default: False)')
    
    # Strategy parameters
    parser.add_argument('--ai-threshold', type=float, default=0.6,
                       help='Threshold for AI signals (default: 0.6)')
    parser.add_argument('--sentiment-threshold', type=float, default=0.3,
                       help='Threshold for sentiment signals (default: 0.3)')
    parser.add_argument('--ai-weight', type=float, default=0.7,
                       help='Weight for AI signals vs sentiment (default: 0.7)')
    parser.add_argument('--stop-loss', type=float, default=0.05,
                       help='Stop loss percentage (default: 0.05 or 5%)')
    parser.add_argument('--take-profit', type=float, default=0.15,
                       help='Take profit percentage (default: 0.15 or 15%)')
    
    args = parser.parse_args()
    
    # Create strategy parameters dictionary
    strategy_params = {
        'ai_threshold': args.ai_threshold,
        'sentiment_threshold': args.sentiment_threshold,
        'ai_weight': args.ai_weight,
        'stop_loss': args.stop_loss,
        'take_profit': args.take_profit
    }
    
    # Create paper trader
    trader = PaperTrader(
        exchange_id=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        strategy=CryptoStrategy,
        strategy_params=strategy_params,
        initial_cash=args.cash,
        commission=args.commission,
        output_dir=args.output_dir,
        enable_sentiment=args.enable_sentiment
    )
    
    # Start paper trading
    trader.start(duration_minutes=args.duration)


if __name__ == "__main__":
    main() 