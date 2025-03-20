"""
Live Trader module for paper trading.

This module provides functionality to run live paper trading simulations
using Backtrader with strategies from the backtesting module.
"""

import os
import pandas as pd
import numpy as np
import backtrader as bt
import ccxt
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from .live_feed import CCXTStore
from ..backtesting.strategies import (
    BasicSMAStrategy,
    SentimentStrategy,
    AIModelStrategy,
    CombinedStrategy,
)


class LiveTrader:
    """
    Class for running live paper trading simulations.
    
    This class uses Backtrader to run paper trading simulations with
    the same strategies used in backtesting, but with live data.
    """
    
    def __init__(
        self,
        symbol: str = "BTC/USDT",
        cash: float = 10000.0,
        timeframe: str = "1m",
        commission: float = 0.001,  # 0.1% commission
        exchange: str = "binance",
        output_dir: str = "output/paper_trading",
        log_level: str = "INFO"
    ):
        """
        Initialize the live trader.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            cash: Initial cash for paper trading
            timeframe: Data timeframe ('1m', '5m', '1h', etc.)
            commission: Commission rate for trades
            exchange: Exchange to use (e.g., 'binance')
            output_dir: Directory to save trading results
            log_level: Logging level
        """
        self.symbol = symbol
        self.cash = cash
        self.timeframe = timeframe
        self.commission = commission
        self.exchange = exchange
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging(log_level)
        
        # Initialize cerebro
        self.cerebro = bt.Cerebro()
        
        # Set initial cash
        self.cerebro.broker.setcash(cash)
        
        # Set commission
        self.cerebro.broker.setcommission(commission=commission)
        
        # Store trade history
        self.trade_history = []
        
        # Store cerebro instances for each strategy
        self.strategy_cerebros = {}
        
        # Initialize exchange store
        self.store = None
        
    def setup_logging(self, log_level: str):
        """
        Set up logging for the live trader.
        
        Args:
            log_level: Logging level
        """
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, "live_trader.log")),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("LiveTrader")
        self.logger.info(f"Initialized LiveTrader for {self.symbol}")
        
    def _load_api_keys(self) -> Dict:
        """
        Load API keys from environment variables.
        
        Returns:
            Dict containing API configuration
        """
        import os
        from dotenv import load_dotenv
        
        # Load .env file
        load_dotenv()
        
        # Exchange-specific configurations
        if self.exchange.lower() == "binance":
            return {
                "apiKey": os.getenv("BINANCE_API_KEY", ""),
                "secret": os.getenv("BINANCE_SECRET", ""),
                "enableRateLimit": True,
                "options": {"defaultType": "spot"}
            }
        elif self.exchange.lower() == "coinbase":
            return {
                "apiKey": os.getenv("COINBASE_API_KEY", ""),
                "secret": os.getenv("COINBASE_SECRET", ""),
                "enableRateLimit": True
            }
        else:
            # Generic configuration
            return {
                "enableRateLimit": True
            }
    
    def connect_exchange(self) -> bool:
        """
        Connect to the exchange.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load API keys
            config = self._load_api_keys()
            
            # Create store
            self.store = CCXTStore(self.exchange, config=config)
            
            # Log success
            self.logger.info(f"Connected to exchange: {self.exchange}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to exchange: {e}")
            return False
    
    def add_data_feed(
        self,
        historical_days: int = 1,
        from_date: Optional[datetime] = None
    ):
        """
        Add data feed to Backtrader engine.
        
        Args:
            historical_days: Number of days of historical data to load
            from_date: Start date for historical data
        """
        if self.store is None:
            self.logger.error("Exchange connection not initialized")
            return False
        
        # Determine start date for historical data
        if from_date is None:
            from_date = datetime.now() - timedelta(days=historical_days)
        
        # Get data feed
        data = self.store.get_datafeed(
            dataname=self.symbol,
            timeframe=self.timeframe,
            from_date=from_date,
            ohlcv_limit=500  # Reasonable limit for initial historical data
        )
        
        # Add data feed to cerebro
        self.cerebro.adddata(data)
        
        self.logger.info(f"Added data feed for {self.symbol} with timeframe {self.timeframe}")
        return True
        
    def add_strategy(
        self,
        strategy_class: bt.Strategy,
        strategy_name: str,
        strategy_params: Optional[Dict] = None
    ):
        """
        Add a strategy to the Backtrader engine.
        
        Args:
            strategy_class: Backtrader strategy class
            strategy_name: Name for the strategy
            strategy_params: Parameters for the strategy
        """
        # Create a new cerebro instance for this strategy
        cerebro = bt.Cerebro()
        
        # Set initial cash
        cerebro.broker.setcash(self.cash)
        
        # Set commission
        cerebro.broker.setcommission(commission=self.commission)
        
        # Add strategy
        if strategy_params:
            cerebro.addstrategy(strategy_class, **strategy_params)
        else:
            cerebro.addstrategy(strategy_class)
        
        # Store for later use
        self.strategy_cerebros[strategy_name] = {
            "cerebro": cerebro,
            "strategy_class": strategy_class,
            "strategy_params": strategy_params
        }
        
        self.logger.info(f"Added strategy: {strategy_name}")
        
    def run_paper_trading(
        self,
        strategy_name: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        max_trades: Optional[int] = None
    ):
        """
        Run paper trading simulation.
        
        Args:
            strategy_name: Name of the strategy to run (if None, run all added strategies)
            duration_seconds: Maximum duration in seconds (if None, run indefinitely)
            max_trades: Maximum number of trades to execute (if None, no limit)
        """
        # Check if we have an exchange connection
        if self.store is None:
            self.logger.error("Exchange connection not initialized")
            return False
        
        if strategy_name and strategy_name not in self.strategy_cerebros:
            self.logger.error(f"Strategy not found: {strategy_name}")
            return False
        
        # Get the strategies to run
        if strategy_name:
            strategies_to_run = {strategy_name: self.strategy_cerebros[strategy_name]}
        else:
            strategies_to_run = self.strategy_cerebros
        
        # Record start time
        start_time = time.time()
        trade_counts = {name: 0 for name in strategies_to_run}
        
        # Run the strategies
        for name, strategy_info in strategies_to_run.items():
            cerebro = strategy_info["cerebro"]
            
            # Add data feed
            data = self.store.get_datafeed(
                dataname=self.symbol,
                timeframe=self.timeframe,
                from_date=datetime.now() - timedelta(days=1)
            )
            cerebro.adddata(data)
            
            # Add observers for trade tracking
            cerebro.addobserver(bt.observers.Trades)
            cerebro.addobserver(bt.observers.BuySell)
            cerebro.addobserver(bt.observers.Value)
            cerebro.addobserver(bt.observers.Broker)
            cerebro.addobserver(bt.observers.DrawDown)
            
            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
            
            # Start strategy in a separate thread (for demo purposes we'll run sequentially)
            self.logger.info(f"Starting paper trading with strategy: {name}")
            
            # Run the strategy for a set number of steps
            strategy = cerebro.run()[0]
            
            # Save results
            self._save_results(name, strategy)
            
        self.logger.info("Paper trading completed")
        return True
    
    def _save_results(self, strategy_name: str, strategy: bt.Strategy):
        """
        Save trading results.
        
        Args:
            strategy_name: Name of the strategy
            strategy: Backtrader strategy instance
        """
        # Create results directory
        results_dir = os.path.join(self.output_dir, strategy_name)
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save portfolio value
        portfolio_value = {
            "timestamp": timestamp,
            "initial_cash": self.cash,
            "final_value": strategy.broker.getvalue(),
            "returns_pct": (strategy.broker.getvalue() / self.cash - 1) * 100
        }
        
        with open(os.path.join(results_dir, f"portfolio_{timestamp}.json"), "w") as f:
            json.dump(portfolio_value, f, indent=4)
        
        # Save trade history if available
        if hasattr(strategy, "trades") and strategy.trades:
            with open(os.path.join(results_dir, f"trades_{timestamp}.json"), "w") as f:
                json.dump(strategy.trades, f, indent=4)
        
        # Save analyzer results
        analyzers = {}
        
        # Get trade analysis
        if hasattr(strategy, "analyzers") and hasattr(strategy.analyzers, "trades"):
            trade_analysis = strategy.analyzers.trades.get_analysis()
            analyzers["trades"] = {
                "total": trade_analysis.get("total", {}).get("total", 0),
                "won": trade_analysis.get("won", {}).get("total", 0),
                "lost": trade_analysis.get("lost", {}).get("total", 0),
                "pnl": {
                    "net": trade_analysis.get("pnl", {}).get("net", 0),
                    "average": trade_analysis.get("pnl", {}).get("average", 0)
                }
            }
        
        # Get Sharpe ratio
        if hasattr(strategy, "analyzers") and hasattr(strategy.analyzers, "sharpe"):
            sharpe = strategy.analyzers.sharpe.get_analysis()
            analyzers["sharpe"] = sharpe.get("sharperatio", 0)
        
        # Get drawdown
        if hasattr(strategy, "analyzers") and hasattr(strategy.analyzers, "drawdown"):
            drawdown = strategy.analyzers.drawdown.get_analysis()
            analyzers["drawdown"] = {
                "max": drawdown.get("max", {}).get("drawdown", 0),
                "len": drawdown.get("max", {}).get("len", 0)
            }
        
        # Get returns
        if hasattr(strategy, "analyzers") and hasattr(strategy.analyzers, "returns"):
            returns = strategy.analyzers.returns.get_analysis()
            analyzers["returns"] = {
                "total": returns.get("rtot", 0),
                "average": returns.get("ravg", 0),
                "annualized": returns.get("rnorm", 0)
            }
        
        with open(os.path.join(results_dir, f"analyzers_{timestamp}.json"), "w") as f:
            json.dump(analyzers, f, indent=4)
        
        self.logger.info(f"Saved results for strategy: {strategy_name}")


def start_standard_paper_trading(
    symbol: str = "BTC/USDT",
    cash: float = 10000.0,
    timeframe: str = "5m",
    exchange: str = "binance",
    duration_hours: Optional[float] = 24
):
    """
    Start standard paper trading with all available strategies.
    
    Args:
        symbol: Trading pair symbol
        cash: Initial cash
        timeframe: Data timeframe
        exchange: Exchange to use
        duration_hours: Duration in hours (if None, run indefinitely)
    """
    # Create live trader
    trader = LiveTrader(
        symbol=symbol,
        cash=cash,
        timeframe=timeframe,
        exchange=exchange
    )
    
    # Connect to exchange
    if not trader.connect_exchange():
        return False
    
    # Add data feed
    trader.add_data_feed(historical_days=1)
    
    # Add strategies
    trader.add_strategy(BasicSMAStrategy, "SMA_Crossover", {"fast_period": 10, "slow_period": 30})
    trader.add_strategy(SentimentStrategy, "Sentiment", {"sentiment_threshold": 0.2})
    trader.add_strategy(AIModelStrategy, "AI_Model", {"threshold": 0.05})
    trader.add_strategy(CombinedStrategy, "Combined", {"confidence_threshold": 0.3})
    
    # Convert duration to seconds
    duration_seconds = None
    if duration_hours:
        duration_seconds = int(duration_hours * 3600)
    
    # Run paper trading
    trader.run_paper_trading(duration_seconds=duration_seconds)
    
    return True 