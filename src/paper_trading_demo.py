#!/usr/bin/env python
"""
Paper Trading Demo Script for Cryptocurrency Trading Bot.

This script demonstrates the paper trading capabilities of the trading bot,
using the same strategies as backtesting but with live market data.
"""

import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from src.paper_trading.live_trader import LiveTrader, start_standard_paper_trading
from src.backtesting.strategies import (
    BasicSMAStrategy,
    SentimentStrategy,
    AIModelStrategy,
    CombinedStrategy
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run cryptocurrency trading paper trading simulation')
    
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                       help='Trading pair to use (default: BTC/USDT)')
    
    parser.add_argument('--cash', type=float, default=10000.0,
                       help='Initial cash for paper trading (default: 10000.0)')
    
    parser.add_argument('--timeframe', type=str, default='5m',
                       help='Data timeframe (default: 5m)')
    
    parser.add_argument('--exchange', type=str, default='binance',
                       help='Exchange to use (default: binance)')
    
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Trading commission (default: 0.001 or 0.1%)')
    
    parser.add_argument('--duration', type=float, default=24.0,
                       help='Duration to run in hours (default: 24 hours)')
    
    parser.add_argument('--strategy', type=str, default=None,
                       choices=['basic', 'sentiment', 'ai', 'combined', None],
                       help='Strategy to run (default: run all)')
    
    parser.add_argument('--output-dir', type=str, default='output/paper_trading',
                       help='Directory to save results (default: output/paper_trading)')
    
    return parser.parse_args()


def run_single_strategy(args):
    """Run a single strategy for paper trading."""
    # Create live trader
    trader = LiveTrader(
        symbol=args.symbol,
        cash=args.cash,
        timeframe=args.timeframe,
        commission=args.commission,
        exchange=args.exchange,
        output_dir=args.output_dir
    )
    
    # Connect to exchange
    if not trader.connect_exchange():
        print("Failed to connect to exchange")
        return
    
    # Add data feed
    trader.add_data_feed(historical_days=1)
    
    # Add the selected strategy
    if args.strategy == 'basic':
        trader.add_strategy(
            BasicSMAStrategy, 
            "SMA_Crossover", 
            {"fast_period": 10, "slow_period": 30}
        )
    elif args.strategy == 'sentiment':
        trader.add_strategy(
            SentimentStrategy, 
            "Sentiment", 
            {"sentiment_threshold": 0.2}
        )
    elif args.strategy == 'ai':
        trader.add_strategy(
            AIModelStrategy, 
            "AI_Model", 
            {"threshold": 0.05}
        )
    elif args.strategy == 'combined':
        trader.add_strategy(
            CombinedStrategy, 
            "Combined", 
            {"confidence_threshold": 0.3}
        )
    
    # Convert duration to seconds
    duration_seconds = int(args.duration * 3600) if args.duration else None
    
    # Run paper trading
    trader.run_paper_trading(duration_seconds=duration_seconds)
    
    print(f"Paper trading completed for {args.strategy} strategy")


def main():
    """Run the paper trading demo."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Starting paper trading for {args.symbol} on {args.exchange}...")
    print(f"Initial cash: ${args.cash:.2f}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Commission: {args.commission * 100:.2f}%")
    print(f"Duration: {args.duration} hours")
    
    if args.strategy:
        # Run a single strategy
        print(f"Running {args.strategy} strategy")
        run_single_strategy(args)
    else:
        # Run all strategies
        print("Running all strategies")
        start_standard_paper_trading(
            symbol=args.symbol,
            cash=args.cash,
            timeframe=args.timeframe,
            exchange=args.exchange,
            duration_hours=args.duration
        )
    
    print("Paper trading demo completed")


if __name__ == "__main__":
    main() 