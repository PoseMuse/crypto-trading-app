#!/usr/bin/env python
"""
Risk Management Demo Script.

This script demonstrates the usage of the risk management system for a
cryptocurrency trading bot.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

from risk_management.position_sizer import PositionSizer
from risk_management.circuit_breakers import CircuitBreaker
from risk_management.risk_manager import RiskManager


def plot_equity_curve(equity_history, drawdowns, safe_mode_periods, output_path):
    """
    Plot an equity curve with drawdowns and safe mode periods.
    
    Args:
        equity_history: List of (timestamp, equity) tuples
        drawdowns: List of (timestamp, drawdown) tuples
        safe_mode_periods: List of (start, end) tuples for safe mode periods
        output_path: Path to save the plot
    """
    # Convert to DataFrames
    equity_df = pd.DataFrame(equity_history, columns=['time', 'equity'])
    equity_df.set_index('time', inplace=True)
    
    drawdown_df = pd.DataFrame(drawdowns, columns=['time', 'drawdown'])
    drawdown_df.set_index('time', inplace=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    ax1.plot(equity_df.index, equity_df['equity'], label='Equity', color='blue')
    
    # Highlight safe mode periods
    for start, end in safe_mode_periods:
        ax1.axvspan(start, end, alpha=0.2, color='red', label='_nolegend_')
    
    # Mark the first safe mode period for the legend
    if safe_mode_periods:
        ax1.axvspan(safe_mode_periods[0][0], safe_mode_periods[0][0], alpha=0.2, color='red', label='Safe Mode')
    
    # Plot drawdowns
    ax2.fill_between(drawdown_df.index, 0, drawdown_df['drawdown'] * 100, color='red', alpha=0.5)
    ax2.plot(drawdown_df.index, drawdown_df['drawdown'] * 100, color='red', label='Drawdown')
    
    # Add horizontal line at 5% drawdown (typical circuit breaker level)
    ax2.axhline(y=5, color='black', linestyle='--', alpha=0.7, label='Circuit Breaker (5%)')
    
    # Set labels and titles
    ax1.set_title('Equity Curve with Risk Management')
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_ylim(0, max(drawdown_df['drawdown'] * 100) * 1.2)  # Add some space above max drawdown
    ax2.grid(True)
    ax2.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    
    plt.close(fig)


def simulate_trading_with_risk_management(
    initial_equity=10000.0,
    days=90,
    trades_per_day=2,
    win_rate=0.55,
    avg_win_pct=0.03,
    avg_loss_pct=0.02,
    volatility=0.015
):
    """
    Simulate trading with risk management.
    
    Args:
        initial_equity: Starting equity
        days: Number of days to simulate
        trades_per_day: Average number of trades per day
        win_rate: Probability of a winning trade
        avg_win_pct: Average win percentage
        avg_loss_pct: Average loss percentage
        volatility: Daily market volatility
        
    Returns:
        Tuple of (equity_history, drawdowns, safe_mode_periods, trade_journal)
    """
    # Initialize risk manager
    risk_manager = RiskManager(
        initial_equity=initial_equity,
        max_risk_per_trade=0.01,  # 1% risk per trade
        max_leverage=1.0,
        daily_drawdown_limit=0.05,  # 5% daily drawdown
        weekly_drawdown_limit=0.10,  # 10% weekly drawdown
        consecutive_loss_limit=3,  # 3 consecutive losses activates safe mode
        auto_reduce_risk=True,
        journal_path="output/risk_demo/trade_journal.json"
    )
    
    # Initialize simulation variables
    current_equity = initial_equity
    start_date = datetime.now() - timedelta(days=days)
    end_date = datetime.now()
    
    # Create daily timestamps
    timestamps = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Track safe mode periods
    safe_mode_periods = []
    safe_mode_start = None
    
    # Generate random daily prices for a simulated asset
    np.random.seed(42)  # For reproducibility
    price = 40000.0  # Starting price (e.g., BTC)
    prices = [price]
    
    # Generate daily returns with a slight upward bias
    daily_returns = np.random.normal(0.0005, volatility, size=len(timestamps))
    
    # Calculate daily prices
    for ret in daily_returns:
        price *= (1 + ret)
        prices.append(price)
    
    # Create price series
    price_series = pd.Series(prices[:len(timestamps)], index=timestamps)
    
    # Simulate trading day by day
    equity_history = [(start_date, current_equity)]
    drawdowns = [(start_date, 0.0)]
    
    # Track circuit breaker status
    circuit_breaker_status = risk_manager.get_circuit_breaker_status()
    
    for day, timestamp in enumerate(timestamps):
        # Get current price
        current_price = price_series[timestamp]
        
        # Decide how many trades to execute today (random around trades_per_day)
        num_trades = np.random.poisson(trades_per_day)
        
        # Execute trades for the day
        for _ in range(num_trades):
            # Generate trade direction (70% long, 30% short)
            direction = "long" if np.random.random() < 0.7 else "short"
            
            # Generate entry price with some intraday noise
            entry_price = current_price * (1 + np.random.normal(0, volatility * 0.3))
            
            # Calculate stop loss price (about 2% away)
            if direction == "long":
                stop_loss = entry_price * 0.98
            else:
                stop_loss = entry_price * 1.02
            
            # Check if we can trade (circuit breakers, etc.)
            position_info = risk_manager.calculate_position_size(
                entry_price=entry_price,
                stop_loss_price=stop_loss,
                symbol="BTC/USDT",
                direction=direction
            )
            
            # Skip if not allowed to trade
            if not position_info["allowed_to_trade"]:
                print(f"Trade skipped on {timestamp.date()}: {position_info.get('reason', 'Not allowed to trade')}")
                continue
            
            # Get position size
            position_size = position_info["position_size"]
            
            # Generate random exit price based on win rate
            is_win = np.random.random() < win_rate
            
            if is_win:
                # Winning trade
                if direction == "long":
                    exit_price = entry_price * (1 + np.random.normal(avg_win_pct, avg_win_pct * 0.3))
                else:
                    exit_price = entry_price * (1 - np.random.normal(avg_win_pct, avg_win_pct * 0.3))
            else:
                # Losing trade - usually stop loss with some slippage
                if direction == "long":
                    exit_price = stop_loss * (1 - np.random.normal(0.002, 0.001))  # Slippage
                else:
                    exit_price = stop_loss * (1 + np.random.normal(0.002, 0.001))  # Slippage
            
            # Calculate P&L
            if direction == "long":
                pnl = (exit_price - entry_price) * position_size
            else:
                pnl = (entry_price - exit_price) * position_size
            
            # Random trade duration (1 min to 1 day)
            duration_hours = np.random.uniform(1/60, 24)
            entry_time = timestamp + timedelta(hours=np.random.uniform(0, 24 - duration_hours))
            exit_time = entry_time + timedelta(hours=duration_hours)
            
            # Record the trade
            risk_manager.record_trade(
                symbol="BTC/USDT",
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=position_size,
                entry_time=entry_time,
                exit_time=exit_time,
                pnl=pnl,
                stop_loss=stop_loss,
                fees=pnl * 0.001 if pnl > 0 else 0  # Simple fee model
            )
            
            # Update equity
            current_equity += pnl
            
            # Update risk manager with new equity
            circuit_breaker_status = risk_manager.update_equity(current_equity)
        
        # Record equity and drawdown for the day
        equity_history.append((timestamp, current_equity))
        drawdowns.append((timestamp, circuit_breaker_status["daily_drawdown"]))
        
        # Track safe mode periods
        if circuit_breaker_status["safe_mode_active"]:
            if safe_mode_start is None:
                safe_mode_start = timestamp
        elif safe_mode_start is not None:
            # Safe mode ended
            safe_mode_periods.append((safe_mode_start, timestamp))
            safe_mode_start = None
    
    # Close any open safe mode period
    if safe_mode_start is not None:
        safe_mode_periods.append((safe_mode_start, timestamps[-1]))
    
    # Get trade analytics
    trade_analytics = risk_manager.get_trade_analytics()
    
    return equity_history, drawdowns, safe_mode_periods, trade_analytics


def main():
    """Run the risk management demo."""
    parser = argparse.ArgumentParser(description='Risk Management Demo')
    
    parser.add_argument('--initial-equity', type=float, default=10000.0,
                       help='Initial equity (default: 10000.0)')
    
    parser.add_argument('--days', type=int, default=90,
                       help='Number of days to simulate (default: 90)')
    
    parser.add_argument('--trades-per-day', type=float, default=2.0,
                       help='Average number of trades per day (default: 2.0)')
    
    parser.add_argument('--win-rate', type=float, default=0.55,
                       help='Win rate (default: 0.55)')
    
    parser.add_argument('--output-dir', type=str, default='output/risk_demo',
                       help='Output directory (default: output/risk_demo)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting risk management demo with:")
    print(f"  Initial equity: ${args.initial_equity:.2f}")
    print(f"  Simulation days: {args.days}")
    print(f"  Trades per day: {args.trades_per_day}")
    print(f"  Win rate: {args.win_rate:.2f}")
    print()
    
    # Run simulation
    print("Running trading simulation with risk management...")
    equity_history, drawdowns, safe_mode_periods, trade_analytics = simulate_trading_with_risk_management(
        initial_equity=args.initial_equity,
        days=args.days,
        trades_per_day=args.trades_per_day,
        win_rate=args.win_rate
    )
    
    # Generate equity curve plot
    plot_path = os.path.join(args.output_dir, "equity_curve.png")
    plot_equity_curve(equity_history, drawdowns, safe_mode_periods, plot_path)
    
    # Save trade analytics to file
    analytics_path = os.path.join(args.output_dir, "trade_analytics.txt")
    with open(analytics_path, 'w') as f:
        f.write("Trade Analytics Summary\n")
        f.write("======================\n\n")
        f.write(f"Total trades: {trade_analytics['total_trades']}\n")
        f.write(f"Winning trades: {trade_analytics['winning_trades']} ({trade_analytics['win_rate']:.2%})\n")
        f.write(f"Losing trades: {trade_analytics['losing_trades']}\n")
        f.write(f"Net profit: ${trade_analytics['net_profit']:.2f}\n")
        f.write(f"Initial equity: ${trade_analytics['initial_equity']:.2f}\n")
        f.write(f"Final equity: ${trade_analytics['current_equity']:.2f}\n")
        f.write(f"Total return: {trade_analytics['overall_return_pct']:.2f}%\n")
        f.write(f"Profit factor: {trade_analytics['profit_factor']:.2f}\n")
        f.write(f"Average win: ${trade_analytics['avg_win']:.2f}\n")
        f.write(f"Average loss: ${trade_analytics['avg_loss']:.2f}\n")
        f.write(f"Maximum drawdown: {trade_analytics['max_drawdown']:.2%}\n")
        f.write("\nRisk Management Settings:\n")
        f.write(f"Max risk per trade: {trade_analytics['risk_settings']['max_risk_per_trade']:.2%}\n")
        f.write(f"Daily drawdown limit: {trade_analytics['risk_settings']['daily_drawdown_limit']:.2%}\n")
        f.write(f"Weekly drawdown limit: {trade_analytics['risk_settings']['weekly_drawdown_limit']:.2%}\n")
        f.write(f"Consecutive loss limit: {trade_analytics['risk_settings']['consecutive_loss_limit']}\n")
    
    print(f"Trade analytics saved to: {analytics_path}")
    
    # Print summary
    print("\nSimulation Summary:")
    print(f"  Initial equity: ${args.initial_equity:.2f}")
    print(f"  Final equity: ${trade_analytics['current_equity']:.2f}")
    print(f"  Total return: {trade_analytics['overall_return_pct']:.2f}%")
    print(f"  Total trades: {trade_analytics['total_trades']}")
    print(f"  Win rate: {trade_analytics['win_rate']:.2%}")
    print(f"  Maximum drawdown: {trade_analytics['max_drawdown']:.2%}")
    print(f"  Safe mode activations: {len(safe_mode_periods)}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main() 