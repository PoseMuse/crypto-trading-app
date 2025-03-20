#!/usr/bin/env python
"""
Backtesting Demo Script for Cryptocurrency Trading Bot.

This script demonstrates the backtesting capabilities of the trading bot,
comparing different trading strategies including ones that utilize
sentiment analysis from social media.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
from pathlib import Path

from src.backtesting.backtest_runner import BacktestRunner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run cryptocurrency trading backtests')
    
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                       help='Trading pair to backtest (default: BTC/USDT)')
    
    parser.add_argument('--cash', type=float, default=10000.0,
                       help='Initial cash for backtesting (default: 10000.0)')
    
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date for backtesting (default: 2020-01-01)')
    
    parser.add_argument('--end-date', type=str, default='2022-12-31',
                       help='End date for backtesting (default: 2022-12-31)')
    
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Trading commission (default: 0.001 or 0.1%)')
    
    parser.add_argument('--strategies', type=str, nargs='+', 
                       default=['basic', 'sentiment', 'ai', 'combined'],
                       help='Strategies to backtest (default: all)')
    
    parser.add_argument('--output-dir', type=str, default='output/backtests',
                       help='Directory to save backtest results (default: output/backtests)')
    
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plotting (default: False)')
    
    parser.add_argument('--use-cached', action='store_true',
                       help='Use cached data if available (default: False)')
    
    parser.add_argument('--report', action='store_true',
                       help='Generate a detailed PDF report (default: False)')
    
    return parser.parse_args()


def generate_report(runner, output_dir):
    """Generate a detailed PDF report of backtest results."""
    try:
        from fpdf import FPDF
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        print("Generating detailed backtest report...")
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, f'Cryptocurrency Trading Bot Backtest Report', 0, 1, 'C')
        pdf.ln(5)
        
        # Summary information
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Symbol: {runner.symbol}', 0, 1)
        pdf.cell(0, 10, f'Initial Cash: ${runner.cash:.2f}', 0, 1)
        pdf.cell(0, 10, f'Commission: {runner.commission:.3f}', 0, 1)
        pdf.ln(5)
        
        # Strategy comparison table
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Strategy Comparison', 0, 1)
        
        comparison_df = runner.compare_strategies(save_to_csv=True)
        
        # Add comparison table
        pdf.set_font('Arial', 'B', 10)
        
        # Table header
        cols = ['Strategy', 'Annual Return', 'Sharpe', 'Max DD', 'Sortino']
        col_widths = [60, 30, 30, 30, 30]
        for i, col in enumerate(cols):
            pdf.cell(col_widths[i], 10, col, 1, 0, 'C')
        pdf.ln()
        
        # Table data
        pdf.set_font('Arial', '', 10)
        for _, row in comparison_df.iterrows():
            pdf.cell(col_widths[0], 10, str(row['strategy']), 1)
            
            annual_return = row['annual_return']
            if annual_return is not None:
                pdf.cell(col_widths[1], 10, f"{annual_return:.2%}", 1)
            else:
                pdf.cell(col_widths[1], 10, "N/A", 1)
                
            sharpe = row['sharpe_ratio']
            if sharpe is not None:
                pdf.cell(col_widths[2], 10, f"{sharpe:.2f}", 1)
            else:
                pdf.cell(col_widths[2], 10, "N/A", 1)
                
            drawdown = row['max_drawdown']
            if drawdown is not None:
                pdf.cell(col_widths[3], 10, f"{drawdown:.2%}", 1)
            else:
                pdf.cell(col_widths[3], 10, "N/A", 1)
                
            sortino = row['sortino_ratio']
            if sortino is not None:
                pdf.cell(col_widths[4], 10, f"{sortino:.2f}", 1)
            else:
                pdf.cell(col_widths[4], 10, "N/A", 1)
            
            pdf.ln()
        
        pdf.ln(10)
        
        # Add strategy charts
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Strategy Performance Charts', 0, 1)
        pdf.ln(5)
        
        # For each strategy, add its chart
        for strategy_name in runner.results.keys():
            strategy_dir = os.path.join(output_dir, strategy_name.replace(" ", "_"))
            chart_path = os.path.join(strategy_dir, f"{strategy_name}_backtest.png")
            
            if os.path.exists(chart_path):
                # Add strategy title
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, strategy_name, 0, 1)
                
                # Add chart
                pdf.image(chart_path, x=10, y=None, w=180)
                pdf.ln(10)
        
        # Save the PDF
        pdf_path = os.path.join(output_dir, "backtest_report.pdf")
        pdf.output(pdf_path, 'F')
        print(f"Report saved to {pdf_path}")
        
    except ImportError:
        print("FPDF library not found. Please install it with: pip install fpdf")
    except Exception as e:
        print(f"Error generating report: {e}")


def main():
    """Run the backtesting demo."""
    # Parse command line arguments
    args = parse_args()
    
    # Convert date strings to datetime objects
    start_date = datetime.fromisoformat(args.start_date)
    end_date = datetime.fromisoformat(args.end_date)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print banner
    print("="*80)
    print("Cryptocurrency Trading Bot - Backtesting Demo".center(80))
    print("="*80)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Initial Cash: ${args.cash:.2f}")
    print(f"Commission: {args.commission:.3f} ({args.commission*100:.1f}%)")
    print(f"Strategies: {', '.join(args.strategies)}")
    print("="*80)
    
    # Initialize backtest runner
    runner = BacktestRunner(
        symbol=args.symbol,
        cash=args.cash,
        timeframe="1d",
        commission=args.commission,
        output_dir=args.output_dir
    )
    
    # Prepare data
    print("\nPreparing data...")
    data_path = runner.prepare_data(
        start_date=start_date,
        end_date=end_date,
        include_sentiment=True,
        sentiment_lookback=7,
        use_cached=args.use_cached
    )
    
    # Configure strategies to run
    include_basic = 'basic' in args.strategies
    include_sentiment = 'sentiment' in args.strategies
    include_ai = 'ai' in args.strategies
    include_combined = 'combined' in args.strategies
    
    # Run backtests
    print("\nRunning backtests...")
    results = runner.run_all_strategies(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        include_basic=include_basic,
        include_sentiment=include_sentiment,
        include_ai=include_ai,
        include_combined=include_combined,
        plot_results=not args.no_plots
    )
    
    # Compare strategies
    print("\nStrategy Comparison:")
    comparison_df = runner.compare_strategies()
    print(comparison_df)
    
    # Generate PDF report if requested
    if args.report:
        generate_report(runner, args.output_dir)
    
    print("\nBacktesting completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 