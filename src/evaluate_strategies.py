#!/usr/bin/env python
"""
Strategy Evaluation Script for Cryptocurrency Trading Bot.

This script compares the performance of trading strategies between
backtesting and paper trading to evaluate their real-world effectiveness.
"""

import os
import argparse
from datetime import datetime
from pathlib import Path

from src.paper_trading.strategy_evaluator import StrategyEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trading strategies')
    
    parser.add_argument('--backtest-dir', type=str, default='output/backtests',
                       help='Directory containing backtest results (default: output/backtests)')
    
    parser.add_argument('--paper-dir', type=str, default='output/paper_trading',
                       help='Directory containing paper trading results (default: output/paper_trading)')
    
    parser.add_argument('--output-dir', type=str, default='output/evaluation',
                       help='Directory to save evaluation results (default: output/evaluation)')
    
    parser.add_argument('--strategy', type=str, default=None,
                       help='Specific strategy to evaluate (default: evaluate all)')
    
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate an HTML report of the evaluation')
    
    return parser.parse_args()


def main():
    """Run the strategy evaluation."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Starting strategy evaluation...")
    print(f"Backtest results directory: {args.backtest_dir}")
    print(f"Paper trading results directory: {args.paper_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize evaluator
    evaluator = StrategyEvaluator(
        backtest_dir=args.backtest_dir,
        paper_trading_dir=args.paper_dir,
        output_dir=args.output_dir
    )
    
    # Load results
    print("Loading backtest results...")
    evaluator.load_backtest_results(args.strategy)
    
    print("Loading paper trading results...")
    evaluator.load_paper_trading_results(args.strategy)
    
    # Compare strategies
    print("Comparing strategies...")
    comparison = evaluator.compare_strategies(args.strategy)
    
    # Display comparison
    if comparison.empty:
        print("No valid comparison data found.")
        return
    
    print("\nStrategy Comparison:")
    print(comparison)
    
    # Generate plots
    print("\nGenerating plots...")
    evaluator.plot_returns_comparison()
    
    # Generate radar charts for each strategy
    for strategy in comparison["Strategy"]:
        print(f"Generating radar chart for {strategy}...")
        evaluator.plot_metrics_radar(strategy)
    
    # Generate report if requested
    if args.generate_report:
        print("\nGenerating HTML report...")
        report_path = evaluator.generate_evaluation_report()
        print(f"Report saved to: {report_path}")
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main() 