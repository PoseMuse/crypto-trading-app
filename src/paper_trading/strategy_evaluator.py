"""
Strategy Evaluator module for comparing paper trading and backtesting results.

This module provides functionality to compare the performance of trading strategies
between paper trading and backtesting to evaluate their real-world effectiveness.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

# Visualization styling
import matplotlib as mpl
mpl.style.use('seaborn-v0_8-darkgrid')


class StrategyEvaluator:
    """
    Class for evaluating and comparing strategy performance between
    backtesting and paper trading.
    """
    
    def __init__(
        self,
        backtest_dir: str = "output/backtests",
        paper_trading_dir: str = "output/paper_trading",
        output_dir: str = "output/evaluation"
    ):
        """
        Initialize the strategy evaluator.
        
        Args:
            backtest_dir: Directory containing backtest results
            paper_trading_dir: Directory containing paper trading results
            output_dir: Directory to save evaluation results
        """
        self.backtest_dir = backtest_dir
        self.paper_trading_dir = paper_trading_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize data structures
        self.backtest_results = {}
        self.paper_trading_results = {}
        self.comparison_metrics = {}
        
    def load_backtest_results(self, strategy_name: Optional[str] = None) -> Dict:
        """
        Load backtest results from files.
        
        Args:
            strategy_name: Name of strategy to load (if None, load all)
            
        Returns:
            Dictionary containing loaded backtest results
        """
        results = {}
        
        # Determine which strategy directories to process
        if strategy_name:
            strategy_dirs = [os.path.join(self.backtest_dir, strategy_name)]
        else:
            strategy_dirs = [
                os.path.join(self.backtest_dir, d) 
                for d in os.listdir(self.backtest_dir) 
                if os.path.isdir(os.path.join(self.backtest_dir, d))
            ]
        
        # Process each strategy directory
        for strategy_dir in strategy_dirs:
            if not os.path.isdir(strategy_dir):
                continue
                
            strategy_name = os.path.basename(strategy_dir)
            results[strategy_name] = {
                "portfolio": [],
                "trades": [],
                "analyzers": []
            }
            
            # Load portfolio files
            portfolio_files = [
                f for f in os.listdir(strategy_dir) 
                if f.startswith("portfolio_") and f.endswith(".json")
            ]
            
            for file in portfolio_files:
                try:
                    with open(os.path.join(strategy_dir, file), "r") as f:
                        data = json.load(f)
                        results[strategy_name]["portfolio"].append(data)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
            
            # Load trade files
            trade_files = [
                f for f in os.listdir(strategy_dir) 
                if f.startswith("trades_") and f.endswith(".json")
            ]
            
            for file in trade_files:
                try:
                    with open(os.path.join(strategy_dir, file), "r") as f:
                        data = json.load(f)
                        results[strategy_name]["trades"].append(data)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
            
            # Load analyzer files
            analyzer_files = [
                f for f in os.listdir(strategy_dir) 
                if f.startswith("analyzers_") and f.endswith(".json")
            ]
            
            for file in analyzer_files:
                try:
                    with open(os.path.join(strategy_dir, file), "r") as f:
                        data = json.load(f)
                        results[strategy_name]["analyzers"].append(data)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        
        # Store results
        self.backtest_results = results
        return results
    
    def load_paper_trading_results(self, strategy_name: Optional[str] = None) -> Dict:
        """
        Load paper trading results from files.
        
        Args:
            strategy_name: Name of strategy to load (if None, load all)
            
        Returns:
            Dictionary containing loaded paper trading results
        """
        results = {}
        
        # Determine which strategy directories to process
        if strategy_name:
            strategy_dirs = [os.path.join(self.paper_trading_dir, strategy_name)]
        else:
            strategy_dirs = [
                os.path.join(self.paper_trading_dir, d) 
                for d in os.listdir(self.paper_trading_dir) 
                if os.path.isdir(os.path.join(self.paper_trading_dir, d))
            ]
        
        # Process each strategy directory
        for strategy_dir in strategy_dirs:
            if not os.path.isdir(strategy_dir):
                continue
                
            strategy_name = os.path.basename(strategy_dir)
            results[strategy_name] = {
                "portfolio": [],
                "trades": [],
                "analyzers": []
            }
            
            # Load portfolio files
            portfolio_files = [
                f for f in os.listdir(strategy_dir) 
                if f.startswith("portfolio_") and f.endswith(".json")
            ]
            
            for file in portfolio_files:
                try:
                    with open(os.path.join(strategy_dir, file), "r") as f:
                        data = json.load(f)
                        results[strategy_name]["portfolio"].append(data)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
            
            # Load trade files
            trade_files = [
                f for f in os.listdir(strategy_dir) 
                if f.startswith("trades_") and f.endswith(".json")
            ]
            
            for file in trade_files:
                try:
                    with open(os.path.join(strategy_dir, file), "r") as f:
                        data = json.load(f)
                        results[strategy_name]["trades"].append(data)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
            
            # Load analyzer files
            analyzer_files = [
                f for f in os.listdir(strategy_dir) 
                if f.startswith("analyzers_") and f.endswith(".json")
            ]
            
            for file in analyzer_files:
                try:
                    with open(os.path.join(strategy_dir, file), "r") as f:
                        data = json.load(f)
                        results[strategy_name]["analyzers"].append(data)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        
        # Store results
        self.paper_trading_results = results
        return results
    
    def compare_strategies(self, strategy_name: Optional[str] = None) -> pd.DataFrame:
        """
        Compare strategy performance between backtesting and paper trading.
        
        Args:
            strategy_name: Name of strategy to compare (if None, compare all)
            
        Returns:
            DataFrame with comparison metrics
        """
        # Load results if not already loaded
        if not self.backtest_results:
            self.load_backtest_results(strategy_name)
            
        if not self.paper_trading_results:
            self.load_paper_trading_results(strategy_name)
        
        # Determine which strategies to compare
        if strategy_name:
            strategies = [strategy_name]
        else:
            # Find common strategies between backtest and paper trading
            strategies = list(
                set(self.backtest_results.keys()) & 
                set(self.paper_trading_results.keys())
            )
        
        # Create comparison data
        comparison_data = []
        
        for strategy in strategies:
            # Skip if strategy isn't in both results
            if (strategy not in self.backtest_results or 
                strategy not in self.paper_trading_results):
                print(f"Strategy {strategy} not found in both results")
                continue
            
            # Get most recent results
            backtest_portfolio = (
                self.backtest_results[strategy]["portfolio"][-1] 
                if self.backtest_results[strategy]["portfolio"] else {}
            )
            
            paper_portfolio = (
                self.paper_trading_results[strategy]["portfolio"][-1] 
                if self.paper_trading_results[strategy]["portfolio"] else {}
            )
            
            backtest_analyzers = (
                self.backtest_results[strategy]["analyzers"][-1] 
                if self.backtest_results[strategy]["analyzers"] else {}
            )
            
            paper_analyzers = (
                self.paper_trading_results[strategy]["analyzers"][-1] 
                if self.paper_trading_results[strategy]["analyzers"] else {}
            )
            
            # Calculate comparison metrics
            try:
                # Returns
                backtest_returns = backtest_portfolio.get("returns_pct", 0)
                paper_returns = paper_portfolio.get("returns_pct", 0)
                returns_diff = paper_returns - backtest_returns
                
                # Sharpe ratio
                backtest_sharpe = backtest_analyzers.get("sharpe", 0)
                paper_sharpe = paper_analyzers.get("sharpe", 0)
                sharpe_diff = paper_sharpe - backtest_sharpe
                
                # Max drawdown
                backtest_drawdown = (
                    backtest_analyzers.get("drawdown", {}).get("max", 0)
                    if isinstance(backtest_analyzers.get("drawdown"), dict) else 0
                )
                
                paper_drawdown = (
                    paper_analyzers.get("drawdown", {}).get("max", 0)
                    if isinstance(paper_analyzers.get("drawdown"), dict) else 0
                )
                
                drawdown_diff = paper_drawdown - backtest_drawdown
                
                # Win rate
                backtest_trades = backtest_analyzers.get("trades", {})
                paper_trades = paper_analyzers.get("trades", {})
                
                backtest_total = (
                    backtest_trades.get("total", 0) 
                    if isinstance(backtest_trades, dict) else 0
                )
                
                backtest_won = (
                    backtest_trades.get("won", 0) 
                    if isinstance(backtest_trades, dict) else 0
                )
                
                paper_total = (
                    paper_trades.get("total", 0) 
                    if isinstance(paper_trades, dict) else 0
                )
                
                paper_won = (
                    paper_trades.get("won", 0) 
                    if isinstance(paper_trades, dict) else 0
                )
                
                backtest_winrate = (
                    backtest_won / backtest_total if backtest_total > 0 else 0
                )
                
                paper_winrate = (
                    paper_won / paper_total if paper_total > 0 else 0
                )
                
                winrate_diff = paper_winrate - backtest_winrate
                
                # Add to comparison data
                comparison_data.append({
                    "Strategy": strategy,
                    "Backtest Returns (%)": backtest_returns,
                    "Paper Returns (%)": paper_returns,
                    "Returns Diff (%)": returns_diff,
                    "Backtest Sharpe": backtest_sharpe,
                    "Paper Sharpe": paper_sharpe,
                    "Sharpe Diff": sharpe_diff,
                    "Backtest Drawdown (%)": backtest_drawdown * 100,
                    "Paper Drawdown (%)": paper_drawdown * 100,
                    "Drawdown Diff (%)": drawdown_diff * 100,
                    "Backtest Win Rate (%)": backtest_winrate * 100,
                    "Paper Win Rate (%)": paper_winrate * 100,
                    "Win Rate Diff (%)": winrate_diff * 100
                })
            except Exception as e:
                print(f"Error comparing {strategy}: {e}")
        
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Store comparison metrics
        self.comparison_metrics = comparison_df
        
        return comparison_df
    
    def plot_returns_comparison(self, save_path: Optional[str] = None):
        """
        Plot a comparison of returns between backtesting and paper trading.
        
        Args:
            save_path: Path to save the plot (if None, use default path)
        """
        if not isinstance(self.comparison_metrics, pd.DataFrame) or self.comparison_metrics.empty:
            print("No comparison metrics available. Run compare_strategies() first.")
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Extract data
        strategies = self.comparison_metrics["Strategy"]
        backtest_returns = self.comparison_metrics["Backtest Returns (%)"]
        paper_returns = self.comparison_metrics["Paper Returns (%)"]
        
        # Create bar positions
        x = np.arange(len(strategies))
        width = 0.35
        
        # Create bars
        plt.bar(x - width/2, backtest_returns, width, label="Backtest")
        plt.bar(x + width/2, paper_returns, width, label="Paper Trading")
        
        # Add labels and title
        plt.xlabel("Strategy")
        plt.ylabel("Returns (%)")
        plt.title("Returns Comparison: Backtesting vs. Paper Trading")
        plt.xticks(x, strategies, rotation=45)
        plt.legend()
        
        # Add grid
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate(backtest_returns):
            plt.text(i - width/2, v + 1, f"{v:.1f}%", ha="center")
            
        for i, v in enumerate(paper_returns):
            plt.text(i + width/2, v + 1, f"{v:.1f}%", ha="center")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
        else:
            # Use default path
            save_path = os.path.join(self.output_dir, "returns_comparison.png")
            plt.savefig(save_path)
            
        plt.close()
        
    def plot_metrics_radar(self, strategy_name: str, save_path: Optional[str] = None):
        """
        Plot a radar chart of performance metrics for a strategy.
        
        Args:
            strategy_name: Name of the strategy to plot
            save_path: Path to save the plot (if None, use default path)
        """
        if not isinstance(self.comparison_metrics, pd.DataFrame) or self.comparison_metrics.empty:
            print("No comparison metrics available. Run compare_strategies() first.")
            return
        
        # Find strategy in metrics
        strategy_metrics = self.comparison_metrics[
            self.comparison_metrics["Strategy"] == strategy_name
        ]
        
        if strategy_metrics.empty:
            print(f"Strategy {strategy_name} not found in comparison metrics.")
            return
        
        # Extract metrics (normalize to 0-1 range for radar chart)
        metrics = [
            "Returns",
            "Sharpe Ratio",
            "Win Rate",
            "Low Drawdown"  # Invert drawdown so higher is better
        ]
        
        # Get values (first row since we filtered for one strategy)
        backtest_values = [
            strategy_metrics["Backtest Returns (%)"].values[0] / 100,  # Convert % to ratio
            strategy_metrics["Backtest Sharpe"].values[0] / 3,  # Normalize Sharpe
            strategy_metrics["Backtest Win Rate (%)"].values[0] / 100,  # Convert % to ratio
            1 - (strategy_metrics["Backtest Drawdown (%)"].values[0] / 100)  # Invert drawdown
        ]
        
        paper_values = [
            strategy_metrics["Paper Returns (%)"].values[0] / 100,
            strategy_metrics["Paper Sharpe"].values[0] / 3,
            strategy_metrics["Paper Win Rate (%)"].values[0] / 100,
            1 - (strategy_metrics["Paper Drawdown (%)"].values[0] / 100)
        ]
        
        # Create radar chart
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Set number of angles (metrics)
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Add values for plotting
        backtest_values += backtest_values[:1]
        paper_values += paper_values[:1]
        metrics += metrics[:1]  # Add the first metric again to close the loop
        
        # Plot data
        ax.plot(angles, backtest_values, 'b-', linewidth=2, label="Backtest")
        ax.fill(angles, backtest_values, 'b', alpha=0.1)
        
        ax.plot(angles, paper_values, 'r-', linewidth=2, label="Paper Trading")
        ax.fill(angles, paper_values, 'r', alpha=0.1)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics[:-1])
        
        # Set title
        plt.title(f"Performance Metrics: {strategy_name}")
        
        # Add legend
        plt.legend(loc="upper right")
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
        else:
            # Use default path
            save_path = os.path.join(self.output_dir, f"radar_{strategy_name}.png")
            plt.savefig(save_path)
            
        plt.close()
    
    def generate_evaluation_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report (if None, use default path)
            
        Returns:
            Path to the generated report
        """
        if not isinstance(self.comparison_metrics, pd.DataFrame) or self.comparison_metrics.empty:
            print("No comparison metrics available. Run compare_strategies() first.")
            return ""
        
        # Determine output path
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"evaluation_report_{timestamp}.html")
        
        # Generate report content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Strategy Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .chart-container {{ margin: 20px 0; text-align: center; }}
                img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <h1>Strategy Evaluation Report</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Performance Comparison</h2>
            <table>
                <tr>
                    <th>Strategy</th>
                    <th>Backtest Returns (%)</th>
                    <th>Paper Returns (%)</th>
                    <th>Difference (%)</th>
                    <th>Backtest Sharpe</th>
                    <th>Paper Sharpe</th>
                    <th>Backtest Win Rate (%)</th>
                    <th>Paper Win Rate (%)</th>
                </tr>
        """
        
        # Add rows for each strategy
        for _, row in self.comparison_metrics.iterrows():
            returns_diff_class = "positive" if row["Returns Diff (%)"] >= 0 else "negative"
            html_content += f"""
                <tr>
                    <td>{row["Strategy"]}</td>
                    <td>{row["Backtest Returns (%)"]:.2f}%</td>
                    <td>{row["Paper Returns (%)"]:.2f}%</td>
                    <td class="{returns_diff_class}">{row["Returns Diff (%)"]:.2f}%</td>
                    <td>{row["Backtest Sharpe"]:.2f}</td>
                    <td>{row["Paper Sharpe"]:.2f}</td>
                    <td>{row["Backtest Win Rate (%)"]:.2f}%</td>
                    <td>{row["Paper Win Rate (%)"]:.2f}%</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Charts</h2>
        """
        
        # Generate and add returns comparison chart
        returns_chart_path = os.path.join(self.output_dir, "returns_comparison.png")
        self.plot_returns_comparison(returns_chart_path)
        html_content += f"""
            <div class="chart-container">
                <h3>Returns Comparison</h3>
                <img src="{os.path.relpath(returns_chart_path, os.path.dirname(output_path))}" alt="Returns Comparison">
            </div>
        """
        
        # Generate and add radar charts for each strategy
        for strategy in self.comparison_metrics["Strategy"]:
            radar_chart_path = os.path.join(self.output_dir, f"radar_{strategy}.png")
            self.plot_metrics_radar(strategy, radar_chart_path)
            html_content += f"""
                <div class="chart-container">
                    <h3>Performance Metrics: {strategy}</h3>
                    <img src="{os.path.relpath(radar_chart_path, os.path.dirname(output_path))}" alt="Performance Metrics: {strategy}">
                </div>
            """
        
        # Add summary and recommendations
        html_content += """
            <h2>Summary and Recommendations</h2>
            <p>
                The above comparison provides insights into how strategies perform in backtesting versus
                paper trading environments. Significant differences between backtest and paper trading
                results can indicate:
            </p>
            <ul>
                <li>Market impact effects not captured in backtesting</li>
                <li>Execution delays in live trading</li>
                <li>Data quality differences between historical and live data</li>
                <li>Overfitting in strategy development</li>
            </ul>
            <p>
                Strategies with the smallest performance gap between backtesting and paper trading
                should be considered more reliable for potential live trading.
            </p>
        """
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, "w") as f:
            f.write(html_content)
        
        print(f"Evaluation report generated at: {output_path}")
        return output_path 