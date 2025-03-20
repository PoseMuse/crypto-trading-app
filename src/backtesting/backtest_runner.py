"""
Backtest Runner module for cryptocurrency trading strategies.

This module provides functionality to run and analyze backtests for various
trading strategies, evaluate performance, and generate reports.
"""

import os
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import empyrical
import pyfolio as pf
from pathlib import Path

# Import our custom data loaders and strategies
from .data_loader import fetch_and_prepare_data, load_backtest_data
from .strategies import (
    BasicSMAStrategy,
    SentimentStrategy,
    AIModelStrategy,
    CombinedStrategy,
)


class BacktestRunner:
    """
    Class for running and analyzing backtests for cryptocurrency trading strategies.
    """
    
    def __init__(
        self,
        symbol: str = "BTC/USDT",
        cash: float = 10000.0,
        timeframe: str = "1d",
        commission: float = 0.001,  # 0.1% commission
        output_dir: str = "output/backtests",
        data_dir: str = "data/backtest"
    ):
        """
        Initialize the backtest runner.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            cash: Initial cash for backtesting
            timeframe: Data timeframe ('1d', '1h', etc.)
            commission: Commission rate for trades
            output_dir: Directory to save backtest results
            data_dir: Directory for data files
        """
        self.symbol = symbol
        self.cash = cash
        self.timeframe = timeframe
        self.commission = commission
        self.output_dir = output_dir
        self.data_dir = data_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Store backtest results
        self.results = {}
        
        # Store performance metrics
        self.metrics = {}
    
    def prepare_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_sentiment: bool = True,
        sentiment_lookback: int = 7,
        use_cached: bool = True
    ) -> str:
        """
        Prepare data for backtesting.
        
        Args:
            start_date: Start date for the backtest
            end_date: End date for the backtest
            include_sentiment: Whether to include sentiment data
            sentiment_lookback: Days to apply sentiment data retrospectively
            use_cached: Whether to use cached data if available
            
        Returns:
            Path to the prepared CSV file
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)  # 1 year of data
        
        # Fetch and prepare data
        filepath = fetch_and_prepare_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=start_date,
            end_date=end_date,
            include_sentiment=include_sentiment,
            sentiment_lookback=sentiment_lookback,
            output_dir=self.data_dir,
            use_cached=use_cached
        )
        
        return filepath
    
    def run_backtest(
        self,
        strategy: bt.Strategy,
        strategy_name: str,
        data_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        strategy_params: Optional[Dict] = None,
        plot_results: bool = True
    ) -> Dict:
        """
        Run a backtest for a given strategy.
        
        Args:
            strategy: Strategy class to use
            strategy_name: Name for this strategy run
            data_path: Path to the data file
            start_date: Start date for the backtest
            end_date: End date for the backtest
            strategy_params: Parameters to pass to the strategy
            plot_results: Whether to generate plots
            
        Returns:
            Dictionary with backtest results
        """
        # Create a cerebro engine
        cerebro = bt.Cerebro()
        
        # Add data feed
        data = load_backtest_data(data_path, start_date, end_date)
        cerebro.adddata(data)
        
        # Add strategy
        if strategy_params:
            cerebro.addstrategy(strategy, **strategy_params)
        else:
            cerebro.addstrategy(strategy)
        
        # Set our desired cash start
        cerebro.broker.setcash(self.cash)
        
        # Set commission
        cerebro.broker.setcommission(commission=self.commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        
        # Add observers
        cerebro.addobserver(bt.observers.Value)
        cerebro.addobserver(bt.observers.Trades)
        cerebro.addobserver(bt.observers.BuySell)
        
        # Run the backtest
        print(f"\nRunning backtest for {strategy_name}...")
        results = cerebro.run()
        strat = results[0]
        
        # Extract results
        portfolio_value = cerebro.broker.getvalue()
        sharpe = strat.analyzers.sharpe.get_analysis()['sharperatio']
        returns = strat.analyzers.returns.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        sqn = strat.analyzers.sqn.get_analysis()
        
        # Print results
        print(f"Final Portfolio Value: ${portfolio_value:.2f}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {drawdown.max.drawdown:.2%}")
        print(f"System Quality Number (SQN): {sqn.sqn:.2f}")
        
        # Calculate additional metrics
        total_return = (portfolio_value / self.cash) - 1
        print(f"Total Return: {total_return:.2%}")
        
        # Plot if requested
        if plot_results:
            self._generate_plots(cerebro, strategy_name)
        
        # Compute comprehensive metrics
        metrics = self._compute_metrics(strat, strategy_name)
        
        # Save results
        self.results[strategy_name] = strat
        self.metrics[strategy_name] = metrics
        
        # Return summary
        return {
            'strategy_name': strategy_name,
            'final_value': portfolio_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': drawdown.max.drawdown,
            'sqn': sqn.sqn,
            'n_trades': trades.total.total if hasattr(trades, 'total') else 0,
            'win_rate': trades.won.total / trades.total.total if hasattr(trades, 'won') and hasattr(trades, 'total') else 0,
            'metrics': metrics
        }
    
    def _generate_plots(self, cerebro: bt.Cerebro, strategy_name: str) -> None:
        """
        Generate and save plots for the backtest.
        
        Args:
            cerebro: Cerebro instance with completed backtest
            strategy_name: Name of the strategy
        """
        # Create a directory for this strategy
        strategy_dir = os.path.join(self.output_dir, strategy_name.replace(" ", "_"))
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Plot main results
        fig = cerebro.plot(style='candlestick', barup='green', bardown='red', 
                          volup='green', voldown='red', 
                          fill_lt=True, tickrotation=30)[0][0]
        
        # Adjust figure size
        fig.set_size_inches(12, 8)
        
        # Save figure
        plt.savefig(os.path.join(strategy_dir, f"{strategy_name}_backtest.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _compute_metrics(self, strat: bt.Strategy, strategy_name: str) -> Dict:
        """
        Compute comprehensive performance metrics.
        
        Args:
            strat: Completed strategy instance
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary of performance metrics
        """
        # Extract returns
        analyzer = strat.analyzers.returns
        returns = pd.Series(analyzer.get_analysis())
        
        # Convert to daily returns if not already
        if not returns.empty:
            daily_returns = returns.get('daily_returns', returns.get('returns', None))
            
            if daily_returns is not None:
                metrics = {
                    'annual_return': empyrical.annual_return(daily_returns),
                    'annual_volatility': empyrical.annual_volatility(daily_returns),
                    'sharpe_ratio': empyrical.sharpe_ratio(daily_returns),
                    'calmar_ratio': empyrical.calmar_ratio(daily_returns),
                    'stability': empyrical.stability_of_timeseries(daily_returns),
                    'max_drawdown': empyrical.max_drawdown(daily_returns),
                    'omega_ratio': empyrical.omega_ratio(daily_returns),
                    'sortino_ratio': empyrical.sortino_ratio(daily_returns),
                    'tail_ratio': empyrical.tail_ratio(daily_returns),
                    'value_at_risk': empyrical.value_at_risk(daily_returns)
                }
                
                return metrics
        
        # If metrics calculation fails, return basic metrics
        return {
            'annual_return': None,
            'annual_volatility': None,
            'sharpe_ratio': strat.analyzers.sharpe.get_analysis()['sharperatio'],
            'max_drawdown': strat.analyzers.drawdown.get_analysis().max.drawdown
        }
    
    def compare_strategies(self, save_to_csv: bool = True) -> pd.DataFrame:
        """
        Compare performance metrics across different strategies.
        
        Args:
            save_to_csv: Whether to save comparison to CSV
            
        Returns:
            DataFrame with strategy comparisons
        """
        # Create a comparison dataframe
        metrics_list = []
        
        for strategy_name, metrics in self.metrics.items():
            # Get base metrics
            base_metrics = {
                'strategy': strategy_name,
                'final_value': None,
                'total_return': None,
                'sharpe_ratio': metrics.get('sharpe_ratio'),
                'max_drawdown': metrics.get('max_drawdown'),
                'annual_return': metrics.get('annual_return'),
                'annual_volatility': metrics.get('annual_volatility'),
                'sortino_ratio': metrics.get('sortino_ratio'),
                'calmar_ratio': metrics.get('calmar_ratio')
            }
            
            metrics_list.append(base_metrics)
        
        # Create DataFrame
        comparison_df = pd.DataFrame(metrics_list)
        
        # Save to CSV if requested
        if save_to_csv:
            comparison_path = os.path.join(self.output_dir, "strategy_comparison.csv")
            comparison_df.to_csv(comparison_path, index=False)
            print(f"Strategy comparison saved to {comparison_path}")
        
        return comparison_df
    
    def run_all_strategies(
        self,
        data_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_basic: bool = True,
        include_sentiment: bool = True,
        include_ai: bool = True,
        include_combined: bool = True,
        custom_strategies: Optional[List[Tuple[bt.Strategy, str, Dict]]] = None,
        plot_results: bool = True
    ) -> Dict:
        """
        Run backtests for multiple strategies.
        
        Args:
            data_path: Path to the data file
            start_date: Start date for the backtest
            end_date: End date for the backtest
            include_basic: Whether to include the BasicSMAStrategy
            include_sentiment: Whether to include the SentimentStrategy
            include_ai: Whether to include the AIModelStrategy
            include_combined: Whether to include the CombinedStrategy
            custom_strategies: List of custom strategies to include, each as (strategy_class, name, params)
            plot_results: Whether to generate plots
            
        Returns:
            Dictionary with all backtest results
        """
        results = {}
        
        # Run basic strategy
        if include_basic:
            basic_result = self.run_backtest(
                strategy=BasicSMAStrategy,
                strategy_name="Basic SMA Crossover",
                data_path=data_path,
                start_date=start_date,
                end_date=end_date,
                plot_results=plot_results
            )
            results["Basic SMA Crossover"] = basic_result
        
        # Run sentiment strategy
        if include_sentiment:
            sentiment_result = self.run_backtest(
                strategy=SentimentStrategy,
                strategy_name="Sentiment Strategy",
                data_path=data_path,
                start_date=start_date,
                end_date=end_date,
                plot_results=plot_results
            )
            results["Sentiment Strategy"] = sentiment_result
        
        # Run AI model strategy
        if include_ai:
            ai_result = self.run_backtest(
                strategy=AIModelStrategy,
                strategy_name="AI Model Strategy",
                data_path=data_path,
                start_date=start_date,
                end_date=end_date,
                strategy_params={'symbol': self.symbol},
                plot_results=plot_results
            )
            results["AI Model Strategy"] = ai_result
        
        # Run combined strategy
        if include_combined:
            combined_result = self.run_backtest(
                strategy=CombinedStrategy,
                strategy_name="Combined Strategy",
                data_path=data_path,
                start_date=start_date,
                end_date=end_date,
                strategy_params={'symbol': self.symbol},
                plot_results=plot_results
            )
            results["Combined Strategy"] = combined_result
        
        # Run custom strategies
        if custom_strategies:
            for strategy_class, strategy_name, strategy_params in custom_strategies:
                custom_result = self.run_backtest(
                    strategy=strategy_class,
                    strategy_name=strategy_name,
                    data_path=data_path,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_params=strategy_params,
                    plot_results=plot_results
                )
                results[strategy_name] = custom_result
        
        # Compare strategies
        self.compare_strategies()
        
        return results


def main():
    """Run a sample backtest with multiple strategies."""
    # Initialize the backtest runner
    print("Initializing backtest runner...")
    runner = BacktestRunner(
        symbol="BTC/USDT",
        cash=10000.0,
        timeframe="1d",
        commission=0.001
    )
    
    # Prepare data
    print("Preparing data...")
    data_path = runner.prepare_data(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2022, 12, 31),
        include_sentiment=True,
        sentiment_lookback=7
    )
    
    # Run all strategies
    print("Running backtests for all strategies...")
    results = runner.run_all_strategies(
        data_path=data_path,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2022, 12, 31)
    )
    
    # Print final comparison
    print("\nStrategy Comparison:")
    comparison_df = runner.compare_strategies()
    print(comparison_df)
    
    return comparison_df


if __name__ == "__main__":
    main() 