"""
Circuit Breakers Module for Risk Management.

This module implements circuit breakers to halt trading during excessive drawdowns,
track consecutive losses, and manage safe mode operations.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np


class CircuitBreaker:
    """
    Circuit breaker for managing trading limits and drawdowns.
    
    Primary functionality:
    - Monitors daily/weekly drawdowns and halts trading if thresholds are exceeded
    - Tracks consecutive losses and activates safe mode if needed
    - Provides mechanisms to exit positions during emergency situations
    - Monitors unrealized PnL in real-time
    """
    
    def __init__(
        self,
        daily_drawdown_limit: float = 0.05,  # 5% max daily drawdown
        weekly_drawdown_limit: float = 0.10,  # 10% max weekly drawdown
        monthly_drawdown_limit: float = 0.20,  # 20% max monthly drawdown
        unrealized_loss_limit: float = 0.10,  # 10% unrealized loss limit
        consecutive_loss_limit: int = 3,  # Max number of consecutive losses
        safe_mode_duration: int = 1,  # Days to remain in safe mode after trigger
    ):
        """
        Initialize the circuit breaker with risk parameters.
        
        Args:
            daily_drawdown_limit: Max allowed daily drawdown (e.g., 0.05 for 5%)
            weekly_drawdown_limit: Max allowed weekly drawdown
            monthly_drawdown_limit: Max allowed monthly drawdown
            unrealized_loss_limit: Max allowed unrealized loss before closing
            consecutive_loss_limit: Number of consecutive losses to trigger safe mode
            safe_mode_duration: Days to remain in safe mode after trigger
        """
        self.daily_drawdown_limit = daily_drawdown_limit
        self.weekly_drawdown_limit = weekly_drawdown_limit
        self.monthly_drawdown_limit = monthly_drawdown_limit
        self.unrealized_loss_limit = unrealized_loss_limit
        self.consecutive_loss_limit = consecutive_loss_limit
        self.safe_mode_duration = safe_mode_duration
        
        # Initialize tracking variables
        self.safe_mode_active = False
        self.safe_mode_end_date = None
        self.trades_history = []
        self.consecutive_losses = 0
        self.daily_high_equity = None
        self.weekly_high_equity = None
        self.monthly_high_equity = None
        self.current_unrealized_pnl = 0.0
        self.current_equity = 0.0
        self.initial_equity = 0.0
        self.last_update_time = None
        
        # Initialize drawdown trackers
        self.daily_drawdowns = []
        self.weekly_drawdowns = []
        self.monthly_drawdowns = []
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def initialize(self, initial_equity: float) -> None:
        """
        Initialize the circuit breaker with starting equity.
        
        Args:
            initial_equity: Starting equity value
        """
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.daily_high_equity = initial_equity
        self.weekly_high_equity = initial_equity
        self.monthly_high_equity = initial_equity
        self.last_update_time = datetime.now()
        
        self.logger.info(f"Circuit breaker initialized with equity: ${initial_equity:.2f}")
    
    def update_equity(self, current_equity: float, unrealized_pnl: float = 0.0) -> Dict[str, Any]:
        """
        Update current equity and check for circuit breaker triggers.
        
        Args:
            current_equity: Current equity value
            unrealized_pnl: Current unrealized profit/loss
            
        Returns:
            Dictionary with circuit breaker status
        """
        # Store previous values
        previous_equity = self.current_equity
        
        # Update values
        self.current_equity = current_equity
        self.current_unrealized_pnl = unrealized_pnl
        current_time = datetime.now()
        
        # Handle time period transitions
        self._check_time_period_reset(current_time)
        
        # Update high water marks
        if current_equity > self.daily_high_equity:
            self.daily_high_equity = current_equity
        if current_equity > self.weekly_high_equity:
            self.weekly_high_equity = current_equity
        if current_equity > self.monthly_high_equity:
            self.monthly_high_equity = current_equity
        
        # Calculate current drawdowns
        daily_drawdown = 1 - (current_equity / self.daily_high_equity)
        weekly_drawdown = 1 - (current_equity / self.weekly_high_equity)
        monthly_drawdown = 1 - (current_equity / self.monthly_high_equity)
        unrealized_drawdown = -unrealized_pnl / self.current_equity if self.current_equity > 0 else 0
        
        # Track drawdowns
        self.daily_drawdowns.append((current_time, daily_drawdown))
        self.weekly_drawdowns.append((current_time, weekly_drawdown))
        self.monthly_drawdowns.append((current_time, monthly_drawdown))
        
        # Check if we need to exit safe mode
        if self.safe_mode_active and self.safe_mode_end_date and current_time > self.safe_mode_end_date:
            self.logger.info("Safe mode duration completed. Returning to normal trading.")
            self.safe_mode_active = False
            self.safe_mode_end_date = None
        
        # Check for circuit breaker triggers
        triggers = {
            "daily_drawdown_triggered": daily_drawdown >= self.daily_drawdown_limit,
            "weekly_drawdown_triggered": weekly_drawdown >= self.weekly_drawdown_limit,
            "monthly_drawdown_triggered": monthly_drawdown >= self.monthly_drawdown_limit,
            "unrealized_loss_triggered": unrealized_drawdown >= self.unrealized_loss_limit,
            "consecutive_losses_triggered": self.consecutive_losses >= self.consecutive_loss_limit,
            "safe_mode_active": self.safe_mode_active,
            "current_equity": current_equity,
            "unrealized_pnl": unrealized_pnl,
            "daily_drawdown": daily_drawdown,
            "weekly_drawdown": weekly_drawdown,
            "monthly_drawdown": monthly_drawdown,
            "unrealized_drawdown": unrealized_drawdown,
            "consecutive_losses": self.consecutive_losses,
            "halt_trading": False  # Default
        }
        
        # Determine if we should halt trading
        if (triggers["daily_drawdown_triggered"] or 
            triggers["weekly_drawdown_triggered"] or 
            triggers["monthly_drawdown_triggered"]):
            
            # Log the circuit breaker activation
            self.logger.warning(f"CIRCUIT BREAKER TRIGGERED: Daily drawdown: {daily_drawdown:.2%}, "
                               f"Weekly: {weekly_drawdown:.2%}, Monthly: {monthly_drawdown:.2%}")
            
            # Activate safe mode
            self._activate_safe_mode()
            
            # Set halt trading flag
            triggers["halt_trading"] = True
        
        # Check for unrealized loss limit
        if triggers["unrealized_loss_triggered"]:
            self.logger.warning(f"UNREALIZED LOSS LIMIT TRIGGERED: {unrealized_drawdown:.2%} exceeds {self.unrealized_loss_limit:.2%}")
            triggers["halt_trading"] = True
            triggers["exit_positions"] = True
        
        # Check for consecutive losses safe mode
        if triggers["consecutive_losses_triggered"] and not self.safe_mode_active:
            self.logger.warning(f"CONSECUTIVE LOSSES TRIGGERED: {self.consecutive_losses} losses in a row")
            self._activate_safe_mode()
            triggers["safe_mode_active"] = True
        
        # Update last update time
        self.last_update_time = current_time
        
        return triggers
    
    def _check_time_period_reset(self, current_time: datetime) -> None:
        """
        Check if we need to reset high water marks for time periods.
        
        Args:
            current_time: Current time
        """
        if not self.last_update_time:
            return
            
        # Check if we crossed to a new day
        if current_time.date() > self.last_update_time.date():
            self.daily_high_equity = self.current_equity
            self.logger.info(f"New day started. Resetting daily high equity to ${self.current_equity:.2f}")
            
            # Trim drawdown history - keep only last 30 days
            if len(self.daily_drawdowns) > 30:
                self.daily_drawdowns = self.daily_drawdowns[-30:]
        
        # Check if we crossed to a new week
        if current_time.isocalendar()[1] != self.last_update_time.isocalendar()[1]:
            self.weekly_high_equity = self.current_equity
            self.logger.info(f"New week started. Resetting weekly high equity to ${self.current_equity:.2f}")
            
            # Trim drawdown history - keep only last 12 weeks
            if len(self.weekly_drawdowns) > 12:
                self.weekly_drawdowns = self.weekly_drawdowns[-12:]
        
        # Check if we crossed to a new month
        if current_time.month != self.last_update_time.month:
            self.monthly_high_equity = self.current_equity
            self.logger.info(f"New month started. Resetting monthly high equity to ${self.current_equity:.2f}")
            
            # Trim drawdown history - keep only last 12 months
            if len(self.monthly_drawdowns) > 12:
                self.monthly_drawdowns = self.monthly_drawdowns[-12:]
    
    def _activate_safe_mode(self) -> None:
        """Activate safe mode for the specified duration."""
        self.safe_mode_active = True
        self.safe_mode_end_date = datetime.now() + timedelta(days=self.safe_mode_duration)
        self.logger.warning(f"SAFE MODE ACTIVATED until {self.safe_mode_end_date.strftime('%Y-%m-%d')}")
    
    def record_trade_result(self, trade_result: float, trade_info: Optional[Dict] = None) -> None:
        """
        Record the result of a completed trade.
        
        Args:
            trade_result: Profit/loss from the trade
            trade_info: Additional trade information (optional)
        """
        trade_time = datetime.now()
        
        # Create trade record
        trade_record = {
            "time": trade_time,
            "result": trade_result,
            "profitable": trade_result > 0,
            **({} if trade_info is None else trade_info)
        }
        
        # Add to trade history
        self.trades_history.append(trade_record)
        
        # Update consecutive losses counter
        if trade_result > 0:
            # Reset counter on profitable trade
            self.consecutive_losses = 0
            self.logger.info(f"Profitable trade: ${trade_result:.2f}. Consecutive losses reset to 0.")
        else:
            # Increment counter on losing trade
            self.consecutive_losses += 1
            self.logger.warning(f"Losing trade: ${trade_result:.2f}. Consecutive losses: {self.consecutive_losses}")
            
            # Check if we need to activate safe mode
            if self.consecutive_losses >= self.consecutive_loss_limit and not self.safe_mode_active:
                self._activate_safe_mode()
    
    def should_exit_position(self, unrealized_pnl: float, entry_price: float, current_price: float) -> Tuple[bool, str]:
        """
        Determine if a position should be exited based on circuit breaker rules.
        
        Args:
            unrealized_pnl: Current unrealized profit/loss
            entry_price: Entry price of the position
            current_price: Current price
            
        Returns:
            Tuple of (exit_signal, reason)
        """
        # Calculate unrealized drawdown
        unrealized_drawdown = -unrealized_pnl / self.current_equity if self.current_equity > 0 else 0
        
        # Check if we hit the unrealized loss limit
        if unrealized_drawdown >= self.unrealized_loss_limit:
            return True, f"Unrealized loss limit hit: {unrealized_drawdown:.2%}"
        
        # Check daily drawdown limit
        daily_drawdown = 1 - (self.current_equity / self.daily_high_equity)
        if daily_drawdown >= self.daily_drawdown_limit:
            return True, f"Daily drawdown limit hit: {daily_drawdown:.2%}"
        
        # Not exiting based on circuit breakers
        return False, ""
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the circuit breaker.
        
        Returns:
            Dictionary with current status
        """
        current_time = datetime.now()
        
        # Calculate current drawdowns
        daily_drawdown = 1 - (self.current_equity / self.daily_high_equity) if self.daily_high_equity > 0 else 0
        weekly_drawdown = 1 - (self.current_equity / self.weekly_high_equity) if self.weekly_high_equity > 0 else 0
        monthly_drawdown = 1 - (self.current_equity / self.monthly_high_equity) if self.monthly_high_equity > 0 else 0
        
        # Create status dictionary
        status = {
            "time": current_time,
            "safe_mode_active": self.safe_mode_active,
            "safe_mode_end_date": self.safe_mode_end_date,
            "consecutive_losses": self.consecutive_losses,
            "initial_equity": self.initial_equity,
            "current_equity": self.current_equity,
            "daily_high_equity": self.daily_high_equity,
            "weekly_high_equity": self.weekly_high_equity,
            "monthly_high_equity": self.monthly_high_equity,
            "daily_drawdown": daily_drawdown,
            "weekly_drawdown": weekly_drawdown,
            "monthly_drawdown": monthly_drawdown,
            "daily_drawdown_limit": self.daily_drawdown_limit,
            "weekly_drawdown_limit": self.weekly_drawdown_limit,
            "monthly_drawdown_limit": self.monthly_drawdown_limit,
            "unrealized_loss_limit": self.unrealized_loss_limit,
            "consecutive_loss_limit": self.consecutive_loss_limit,
            "total_trades": len(self.trades_history),
            "winning_trades": sum(1 for trade in self.trades_history if trade["profitable"]),
            "losing_trades": sum(1 for trade in self.trades_history if not trade["profitable"]),
        }
        
        return status
    
    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        self.safe_mode_active = False
        self.safe_mode_end_date = None
        self.consecutive_losses = 0
        
        self.logger.info("Circuit breaker reset - safe mode deactivated, consecutive losses reset.")
    
    def get_drawdown_report(self) -> Dict[str, pd.DataFrame]:
        """
        Generate a report of drawdown statistics.
        
        Returns:
            Dictionary with drawdown DataFrames
        """
        # Convert drawdown lists to DataFrames
        daily_df = pd.DataFrame(self.daily_drawdowns, columns=['time', 'drawdown'])
        weekly_df = pd.DataFrame(self.weekly_drawdowns, columns=['time', 'drawdown'])
        monthly_df = pd.DataFrame(self.monthly_drawdowns, columns=['time', 'drawdown'])
        
        # Calculate statistics
        def calculate_stats(df):
            if df.empty:
                return pd.DataFrame()
            
            return pd.DataFrame({
                'max_drawdown': [df['drawdown'].max()],
                'avg_drawdown': [df['drawdown'].mean()],
                'current_drawdown': [df['drawdown'].iloc[-1] if not df.empty else 0],
                'num_observations': [len(df)]
            })
        
        daily_stats = calculate_stats(daily_df)
        weekly_stats = calculate_stats(weekly_df)
        monthly_stats = calculate_stats(monthly_df)
        
        return {
            'daily_drawdowns': daily_df,
            'weekly_drawdowns': weekly_df,
            'monthly_drawdowns': monthly_df,
            'daily_stats': daily_stats,
            'weekly_stats': weekly_stats,
            'monthly_stats': monthly_stats
        } 