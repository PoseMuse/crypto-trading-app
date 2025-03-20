"""
Risk-Managed Strategy that extends CryptoStrategy.

This module provides a trading strategy that incorporates risk management
rules from the risk management package.
"""

import backtrader as bt
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from .backtesting_pipeline import CryptoStrategy
from ..risk_management.position_sizer import PositionSizer
from ..risk_management.circuit_breakers import CircuitBreaker


class RiskManagedStrategy(CryptoStrategy):
    """
    Risk-managed strategy that incorporates advanced risk management rules.
    
    This strategy extends CryptoStrategy with:
    - Position sizing based on the 1% rule
    - Circuit breakers to protect capital during drawdowns
    - Safe mode behavior during periods of poor performance
    - Trade journaling
    """
    
    params = (
        # Default parameters from CryptoStrategy
        ('ai_threshold', 0.6),        # Threshold for AI signals
        ('sentiment_threshold', 0.3),  # Threshold for sentiment signals
        ('ai_weight', 0.7),            # Weight for AI signals vs sentiment
        ('stop_loss', 0.05),           # Stop loss percentage
        ('take_profit', 0.15),         # Take profit percentage
        ('position_size', 0.95),       # Position size as percentage of portfolio
        
        # Risk management parameters
        ('max_risk_pct', 0.01),        # Maximum risk per trade (1% rule)
        ('safe_mode_risk_pct', 0.005), # Maximum risk in safe mode (0.5%)
        ('daily_drawdown_limit', 0.05), # 5% daily drawdown limit
        ('weekly_drawdown_limit', 0.10), # 10% weekly drawdown limit
        ('monthly_drawdown_limit', 0.20), # 20% monthly drawdown limit
        ('consecutive_loss_limit', 3),   # Max consecutive losses
        ('safe_mode_duration', 1),       # Days to stay in safe mode
        ('max_leverage', 1.0),           # Max leverage (1.0 = no leverage)
    )
    
    def __init__(self):
        """Initialize the risk-managed strategy."""
        # Call parent's init first
        super(RiskManagedStrategy, self).__init__()
        
        # Initialize risk management components
        self.position_sizer = PositionSizer(
            max_risk_pct=self.params.max_risk_pct,
            safe_mode_risk_pct=self.params.safe_mode_risk_pct,
            max_position_pct=self.params.position_size,
            max_leverage=self.params.max_leverage
        )
        
        self.circuit_breaker = CircuitBreaker(
            daily_drawdown_limit=self.params.daily_drawdown_limit,
            weekly_drawdown_limit=self.params.weekly_drawdown_limit,
            monthly_drawdown_limit=self.params.monthly_drawdown_limit,
            consecutive_loss_limit=self.params.consecutive_loss_limit,
            safe_mode_duration=self.params.safe_mode_duration
        )
        
        # Initialize trade tracking
        self.equity_history = []
        self.drawdown_history = []
        self.safe_mode_history = []
        
        self.equity_high_watermark = 0
        self.current_drawdown = 0
        self.consecutive_losses = 0
        self.safe_mode = False
        
        # Log initialization with risk parameters
        self.log(f"Risk-Managed Strategy initialized with max risk: {self.params.max_risk_pct:.2%}, "
                f"daily drawdown limit: {self.params.daily_drawdown_limit:.2%}")
    
    def start(self):
        """Called when strategy starts."""
        super(RiskManagedStrategy, self).start()
        
        # Initialize circuit breaker with starting equity
        initial_equity = self.broker.getvalue()
        self.circuit_breaker.initialize(initial_equity)
        self.equity_high_watermark = initial_equity
        
        # Record initial equity
        self._record_equity()
    
    def next(self):
        """
        Core strategy logic executed for each candle.
        """
        # Update equity and risk metrics
        self._record_equity()
        
        # Update circuit breaker with current equity and check for triggers
        current_equity = self.broker.getvalue()
        unrealized_pnl = self._get_unrealized_pnl()
        circuit_status = self.circuit_breaker.update_equity(current_equity, unrealized_pnl)
        
        # Check if trading should be halted
        if circuit_status["halt_trading"]:
            self.log(f"TRADING HALTED - Circuit breaker triggered - "
                    f"Daily drawdown: {circuit_status['daily_drawdown']:.2%}")
            
            # Exit any existing positions if needed
            if circuit_status.get("exit_positions", False) and self.position:
                self.log("EMERGENCY EXIT - Exiting all positions")
                self.order = self.close()
            
            return
        
        # Update safe mode status
        self.safe_mode = circuit_status["safe_mode_active"]
        
        # Call the parent's next method if we're in a position
        # For position management (stop loss, take profit)
        if self.position:
            # Check if we should exit based on circuit breakers
            should_exit, reason = self.circuit_breaker.should_exit_position(
                unrealized_pnl=unrealized_pnl,
                entry_price=self.price_entry if self.price_entry else self.position.price,
                current_price=self.data.close[0]
            )
            
            if should_exit:
                self.log(f"RISK EXIT - {reason}")
                self.order = self.close()
                return
            
            # Default position management (stop loss, take profit)
            if self.stop_loss_price is not None and self.data.close[0] < self.stop_loss_price:
                self.log(f"STOP LOSS TRIGGERED - Entry: {self.price_entry:.2f}, Current: {self.data.close[0]:.2f}")
                self.order = self.sell(size=self.position.size)
                return
            
            if self.take_profit_price is not None and self.data.close[0] > self.take_profit_price:
                self.log(f"TAKE PROFIT TRIGGERED - Entry: {self.price_entry:.2f}, Current: {self.data.close[0]:.2f}")
                self.order = self.sell(size=self.position.size)
                return
            
            # Signal-based exit
            combined_signal = self.get_combined_signal()
            if combined_signal < -self.params.ai_threshold:
                self.log(f"SELL SIGNAL - Combined: {combined_signal:.2f}, AI: {self.ai_signal[0]:.2f}, Sentiment: {self.sentiment[0]:.2f}")
                self.order = self.sell(size=self.position.size)
                return
        
        # Skip if an order is pending
        if self.order:
            return
        
        # Process new trade entry if we're not in a position
        if not self.position:
            # Check if we should enter
            combined_signal = self.get_combined_signal()
            
            if combined_signal > self.params.ai_threshold:
                # Calculate entry price
                entry_price = self.data.close[0]
                
                # Calculate stop loss price
                stop_loss_price = entry_price * (1.0 - self.params.stop_loss)
                
                # Calculate risk-managed position size
                position_size, details = self.position_sizer.calculate_position_size(
                    capital=self.broker.getvalue(),
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    safe_mode=self.safe_mode
                )
                
                # Submit buy order if position size is valid
                if position_size > 0:
                    self.log(f"BUY SIGNAL - Combined: {combined_signal:.2f}, "
                            f"Size: {position_size:.4f}, Risk: {details['capital_at_risk_pct']:.2%}"
                            f"{' (SAFE MODE)' if self.safe_mode else ''}")
                    
                    self.order = self.buy(size=position_size)
                    
                    # Record stop loss and take profit prices
                    self.price_entry = entry_price
                    self.stop_loss_price = stop_loss_price
                    self.take_profit_price = entry_price * (1.0 + self.params.take_profit)
                else:
                    self.log(f"BUY SIGNAL IGNORED - Combined: {combined_signal:.2f} - "
                            f"Zero position size due to risk limits")
    
    def notify_trade(self, trade):
        """Called on each trade completion."""
        super(RiskManagedStrategy, self).notify_trade(trade)
        
        if trade.isclosed:
            # Record the trade result to circuit breaker for consecutive loss tracking
            self.circuit_breaker.record_trade_result(
                trade_result=trade.pnlcomm,
                trade_info={
                    "entry_price": trade.price,
                    "exit_price": trade.price + trade.pnl / trade.size,
                    "size": trade.size
                }
            )
            
            # Update our own consecutive loss counter
            if trade.pnlcomm > 0:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
    
    def _record_equity(self):
        """Record current equity and calculate drawdown."""
        current_equity = self.broker.getvalue()
        current_time = self.data.datetime.datetime()
        
        # Update high watermark if needed
        if current_equity > self.equity_high_watermark:
            self.equity_high_watermark = current_equity
        
        # Calculate current drawdown
        if self.equity_high_watermark > 0:
            self.current_drawdown = 1 - (current_equity / self.equity_high_watermark)
        
        # Record equity history
        self.equity_history.append((current_time, current_equity))
        
        # Record drawdown history
        self.drawdown_history.append((current_time, self.current_drawdown))
        
        # Record safe mode history
        self.safe_mode_history.append((current_time, self.safe_mode))
    
    def _get_unrealized_pnl(self) -> float:
        """Calculate unrealized PnL of current position."""
        if not self.position:
            return 0.0
        
        current_value = self.position.size * self.data.close[0]
        cost_basis = self.position.price * self.position.size
        
        return current_value - cost_basis
    
    def stop(self):
        """Called when the strategy is stopped."""
        super(RiskManagedStrategy, self).stop()
        
        # Calculate final statistics
        final_equity = self.broker.getvalue()
        initial_equity = self.equity_history[0][1] if self.equity_history else 0
        total_return_pct = ((final_equity / initial_equity) - 1) * 100 if initial_equity > 0 else 0
        
        # Calculate max drawdown from history
        max_drawdown = max([dd[1] for dd in self.drawdown_history]) if self.drawdown_history else 0
        
        # Log final results with risk metrics
        self.log(f"Risk-Managed Strategy finished - "
                f"Return: {total_return_pct:.2f}%, Max Drawdown: {max_drawdown:.2%}, "
                f"Safe Mode Activations: {sum(1 for sm in self.safe_mode_history if sm[1])}")
        
        # TODO: Save detailed risk metrics to a file for analysis 