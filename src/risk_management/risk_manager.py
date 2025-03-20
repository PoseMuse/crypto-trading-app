"""
Risk Manager Module.

This module integrates position sizing and circuit breakers into a unified
risk management system for cryptocurrency trading.
"""

import logging
import json
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

from .position_sizer import PositionSizer
from .circuit_breakers import CircuitBreaker


class RiskManager:
    """
    Risk manager that combines position sizing and circuit breakers.
    
    This class provides a unified interface for risk management, including:
    - Position sizing based on the 1% rule
    - Circuit breakers for drawdown limits
    - Safe mode operations
    - Trade journaling and analytics
    """
    
    def __init__(
        self,
        initial_equity: float,
        max_risk_per_trade: float = 0.01,  # 1% risk per trade
        max_leverage: float = 1.0,  # No leverage by default
        daily_drawdown_limit: float = 0.05,  # 5% max daily drawdown
        weekly_drawdown_limit: float = 0.10,  # 10% max weekly drawdown
        monthly_drawdown_limit: float = 0.20,  # 20% max monthly drawdown
        consecutive_loss_limit: int = 3,  # Consecutive loss limit
        auto_reduce_risk: bool = True,  # Reduce risk after drawdowns
        journal_path: str = "data/trade_journal.json"
    ):
        """
        Initialize the risk manager.
        
        Args:
            initial_equity: Initial equity amount
            max_risk_per_trade: Maximum risk per trade (as % of equity)
            max_leverage: Maximum allowed leverage
            daily_drawdown_limit: Maximum daily drawdown before halting
            weekly_drawdown_limit: Maximum weekly drawdown before halting
            monthly_drawdown_limit: Maximum monthly drawdown before halting
            consecutive_loss_limit: Number of consecutive losses before safe mode
            auto_reduce_risk: Whether to automatically reduce risk after drawdowns
            journal_path: Path to save the trade journal
        """
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.max_risk_per_trade = max_risk_per_trade
        self.max_leverage = max_leverage
        self.auto_reduce_risk = auto_reduce_risk
        self.journal_path = journal_path
        
        # Initialize position sizer
        self.position_sizer = PositionSizer(
            max_risk_pct=max_risk_per_trade,
            safe_mode_risk_pct=max_risk_per_trade / 2,  # Half risk in safe mode
            max_leverage=max_leverage
        )
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            daily_drawdown_limit=daily_drawdown_limit,
            weekly_drawdown_limit=weekly_drawdown_limit,
            monthly_drawdown_limit=monthly_drawdown_limit,
            consecutive_loss_limit=consecutive_loss_limit
        )
        self.circuit_breaker.initialize(initial_equity)
        
        # Initialize trade journal
        self.trade_journal = []
        self._load_trade_journal()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Risk Manager initialized with equity: ${initial_equity:.2f}")
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        symbol: str = "",
        direction: str = "long"
    ) -> Dict[str, Any]:
        """
        Calculate position size based on risk management rules.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price for the trade
            symbol: Trading symbol (for reference)
            direction: Trade direction ("long" or "short")
            
        Returns:
            Dictionary with position sizing information
        """
        # Check if trading is allowed
        circuit_status = self.get_circuit_breaker_status()
        if circuit_status["halt_trading"]:
            return {
                "position_size": 0.0,
                "allowed_to_trade": False,
                "reason": "Circuit breaker triggered - trading halted",
                "safe_mode": circuit_status["safe_mode_active"],
                "capital_at_risk": 0.0
            }
        
        # Determine if we're in safe mode
        safe_mode = circuit_status["safe_mode_active"]
        
        # Calculate volatility factor based on drawdowns
        volatility_factor = 1.0
        if self.auto_reduce_risk:
            # If in drawdown, reduce position size proportionally
            daily_drawdown = circuit_status["daily_drawdown"]
            drawdown_ratio = daily_drawdown / circuit_status["daily_drawdown_limit"]
            
            # Scale down as we approach limits
            if drawdown_ratio > 0.5:  # If we're at least 50% of the way to limit
                volatility_factor = 1.0 - (drawdown_ratio - 0.5)  # Linear scale down
                volatility_factor = max(0.3, volatility_factor)  # Don't go below 30%
        
        # Calculate position size
        position_size, details = self.position_sizer.calculate_position_size(
            capital=self.current_equity,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            safe_mode=safe_mode,
            volatility_factor=volatility_factor
        )
        
        # Create result dictionary
        result = {
            "position_size": position_size,
            "allowed_to_trade": position_size > 0,
            "safe_mode": safe_mode,
            "capital_at_risk": details["capital_at_risk_pct"],
            "volatility_adjustment": volatility_factor,
            "max_loss_amount": details["max_loss"],
            "direction": direction,
            "symbol": symbol,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "current_equity": self.current_equity
        }
        
        # Log position size calculation
        if position_size > 0:
            self.logger.info(
                f"Position size calculated: {position_size:.4f} {symbol} - Risk: {details['capital_at_risk_pct']:.2%}"
                f"{' (SAFE MODE)' if safe_mode else ''}"
            )
        else:
            self.logger.warning(f"Position size zero - not allowed to trade")
        
        return result
    
    def update_equity(self, current_equity: float, unrealized_pnl: float = 0.0) -> Dict[str, Any]:
        """
        Update current equity and check circuit breakers.
        
        Args:
            current_equity: Current equity amount
            unrealized_pnl: Current unrealized profit/loss
            
        Returns:
            Dictionary with status information
        """
        # Update current equity
        self.current_equity = current_equity
        
        # Update circuit breaker
        circuit_status = self.circuit_breaker.update_equity(current_equity, unrealized_pnl)
        
        # Return status
        return circuit_status
    
    def record_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        position_size: float,
        entry_time: datetime,
        exit_time: datetime,
        pnl: float,
        stop_loss: float,
        take_profit: Optional[float] = None,
        fees: float = 0.0,
        trade_id: Optional[str] = None,
        notes: Optional[str] = None
    ) -> None:
        """
        Record a completed trade in the journal.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ("long" or "short")
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size
            entry_time: Entry time
            exit_time: Exit time
            pnl: Profit/loss amount
            stop_loss: Stop loss price
            take_profit: Take profit price (optional)
            fees: Trading fees
            trade_id: Unique trade ID (optional)
            notes: Additional notes (optional)
        """
        # Calculate trade metrics
        pnl_pct = ((exit_price / entry_price) - 1) * 100 if direction == "long" else \
                  ((entry_price / exit_price) - 1) * 100
        risk_amount = abs(entry_price - stop_loss) * position_size
        risk_pct = risk_amount / self.current_equity
        
        # Calculate reward to risk ratio
        r_multiple = None
        if take_profit and stop_loss:
            potential_gain = abs(take_profit - entry_price) * position_size
            potential_loss = abs(stop_loss - entry_price) * position_size
            if potential_loss > 0:
                r_multiple = potential_gain / potential_loss
        
        # Create trade record
        trade_record = {
            "trade_id": trade_id or f"{symbol}-{entry_time.strftime('%Y%m%d%H%M%S')}",
            "symbol": symbol,
            "direction": direction,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "position_size": float(position_size),
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "duration": (exit_time - entry_time).total_seconds() / 60,  # minutes
            "pnl": float(pnl),
            "pnl_pct": float(pnl_pct),
            "fees": float(fees),
            "net_pnl": float(pnl - fees),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit) if take_profit else None,
            "risk_amount": float(risk_amount),
            "risk_pct": float(risk_pct),
            "r_multiple": float(r_multiple) if r_multiple else None,
            "equity_before": float(self.current_equity - pnl),
            "equity_after": float(self.current_equity),
            "notes": notes,
            "safe_mode": self.circuit_breaker.safe_mode_active
        }
        
        # Add to trade journal
        self.trade_journal.append(trade_record)
        
        # Record in circuit breaker
        self.circuit_breaker.record_trade_result(pnl, {
            "symbol": symbol,
            "direction": direction,
            "position_size": position_size
        })
        
        # Save to disk
        self._save_trade_journal()
        
        # Log trade
        if pnl > 0:
            self.logger.info(f"Trade recorded - {symbol} {direction}: ${pnl:.2f} profit")
        else:
            self.logger.warning(f"Trade recorded - {symbol} {direction}: ${-pnl:.2f} loss")
    
    def should_exit_position(
        self,
        unrealized_pnl: float,
        entry_price: float,
        current_price: float,
        position_size: float
    ) -> Tuple[bool, str]:
        """
        Check if a position should be exited based on risk management rules.
        
        Args:
            unrealized_pnl: Current unrealized profit/loss
            entry_price: Entry price of the position
            current_price: Current price
            position_size: Position size
            
        Returns:
            Tuple of (exit_signal, reason)
        """
        return self.circuit_breaker.should_exit_position(
            unrealized_pnl, entry_price, current_price
        )
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """
        Get the current status of the circuit breaker.
        
        Returns:
            Dictionary with circuit breaker status
        """
        return self.circuit_breaker.get_status()
    
    def get_trade_analytics(self) -> Dict[str, Any]:
        """
        Generate trade analytics from the journal.
        
        Returns:
            Dictionary with trade analytics
        """
        if not self.trade_journal:
            return {
                "total_trades": 0,
                "message": "No trades recorded yet"
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.trade_journal)
        
        # Basic analytics
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit metrics
        total_profit = df[df['pnl'] > 0]['pnl'].sum() if not df.empty else 0
        total_loss = df[df['pnl'] <= 0]['pnl'].sum() if not df.empty else 0
        net_profit = total_profit + total_loss
        
        # Calculate average values
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Calculate profit factor
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Get equity curve
        equity_series = df['equity_after'].tolist()
        
        # Calculate drawdown
        max_equity = 0
        max_drawdown = 0
        drawdown_series = []
        
        for equity in equity_series:
            max_equity = max(max_equity, equity)
            drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
            drawdown_series.append(drawdown)
        
        # Return analytics
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": float(total_profit),
            "total_loss": float(total_loss),
            "net_profit": float(net_profit),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "max_drawdown": float(max_drawdown),
            "current_equity": float(self.current_equity),
            "initial_equity": float(self.initial_equity),
            "overall_return_pct": (self.current_equity / self.initial_equity - 1) * 100,
            "risk_settings": {
                "max_risk_per_trade": self.max_risk_per_trade,
                "max_leverage": self.max_leverage,
                "daily_drawdown_limit": self.circuit_breaker.daily_drawdown_limit,
                "weekly_drawdown_limit": self.circuit_breaker.weekly_drawdown_limit,
                "consecutive_loss_limit": self.circuit_breaker.consecutive_loss_limit
            }
        }
    
    def _load_trade_journal(self) -> None:
        """Load the trade journal from disk if it exists."""
        if os.path.exists(self.journal_path):
            try:
                with open(self.journal_path, 'r') as f:
                    self.trade_journal = json.load(f)
                self.logger.info(f"Loaded {len(self.trade_journal)} trades from journal")
            except Exception as e:
                self.logger.error(f"Error loading trade journal: {e}")
    
    def _save_trade_journal(self) -> None:
        """Save the trade journal to disk."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.journal_path), exist_ok=True)
        
        # Save journal
        try:
            with open(self.journal_path, 'w') as f:
                json.dump(self.trade_journal, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving trade journal: {e}")
    
    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers to their initial state."""
        self.circuit_breaker.reset()
        self.logger.info("Circuit breakers reset")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive risk management report.
        
        Returns:
            Dictionary with risk management report
        """
        # Get circuit breaker status
        circuit_status = self.circuit_breaker.get_status()
        
        # Get drawdown report
        drawdown_report = self.circuit_breaker.get_drawdown_report()
        
        # Get trade analytics
        trade_analytics = self.get_trade_analytics()
        
        # Combine into a report
        return {
            "current_status": circuit_status,
            "drawdown_stats": {
                "daily_max": float(drawdown_report["daily_stats"]["max_drawdown"].iloc[0]) if not drawdown_report["daily_stats"].empty else 0,
                "weekly_max": float(drawdown_report["weekly_stats"]["max_drawdown"].iloc[0]) if not drawdown_report["weekly_stats"].empty else 0,
                "monthly_max": float(drawdown_report["monthly_stats"]["max_drawdown"].iloc[0]) if not drawdown_report["monthly_stats"].empty else 0,
            },
            "trade_analytics": trade_analytics,
            "risk_settings": {
                "max_risk_per_trade": self.max_risk_per_trade,
                "safe_mode_risk": self.position_sizer.safe_mode_risk_pct,
                "max_leverage": self.max_leverage,
                "daily_drawdown_limit": self.circuit_breaker.daily_drawdown_limit,
                "weekly_drawdown_limit": self.circuit_breaker.weekly_drawdown_limit,
                "monthly_drawdown_limit": self.circuit_breaker.monthly_drawdown_limit,
                "consecutive_loss_limit": self.circuit_breaker.consecutive_loss_limit,
                "auto_reduce_risk": self.auto_reduce_risk
            }
        } 