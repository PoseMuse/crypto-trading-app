"""
Position Sizing Module based on Risk Management Rules.

This module provides functionality for calculating position sizes
based on risk management rules like the 1% rule.
"""

from typing import Dict, Optional, Tuple, Union
import logging


class PositionSizer:
    """
    Position sizer that implements various risk management rules.
    
    Primary functionality:
    - Implements the 1% rule (risk no more than 1% of capital per trade)
    - Adjusts position size based on stop loss distance
    - Supports scaling position sizes based on volatility
    - Handles safe mode with reduced position sizes
    """
    
    def __init__(
        self,
        max_risk_pct: float = 0.01,  # 1% risk per trade
        safe_mode_risk_pct: float = 0.005,  # 0.5% risk in safe mode
        min_position_size: float = 0.0,  # Minimum position size
        max_position_pct: float = 0.95,  # Maximum position as % of available capital
        max_leverage: float = 1.0  # Maximum allowed leverage (1.0 = no leverage)
    ):
        """
        Initialize the position sizer with risk parameters.
        
        Args:
            max_risk_pct: Maximum risk per trade as percentage of capital
            safe_mode_risk_pct: Reduced risk percentage when in safe mode
            min_position_size: Minimum position size (for minimum order requirements)
            max_position_pct: Maximum position size as % of available capital
            max_leverage: Maximum allowed leverage
        """
        self.max_risk_pct = max_risk_pct
        self.safe_mode_risk_pct = safe_mode_risk_pct
        self.min_position_size = min_position_size
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: float,
        safe_mode: bool = False,
        volatility_factor: float = 1.0,
        current_margin_used: float = 0.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate position size based on the 1% rule and other constraints.
        
        The 1% rule states that no more than 1% of total capital should be
        at risk in any single trade. Position size is adjusted to ensure that
        if the stop loss is hit, the loss is limited to 1% of capital.
        
        Args:
            capital: Total available capital
            entry_price: Entry price of the asset
            stop_loss_price: Stop loss price
            safe_mode: Whether to use safe mode (reduced risk)
            volatility_factor: Factor to adjust position size based on volatility (< 1 for high volatility)
            current_margin_used: Amount of margin currently in use
            
        Returns:
            Tuple of (position_size, details)
        """
        # Determine which risk percentage to use
        risk_pct = self.safe_mode_risk_pct if safe_mode else self.max_risk_pct
        
        # Calculate risk amount in currency units
        risk_amount = capital * risk_pct
        
        # Calculate the price difference between entry and stop loss
        if entry_price > stop_loss_price:
            # Long position
            price_difference = entry_price - stop_loss_price
            position_direction = "long"
        else:
            # Short position
            price_difference = stop_loss_price - entry_price
            position_direction = "short"
        
        # Calculate risk per unit
        risk_per_unit = price_difference
        
        # If price difference is zero or negative, return minimum position
        if risk_per_unit <= 0:
            self.logger.warning("Invalid stop loss placement - entry and stop loss prices too close")
            return self.min_position_size, {
                "position_size": self.min_position_size,
                "risk_amount": 0.0,
                "max_loss": 0.0,
                "capital_at_risk_pct": 0.0,
                "reason": "Invalid stop loss",
                "direction": position_direction
            }
        
        # Calculate raw position size based on risk
        position_size = risk_amount / risk_per_unit
        
        # Adjust for volatility
        position_size *= volatility_factor
        
        # Check against minimum position size
        if position_size < self.min_position_size:
            position_size = self.min_position_size
            actual_risk = position_size * risk_per_unit
            actual_risk_pct = actual_risk / capital
            
            self.logger.info(f"Position size adjusted to minimum. Actual risk: {actual_risk_pct:.2%}")
        
        # Check against maximum position size (% of capital)
        max_position_size = (capital * self.max_position_pct) / entry_price
        if position_size > max_position_size:
            position_size = max_position_size
            actual_risk = position_size * risk_per_unit
            actual_risk_pct = actual_risk / capital
            
            self.logger.info(f"Position size limited by max capital %. Actual risk: {actual_risk_pct:.2%}")
        
        # Check leverage limits
        available_margin = capital * self.max_leverage - current_margin_used
        if position_size * entry_price > available_margin:
            # Adjust position size to stay within leverage limits
            position_size = available_margin / entry_price
            actual_risk = position_size * risk_per_unit
            actual_risk_pct = actual_risk / capital
            
            self.logger.info(f"Position size limited by max leverage. Actual risk: {actual_risk_pct:.2%}")
        
        # Calculate actual maximum loss
        max_loss = position_size * risk_per_unit
        
        # Return position size and details
        details = {
            "position_size": position_size,
            "risk_amount": risk_amount,
            "max_loss": max_loss,
            "capital_at_risk_pct": max_loss / capital,
            "risk_reward_ratio": None,  # Would require a target price
            "direction": position_direction
        }
        
        return position_size, details
    
    def calculate_stop_loss_price(
        self,
        entry_price: float,
        position_size: float,
        capital: float,
        direction: str = "long"
    ) -> float:
        """
        Calculate a stop loss price based on the 1% rule.
        
        Given an entry price and position size, calculate where
        the stop loss should be placed to limit risk to 1%.
        
        Args:
            entry_price: Entry price of the asset
            position_size: Position size
            capital: Total available capital
            direction: Trade direction ("long" or "short")
            
        Returns:
            Stop loss price
        """
        # Calculate max allowed loss
        max_loss = capital * self.max_risk_pct
        
        # Calculate price change that would cause max loss
        price_change = max_loss / position_size
        
        # Calculate stop loss price based on direction
        if direction.lower() == "long":
            stop_loss_price = entry_price - price_change
        else:
            stop_loss_price = entry_price + price_change
        
        return stop_loss_price
    
    def adjust_for_volatility(
        self,
        position_size: float,
        atr: float,
        price: float,
        atr_factor: float = 2.0
    ) -> float:
        """
        Adjust position size based on volatility using ATR.
        
        Args:
            position_size: Original position size
            atr: Average True Range
            price: Current price
            atr_factor: Factor to multiply ATR by
            
        Returns:
            Adjusted position size
        """
        # Calculate volatility as percentage of price
        volatility_pct = atr / price
        
        # Calculate adjustment factor
        # Higher volatility = smaller position
        adjustment = 1.0 / (1.0 + volatility_pct * atr_factor)
        
        # Adjust position size
        adjusted_position = position_size * adjustment
        
        return adjusted_position 