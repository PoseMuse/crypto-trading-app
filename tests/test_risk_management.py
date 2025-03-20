"""
Tests for risk management functionality.

This module contains tests for the risk management components, including
position sizing and circuit breakers.
"""

import os
import unittest
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import risk management components
from src.risk_management.position_sizer import PositionSizer
from src.risk_management.circuit_breakers import CircuitBreaker
from src.risk_management.risk_manager import RiskManager


class TestPositionSizer(unittest.TestCase):
    """
    Test cases for the PositionSizer class.
    """
    
    def setUp(self):
        """Set up the test environment."""
        self.position_sizer = PositionSizer(
            max_risk_pct=0.01,  # 1% risk
            safe_mode_risk_pct=0.005,  # 0.5% risk in safe mode
            min_position_size=0.001,
            max_position_pct=0.95,
            max_leverage=1.0
        )
    
    def test_calculate_position_size_long(self):
        """Test position size calculation for a long trade."""
        # Parameters
        capital = 10000.0
        entry_price = 40000.0  # BTC price
        stop_loss_price = 38000.0  # 5% stop loss
        
        # Calculate position size
        position_size, details = self.position_sizer.calculate_position_size(
            capital=capital,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price
        )
        
        # Expected position size based on 1% risk
        risk_amount = capital * 0.01  # $100
        price_diff = entry_price - stop_loss_price  # $2000
        expected_position_size = risk_amount / price_diff  # 0.05 BTC
        
        # Check position size
        self.assertAlmostEqual(position_size, expected_position_size, places=6)
        
        # Check risk details
        self.assertEqual(details["direction"], "long")
        self.assertAlmostEqual(details["risk_amount"], risk_amount, places=2)
        self.assertAlmostEqual(details["max_loss"], risk_amount, places=2)
        self.assertAlmostEqual(details["capital_at_risk_pct"], 0.01, places=4)
    
    def test_calculate_position_size_short(self):
        """Test position size calculation for a short trade."""
        # Parameters
        capital = 10000.0
        entry_price = 38000.0  # BTC price
        stop_loss_price = 40000.0  # Stop above entry for short
        
        # Calculate position size
        position_size, details = self.position_sizer.calculate_position_size(
            capital=capital,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price
        )
        
        # Expected position size based on 1% risk
        risk_amount = capital * 0.01  # $100
        price_diff = stop_loss_price - entry_price  # $2000
        expected_position_size = risk_amount / price_diff  # 0.05 BTC
        
        # Check position size
        self.assertAlmostEqual(position_size, expected_position_size, places=6)
        
        # Check risk details
        self.assertEqual(details["direction"], "short")
        self.assertAlmostEqual(details["risk_amount"], risk_amount, places=2)
        self.assertAlmostEqual(details["max_loss"], risk_amount, places=2)
        self.assertAlmostEqual(details["capital_at_risk_pct"], 0.01, places=4)
    
    def test_safe_mode_reduces_risk(self):
        """Test that safe mode reduces position size."""
        # Parameters
        capital = 10000.0
        entry_price = 40000.0
        stop_loss_price = 38000.0
        
        # Calculate normal position size
        normal_position, normal_details = self.position_sizer.calculate_position_size(
            capital=capital,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            safe_mode=False
        )
        
        # Calculate safe mode position size
        safe_position, safe_details = self.position_sizer.calculate_position_size(
            capital=capital,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            safe_mode=True
        )
        
        # Safe mode should reduce position size by 50%
        self.assertAlmostEqual(safe_position, normal_position * 0.5, places=6)
        self.assertAlmostEqual(safe_details["risk_amount"], normal_details["risk_amount"] * 0.5, places=2)
    
    def test_max_position_limit(self):
        """Test that position size is limited by max position percentage."""
        # Parameters
        capital = 10000.0
        entry_price = 40000.0
        stop_loss_price = 39900.0  # Very tight stop loss
        
        # Calculate position size with tight stop loss
        position_size, details = self.position_sizer.calculate_position_size(
            capital=capital,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price
        )
        
        # Expected position size based on max position percentage
        max_expected = (capital * 0.95) / entry_price
        
        # Position should be limited by max position
        self.assertLessEqual(position_size, max_expected)
        
        # Risk should be less than 1% due to the limit
        self.assertLess(details["capital_at_risk_pct"], 0.01)
    
    def test_invalid_stop_loss(self):
        """Test behavior with invalid stop loss (same as entry)."""
        # Parameters
        capital = 10000.0
        entry_price = 40000.0
        stop_loss_price = 40000.0  # Same as entry
        
        # Calculate position size
        position_size, details = self.position_sizer.calculate_position_size(
            capital=capital,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price
        )
        
        # Should return minimum position size
        self.assertEqual(position_size, self.position_sizer.min_position_size)
        self.assertEqual(details["capital_at_risk_pct"], 0.0)
        self.assertIn("reason", details)
    
    def test_calculate_stop_loss_price(self):
        """Test calculation of stop loss price from risk."""
        # Parameters
        capital = 10000.0
        entry_price = 40000.0
        position_size = 0.05  # BTC
        
        # Calculate stop loss price for a long position
        stop_loss_price = self.position_sizer.calculate_stop_loss_price(
            entry_price=entry_price,
            position_size=position_size,
            capital=capital,
            direction="long"
        )
        
        # Expected stop loss price based on 1% risk
        risk_amount = capital * 0.01  # $100
        price_change = risk_amount / position_size  # $2000
        expected_stop_loss = entry_price - price_change  # $38000
        
        # Check stop loss price
        self.assertAlmostEqual(stop_loss_price, expected_stop_loss, places=2)


class TestCircuitBreaker(unittest.TestCase):
    """
    Test cases for the CircuitBreaker class.
    """
    
    def setUp(self):
        """Set up the test environment."""
        self.circuit_breaker = CircuitBreaker(
            daily_drawdown_limit=0.05,  # 5% daily drawdown limit
            weekly_drawdown_limit=0.10,  # 10% weekly drawdown limit
            monthly_drawdown_limit=0.20,  # 20% monthly drawdown limit
            unrealized_loss_limit=0.10,  # 10% unrealized loss limit
            consecutive_loss_limit=3,    # 3 consecutive losses
            safe_mode_duration=1         # 1 day safe mode
        )
        
        # Initialize with starting equity
        self.initial_equity = 10000.0
        self.circuit_breaker.initialize(self.initial_equity)
    
    def test_update_equity_no_triggers(self):
        """Test equity update with no circuit breaker triggers."""
        # Update with small equity change
        current_equity = 9900.0  # 1% drawdown
        unrealized_pnl = -50.0
        
        # Update equity
        status = self.circuit_breaker.update_equity(current_equity, unrealized_pnl)
        
        # Check status
        self.assertFalse(status["daily_drawdown_triggered"])
        self.assertFalse(status["weekly_drawdown_triggered"])
        self.assertFalse(status["monthly_drawdown_triggered"])
        self.assertFalse(status["unrealized_loss_triggered"])
        self.assertFalse(status["halt_trading"])
        self.assertFalse(status["safe_mode_active"])
        
        # Check drawdown calculations
        self.assertAlmostEqual(status["daily_drawdown"], 0.01, places=4)  # 1% drawdown
    
    def test_daily_drawdown_trigger(self):
        """Test that daily drawdown limit triggers circuit breaker."""
        # Update with large equity drop
        current_equity = 9400.0  # 6% drawdown
        unrealized_pnl = -200.0
        
        # Update equity
        status = self.circuit_breaker.update_equity(current_equity, unrealized_pnl)
        
        # Check status
        self.assertTrue(status["daily_drawdown_triggered"])
        self.assertTrue(status["halt_trading"])
        self.assertTrue(status["safe_mode_active"])
        
        # Check drawdown calculations
        self.assertAlmostEqual(status["daily_drawdown"], 0.06, places=4)  # 6% drawdown
    
    def test_consecutive_losses(self):
        """Test that consecutive losses trigger safe mode."""
        # Record three consecutive losses
        self.circuit_breaker.record_trade_result(-50.0)
        self.circuit_breaker.record_trade_result(-75.0)
        
        # After two losses, safe mode should not be active
        self.assertEqual(self.circuit_breaker.consecutive_losses, 2)
        self.assertFalse(self.circuit_breaker.safe_mode_active)
        
        # Record third loss
        self.circuit_breaker.record_trade_result(-100.0)
        
        # After three losses, safe mode should be active
        self.assertEqual(self.circuit_breaker.consecutive_losses, 3)
        self.assertTrue(self.circuit_breaker.safe_mode_active)
        
        # Record a win
        self.circuit_breaker.record_trade_result(200.0)
        
        # Consecutive losses should be reset
        self.assertEqual(self.circuit_breaker.consecutive_losses, 0)
        
        # Safe mode should still be active until duration expires
        self.assertTrue(self.circuit_breaker.safe_mode_active)
        self.assertIsNotNone(self.circuit_breaker.safe_mode_end_date)
    
    def test_should_exit_position(self):
        """Test the should_exit_position method."""
        # Set up a position with unrealized loss
        unrealized_pnl = -1100.0  # 11% of equity
        entry_price = 40000.0
        current_price = 36000.0
        
        # Check if position should be exited
        should_exit, reason = self.circuit_breaker.should_exit_position(
            unrealized_pnl=unrealized_pnl,
            entry_price=entry_price,
            current_price=current_price
        )
        
        # Should exit due to unrealized loss > 10%
        self.assertTrue(should_exit)
        self.assertIn("Unrealized loss limit", reason)
    
    def test_reset(self):
        """Test resetting the circuit breaker."""
        # Trigger circuit breaker first
        self.circuit_breaker.record_trade_result(-50.0)
        self.circuit_breaker.record_trade_result(-75.0)
        self.circuit_breaker.record_trade_result(-100.0)
        
        # Verify safe mode is active
        self.assertTrue(self.circuit_breaker.safe_mode_active)
        self.assertEqual(self.circuit_breaker.consecutive_losses, 3)
        
        # Reset circuit breaker
        self.circuit_breaker.reset()
        
        # Verify reset state
        self.assertFalse(self.circuit_breaker.safe_mode_active)
        self.assertEqual(self.circuit_breaker.consecutive_losses, 0)
        self.assertIsNone(self.circuit_breaker.safe_mode_end_date)


class TestRiskManager(unittest.TestCase):
    """
    Test cases for the RiskManager class.
    """
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for the trade journal
        self.test_dir = tempfile.mkdtemp()
        self.journal_path = os.path.join(self.test_dir, "test_journal.json")
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            initial_equity=10000.0,
            max_risk_per_trade=0.01,
            max_leverage=1.0,
            daily_drawdown_limit=0.05,
            weekly_drawdown_limit=0.10,
            monthly_drawdown_limit=0.20,
            consecutive_loss_limit=3,
            auto_reduce_risk=True,
            journal_path=self.journal_path
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory and files
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)
    
    def test_calculate_position_size(self):
        """Test position size calculation with risk manager."""
        # Parameters
        entry_price = 40000.0
        stop_loss_price = 38000.0
        symbol = "BTC/USDT"
        direction = "long"
        
        # Calculate position size
        result = self.risk_manager.calculate_position_size(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            symbol=symbol,
            direction=direction
        )
        
        # Check result
        self.assertGreater(result["position_size"], 0)
        self.assertTrue(result["allowed_to_trade"])
        self.assertFalse(result["safe_mode"])
        self.assertAlmostEqual(result["capital_at_risk"], 0.01, places=4)  # 1% risk
    
    def test_halt_trading_on_drawdown(self):
        """Test that trading is halted on excessive drawdown."""
        # First update equity with a large drawdown
        self.risk_manager.update_equity(9400.0)  # 6% drawdown
        
        # Now try to calculate position size
        result = self.risk_manager.calculate_position_size(
            entry_price=40000.0,
            stop_loss_price=38000.0,
            symbol="BTC/USDT",
            direction="long"
        )
        
        # Check that trading is halted
        self.assertEqual(result["position_size"], 0.0)
        self.assertFalse(result["allowed_to_trade"])
        self.assertTrue(result["safe_mode"])
        self.assertIn("Circuit breaker triggered", result["reason"])
    
    def test_record_trade(self):
        """Test recording a trade in the journal."""
        # Record a trade
        self.risk_manager.record_trade(
            symbol="BTC/USDT",
            direction="long",
            entry_price=35000.0,
            exit_price=38000.0,
            position_size=0.05,
            entry_time=datetime.now() - timedelta(hours=2),
            exit_time=datetime.now(),
            pnl=150.0,
            stop_loss=33000.0,
            take_profit=40000.0,
            fees=10.0
        )
        
        # Check that trade was recorded
        self.assertEqual(len(self.risk_manager.trade_journal), 1)
        
        # Check that journal file was created
        self.assertTrue(os.path.exists(self.journal_path))
        
        # Load journal and check content
        with open(self.journal_path, 'r') as f:
            import json
            journal = json.load(f)
            self.assertEqual(len(journal), 1)
            self.assertEqual(journal[0]["symbol"], "BTC/USDT")
            self.assertEqual(journal[0]["direction"], "long")
            self.assertEqual(journal[0]["pnl"], 150.0)
    
    def test_get_trade_analytics(self):
        """Test generating trade analytics."""
        # Record some trades
        # Win
        self.risk_manager.record_trade(
            symbol="BTC/USDT",
            direction="long",
            entry_price=35000.0,
            exit_price=38000.0,
            position_size=0.05,
            entry_time=datetime.now() - timedelta(hours=4),
            exit_time=datetime.now() - timedelta(hours=3),
            pnl=150.0,
            stop_loss=33000.0
        )
        
        # Loss
        self.risk_manager.record_trade(
            symbol="ETH/USDT",
            direction="long",
            entry_price=2000.0,
            exit_price=1900.0,
            position_size=0.5,
            entry_time=datetime.now() - timedelta(hours=2),
            exit_time=datetime.now() - timedelta(hours=1),
            pnl=-50.0,
            stop_loss=1850.0
        )
        
        # Get analytics
        analytics = self.risk_manager.get_trade_analytics()
        
        # Check analytics
        self.assertEqual(analytics["total_trades"], 2)
        self.assertEqual(analytics["winning_trades"], 1)
        self.assertEqual(analytics["losing_trades"], 1)
        self.assertAlmostEqual(analytics["win_rate"], 0.5, places=4)
        self.assertEqual(analytics["total_profit"], 150.0)
        self.assertEqual(analytics["total_loss"], -50.0)
        self.assertEqual(analytics["net_profit"], 100.0)
    
    def test_risk_report(self):
        """Test generating a risk report."""
        # Update equity with a small drawdown
        self.risk_manager.update_equity(9800.0)  # 2% drawdown
        
        # Get risk report
        report = self.risk_manager.get_risk_report()
        
        # Check report structure
        self.assertIn("current_status", report)
        self.assertIn("drawdown_stats", report)
        self.assertIn("trade_analytics", report)
        self.assertIn("risk_settings", report)
        
        # Check some values
        self.assertAlmostEqual(
            report["current_status"]["daily_drawdown"], 
            0.02, 
            places=4
        )
        
        self.assertEqual(
            report["risk_settings"]["max_risk_per_trade"],
            0.01
        )


if __name__ == '__main__':
    unittest.main() 