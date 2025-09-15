"""
Comprehensive Unit Tests for Greeks Risk Manager

Tests portfolio-wide Greeks aggregation, exposure limits,
risk calculations, and position sizing validations.
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import asdict

from app.trading.greeks_risk_manager import (
    GreeksRiskManager,
    GreeksLimits,
    GreeksRiskMetrics
)
from app.trading.options_trading import OptionPosition, GreeksData
from app.trading.risk_manager import RiskManager
from app.core.exceptions import TradingError


class TestGreeksLimits:
    """Test Greeks limits configuration"""

    def test_default_limits_values(self):
        """Test default Greeks limits are reasonable"""
        limits = GreeksLimits()

        # Delta limits
        assert limits.max_portfolio_delta == 1000.0
        assert limits.max_single_position_delta == 100.0
        assert limits.max_sector_delta == 500.0

        # Gamma limits
        assert limits.max_portfolio_gamma == 10.0
        assert limits.max_single_position_gamma == 2.0
        assert limits.gamma_scalping_threshold == 5.0

        # Vega limits
        assert limits.max_portfolio_vega == 1000.0
        assert limits.max_single_position_vega == 100.0
        assert limits.max_iv_exposure_pct == 25.0

        # Theta limits
        assert limits.max_portfolio_theta == -100.0
        assert limits.min_theta_yield_pct == 0.1
        assert limits.max_theta_concentration == 0.5

        # Rho limits
        assert limits.max_portfolio_rho == 500.0
        assert limits.max_duration_years == 1.0

    def test_custom_limits_values(self):
        """Test custom Greeks limits"""
        limits = GreeksLimits(
            max_portfolio_delta=2000.0,
            max_portfolio_gamma=20.0,
            max_portfolio_vega=2000.0,
            gamma_scalping_threshold=10.0
        )

        assert limits.max_portfolio_delta == 2000.0
        assert limits.max_portfolio_gamma == 20.0
        assert limits.max_portfolio_vega == 2000.0
        assert limits.gamma_scalping_threshold == 10.0


class TestGreeksRiskMetrics:
    """Test Greeks risk metrics data structure"""

    def test_metrics_creation(self):
        """Test creating Greeks risk metrics"""
        timestamp = datetime.now()
        metrics = GreeksRiskMetrics(
            portfolio_delta=150.0,
            portfolio_gamma=5.5,
            portfolio_theta=-45.0,
            portfolio_vega=250.0,
            portfolio_rho=100.0,
            delta_utilization=0.15,
            gamma_utilization=0.55,
            vega_utilization=0.25,
            theta_utilization=0.45,
            largest_position_delta=50.0,
            delta_concentration=0.33,
            gamma_concentration=0.40,
            vega_concentration=0.20,
            delta_limit_breach=False,
            gamma_limit_breach=False,
            vega_limit_breach=False,
            concentration_warning=True,
            last_updated=timestamp,
            next_rebalance=timestamp + timedelta(hours=1)
        )

        assert metrics.portfolio_delta == 150.0
        assert metrics.portfolio_gamma == 5.5
        assert metrics.portfolio_theta == -45.0
        assert metrics.portfolio_vega == 250.0
        assert metrics.delta_utilization == 0.15
        assert metrics.concentration_warning is True
        assert metrics.last_updated == timestamp

    def test_metrics_default_values(self):
        """Test default values for risk metrics"""
        metrics = GreeksRiskMetrics()

        assert metrics.portfolio_delta == 0.0
        assert metrics.portfolio_gamma == 0.0
        assert metrics.portfolio_theta == 0.0
        assert metrics.portfolio_vega == 0.0
        assert metrics.portfolio_rho == 0.0
        assert metrics.delta_limit_breach is False
        assert metrics.gamma_limit_breach is False
        assert metrics.vega_limit_breach is False


class TestGreeksRiskManager:
    """Comprehensive tests for Greeks Risk Manager"""

    @pytest.fixture
    def mock_base_risk_manager(self):
        """Mock base risk manager"""
        manager = MagicMock(spec=RiskManager)
        manager.validate_order = AsyncMock(return_value=(True, ""))
        manager.get_portfolio_value = AsyncMock(return_value=100000.0)
        manager.get_buying_power = AsyncMock(return_value=50000.0)
        return manager

    @pytest.fixture
    def default_limits(self):
        """Default Greeks limits fixture"""
        return GreeksLimits()

    @pytest.fixture
    def custom_limits(self):
        """Custom Greeks limits for testing"""
        return GreeksLimits(
            max_portfolio_delta=500.0,
            max_portfolio_gamma=5.0,
            max_portfolio_vega=500.0,
            max_single_position_delta=50.0
        )

    @pytest.fixture
    def greeks_manager(self, mock_base_risk_manager, default_limits):
        """Greeks risk manager fixture"""
        return GreeksRiskManager(
            alpaca_client=MagicMock(),
            config=MagicMock(),
            limits=default_limits
        )

    @pytest.fixture
    def sample_option_positions(self):
        """Sample option positions for testing"""
        return [
            OptionPosition(
                symbol="AAPL250221C00150000",
                underlying="AAPL",
                quantity=10,
                side="long",
                avg_price=3.50,
                current_price=3.75,
                greeks=GreeksData(
                    delta=0.50, gamma=0.08, theta=-0.05, vega=0.12, rho=0.03
                ),
                expiration_date=datetime(2025, 2, 21),
                strike_price=150.0,
                option_type="call"
            ),
            OptionPosition(
                symbol="AAPL250221P00145000",
                underlying="AAPL",
                quantity=5,
                side="short",
                avg_price=2.00,
                current_price=1.85,
                greeks=GreeksData(
                    delta=-0.30, gamma=0.06, theta=-0.03, vega=0.08, rho=-0.02
                ),
                expiration_date=datetime(2025, 2, 21),
                strike_price=145.0,
                option_type="put"
            ),
            OptionPosition(
                symbol="MSFT250328C00350000",
                underlying="MSFT",
                quantity=8,
                side="long",
                avg_price=5.25,
                current_price=5.50,
                greeks=GreeksData(
                    delta=0.45, gamma=0.05, theta=-0.08, vega=0.15, rho=0.05
                ),
                expiration_date=datetime(2025, 3, 28),
                strike_price=350.0,
                option_type="call"
            )
        ]

    def test_manager_initialization_default_limits(self, mock_base_risk_manager):
        """Test manager initialization with default limits"""
        manager = GreeksRiskManager(
            alpaca_client=MagicMock(),
            config=MagicMock()
        )

        assert isinstance(manager.limits, GreeksLimits)
        assert manager.limits.max_portfolio_delta == 1000.0
        assert manager.current_metrics is None

    def test_manager_initialization_custom_limits(self, mock_base_risk_manager, custom_limits):
        """Test manager initialization with custom limits"""
        manager = GreeksRiskManager(
            alpaca_client=MagicMock(),
            config=MagicMock(),
            limits=custom_limits
        )

        assert manager.limits == custom_limits
        assert manager.limits.max_portfolio_delta == 500.0

    @pytest.mark.asyncio
    async def test_update_portfolio_greeks_basic(self, greeks_manager, sample_option_positions):
        """Test basic portfolio Greeks aggregation"""
        metrics = await greeks_manager.update_portfolio_greeks(sample_option_positions)

        # Calculate expected values
        expected_delta = (10 * 0.50) + (5 * -0.30 * -1) + (8 * 0.45)  # Short positions inverted
        expected_gamma = (10 * 0.08) + (5 * 0.06) + (8 * 0.05)
        expected_theta = (10 * -0.05) + (5 * -0.03 * -1) + (8 * -0.08)  # Short theta is positive income
        expected_vega = (10 * 0.12) + (5 * 0.08 * -1) + (8 * 0.15)  # Short vega is negative
        expected_rho = (10 * 0.03) + (5 * -0.02 * -1) + (8 * 0.05)

        assert metrics.portfolio_delta == pytest.approx(expected_delta, rel=1e-2)
        assert metrics.portfolio_gamma == pytest.approx(expected_gamma, rel=1e-2)
        assert metrics.portfolio_theta == pytest.approx(expected_theta, rel=1e-2)
        assert metrics.portfolio_vega == pytest.approx(expected_vega, rel=1e-2)
        assert metrics.portfolio_rho == pytest.approx(expected_rho, rel=1e-2)

    @pytest.mark.asyncio
    async def test_update_portfolio_greeks_utilization(self, greeks_manager, sample_option_positions):
        """Test Greeks utilization calculation"""
        metrics = await greeks_manager.update_portfolio_greeks(sample_option_positions)

        # Calculate expected utilizations based on limits
        expected_delta_util = abs(metrics.portfolio_delta) / greeks_manager.limits.max_portfolio_delta
        expected_gamma_util = metrics.portfolio_gamma / greeks_manager.limits.max_portfolio_gamma
        expected_vega_util = abs(metrics.portfolio_vega) / greeks_manager.limits.max_portfolio_vega
        expected_theta_util = abs(metrics.portfolio_theta) / abs(greeks_manager.limits.max_portfolio_theta)

        assert metrics.delta_utilization == pytest.approx(expected_delta_util, rel=1e-2)
        assert metrics.gamma_utilization == pytest.approx(expected_gamma_util, rel=1e-2)
        assert metrics.vega_utilization == pytest.approx(expected_vega_util, rel=1e-2)
        assert metrics.theta_utilization == pytest.approx(expected_theta_util, rel=1e-2)

    @pytest.mark.asyncio
    async def test_update_portfolio_greeks_concentration(self, greeks_manager, sample_option_positions):
        """Test concentration risk calculation"""
        metrics = await greeks_manager.update_portfolio_greeks(sample_option_positions)

        # Find largest position delta
        position_deltas = [
            abs(10 * 0.50),           # AAPL long call
            abs(5 * -0.30 * -1),      # AAPL short put (inverted)
            abs(8 * 0.45)             # MSFT long call
        ]
        largest_delta = max(position_deltas)
        expected_concentration = largest_delta / abs(metrics.portfolio_delta) if metrics.portfolio_delta != 0 else 0

        assert metrics.largest_position_delta == largest_delta
        assert metrics.delta_concentration == pytest.approx(expected_concentration, rel=1e-2)

    @pytest.mark.asyncio
    async def test_update_portfolio_greeks_limit_breaches(self, greeks_manager, custom_limits):
        """Test limit breach detection"""
        greeks_manager.limits = custom_limits  # Use smaller limits

        # Create positions that exceed limits
        high_risk_positions = [
            OptionPosition(
                symbol="AAPL250221C00150000",
                underlying="AAPL",
                quantity=50,  # Large quantity
                side="long",
                avg_price=3.50,
                current_price=3.75,
                greeks=GreeksData(
                    delta=0.80, gamma=0.20, theta=-0.10, vega=0.25, rho=0.05
                ),
                expiration_date=datetime(2025, 2, 21),
                strike_price=150.0,
                option_type="call"
            )
        ]

        metrics = await greeks_manager.update_portfolio_greeks(high_risk_positions)

        # Check for limit breaches
        assert metrics.portfolio_delta > custom_limits.max_portfolio_delta
        assert metrics.portfolio_gamma > custom_limits.max_portfolio_gamma
        assert metrics.portfolio_vega > custom_limits.max_portfolio_vega
        assert metrics.delta_limit_breach is True
        assert metrics.gamma_limit_breach is True
        assert metrics.vega_limit_breach is True

    @pytest.mark.asyncio
    async def test_validate_greeks_order_valid(self, greeks_manager, sample_option_positions):
        """Test validating a valid Greeks order"""
        # Update current portfolio
        await greeks_manager.update_portfolio_greeks(sample_option_positions)

        # Test adding a reasonable position
        new_position = OptionPosition(
            symbol="TSLA250221C00200000",
            underlying="TSLA",
            quantity=5,
            side="long",
            avg_price=4.00,
            current_price=4.20,
            greeks=GreeksData(
                delta=0.40, gamma=0.06, theta=-0.04, vega=0.10, rho=0.04
            ),
            expiration_date=datetime(2025, 2, 21),
            strike_price=200.0,
            option_type="call"
        )

        is_valid, message, adjustments = greeks_manager.validate_greeks_order(
            new_position, "buy", 5
        )

        assert is_valid is True
        assert message == ""
        assert adjustments is None

    @pytest.mark.asyncio
    async def test_validate_greeks_order_exceeds_limits(self, greeks_manager, sample_option_positions, custom_limits):
        """Test validating order that exceeds limits"""
        greeks_manager.limits = custom_limits
        await greeks_manager.update_portfolio_greeks(sample_option_positions)

        # Test adding position that would exceed limits
        large_position = OptionPosition(
            symbol="TSLA250221C00200000",
            underlying="TSLA",
            quantity=100,  # Very large quantity
            side="long",
            avg_price=4.00,
            current_price=4.20,
            greeks=GreeksData(
                delta=0.80, gamma=0.15, theta=-0.10, vega=0.20, rho=0.08
            ),
            expiration_date=datetime(2025, 2, 21),
            strike_price=200.0,
            option_type="call"
        )

        is_valid, message, adjustments = greeks_manager.validate_greeks_order(
            large_position, "buy", 100
        )

        assert is_valid is False
        assert "limit" in message.lower()
        assert adjustments is not None
        assert adjustments["recommended_quantity"] < 100

    def test_calculate_position_greeks(self, greeks_manager):
        """Test position Greeks calculation"""
        position = OptionPosition(
            symbol="AAPL250221C00150000",
            underlying="AAPL",
            quantity=10,
            side="long",
            avg_price=3.50,
            current_price=3.75,
            greeks=GreeksData(
                delta=0.50, gamma=0.08, theta=-0.05, vega=0.12, rho=0.03
            ),
            expiration_date=datetime(2025, 2, 21),
            strike_price=150.0,
            option_type="call"
        )

        greeks = greeks_manager._calculate_position_greeks(position)

        # For long position, Greeks should be quantity * individual Greeks
        assert greeks["delta"] == 10 * 0.50
        assert greeks["gamma"] == 10 * 0.08
        assert greeks["theta"] == 10 * -0.05
        assert greeks["vega"] == 10 * 0.12
        assert greeks["rho"] == 10 * 0.03

    def test_calculate_position_greeks_short(self, greeks_manager):
        """Test position Greeks calculation for short position"""
        position = OptionPosition(
            symbol="AAPL250221P00145000",
            underlying="AAPL",
            quantity=5,
            side="short",
            avg_price=2.00,
            current_price=1.85,
            greeks=GreeksData(
                delta=-0.30, gamma=0.06, theta=-0.03, vega=0.08, rho=-0.02
            ),
            expiration_date=datetime(2025, 2, 21),
            strike_price=145.0,
            option_type="put"
        )

        greeks = greeks_manager._calculate_position_greeks(position)

        # For short position, delta/theta/vega are inverted, gamma/rho remain same
        assert greeks["delta"] == 5 * -0.30 * -1  # Short delta becomes positive income
        assert greeks["gamma"] == 5 * 0.06         # Gamma is always positive for seller
        assert greeks["theta"] == 5 * -0.03 * -1   # Short theta becomes positive income
        assert greeks["vega"] == 5 * 0.08 * -1     # Short vega is negative (benefit from vol drop)
        assert greeks["rho"] == 5 * -0.02 * -1     # Short rho is inverted

    def test_calculate_max_position_size_delta_limit(self, greeks_manager):
        """Test max position size based on delta limit"""
        # Position with delta 0.50 per contract
        greeks_data = GreeksData(delta=0.50, gamma=0.08, theta=-0.05, vega=0.12, rho=0.03)

        max_size = greeks_manager._calculate_max_position_size_delta_limit(greeks_data)

        # Max single position delta is 100, so max size = 100 / 0.50 = 200 contracts
        expected_max = int(greeks_manager.limits.max_single_position_delta / abs(greeks_data.delta))
        assert max_size == expected_max

    def test_calculate_max_position_size_gamma_limit(self, greeks_manager):
        """Test max position size based on gamma limit"""
        greeks_data = GreeksData(delta=0.50, gamma=0.08, theta=-0.05, vega=0.12, rho=0.03)

        max_size = greeks_manager._calculate_max_position_size_gamma_limit(greeks_data)

        # Max single position gamma is 2.0, so max size = 2.0 / 0.08 = 25 contracts
        expected_max = int(greeks_manager.limits.max_single_position_gamma / greeks_data.gamma)
        assert max_size == expected_max

    def test_calculate_max_position_size_vega_limit(self, greeks_manager):
        """Test max position size based on vega limit"""
        greeks_data = GreeksData(delta=0.50, gamma=0.08, theta=-0.05, vega=0.12, rho=0.03)

        max_size = greeks_manager._calculate_max_position_size_vega_limit(greeks_data)

        # Max single position vega is 100, so max size = 100 / 0.12 = 833 contracts
        expected_max = int(greeks_manager.limits.max_single_position_vega / greeks_data.vega)
        assert max_size == expected_max

    def test_get_position_size_recommendation(self, greeks_manager):
        """Test position size recommendation logic"""
        greeks_data = GreeksData(delta=0.50, gamma=0.08, theta=-0.05, vega=0.12, rho=0.03)
        desired_quantity = 50

        recommendation = greeks_manager.get_position_size_recommendation(
            greeks_data, desired_quantity
        )

        # Should recommend the most restrictive limit
        max_delta = int(greeks_manager.limits.max_single_position_delta / abs(greeks_data.delta))
        max_gamma = int(greeks_manager.limits.max_single_position_gamma / greeks_data.gamma)
        max_vega = int(greeks_manager.limits.max_single_position_vega / greeks_data.vega)

        most_restrictive = min(max_delta, max_gamma, max_vega, desired_quantity)

        assert recommendation["recommended_quantity"] == most_restrictive
        assert recommendation["limiting_factor"] in ["delta", "gamma", "vega", "desired"]

    def test_check_concentration_risk(self, greeks_manager):
        """Test concentration risk checking"""
        total_delta = 500.0
        position_delta = 300.0

        is_concentrated = greeks_manager._check_concentration_risk(total_delta, position_delta)

        concentration = position_delta / total_delta  # 0.6
        assert concentration > 0.5  # Above 50% threshold
        assert is_concentrated is True

    def test_check_concentration_risk_acceptable(self, greeks_manager):
        """Test acceptable concentration risk"""
        total_delta = 500.0
        position_delta = 200.0

        is_concentrated = greeks_manager._check_concentration_risk(total_delta, position_delta)

        concentration = position_delta / total_delta  # 0.4
        assert concentration <= 0.5  # Below 50% threshold
        assert is_concentrated is False

    @pytest.mark.asyncio
    async def test_get_portfolio_summary(self, greeks_manager, sample_option_positions):
        """Test portfolio summary generation"""
        await greeks_manager.update_portfolio_greeks(sample_option_positions)

        summary = await greeks_manager.get_portfolio_summary()

        assert "total_positions" in summary
        assert "portfolio_greeks" in summary
        assert "risk_utilization" in summary
        assert "limit_breaches" in summary
        assert summary["total_positions"] == len(sample_option_positions)

    @pytest.mark.asyncio
    async def test_suggest_hedge_trades(self, greeks_manager, sample_option_positions):
        """Test hedge trade suggestions"""
        await greeks_manager.update_portfolio_greeks(sample_option_positions)

        suggestions = await greeks_manager.suggest_hedge_trades()

        assert isinstance(suggestions, list)
        # Should suggest hedges if portfolio is not delta neutral
        if abs(greeks_manager.current_metrics.portfolio_delta) > greeks_manager.limits.target_delta:
            assert len(suggestions) > 0

    def test_calculate_greeks_pnl(self, greeks_manager):
        """Test Greeks P&L calculation"""
        price_change = 2.0
        vol_change = 0.02
        time_decay_days = 1

        greeks = {
            "delta": 100.0,
            "gamma": 5.0,
            "theta": -50.0,
            "vega": 200.0
        }

        pnl_breakdown = greeks_manager.calculate_greeks_pnl(
            greeks, price_change, vol_change, time_decay_days
        )

        expected_delta_pnl = greeks["delta"] * price_change
        expected_gamma_pnl = 0.5 * greeks["gamma"] * (price_change ** 2)
        expected_theta_pnl = greeks["theta"] * time_decay_days
        expected_vega_pnl = greeks["vega"] * vol_change

        assert pnl_breakdown["delta_pnl"] == pytest.approx(expected_delta_pnl, rel=1e-2)
        assert pnl_breakdown["gamma_pnl"] == pytest.approx(expected_gamma_pnl, rel=1e-2)
        assert pnl_breakdown["theta_pnl"] == pytest.approx(expected_theta_pnl, rel=1e-2)
        assert pnl_breakdown["vega_pnl"] == pytest.approx(expected_vega_pnl, rel=1e-2)


# Edge Cases and Error Handling Tests
class TestGreeksRiskManagerEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def greeks_manager(self):
        """Greeks manager for edge case testing"""
        return GreeksRiskManager(
            alpaca_client=MagicMock(),
            config=MagicMock(),
            limits=GreeksLimits()
        )

    @pytest.mark.asyncio
    async def test_update_portfolio_greeks_empty_positions(self, greeks_manager):
        """Test updating Greeks with empty positions list"""
        metrics = await greeks_manager.update_portfolio_greeks([])

        assert metrics.portfolio_delta == 0.0
        assert metrics.portfolio_gamma == 0.0
        assert metrics.portfolio_theta == 0.0
        assert metrics.portfolio_vega == 0.0
        assert metrics.portfolio_rho == 0.0
        assert metrics.delta_limit_breach is False

    @pytest.mark.asyncio
    async def test_update_portfolio_greeks_none_positions(self, greeks_manager):
        """Test updating Greeks with None positions"""
        metrics = await greeks_manager.update_portfolio_greeks(None)

        assert metrics.portfolio_delta == 0.0
        assert metrics.portfolio_gamma == 0.0

    def test_calculate_position_greeks_missing_greeks(self, greeks_manager):
        """Test position with missing Greeks data"""
        position = OptionPosition(
            symbol="AAPL250221C00150000",
            underlying="AAPL",
            quantity=10,
            side="long",
            avg_price=3.50,
            current_price=3.75,
            greeks=None,  # Missing Greeks
            expiration_date=datetime(2025, 2, 21),
            strike_price=150.0,
            option_type="call"
        )

        greeks = greeks_manager._calculate_position_greeks(position)

        # Should return zeros for missing Greeks
        assert greeks["delta"] == 0.0
        assert greeks["gamma"] == 0.0
        assert greeks["theta"] == 0.0
        assert greeks["vega"] == 0.0
        assert greeks["rho"] == 0.0

    def test_position_size_recommendation_zero_greeks(self, greeks_manager):
        """Test position size recommendation with zero Greeks"""
        greeks_data = GreeksData(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
        desired_quantity = 100

        recommendation = greeks_manager.get_position_size_recommendation(
            greeks_data, desired_quantity
        )

        # Should return desired quantity when Greeks are zero
        assert recommendation["recommended_quantity"] == desired_quantity
        assert recommendation["limiting_factor"] == "desired"

    def test_check_concentration_risk_zero_total(self, greeks_manager):
        """Test concentration risk with zero total delta"""
        is_concentrated = greeks_manager._check_concentration_risk(0.0, 100.0)

        # Should handle division by zero gracefully
        assert is_concentrated is False

    def test_validate_greeks_order_invalid_side(self, greeks_manager):
        """Test validating order with invalid side"""
        position = OptionPosition(
            symbol="AAPL250221C00150000",
            underlying="AAPL",
            quantity=10,
            side="long",
            avg_price=3.50,
            current_price=3.75,
            greeks=GreeksData(delta=0.50, gamma=0.08, theta=-0.05, vega=0.12, rho=0.03),
            expiration_date=datetime(2025, 2, 21),
            strike_price=150.0,
            option_type="call"
        )

        is_valid, message, adjustments = greeks_manager.validate_greeks_order(
            position, "invalid_side", 10
        )

        assert is_valid is False
        assert "invalid" in message.lower()

    def test_calculate_greeks_pnl_extreme_values(self, greeks_manager):
        """Test Greeks P&L with extreme values"""
        greeks = {
            "delta": 1000000.0,  # Very large delta
            "gamma": 0.0,        # Zero gamma
            "theta": -1000.0,    # Large negative theta
            "vega": 5000.0       # Large vega
        }

        pnl_breakdown = greeks_manager.calculate_greeks_pnl(
            greeks, 0.01, 0.01, 1  # Small changes
        )

        # Should handle extreme values without error
        assert isinstance(pnl_breakdown["delta_pnl"], float)
        assert pnl_breakdown["gamma_pnl"] == 0.0  # Zero gamma should give zero P&L
        assert pnl_breakdown["theta_pnl"] < 0  # Negative theta should lose money over time


if __name__ == "__main__":
    pytest.main([__file__])