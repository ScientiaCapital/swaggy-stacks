"""
Comprehensive Unit Tests for Wheel Options Strategy

Tests the two-phase Wheel strategy including cash-secured puts,
covered calls, assignment handling, and Bollinger Band integration.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import asdict

from app.strategies.options.wheel_strategy import (
    WheelStrategy,
    WheelConfig,
    WheelPosition,
    WheelPhase,
    BollingerBands
)
from app.trading.alpaca_client import AlpacaClient
from app.trading.options_trading import OptionContract, GreeksData, OptionType
from app.rag.strategies.strategy_engines import StrategySignal


class TestWheelConfig:
    """Test Wheel strategy configuration"""

    def test_default_config_values(self):
        """Test default configuration values"""
        config = WheelConfig()

        # CSP Configuration
        assert config.csp_delta_target == -0.30
        assert config.csp_delta_tolerance == 0.05
        assert config.csp_min_premium == 0.50
        assert config.csp_max_dte == 45
        assert config.csp_min_dte == 7

        # CC Configuration
        assert config.cc_delta_target == 0.30
        assert config.cc_delta_tolerance == 0.05
        assert config.cc_min_premium == 0.30
        assert config.cc_max_dte == 30
        assert config.cc_min_dte == 7

        # Position sizing
        assert config.max_wheel_positions == 3
        assert config.capital_per_wheel == 10000.0

        # Risk management
        assert config.profit_target_pct == 25.0
        assert config.loss_limit_pct == 200.0
        assert config.assignment_handling == "accept"

        # Bollinger Bands
        assert config.bb_period == 20
        assert config.bb_std_dev == 2.0

    def test_custom_config_values(self):
        """Test custom configuration values"""
        config = WheelConfig(
            csp_delta_target=-0.25,
            cc_delta_target=0.35,
            max_wheel_positions=5,
            profit_target_pct=30.0
        )

        assert config.csp_delta_target == -0.25
        assert config.cc_delta_target == 0.35
        assert config.max_wheel_positions == 5
        assert config.profit_target_pct == 30.0


class TestBollingerBands:
    """Test Bollinger Bands data structure"""

    def test_bollinger_bands_creation(self):
        """Test Bollinger Bands creation"""
        bb = BollingerBands(
            upper_band=155.0,
            middle_band=150.0,
            lower_band=145.0,
            current_price=148.0,
            position_relative_to_bands="below_middle"
        )

        assert bb.upper_band == 155.0
        assert bb.middle_band == 150.0
        assert bb.lower_band == 145.0
        assert bb.current_price == 148.0
        assert bb.position_relative_to_bands == "below_middle"

    def test_bollinger_bands_width_calculation(self):
        """Test Bollinger Bands width calculation"""
        bb = BollingerBands(
            upper_band=155.0,
            middle_band=150.0,
            lower_band=145.0,
            current_price=148.0,
            position_relative_to_bands="below_middle"
        )

        width = bb.calculate_width()
        expected_width = (155.0 - 145.0) / 150.0 * 100
        assert width == pytest.approx(expected_width, rel=1e-2)


class TestWheelPosition:
    """Test Wheel position data structure"""

    def test_csp_position_creation(self):
        """Test CSP position creation"""
        entry_time = datetime.now()
        position = WheelPosition(
            underlying_symbol="AAPL",
            phase=WheelPhase.CSP,
            option_symbol="AAPL250221P00150000",
            strike_price=150.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=2.50,
            quantity=1,
            entry_time=entry_time,
            target_profit=1.875,  # 25% profit target
            stop_loss=5.00
        )

        assert position.underlying_symbol == "AAPL"
        assert position.phase == WheelPhase.CSP
        assert position.option_symbol == "AAPL250221P00150000"
        assert position.strike_price == 150.0
        assert position.entry_price == 2.50
        assert position.quantity == 1
        assert position.target_profit == 1.875
        assert position.stock_position is None

    def test_cc_position_creation(self):
        """Test CC position creation"""
        entry_time = datetime.now()
        position = WheelPosition(
            underlying_symbol="AAPL",
            phase=WheelPhase.CC,
            option_symbol="AAPL250221C00155000",
            strike_price=155.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=1.50,
            quantity=1,
            entry_time=entry_time,
            target_profit=1.125,
            stop_loss=3.00,
            stock_position={"shares": 100, "avg_cost": 150.0}
        )

        assert position.phase == WheelPhase.CC
        assert position.stock_position["shares"] == 100
        assert position.stock_position["avg_cost"] == 150.0


class TestWheelStrategy:
    """Comprehensive tests for Wheel strategy implementation"""

    @pytest.fixture
    def mock_alpaca_client(self):
        """Mock Alpaca client fixture"""
        client = MagicMock(spec=AlpacaClient)
        client.get_option_chain = AsyncMock()
        client.get_option_quote = AsyncMock()
        client.execute_order = AsyncMock()
        client.get_positions = AsyncMock()
        client.get_account = AsyncMock()
        client.get_market_data = AsyncMock()
        return client

    @pytest.fixture
    def default_config(self):
        """Default configuration fixture"""
        return WheelConfig()

    @pytest.fixture
    def strategy(self, mock_alpaca_client, default_config):
        """Wheel strategy fixture"""
        return WheelStrategy(mock_alpaca_client, default_config)

    @pytest.fixture
    def mock_price_data(self):
        """Mock historical price data for Bollinger Bands"""
        base_price = 150.0
        return [
            {"close": base_price + i * 0.5, "timestamp": datetime.now() - timedelta(days=20-i)}
            for i in range(20)
        ]

    def test_strategy_initialization(self, mock_alpaca_client, default_config):
        """Test strategy initialization"""
        strategy = WheelStrategy(mock_alpaca_client, default_config)

        assert strategy.alpaca_client == mock_alpaca_client
        assert strategy.config == default_config
        assert strategy.active_wheels == []

    @pytest.mark.asyncio
    async def test_analyze_symbol_no_existing_position(self, strategy, mock_alpaca_client, mock_price_data):
        """Test analyze_symbol for new wheel opportunity"""
        # Mock Bollinger Bands calculation
        mock_alpaca_client.get_market_data.return_value = mock_price_data

        # Mock option chain for CSP
        mock_option_chain = [
            {
                "symbol": "AAPL250221P00150000",
                "underlying_symbol": "AAPL",
                "strike_price": 150.0,
                "option_type": "put",
                "expiration_date": "2025-02-21",
                "bid": 2.40,
                "ask": 2.60,
                "last_price": 2.50,
                "implied_volatility": 0.25,
                "delta": -0.30,
                "gamma": 0.05,
                "theta": -0.05,
                "vega": 0.08,
                "open_interest": 1000,
                "volume": 150
            }
        ]

        mock_alpaca_client.get_option_chain.return_value = mock_option_chain

        result = await strategy.analyze_symbol("AAPL")

        assert result is not None
        assert isinstance(result, StrategySignal)
        assert result.symbol == "AAPL"
        assert result.action == "SELL"  # Selling CSP
        assert "Wheel CSP" in result.reasoning

    @pytest.mark.asyncio
    async def test_analyze_symbol_existing_csp_to_cc(self, strategy, mock_alpaca_client):
        """Test analyze_symbol transitioning from CSP to CC after assignment"""
        # Add existing CSP position that was assigned
        existing_position = WheelPosition(
            underlying_symbol="AAPL",
            phase=WheelPhase.CSP,
            option_symbol="AAPL250221P00150000",
            strike_price=150.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=2.50,
            quantity=1,
            entry_time=datetime.now() - timedelta(days=10),
            target_profit=1.875,
            stop_loss=5.00,
            stock_position={"shares": 100, "avg_cost": 150.0}  # Assigned
        )

        strategy.active_wheels = [existing_position]

        # Mock CC option chain
        mock_cc_chain = [
            {
                "symbol": "AAPL250221C00155000",
                "underlying_symbol": "AAPL",
                "strike_price": 155.0,
                "option_type": "call",
                "expiration_date": "2025-02-21",
                "bid": 1.40,
                "ask": 1.60,
                "last_price": 1.50,
                "delta": 0.30,
                "open_interest": 800
            }
        ]

        mock_alpaca_client.get_option_chain.return_value = mock_cc_chain

        result = await strategy.analyze_symbol("AAPL")

        assert result is not None
        assert result.action == "SELL"  # Selling CC
        assert "Wheel CC" in result.reasoning

    @pytest.mark.asyncio
    async def test_calculate_bollinger_bands(self, strategy, mock_alpaca_client, mock_price_data):
        """Test Bollinger Bands calculation"""
        mock_alpaca_client.get_market_data.return_value = mock_price_data

        bb = await strategy._calculate_bollinger_bands("AAPL")

        assert bb is not None
        assert isinstance(bb, BollingerBands)
        assert bb.upper_band > bb.middle_band > bb.lower_band
        assert bb.current_price > 0

    @pytest.mark.asyncio
    async def test_calculate_bollinger_bands_insufficient_data(self, strategy, mock_alpaca_client):
        """Test Bollinger Bands with insufficient data"""
        # Only 5 data points instead of required 20
        insufficient_data = [
            {"close": 150.0 + i, "timestamp": datetime.now() - timedelta(days=5-i)}
            for i in range(5)
        ]

        mock_alpaca_client.get_market_data.return_value = insufficient_data

        bb = await strategy._calculate_bollinger_bands("AAPL")

        assert bb is None

    def test_find_best_csp_opportunity_valid(self, strategy):
        """Test finding best CSP opportunity"""
        option_chain = [
            {
                "symbol": "AAPL250221P00150000",
                "strike_price": 150.0,
                "delta": -0.30,
                "bid": 2.40,
                "ask": 2.60,
                "open_interest": 1000,
                "expiration_date": "2025-02-21"
            },
            {
                "symbol": "AAPL250221P00148000",
                "strike_price": 148.0,
                "delta": -0.25,
                "bid": 2.00,
                "ask": 2.20,
                "open_interest": 800,
                "expiration_date": "2025-02-21"
            }
        ]

        bb = BollingerBands(
            upper_band=155.0,
            middle_band=150.0,
            lower_band=145.0,
            current_price=149.0,
            position_relative_to_bands="near_middle"
        )

        best_option = strategy._find_best_csp_opportunity("AAPL", option_chain, bb)

        assert best_option is not None
        assert best_option["symbol"] == "AAPL250221P00150000"

    def test_find_best_csp_opportunity_no_valid(self, strategy):
        """Test finding CSP when no valid opportunities"""
        option_chain = [
            {
                "symbol": "AAPL250221P00150000",
                "strike_price": 150.0,
                "delta": -0.15,  # Delta too low
                "bid": 2.40,
                "ask": 2.60,
                "open_interest": 1000,
                "expiration_date": "2025-02-21"
            }
        ]

        bb = BollingerBands(
            upper_band=155.0,
            middle_band=150.0,
            lower_band=145.0,
            current_price=149.0,
            position_relative_to_bands="near_middle"
        )

        best_option = strategy._find_best_csp_opportunity("AAPL", option_chain, bb)

        assert best_option is None

    def test_find_best_cc_opportunity_valid(self, strategy):
        """Test finding best CC opportunity"""
        option_chain = [
            {
                "symbol": "AAPL250221C00155000",
                "strike_price": 155.0,
                "delta": 0.30,
                "bid": 1.40,
                "ask": 1.60,
                "open_interest": 800,
                "expiration_date": "2025-02-21"
            },
            {
                "symbol": "AAPL250221C00157000",
                "strike_price": 157.0,
                "delta": 0.25,
                "bid": 1.00,
                "ask": 1.20,
                "open_interest": 600,
                "expiration_date": "2025-02-21"
            }
        ]

        stock_position = {"shares": 100, "avg_cost": 150.0}

        best_option = strategy._find_best_cc_opportunity("AAPL", option_chain, stock_position)

        assert best_option is not None
        assert best_option["symbol"] == "AAPL250221C00155000"

    def test_find_best_cc_opportunity_below_cost_basis(self, strategy):
        """Test CC opportunity filtering based on cost basis"""
        option_chain = [
            {
                "symbol": "AAPL250221C00148000",
                "strike_price": 148.0,  # Below cost basis
                "delta": 0.35,
                "bid": 2.40,
                "ask": 2.60,
                "open_interest": 800,
                "expiration_date": "2025-02-21"
            }
        ]

        stock_position = {"shares": 100, "avg_cost": 150.0}

        best_option = strategy._find_best_cc_opportunity("AAPL", option_chain, stock_position)

        # Should reject option below cost basis
        assert best_option is None

    @pytest.mark.asyncio
    async def test_execute_csp_signal(self, strategy, mock_alpaca_client):
        """Test executing CSP signal"""
        signal = StrategySignal(
            symbol="AAPL",
            action="SELL",
            quantity=1,
            confidence=0.75,
            reasoning="Wheel CSP opportunity",
            metadata={
                "option_symbol": "AAPL250221P00150000",
                "option_type": "put",
                "strike_price": 150.0,
                "premium": 2.50,
                "phase": "CSP"
            }
        )

        mock_alpaca_client.execute_order.return_value = {
            "id": "order_123",
            "status": "filled",
            "filled_price": 2.50
        }

        result = await strategy.execute(signal)

        assert result["status"] == "success"
        assert result["order_id"] == "order_123"
        assert len(strategy.active_wheels) == 1
        assert strategy.active_wheels[0].phase == WheelPhase.CSP

    @pytest.mark.asyncio
    async def test_execute_cc_signal(self, strategy, mock_alpaca_client):
        """Test executing CC signal"""
        signal = StrategySignal(
            symbol="AAPL",
            action="SELL",
            quantity=1,
            confidence=0.75,
            reasoning="Wheel CC opportunity",
            metadata={
                "option_symbol": "AAPL250221C00155000",
                "option_type": "call",
                "strike_price": 155.0,
                "premium": 1.50,
                "phase": "CC"
            }
        )

        mock_alpaca_client.execute_order.return_value = {
            "id": "order_456",
            "status": "filled",
            "filled_price": 1.50
        }

        result = await strategy.execute(signal)

        assert result["status"] == "success"
        assert len(strategy.active_wheels) == 1
        assert strategy.active_wheels[0].phase == WheelPhase.CC

    def test_should_close_position_profit_target(self, strategy):
        """Test closing position at profit target"""
        position = WheelPosition(
            underlying_symbol="AAPL",
            phase=WheelPhase.CSP,
            option_symbol="AAPL250221P00150000",
            strike_price=150.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=2.50,
            quantity=1,
            entry_time=datetime.now() - timedelta(days=5),
            target_profit=1.875,  # 25% profit target
            stop_loss=5.00,
            current_price=1.80  # Below profit target
        )

        should_close, reason = strategy._should_close_position(position)

        assert should_close is True
        assert "profit target" in reason.lower()

    def test_should_close_position_stop_loss(self, strategy):
        """Test closing position at stop loss"""
        position = WheelPosition(
            underlying_symbol="AAPL",
            phase=WheelPhase.CSP,
            option_symbol="AAPL250221P00150000",
            strike_price=150.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=2.50,
            quantity=1,
            entry_time=datetime.now() - timedelta(days=5),
            target_profit=1.875,
            stop_loss=5.00,
            current_price=6.00  # Above stop loss
        )

        should_close, reason = strategy._should_close_position(position)

        assert should_close is True
        assert "stop loss" in reason.lower()

    def test_should_close_position_near_expiration(self, strategy):
        """Test closing position near expiration"""
        position = WheelPosition(
            underlying_symbol="AAPL",
            phase=WheelPhase.CSP,
            option_symbol="AAPL250221P00150000",
            strike_price=150.0,
            expiration_date=datetime.now() + timedelta(days=1),  # Tomorrow
            entry_price=2.50,
            quantity=1,
            entry_time=datetime.now() - timedelta(days=20),
            target_profit=1.875,
            stop_loss=5.00,
            current_price=2.00
        )

        should_close, reason = strategy._should_close_position(position)

        assert should_close is True
        assert "expiration" in reason.lower()

    def test_should_close_position_no_close(self, strategy):
        """Test position that should not close"""
        position = WheelPosition(
            underlying_symbol="AAPL",
            phase=WheelPhase.CSP,
            option_symbol="AAPL250221P00150000",
            strike_price=150.0,
            expiration_date=datetime.now() + timedelta(days=10),
            entry_price=2.50,
            quantity=1,
            entry_time=datetime.now() - timedelta(days=5),
            target_profit=1.875,
            stop_loss=5.00,
            current_price=2.20  # Between targets
        )

        should_close, reason = strategy._should_close_position(position)

        assert should_close is False
        assert reason == ""

    def test_calculate_days_to_expiration(self, strategy):
        """Test days to expiration calculation"""
        future_date = datetime.now() + timedelta(days=15)
        dte = strategy._calculate_days_to_expiration(future_date)

        assert dte == 15

    def test_calculate_days_to_expiration_past(self, strategy):
        """Test DTE for past expiration"""
        past_date = datetime.now() - timedelta(days=5)
        dte = strategy._calculate_days_to_expiration(past_date)

        assert dte == -5

    def test_get_strategy_name(self, strategy):
        """Test strategy name"""
        assert strategy.get_strategy_name() == "Wheel"

    def test_get_active_wheels_count(self, strategy):
        """Test active wheels count"""
        assert strategy.get_active_wheels_count() == 0

        # Add position
        position = WheelPosition(
            underlying_symbol="AAPL",
            phase=WheelPhase.CSP,
            option_symbol="AAPL250221P00150000",
            strike_price=150.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=2.50,
            quantity=1,
            entry_time=datetime.now(),
            target_profit=1.875,
            stop_loss=5.00
        )
        strategy.active_wheels.append(position)

        assert strategy.get_active_wheels_count() == 1

    @pytest.mark.asyncio
    async def test_monitor_wheels_no_positions(self, strategy):
        """Test monitoring with no active wheels"""
        await strategy._monitor_wheels()
        # Should complete without errors

    @pytest.mark.asyncio
    async def test_handle_assignment_csp(self, strategy, mock_alpaca_client):
        """Test handling CSP assignment"""
        position = WheelPosition(
            underlying_symbol="AAPL",
            phase=WheelPhase.CSP,
            option_symbol="AAPL250221P00150000",
            strike_price=150.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=2.50,
            quantity=1,
            entry_time=datetime.now() - timedelta(days=10),
            target_profit=1.875,
            stop_loss=5.00
        )

        strategy.active_wheels = [position]

        # Mock position query showing assignment
        mock_alpaca_client.get_positions.return_value = [
            {"symbol": "AAPL", "qty": 100, "avg_entry_price": 150.0}
        ]

        await strategy._check_for_assignments()

        # Position should be updated with stock position
        assert strategy.active_wheels[0].stock_position is not None
        assert strategy.active_wheels[0].stock_position["shares"] == 100


# Edge Cases and Error Handling Tests
class TestWheelEdgeCases:
    """Test edge cases and error handling for Wheel strategy"""

    @pytest.fixture
    def strategy(self):
        """Strategy fixture for edge case testing"""
        mock_client = MagicMock(spec=AlpacaClient)
        return WheelStrategy(mock_client)

    @pytest.mark.asyncio
    async def test_analyze_symbol_api_error(self, strategy):
        """Test analyze_symbol when API calls fail"""
        strategy.alpaca_client.get_market_data.side_effect = Exception("API Error")

        result = await strategy.analyze_symbol("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_calculate_bollinger_bands_api_error(self, strategy):
        """Test Bollinger Bands calculation with API error"""
        strategy.alpaca_client.get_market_data.side_effect = Exception("Data Error")

        bb = await strategy._calculate_bollinger_bands("AAPL")

        assert bb is None

    def test_filter_valid_options_empty_chain(self, strategy):
        """Test option filtering with empty chain"""
        valid_options = strategy._filter_valid_csp_options([], None)
        assert valid_options == []

    def test_position_max_wheels_limit(self, strategy):
        """Test maximum wheels limit"""
        strategy.config.max_wheel_positions = 1

        # Add one wheel position
        position = WheelPosition(
            underlying_symbol="AAPL",
            phase=WheelPhase.CSP,
            option_symbol="AAPL250221P00150000",
            strike_price=150.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=2.50,
            quantity=1,
            entry_time=datetime.now(),
            target_profit=1.875,
            stop_loss=5.00
        )
        strategy.active_wheels.append(position)

        # Should reject new position due to limit
        can_add = strategy._can_add_new_wheel("MSFT")
        assert can_add is False

    def test_calculate_premium_too_low(self, strategy):
        """Test option with premium below minimum"""
        option = {"bid": 0.30, "ask": 0.50}  # Mid-price 0.40, below 0.50 minimum

        is_valid = strategy._is_premium_adequate(option, is_csp=True)
        assert is_valid is False

    def test_bollinger_bands_edge_values(self, strategy):
        """Test Bollinger Bands with edge values"""
        bb = BollingerBands(
            upper_band=150.0,
            middle_band=150.0,  # No volatility
            lower_band=150.0,
            current_price=150.0,
            position_relative_to_bands="at_middle"
        )

        width = bb.calculate_width()
        assert width == 0.0  # No width when bands collapse


if __name__ == "__main__":
    pytest.main([__file__])