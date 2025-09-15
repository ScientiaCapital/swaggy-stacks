"""
Comprehensive Unit Tests for Zero-DTE Options Strategy

Tests all functionality of the Zero Days to Expiration strategy including
initialization, signal processing, order creation, exit conditions, and risk validation.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import asdict

from app.strategies.options.zero_dte_strategy import (
    ZeroDTEStrategy,
    ZeroDTEConfig,
    ZeroDTEPosition
)
from app.trading.alpaca_client import AlpacaClient
from app.trading.options_trading import OptionContract, GreeksData, OptionType
from app.rag.strategies.strategy_engines import StrategySignal


class TestZeroDTEConfig:
    """Test Zero-DTE configuration validation"""

    def test_default_config_values(self):
        """Test default configuration values are reasonable"""
        config = ZeroDTEConfig()

        # Delta thresholds
        assert config.short_delta_min == -0.42
        assert config.short_delta_max == -0.38
        assert config.long_delta_min == -0.22
        assert config.long_delta_max == -0.18

        # Risk management
        assert config.profit_target_pct == 50.0
        assert config.stop_loss_multiplier == 2.0

        # Filtering criteria
        assert config.min_open_interest == 500
        assert config.min_spread_width == 2.0
        assert config.max_spread_width == 5.0

        # Monitoring
        assert config.monitoring_interval_minutes == 3

        # IV constraints
        assert config.min_implied_volatility == 0.15
        assert config.max_implied_volatility == 0.80

    def test_custom_config_values(self):
        """Test custom configuration values"""
        config = ZeroDTEConfig(
            short_delta_min=-0.5,
            profit_target_pct=75.0,
            monitoring_interval_minutes=5
        )

        assert config.short_delta_min == -0.5
        assert config.profit_target_pct == 75.0
        assert config.monitoring_interval_minutes == 5


class TestZeroDTEPosition:
    """Test Zero-DTE position data structure"""

    def test_position_creation(self):
        """Test position creation with required fields"""
        entry_time = datetime.now()
        position = ZeroDTEPosition(
            option_symbol="AAPL250117C00150000",
            underlying_symbol="AAPL",
            position_type="SHORT",
            entry_price=2.50,
            entry_delta=-0.40,
            quantity=1,
            profit_target=1.25,
            stop_loss=5.00,
            entry_time=entry_time,
            last_monitoring_time=entry_time
        )

        assert position.option_symbol == "AAPL250117C00150000"
        assert position.underlying_symbol == "AAPL"
        assert position.position_type == "SHORT"
        assert position.entry_price == 2.50
        assert position.entry_delta == -0.40
        assert position.quantity == 1
        assert position.profit_target == 1.25
        assert position.stop_loss == 5.00
        assert position.entry_time == entry_time
        assert position.current_price is None
        assert position.unrealized_pnl is None


class TestZeroDTEStrategy:
    """Comprehensive tests for Zero-DTE strategy implementation"""

    @pytest.fixture
    def mock_alpaca_client(self):
        """Mock Alpaca client fixture"""
        client = MagicMock(spec=AlpacaClient)
        client.get_option_chain = AsyncMock()
        client.get_option_quote = AsyncMock()
        client.execute_order = AsyncMock()
        client.get_positions = AsyncMock()
        return client

    @pytest.fixture
    def default_config(self):
        """Default configuration fixture"""
        return ZeroDTEConfig()

    @pytest.fixture
    def custom_config(self):
        """Custom configuration fixture for testing"""
        return ZeroDTEConfig(
            short_delta_min=-0.45,
            short_delta_max=-0.35,
            profit_target_pct=40.0,
            monitoring_interval_minutes=2
        )

    @pytest.fixture
    def strategy(self, mock_alpaca_client, default_config):
        """Zero-DTE strategy fixture"""
        return ZeroDTEStrategy(mock_alpaca_client, default_config)

    @pytest.fixture
    def mock_option_chain(self):
        """Mock option chain data fixture"""
        return [
            {
                "symbol": "AAPL250117C00150000",
                "underlying_symbol": "AAPL",
                "strike_price": 150.0,
                "option_type": "call",
                "expiration_date": "2025-01-17",
                "bid": 2.40,
                "ask": 2.60,
                "last_price": 2.50,
                "implied_volatility": 0.25,
                "delta": -0.40,
                "gamma": 0.05,
                "theta": -0.15,
                "vega": 0.08,
                "open_interest": 750,
                "volume": 100
            },
            {
                "symbol": "AAPL250117P00148000",
                "underlying_symbol": "AAPL",
                "strike_price": 148.0,
                "option_type": "put",
                "expiration_date": "2025-01-17",
                "bid": 1.80,
                "ask": 2.00,
                "last_price": 1.90,
                "implied_volatility": 0.23,
                "delta": -0.20,
                "gamma": 0.04,
                "theta": -0.12,
                "vega": 0.07,
                "open_interest": 600,
                "volume": 80
            }
        ]

    def test_strategy_initialization_default_config(self, mock_alpaca_client):
        """Test strategy initialization with default config"""
        strategy = ZeroDTEStrategy(mock_alpaca_client)

        assert strategy.alpaca_client == mock_alpaca_client
        assert isinstance(strategy.config, ZeroDTEConfig)
        assert strategy.active_positions == []
        assert strategy.monitoring_task is None

    def test_strategy_initialization_custom_config(self, mock_alpaca_client, custom_config):
        """Test strategy initialization with custom config"""
        strategy = ZeroDTEStrategy(mock_alpaca_client, custom_config)

        assert strategy.config == custom_config
        assert strategy.config.short_delta_min == -0.45
        assert strategy.config.profit_target_pct == 40.0

    @pytest.mark.asyncio
    async def test_analyze_symbol_no_option_chain(self, strategy, mock_alpaca_client):
        """Test analyze_symbol when no option chain is available"""
        mock_alpaca_client.get_option_chain.return_value = []

        result = await strategy.analyze_symbol("AAPL")

        assert result is None
        mock_alpaca_client.get_option_chain.assert_called_once_with(
            "AAPL",
            expiration_date=None
        )

    @pytest.mark.asyncio
    async def test_analyze_symbol_no_valid_options(self, strategy, mock_alpaca_client):
        """Test analyze_symbol when no options meet criteria"""
        # Option with delta outside range
        mock_option_chain = [{
            "symbol": "AAPL250117C00150000",
            "underlying_symbol": "AAPL",
            "strike_price": 150.0,
            "delta": -0.60,  # Outside range
            "bid": 2.40,
            "ask": 2.60,
            "implied_volatility": 0.25,
            "open_interest": 750,
            "expiration_date": "2025-01-17"
        }]

        mock_alpaca_client.get_option_chain.return_value = mock_option_chain

        result = await strategy.analyze_symbol("AAPL")
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_symbol_valid_opportunity(self, strategy, mock_alpaca_client, mock_option_chain):
        """Test analyze_symbol with valid Zero-DTE opportunity"""
        mock_alpaca_client.get_option_chain.return_value = mock_option_chain

        result = await strategy.analyze_symbol("AAPL")

        assert result is not None
        assert isinstance(result, StrategySignal)
        assert result.symbol == "AAPL"
        assert result.action in ["BUY", "SELL"]
        assert result.confidence > 0
        assert "Zero-DTE" in result.reasoning

    @pytest.mark.asyncio
    async def test_analyze_symbol_with_market_data(self, strategy, mock_alpaca_client, mock_option_chain):
        """Test analyze_symbol with market data context"""
        market_data = {
            "current_price": 149.50,
            "volume": 1000000,
            "volatility": 0.25
        }

        mock_alpaca_client.get_option_chain.return_value = mock_option_chain

        result = await strategy.analyze_symbol("AAPL", market_data)

        assert result is not None
        assert result.symbol == "AAPL"

    def test_filter_valid_options_delta_criteria(self, strategy):
        """Test option filtering based on delta criteria"""
        options = [
            {"delta": -0.40, "implied_volatility": 0.25, "open_interest": 750},  # Valid
            {"delta": -0.60, "implied_volatility": 0.25, "open_interest": 750},  # Invalid delta
            {"delta": -0.39, "implied_volatility": 0.25, "open_interest": 750},  # Valid
            {"delta": -0.20, "implied_volatility": 0.25, "open_interest": 750}   # Valid long delta
        ]

        valid_options = strategy._filter_valid_options(options)

        # Should have 3 valid options (delta -0.40, -0.39, -0.20)
        assert len(valid_options) == 3
        assert all(opt["delta"] >= -0.42 for opt in valid_options)

    def test_filter_valid_options_iv_criteria(self, strategy):
        """Test option filtering based on implied volatility criteria"""
        options = [
            {"delta": -0.40, "implied_volatility": 0.25, "open_interest": 750},  # Valid
            {"delta": -0.40, "implied_volatility": 0.10, "open_interest": 750},  # IV too low
            {"delta": -0.40, "implied_volatility": 0.90, "open_interest": 750},  # IV too high
            {"delta": -0.40, "implied_volatility": 0.30, "open_interest": 750}   # Valid
        ]

        valid_options = strategy._filter_valid_options(options)

        # Should have 2 valid options
        assert len(valid_options) == 2
        assert all(0.15 <= opt["implied_volatility"] <= 0.80 for opt in valid_options)

    def test_filter_valid_options_open_interest_criteria(self, strategy):
        """Test option filtering based on open interest criteria"""
        options = [
            {"delta": -0.40, "implied_volatility": 0.25, "open_interest": 750},  # Valid
            {"delta": -0.40, "implied_volatility": 0.25, "open_interest": 200},  # OI too low
            {"delta": -0.40, "implied_volatility": 0.25, "open_interest": 1000}  # Valid
        ]

        valid_options = strategy._filter_valid_options(options)

        # Should have 2 valid options
        assert len(valid_options) == 2
        assert all(opt["open_interest"] >= 500 for opt in valid_options)

    def test_calculate_spread_width(self, strategy):
        """Test spread width calculation"""
        option = {"bid": 2.40, "ask": 2.60}

        spread_width = strategy._calculate_spread_width(option)
        assert spread_width == 0.20

    def test_calculate_spread_width_percentage(self, strategy):
        """Test spread width percentage calculation"""
        option = {"bid": 2.40, "ask": 2.60}

        spread_pct = strategy._calculate_spread_width_percentage(option)
        expected_pct = (0.20 / 2.50) * 100  # (ask-bid) / midpoint * 100
        assert spread_pct == pytest.approx(expected_pct, rel=1e-2)

    @pytest.mark.asyncio
    async def test_execute_signal_buy(self, strategy, mock_alpaca_client):
        """Test execute method with BUY signal"""
        signal = StrategySignal(
            symbol="AAPL",
            action="BUY",
            quantity=1,
            confidence=0.75,
            reasoning="Zero-DTE opportunity",
            metadata={"option_symbol": "AAPL250117C00150000", "entry_price": 2.50}
        )

        mock_alpaca_client.execute_order.return_value = {
            "id": "order_123",
            "status": "filled",
            "filled_price": 2.50
        }

        result = await strategy.execute(signal)

        assert result["status"] == "success"
        assert result["order_id"] == "order_123"
        mock_alpaca_client.execute_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_signal_invalid_action(self, strategy):
        """Test execute method with invalid action"""
        signal = StrategySignal(
            symbol="AAPL",
            action="INVALID",
            quantity=1,
            confidence=0.75,
            reasoning="Test",
            metadata={}
        )

        result = await strategy.execute(signal)

        assert result["status"] == "error"
        assert "Invalid action" in result["error"]

    def test_should_exit_position_profit_target(self, strategy):
        """Test exit condition based on profit target"""
        position = ZeroDTEPosition(
            option_symbol="AAPL250117C00150000",
            underlying_symbol="AAPL",
            position_type="SHORT",
            entry_price=2.50,
            entry_delta=-0.40,
            quantity=1,
            profit_target=1.25,  # 50% profit target
            stop_loss=5.00,
            entry_time=datetime.now(),
            last_monitoring_time=datetime.now(),
            current_price=1.20  # Below profit target
        )

        should_exit, reason = strategy._should_exit_position(position)

        assert should_exit is True
        assert "profit target" in reason.lower()

    def test_should_exit_position_stop_loss(self, strategy):
        """Test exit condition based on stop loss"""
        position = ZeroDTEPosition(
            option_symbol="AAPL250117C00150000",
            underlying_symbol="AAPL",
            position_type="SHORT",
            entry_price=2.50,
            entry_delta=-0.40,
            quantity=1,
            profit_target=1.25,
            stop_loss=5.00,
            entry_time=datetime.now(),
            last_monitoring_time=datetime.now(),
            current_price=5.50  # Above stop loss
        )

        should_exit, reason = strategy._should_exit_position(position)

        assert should_exit is True
        assert "stop loss" in reason.lower()

    def test_should_exit_position_no_exit(self, strategy):
        """Test no exit condition"""
        position = ZeroDTEPosition(
            option_symbol="AAPL250117C00150000",
            underlying_symbol="AAPL",
            position_type="SHORT",
            entry_price=2.50,
            entry_delta=-0.40,
            quantity=1,
            profit_target=1.25,
            stop_loss=5.00,
            entry_time=datetime.now(),
            last_monitoring_time=datetime.now(),
            current_price=2.00  # Between targets
        )

        should_exit, reason = strategy._should_exit_position(position)

        assert should_exit is False
        assert reason == ""

    def test_calculate_profit_target(self, strategy):
        """Test profit target calculation"""
        entry_price = 2.50
        expected_target = entry_price * (1 - strategy.config.profit_target_pct / 100)

        profit_target = strategy._calculate_profit_target(entry_price)

        assert profit_target == pytest.approx(expected_target, rel=1e-2)

    def test_calculate_stop_loss(self, strategy):
        """Test stop loss calculation"""
        entry_price = 2.50
        entry_delta = -0.40
        expected_stop = entry_price + (abs(entry_delta) * strategy.config.stop_loss_multiplier)

        stop_loss = strategy._calculate_stop_loss(entry_price, entry_delta)

        assert stop_loss == pytest.approx(expected_stop, rel=1e-2)

    @pytest.mark.asyncio
    async def test_monitor_positions_empty(self, strategy):
        """Test monitoring with no active positions"""
        await strategy._monitor_positions()
        # Should complete without errors

    @pytest.mark.asyncio
    async def test_monitor_positions_with_exit(self, strategy, mock_alpaca_client):
        """Test monitoring positions that need to exit"""
        position = ZeroDTEPosition(
            option_symbol="AAPL250117C00150000",
            underlying_symbol="AAPL",
            position_type="SHORT",
            entry_price=2.50,
            entry_delta=-0.40,
            quantity=1,
            profit_target=1.25,
            stop_loss=5.00,
            entry_time=datetime.now(),
            last_monitoring_time=datetime.now(),
            current_price=1.20  # Profitable exit
        )

        strategy.active_positions = [position]

        mock_alpaca_client.get_option_quote.return_value = {
            "bid": 1.15,
            "ask": 1.25,
            "last_price": 1.20
        }

        mock_alpaca_client.execute_order.return_value = {
            "id": "exit_order_123",
            "status": "filled"
        }

        await strategy._monitor_positions()

        # Position should be removed after exit
        assert len(strategy.active_positions) == 0
        mock_alpaca_client.execute_order.assert_called_once()

    def test_get_strategy_name(self, strategy):
        """Test strategy name retrieval"""
        assert strategy.get_strategy_name() == "Zero-DTE"

    def test_get_active_positions_count(self, strategy):
        """Test active positions count"""
        assert strategy.get_active_positions_count() == 0

        # Add a position
        position = ZeroDTEPosition(
            option_symbol="AAPL250117C00150000",
            underlying_symbol="AAPL",
            position_type="SHORT",
            entry_price=2.50,
            entry_delta=-0.40,
            quantity=1,
            profit_target=1.25,
            stop_loss=5.00,
            entry_time=datetime.now(),
            last_monitoring_time=datetime.now()
        )
        strategy.active_positions.append(position)

        assert strategy.get_active_positions_count() == 1

    @pytest.mark.asyncio
    async def test_cleanup_positions(self, strategy):
        """Test position cleanup"""
        # Add expired position
        old_position = ZeroDTEPosition(
            option_symbol="AAPL250117C00150000",
            underlying_symbol="AAPL",
            position_type="SHORT",
            entry_price=2.50,
            entry_delta=-0.40,
            quantity=1,
            profit_target=1.25,
            stop_loss=5.00,
            entry_time=datetime.now() - timedelta(days=1),  # Yesterday
            last_monitoring_time=datetime.now() - timedelta(hours=1)
        )

        strategy.active_positions = [old_position]

        await strategy._cleanup_expired_positions()

        # Position should be removed as it's from yesterday (expired)
        assert len(strategy.active_positions) == 0


# Edge Cases and Error Handling Tests
class TestZeroDTEEdgeCases:
    """Test edge cases and error handling for Zero-DTE strategy"""

    @pytest.fixture
    def strategy(self):
        """Strategy fixture for edge case testing"""
        mock_client = MagicMock(spec=AlpacaClient)
        return ZeroDTEStrategy(mock_client)

    @pytest.mark.asyncio
    async def test_analyze_symbol_api_error(self, strategy):
        """Test analyze_symbol when API call fails"""
        strategy.alpaca_client.get_option_chain.side_effect = Exception("API Error")

        result = await strategy.analyze_symbol("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_execute_order_failure(self, strategy):
        """Test execute when order placement fails"""
        signal = StrategySignal(
            symbol="AAPL",
            action="BUY",
            quantity=1,
            confidence=0.75,
            reasoning="Test",
            metadata={"option_symbol": "AAPL250117C00150000"}
        )

        strategy.alpaca_client.execute_order.side_effect = Exception("Order failed")

        result = await strategy.execute(signal)

        assert result["status"] == "error"
        assert "Order failed" in str(result["error"])

    def test_position_with_missing_current_price(self, strategy):
        """Test position monitoring with missing current price"""
        position = ZeroDTEPosition(
            option_symbol="AAPL250117C00150000",
            underlying_symbol="AAPL",
            position_type="SHORT",
            entry_price=2.50,
            entry_delta=-0.40,
            quantity=1,
            profit_target=1.25,
            stop_loss=5.00,
            entry_time=datetime.now(),
            last_monitoring_time=datetime.now(),
            current_price=None  # Missing price
        )

        should_exit, reason = strategy._should_exit_position(position)

        # Should not exit without current price
        assert should_exit is False

    def test_filter_options_empty_list(self, strategy):
        """Test filtering with empty options list"""
        valid_options = strategy._filter_valid_options([])
        assert valid_options == []

    def test_filter_options_missing_fields(self, strategy):
        """Test filtering with options missing required fields"""
        incomplete_options = [
            {"delta": -0.40},  # Missing IV and OI
            {"implied_volatility": 0.25},  # Missing delta and OI
            {}  # Empty option
        ]

        # Should handle gracefully and return empty list
        valid_options = strategy._filter_valid_options(incomplete_options)
        assert valid_options == []


if __name__ == "__main__":
    pytest.main([__file__])