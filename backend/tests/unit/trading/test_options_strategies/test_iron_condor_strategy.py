"""
Comprehensive Unit Tests for Iron Condor Options Strategy

Tests the four-leg Iron Condor strategy including bull put spread,
bear call spread, optimal strike selection, and risk management.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import asdict

from app.strategies.options.iron_condor_strategy import (
    IronCondorStrategy,
    IronCondorConfig,
    IronCondorPosition,
    IronCondorLeg,
    CondorType
)
from app.trading.alpaca_client import AlpacaClient
from app.trading.options_trading import OptionContract, GreeksData, OptionType
from app.rag.strategies.strategy_engines import StrategySignal


class TestIronCondorConfig:
    """Test Iron Condor configuration validation"""

    def test_default_config_values(self):
        """Test default configuration values"""
        config = IronCondorConfig()

        # Strike selection
        assert config.wing_width == 5.0
        assert config.max_width_difference == 1.0
        assert config.target_delta_put == -0.15
        assert config.target_delta_call == 0.15
        assert config.delta_tolerance == 0.05

        # Risk management
        assert config.max_loss_multiplier == 3.0
        assert config.profit_target_pct == 25.0
        assert config.stop_loss_pct == 200.0

        # Filtering criteria
        assert config.min_net_credit == 1.0
        assert config.min_open_interest == 100
        assert config.max_bid_ask_spread_pct == 20.0

        # Time management
        assert config.min_dte == 14
        assert config.max_dte == 45
        assert config.close_dte_threshold == 7

        # Position limits
        assert config.max_condors == 5
        assert config.max_same_expiration == 2

    def test_custom_config_values(self):
        """Test custom configuration values"""
        config = IronCondorConfig(
            wing_width=10.0,
            profit_target_pct=30.0,
            max_condors=3,
            target_delta_put=-0.20
        )

        assert config.wing_width == 10.0
        assert config.profit_target_pct == 30.0
        assert config.max_condors == 3
        assert config.target_delta_put == -0.20


class TestIronCondorLeg:
    """Test Iron Condor leg data structure"""

    def test_leg_creation(self):
        """Test creating an Iron Condor leg"""
        leg = IronCondorLeg(
            option_symbol="AAPL250221P00150000",
            leg_type="short_put",
            strike_price=150.0,
            action="sell",
            premium=2.50,
            delta=-0.15,
            quantity=1
        )

        assert leg.option_symbol == "AAPL250221P00150000"
        assert leg.leg_type == "short_put"
        assert leg.strike_price == 150.0
        assert leg.action == "sell"
        assert leg.premium == 2.50
        assert leg.delta == -0.15
        assert leg.quantity == 1


class TestIronCondorPosition:
    """Test Iron Condor position data structure"""

    def test_position_creation(self):
        """Test complete Iron Condor position creation"""
        legs = [
            IronCondorLeg("AAPL250221P00145000", "long_put", 145.0, "buy", 1.50, -0.08, 1),
            IronCondorLeg("AAPL250221P00150000", "short_put", 150.0, "sell", 2.50, -0.15, 1),
            IronCondorLeg("AAPL250221C00155000", "short_call", 155.0, "sell", 2.00, 0.15, 1),
            IronCondorLeg("AAPL250221C00160000", "long_call", 160.0, "buy", 1.20, 0.08, 1)
        ]

        entry_time = datetime.now()
        position = IronCondorPosition(
            underlying_symbol="AAPL",
            expiration_date=datetime(2025, 2, 21),
            legs=legs,
            net_credit=1.80,  # 2.50 + 2.00 - 1.50 - 1.20
            max_profit=1.80,
            max_loss=3.20,  # Wing width (5) - net credit (1.80)
            entry_time=entry_time,
            profit_target=1.35,  # 75% of max profit
            stop_loss=5.60,  # 200% of net credit
            condor_type=CondorType.BALANCED
        )

        assert position.underlying_symbol == "AAPL"
        assert len(position.legs) == 4
        assert position.net_credit == 1.80
        assert position.max_profit == 1.80
        assert position.max_loss == 3.20
        assert position.condor_type == CondorType.BALANCED

    def test_position_greeks_calculation(self):
        """Test position-level Greeks calculation"""
        legs = [
            IronCondorLeg("AAPL250221P00145000", "long_put", 145.0, "buy", 1.50, -0.08, 1),
            IronCondorLeg("AAPL250221P00150000", "short_put", 150.0, "sell", 2.50, -0.15, 1),
            IronCondorLeg("AAPL250221C00155000", "short_call", 155.0, "sell", 2.00, 0.15, 1),
            IronCondorLeg("AAPL250221C00160000", "long_call", 160.0, "buy", 1.20, 0.08, 1)
        ]

        position = IronCondorPosition(
            underlying_symbol="AAPL",
            expiration_date=datetime(2025, 2, 21),
            legs=legs,
            net_credit=1.80,
            max_profit=1.80,
            max_loss=3.20,
            entry_time=datetime.now(),
            profit_target=1.35,
            stop_loss=5.60,
            condor_type=CondorType.BALANCED
        )

        # Calculate position delta (should be near zero for balanced condor)
        position_delta = position.calculate_position_delta()
        expected_delta = (-0.08 * -1) + (-0.15 * -1) + (0.15 * -1) + (0.08 * 1)
        assert position_delta == pytest.approx(expected_delta, rel=1e-2)


class TestIronCondorStrategy:
    """Comprehensive tests for Iron Condor strategy implementation"""

    @pytest.fixture
    def mock_alpaca_client(self):
        """Mock Alpaca client fixture"""
        client = MagicMock(spec=AlpacaClient)
        client.get_option_chain = AsyncMock()
        client.get_option_quote = AsyncMock()
        client.execute_multi_leg_order = AsyncMock()
        client.get_positions = AsyncMock()
        client.get_market_data = AsyncMock()
        return client

    @pytest.fixture
    def default_config(self):
        """Default configuration fixture"""
        return IronCondorConfig()

    @pytest.fixture
    def strategy(self, mock_alpaca_client, default_config):
        """Iron Condor strategy fixture"""
        return IronCondorStrategy(mock_alpaca_client, default_config)

    @pytest.fixture
    def mock_option_chain(self):
        """Mock comprehensive option chain"""
        return [
            # Puts
            {"symbol": "AAPL250221P00140000", "strike_price": 140.0, "option_type": "put",
             "bid": 0.80, "ask": 1.00, "delta": -0.05, "open_interest": 500, "expiration_date": "2025-02-21"},
            {"symbol": "AAPL250221P00145000", "strike_price": 145.0, "option_type": "put",
             "bid": 1.40, "ask": 1.60, "delta": -0.08, "open_interest": 800, "expiration_date": "2025-02-21"},
            {"symbol": "AAPL250221P00150000", "strike_price": 150.0, "option_type": "put",
             "bid": 2.40, "ask": 2.60, "delta": -0.15, "open_interest": 1200, "expiration_date": "2025-02-21"},
            {"symbol": "AAPL250221P00155000", "strike_price": 155.0, "option_type": "put",
             "bid": 4.20, "ask": 4.50, "delta": -0.25, "open_interest": 900, "expiration_date": "2025-02-21"},

            # Calls
            {"symbol": "AAPL250221C00155000", "strike_price": 155.0, "option_type": "call",
             "bid": 1.80, "ask": 2.00, "delta": 0.15, "open_interest": 1000, "expiration_date": "2025-02-21"},
            {"symbol": "AAPL250221C00160000", "strike_price": 160.0, "option_type": "call",
             "bid": 1.10, "ask": 1.30, "delta": 0.08, "open_interest": 700, "expiration_date": "2025-02-21"},
            {"symbol": "AAPL250221C00165000", "strike_price": 165.0, "option_type": "call",
             "bid": 0.60, "ask": 0.80, "delta": 0.05, "open_interest": 400, "expiration_date": "2025-02-21"},
            {"symbol": "AAPL250221C00170000", "strike_price": 170.0, "option_type": "call",
             "bid": 0.30, "ask": 0.50, "delta": 0.03, "open_interest": 200, "expiration_date": "2025-02-21"}
        ]

    def test_strategy_initialization(self, mock_alpaca_client, default_config):
        """Test strategy initialization"""
        strategy = IronCondorStrategy(mock_alpaca_client, default_config)

        assert strategy.alpaca_client == mock_alpaca_client
        assert strategy.config == default_config
        assert strategy.active_condors == []

    @pytest.mark.asyncio
    async def test_analyze_symbol_valid_opportunity(self, strategy, mock_alpaca_client, mock_option_chain):
        """Test analyze_symbol with valid Iron Condor opportunity"""
        mock_alpaca_client.get_option_chain.return_value = mock_option_chain

        # Mock current price around middle of range
        market_data = {"current_price": 152.5, "implied_volatility": 0.25}

        result = await strategy.analyze_symbol("AAPL", market_data)

        assert result is not None
        assert isinstance(result, StrategySignal)
        assert result.symbol == "AAPL"
        assert result.action == "SELL"  # Iron Condor is net credit
        assert "Iron Condor" in result.reasoning

    @pytest.mark.asyncio
    async def test_analyze_symbol_no_suitable_strikes(self, strategy, mock_alpaca_client):
        """Test analyze_symbol when no suitable strikes available"""
        # Limited option chain with no good strikes
        limited_chain = [
            {"symbol": "AAPL250221P00150000", "strike_price": 150.0, "option_type": "put",
             "bid": 2.40, "ask": 2.60, "delta": -0.30, "open_interest": 1200, "expiration_date": "2025-02-21"},
            {"symbol": "AAPL250221C00155000", "strike_price": 155.0, "option_type": "call",
             "bid": 1.80, "ask": 2.00, "delta": 0.30, "open_interest": 1000, "expiration_date": "2025-02-21"}
        ]

        mock_alpaca_client.get_option_chain.return_value = limited_chain

        result = await strategy.analyze_symbol("AAPL")

        assert result is None

    def test_find_optimal_strikes_balanced(self, strategy, mock_option_chain):
        """Test finding optimal strikes for balanced condor"""
        current_price = 152.5

        strikes = strategy._find_optimal_strikes("AAPL", mock_option_chain, current_price)

        assert strikes is not None
        assert len(strikes) == 4
        assert strikes["long_put"]["strike_price"] < strikes["short_put"]["strike_price"]
        assert strikes["short_put"]["strike_price"] < current_price < strikes["short_call"]["strike_price"]
        assert strikes["short_call"]["strike_price"] < strikes["long_call"]["strike_price"]

    def test_find_optimal_strikes_insufficient_options(self, strategy):
        """Test strike selection with insufficient options"""
        limited_options = [
            {"symbol": "AAPL250221P00150000", "strike_price": 150.0, "option_type": "put",
             "delta": -0.15, "open_interest": 100}
        ]

        strikes = strategy._find_optimal_strikes("AAPL", limited_options, 152.5)

        assert strikes is None

    def test_validate_strike_combination(self, strategy):
        """Test strike combination validation"""
        valid_strikes = {
            "long_put": {"strike_price": 145.0, "bid": 1.40, "ask": 1.60},
            "short_put": {"strike_price": 150.0, "bid": 2.40, "ask": 2.60},
            "short_call": {"strike_price": 155.0, "bid": 1.80, "ask": 2.00},
            "long_call": {"strike_price": 160.0, "bid": 1.10, "ask": 1.30}
        }

        is_valid = strategy._validate_strike_combination(valid_strikes)
        assert is_valid is True

    def test_validate_strike_combination_invalid_ordering(self, strategy):
        """Test strike validation with invalid ordering"""
        invalid_strikes = {
            "long_put": {"strike_price": 155.0, "bid": 1.40, "ask": 1.60},  # Wrong order
            "short_put": {"strike_price": 150.0, "bid": 2.40, "ask": 2.60},
            "short_call": {"strike_price": 155.0, "bid": 1.80, "ask": 2.00},
            "long_call": {"strike_price": 160.0, "bid": 1.10, "ask": 1.30}
        }

        is_valid = strategy._validate_strike_combination(invalid_strikes)
        assert is_valid is False

    def test_calculate_net_credit(self, strategy):
        """Test net credit calculation"""
        strikes = {
            "long_put": {"strike_price": 145.0, "bid": 1.40, "ask": 1.60},
            "short_put": {"strike_price": 150.0, "bid": 2.40, "ask": 2.60},
            "short_call": {"strike_price": 155.0, "bid": 1.80, "ask": 2.00},
            "long_call": {"strike_price": 160.0, "bid": 1.10, "ask": 1.30}
        }

        net_credit = strategy._calculate_net_credit(strikes)
        expected_credit = 2.50 + 1.90 - 1.50 - 1.20  # Mid-prices
        assert net_credit == pytest.approx(expected_credit, rel=1e-2)

    def test_calculate_max_loss(self, strategy):
        """Test maximum loss calculation"""
        strikes = {
            "long_put": {"strike_price": 145.0},
            "short_put": {"strike_price": 150.0},
            "short_call": {"strike_price": 155.0},
            "long_call": {"strike_price": 160.0}
        }
        net_credit = 1.70

        max_loss = strategy._calculate_max_loss(strikes, net_credit)
        expected_loss = 5.0 - 1.70  # Wing width - net credit
        assert max_loss == pytest.approx(expected_loss, rel=1e-2)

    def test_calculate_breakeven_points(self, strategy):
        """Test breakeven point calculation"""
        strikes = {
            "short_put": {"strike_price": 150.0},
            "short_call": {"strike_price": 155.0}
        }
        net_credit = 1.70

        breakevens = strategy._calculate_breakeven_points(strikes, net_credit)

        assert len(breakevens) == 2
        assert breakevens["lower"] == 150.0 - 1.70
        assert breakevens["upper"] == 155.0 + 1.70

    @pytest.mark.asyncio
    async def test_execute_iron_condor(self, strategy, mock_alpaca_client):
        """Test executing Iron Condor signal"""
        signal = StrategySignal(
            symbol="AAPL",
            action="SELL",
            quantity=1,
            confidence=0.80,
            reasoning="Iron Condor opportunity",
            metadata={
                "condor_legs": [
                    {"option_symbol": "AAPL250221P00145000", "action": "buy", "quantity": 1},
                    {"option_symbol": "AAPL250221P00150000", "action": "sell", "quantity": 1},
                    {"option_symbol": "AAPL250221C00155000", "action": "sell", "quantity": 1},
                    {"option_symbol": "AAPL250221C00160000", "action": "buy", "quantity": 1}
                ],
                "net_credit": 1.70,
                "max_loss": 3.30
            }
        )

        mock_alpaca_client.execute_multi_leg_order.return_value = {
            "id": "condor_order_123",
            "status": "filled",
            "legs": [
                {"symbol": "AAPL250221P00145000", "filled_price": 1.50},
                {"symbol": "AAPL250221P00150000", "filled_price": 2.50},
                {"symbol": "AAPL250221C00155000", "filled_price": 1.90},
                {"symbol": "AAPL250221C00160000", "filled_price": 1.20}
            ]
        }

        result = await strategy.execute(signal)

        assert result["status"] == "success"
        assert result["order_id"] == "condor_order_123"
        assert len(strategy.active_condors) == 1

    def test_should_close_position_profit_target(self, strategy):
        """Test closing position at profit target"""
        position = IronCondorPosition(
            underlying_symbol="AAPL",
            expiration_date=datetime(2025, 2, 21),
            legs=[],  # Simplified for test
            net_credit=1.80,
            max_profit=1.80,
            max_loss=3.20,
            entry_time=datetime.now() - timedelta(days=5),
            profit_target=1.35,  # 75% target
            stop_loss=5.60,
            condor_type=CondorType.BALANCED,
            current_value=0.40  # Below profit target
        )

        should_close, reason = strategy._should_close_position(position)

        assert should_close is True
        assert "profit target" in reason.lower()

    def test_should_close_position_stop_loss(self, strategy):
        """Test closing position at stop loss"""
        position = IronCondorPosition(
            underlying_symbol="AAPL",
            expiration_date=datetime(2025, 2, 21),
            legs=[],
            net_credit=1.80,
            max_profit=1.80,
            max_loss=3.20,
            entry_time=datetime.now() - timedelta(days=5),
            profit_target=1.35,
            stop_loss=5.60,
            condor_type=CondorType.BALANCED,
            current_value=6.00  # Above stop loss
        )

        should_close, reason = strategy._should_close_position(position)

        assert should_close is True
        assert "stop loss" in reason.lower()

    def test_should_close_position_near_expiration(self, strategy):
        """Test closing position near expiration"""
        position = IronCondorPosition(
            underlying_symbol="AAPL",
            expiration_date=datetime.now() + timedelta(days=5),  # Within threshold
            legs=[],
            net_credit=1.80,
            max_profit=1.80,
            max_loss=3.20,
            entry_time=datetime.now() - timedelta(days=25),
            profit_target=1.35,
            stop_loss=5.60,
            condor_type=CondorType.BALANCED,
            current_value=1.00
        )

        should_close, reason = strategy._should_close_position(position)

        assert should_close is True
        assert "expiration" in reason.lower()

    def test_determine_condor_type_balanced(self, strategy):
        """Test condor type determination - balanced"""
        strikes = {
            "short_put": {"strike_price": 150.0},
            "short_call": {"strike_price": 155.0}
        }
        current_price = 152.5

        condor_type = strategy._determine_condor_type(strikes, current_price)
        assert condor_type == CondorType.BALANCED

    def test_determine_condor_type_bullish(self, strategy):
        """Test condor type determination - bullish"""
        strikes = {
            "short_put": {"strike_price": 150.0},
            "short_call": {"strike_price": 155.0}
        }
        current_price = 151.0  # Closer to short put

        condor_type = strategy._determine_condor_type(strikes, current_price)
        assert condor_type == CondorType.BULLISH

    def test_determine_condor_type_bearish(self, strategy):
        """Test condor type determination - bearish"""
        strikes = {
            "short_put": {"strike_price": 150.0},
            "short_call": {"strike_price": 155.0}
        }
        current_price = 154.0  # Closer to short call

        condor_type = strategy._determine_condor_type(strikes, current_price)
        assert condor_type == CondorType.BEARISH

    def test_check_position_limits(self, strategy):
        """Test position limit checking"""
        # Add max condors
        for i in range(strategy.config.max_condors):
            position = IronCondorPosition(
                underlying_symbol=f"STOCK{i}",
                expiration_date=datetime(2025, 2, 21),
                legs=[],
                net_credit=1.80,
                max_profit=1.80,
                max_loss=3.20,
                entry_time=datetime.now(),
                profit_target=1.35,
                stop_loss=5.60,
                condor_type=CondorType.BALANCED
            )
            strategy.active_condors.append(position)

        can_add = strategy._can_add_new_condor("AAPL", datetime(2025, 2, 21))
        assert can_add is False

    def test_check_same_expiration_limit(self, strategy):
        """Test same expiration limit checking"""
        expiration = datetime(2025, 2, 21)

        # Add max same expiration condors
        for i in range(strategy.config.max_same_expiration):
            position = IronCondorPosition(
                underlying_symbol=f"STOCK{i}",
                expiration_date=expiration,
                legs=[],
                net_credit=1.80,
                max_profit=1.80,
                max_loss=3.20,
                entry_time=datetime.now(),
                profit_target=1.35,
                stop_loss=5.60,
                condor_type=CondorType.BALANCED
            )
            strategy.active_condors.append(position)

        can_add = strategy._can_add_new_condor("AAPL", expiration)
        assert can_add is False

    def test_get_strategy_name(self, strategy):
        """Test strategy name"""
        assert strategy.get_strategy_name() == "Iron Condor"

    def test_get_active_condors_count(self, strategy):
        """Test active condors count"""
        assert strategy.get_active_condors_count() == 0

        # Add position
        position = IronCondorPosition(
            underlying_symbol="AAPL",
            expiration_date=datetime(2025, 2, 21),
            legs=[],
            net_credit=1.80,
            max_profit=1.80,
            max_loss=3.20,
            entry_time=datetime.now(),
            profit_target=1.35,
            stop_loss=5.60,
            condor_type=CondorType.BALANCED
        )
        strategy.active_condors.append(position)

        assert strategy.get_active_condors_count() == 1

    @pytest.mark.asyncio
    async def test_monitor_condors_no_positions(self, strategy):
        """Test monitoring with no active condors"""
        await strategy._monitor_condors()
        # Should complete without errors


# Edge Cases and Error Handling Tests
class TestIronCondorEdgeCases:
    """Test edge cases and error handling for Iron Condor strategy"""

    @pytest.fixture
    def strategy(self):
        """Strategy fixture for edge case testing"""
        mock_client = MagicMock(spec=AlpacaClient)
        return IronCondorStrategy(mock_client)

    @pytest.mark.asyncio
    async def test_analyze_symbol_api_error(self, strategy):
        """Test analyze_symbol when API call fails"""
        strategy.alpaca_client.get_option_chain.side_effect = Exception("API Error")

        result = await strategy.analyze_symbol("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_execute_multi_leg_order_failure(self, strategy):
        """Test execute when multi-leg order fails"""
        signal = StrategySignal(
            symbol="AAPL",
            action="SELL",
            quantity=1,
            confidence=0.80,
            reasoning="Test",
            metadata={
                "condor_legs": [],
                "net_credit": 1.70
            }
        )

        strategy.alpaca_client.execute_multi_leg_order.side_effect = Exception("Order failed")

        result = await strategy.execute(signal)

        assert result["status"] == "error"
        assert "Order failed" in str(result["error"])

    def test_filter_options_insufficient_open_interest(self, strategy):
        """Test filtering options with low open interest"""
        options = [
            {"open_interest": 50, "delta": -0.15, "bid": 2.0, "ask": 2.2},  # Below minimum
            {"open_interest": 150, "delta": -0.15, "bid": 2.0, "ask": 2.2}  # Above minimum
        ]

        valid_options = strategy._filter_options_by_criteria(options)
        assert len(valid_options) == 1
        assert valid_options[0]["open_interest"] == 150

    def test_filter_options_wide_bid_ask_spread(self, strategy):
        """Test filtering options with wide bid-ask spreads"""
        options = [
            {"bid": 1.0, "ask": 2.0, "open_interest": 200, "delta": -0.15},  # 50% spread - too wide
            {"bid": 2.0, "ask": 2.2, "open_interest": 200, "delta": -0.15}   # 10% spread - acceptable
        ]

        valid_options = strategy._filter_options_by_criteria(options)
        assert len(valid_options) == 1
        assert valid_options[0]["bid"] == 2.0

    def test_strikes_with_same_expiration_different_symbols(self, strategy):
        """Test strike selection with different underlying symbols"""
        mixed_chain = [
            {"symbol": "AAPL250221P00150000", "underlying_symbol": "AAPL", "strike_price": 150.0,
             "option_type": "put", "delta": -0.15, "open_interest": 200, "expiration_date": "2025-02-21"},
            {"symbol": "MSFT250221P00300000", "underlying_symbol": "MSFT", "strike_price": 300.0,
             "option_type": "put", "delta": -0.15, "open_interest": 200, "expiration_date": "2025-02-21"}
        ]

        strikes = strategy._find_optimal_strikes("AAPL", mixed_chain, 152.5)

        # Should only consider AAPL options
        assert strikes is None  # Not enough AAPL options

    def test_calculate_net_credit_zero(self, strategy):
        """Test net credit calculation resulting in zero or negative"""
        strikes = {
            "long_put": {"bid": 2.0, "ask": 2.2},
            "short_put": {"bid": 1.0, "ask": 1.2},
            "short_call": {"bid": 1.0, "ask": 1.2},
            "long_call": {"bid": 2.0, "ask": 2.2}
        }

        net_credit = strategy._calculate_net_credit(strikes)
        # Should be negative: 1.1 + 1.1 - 2.1 - 2.1 = -2.0
        assert net_credit < 0

    def test_position_with_missing_current_value(self, strategy):
        """Test position evaluation with missing current value"""
        position = IronCondorPosition(
            underlying_symbol="AAPL",
            expiration_date=datetime(2025, 2, 21),
            legs=[],
            net_credit=1.80,
            max_profit=1.80,
            max_loss=3.20,
            entry_time=datetime.now(),
            profit_target=1.35,
            stop_loss=5.60,
            condor_type=CondorType.BALANCED,
            current_value=None  # Missing value
        )

        should_close, reason = strategy._should_close_position(position)

        # Should not close without current value
        assert should_close is False


if __name__ == "__main__":
    pytest.main([__file__])