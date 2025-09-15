"""
Comprehensive Unit Tests for Gamma Scalping Options Strategy

Tests delta-neutral gamma scalping strategy including position setup,
dynamic rebalancing, volatility tracking, and risk management.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import asdict

from app.strategies.options.gamma_scalping_strategy import (
    GammaScalpingStrategy,
    GammaScalpingConfig,
    GammaPosition,
    RebalanceOrder,
    VolatilityRegime
)
from app.trading.alpaca_client import AlpacaClient
from app.trading.options_trading import OptionContract, GreeksData, OptionType
from app.rag.strategies.strategy_engines import StrategySignal


class TestGammaScalpingConfig:
    """Test Gamma Scalping configuration validation"""

    def test_default_config_values(self):
        """Test default configuration values"""
        config = GammaScalpingConfig()

        # Delta neutrality
        assert config.target_delta == 0.0
        assert config.delta_tolerance == 0.05
        assert config.rebalance_threshold == 0.10

        # Gamma targeting
        assert config.min_gamma == 0.01
        assert config.max_gamma == 0.50
        assert config.target_gamma_exposure == 1000.0

        # Volatility management
        assert config.min_implied_vol == 0.15
        assert config.max_implied_vol == 0.80
        assert config.vol_regime_threshold == 0.25

        # Time management
        assert config.min_dte == 7
        assert config.max_dte == 60
        assert config.rebalance_frequency_minutes == 15

        # Risk management
        assert config.max_loss_per_position == 5000.0
        assert config.profit_target_pct == 20.0
        assert config.stop_loss_pct == 100.0

        # Position limits
        assert config.max_positions == 3
        assert config.max_gamma_exposure == 3000.0

    def test_custom_config_values(self):
        """Test custom configuration values"""
        config = GammaScalpingConfig(
            delta_tolerance=0.08,
            target_gamma_exposure=1500.0,
            max_positions=5,
            rebalance_frequency_minutes=10
        )

        assert config.delta_tolerance == 0.08
        assert config.target_gamma_exposure == 1500.0
        assert config.max_positions == 5
        assert config.rebalance_frequency_minutes == 10


class TestRebalanceOrder:
    """Test rebalance order data structure"""

    def test_rebalance_order_creation(self):
        """Test creating a rebalance order"""
        order = RebalanceOrder(
            symbol="AAPL",
            action="buy",
            quantity=50,
            order_type="market",
            reasoning="Delta too negative, buying stock to neutralize",
            target_delta=0.02,
            current_delta=-0.08
        )

        assert order.symbol == "AAPL"
        assert order.action == "buy"
        assert order.quantity == 50
        assert order.order_type == "market"
        assert "Delta too negative" in order.reasoning
        assert order.target_delta == 0.02
        assert order.current_delta == -0.08


class TestGammaPosition:
    """Test Gamma position data structure"""

    def test_position_creation(self):
        """Test complete Gamma position creation"""
        entry_time = datetime.now()
        position = GammaPosition(
            underlying_symbol="AAPL",
            option_symbol="AAPL250221C00155000",
            option_type="call",
            strike_price=155.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=3.50,
            quantity=10,
            entry_time=entry_time,
            stock_position=0,  # Start delta neutral
            target_delta=0.0,
            current_delta=0.02,
            current_gamma=0.08,
            realized_pnl=0.0,
            unrealized_pnl=150.0,
            total_rebalances=3,
            last_rebalance_time=entry_time,
            volatility_regime=VolatilityRegime.NORMAL
        )

        assert position.underlying_symbol == "AAPL"
        assert position.option_symbol == "AAPL250221C00155000"
        assert position.option_type == "call"
        assert position.strike_price == 155.0
        assert position.quantity == 10
        assert position.stock_position == 0
        assert position.current_delta == 0.02
        assert position.current_gamma == 0.08
        assert position.total_rebalances == 3
        assert position.volatility_regime == VolatilityRegime.NORMAL

    def test_position_gamma_exposure(self):
        """Test gamma exposure calculation"""
        position = GammaPosition(
            underlying_symbol="AAPL",
            option_symbol="AAPL250221C00155000",
            option_type="call",
            strike_price=155.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=3.50,
            quantity=10,
            entry_time=datetime.now(),
            stock_position=0,
            target_delta=0.0,
            current_delta=0.02,
            current_gamma=0.08,
            realized_pnl=0.0,
            unrealized_pnl=150.0,
            total_rebalances=0,
            last_rebalance_time=datetime.now(),
            volatility_regime=VolatilityRegime.NORMAL
        )

        gamma_exposure = position.calculate_gamma_exposure()
        expected_exposure = 10 * 0.08 * 100  # quantity * gamma * 100 shares per contract
        assert gamma_exposure == expected_exposure

    def test_position_total_pnl(self):
        """Test total P&L calculation"""
        position = GammaPosition(
            underlying_symbol="AAPL",
            option_symbol="AAPL250221C00155000",
            option_type="call",
            strike_price=155.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=3.50,
            quantity=10,
            entry_time=datetime.now(),
            stock_position=0,
            target_delta=0.0,
            current_delta=0.02,
            current_gamma=0.08,
            realized_pnl=250.0,
            unrealized_pnl=150.0,
            total_rebalances=3,
            last_rebalance_time=datetime.now(),
            volatility_regime=VolatilityRegime.NORMAL
        )

        total_pnl = position.calculate_total_pnl()
        assert total_pnl == 400.0  # 250 + 150


class TestGammaScalpingStrategy:
    """Comprehensive tests for Gamma Scalping strategy implementation"""

    @pytest.fixture
    def mock_alpaca_client(self):
        """Mock Alpaca client fixture"""
        client = MagicMock(spec=AlpacaClient)
        client.get_option_chain = AsyncMock()
        client.get_option_quote = AsyncMock()
        client.execute_order = AsyncMock()
        client.get_positions = AsyncMock()
        client.get_market_data = AsyncMock()
        client.get_account = AsyncMock()
        return client

    @pytest.fixture
    def default_config(self):
        """Default configuration fixture"""
        return GammaScalpingConfig()

    @pytest.fixture
    def strategy(self, mock_alpaca_client, default_config):
        """Gamma Scalping strategy fixture"""
        return GammaScalpingStrategy(mock_alpaca_client, default_config)

    @pytest.fixture
    def mock_option_chain(self):
        """Mock option chain for ATM options"""
        return [
            # ATM Call
            {
                "symbol": "AAPL250221C00150000",
                "underlying_symbol": "AAPL",
                "strike_price": 150.0,
                "option_type": "call",
                "expiration_date": "2025-02-21",
                "bid": 3.40,
                "ask": 3.60,
                "last_price": 3.50,
                "implied_volatility": 0.25,
                "delta": 0.50,
                "gamma": 0.08,
                "theta": -0.05,
                "vega": 0.12,
                "open_interest": 2000,
                "volume": 500
            },
            # ATM Put
            {
                "symbol": "AAPL250221P00150000",
                "underlying_symbol": "AAPL",
                "strike_price": 150.0,
                "option_type": "put",
                "expiration_date": "2025-02-21",
                "bid": 3.40,
                "ask": 3.60,
                "last_price": 3.50,
                "implied_volatility": 0.25,
                "delta": -0.50,
                "gamma": 0.08,
                "theta": -0.05,
                "vega": 0.12,
                "open_interest": 1800,
                "volume": 400
            }
        ]

    def test_strategy_initialization(self, mock_alpaca_client, default_config):
        """Test strategy initialization"""
        strategy = GammaScalpingStrategy(mock_alpaca_client, default_config)

        assert strategy.alpaca_client == mock_alpaca_client
        assert strategy.config == default_config
        assert strategy.active_positions == []
        assert strategy.rebalancing_task is None

    @pytest.mark.asyncio
    async def test_analyze_symbol_valid_opportunity(self, strategy, mock_alpaca_client, mock_option_chain):
        """Test analyze_symbol with valid gamma scalping opportunity"""
        mock_alpaca_client.get_option_chain.return_value = mock_option_chain

        # Mock market data with current price at strike (ATM)
        market_data = {
            "current_price": 150.0,
            "volume": 1000000,
            "implied_volatility": 0.25
        }

        result = await strategy.analyze_symbol("AAPL", market_data)

        assert result is not None
        assert isinstance(result, StrategySignal)
        assert result.symbol == "AAPL"
        assert "Gamma Scalping" in result.reasoning

    @pytest.mark.asyncio
    async def test_analyze_symbol_low_volatility(self, strategy, mock_alpaca_client, mock_option_chain):
        """Test analyze_symbol with low volatility (unfavorable)"""
        # Modify option chain for low volatility
        low_vol_chain = mock_option_chain.copy()
        for option in low_vol_chain:
            option["implied_volatility"] = 0.10  # Below minimum

        mock_alpaca_client.get_option_chain.return_value = low_vol_chain

        market_data = {"current_price": 150.0, "implied_volatility": 0.10}

        result = await strategy.analyze_symbol("AAPL", market_data)

        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_symbol_high_volatility(self, strategy, mock_alpaca_client, mock_option_chain):
        """Test analyze_symbol with high volatility (unfavorable)"""
        # Modify option chain for high volatility
        high_vol_chain = mock_option_chain.copy()
        for option in high_vol_chain:
            option["implied_volatility"] = 0.90  # Above maximum

        mock_alpaca_client.get_option_chain.return_value = high_vol_chain

        market_data = {"current_price": 150.0, "implied_volatility": 0.90}

        result = await strategy.analyze_symbol("AAPL", market_data)

        assert result is None

    def test_find_best_atm_option_call(self, strategy, mock_option_chain):
        """Test finding best ATM option - call selected"""
        current_price = 150.0

        best_option = strategy._find_best_atm_option("AAPL", mock_option_chain, current_price)

        assert best_option is not None
        assert best_option["option_type"] == "call"  # Should prefer call for positive gamma exposure
        assert best_option["strike_price"] == 150.0

    def test_find_best_atm_option_no_suitable(self, strategy):
        """Test finding ATM option when none suitable"""
        poor_options = [
            {
                "symbol": "AAPL250221C00150000",
                "strike_price": 150.0,
                "option_type": "call",
                "gamma": 0.005,  # Too low gamma
                "implied_volatility": 0.25,
                "open_interest": 2000
            }
        ]

        best_option = strategy._find_best_atm_option("AAPL", poor_options, 150.0)

        assert best_option is None

    def test_calculate_volatility_regime_normal(self, strategy):
        """Test volatility regime calculation - normal"""
        implied_vol = 0.25

        regime = strategy._calculate_volatility_regime(implied_vol)

        assert regime == VolatilityRegime.NORMAL

    def test_calculate_volatility_regime_low(self, strategy):
        """Test volatility regime calculation - low"""
        implied_vol = 0.18

        regime = strategy._calculate_volatility_regime(implied_vol)

        assert regime == VolatilityRegime.LOW

    def test_calculate_volatility_regime_high(self, strategy):
        """Test volatility regime calculation - high"""
        implied_vol = 0.35

        regime = strategy._calculate_volatility_regime(implied_vol)

        assert regime == VolatilityRegime.HIGH

    @pytest.mark.asyncio
    async def test_execute_gamma_signal(self, strategy, mock_alpaca_client):
        """Test executing gamma scalping signal"""
        signal = StrategySignal(
            symbol="AAPL",
            action="BUY",
            quantity=10,
            confidence=0.75,
            reasoning="Gamma scalping opportunity",
            metadata={
                "option_symbol": "AAPL250221C00150000",
                "option_type": "call",
                "strike_price": 150.0,
                "entry_price": 3.50,
                "target_gamma": 0.08
            }
        )

        mock_alpaca_client.execute_order.return_value = {
            "id": "gamma_order_123",
            "status": "filled",
            "filled_price": 3.50
        }

        result = await strategy.execute(signal)

        assert result["status"] == "success"
        assert result["order_id"] == "gamma_order_123"
        assert len(strategy.active_positions) == 1
        assert strategy.active_positions[0].option_type == "call"

    def test_check_rebalancing_needed_within_tolerance(self, strategy):
        """Test rebalancing check when delta within tolerance"""
        position = GammaPosition(
            underlying_symbol="AAPL",
            option_symbol="AAPL250221C00150000",
            option_type="call",
            strike_price=150.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=3.50,
            quantity=10,
            entry_time=datetime.now(),
            stock_position=0,
            target_delta=0.0,
            current_delta=0.03,  # Within tolerance (0.05)
            current_gamma=0.08,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_rebalances=0,
            last_rebalance_time=datetime.now(),
            volatility_regime=VolatilityRegime.NORMAL
        )

        needs_rebalance, order = strategy._check_rebalancing_needed(position)

        assert needs_rebalance is False
        assert order is None

    def test_check_rebalancing_needed_delta_too_positive(self, strategy):
        """Test rebalancing when delta too positive"""
        position = GammaPosition(
            underlying_symbol="AAPL",
            option_symbol="AAPL250221C00150000",
            option_type="call",
            strike_price=150.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=3.50,
            quantity=10,
            entry_time=datetime.now(),
            stock_position=0,
            target_delta=0.0,
            current_delta=0.15,  # Too positive
            current_gamma=0.08,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_rebalances=0,
            last_rebalance_time=datetime.now(),
            volatility_regime=VolatilityRegime.NORMAL
        )

        needs_rebalance, order = strategy._check_rebalancing_needed(position)

        assert needs_rebalance is True
        assert order is not None
        assert order.action == "sell"  # Sell stock to reduce delta
        assert order.quantity > 0

    def test_check_rebalancing_needed_delta_too_negative(self, strategy):
        """Test rebalancing when delta too negative"""
        position = GammaPosition(
            underlying_symbol="AAPL",
            option_symbol="AAPL250221C00150000",
            option_type="call",
            strike_price=150.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=3.50,
            quantity=10,
            entry_time=datetime.now(),
            stock_position=-200,  # Short stock position
            target_delta=0.0,
            current_delta=-0.12,  # Too negative
            current_gamma=0.08,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_rebalances=0,
            last_rebalance_time=datetime.now(),
            volatility_regime=VolatilityRegime.NORMAL
        )

        needs_rebalance, order = strategy._check_rebalancing_needed(position)

        assert needs_rebalance is True
        assert order is not None
        assert order.action == "buy"  # Buy stock to increase delta
        assert order.quantity > 0

    def test_calculate_hedge_quantity(self, strategy):
        """Test hedge quantity calculation"""
        current_delta = 0.15
        target_delta = 0.0
        delta_per_share = 0.01  # How much delta changes per share

        hedge_quantity = strategy._calculate_hedge_quantity(current_delta, target_delta, delta_per_share)

        expected_quantity = abs(current_delta - target_delta) / delta_per_share
        assert hedge_quantity == int(expected_quantity)

    def test_should_close_position_profit_target(self, strategy):
        """Test closing position at profit target"""
        position = GammaPosition(
            underlying_symbol="AAPL",
            option_symbol="AAPL250221C00150000",
            option_type="call",
            strike_price=150.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=3.50,
            quantity=10,
            entry_time=datetime.now() - timedelta(days=5),
            stock_position=0,
            target_delta=0.0,
            current_delta=0.02,
            current_gamma=0.08,
            realized_pnl=500.0,  # Good realized profits
            unrealized_pnl=200.0,
            total_rebalances=5,
            last_rebalance_time=datetime.now(),
            volatility_regime=VolatilityRegime.NORMAL
        )

        should_close, reason = strategy._should_close_position(position)

        assert should_close is True
        assert "profit target" in reason.lower()

    def test_should_close_position_stop_loss(self, strategy):
        """Test closing position at stop loss"""
        position = GammaPosition(
            underlying_symbol="AAPL",
            option_symbol="AAPL250221C00150000",
            option_type="call",
            strike_price=150.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=3.50,
            quantity=10,
            entry_time=datetime.now() - timedelta(days=5),
            stock_position=0,
            target_delta=0.0,
            current_delta=0.02,
            current_gamma=0.08,
            realized_pnl=-3000.0,  # Large loss
            unrealized_pnl=-500.0,
            total_rebalances=5,
            last_rebalance_time=datetime.now(),
            volatility_regime=VolatilityRegime.NORMAL
        )

        should_close, reason = strategy._should_close_position(position)

        assert should_close is True
        assert "stop loss" in reason.lower()

    def test_should_close_position_near_expiration(self, strategy):
        """Test closing position near expiration"""
        position = GammaPosition(
            underlying_symbol="AAPL",
            option_symbol="AAPL250221C00150000",
            option_type="call",
            strike_price=150.0,
            expiration_date=datetime.now() + timedelta(days=3),  # Very close to expiration
            entry_price=3.50,
            quantity=10,
            entry_time=datetime.now() - timedelta(days=20),
            stock_position=0,
            target_delta=0.0,
            current_delta=0.02,
            current_gamma=0.08,
            realized_pnl=100.0,
            unrealized_pnl=50.0,
            total_rebalances=5,
            last_rebalance_time=datetime.now(),
            volatility_regime=VolatilityRegime.NORMAL
        )

        should_close, reason = strategy._should_close_position(position)

        assert should_close is True
        assert "expiration" in reason.lower()

    def test_should_close_position_no_close(self, strategy):
        """Test position that should not close"""
        position = GammaPosition(
            underlying_symbol="AAPL",
            option_symbol="AAPL250221C00150000",
            option_type="call",
            strike_price=150.0,
            expiration_date=datetime.now() + timedelta(days=15),
            entry_price=3.50,
            quantity=10,
            entry_time=datetime.now() - timedelta(days=5),
            stock_position=0,
            target_delta=0.0,
            current_delta=0.02,
            current_gamma=0.08,
            realized_pnl=200.0,  # Moderate profit
            unrealized_pnl=100.0,
            total_rebalances=3,
            last_rebalance_time=datetime.now(),
            volatility_regime=VolatilityRegime.NORMAL
        )

        should_close, reason = strategy._should_close_position(position)

        assert should_close is False
        assert reason == ""

    def test_check_position_limits(self, strategy):
        """Test position limit checking"""
        # Add max positions
        for i in range(strategy.config.max_positions):
            position = GammaPosition(
                underlying_symbol=f"STOCK{i}",
                option_symbol=f"STOCK{i}250221C00150000",
                option_type="call",
                strike_price=150.0,
                expiration_date=datetime(2025, 2, 21),
                entry_price=3.50,
                quantity=10,
                entry_time=datetime.now(),
                stock_position=0,
                target_delta=0.0,
                current_delta=0.02,
                current_gamma=0.08,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                total_rebalances=0,
                last_rebalance_time=datetime.now(),
                volatility_regime=VolatilityRegime.NORMAL
            )
            strategy.active_positions.append(position)

        can_add = strategy._can_add_new_position("AAPL")
        assert can_add is False

    def test_check_gamma_exposure_limit(self, strategy):
        """Test gamma exposure limit checking"""
        # Add position with high gamma exposure
        position = GammaPosition(
            underlying_symbol="AAPL",
            option_symbol="AAPL250221C00150000",
            option_type="call",
            strike_price=150.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=3.50,
            quantity=100,  # Large quantity
            entry_time=datetime.now(),
            stock_position=0,
            target_delta=0.0,
            current_delta=0.02,
            current_gamma=0.30,  # High gamma
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_rebalances=0,
            last_rebalance_time=datetime.now(),
            volatility_regime=VolatilityRegime.NORMAL
        )
        strategy.active_positions.append(position)

        total_exposure = strategy._calculate_total_gamma_exposure()
        expected_exposure = 100 * 0.30 * 100  # 3000
        assert total_exposure == expected_exposure
        assert total_exposure >= strategy.config.max_gamma_exposure

    def test_get_strategy_name(self, strategy):
        """Test strategy name"""
        assert strategy.get_strategy_name() == "Gamma Scalping"

    def test_get_active_positions_count(self, strategy):
        """Test active positions count"""
        assert strategy.get_active_positions_count() == 0

        # Add position
        position = GammaPosition(
            underlying_symbol="AAPL",
            option_symbol="AAPL250221C00150000",
            option_type="call",
            strike_price=150.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=3.50,
            quantity=10,
            entry_time=datetime.now(),
            stock_position=0,
            target_delta=0.0,
            current_delta=0.02,
            current_gamma=0.08,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_rebalances=0,
            last_rebalance_time=datetime.now(),
            volatility_regime=VolatilityRegime.NORMAL
        )
        strategy.active_positions.append(position)

        assert strategy.get_active_positions_count() == 1

    @pytest.mark.asyncio
    async def test_start_rebalancing_monitor(self, strategy):
        """Test starting rebalancing monitor"""
        await strategy._start_rebalancing_monitor()

        assert strategy.rebalancing_task is not None
        assert not strategy.rebalancing_task.done()

        # Clean up
        strategy.rebalancing_task.cancel()
        try:
            await strategy.rebalancing_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_monitor_positions_no_positions(self, strategy):
        """Test monitoring with no active positions"""
        await strategy._monitor_positions()
        # Should complete without errors


# Edge Cases and Error Handling Tests
class TestGammaScalpingEdgeCases:
    """Test edge cases and error handling for Gamma Scalping strategy"""

    @pytest.fixture
    def strategy(self):
        """Strategy fixture for edge case testing"""
        mock_client = MagicMock(spec=AlpacaClient)
        return GammaScalpingStrategy(mock_client)

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
            quantity=10,
            confidence=0.75,
            reasoning="Test",
            metadata={
                "option_symbol": "AAPL250221C00150000",
                "option_type": "call"
            }
        )

        strategy.alpaca_client.execute_order.side_effect = Exception("Order failed")

        result = await strategy.execute(signal)

        assert result["status"] == "error"
        assert "Order failed" in str(result["error"])

    def test_rebalancing_with_zero_gamma(self, strategy):
        """Test rebalancing calculation with zero gamma"""
        position = GammaPosition(
            underlying_symbol="AAPL",
            option_symbol="AAPL250221C00150000",
            option_type="call",
            strike_price=150.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=3.50,
            quantity=10,
            entry_time=datetime.now(),
            stock_position=0,
            target_delta=0.0,
            current_delta=0.15,
            current_gamma=0.0,  # Zero gamma
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_rebalances=0,
            last_rebalance_time=datetime.now(),
            volatility_regime=VolatilityRegime.NORMAL
        )

        needs_rebalance, order = strategy._check_rebalancing_needed(position)

        # Should still try to rebalance based on delta
        assert needs_rebalance is True
        assert order is not None

    def test_position_with_extreme_delta(self, strategy):
        """Test position with extreme delta values"""
        position = GammaPosition(
            underlying_symbol="AAPL",
            option_symbol="AAPL250221C00150000",
            option_type="call",
            strike_price=150.0,
            expiration_date=datetime(2025, 2, 21),
            entry_price=3.50,
            quantity=10,
            entry_time=datetime.now(),
            stock_position=0,
            target_delta=0.0,
            current_delta=0.95,  # Extreme positive delta
            current_gamma=0.01,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_rebalances=0,
            last_rebalance_time=datetime.now(),
            volatility_regime=VolatilityRegime.NORMAL
        )

        needs_rebalance, order = strategy._check_rebalancing_needed(position)

        assert needs_rebalance is True
        assert order.quantity > 0  # Should calculate reasonable hedge quantity

    def test_calculate_hedge_quantity_edge_cases(self, strategy):
        """Test hedge quantity calculation edge cases"""
        # Zero delta difference
        hedge_qty = strategy._calculate_hedge_quantity(0.05, 0.05, 0.01)
        assert hedge_qty == 0

        # Very small delta per share
        hedge_qty = strategy._calculate_hedge_quantity(0.10, 0.0, 0.0001)
        assert hedge_qty > 0  # Should handle small values

    def test_volatility_regime_boundary_values(self, strategy):
        """Test volatility regime calculation at boundaries"""
        # Exactly at threshold
        regime = strategy._calculate_volatility_regime(0.25)
        assert regime == VolatilityRegime.NORMAL

        # Just below threshold
        regime = strategy._calculate_volatility_regime(0.24)
        assert regime == VolatilityRegime.LOW

        # Just above threshold
        regime = strategy._calculate_volatility_regime(0.26)
        assert regime == VolatilityRegime.HIGH


if __name__ == "__main__":
    pytest.main([__file__])