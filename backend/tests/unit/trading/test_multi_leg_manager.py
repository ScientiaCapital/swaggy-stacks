"""
Unit tests for MultiLegOrderManager

Tests the atomic execution, rollback mechanisms, and order validation
for multi-leg options strategies like Iron Condors and complex spreads.

Test Coverage:
- Atomic order execution (all-or-nothing)
- Rollback mechanisms for partial failures
- Order validation and error handling
- Complex multi-leg strategy execution
- Error recovery and retry logic
- Position tracking and coordination
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timezone
import uuid
from decimal import Decimal

from app.trading.multi_leg_manager import MultiLegOrderManager
from app.trading.alpaca_client import AlpacaClient
from app.core.exceptions import TradingError, OrderExecutionError
from app.models.trading import (
    OptionOrder, OrderStatus, OrderAction, OrderType,
    Position, StrategySignal, OrderResult
)


class TestMultiLegOrderManager:
    """Test suite for MultiLegOrderManager atomic execution and rollback"""

    @pytest.fixture
    def mock_alpaca_client(self):
        """Mock Alpaca client for testing"""
        client = AsyncMock(spec=AlpacaClient)
        client.place_option_order = AsyncMock()
        client.cancel_order = AsyncMock()
        client.get_order_status = AsyncMock()
        client.get_positions = AsyncMock(return_value=[])
        return client

    @pytest.fixture
    def multi_leg_manager(self, mock_alpaca_client):
        """MultiLegOrderManager instance with mocked dependencies"""
        return MultiLegOrderManager(alpaca_client=mock_alpaca_client)

    @pytest.fixture
    def iron_condor_orders(self):
        """Sample Iron Condor four-leg order set"""
        base_time = datetime.now(timezone.utc)

        return [
            OptionOrder(
                id=str(uuid.uuid4()),
                symbol="AAPL250221P00150000",  # Long Put
                action=OrderAction.BUY,
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=Decimal("2.50"),
                time_in_force="DAY",
                created_at=base_time,
                status=OrderStatus.PENDING_NEW
            ),
            OptionOrder(
                id=str(uuid.uuid4()),
                symbol="AAPL250221P00155000",  # Short Put
                action=OrderAction.SELL,
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=Decimal("4.25"),
                time_in_force="DAY",
                created_at=base_time,
                status=OrderStatus.PENDING_NEW
            ),
            OptionOrder(
                id=str(uuid.uuid4()),
                symbol="AAPL250221C00165000",  # Short Call
                action=OrderAction.SELL,
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=Decimal("3.75"),
                time_in_force="DAY",
                created_at=base_time,
                status=OrderStatus.PENDING_NEW
            ),
            OptionOrder(
                id=str(uuid.uuid4()),
                symbol="AAPL250221C00170000",  # Long Call
                action=OrderAction.BUY,
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=Decimal("1.80"),
                time_in_force="DAY",
                created_at=base_time,
                status=OrderStatus.PENDING_NEW
            )
        ]

    @pytest.fixture
    def spread_orders(self):
        """Sample two-leg spread order set"""
        base_time = datetime.now(timezone.utc)

        return [
            OptionOrder(
                id=str(uuid.uuid4()),
                symbol="SPY250221C00450000",  # Long Call
                action=OrderAction.BUY,
                quantity=5,
                order_type=OrderType.LIMIT,
                limit_price=Decimal("3.20"),
                time_in_force="DAY",
                created_at=base_time,
                status=OrderStatus.PENDING_NEW
            ),
            OptionOrder(
                id=str(uuid.uuid4()),
                symbol="SPY250221C00455000",  # Short Call
                action=OrderAction.SELL,
                quantity=5,
                order_type=OrderType.LIMIT,
                limit_price=Decimal("1.95"),
                time_in_force="DAY",
                created_at=base_time,
                status=OrderStatus.PENDING_NEW
            )
        ]

    @pytest.fixture
    def successful_order_results(self):
        """Mock successful order execution results"""
        return [
            OrderResult(
                order_id="order_1",
                status=OrderStatus.FILLED,
                filled_quantity=10,
                filled_price=Decimal("2.48"),
                commission=Decimal("1.00"),
                timestamp=datetime.now(timezone.utc)
            ),
            OrderResult(
                order_id="order_2",
                status=OrderStatus.FILLED,
                filled_quantity=10,
                filled_price=Decimal("4.28"),
                commission=Decimal("1.00"),
                timestamp=datetime.now(timezone.utc)
            ),
            OrderResult(
                order_id="order_3",
                status=OrderStatus.FILLED,
                filled_quantity=10,
                filled_price=Decimal("3.73"),
                commission=Decimal("1.00"),
                timestamp=datetime.now(timezone.utc)
            ),
            OrderResult(
                order_id="order_4",
                status=OrderStatus.FILLED,
                filled_quantity=10,
                filled_price=Decimal("1.82"),
                commission=Decimal("1.00"),
                timestamp=datetime.now(timezone.utc)
            )
        ]

    @pytest.mark.asyncio
    async def test_execute_multi_leg_atomic_success(
        self, multi_leg_manager, iron_condor_orders, successful_order_results
    ):
        """Test successful atomic execution of all legs"""
        # Mock successful order placement
        multi_leg_manager.alpaca_client.place_option_order.side_effect = [
            result.order_id for result in successful_order_results
        ]

        # Mock order status checks showing all filled
        multi_leg_manager.alpaca_client.get_order_status.side_effect = successful_order_results

        # Execute multi-leg order
        strategy_id = "iron_condor_001"
        result = await multi_leg_manager.execute_multi_leg_order(
            orders=iron_condor_orders,
            strategy_id=strategy_id,
            max_wait_time=30.0
        )

        # Verify all orders were placed
        assert multi_leg_manager.alpaca_client.place_option_order.call_count == 4

        # Verify successful execution
        assert result.success is True
        assert len(result.executed_orders) == 4
        assert result.strategy_id == strategy_id
        assert result.net_credit > 0  # Iron Condor should generate credit

        # Verify no cancellations were needed
        multi_leg_manager.alpaca_client.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_multi_leg_partial_failure_rollback(
        self, multi_leg_manager, iron_condor_orders
    ):
        """Test rollback when some legs fail to execute"""
        # Mock first two orders succeed, third fails, fourth not attempted
        successful_results = [
            OrderResult(
                order_id="order_1",
                status=OrderStatus.FILLED,
                filled_quantity=10,
                filled_price=Decimal("2.48"),
                commission=Decimal("1.00"),
                timestamp=datetime.now(timezone.utc)
            ),
            OrderResult(
                order_id="order_2",
                status=OrderStatus.FILLED,
                filled_quantity=10,
                filled_price=Decimal("4.28"),
                commission=Decimal("1.00"),
                timestamp=datetime.now(timezone.utc)
            )
        ]

        # Mock order placement - first two succeed, third fails
        multi_leg_manager.alpaca_client.place_option_order.side_effect = [
            "order_1", "order_2", TradingError("Order rejected by exchange"), "order_4"
        ]

        # Mock status checks for successful orders
        multi_leg_manager.alpaca_client.get_order_status.side_effect = successful_results

        # Mock successful cancellations during rollback
        multi_leg_manager.alpaca_client.cancel_order.return_value = True

        strategy_id = "iron_condor_002"
        result = await multi_leg_manager.execute_multi_leg_order(
            orders=iron_condor_orders,
            strategy_id=strategy_id,
            max_wait_time=30.0
        )

        # Verify failure and rollback
        assert result.success is False
        assert len(result.executed_orders) == 0  # All should be rolled back
        assert "Order rejected by exchange" in result.error_message

        # Verify rollback cancellations were attempted
        assert multi_leg_manager.alpaca_client.cancel_order.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_multi_leg_timeout_rollback(
        self, multi_leg_manager, spread_orders
    ):
        """Test rollback when orders timeout waiting for fills"""
        # Mock orders get placed but never fill (timeout scenario)
        multi_leg_manager.alpaca_client.place_option_order.side_effect = [
            "order_1", "order_2"
        ]

        # Mock orders remaining pending (timeout)
        timeout_results = [
            OrderResult(
                order_id="order_1",
                status=OrderStatus.PENDING_NEW,
                filled_quantity=0,
                filled_price=None,
                commission=Decimal("0.00"),
                timestamp=datetime.now(timezone.utc)
            ),
            OrderResult(
                order_id="order_2",
                status=OrderStatus.PENDING_NEW,
                filled_quantity=0,
                filled_price=None,
                commission=Decimal("0.00"),
                timestamp=datetime.now(timezone.utc)
            )
        ]

        multi_leg_manager.alpaca_client.get_order_status.side_effect = timeout_results * 10
        multi_leg_manager.alpaca_client.cancel_order.return_value = True

        strategy_id = "spread_timeout"
        result = await multi_leg_manager.execute_multi_leg_order(
            orders=spread_orders,
            strategy_id=strategy_id,
            max_wait_time=1.0  # Short timeout for testing
        )

        # Verify timeout failure and rollback
        assert result.success is False
        assert "timeout" in result.error_message.lower()
        assert len(result.executed_orders) == 0

        # Verify cancellation attempts
        assert multi_leg_manager.alpaca_client.cancel_order.call_count == 2

    @pytest.mark.asyncio
    async def test_validate_multi_leg_order_success(self, multi_leg_manager, iron_condor_orders):
        """Test successful order validation for Iron Condor"""
        # Should pass validation
        is_valid, errors = await multi_leg_manager.validate_multi_leg_order(iron_condor_orders)

        assert is_valid is True
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_validate_multi_leg_order_quantity_mismatch(self, multi_leg_manager, iron_condor_orders):
        """Test validation failure for quantity mismatches"""
        # Modify one order to have different quantity
        iron_condor_orders[1].quantity = 15  # Different from others (10)

        is_valid, errors = await multi_leg_manager.validate_multi_leg_order(iron_condor_orders)

        assert is_valid is False
        assert any("quantity" in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_validate_multi_leg_order_missing_legs(self, multi_leg_manager):
        """Test validation failure for incomplete strategies"""
        # Single leg order (invalid for multi-leg strategy)
        incomplete_orders = [
            OptionOrder(
                id=str(uuid.uuid4()),
                symbol="AAPL250221C00160000",
                action=OrderAction.BUY,
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=Decimal("3.50"),
                time_in_force="DAY",
                created_at=datetime.now(timezone.utc),
                status=OrderStatus.PENDING_NEW
            )
        ]

        is_valid, errors = await multi_leg_manager.validate_multi_leg_order(incomplete_orders)

        assert is_valid is False
        assert any("legs" in error.lower() or "minimum" in error.lower() for error in errors)

    @pytest.mark.asyncio
    async def test_calculate_net_credit_debit(self, multi_leg_manager, iron_condor_orders):
        """Test net credit/debit calculation for multi-leg orders"""
        # Iron Condor should typically be a net credit strategy
        net_amount = multi_leg_manager._calculate_net_credit_debit(iron_condor_orders)

        expected_net = (
            Decimal("4.25") + Decimal("3.75") -  # Credits from short positions
            Decimal("2.50") - Decimal("1.80")    # Debits from long positions
        ) * 10  # Quantity

        assert net_amount == expected_net
        assert net_amount > 0  # Should be net credit

    @pytest.mark.asyncio
    async def test_retry_failed_order_success(self, multi_leg_manager):
        """Test successful retry of a failed order"""
        failed_order = OptionOrder(
            id=str(uuid.uuid4()),
            symbol="AAPL250221C00160000",
            action=OrderAction.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("3.50"),
            time_in_force="DAY",
            created_at=datetime.now(timezone.utc),
            status=OrderStatus.REJECTED
        )

        # Mock successful retry
        multi_leg_manager.alpaca_client.place_option_order.return_value = "retry_order_1"
        multi_leg_manager.alpaca_client.get_order_status.return_value = OrderResult(
            order_id="retry_order_1",
            status=OrderStatus.FILLED,
            filled_quantity=10,
            filled_price=Decimal("3.48"),
            commission=Decimal("1.00"),
            timestamp=datetime.now(timezone.utc)
        )

        result = await multi_leg_manager._retry_failed_order(
            order=failed_order,
            max_retries=3,
            retry_delay=0.1
        )

        assert result.success is True
        assert result.order_id == "retry_order_1"

    @pytest.mark.asyncio
    async def test_retry_failed_order_max_retries(self, multi_leg_manager):
        """Test retry failure after max attempts"""
        failed_order = OptionOrder(
            id=str(uuid.uuid4()),
            symbol="AAPL250221C00160000",
            action=OrderAction.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("3.50"),
            time_in_force="DAY",
            created_at=datetime.now(timezone.utc),
            status=OrderStatus.REJECTED
        )

        # Mock continued failures
        multi_leg_manager.alpaca_client.place_option_order.side_effect = TradingError("Order rejected")

        result = await multi_leg_manager._retry_failed_order(
            order=failed_order,
            max_retries=2,
            retry_delay=0.05
        )

        assert result.success is False
        assert "max retries" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_monitor_order_execution_all_filled(self, multi_leg_manager, spread_orders):
        """Test monitoring until all orders are filled"""
        order_ids = ["order_1", "order_2"]

        # Mock progressive fills
        fill_sequence = [
            # First check - order 1 filled, order 2 pending
            [
                OrderResult(
                    order_id="order_1",
                    status=OrderStatus.FILLED,
                    filled_quantity=5,
                    filled_price=Decimal("3.18"),
                    commission=Decimal("1.00"),
                    timestamp=datetime.now(timezone.utc)
                ),
                OrderResult(
                    order_id="order_2",
                    status=OrderStatus.PENDING_NEW,
                    filled_quantity=0,
                    filled_price=None,
                    commission=Decimal("0.00"),
                    timestamp=datetime.now(timezone.utc)
                )
            ],
            # Second check - both filled
            [
                OrderResult(
                    order_id="order_1",
                    status=OrderStatus.FILLED,
                    filled_quantity=5,
                    filled_price=Decimal("3.18"),
                    commission=Decimal("1.00"),
                    timestamp=datetime.now(timezone.utc)
                ),
                OrderResult(
                    order_id="order_2",
                    status=OrderStatus.FILLED,
                    filled_quantity=5,
                    filled_price=Decimal("1.93"),
                    commission=Decimal("1.00"),
                    timestamp=datetime.now(timezone.utc)
                )
            ]
        ]

        multi_leg_manager.alpaca_client.get_order_status.side_effect = [
            result for batch in fill_sequence for result in batch
        ]

        results = await multi_leg_manager._monitor_order_execution(
            order_ids=order_ids,
            max_wait_time=10.0,
            check_interval=0.1
        )

        assert len(results) == 2
        assert all(result.status == OrderStatus.FILLED for result in results)

    @pytest.mark.asyncio
    async def test_calculate_strategy_performance(self, multi_leg_manager, successful_order_results):
        """Test performance calculation for executed strategy"""
        strategy_id = "iron_condor_perf"

        performance = multi_leg_manager._calculate_strategy_performance(
            executed_orders=successful_order_results,
            strategy_id=strategy_id
        )

        assert performance.strategy_id == strategy_id
        assert performance.total_legs == 4
        assert performance.total_commission == Decimal("4.00")  # 4 orders × $1.00
        assert performance.net_credit > 0  # Should be positive for Iron Condor
        assert performance.execution_time_seconds > 0

    @pytest.mark.asyncio
    async def test_emergency_cancel_all_orders(self, multi_leg_manager):
        """Test emergency cancellation of all pending orders"""
        order_ids = ["order_1", "order_2", "order_3"]

        # Mock successful cancellations
        multi_leg_manager.alpaca_client.cancel_order.return_value = True

        cancelled_count = await multi_leg_manager._emergency_cancel_all_orders(order_ids)

        assert cancelled_count == 3
        assert multi_leg_manager.alpaca_client.cancel_order.call_count == 3

    @pytest.mark.asyncio
    async def test_handle_partial_execution_error(self, multi_leg_manager):
        """Test handling of partial execution errors"""
        executed_orders = [
            OrderResult(
                order_id="order_1",
                status=OrderStatus.FILLED,
                filled_quantity=10,
                filled_price=Decimal("2.48"),
                commission=Decimal("1.00"),
                timestamp=datetime.now(timezone.utc)
            )
        ]

        pending_order_ids = ["order_2", "order_3"]
        error_message = "Third leg rejected by exchange"

        # Mock successful cancellations
        multi_leg_manager.alpaca_client.cancel_order.return_value = True

        result = await multi_leg_manager._handle_partial_execution_error(
            executed_orders=executed_orders,
            pending_order_ids=pending_order_ids,
            error_message=error_message
        )

        assert result.success is False
        assert result.error_message == error_message
        assert len(result.executed_orders) == 0  # Should be empty after rollback

        # Verify cancellation attempts
        assert multi_leg_manager.alpaca_client.cancel_order.call_count == 2

    def test_validate_iron_condor_structure(self, multi_leg_manager, iron_condor_orders):
        """Test validation of Iron Condor structure"""
        is_valid = multi_leg_manager._validate_iron_condor_structure(iron_condor_orders)
        assert is_valid is True

    def test_validate_iron_condor_structure_invalid(self, multi_leg_manager):
        """Test validation failure for invalid Iron Condor structure"""
        # Create invalid structure (two calls, no puts)
        invalid_orders = [
            OptionOrder(
                id=str(uuid.uuid4()),
                symbol="AAPL250221C00160000",
                action=OrderAction.BUY,
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=Decimal("3.50"),
                time_in_force="DAY",
                created_at=datetime.now(timezone.utc),
                status=OrderStatus.PENDING_NEW
            ),
            OptionOrder(
                id=str(uuid.uuid4()),
                symbol="AAPL250221C00165000",
                action=OrderAction.SELL,
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=Decimal("2.25"),
                time_in_force="DAY",
                created_at=datetime.now(timezone.utc),
                status=OrderStatus.PENDING_NEW
            )
        ]

        is_valid = multi_leg_manager._validate_iron_condor_structure(invalid_orders)
        assert is_valid is False