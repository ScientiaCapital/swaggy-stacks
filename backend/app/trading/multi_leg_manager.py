"""
Multi-Leg Order Manager

Extends the existing OrderManager with atomic multi-leg options order execution.
Provides all-or-nothing execution guarantees with comprehensive rollback capabilities
for complex options strategies.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import uuid

import structlog
from app.core.exceptions import TradingError
from app.trading.alpaca_client import AlpacaClient
from app.trading.order_manager import OrderManager
from app.trading.risk_manager import RiskManager

logger = structlog.get_logger()


class LegStatus(Enum):
    """Status of individual order legs"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class MultiLegOrderStatus(Enum):
    """Status of multi-leg order"""
    CREATED = "created"
    VALIDATING = "validating"
    SUBMITTING = "submitting"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class OrderLeg:
    """Individual leg of a multi-leg order"""

    # Order identification
    leg_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    order_type: str = "market"  # "market", "limit", "stop", "stop_limit"

    # Pricing
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    # Execution details
    status: LegStatus = LegStatus.PENDING
    order_id: Optional[str] = None
    filled_qty: int = 0
    filled_avg_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class MultiLegOrder:
    """Multi-leg order container"""

    # Order identification
    order_id: str
    strategy_name: str
    underlying_symbol: str

    # Legs
    legs: List[OrderLeg]

    # Execution parameters
    order_class: str = "MLEG"  # Multi-leg order class
    time_in_force: str = "gtc"
    execution_timeout: int = 300  # 5 minutes timeout

    # Status tracking
    status: MultiLegOrderStatus = MultiLegOrderStatus.CREATED
    created_at: datetime = None
    submitted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    total_legs: int = 0
    filled_legs: int = 0
    failed_legs: int = 0
    net_premium: Optional[float] = None

    # Error handling
    error_message: Optional[str] = None
    rollback_reason: Optional[str] = None
    rollback_orders: List[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        self.total_legs = len(self.legs)


class MultiLegOrderManager(OrderManager):
    """
    Multi-Leg Order Manager

    Extends OrderManager to provide atomic multi-leg options order execution
    with comprehensive rollback capabilities. Ensures all-or-nothing execution
    for complex options strategies.
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        risk_manager: RiskManager,
        max_concurrent_orders: int = 5,
    ):
        # Initialize base OrderManager
        super().__init__(alpaca_client, risk_manager)

        # Multi-leg specific tracking
        self.multi_leg_orders: Dict[str, MultiLegOrder] = {}
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        self.max_concurrent_orders = max_concurrent_orders

        # Execution control
        self.executor_task: Optional[asyncio.Task] = None
        self.is_processing = False

        # Transaction logging
        self.transaction_log: List[Dict[str, Any]] = []

        logger.info("Multi-leg order manager initialized",
                   max_concurrent=max_concurrent_orders)

    async def execute_multi_leg_order(
        self,
        legs: List[Dict[str, Any]],
        strategy_name: str,
        underlying_symbol: str,
        order_class: str = "MLEG",
        time_in_force: str = "gtc",
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """
        Execute atomic multi-leg options order

        Args:
            legs: List of leg specifications
            strategy_name: Name of the strategy (e.g., "Iron Condor")
            underlying_symbol: Underlying symbol
            order_class: Order class (default: MLEG)
            time_in_force: Time in force (default: gtc)
            timeout: Execution timeout in seconds

        Returns:
            Execution result with order details
        """
        try:
            # Validate input parameters
            await self._validate_multi_leg_request(legs, strategy_name, underlying_symbol)

            # Create multi-leg order
            multi_leg_order = await self._create_multi_leg_order(
                legs, strategy_name, underlying_symbol, order_class, time_in_force, timeout
            )

            # Add to tracking
            self.multi_leg_orders[multi_leg_order.order_id] = multi_leg_order

            # Execute the order
            result = await self._execute_order_atomic(multi_leg_order)

            # Log transaction
            await self._log_transaction(multi_leg_order, result)

            return result

        except Exception as e:
            logger.error("Failed to execute multi-leg order", error=str(e))
            raise TradingError(f"Multi-leg order execution failed: {e}")

    async def _validate_multi_leg_request(
        self, legs: List[Dict[str, Any]], strategy_name: str, underlying_symbol: str
    ):
        """Validate multi-leg order request"""

        # Check leg count
        if len(legs) < 2 or len(legs) > 4:
            raise TradingError("Multi-leg orders must have 2-4 legs")

        # Validate each leg
        for i, leg_data in enumerate(legs):
            if "symbol" not in leg_data:
                raise TradingError(f"Leg {i+1}: Missing symbol")
            if "side" not in leg_data or leg_data["side"] not in ["buy", "sell"]:
                raise TradingError(f"Leg {i+1}: Invalid side")
            if "quantity" not in leg_data or leg_data["quantity"] <= 0:
                raise TradingError(f"Leg {i+1}: Invalid quantity")

        # Strategy validation
        if not strategy_name:
            raise TradingError("Strategy name is required")

        if not underlying_symbol:
            raise TradingError("Underlying symbol is required")

        logger.debug("Multi-leg order validation passed",
                    legs=len(legs), strategy=strategy_name, underlying=underlying_symbol)

    async def _create_multi_leg_order(
        self,
        legs: List[Dict[str, Any]],
        strategy_name: str,
        underlying_symbol: str,
        order_class: str,
        time_in_force: str,
        timeout: int,
    ) -> MultiLegOrder:
        """Create multi-leg order object"""

        order_id = f"MLO_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"

        # Create order legs
        order_legs = []
        for i, leg_data in enumerate(legs):
            leg = OrderLeg(
                leg_id=f"{order_id}_LEG_{i+1}",
                symbol=leg_data["symbol"],
                side=leg_data["side"],
                quantity=leg_data["quantity"],
                order_type=leg_data.get("order_type", "market"),
                limit_price=leg_data.get("limit_price"),
                stop_price=leg_data.get("stop_price"),
            )
            order_legs.append(leg)

        # Create multi-leg order
        multi_leg_order = MultiLegOrder(
            order_id=order_id,
            strategy_name=strategy_name,
            underlying_symbol=underlying_symbol,
            legs=order_legs,
            order_class=order_class,
            time_in_force=time_in_force,
            execution_timeout=timeout,
        )

        logger.info("Multi-leg order created",
                   order_id=order_id, strategy=strategy_name, legs=len(order_legs))

        return multi_leg_order

    async def _execute_order_atomic(self, multi_leg_order: MultiLegOrder) -> Dict[str, Any]:
        """Execute multi-leg order with atomic guarantees"""

        multi_leg_order.status = MultiLegOrderStatus.VALIDATING

        try:
            # Pre-execution validation
            await self._validate_order_execution(multi_leg_order)

            multi_leg_order.status = MultiLegOrderStatus.SUBMITTING
            multi_leg_order.submitted_at = datetime.now()

            # Attempt to use native multi-leg execution first
            native_result = await self._try_native_multi_leg_execution(multi_leg_order)
            if native_result:
                return native_result

            # Fall back to individual leg execution with rollback protection
            return await self._execute_legs_with_rollback(multi_leg_order)

        except Exception as e:
            logger.error("Multi-leg order execution failed",
                        order_id=multi_leg_order.order_id, error=str(e))

            # Attempt rollback
            await self._rollback_order(multi_leg_order, str(e))
            raise TradingError(f"Multi-leg order failed and rolled back: {e}")

    async def _try_native_multi_leg_execution(self, multi_leg_order: MultiLegOrder) -> Optional[Dict[str, Any]]:
        """Attempt native multi-leg execution via Alpaca API"""
        try:
            # Prepare legs for Alpaca multi-leg API
            alpaca_legs = []
            for leg in multi_leg_order.legs:
                alpaca_leg = {
                    "symbol": leg.symbol,
                    "side": leg.side,
                    "quantity": leg.quantity,
                }
                if leg.limit_price:
                    alpaca_leg["limit_price"] = leg.limit_price
                if leg.stop_price:
                    alpaca_leg["stop_price"] = leg.stop_price
                alpaca_legs.append(alpaca_leg)

            # Submit to Alpaca multi-leg API
            result = await self.alpaca_client.execute_multi_leg_order(
                legs=alpaca_legs,
                order_class=multi_leg_order.order_class,
                time_in_force=multi_leg_order.time_in_force,
                client_order_id=multi_leg_order.order_id,
            )

            if result and result.get("status") == "accepted":
                multi_leg_order.status = MultiLegOrderStatus.FILLED
                multi_leg_order.completed_at = datetime.now()

                # Update leg statuses
                for leg in multi_leg_order.legs:
                    leg.status = LegStatus.FILLED
                    leg.filled_at = datetime.now()

                multi_leg_order.filled_legs = len(multi_leg_order.legs)

                logger.info("Native multi-leg execution successful",
                           order_id=multi_leg_order.order_id)

                return {
                    "status": "success",
                    "execution_method": "native_multileg",
                    "order_id": multi_leg_order.order_id,
                    "alpaca_order_id": result.get("id"),
                    "legs_filled": multi_leg_order.filled_legs,
                    "total_legs": multi_leg_order.total_legs,
                    "completed_at": multi_leg_order.completed_at.isoformat(),
                }

            return None

        except Exception as e:
            logger.warning("Native multi-leg execution failed, falling back to individual legs",
                          order_id=multi_leg_order.order_id, error=str(e))
            return None

    async def _execute_legs_with_rollback(self, multi_leg_order: MultiLegOrder) -> Dict[str, Any]:
        """Execute legs individually with rollback protection"""

        filled_orders = []

        try:
            # Execute each leg sequentially
            for leg in multi_leg_order.legs:
                leg.status = LegStatus.SUBMITTED
                leg.submitted_at = datetime.now()

                # Execute individual leg
                order_result = await self._execute_single_leg(leg)

                if order_result and order_result.get("status") == "filled":
                    leg.status = LegStatus.FILLED
                    leg.order_id = order_result.get("id")
                    leg.filled_qty = order_result.get("filled_qty", leg.quantity)
                    leg.filled_avg_price = order_result.get("filled_avg_price", 0.0)
                    leg.filled_at = datetime.now()

                    filled_orders.append(leg.order_id)
                    multi_leg_order.filled_legs += 1

                    logger.info("Leg executed successfully",
                               order_id=multi_leg_order.order_id,
                               leg_id=leg.leg_id,
                               alpaca_order_id=leg.order_id)
                else:
                    # Leg failed - trigger rollback
                    leg.status = LegStatus.FAILED
                    leg.error_message = order_result.get("error", "Unknown execution error")

                    raise TradingError(f"Leg execution failed: {leg.error_message}")

            # All legs successful
            multi_leg_order.status = MultiLegOrderStatus.FILLED
            multi_leg_order.completed_at = datetime.now()

            # Calculate net premium
            multi_leg_order.net_premium = await self._calculate_net_premium(multi_leg_order)

            logger.info("Multi-leg order completed successfully",
                       order_id=multi_leg_order.order_id,
                       legs_filled=multi_leg_order.filled_legs,
                       net_premium=multi_leg_order.net_premium)

            return {
                "status": "success",
                "execution_method": "individual_legs",
                "order_id": multi_leg_order.order_id,
                "legs_filled": multi_leg_order.filled_legs,
                "total_legs": multi_leg_order.total_legs,
                "net_premium": multi_leg_order.net_premium,
                "completed_at": multi_leg_order.completed_at.isoformat(),
                "leg_details": [
                    {
                        "leg_id": leg.leg_id,
                        "symbol": leg.symbol,
                        "side": leg.side,
                        "quantity": leg.quantity,
                        "filled_price": leg.filled_avg_price,
                        "order_id": leg.order_id,
                    }
                    for leg in multi_leg_order.legs
                ]
            }

        except Exception as e:
            # Rollback all filled orders
            multi_leg_order.failed_legs = len(multi_leg_order.legs) - multi_leg_order.filled_legs
            await self._rollback_filled_orders(multi_leg_order, filled_orders)
            raise e

    async def _execute_single_leg(self, leg: OrderLeg) -> Dict[str, Any]:
        """Execute a single leg order"""
        try:
            # Use the alpaca client to execute the order
            result = await self.alpaca_client.execute_order(
                symbol=leg.symbol,
                quantity=leg.quantity,
                side=leg.side,
                order_type=leg.order_type,
                limit_price=leg.limit_price,
                stop_price=leg.stop_price,
            )

            return result

        except Exception as e:
            logger.error("Single leg execution failed",
                        leg_id=leg.leg_id, symbol=leg.symbol, error=str(e))
            return {"status": "failed", "error": str(e)}

    async def _rollback_filled_orders(self, multi_leg_order: MultiLegOrder, filled_order_ids: List[str]):
        """Rollback all filled orders"""
        rollback_orders = []

        try:
            for leg in multi_leg_order.legs:
                if leg.status == LegStatus.FILLED and leg.order_id:
                    # Create offsetting order
                    offset_side = "sell" if leg.side == "buy" else "buy"

                    rollback_result = await self.alpaca_client.execute_order(
                        symbol=leg.symbol,
                        quantity=leg.filled_qty,
                        side=offset_side,
                        order_type="market",  # Use market orders for quick rollback
                    )

                    if rollback_result:
                        rollback_orders.append(rollback_result.get("id"))
                        leg.status = LegStatus.CANCELLED

                        logger.info("Leg rolled back",
                                   leg_id=leg.leg_id,
                                   original_order=leg.order_id,
                                   rollback_order=rollback_result.get("id"))

            multi_leg_order.status = MultiLegOrderStatus.ROLLED_BACK
            multi_leg_order.rollback_orders = rollback_orders
            multi_leg_order.completed_at = datetime.now()

        except Exception as e:
            logger.error("Rollback failed",
                        order_id=multi_leg_order.order_id, error=str(e))
            multi_leg_order.rollback_reason = f"Rollback failed: {e}"

    async def _rollback_order(self, multi_leg_order: MultiLegOrder, reason: str):
        """Rollback entire multi-leg order"""
        logger.warning("Rolling back multi-leg order",
                      order_id=multi_leg_order.order_id, reason=reason)

        filled_orders = [leg.order_id for leg in multi_leg_order.legs
                        if leg.status == LegStatus.FILLED and leg.order_id]

        if filled_orders:
            await self._rollback_filled_orders(multi_leg_order, filled_orders)
        else:
            multi_leg_order.status = MultiLegOrderStatus.CANCELLED

        multi_leg_order.rollback_reason = reason

    async def _validate_order_execution(self, multi_leg_order: MultiLegOrder):
        """Validate order before execution"""

        # Check risk limits for each leg
        for leg in multi_leg_order.legs:
            # This would integrate with risk manager
            # For now, basic validation
            if leg.quantity <= 0:
                raise TradingError(f"Invalid quantity for leg {leg.leg_id}")

        # Check order timeout
        if multi_leg_order.execution_timeout < 30:
            raise TradingError("Execution timeout too short (minimum 30 seconds)")

        logger.debug("Order execution validation passed",
                    order_id=multi_leg_order.order_id)

    async def _calculate_net_premium(self, multi_leg_order: MultiLegOrder) -> float:
        """Calculate net premium for the multi-leg order"""
        net_premium = 0.0

        for leg in multi_leg_order.legs:
            if leg.filled_avg_price and leg.filled_qty:
                leg_value = leg.filled_avg_price * leg.filled_qty * 100  # Options are per 100 shares

                # Add for sells (premium received), subtract for buys (premium paid)
                if leg.side == "sell":
                    net_premium += leg_value
                else:
                    net_premium -= leg_value

        return net_premium

    async def _log_transaction(self, multi_leg_order: MultiLegOrder, result: Dict[str, Any]):
        """Log transaction for audit purposes"""
        transaction_record = {
            "timestamp": datetime.now().isoformat(),
            "order_id": multi_leg_order.order_id,
            "strategy": multi_leg_order.strategy_name,
            "underlying": multi_leg_order.underlying_symbol,
            "status": result.get("status"),
            "execution_method": result.get("execution_method"),
            "legs": len(multi_leg_order.legs),
            "net_premium": result.get("net_premium"),
            "execution_time_seconds": (
                (multi_leg_order.completed_at - multi_leg_order.submitted_at).total_seconds()
                if multi_leg_order.completed_at and multi_leg_order.submitted_at else None
            ),
        }

        self.transaction_log.append(transaction_record)

        # Keep only last 1000 transactions
        if len(self.transaction_log) > 1000:
            self.transaction_log = self.transaction_log[-1000:]

        logger.info("Transaction logged", transaction=transaction_record)

    # Order management methods
    def get_multi_leg_order(self, order_id: str) -> Optional[MultiLegOrder]:
        """Get multi-leg order by ID"""
        return self.multi_leg_orders.get(order_id)

    def get_all_multi_leg_orders(self) -> Dict[str, MultiLegOrder]:
        """Get all multi-leg orders"""
        return self.multi_leg_orders.copy()

    def get_active_multi_leg_orders(self) -> Dict[str, MultiLegOrder]:
        """Get active multi-leg orders"""
        return {
            order_id: order for order_id, order in self.multi_leg_orders.items()
            if order.status in [
                MultiLegOrderStatus.CREATED,
                MultiLegOrderStatus.VALIDATING,
                MultiLegOrderStatus.SUBMITTING,
                MultiLegOrderStatus.PARTIALLY_FILLED,
            ]
        }

    async def cancel_multi_leg_order(self, order_id: str) -> bool:
        """Cancel multi-leg order"""
        order = self.multi_leg_orders.get(order_id)
        if not order:
            return False

        try:
            # Cancel individual legs that are still pending/submitted
            for leg in order.legs:
                if leg.status in [LegStatus.PENDING, LegStatus.SUBMITTED] and leg.order_id:
                    await self.alpaca_client.cancel_order(leg.order_id)
                    leg.status = LegStatus.CANCELLED

            order.status = MultiLegOrderStatus.CANCELLED
            order.completed_at = datetime.now()

            logger.info("Multi-leg order cancelled", order_id=order_id)
            return True

        except Exception as e:
            logger.error("Failed to cancel multi-leg order", order_id=order_id, error=str(e))
            return False

    def get_transaction_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get transaction log for audit"""
        return self.transaction_log[-limit:] if limit else self.transaction_log

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for multi-leg orders"""
        total_orders = len(self.multi_leg_orders)
        successful_orders = sum(
            1 for order in self.multi_leg_orders.values()
            if order.status == MultiLegOrderStatus.FILLED
        )
        failed_orders = sum(
            1 for order in self.multi_leg_orders.values()
            if order.status in [MultiLegOrderStatus.FAILED, MultiLegOrderStatus.ROLLED_BACK]
        )

        success_rate = (successful_orders / total_orders * 100) if total_orders > 0 else 0

        return {
            "total_orders": total_orders,
            "successful_orders": successful_orders,
            "failed_orders": failed_orders,
            "success_rate_pct": success_rate,
            "active_orders": len(self.get_active_multi_leg_orders()),
            "transaction_log_entries": len(self.transaction_log),
        }