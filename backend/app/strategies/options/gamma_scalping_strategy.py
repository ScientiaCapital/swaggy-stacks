"""
Gamma Scalping Strategy

Delta-neutral strategy that profits from volatility by continuously rebalancing
to maintain delta neutrality. Captures profits from gamma exposure as the
underlying moves while hedging directional risk.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import math

import structlog
from app.core.exceptions import TradingError
from app.trading.alpaca_client import AlpacaClient
from app.trading.options_trading import OptionPosition, GreeksData, OptionType, get_options_trader
from app.trading.greeks_risk_manager import GreeksRiskManager, GreeksRiskMetrics
from app.rag.strategies.strategy_engines import StrategySignal

logger = structlog.get_logger()


class RebalanceAction(Enum):
    """Rebalancing action types"""
    BUY_STOCK = "buy_stock"
    SELL_STOCK = "sell_stock"
    NO_ACTION = "no_action"


@dataclass
class GammaScalpingConfig:
    """Configuration for Gamma Scalping strategy"""

    # Delta neutrality parameters
    max_absolute_delta: float = 500.0  # Maximum $500 delta exposure
    rebalance_threshold: float = 250.0  # Rebalance when delta exceeds $250
    min_rebalance_amount: float = 50.0  # Minimum $50 delta to rebalance

    # Monitoring intervals
    monitoring_interval_min: int = 1    # 1 minute minimum
    monitoring_interval_max: int = 3    # 3 minutes maximum
    high_volatility_interval: int = 30  # 30 seconds during high volatility

    # Position management
    min_gamma_threshold: float = 0.1    # Minimum gamma exposure for scalping
    max_gamma_exposure: float = 10.0    # Maximum total gamma exposure
    target_gamma_ratio: float = 0.05    # Target gamma as % of portfolio

    # Rebalancing parameters
    slippage_buffer: float = 0.002      # 0.2% slippage buffer
    max_rebalance_frequency: int = 10   # Max rebalances per hour
    commission_threshold: float = 2.0   # Min profit needed to cover commissions

    # Risk management
    max_overnight_delta: float = 100.0  # Max delta to hold overnight
    daily_pnl_limit: float = 1000.0     # Daily P&L limit
    max_position_size: int = 1000       # Max shares for rebalancing


@dataclass
class GammaPosition:
    """Individual gamma scalping position"""

    symbol: str
    option_positions: List[OptionPosition]
    stock_position: int = 0  # Net stock position for hedging
    target_delta: float = 0.0  # Target delta (usually 0 for neutral)

    # Greeks tracking
    current_delta: float = 0.0
    current_gamma: float = 0.0
    portfolio_weight: float = 0.0

    # Performance tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_rebalances: int = 0
    last_rebalance_time: Optional[datetime] = None

    # Risk metrics
    max_delta_today: float = 0.0
    min_delta_today: float = 0.0
    volatility_estimate: float = 0.0


@dataclass
class RebalanceOrder:
    """Rebalancing order details"""

    symbol: str
    action: RebalanceAction
    quantity: int
    expected_delta_change: float
    reason: str
    urgency: str  # LOW, MEDIUM, HIGH
    estimated_cost: float


class GammaScalpingStrategy:
    """
    Gamma Scalping Strategy Implementation

    A delta-neutral strategy that:
    1. Maintains delta neutrality through continuous rebalancing
    2. Profits from gamma exposure as underlying price moves
    3. Uses real-time Greeks monitoring for position management
    4. Automatically hedges with underlying stock positions
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        greeks_risk_manager: GreeksRiskManager,
        config: GammaScalpingConfig = None,
    ):
        self.alpaca_client = alpaca_client
        self.greeks_risk_manager = greeks_risk_manager
        self.config = config or GammaScalpingConfig()

        # Position tracking
        self.active_positions: Dict[str, GammaPosition] = {}
        self.rebalance_history: List[RebalanceOrder] = []

        # Monitoring state
        self.monitoring_task: Optional[asyncio.Task] = None
        self.last_monitoring_time: Optional[datetime] = None
        self.current_monitoring_interval: int = self.config.monitoring_interval_min

        # Performance tracking
        self.daily_pnl: float = 0.0
        self.total_rebalances_today: int = 0
        self.last_reset_date: Optional[datetime] = None

        # Options trader for Greeks calculations
        self.options_trader = get_options_trader()

        logger.info("Gamma Scalping strategy initialized", config=self.config)

    async def analyze_symbol(
        self,
        symbol: str,
        market_data: Dict[str, Any] = None,
    ) -> Optional[StrategySignal]:
        """
        Analyze symbol for gamma scalping opportunities

        Args:
            symbol: Underlying symbol to analyze
            market_data: Current market data context

        Returns:
            StrategySignal if opportunity found, None otherwise
        """
        try:
            # Check if we already have a gamma scalping position
            if symbol in self.active_positions:
                # Analyze existing position for rebalancing
                return await self._analyze_existing_position(symbol, market_data)
            else:
                # Look for new gamma scalping opportunities
                return await self._analyze_new_opportunity(symbol, market_data)

        except Exception as e:
            logger.error("Error analyzing symbol for Gamma Scalping", symbol=symbol, error=str(e))
            return None

    async def _analyze_existing_position(
        self, symbol: str, market_data: Dict[str, Any]
    ) -> Optional[StrategySignal]:
        """Analyze existing gamma scalping position for rebalancing needs"""

        position = self.active_positions[symbol]

        # Update current Greeks
        await self._update_position_greeks(position)

        # Check if rebalancing is needed
        rebalance_needed, rebalance_order = await self._check_rebalancing_needed(position)

        if rebalance_needed and rebalance_order:
            return await self._create_rebalancing_signal(position, rebalance_order, market_data)

        return None

    async def _analyze_new_opportunity(
        self, symbol: str, market_data: Dict[str, Any]
    ) -> Optional[StrategySignal]:
        """Analyze new gamma scalping opportunity"""

        # Get current options positions for this symbol
        options_positions = await self._get_current_options_positions(symbol)

        if not options_positions:
            logger.debug("No options positions found for gamma scalping", symbol=symbol)
            return None

        # Calculate current portfolio Greeks
        portfolio_greeks = await self._calculate_position_greeks(options_positions)

        if not portfolio_greeks:
            return None

        # Check if position has sufficient gamma for scalping
        if abs(portfolio_greeks["gamma"]) < self.config.min_gamma_threshold:
            logger.debug("Insufficient gamma for scalping", symbol=symbol, gamma=portfolio_greeks["gamma"])
            return None

        # Check if delta is outside neutral zone
        current_delta = portfolio_greeks["delta"]
        if abs(current_delta) > self.config.rebalance_threshold:
            # Create initial rebalancing signal
            return await self._create_initial_scalping_signal(symbol, portfolio_greeks, market_data)

        return None

    async def _update_position_greeks(self, position: GammaPosition):
        """Update Greeks for existing position"""
        try:
            if not position.option_positions:
                return

            # Calculate current Greeks
            portfolio_greeks = await self._calculate_position_greeks(position.option_positions)

            if portfolio_greeks:
                position.current_delta = portfolio_greeks["delta"]
                position.current_gamma = portfolio_greeks["gamma"]

                # Update daily min/max delta tracking
                position.max_delta_today = max(position.max_delta_today, abs(position.current_delta))
                position.min_delta_today = min(position.min_delta_today, abs(position.current_delta))

        except Exception as e:
            logger.error("Error updating position Greeks", symbol=position.symbol, error=str(e))

    async def _check_rebalancing_needed(self, position: GammaPosition) -> Tuple[bool, Optional[RebalanceOrder]]:
        """Check if position needs rebalancing"""
        try:
            current_delta = position.current_delta

            # Check delta threshold
            if abs(current_delta) < self.config.rebalance_threshold:
                return False, None

            # Check minimum rebalance amount
            if abs(current_delta) < self.config.min_rebalance_amount:
                return False, None

            # Check rebalance frequency limits
            if await self._is_rebalancing_too_frequent(position):
                return False, None

            # Calculate rebalancing order
            rebalance_order = await self._calculate_rebalancing_order(position)

            return True, rebalance_order

        except Exception as e:
            logger.error("Error checking rebalancing needs", symbol=position.symbol, error=str(e))
            return False, None

    async def _calculate_rebalancing_order(self, position: GammaPosition) -> RebalanceOrder:
        """Calculate the required rebalancing order"""
        try:
            current_delta = position.current_delta
            target_delta = position.target_delta  # Usually 0 for delta neutral

            # Calculate required delta change
            delta_change_needed = target_delta - current_delta

            # Convert delta to shares (1 delta ≈ 1 share)
            shares_needed = int(delta_change_needed)

            # Determine action
            if shares_needed > 0:
                action = RebalanceAction.BUY_STOCK
                quantity = abs(shares_needed)
            elif shares_needed < 0:
                action = RebalanceAction.SELL_STOCK
                quantity = abs(shares_needed)
            else:
                action = RebalanceAction.NO_ACTION
                quantity = 0

            # Determine urgency
            delta_ratio = abs(current_delta) / self.config.max_absolute_delta
            if delta_ratio > 0.8:
                urgency = "HIGH"
            elif delta_ratio > 0.5:
                urgency = "MEDIUM"
            else:
                urgency = "LOW"

            # Estimate cost (simplified)
            current_price = await self.alpaca_client.get_latest_price(position.symbol)
            estimated_cost = quantity * current_price * self.config.slippage_buffer if current_price else 0.0

            reason = f"Delta rebalancing: current={current_delta:.1f}, target={target_delta:.1f}"

            return RebalanceOrder(
                symbol=position.symbol,
                action=action,
                quantity=quantity,
                expected_delta_change=delta_change_needed,
                reason=reason,
                urgency=urgency,
                estimated_cost=estimated_cost,
            )

        except Exception as e:
            logger.error("Error calculating rebalancing order", symbol=position.symbol, error=str(e))
            raise TradingError(f"Failed to calculate rebalancing order: {e}")

    async def _create_rebalancing_signal(
        self,
        position: GammaPosition,
        rebalance_order: RebalanceOrder,
        market_data: Dict[str, Any],
    ) -> StrategySignal:
        """Create strategy signal for rebalancing"""

        if rebalance_order.action == RebalanceAction.NO_ACTION:
            return None

        # Map action to signal direction
        direction = "BUY" if rebalance_order.action == RebalanceAction.BUY_STOCK else "SELL"

        # Get current price for entry price
        current_price = await self.alpaca_client.get_latest_price(position.symbol)
        entry_price = current_price if current_price else 0.0

        # Set profit target and stop loss (for hedging, these are less relevant)
        profit_target = entry_price * 1.001  # Minimal profit target
        stop_loss = entry_price * 0.999     # Minimal stop loss

        # Calculate confidence based on urgency and gamma exposure
        confidence_map = {"LOW": 0.6, "MEDIUM": 0.75, "HIGH": 0.9}
        confidence = confidence_map.get(rebalance_order.urgency, 0.7)

        rationale = (
            f"Gamma scalping rebalance: {rebalance_order.reason}. "
            f"Current gamma: {position.current_gamma:.3f}, "
            f"Urgency: {rebalance_order.urgency}"
        )

        return StrategySignal(
            strategy="Gamma Scalping",
            symbol=position.symbol,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=profit_target,
            rationale=rationale,
            indicators_used=[
                "Portfolio Delta", "Gamma Exposure", "Greeks Risk Manager",
                "Delta Neutrality", "Rebalancing Threshold"
            ],
            market_context={
                "underlying_symbol": position.symbol,
                "strategy_type": "gamma_scalping_rebalance",
                "rebalance_order": {
                    "action": rebalance_order.action.value,
                    "quantity": rebalance_order.quantity,
                    "expected_delta_change": rebalance_order.expected_delta_change,
                    "urgency": rebalance_order.urgency,
                    "estimated_cost": rebalance_order.estimated_cost,
                },
                "position_metrics": {
                    "current_delta": position.current_delta,
                    "current_gamma": position.current_gamma,
                    "target_delta": position.target_delta,
                    "stock_position": position.stock_position,
                    "total_rebalances": position.total_rebalances,
                },
                "market_data": market_data or {},
            }
        )

    async def _create_initial_scalping_signal(
        self,
        symbol: str,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, Any],
    ) -> StrategySignal:
        """Create initial gamma scalping setup signal"""

        current_delta = portfolio_greeks["delta"]
        current_gamma = portfolio_greeks["gamma"]

        # Calculate initial hedging requirement
        shares_needed = int(-current_delta)  # Negative to hedge
        direction = "BUY" if shares_needed > 0 else "SELL"
        quantity = abs(shares_needed)

        # Get current price
        current_price = await self.alpaca_client.get_latest_price(symbol)
        entry_price = current_price if current_price else 0.0

        confidence = 0.8  # High confidence for initial setup

        rationale = (
            f"Initial gamma scalping setup for {symbol}. "
            f"Portfolio delta: {current_delta:.1f}, gamma: {current_gamma:.3f}. "
            f"Hedging with {quantity} shares to achieve delta neutrality."
        )

        return StrategySignal(
            strategy="Gamma Scalping",
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=entry_price * 0.95,  # 5% stop loss
            take_profit=entry_price * 1.05,  # 5% profit target
            rationale=rationale,
            indicators_used=[
                "Portfolio Greeks", "Delta Neutrality", "Gamma Exposure"
            ],
            market_context={
                "underlying_symbol": symbol,
                "strategy_type": "gamma_scalping_initial",
                "initial_hedge": {
                    "shares_needed": shares_needed,
                    "portfolio_delta": current_delta,
                    "portfolio_gamma": current_gamma,
                },
                "market_data": market_data or {},
            }
        )

    async def start_monitoring(self):
        """Start continuous delta monitoring and rebalancing"""
        if self.monitoring_task and not self.monitoring_task.done():
            logger.warning("Gamma scalping monitoring already active")
            return

        logger.info("Starting gamma scalping monitoring")
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop gamma scalping monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Gamma scalping monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop for gamma scalping"""
        while True:
            try:
                # Reset daily tracking if new day
                await self._check_daily_reset()

                # Update all positions
                for symbol, position in self.active_positions.items():
                    await self._update_position_greeks(position)

                    # Check for rebalancing needs
                    rebalance_needed, rebalance_order = await self._check_rebalancing_needed(position)

                    if rebalance_needed and rebalance_order:
                        logger.info("Rebalancing needed",
                                   symbol=symbol,
                                   action=rebalance_order.action.value,
                                   quantity=rebalance_order.quantity,
                                   urgency=rebalance_order.urgency)

                        # Execute rebalancing (in a real implementation)
                        await self._execute_rebalancing(position, rebalance_order)

                # Adjust monitoring interval based on market conditions
                await self._adjust_monitoring_interval()

                # Sleep until next monitoring cycle
                await asyncio.sleep(self.current_monitoring_interval * 60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in gamma scalping monitoring loop", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying

    async def _execute_rebalancing(self, position: GammaPosition, rebalance_order: RebalanceOrder):
        """Execute the rebalancing order"""
        try:
            if rebalance_order.action == RebalanceAction.NO_ACTION:
                return

            # Map to order parameters
            side = "buy" if rebalance_order.action == RebalanceAction.BUY_STOCK else "sell"

            # Execute stock order for hedging
            order = await self.alpaca_client.execute_order(
                symbol=position.symbol,
                quantity=rebalance_order.quantity,
                side=side,
                order_type="market",  # Use market orders for quick execution
            )

            # Update position tracking
            delta_change = rebalance_order.quantity if side == "buy" else -rebalance_order.quantity
            position.stock_position += delta_change
            position.total_rebalances += 1
            position.last_rebalance_time = datetime.now()

            # Update daily tracking
            self.total_rebalances_today += 1

            # Add to history
            self.rebalance_history.append(rebalance_order)

            logger.info("Rebalancing executed",
                       symbol=position.symbol,
                       side=side,
                       quantity=rebalance_order.quantity,
                       order_id=order.get("id"),
                       new_stock_position=position.stock_position)

        except Exception as e:
            logger.error("Failed to execute rebalancing",
                        symbol=position.symbol,
                        error=str(e))

    # Helper methods
    async def _get_current_options_positions(self, symbol: str) -> List[OptionPosition]:
        """Get current options positions for symbol"""
        try:
            # This would typically fetch from the portfolio management system
            # For now, return empty list (would be implemented with actual position tracking)
            return []
        except Exception as e:
            logger.error("Error getting options positions", symbol=symbol, error=str(e))
            return []

    async def _calculate_position_greeks(self, positions: List[OptionPosition]) -> Optional[Dict[str, float]]:
        """Calculate Greeks for a list of positions"""
        try:
            if not positions:
                return None

            return await self.options_trader.get_portfolio_greeks(positions)
        except Exception as e:
            logger.error("Error calculating position Greeks", error=str(e))
            return None

    async def _is_rebalancing_too_frequent(self, position: GammaPosition) -> bool:
        """Check if rebalancing is happening too frequently"""
        if not position.last_rebalance_time:
            return False

        time_since_last = datetime.now() - position.last_rebalance_time
        min_interval = timedelta(minutes=self.config.monitoring_interval_min)

        return time_since_last < min_interval

    async def _adjust_monitoring_interval(self):
        """Adjust monitoring interval based on market conditions"""
        try:
            # Get current volatility estimate
            max_volatility = max(
                (pos.volatility_estimate for pos in self.active_positions.values()),
                default=0.0
            )

            # Adjust interval based on volatility
            if max_volatility > 0.5:  # High volatility
                self.current_monitoring_interval = self.config.high_volatility_interval // 60  # Convert to minutes
            else:
                self.current_monitoring_interval = self.config.monitoring_interval_min

        except Exception as e:
            logger.error("Error adjusting monitoring interval", error=str(e))

    async def _check_daily_reset(self):
        """Check if we need to reset daily tracking"""
        current_date = datetime.now().date()

        if self.last_reset_date != current_date:
            self.daily_pnl = 0.0
            self.total_rebalances_today = 0
            self.last_reset_date = current_date

            # Reset daily min/max tracking for all positions
            for position in self.active_positions.values():
                position.max_delta_today = 0.0
                position.min_delta_today = 0.0

            logger.info("Daily gamma scalping metrics reset")

    # Position management methods
    def add_position(
        self,
        symbol: str,
        option_positions: List[OptionPosition],
        target_delta: float = 0.0,
    ):
        """Add new gamma scalping position"""
        position = GammaPosition(
            symbol=symbol,
            option_positions=option_positions,
            target_delta=target_delta,
        )

        self.active_positions[symbol] = position
        logger.info("Added gamma scalping position", symbol=symbol, target_delta=target_delta)

    def remove_position(self, symbol: str):
        """Remove gamma scalping position"""
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            del self.active_positions[symbol]

            logger.info("Removed gamma scalping position",
                       symbol=symbol,
                       final_pnl=position.realized_pnl + position.unrealized_pnl,
                       total_rebalances=position.total_rebalances)

    def get_position(self, symbol: str) -> Optional[GammaPosition]:
        """Get gamma scalping position"""
        return self.active_positions.get(symbol)

    def get_all_positions(self) -> Dict[str, GammaPosition]:
        """Get all active gamma scalping positions"""
        return self.active_positions.copy()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        total_positions = len(self.active_positions)
        total_pnl = sum(pos.realized_pnl + pos.unrealized_pnl for pos in self.active_positions.values())
        total_rebalances = sum(pos.total_rebalances for pos in self.active_positions.values())

        return {
            "strategy": "Gamma Scalping",
            "active_positions": total_positions,
            "daily_pnl": self.daily_pnl,
            "total_pnl": total_pnl,
            "total_rebalances_today": self.total_rebalances_today,
            "total_rebalances_all_time": total_rebalances,
            "positions": {
                symbol: {
                    "current_delta": pos.current_delta,
                    "current_gamma": pos.current_gamma,
                    "stock_position": pos.stock_position,
                    "total_rebalances": pos.total_rebalances,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                for symbol, pos in self.active_positions.items()
            },
            "monitoring_interval": self.current_monitoring_interval,
            "last_updated": datetime.now().isoformat(),
        }