"""
Zero Days to Expiration (Zero-DTE) Options Strategy

High-risk, high-reward strategy targeting options expiring within the same trading day.
Focuses on delta-based selection with sophisticated monitoring and exit conditions.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import structlog
from app.core.exceptions import TradingError
from app.trading.alpaca_client import AlpacaClient
from app.trading.options_trading import OptionContract, GreeksData, OptionType
from app.rag.strategies.strategy_engines import StrategySignal

logger = structlog.get_logger()


@dataclass
class ZeroDTEConfig:
    """Configuration for Zero-DTE strategy"""

    # Delta thresholds for option selection
    short_delta_min: float = -0.42
    short_delta_max: float = -0.38
    long_delta_min: float = -0.22
    long_delta_max: float = -0.18

    # Risk management
    profit_target_pct: float = 50.0  # 50% profit target
    stop_loss_multiplier: float = 2.0  # 2x delta stop-loss

    # Filtering criteria
    min_open_interest: int = 500
    min_spread_width: float = 2.0
    max_spread_width: float = 5.0

    # Monitoring
    monitoring_interval_minutes: int = 3

    # IV constraints
    min_implied_volatility: float = 0.15
    max_implied_volatility: float = 0.80


@dataclass
class ZeroDTEPosition:
    """Active Zero-DTE position tracking"""

    option_symbol: str
    underlying_symbol: str
    position_type: str  # SHORT or LONG
    entry_price: float
    entry_delta: float
    quantity: int
    profit_target: float
    stop_loss: float
    entry_time: datetime
    last_monitoring_time: datetime
    current_price: Optional[float] = None
    current_delta: Optional[float] = None
    unrealized_pnl: Optional[float] = None


class ZeroDTEStrategy:
    """
    Zero Days to Expiration Options Strategy

    This strategy trades options that expire within the same trading day,
    focusing on time decay (theta) and delta movements. High-risk strategy
    requiring sophisticated monitoring and quick exit conditions.
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        config: ZeroDTEConfig = None,
    ):
        self.alpaca_client = alpaca_client
        self.config = config or ZeroDTEConfig()
        self.active_positions: List[ZeroDTEPosition] = []
        self.monitoring_task: Optional[asyncio.Task] = None

        logger.info("Zero-DTE strategy initialized", config=self.config)

    async def analyze_symbol(
        self,
        symbol: str,
        market_data: Dict[str, Any] = None,
    ) -> Optional[StrategySignal]:
        """
        Analyze symbol for Zero-DTE opportunities

        Args:
            symbol: Underlying symbol to analyze
            market_data: Current market data context

        Returns:
            StrategySignal if opportunity found, None otherwise
        """
        try:
            # Get today's expiring options
            today = datetime.now().date()
            option_chain = await self.alpaca_client.get_option_chain(
                symbol=symbol,
                expiration_date=today.strftime("%Y-%m-%d"),
                limit=200
            )

            if not option_chain:
                logger.debug("No options found for today's expiration", symbol=symbol)
                return None

            # Get current underlying price
            underlying_price = await self.alpaca_client.get_latest_price(symbol)
            if not underlying_price:
                raise TradingError(f"Unable to get current price for {symbol}")

            # Find best Zero-DTE opportunity
            best_opportunity = await self._find_best_zero_dte_opportunity(
                symbol, option_chain, underlying_price, market_data
            )

            if not best_opportunity:
                return None

            return await self._create_strategy_signal(
                symbol, best_opportunity, underlying_price, market_data
            )

        except Exception as e:
            logger.error("Error analyzing symbol for Zero-DTE", symbol=symbol, error=str(e))
            return None

    async def _find_best_zero_dte_opportunity(
        self,
        symbol: str,
        option_chain: List[Dict[str, Any]],
        underlying_price: float,
        market_data: Dict[str, Any],
    ) -> Optional[Tuple[OptionContract, str, float]]:
        """Find the best Zero-DTE opportunity from option chain"""

        best_short_opportunity = None
        best_long_opportunity = None
        best_short_score = 0
        best_long_score = 0

        for option_data in option_chain:
            try:
                # Convert to OptionContract
                option = await self._convert_to_option_contract(option_data, symbol)

                # Apply basic filters
                if not await self._passes_basic_filters(option):
                    continue

                # Calculate Greeks if not provided
                if not option.greeks:
                    option.greeks = await self._calculate_greeks(option, underlying_price)

                if not option.greeks:
                    continue

                # Check for short opportunity (high delta puts/calls)
                if self.config.short_delta_min <= option.greeks.delta <= self.config.short_delta_max:
                    score = await self._score_short_opportunity(option, underlying_price)
                    if score > best_short_score:
                        best_short_score = score
                        best_short_opportunity = (option, "SHORT", score)

                # Check for long opportunity (low delta options)
                elif self.config.long_delta_min <= abs(option.greeks.delta) <= self.config.long_delta_max:
                    score = await self._score_long_opportunity(option, underlying_price)
                    if score > best_long_score:
                        best_long_score = score
                        best_long_opportunity = (option, "LONG", score)

            except Exception as e:
                logger.warning("Error processing option", option_data=option_data, error=str(e))
                continue

        # Return the better opportunity
        if best_short_opportunity and best_long_opportunity:
            return best_short_opportunity if best_short_score > best_long_score else best_long_opportunity
        elif best_short_opportunity:
            return best_short_opportunity
        elif best_long_opportunity:
            return best_long_opportunity
        else:
            return None

    async def _passes_basic_filters(self, option: OptionContract) -> bool:
        """Apply basic filtering criteria"""

        # Open interest filter
        if option.open_interest < self.config.min_open_interest:
            return False

        # Spread width filter
        spread_width = option.ask - option.bid
        if spread_width < self.config.min_spread_width or spread_width > self.config.max_spread_width:
            return False

        # Implied volatility filter
        if option.implied_volatility:
            if (option.implied_volatility < self.config.min_implied_volatility or
                option.implied_volatility > self.config.max_implied_volatility):
                return False

        # Must expire today
        expiry_date = option.expiration_date.date()
        today = datetime.now().date()
        if expiry_date != today:
            return False

        return True

    async def _score_short_opportunity(self, option: OptionContract, underlying_price: float) -> float:
        """Score short selling opportunity"""
        score = 0.0

        # Higher theta (time decay) is better for short positions
        if option.greeks.theta < 0:
            score += abs(option.greeks.theta) * 10

        # Lower gamma is better (less delta movement risk)
        if option.greeks.gamma > 0:
            score += (1.0 / option.greeks.gamma) * 5

        # Higher implied volatility is better for selling
        if option.implied_volatility:
            score += option.implied_volatility * 20

        # Better liquidity (tighter spread) gets higher score
        spread_pct = (option.ask - option.bid) / option.current_price if option.current_price > 0 else 1.0
        score += (1.0 / max(spread_pct, 0.01)) * 5

        # Higher open interest is better
        score += min(option.open_interest / 1000, 10)

        return score

    async def _score_long_opportunity(self, option: OptionContract, underlying_price: float) -> float:
        """Score long buying opportunity"""
        score = 0.0

        # Higher gamma is better for long positions (more delta acceleration)
        if option.greeks.gamma > 0:
            score += option.greeks.gamma * 20

        # Lower time decay (less negative theta) is better
        if option.greeks.theta < 0:
            score += (1.0 / abs(option.greeks.theta)) * 5

        # Lower implied volatility is better for buying
        if option.implied_volatility:
            score += (1.0 / max(option.implied_volatility, 0.1)) * 15

        # Better liquidity gets higher score
        spread_pct = (option.ask - option.bid) / option.current_price if option.current_price > 0 else 1.0
        score += (1.0 / max(spread_pct, 0.01)) * 5

        # Higher open interest is better
        score += min(option.open_interest / 1000, 10)

        return score

    async def _create_strategy_signal(
        self,
        symbol: str,
        opportunity: Tuple[OptionContract, str, float],
        underlying_price: float,
        market_data: Dict[str, Any],
    ) -> StrategySignal:
        """Create strategy signal from opportunity"""

        option, position_type, score = opportunity

        # Calculate entry price (use mid-price)
        entry_price = (option.bid + option.ask) / 2

        # Calculate profit target and stop loss
        if position_type == "SHORT":
            profit_target = entry_price * (1 - self.config.profit_target_pct / 100)
            stop_loss = entry_price * (1 + self.config.stop_loss_multiplier * abs(option.greeks.delta))
            direction = "SELL"
        else:  # LONG
            profit_target = entry_price * (1 + self.config.profit_target_pct / 100)
            stop_loss = entry_price * (1 - self.config.stop_loss_multiplier * abs(option.greeks.delta))
            direction = "BUY"

        # Calculate confidence based on score and market conditions
        confidence = min(score / 100, 0.95)  # Cap at 95%

        # Build rationale
        rationale = (
            f"Zero-DTE {position_type} opportunity on {option.symbol}. "
            f"Delta: {option.greeks.delta:.3f}, "
            f"Theta: {option.greeks.theta:.3f}, "
            f"IV: {option.implied_volatility:.2f}, "
            f"OI: {option.open_interest}, "
            f"Score: {score:.1f}"
        )

        return StrategySignal(
            strategy="Zero-DTE",
            symbol=option.symbol,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=profit_target,
            rationale=rationale,
            indicators_used=[
                "Delta", "Theta", "Gamma", "Implied Volatility",
                "Open Interest", "Bid-Ask Spread"
            ],
            market_context={
                "underlying_symbol": symbol,
                "underlying_price": underlying_price,
                "position_type": position_type,
                "option_type": option.option_type.value,
                "strike_price": option.strike_price,
                "expiration_date": option.expiration_date.isoformat(),
                "greeks": {
                    "delta": option.greeks.delta,
                    "gamma": option.greeks.gamma,
                    "theta": option.greeks.theta,
                    "vega": option.greeks.vega,
                },
                "implied_volatility": option.implied_volatility,
                "open_interest": option.open_interest,
                "volume": option.volume,
                "opportunity_score": score,
                "market_data": market_data or {},
            }
        )

    async def _convert_to_option_contract(
        self, option_data: Dict[str, Any], underlying_symbol: str
    ) -> OptionContract:
        """Convert API option data to OptionContract"""

        # Parse option symbol and extract details
        symbol = option_data.get("symbol", "")

        # Extract option details from symbol or data
        strike_price = float(option_data.get("strike_price", 0))
        option_type_str = option_data.get("type", "call").upper()
        option_type = OptionType.CALL if option_type_str == "CALL" else OptionType.PUT

        # Parse expiration date
        exp_date_str = option_data.get("expiration_date", "")
        expiration_date = datetime.fromisoformat(exp_date_str.replace("Z", "+00:00"))

        return OptionContract(
            symbol=symbol,
            underlying_symbol=underlying_symbol,
            option_type=option_type,
            strike_price=strike_price,
            expiration_date=expiration_date,
            current_price=float(option_data.get("last_price", 0)),
            bid=float(option_data.get("bid_price", 0)),
            ask=float(option_data.get("ask_price", 0)),
            volume=int(option_data.get("volume", 0)),
            open_interest=int(option_data.get("open_interest", 0)),
            implied_volatility=option_data.get("implied_volatility"),
        )

    async def _calculate_greeks(
        self, option: OptionContract, underlying_price: float
    ) -> Optional[GreeksData]:
        """Calculate Greeks for option using Black-Scholes"""
        try:
            from app.trading.options_trading import get_options_trader

            options_trader = get_options_trader()

            # Calculate time to expiration in years
            time_to_expiry = (option.expiration_date - datetime.now()).total_seconds() / (365.25 * 24 * 3600)

            # Use current mid-price if available, otherwise use last price
            option_price = (option.bid + option.ask) / 2 if option.bid > 0 and option.ask > 0 else option.current_price

            if option_price <= 0:
                return None

            # Calculate Greeks using Black-Scholes
            greeks = options_trader.bs_calculator.calculate_greeks(
                underlying_price=underlying_price,
                strike_price=option.strike_price,
                time_to_expiry=time_to_expiry,
                risk_free_rate=0.05,  # Approximate current risk-free rate
                volatility=option.implied_volatility or 0.25,  # Default to 25% if IV not available
                option_type=option.option_type,
            )

            return greeks

        except Exception as e:
            logger.warning("Error calculating Greeks", option=option.symbol, error=str(e))
            return None

    async def start_monitoring(self):
        """Start monitoring active positions"""
        if self.monitoring_task and not self.monitoring_task.done():
            logger.warning("Monitoring already active")
            return

        logger.info("Starting Zero-DTE position monitoring")
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop monitoring active positions"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Zero-DTE monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop for active positions"""
        while True:
            try:
                await self._update_active_positions()
                await self._check_exit_conditions()
                await asyncio.sleep(self.config.monitoring_interval_minutes * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying

    async def _update_active_positions(self):
        """Update current prices and Greeks for active positions"""
        for position in self.active_positions:
            try:
                # Get current option quote
                quote = await self.alpaca_client.get_option_quote(position.option_symbol)
                if quote:
                    position.current_price = (quote["bid_price"] + quote["ask_price"]) / 2
                    position.unrealized_pnl = self._calculate_unrealized_pnl(position)

                position.last_monitoring_time = datetime.now()

            except Exception as e:
                logger.warning("Error updating position", position=position.option_symbol, error=str(e))

    async def _check_exit_conditions(self):
        """Check if any positions should be closed"""
        positions_to_close = []

        for position in self.active_positions:
            should_exit, reason = self._should_exit_position(position)
            if should_exit:
                positions_to_close.append((position, reason))

        for position, reason in positions_to_close:
            await self._close_position(position, reason)

    def _should_exit_position(self, position: ZeroDTEPosition) -> Tuple[bool, str]:
        """Check if position should be exited"""

        if not position.current_price:
            return False, ""

        # Check profit target
        if position.position_type == "SHORT":
            if position.current_price <= position.profit_target:
                return True, "Profit target reached"
            if position.current_price >= position.stop_loss:
                return True, "Stop loss triggered"
        else:  # LONG
            if position.current_price >= position.profit_target:
                return True, "Profit target reached"
            if position.current_price <= position.stop_loss:
                return True, "Stop loss triggered"

        # Check time-based exit (close 30 minutes before expiration)
        time_to_expiry = (position.entry_time.replace(hour=16, minute=0, second=0) - datetime.now()).total_seconds()
        if time_to_expiry <= 30 * 60:  # 30 minutes
            return True, "Time-based exit (30 min before expiration)"

        return False, ""

    async def _close_position(self, position: ZeroDTEPosition, reason: str):
        """Close an active position"""
        try:
            # Determine order side (opposite of entry)
            side = "buy" if position.position_type == "SHORT" else "sell"

            # Execute closing order
            order = await self.alpaca_client.execute_order(
                symbol=position.option_symbol,
                quantity=position.quantity,
                side=side,
                order_type="market",
            )

            logger.info(
                "Zero-DTE position closed",
                symbol=position.option_symbol,
                reason=reason,
                order_id=order.get("id"),
                unrealized_pnl=position.unrealized_pnl,
            )

            # Remove from active positions
            self.active_positions.remove(position)

        except Exception as e:
            logger.error("Error closing position", position=position.option_symbol, error=str(e))

    def _calculate_unrealized_pnl(self, position: ZeroDTEPosition) -> float:
        """Calculate unrealized P&L for position"""
        if not position.current_price:
            return 0.0

        if position.position_type == "SHORT":
            return (position.entry_price - position.current_price) * position.quantity * 100
        else:  # LONG
            return (position.current_price - position.entry_price) * position.quantity * 100

    def add_position(
        self,
        option_symbol: str,
        underlying_symbol: str,
        position_type: str,
        entry_price: float,
        entry_delta: float,
        quantity: int,
        profit_target: float,
        stop_loss: float,
    ):
        """Add a new position to monitoring"""
        position = ZeroDTEPosition(
            option_symbol=option_symbol,
            underlying_symbol=underlying_symbol,
            position_type=position_type,
            entry_price=entry_price,
            entry_delta=entry_delta,
            quantity=quantity,
            profit_target=profit_target,
            stop_loss=stop_loss,
            entry_time=datetime.now(),
            last_monitoring_time=datetime.now(),
        )

        self.active_positions.append(position)
        logger.info("Added Zero-DTE position to monitoring", position=position)

    def get_active_positions(self) -> List[ZeroDTEPosition]:
        """Get all active positions"""
        return self.active_positions.copy()

    async def cleanup_expired_positions(self):
        """Clean up positions that have expired"""
        current_time = datetime.now()
        expired_positions = []

        for position in self.active_positions:
            # Assume options expire at 4 PM ET
            expiry_time = position.entry_time.replace(hour=16, minute=0, second=0, microsecond=0)
            if current_time >= expiry_time:
                expired_positions.append(position)

        for position in expired_positions:
            logger.info("Removing expired Zero-DTE position", position=position.option_symbol)
            self.active_positions.remove(position)