"""
Iron Condor Options Strategy

Four-leg neutral strategy for range-bound markets that profits from low volatility.
Combines a bull put spread and a bear call spread to create a profit zone
between the short strikes.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

import structlog
from app.core.exceptions import TradingError
from app.trading.alpaca_client import AlpacaClient
from app.trading.options_trading import OptionContract, GreeksData, OptionType
from app.rag.strategies.strategy_engines import StrategySignal

logger = structlog.get_logger()


class IronCondorLeg(Enum):
    """Iron Condor legs"""
    SHORT_PUT = "short_put"      # Sell put (bull put spread)
    LONG_PUT = "long_put"        # Buy put (protection)
    SHORT_CALL = "short_call"    # Sell call (bear call spread)
    LONG_CALL = "long_call"      # Buy call (protection)


@dataclass
class IronCondorConfig:
    """Configuration for Iron Condor strategy"""

    # Range parameters
    range_pct: float = 15.0  # 15% range around underlying price
    profit_target_pct: float = 40.0  # Target 40% of max profit

    # IV filtering
    min_iv: float = 0.10  # 10% minimum IV
    max_iv: float = 0.40  # 40% maximum IV

    # Strike selection
    std_dev_multiplier: float = 1.0  # Standard deviation for strike spacing
    min_strike_width: float = 5.0   # Minimum $5 between strikes
    max_strike_width: float = 20.0  # Maximum $20 between strikes

    # Expiration range
    min_days_to_expiry: int = 14    # Minimum 2 weeks
    max_days_to_expiry: int = 45    # Maximum 6 weeks

    # Filtering criteria
    min_open_interest: int = 100
    max_bid_ask_spread_pct: float = 10.0

    # Risk management
    max_loss_pct: float = 200.0     # Stop at 200% of credit received
    manage_at_profit_pct: float = 25.0  # Consider closing at 25% profit


@dataclass
class IronCondorLeg:
    """Individual leg of Iron Condor"""
    leg_type: IronCondorLeg
    option_contract: OptionContract
    quantity: int  # Positive for long, negative for short
    premium: float  # Premium paid (positive) or received (negative)


@dataclass
class IronCondorPosition:
    """Complete Iron Condor position"""

    underlying_symbol: str
    entry_time: datetime
    expiration_date: datetime

    # The four legs
    short_put: IronCondorLeg
    long_put: IronCondorLeg
    short_call: IronCondorLeg
    long_call: IronCondorLeg

    # Position metrics
    net_credit: float  # Total premium received (net credit)
    max_profit: float  # Maximum profit potential
    max_loss: float    # Maximum loss potential
    profit_target: float  # Target profit for closing
    stop_loss: float   # Stop loss threshold

    # Strike prices for easy reference
    short_put_strike: float
    long_put_strike: float
    short_call_strike: float
    long_call_strike: float

    # Current status
    current_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    days_to_expiry: Optional[int] = None
    is_profitable: bool = False


class IronCondorStrategy:
    """
    Iron Condor Options Strategy

    A market-neutral strategy that profits from low volatility and range-bound
    price action. The strategy involves:
    1. Selling an out-of-the-money put (bull put spread lower leg)
    2. Buying a further out-of-the-money put (protection)
    3. Selling an out-of-the-money call (bear call spread upper leg)
    4. Buying a further out-of-the-money call (protection)

    Profits when the underlying stays between the short strikes.
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        config: IronCondorConfig = None,
    ):
        self.alpaca_client = alpaca_client
        self.config = config or IronCondorConfig()
        self.active_positions: Dict[str, IronCondorPosition] = {}

        logger.info("Iron Condor strategy initialized", config=self.config)

    async def analyze_symbol(
        self,
        symbol: str,
        market_data: Dict[str, Any] = None,
    ) -> Optional[StrategySignal]:
        """
        Analyze symbol for Iron Condor opportunities

        Args:
            symbol: Underlying symbol to analyze
            market_data: Current market data context

        Returns:
            StrategySignal if opportunity found, None otherwise
        """
        try:
            # Check if we already have a position
            if symbol in self.active_positions:
                logger.debug("Position already exists", symbol=symbol)
                return None

            # Get current underlying price
            underlying_price = await self.alpaca_client.get_latest_price(symbol)
            if not underlying_price:
                raise TradingError(f"Unable to get current price for {symbol}")

            # Calculate volatility and market metrics
            market_metrics = await self._calculate_market_metrics(symbol, underlying_price)
            if not market_metrics:
                return None

            # Check if market conditions are suitable for Iron Condor
            if not await self._is_suitable_market_conditions(market_metrics):
                logger.debug("Market conditions not suitable", symbol=symbol, metrics=market_metrics)
                return None

            # Find optimal Iron Condor setup
            condor_setup = await self._find_optimal_iron_condor(
                symbol, underlying_price, market_metrics
            )

            if not condor_setup:
                return None

            return await self._create_iron_condor_signal(
                symbol, condor_setup, underlying_price, market_metrics, market_data
            )

        except Exception as e:
            logger.error("Error analyzing symbol for Iron Condor", symbol=symbol, error=str(e))
            return None

    async def _calculate_market_metrics(
        self, symbol: str, underlying_price: float
    ) -> Optional[Dict[str, float]]:
        """Calculate market metrics for Iron Condor suitability"""
        try:
            # Get historical data for volatility calculation
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            historical_data = await self.alpaca_client.get_historical_data(
                symbol=symbol,
                start=start_date,
                end=end_date,
                timeframe="1Day"
            )

            if not historical_data or len(historical_data) < 20:
                return None

            # Calculate realized volatility (20-day)
            closes = [float(bar["close"]) for bar in historical_data[-20:]]
            returns = [(closes[i] / closes[i-1] - 1) for i in range(1, len(closes))]

            if not returns:
                return None

            # Calculate standard deviation and annualize
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            realized_vol = math.sqrt(variance * 252)  # Annualized

            # Calculate average true range for recent volatility
            atr_values = []
            for i in range(1, min(15, len(historical_data))):
                bar = historical_data[-i]
                prev_close = float(historical_data[-i-1]["close"])
                high = float(bar["high"])
                low = float(bar["low"])

                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                atr_values.append(tr)

            avg_atr = sum(atr_values) / len(atr_values) if atr_values else 0
            atr_pct = (avg_atr / underlying_price) * 100

            return {
                "realized_volatility": realized_vol,
                "atr_percent": atr_pct,
                "price_stability": 1.0 / (realized_vol + 0.01),  # Higher = more stable
                "recent_range_pct": (max(closes[-10:]) - min(closes[-10:])) / underlying_price * 100,
            }

        except Exception as e:
            logger.error("Error calculating market metrics", symbol=symbol, error=str(e))
            return None

    async def _is_suitable_market_conditions(self, metrics: Dict[str, float]) -> bool:
        """Check if market conditions are suitable for Iron Condor"""

        # Prefer low to moderate volatility
        if metrics["realized_volatility"] > 0.50:  # 50% annualized vol
            return False

        # Prefer stable, range-bound conditions
        if metrics["recent_range_pct"] > 15.0:  # 15% recent range
            return False

        # Need sufficient but not excessive volatility for premium
        if metrics["realized_volatility"] < 0.10:  # 10% minimum
            return False

        return True

    async def _find_optimal_iron_condor(
        self,
        symbol: str,
        underlying_price: float,
        market_metrics: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Find optimal Iron Condor setup"""

        best_setup = None
        best_score = 0

        # Try different expiration dates
        min_date = datetime.now() + timedelta(days=self.config.min_days_to_expiry)
        max_date = datetime.now() + timedelta(days=self.config.max_days_to_expiry)

        current_date = min_date
        while current_date <= max_date:
            try:
                # Get options chain for this expiration
                options_chain = await self.alpaca_client.get_option_chain(
                    symbol=symbol,
                    expiration_date=current_date.strftime("%Y-%m-%d"),
                    limit=200
                )

                if not options_chain:
                    current_date += timedelta(days=1)
                    continue

                # Find optimal strike configuration for this expiration
                setup = await self._find_optimal_strikes(
                    symbol, options_chain, underlying_price, market_metrics, current_date
                )

                if setup:
                    score = await self._score_iron_condor_setup(setup, market_metrics)
                    if score > best_score:
                        best_score = score
                        best_setup = setup

            except Exception as e:
                logger.warning("Error processing expiration", date=current_date, error=str(e))

            current_date += timedelta(days=7)  # Check weekly expirations

        return best_setup

    async def _find_optimal_strikes(
        self,
        symbol: str,
        options_chain: List[Dict[str, Any]],
        underlying_price: float,
        market_metrics: Dict[str, float],
        expiration_date: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Find optimal strike prices for Iron Condor"""

        # Separate puts and calls
        puts = [opt for opt in options_chain if opt.get("type", "").upper() == "PUT"]
        calls = [opt for opt in options_chain if opt.get("type", "").upper() == "CALL"]

        if len(puts) < 2 or len(calls) < 2:
            return None

        # Calculate target strikes based on range and standard deviation
        vol_estimate = market_metrics.get("realized_volatility", 0.25)
        days_to_expiry = (expiration_date - datetime.now()).days
        time_to_expiry = days_to_expiry / 365.0

        # Calculate expected move (1 standard deviation)
        expected_move = underlying_price * vol_estimate * math.sqrt(time_to_expiry)

        # Target strikes for Iron Condor
        range_width = underlying_price * (self.config.range_pct / 100)
        target_short_put = underlying_price - (range_width / 2)
        target_short_call = underlying_price + (range_width / 2)

        # Find actual strikes closest to targets
        short_put_option = await self._find_closest_strike(puts, target_short_put, "PUT")
        short_call_option = await self._find_closest_strike(calls, target_short_call, "CALL")

        if not short_put_option or not short_call_option:
            return None

        # Find protection strikes (long options)
        short_put_strike = float(short_put_option.get("strike_price", 0))
        short_call_strike = float(short_call_option.get("strike_price", 0))

        # Long put should be below short put
        target_long_put = short_put_strike - max(self.config.min_strike_width, expected_move * 0.5)
        long_put_option = await self._find_closest_strike(puts, target_long_put, "PUT", below=True)

        # Long call should be above short call
        target_long_call = short_call_strike + max(self.config.min_strike_width, expected_move * 0.5)
        long_call_option = await self._find_closest_strike(calls, target_long_call, "CALL", above=True)

        if not long_put_option or not long_call_option:
            return None

        # Validate the setup
        if not await self._validate_iron_condor_setup(
            short_put_option, long_put_option, short_call_option, long_call_option
        ):
            return None

        return {
            "symbol": symbol,
            "underlying_price": underlying_price,
            "expiration_date": expiration_date,
            "short_put": short_put_option,
            "long_put": long_put_option,
            "short_call": short_call_option,
            "long_call": long_call_option,
            "expected_move": expected_move,
            "market_metrics": market_metrics,
        }

    async def _find_closest_strike(
        self,
        options: List[Dict[str, Any]],
        target_strike: float,
        option_type: str,
        below: bool = False,
        above: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Find option with strike closest to target"""

        valid_options = []

        for option in options:
            strike = float(option.get("strike_price", 0))

            # Apply directional filter if specified
            if below and strike >= target_strike:
                continue
            if above and strike <= target_strike:
                continue

            # Apply basic filters
            if not await self._passes_basic_option_filters(option):
                continue

            valid_options.append((option, abs(strike - target_strike)))

        if not valid_options:
            return None

        # Return option with closest strike
        valid_options.sort(key=lambda x: x[1])
        return valid_options[0][0]

    async def _passes_basic_option_filters(self, option: Dict[str, Any]) -> bool:
        """Apply basic filters to option"""

        # Open interest filter
        open_interest = int(option.get("open_interest", 0))
        if open_interest < self.config.min_open_interest:
            return False

        # Bid-ask spread filter
        bid = float(option.get("bid_price", 0))
        ask = float(option.get("ask_price", 0))

        if bid <= 0 or ask <= 0:
            return False

        mid_price = (bid + ask) / 2
        if mid_price <= 0:
            return False

        spread_pct = ((ask - bid) / mid_price) * 100
        if spread_pct > self.config.max_bid_ask_spread_pct:
            return False

        # IV filter
        iv = option.get("implied_volatility")
        if iv and (iv < self.config.min_iv or iv > self.config.max_iv):
            return False

        return True

    async def _validate_iron_condor_setup(
        self,
        short_put: Dict[str, Any],
        long_put: Dict[str, Any],
        short_call: Dict[str, Any],
        long_call: Dict[str, Any],
    ) -> bool:
        """Validate Iron Condor setup"""

        # Extract strike prices
        short_put_strike = float(short_put.get("strike_price", 0))
        long_put_strike = float(long_put.get("strike_price", 0))
        short_call_strike = float(short_call.get("strike_price", 0))
        long_call_strike = float(long_call.get("strike_price", 0))

        # Validate strike ordering
        if not (long_put_strike < short_put_strike < short_call_strike < long_call_strike):
            return False

        # Validate strike widths
        put_spread_width = short_put_strike - long_put_strike
        call_spread_width = long_call_strike - short_call_strike

        if (put_spread_width < self.config.min_strike_width or
            call_spread_width < self.config.min_strike_width):
            return False

        if (put_spread_width > self.config.max_strike_width or
            call_spread_width > self.config.max_strike_width):
            return False

        # Validate that we can collect net credit
        short_put_mid = (float(short_put.get("bid_price", 0)) + float(short_put.get("ask_price", 0))) / 2
        long_put_mid = (float(long_put.get("bid_price", 0)) + float(long_put.get("ask_price", 0))) / 2
        short_call_mid = (float(short_call.get("bid_price", 0)) + float(short_call.get("ask_price", 0))) / 2
        long_call_mid = (float(long_call.get("bid_price", 0)) + float(long_call.get("ask_price", 0))) / 2

        net_credit = short_put_mid + short_call_mid - long_put_mid - long_call_mid

        if net_credit <= 0:
            return False

        return True

    async def _score_iron_condor_setup(
        self, setup: Dict[str, Any], market_metrics: Dict[str, float]
    ) -> float:
        """Score Iron Condor setup quality"""

        score = 0.0

        # Calculate position metrics
        short_put = setup["short_put"]
        long_put = setup["long_put"]
        short_call = setup["short_call"]
        long_call = setup["long_call"]

        # Premium collection score
        short_put_mid = (float(short_put.get("bid_price", 0)) + float(short_put.get("ask_price", 0))) / 2
        long_put_mid = (float(long_put.get("bid_price", 0)) + float(long_put.get("ask_price", 0))) / 2
        short_call_mid = (float(short_call.get("bid_price", 0)) + float(short_call.get("ask_price", 0))) / 2
        long_call_mid = (float(long_call.get("bid_price", 0)) + float(long_call.get("ask_price", 0))) / 2

        net_credit = short_put_mid + short_call_mid - long_put_mid - long_call_mid

        # Calculate max profit/loss
        put_spread_width = float(short_put.get("strike_price", 0)) - float(long_put.get("strike_price", 0))
        call_spread_width = float(long_call.get("strike_price", 0)) - float(short_call.get("strike_price", 0))
        max_loss = max(put_spread_width, call_spread_width) - net_credit

        if max_loss > 0:
            risk_reward_ratio = net_credit / max_loss
            score += risk_reward_ratio * 25  # Reward good risk/reward

        # Probability of profit (simplified)
        underlying_price = setup["underlying_price"]
        short_put_strike = float(short_put.get("strike_price", 0))
        short_call_strike = float(short_call.get("strike_price", 0))

        profit_zone_width = short_call_strike - short_put_strike
        profit_zone_pct = profit_zone_width / underlying_price * 100
        score += min(profit_zone_pct, 20)  # Wider profit zone = higher score

        # Time decay advantage
        days_to_expiry = (setup["expiration_date"] - datetime.now()).days
        if 14 <= days_to_expiry <= 45:  # Sweet spot for Iron Condors
            score += 15

        # Volatility suitability
        realized_vol = market_metrics.get("realized_volatility", 0.25)
        if 0.15 <= realized_vol <= 0.35:  # Optimal vol range
            score += 10

        # Liquidity score
        total_oi = (int(short_put.get("open_interest", 0)) +
                   int(long_put.get("open_interest", 0)) +
                   int(short_call.get("open_interest", 0)) +
                   int(long_call.get("open_interest", 0)))
        score += min(total_oi / 1000, 10)

        return score

    async def _create_iron_condor_signal(
        self,
        symbol: str,
        setup: Dict[str, Any],
        underlying_price: float,
        market_metrics: Dict[str, float],
        market_data: Dict[str, Any],
    ) -> StrategySignal:
        """Create strategy signal for Iron Condor"""

        # Calculate position metrics
        short_put = setup["short_put"]
        long_put = setup["long_put"]
        short_call = setup["short_call"]
        long_call = setup["long_call"]

        # Calculate net credit and profit/loss
        short_put_mid = (float(short_put.get("bid_price", 0)) + float(short_put.get("ask_price", 0))) / 2
        long_put_mid = (float(long_put.get("bid_price", 0)) + float(long_put.get("ask_price", 0))) / 2
        short_call_mid = (float(short_call.get("bid_price", 0)) + float(short_call.get("ask_price", 0))) / 2
        long_call_mid = (float(long_call.get("bid_price", 0)) + float(long_call.get("ask_price", 0))) / 2

        net_credit = short_put_mid + short_call_mid - long_put_mid - long_call_mid

        # Calculate max profit/loss
        put_spread_width = float(short_put.get("strike_price", 0)) - float(long_put.get("strike_price", 0))
        call_spread_width = float(long_call.get("strike_price", 0)) - float(short_call.get("strike_price", 0))
        max_loss = max(put_spread_width, call_spread_width) - net_credit

        # Profit target and stop loss
        profit_target = net_credit * (self.config.profit_target_pct / 100)
        stop_loss = net_credit + (max_loss * 0.5)  # 50% of max loss

        # Calculate confidence
        risk_reward = net_credit / max_loss if max_loss > 0 else 0
        confidence = min(0.75 + (risk_reward * 0.1), 0.95)

        # Build rationale
        short_put_strike = float(short_put.get("strike_price", 0))
        short_call_strike = float(short_call.get("strike_price", 0))
        profit_zone = short_call_strike - short_put_strike

        rationale = (
            f"Iron Condor on {symbol}: Profit zone ${short_put_strike:.0f}-${short_call_strike:.0f} "
            f"(${profit_zone:.0f} wide, {profit_zone/underlying_price:.1%} range). "
            f"Net credit: ${net_credit:.2f}, Max profit: ${net_credit:.2f}, "
            f"Max loss: ${max_loss:.2f}, R/R: {risk_reward:.2f}"
        )

        return StrategySignal(
            strategy="Iron Condor",
            symbol=f"{symbol}_IC_{setup['expiration_date'].strftime('%Y%m%d')}",
            direction="SELL",  # Net credit strategy
            confidence=confidence,
            entry_price=net_credit,
            stop_loss=stop_loss,
            take_profit=profit_target,
            rationale=rationale,
            indicators_used=[
                "Realized Volatility", "Expected Move", "Strike Selection",
                "Risk-Reward Ratio", "Time Decay", "Open Interest"
            ],
            market_context={
                "underlying_symbol": symbol,
                "underlying_price": underlying_price,
                "strategy_type": "Iron Condor",
                "legs": {
                    "short_put": {
                        "symbol": short_put.get("symbol"),
                        "strike": float(short_put.get("strike_price", 0)),
                        "premium": short_put_mid,
                        "action": "SELL"
                    },
                    "long_put": {
                        "symbol": long_put.get("symbol"),
                        "strike": float(long_put.get("strike_price", 0)),
                        "premium": long_put_mid,
                        "action": "BUY"
                    },
                    "short_call": {
                        "symbol": short_call.get("symbol"),
                        "strike": float(short_call.get("strike_price", 0)),
                        "premium": short_call_mid,
                        "action": "SELL"
                    },
                    "long_call": {
                        "symbol": long_call.get("symbol"),
                        "strike": float(long_call.get("strike_price", 0)),
                        "premium": long_call_mid,
                        "action": "BUY"
                    }
                },
                "position_metrics": {
                    "net_credit": net_credit,
                    "max_profit": net_credit,
                    "max_loss": max_loss,
                    "profit_zone_width": profit_zone,
                    "profit_zone_pct": profit_zone / underlying_price * 100,
                    "break_even_lower": short_put_strike - net_credit,
                    "break_even_upper": short_call_strike + net_credit,
                    "days_to_expiry": (setup["expiration_date"] - datetime.now()).days,
                },
                "market_metrics": market_metrics,
                "expiration_date": setup["expiration_date"].isoformat(),
            }
        )

    async def execute_iron_condor(self, signal: StrategySignal) -> Dict[str, Any]:
        """Execute Iron Condor using multi-leg order"""
        try:
            legs_data = signal.market_context.get("legs", {})

            # Prepare multi-leg order
            legs = []

            # Short Put
            if "short_put" in legs_data:
                legs.append({
                    "symbol": legs_data["short_put"]["symbol"],
                    "side": "sell",
                    "quantity": 1,
                })

            # Long Put
            if "long_put" in legs_data:
                legs.append({
                    "symbol": legs_data["long_put"]["symbol"],
                    "side": "buy",
                    "quantity": 1,
                })

            # Short Call
            if "short_call" in legs_data:
                legs.append({
                    "symbol": legs_data["short_call"]["symbol"],
                    "side": "sell",
                    "quantity": 1,
                })

            # Long Call
            if "long_call" in legs_data:
                legs.append({
                    "symbol": legs_data["long_call"]["symbol"],
                    "side": "buy",
                    "quantity": 1,
                })

            # Execute multi-leg order
            order_result = await self.alpaca_client.execute_multi_leg_order(
                legs=legs,
                order_class="MLEG",
                time_in_force="gtc",
            )

            logger.info("Iron Condor executed",
                       symbol=signal.market_context.get("underlying_symbol"),
                       order_id=order_result.get("id"),
                       net_credit=signal.market_context.get("position_metrics", {}).get("net_credit"))

            return order_result

        except Exception as e:
            logger.error("Failed to execute Iron Condor", error=str(e))
            raise TradingError(f"Failed to execute Iron Condor: {e}")

    # Position management methods
    def add_position(self, position: IronCondorPosition):
        """Add new Iron Condor position to tracking"""
        self.active_positions[position.underlying_symbol] = position
        logger.info("Added Iron Condor position", symbol=position.underlying_symbol)

    def get_position(self, symbol: str) -> Optional[IronCondorPosition]:
        """Get Iron Condor position for symbol"""
        return self.active_positions.get(symbol)

    def remove_position(self, symbol: str):
        """Remove Iron Condor position"""
        if symbol in self.active_positions:
            del self.active_positions[symbol]
            logger.info("Removed Iron Condor position", symbol=symbol)

    def get_all_positions(self) -> Dict[str, IronCondorPosition]:
        """Get all active Iron Condor positions"""
        return self.active_positions.copy()

    async def monitor_positions(self) -> List[Dict[str, Any]]:
        """Monitor all active Iron Condor positions"""
        monitoring_results = []

        for symbol, position in self.active_positions.items():
            try:
                # Update position metrics
                current_value = await self._calculate_current_position_value(position)
                unrealized_pnl = current_value - position.net_credit if current_value else None

                position.current_value = current_value
                position.unrealized_pnl = unrealized_pnl
                position.days_to_expiry = (position.expiration_date - datetime.now()).days

                # Check if position should be managed
                should_close, reason = self._should_close_position(position)

                monitoring_results.append({
                    "symbol": symbol,
                    "current_value": current_value,
                    "unrealized_pnl": unrealized_pnl,
                    "days_to_expiry": position.days_to_expiry,
                    "should_close": should_close,
                    "close_reason": reason,
                    "profit_zone": f"${position.short_put_strike:.0f}-${position.short_call_strike:.0f}",
                })

            except Exception as e:
                logger.warning("Error monitoring position", symbol=symbol, error=str(e))
                monitoring_results.append({
                    "symbol": symbol,
                    "error": str(e)
                })

        return monitoring_results

    async def _calculate_current_position_value(self, position: IronCondorPosition) -> Optional[float]:
        """Calculate current value of Iron Condor position"""
        try:
            total_value = 0.0

            # Get current quotes for all legs
            leg_symbols = [
                position.short_put.option_contract.symbol,
                position.long_put.option_contract.symbol,
                position.short_call.option_contract.symbol,
                position.long_call.option_contract.symbol,
            ]

            for i, symbol in enumerate(leg_symbols):
                quote = await self.alpaca_client.get_option_quote(symbol)
                if quote:
                    mid_price = (quote["bid_price"] + quote["ask_price"]) / 2
                    # Short positions contribute negatively to current value
                    if i in [0, 2]:  # Short put and short call
                        total_value -= mid_price
                    else:  # Long positions
                        total_value += mid_price

            return total_value

        except Exception as e:
            logger.warning("Error calculating position value", error=str(e))
            return None

    def _should_close_position(self, position: IronCondorPosition) -> Tuple[bool, str]:
        """Check if Iron Condor position should be closed"""

        if not position.unrealized_pnl:
            return False, ""

        # Profit target reached
        if position.unrealized_pnl >= position.profit_target:
            return True, f"Profit target reached (${position.unrealized_pnl:.2f})"

        # Stop loss triggered
        if position.unrealized_pnl <= -position.stop_loss:
            return True, f"Stop loss triggered (${position.unrealized_pnl:.2f})"

        # Time-based management
        if position.days_to_expiry <= 7:
            return True, "Time-based close (7 days to expiry)"

        # Early profit taking
        profit_pct = (position.unrealized_pnl / position.max_profit) * 100 if position.max_profit > 0 else 0
        if profit_pct >= self.config.manage_at_profit_pct:
            return True, f"Early profit taking ({profit_pct:.1f}%)"

        return False, ""