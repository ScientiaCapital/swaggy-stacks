"""
Iron Butterfly Options Strategy

Income strategy that profits from low volatility and time decay.
Combines a short straddle (sell ATM call and put) with protective long wings.
Maximum profit occurs when underlying closes exactly at the short strike at expiration.
Limited risk with defined maximum loss.
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


class IronButterflyLeg(Enum):
    """Iron Butterfly legs"""
    LONG_PUT = "long_put"        # Buy OTM put (protection)
    SHORT_PUT = "short_put"      # Sell ATM put
    SHORT_CALL = "short_call"    # Sell ATM call
    LONG_CALL = "long_call"      # Buy OTM call (protection)


class ButterflyPhase(Enum):
    """Iron Butterfly phases"""
    ENTRY = "entry"                 # Looking for entry opportunity
    MONITORING = "monitoring"       # Monitoring existing position
    PROFIT_TAKING = "profit_taking" # Taking profits
    ADJUSTMENT = "adjustment"       # Adjusting position
    EXPIRATION = "expiration"       # Near expiration management


@dataclass
class IronButterflyConfig:
    """Configuration for Iron Butterfly strategy"""

    # Volatility criteria (prefer low IV environment)
    max_iv: float = 0.30  # 30% maximum implied volatility
    min_iv: float = 0.08  # 8% minimum implied volatility
    iv_rank_max: float = 0.70  # Maximum IV rank (70th percentile)
    iv_rank_min: float = 0.10  # Minimum IV rank (10th percentile)

    # Entry criteria
    min_days_to_expiry: int = 14    # Minimum 2 weeks
    max_days_to_expiry: int = 45    # Maximum 45 days (sweet spot 30-35)
    min_open_interest: int = 50     # Minimum open interest per leg
    max_bid_ask_spread_pct: float = 15.0  # Max 15% spread for butterflies

    # Strike selection
    wing_width: float = 10.0        # Distance from ATM to wings ($10 default)
    min_wing_width: float = 5.0     # Minimum wing width
    max_wing_width: float = 20.0    # Maximum wing width
    atm_threshold: float = 2.0      # Within $2 of ATM for center strike

    # Position management
    profit_target_pct: float = 50.0     # Target 50% of max profit
    stop_loss_pct: float = 200.0        # Stop loss at 200% of credit received
    max_loss_pct: float = 300.0         # Maximum loss threshold
    early_exit_days: int = 7            # Exit 7 days before expiration

    # Risk management
    max_position_size_pct: float = 3.0  # Max 3% of portfolio per butterfly
    min_credit_received: float = 1.0    # Minimum $1.00 credit to enter

    # Greeks thresholds
    max_theta_decay: float = -10.0      # Maximum theta decay per day
    target_delta: float = 0.0           # Target delta neutrality
    max_gamma_risk: float = 0.05        # Maximum gamma exposure


@dataclass
class IronButterflyPosition:
    """Represents an Iron Butterfly position"""
    symbol: str
    short_strike: float             # ATM strike (same for call and put)
    wing_width: float              # Distance to protective strikes
    long_put_contract: OptionContract
    short_put_contract: OptionContract
    short_call_contract: OptionContract
    long_call_contract: OptionContract
    entry_date: datetime
    credit_received: float         # Net credit received at entry
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    phase: ButterflyPhase = ButterflyPhase.ENTRY

    # Greeks for the combined position
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    # Risk metrics
    max_profit: float = 0.0        # Maximum profit potential
    max_loss: float = 0.0          # Maximum loss potential
    breakeven_upper: float = 0.0   # Upper breakeven point
    breakeven_lower: float = 0.0   # Lower breakeven point
    profit_zone_width: float = 0.0 # Width of profit zone
    days_held: int = 0
    iv_at_entry: float = 0.0


class IronButterflyStrategy:
    """
    Iron Butterfly options strategy implementation

    This strategy is designed for low volatility environments where
    the underlying is expected to stay near the current price.
    It's a credit strategy that benefits from time decay.
    """

    def __init__(self, alpaca_client: AlpacaClient, config: Optional[IronButterflyConfig] = None):
        """Initialize Iron Butterfly strategy"""
        self.alpaca_client = alpaca_client
        self.config = config or IronButterflyConfig()
        self.active_positions: Dict[str, IronButterflyPosition] = {}
        self.monitoring_task: Optional[asyncio.Task] = None

        logger.info("Iron Butterfly strategy initialized", config=self.config)

    async def analyze_symbol(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """
        Analyze symbol for Iron Butterfly opportunities

        Args:
            symbol: Stock symbol to analyze
            market_data: Current market data including price, volume, IV

        Returns:
            StrategySignal if opportunity found, None otherwise
        """
        try:
            current_price = market_data.get("price")
            if not current_price:
                return None

            # Check if we already have a position
            if symbol in self.active_positions:
                return await self._analyze_existing_position(symbol, market_data)
            else:
                return await self._analyze_new_position(symbol, market_data)

        except Exception as e:
            logger.error("Error analyzing iron butterfly opportunity", symbol=symbol, error=str(e))
            return None

    async def _analyze_new_position(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze for new Iron Butterfly entry opportunity"""
        current_price = market_data["price"]
        current_iv = market_data.get("implied_volatility", 0.0)
        iv_rank = market_data.get("iv_rank", 0.5)

        # Check volatility criteria (prefer low IV environment)
        if not (self.config.min_iv <= current_iv <= self.config.max_iv):
            logger.debug("IV outside acceptable range", symbol=symbol, iv=current_iv)
            return None

        if not (self.config.iv_rank_min <= iv_rank <= self.config.iv_rank_max):
            logger.debug("IV rank outside acceptable range", symbol=symbol, iv_rank=iv_rank)
            return None

        # Check for low volatility environment preference
        recent_volatility = market_data.get("realized_volatility_20d", current_iv)
        if recent_volatility > current_iv * 1.5:  # Prefer when IV > realized vol
            logger.debug("Realized volatility too high vs IV", symbol=symbol)
            return None

        # Get option chain
        option_chain = await self._get_filtered_option_chain(symbol, current_price)
        if not option_chain:
            return None

        # Find best butterfly opportunity
        best_butterfly = await self._find_best_butterfly_opportunity(symbol, current_price, option_chain)
        if not best_butterfly:
            return None

        return await self._create_butterfly_signal(symbol, best_butterfly, market_data)

    async def _analyze_existing_position(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze existing Iron Butterfly position for management actions"""
        position = self.active_positions[symbol]
        current_price = market_data["price"]

        # Update position metrics
        await self._update_position_metrics(position, market_data)

        # Check for profit taking
        if position.unrealized_pnl >= position.max_profit * (self.config.profit_target_pct / 100):
            return await self._create_exit_signal(symbol, position, "PROFIT_TARGET")

        # Check for stop loss
        if position.unrealized_pnl <= -position.credit_received * (self.config.stop_loss_pct / 100):
            return await self._create_exit_signal(symbol, position, "STOP_LOSS")

        # Check early exit before expiration
        days_to_expiry = (position.short_call_contract.expiration - datetime.now()).days
        if days_to_expiry <= self.config.early_exit_days:
            return await self._create_exit_signal(symbol, position, "TIME_DECAY")

        # Check for adjustment opportunities
        distance_from_short = abs(current_price - position.short_strike)
        wing_distance = position.wing_width * 0.7  # 70% of wing width

        if distance_from_short > wing_distance:
            return await self._create_adjustment_signal(symbol, position, market_data)

        return None

    async def _get_filtered_option_chain(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """Get filtered option chain for butterfly analysis"""
        try:
            # Calculate date range
            min_date = datetime.now() + timedelta(days=self.config.min_days_to_expiry)
            max_date = datetime.now() + timedelta(days=self.config.max_days_to_expiry)

            # Get option chain from Alpaca
            option_chain = await self.alpaca_client.get_option_chain(
                symbol=symbol,
                expiration_date_gte=min_date.strftime("%Y-%m-%d"),
                expiration_date_lte=max_date.strftime("%Y-%m-%d")
            )

            if not option_chain:
                return []

            # Filter for relevant strikes and liquidity
            filtered_chain = []
            for option in option_chain:
                strike = option.get("strike_price", 0)
                open_interest = option.get("open_interest", 0)
                bid_ask_spread = option.get("ask", 0) - option.get("bid", 0)
                bid_ask_spread_pct = (bid_ask_spread / option.get("ask", 1)) * 100 if option.get("ask", 0) > 0 else 100

                # Check if strike is within reasonable range for butterfly
                strike_distance = abs(strike - current_price)
                if strike_distance <= self.config.max_wing_width:
                    # Check liquidity criteria
                    if (open_interest >= self.config.min_open_interest and
                        bid_ask_spread_pct <= self.config.max_bid_ask_spread_pct):
                        filtered_chain.append(option)

            return filtered_chain

        except Exception as e:
            logger.error("Error filtering option chain", symbol=symbol, error=str(e))
            return []

    async def _find_best_butterfly_opportunity(self, symbol: str, current_price: float,
                                             option_chain: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the best Iron Butterfly opportunity from option chain"""
        best_butterfly = None
        best_score = 0

        # Group options by expiration and strike
        expirations = {}
        for option in option_chain:
            exp_date = option.get("expiration")
            strike = option.get("strike_price")
            option_type = option.get("type")

            if exp_date not in expirations:
                expirations[exp_date] = {}
            if strike not in expirations[exp_date]:
                expirations[exp_date][strike] = {}

            expirations[exp_date][strike][option_type] = option

        # Evaluate each potential butterfly
        for exp_date, strikes in expirations.items():
            strike_list = sorted(strikes.keys())

            for i, center_strike in enumerate(strike_list):
                if abs(center_strike - current_price) > self.config.atm_threshold:
                    continue

                # Look for wing strikes
                for wing_width in [5.0, 10.0, 15.0, 20.0]:
                    if wing_width < self.config.min_wing_width or wing_width > self.config.max_wing_width:
                        continue

                    lower_strike = center_strike - wing_width
                    upper_strike = center_strike + wing_width

                    # Check if all required strikes exist
                    if (lower_strike in strikes and center_strike in strikes and upper_strike in strikes):
                        butterfly_data = await self._build_butterfly_position(
                            strikes[lower_strike], strikes[center_strike], strikes[upper_strike],
                            current_price, wing_width
                        )

                        if butterfly_data and self._passes_butterfly_filters(butterfly_data):
                            score = await self._score_butterfly_opportunity(butterfly_data)
                            if score > best_score:
                                best_score = score
                                best_butterfly = butterfly_data

        return best_butterfly

    async def _build_butterfly_position(self, lower_strikes: Dict, center_strikes: Dict,
                                      upper_strikes: Dict, current_price: float,
                                      wing_width: float) -> Optional[Dict[str, Any]]:
        """Build Iron Butterfly position from strike data"""
        try:
            # Verify all legs exist
            required_legs = ["call", "put"]
            for strikes in [lower_strikes, center_strikes, upper_strikes]:
                for leg in required_legs:
                    if leg not in strikes:
                        return None

            # Extract leg data
            long_put = lower_strikes["put"]      # Buy OTM put
            short_put = center_strikes["put"]    # Sell ATM put
            short_call = center_strikes["call"]  # Sell ATM call
            long_call = upper_strikes["call"]    # Buy OTM call

            # Calculate net credit received
            long_put_price = (long_put.get("bid", 0) + long_put.get("ask", 0)) / 2
            short_put_price = (short_put.get("bid", 0) + short_put.get("ask", 0)) / 2
            short_call_price = (short_call.get("bid", 0) + short_call.get("ask", 0)) / 2
            long_call_price = (long_call.get("bid", 0) + long_call.get("ask", 0)) / 2

            net_credit = (short_put_price + short_call_price) - (long_put_price + long_call_price)

            if net_credit < self.config.min_credit_received:
                return None

            # Calculate profit/loss metrics
            center_strike = short_call.get("strike_price")
            max_profit = net_credit
            max_loss = wing_width - net_credit
            breakeven_lower = center_strike - net_credit
            breakeven_upper = center_strike + net_credit
            profit_zone_width = breakeven_upper - breakeven_lower

            # Calculate combined Greeks
            greeks = await self._calculate_butterfly_greeks(
                long_put, short_put, short_call, long_call, current_price
            )

            return {
                "long_put": long_put,
                "short_put": short_put,
                "short_call": short_call,
                "long_call": long_call,
                "center_strike": center_strike,
                "wing_width": wing_width,
                "net_credit": net_credit,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "breakeven_lower": breakeven_lower,
                "breakeven_upper": breakeven_upper,
                "profit_zone_width": profit_zone_width,
                "greeks": greeks,
                "expiration": short_call.get("expiration"),
                "days_to_expiry": (datetime.strptime(short_call.get("expiration"), "%Y-%m-%d") - datetime.now()).days
            }

        except Exception as e:
            logger.error("Error building butterfly position", error=str(e))
            return None

    async def _calculate_butterfly_greeks(self, long_put: Dict, short_put: Dict,
                                        short_call: Dict, long_call: Dict,
                                        current_price: float) -> Dict[str, float]:
        """Calculate combined Greeks for Iron Butterfly position"""
        # Calculate individual Greeks for each leg
        long_put_greeks = await self._calculate_greeks(long_put, current_price)
        short_put_greeks = await self._calculate_greeks(short_put, current_price)
        short_call_greeks = await self._calculate_greeks(short_call, current_price)
        long_call_greeks = await self._calculate_greeks(long_call, current_price)

        # Combine Greeks (account for position direction: long = +1, short = -1)
        combined_delta = (long_put_greeks["delta"] +
                         -short_put_greeks["delta"] +
                         -short_call_greeks["delta"] +
                         long_call_greeks["delta"])

        combined_gamma = (long_put_greeks["gamma"] +
                         -short_put_greeks["gamma"] +
                         -short_call_greeks["gamma"] +
                         long_call_greeks["gamma"])

        combined_theta = (long_put_greeks["theta"] +
                         -short_put_greeks["theta"] +
                         -short_call_greeks["theta"] +
                         long_call_greeks["theta"])

        combined_vega = (long_put_greeks["vega"] +
                        -short_put_greeks["vega"] +
                        -short_call_greeks["vega"] +
                        long_call_greeks["vega"])

        combined_rho = (long_put_greeks["rho"] +
                       -short_put_greeks["rho"] +
                       -short_call_greeks["rho"] +
                       long_call_greeks["rho"])

        return {
            "delta": combined_delta,
            "gamma": combined_gamma,
            "theta": combined_theta,
            "vega": combined_vega,
            "rho": combined_rho
        }

    def _passes_butterfly_filters(self, butterfly_data: Dict[str, Any]) -> bool:
        """Check if butterfly passes all filters"""
        # Check minimum credit received
        if butterfly_data["net_credit"] < self.config.min_credit_received:
            return False

        # Check risk/reward ratio
        risk_reward_ratio = butterfly_data["max_loss"] / butterfly_data["max_profit"]
        if risk_reward_ratio > 10.0:  # Don't risk more than 10:1
            return False

        # Check theta decay (should be positive for credit strategies)
        if butterfly_data["greeks"]["theta"] < -self.config.max_theta_decay:
            return False

        # Check gamma risk
        if abs(butterfly_data["greeks"]["gamma"]) > self.config.max_gamma_risk:
            return False

        return True

    async def _score_butterfly_opportunity(self, butterfly_data: Dict[str, Any]) -> float:
        """Score Iron Butterfly opportunity based on multiple factors"""
        score = 0.0

        # Reward good credit received relative to wing width
        credit_ratio = butterfly_data["net_credit"] / butterfly_data["wing_width"]
        score += credit_ratio * 40.0  # Up to 40 points for good credit

        # Reward good risk/reward ratio
        risk_reward = butterfly_data["max_loss"] / butterfly_data["max_profit"]
        if risk_reward <= 3.0:
            score += 25.0 * (1 - risk_reward / 3.0)

        # Reward positive theta decay
        theta_score = min(-butterfly_data["greeks"]["theta"] * 2, 20.0)
        score += theta_score

        # Reward delta neutrality
        delta_neutrality = 10.0 * (1 - abs(butterfly_data["greeks"]["delta"]) * 20)
        score += max(delta_neutrality, 0)

        # Reward optimal time to expiry (sweet spot around 30-35 days)
        days_to_expiry = butterfly_data["days_to_expiry"]
        if 25 <= days_to_expiry <= 40:
            score += 15.0 * (1 - abs(days_to_expiry - 32.5) / 15)

        return max(score, 0.0)

    async def _create_butterfly_signal(self, symbol: str, butterfly_data: Dict[str, Any],
                                     market_data: Dict[str, Any]) -> StrategySignal:
        """Create strategy signal for Iron Butterfly entry"""
        # Convert to option contracts
        long_put_contract = await self._convert_to_option_contract(butterfly_data["long_put"])
        short_put_contract = await self._convert_to_option_contract(butterfly_data["short_put"])
        short_call_contract = await self._convert_to_option_contract(butterfly_data["short_call"])
        long_call_contract = await self._convert_to_option_contract(butterfly_data["long_call"])

        confidence = min(await self._score_butterfly_opportunity(butterfly_data) / 100.0, 0.95)

        signal = StrategySignal(
            strategy_name="iron_butterfly",
            symbol=symbol,
            action="OPEN_IRON_BUTTERFLY",
            confidence=confidence,
            reasoning=f"Iron Butterfly entry: IV={market_data.get('implied_volatility', 0):.1%}, "
                     f"credit=${butterfly_data['net_credit']:.2f}, "
                     f"max profit=${butterfly_data['max_profit']:.2f}, "
                     f"profit zone={butterfly_data['profit_zone_width']:.1f}",
            metadata={
                "strategy_type": "iron_butterfly",
                "long_put_contract": long_put_contract.__dict__,
                "short_put_contract": short_put_contract.__dict__,
                "short_call_contract": short_call_contract.__dict__,
                "long_call_contract": long_call_contract.__dict__,
                "center_strike": butterfly_data["center_strike"],
                "wing_width": butterfly_data["wing_width"],
                "net_credit": butterfly_data["net_credit"],
                "max_profit": butterfly_data["max_profit"],
                "max_loss": butterfly_data["max_loss"],
                "breakeven_lower": butterfly_data["breakeven_lower"],
                "breakeven_upper": butterfly_data["breakeven_upper"],
                "profit_zone_width": butterfly_data["profit_zone_width"],
                "greeks": butterfly_data["greeks"],
                "days_to_expiry": butterfly_data["days_to_expiry"],
                "iv_at_entry": market_data.get("implied_volatility", 0)
            }
        )

        return signal

    async def _create_exit_signal(self, symbol: str, position: IronButterflyPosition,
                                reason: str) -> StrategySignal:
        """Create exit signal for Iron Butterfly position"""
        return StrategySignal(
            strategy_name="iron_butterfly",
            symbol=symbol,
            action="CLOSE_IRON_BUTTERFLY",
            confidence=0.90,
            reasoning=f"Exit Iron Butterfly: {reason}, P&L: {position.unrealized_pnl:.2f}",
            metadata={
                "strategy_type": "iron_butterfly",
                "exit_reason": reason,
                "credit_received": position.credit_received,
                "current_value": position.current_value,
                "unrealized_pnl": position.unrealized_pnl,
                "days_held": position.days_held,
                "max_profit": position.max_profit,
                "max_loss": position.max_loss
            }
        )

    async def _create_adjustment_signal(self, symbol: str, position: IronButterflyPosition,
                                      market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Create adjustment signal for Iron Butterfly position"""
        current_price = market_data["price"]

        # Determine adjustment type based on position
        if current_price > position.short_strike:
            # Price moved above short strike - consider rolling up
            adjustment_type = "ROLL_UP"
            reasoning = f"Price moved to {current_price:.2f}, above short strike {position.short_strike:.2f}"
        else:
            # Price moved below short strike - consider rolling down
            adjustment_type = "ROLL_DOWN"
            reasoning = f"Price moved to {current_price:.2f}, below short strike {position.short_strike:.2f}"

        return StrategySignal(
            strategy_name="iron_butterfly",
            symbol=symbol,
            action=f"ADJUST_BUTTERFLY_{adjustment_type}",
            confidence=0.70,
            reasoning=f"Adjust Iron Butterfly: {reasoning}",
            metadata={
                "strategy_type": "iron_butterfly_adjustment",
                "adjustment_type": adjustment_type,
                "current_price": current_price,
                "short_strike": position.short_strike,
                "wing_width": position.wing_width
            }
        )

    async def _convert_to_option_contract(self, option_data: Dict[str, Any]) -> OptionContract:
        """Convert option data to OptionContract object"""
        return OptionContract(
            symbol=option_data.get("underlying_symbol", ""),
            strike_price=option_data.get("strike_price", 0.0),
            expiration=datetime.strptime(option_data.get("expiration", ""), "%Y-%m-%d"),
            option_type=OptionType.CALL if option_data.get("type") == "call" else OptionType.PUT,
            bid=option_data.get("bid", 0.0),
            ask=option_data.get("ask", 0.0),
            last=option_data.get("last", 0.0),
            volume=option_data.get("volume", 0),
            open_interest=option_data.get("open_interest", 0),
            implied_volatility=option_data.get("implied_volatility", 0.0)
        )

    async def _calculate_greeks(self, option_data: Dict[str, Any],
                              underlying_price: float) -> Dict[str, float]:
        """Calculate Greeks for an option"""
        # This would typically use the enhanced Black-Scholes calculator
        # For now, return mock Greeks based on option data
        option_type = option_data.get("type", "call")
        strike = option_data.get("strike_price", underlying_price)

        # Simple delta approximation
        if option_type == "call":
            delta = 0.5 if strike == underlying_price else (0.8 if strike < underlying_price else 0.2)
        else:
            delta = -0.5 if strike == underlying_price else (-0.2 if strike < underlying_price else -0.8)

        return {
            "delta": delta,
            "gamma": 0.02 if abs(strike - underlying_price) < 5 else 0.01,
            "theta": -0.05,
            "vega": 0.1,
            "rho": 0.02
        }

    async def _update_position_metrics(self, position: IronButterflyPosition,
                                     market_data: Dict[str, Any]) -> None:
        """Update position metrics with current market data"""
        current_price = market_data["price"]

        # Estimate current value of the butterfly position
        # This is a simplified calculation - in production would use real option prices
        days_to_expiry = (position.short_call_contract.expiration - datetime.now()).days
        time_value_factor = max(days_to_expiry / 30.0, 0.05)

        # Calculate intrinsic values
        center_strike = position.short_strike
        long_put_intrinsic = max(position.long_put_contract.strike_price - current_price, 0)
        short_put_intrinsic = max(center_strike - current_price, 0)
        short_call_intrinsic = max(current_price - center_strike, 0)
        long_call_intrinsic = max(current_price - position.long_call_contract.strike_price, 0)

        # Add time value estimates
        iv_factor = position.iv_at_entry * time_value_factor

        estimated_long_put_price = long_put_intrinsic + iv_factor
        estimated_short_put_price = short_put_intrinsic + iv_factor * 1.2  # ATM has higher time value
        estimated_short_call_price = short_call_intrinsic + iv_factor * 1.2
        estimated_long_call_price = long_call_intrinsic + iv_factor

        # Calculate position value (what we'd pay to close)
        position_cost = (estimated_long_put_price + estimated_long_call_price +
                        estimated_short_put_price + estimated_short_call_price)

        position.current_value = position_cost
        position.unrealized_pnl = position.credit_received - position_cost
        position.days_held = (datetime.now() - position.entry_date).days

        # Update Greeks (simplified)
        distance_from_center = abs(current_price - center_strike)
        position.delta = 0.0  # Iron Butterfly should be delta neutral
        position.gamma = -0.03 if distance_from_center < 2 else -0.01  # Negative gamma
        position.theta = 0.08 * (position.credit_received / 100)  # Positive theta for credit strategy
        position.vega = -0.12 * (position.credit_received / 100)  # Negative vega

    async def add_position(self, symbol: str, center_strike: float, wing_width: float,
                          long_put_contract: OptionContract, short_put_contract: OptionContract,
                          short_call_contract: OptionContract, long_call_contract: OptionContract,
                          credit_received: float, iv_at_entry: float) -> None:
        """Add new Iron Butterfly position to tracking"""
        max_profit = credit_received
        max_loss = wing_width - credit_received
        breakeven_lower = center_strike - credit_received
        breakeven_upper = center_strike + credit_received
        profit_zone_width = breakeven_upper - breakeven_lower

        position = IronButterflyPosition(
            symbol=symbol,
            short_strike=center_strike,
            wing_width=wing_width,
            long_put_contract=long_put_contract,
            short_put_contract=short_put_contract,
            short_call_contract=short_call_contract,
            long_call_contract=long_call_contract,
            entry_date=datetime.now(),
            credit_received=credit_received,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_lower=breakeven_lower,
            breakeven_upper=breakeven_upper,
            profit_zone_width=profit_zone_width,
            iv_at_entry=iv_at_entry
        )

        self.active_positions[symbol] = position
        logger.info("Added Iron Butterfly position", symbol=symbol,
                   credit_received=credit_received, center_strike=center_strike)

    def get_position(self, symbol: str) -> Optional[IronButterflyPosition]:
        """Get Iron Butterfly position for symbol"""
        return self.active_positions.get(symbol)

    async def remove_position(self, symbol: str) -> None:
        """Remove Iron Butterfly position from tracking"""
        if symbol in self.active_positions:
            del self.active_positions[symbol]
            logger.info("Removed Iron Butterfly position", symbol=symbol)

    def get_all_positions(self) -> Dict[str, IronButterflyPosition]:
        """Get all active Iron Butterfly positions"""
        return self.active_positions.copy()