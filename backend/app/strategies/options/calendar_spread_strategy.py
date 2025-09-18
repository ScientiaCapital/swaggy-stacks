"""
Calendar Spread Options Strategy

Time decay arbitrage strategy that profits from differential theta decay between
near-term and far-term options at the same strike. Also known as horizontal spread.
Buys far-dated option and sells near-dated option to benefit from accelerated
time decay of the short option.
"""

import asyncio
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


class CalendarSpreadType(Enum):
    """Calendar spread types"""
    CALL_CALENDAR = "call_calendar"    # Using calls
    PUT_CALENDAR = "put_calendar"      # Using puts
    DOUBLE_CALENDAR = "double_calendar" # Both calls and puts


class CalendarPhase(Enum):
    """Calendar spread phases"""
    ENTRY = "entry"                    # Looking for entry opportunity
    MONITORING = "monitoring"          # Monitoring existing position
    FRONT_MONTH_MGMT = "front_month"   # Managing front month expiration
    ROLL_FORWARD = "roll_forward"      # Rolling front month forward
    VOLATILITY_EXPANSION = "vol_expansion"  # Volatility increased


@dataclass
class CalendarSpreadConfig:
    """Configuration for Calendar Spread strategy"""

    # Time to expiration parameters
    short_min_days: int = 7            # Minimum days for short option
    short_max_days: int = 30           # Maximum days for short option
    long_min_days: int = 35            # Minimum days for long option
    long_max_days: int = 90            # Maximum days for long option
    min_time_spread: int = 21          # Minimum days between expirations

    # Volatility criteria
    min_iv: float = 0.12               # 12% minimum implied volatility
    max_iv: float = 0.45               # 45% maximum implied volatility
    iv_skew_threshold: float = 0.05    # Max IV difference between months
    iv_rank_min: float = 0.20          # Minimum IV rank
    iv_rank_max: float = 0.80          # Maximum IV rank

    # Strike selection
    atm_threshold: float = 2.0         # Within $2 of ATM for calendar strikes
    max_strikes_to_check: int = 5      # Maximum strikes to evaluate

    # Entry criteria
    min_open_interest: int = 50        # Minimum open interest per leg
    max_bid_ask_spread_pct: float = 12.0  # Max 12% spread for calendars
    min_credit_ratio: float = 0.15     # Minimum credit as % of long option price

    # Position management
    profit_target_pct: float = 50.0    # Target 50% of max profit
    stop_loss_pct: float = 100.0       # Stop loss at 100% of debit paid
    max_loss_pct: float = 150.0        # Maximum loss threshold
    vol_expansion_exit_pct: float = 25.0  # Exit if IV increases by 25%

    # Greeks thresholds
    target_theta_ratio: float = 2.0    # Target short theta / long theta ratio
    max_delta_exposure: float = 0.15   # Maximum delta exposure
    min_vega_ratio: float = 0.5        # Minimum long vega / short vega ratio

    # Risk management
    max_position_size_pct: float = 4.0  # Max 4% of portfolio per calendar
    max_positions_per_symbol: int = 2   # Max calendars per underlying


@dataclass
class CalendarSpreadPosition:
    """Represents a Calendar Spread position"""
    symbol: str
    spread_type: CalendarSpreadType
    strike_price: float
    short_option_contract: OptionContract  # Near-term (sell)
    long_option_contract: OptionContract   # Far-term (buy)
    entry_date: datetime
    net_debit: float                       # Net debit paid at entry
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    phase: CalendarPhase = CalendarPhase.ENTRY

    # Greeks for the combined position
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0                     # Should be positive for calendars
    vega: float = 0.0
    rho: float = 0.0

    # Time decay metrics
    short_theta: float = 0.0
    long_theta: float = 0.0
    theta_ratio: float = 0.0               # short_theta / long_theta

    # Volatility metrics
    short_vega: float = 0.0
    long_vega: float = 0.0
    vega_ratio: float = 0.0                # long_vega / short_vega

    # Risk metrics
    max_profit: float = 0.0                # Estimated maximum profit
    max_loss: float = 0.0                  # Maximum loss (debit paid)
    optimal_expiry_price: float = 0.0      # Price for maximum profit at front expiry
    days_held: int = 0
    iv_at_entry: float = 0.0
    short_days_to_expiry: int = 0
    long_days_to_expiry: int = 0


class CalendarSpreadStrategy:
    """
    Calendar Spread options strategy implementation

    This strategy profits from time decay differential between near and far
    options. It's typically delta-neutral and benefits from low volatility
    and stable underlying prices.
    """

    def __init__(self, alpaca_client: AlpacaClient, config: Optional[CalendarSpreadConfig] = None):
        """Initialize Calendar Spread strategy"""
        self.alpaca_client = alpaca_client
        self.config = config or CalendarSpreadConfig()
        self.active_positions: Dict[str, List[CalendarSpreadPosition]] = {}
        self.monitoring_task: Optional[asyncio.Task] = None

        logger.info("Calendar Spread strategy initialized", config=self.config)

    async def analyze_symbol(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """
        Analyze symbol for Calendar Spread opportunities

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

            # Check if we already have positions
            existing_positions = self.active_positions.get(symbol, [])

            # Analyze existing positions first
            for position in existing_positions:
                signal = await self._analyze_existing_position(position, market_data)
                if signal:
                    return signal

            # Check for new position opportunity
            if len(existing_positions) < self.config.max_positions_per_symbol:
                return await self._analyze_new_position(symbol, market_data)

            return None

        except Exception as e:
            logger.error("Error analyzing calendar spread opportunity", symbol=symbol, error=str(e))
            return None

    async def _analyze_new_position(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze for new Calendar Spread entry opportunity"""
        current_price = market_data["price"]
        current_iv = market_data.get("implied_volatility", 0.0)
        iv_rank = market_data.get("iv_rank", 0.5)

        # Check volatility criteria
        if not (self.config.min_iv <= current_iv <= self.config.max_iv):
            logger.debug("IV outside acceptable range", symbol=symbol, iv=current_iv)
            return None

        if not (self.config.iv_rank_min <= iv_rank <= self.config.iv_rank_max):
            logger.debug("IV rank outside acceptable range", symbol=symbol, iv_rank=iv_rank)
            return None

        # Get option chain with multiple expirations
        option_chain = await self._get_calendar_option_chain(symbol, current_price)
        if not option_chain:
            return None

        # Find best calendar spread opportunity
        best_calendar = await self._find_best_calendar_opportunity(symbol, current_price, option_chain)
        if not best_calendar:
            return None

        return await self._create_calendar_signal(symbol, best_calendar, market_data)

    async def _analyze_existing_position(self, position: CalendarSpreadPosition,
                                       market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze existing Calendar Spread position for management actions"""
        current_price = market_data["price"]

        # Update position metrics
        await self._update_position_metrics(position, market_data)

        # Check for profit taking
        profit_pct = (position.unrealized_pnl / abs(position.net_debit)) * 100
        if profit_pct >= self.config.profit_target_pct:
            return await self._create_exit_signal(position, "PROFIT_TARGET")

        # Check for stop loss
        if profit_pct <= -self.config.stop_loss_pct:
            return await self._create_exit_signal(position, "STOP_LOSS")

        # Check for volatility expansion
        current_iv = market_data.get("implied_volatility", 0.0)
        iv_change_pct = ((current_iv - position.iv_at_entry) / position.iv_at_entry) * 100
        if iv_change_pct >= self.config.vol_expansion_exit_pct:
            return await self._create_exit_signal(position, "VOLATILITY_EXPANSION")

        # Check front month expiration management
        if position.short_days_to_expiry <= 7:
            return await self._create_front_month_mgmt_signal(position, market_data)

        # Check for rolling opportunity
        if position.short_days_to_expiry <= 14 and abs(current_price - position.strike_price) < 3.0:
            return await self._create_roll_signal(position, market_data)

        return None

    async def _get_calendar_option_chain(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Get option chain data structured for calendar spread analysis"""
        try:
            # Calculate date ranges for short and long options
            short_min_date = datetime.now() + timedelta(days=self.config.short_min_days)
            short_max_date = datetime.now() + timedelta(days=self.config.short_max_days)
            long_min_date = datetime.now() + timedelta(days=self.config.long_min_days)
            long_max_date = datetime.now() + timedelta(days=self.config.long_max_days)

            # Get full option chain
            option_chain = await self.alpaca_client.get_option_chain(
                symbol=symbol,
                expiration_date_gte=short_min_date.strftime("%Y-%m-%d"),
                expiration_date_lte=long_max_date.strftime("%Y-%m-%d")
            )

            if not option_chain:
                return {}

            # Organize by expiration and strike
            organized_chain = {}
            for option in option_chain:
                exp_date = option.get("expiration")
                strike = option.get("strike_price")
                option_type = option.get("type")

                # Filter for relevant strikes (near ATM)
                if abs(strike - current_price) > self.config.atm_threshold:
                    continue

                # Check liquidity
                open_interest = option.get("open_interest", 0)
                if open_interest < self.config.min_open_interest:
                    continue

                # Organize data
                if exp_date not in organized_chain:
                    organized_chain[exp_date] = {}
                if strike not in organized_chain[exp_date]:
                    organized_chain[exp_date][strike] = {}

                organized_chain[exp_date][strike][option_type] = option

            return organized_chain

        except Exception as e:
            logger.error("Error getting calendar option chain", symbol=symbol, error=str(e))
            return {}

    async def _find_best_calendar_opportunity(self, symbol: str, current_price: float,
                                            option_chain: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the best calendar spread opportunity from option chain"""
        best_calendar = None
        best_score = 0

        # Get available expirations
        expirations = sorted(option_chain.keys())

        # Evaluate calendar spreads between different expirations
        for i, short_exp in enumerate(expirations):
            short_exp_date = datetime.strptime(short_exp, "%Y-%m-%d")
            short_days = (short_exp_date - datetime.now()).days

            if not (self.config.short_min_days <= short_days <= self.config.short_max_days):
                continue

            for j, long_exp in enumerate(expirations[i+1:], i+1):
                long_exp_date = datetime.strptime(long_exp, "%Y-%m-%d")
                long_days = (long_exp_date - datetime.now()).days

                if not (self.config.long_min_days <= long_days <= self.config.long_max_days):
                    continue

                # Check minimum time spread
                if long_days - short_days < self.config.min_time_spread:
                    continue

                # Find common strikes
                short_strikes = set(option_chain[short_exp].keys())
                long_strikes = set(option_chain[long_exp].keys())
                common_strikes = short_strikes.intersection(long_strikes)

                for strike in common_strikes:
                    # Evaluate call calendar
                    call_calendar = await self._evaluate_calendar_spread(
                        option_chain[short_exp][strike], option_chain[long_exp][strike],
                        "call", current_price, short_days, long_days
                    )

                    if call_calendar and self._passes_calendar_filters(call_calendar):
                        score = await self._score_calendar_opportunity(call_calendar)
                        if score > best_score:
                            best_score = score
                            best_calendar = call_calendar

                    # Evaluate put calendar
                    put_calendar = await self._evaluate_calendar_spread(
                        option_chain[short_exp][strike], option_chain[long_exp][strike],
                        "put", current_price, short_days, long_days
                    )

                    if put_calendar and self._passes_calendar_filters(put_calendar):
                        score = await self._score_calendar_opportunity(put_calendar)
                        if score > best_score:
                            best_score = score
                            best_calendar = put_calendar

        return best_calendar

    async def _evaluate_calendar_spread(self, short_options: Dict, long_options: Dict,
                                      option_type: str, current_price: float,
                                      short_days: int, long_days: int) -> Optional[Dict[str, Any]]:
        """Evaluate a specific calendar spread combination"""
        try:
            if option_type not in short_options or option_type not in long_options:
                return None

            short_option = short_options[option_type]
            long_option = long_options[option_type]

            # Calculate prices
            short_price = (short_option.get("bid", 0) + short_option.get("ask", 0)) / 2
            long_price = (long_option.get("bid", 0) + long_option.get("ask", 0)) / 2

            if short_price <= 0 or long_price <= 0:
                return None

            net_debit = long_price - short_price
            if net_debit <= 0:  # Must be a debit spread
                return None

            # Check minimum credit ratio
            credit_ratio = short_price / long_price
            if credit_ratio < self.config.min_credit_ratio:
                return None

            # Calculate Greeks
            short_greeks = await self._calculate_greeks(short_option, current_price)
            long_greeks = await self._calculate_greeks(long_option, current_price)

            # Combined Greeks
            combined_delta = long_greeks["delta"] - short_greeks["delta"]
            combined_gamma = long_greeks["gamma"] - short_greeks["gamma"]
            combined_theta = long_greeks["theta"] - short_greeks["theta"]
            combined_vega = long_greeks["vega"] - short_greeks["vega"]
            combined_rho = long_greeks["rho"] - short_greeks["rho"]

            # Calculate ratios
            theta_ratio = abs(short_greeks["theta"] / long_greeks["theta"]) if long_greeks["theta"] != 0 else 0
            vega_ratio = long_greeks["vega"] / short_greeks["vega"] if short_greeks["vega"] != 0 else 0

            # Estimate maximum profit (simplified)
            strike = short_option.get("strike_price")
            max_profit = short_price * 0.8  # Approximate max profit at expiration
            optimal_price = strike  # Max profit when underlying = strike at short expiration

            return {
                "spread_type": CalendarSpreadType.CALL_CALENDAR if option_type == "call" else CalendarSpreadType.PUT_CALENDAR,
                "short_option": short_option,
                "long_option": long_option,
                "short_price": short_price,
                "long_price": long_price,
                "net_debit": net_debit,
                "strike_price": strike,
                "short_days": short_days,
                "long_days": long_days,
                "time_spread": long_days - short_days,
                "max_profit": max_profit,
                "max_loss": net_debit,
                "optimal_expiry_price": optimal_price,
                "credit_ratio": credit_ratio,
                "delta": combined_delta,
                "gamma": combined_gamma,
                "theta": combined_theta,
                "vega": combined_vega,
                "rho": combined_rho,
                "short_theta": short_greeks["theta"],
                "long_theta": long_greeks["theta"],
                "theta_ratio": theta_ratio,
                "short_vega": short_greeks["vega"],
                "long_vega": long_greeks["vega"],
                "vega_ratio": vega_ratio
            }

        except Exception as e:
            logger.error("Error evaluating calendar spread", error=str(e))
            return None

    def _passes_calendar_filters(self, calendar_data: Dict[str, Any]) -> bool:
        """Check if calendar spread passes all filters"""
        # Check theta ratio (short should decay faster)
        if calendar_data["theta_ratio"] < self.config.target_theta_ratio:
            return False

        # Check delta exposure (should be low)
        if abs(calendar_data["delta"]) > self.config.max_delta_exposure:
            return False

        # Check vega ratio (long should have more vega)
        if calendar_data["vega_ratio"] < self.config.min_vega_ratio:
            return False

        # Check that theta is positive (time decay working for us)
        if calendar_data["theta"] <= 0:
            return False

        return True

    async def _score_calendar_opportunity(self, calendar_data: Dict[str, Any]) -> float:
        """Score calendar spread opportunity based on multiple factors"""
        score = 0.0

        # Reward good credit ratio (short premium vs long premium)
        credit_ratio = calendar_data["credit_ratio"]
        score += credit_ratio * 30.0  # Up to 30 points

        # Reward good theta ratio (short decays faster than long)
        theta_ratio = min(calendar_data["theta_ratio"], 5.0)
        score += theta_ratio * 8.0  # Up to 40 points

        # Reward positive net theta
        net_theta_score = min(calendar_data["theta"] * 100, 25.0)
        score += net_theta_score

        # Reward delta neutrality
        delta_neutrality = 10.0 * (1 - abs(calendar_data["delta"]) * 10)
        score += max(delta_neutrality, 0)

        # Reward good time spread (sweet spot around 30-40 days)
        time_spread = calendar_data["time_spread"]
        if 25 <= time_spread <= 45:
            score += 15.0 * (1 - abs(time_spread - 35) / 20)

        # Reward good vega ratio
        vega_ratio = min(calendar_data["vega_ratio"], 3.0)
        score += vega_ratio * 5.0  # Up to 15 points

        return max(score, 0.0)

    async def _create_calendar_signal(self, symbol: str, calendar_data: Dict[str, Any],
                                    market_data: Dict[str, Any]) -> StrategySignal:
        """Create strategy signal for Calendar Spread entry"""
        short_contract = await self._convert_to_option_contract(calendar_data["short_option"])
        long_contract = await self._convert_to_option_contract(calendar_data["long_option"])

        confidence = min(await self._score_calendar_opportunity(calendar_data) / 100.0, 0.95)

        signal = StrategySignal(
            strategy_name="calendar_spread",
            symbol=symbol,
            action="OPEN_CALENDAR_SPREAD",
            confidence=confidence,
            reasoning=f"Calendar spread entry: {calendar_data['spread_type'].value}, "
                     f"strike=${calendar_data['strike_price']:.2f}, "
                     f"debit=${calendar_data['net_debit']:.2f}, "
                     f"theta={calendar_data['theta']:.3f}, "
                     f"time spread={calendar_data['time_spread']} days",
            metadata={
                "strategy_type": "calendar_spread",
                "spread_type": calendar_data["spread_type"].value,
                "short_contract": short_contract.__dict__,
                "long_contract": long_contract.__dict__,
                "strike_price": calendar_data["strike_price"],
                "net_debit": calendar_data["net_debit"],
                "max_profit": calendar_data["max_profit"],
                "max_loss": calendar_data["max_loss"],
                "optimal_expiry_price": calendar_data["optimal_expiry_price"],
                "short_days": calendar_data["short_days"],
                "long_days": calendar_data["long_days"],
                "time_spread": calendar_data["time_spread"],
                "greeks": {
                    "delta": calendar_data["delta"],
                    "gamma": calendar_data["gamma"],
                    "theta": calendar_data["theta"],
                    "vega": calendar_data["vega"],
                    "rho": calendar_data["rho"]
                },
                "theta_ratio": calendar_data["theta_ratio"],
                "vega_ratio": calendar_data["vega_ratio"],
                "iv_at_entry": market_data.get("implied_volatility", 0)
            }
        )

        return signal

    async def _create_exit_signal(self, position: CalendarSpreadPosition, reason: str) -> StrategySignal:
        """Create exit signal for Calendar Spread position"""
        return StrategySignal(
            strategy_name="calendar_spread",
            symbol=position.symbol,
            action="CLOSE_CALENDAR_SPREAD",
            confidence=0.90,
            reasoning=f"Exit calendar spread: {reason}, P&L: {position.unrealized_pnl:.2f}",
            metadata={
                "strategy_type": "calendar_spread",
                "spread_type": position.spread_type.value,
                "exit_reason": reason,
                "net_debit": position.net_debit,
                "current_value": position.current_value,
                "unrealized_pnl": position.unrealized_pnl,
                "days_held": position.days_held,
                "strike_price": position.strike_price
            }
        )

    async def _create_front_month_mgmt_signal(self, position: CalendarSpreadPosition,
                                            market_data: Dict[str, Any]) -> StrategySignal:
        """Create front month management signal"""
        current_price = market_data["price"]

        if abs(current_price - position.strike_price) < 1.0:
            # Close to strike - manage carefully
            action = "MANAGE_FRONT_MONTH_ATM"
            reasoning = f"Front month expires in {position.short_days_to_expiry} days, price near strike"
        else:
            # Away from strike - let it expire or close
            action = "MANAGE_FRONT_MONTH_OTM"
            reasoning = f"Front month expires in {position.short_days_to_expiry} days, price away from strike"

        return StrategySignal(
            strategy_name="calendar_spread",
            symbol=position.symbol,
            action=action,
            confidence=0.80,
            reasoning=reasoning,
            metadata={
                "strategy_type": "calendar_spread_management",
                "current_price": current_price,
                "strike_price": position.strike_price,
                "short_days_to_expiry": position.short_days_to_expiry,
                "distance_from_strike": abs(current_price - position.strike_price)
            }
        )

    async def _create_roll_signal(self, position: CalendarSpreadPosition,
                                market_data: Dict[str, Any]) -> StrategySignal:
        """Create roll signal for calendar spread"""
        return StrategySignal(
            strategy_name="calendar_spread",
            symbol=position.symbol,
            action="ROLL_CALENDAR_FORWARD",
            confidence=0.75,
            reasoning=f"Roll calendar spread: front month has {position.short_days_to_expiry} days, "
                     f"price stable near ${position.strike_price:.2f}",
            metadata={
                "strategy_type": "calendar_spread_roll",
                "current_strike": position.strike_price,
                "short_days_to_expiry": position.short_days_to_expiry,
                "long_days_to_expiry": position.long_days_to_expiry,
                "current_price": market_data["price"]
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
        option_type = option_data.get("type", "call")
        strike = option_data.get("strike_price", underlying_price)
        days_to_expiry = (datetime.strptime(option_data.get("expiration"), "%Y-%m-%d") - datetime.now()).days

        # Simple approximations for Greeks
        moneyness = underlying_price / strike if strike > 0 else 1.0
        time_factor = math.sqrt(days_to_expiry / 365.0)

        if option_type == "call":
            delta = 0.5 if moneyness == 1.0 else (0.8 if moneyness > 1.0 else 0.2)
        else:
            delta = -0.5 if moneyness == 1.0 else (-0.2 if moneyness > 1.0 else -0.8)

        # Theta is more negative for shorter-term options
        theta = -0.02 * (30 / max(days_to_expiry, 1)) * time_factor

        return {
            "delta": delta,
            "gamma": 0.02 / max(time_factor, 0.1),
            "theta": theta,
            "vega": 0.1 * time_factor,
            "rho": 0.01 * time_factor
        }

    async def _update_position_metrics(self, position: CalendarSpreadPosition,
                                     market_data: Dict[str, Any]) -> None:
        """Update position metrics with current market data"""
        current_price = market_data["price"]

        # Update days to expiry
        now = datetime.now()
        position.short_days_to_expiry = (position.short_option_contract.expiration - now).days
        position.long_days_to_expiry = (position.long_option_contract.expiration - now).days
        position.days_held = (now - position.entry_date).days

        # Estimate current values (simplified)
        short_time_value = max(position.short_days_to_expiry / 30.0, 0.05)
        long_time_value = max(position.long_days_to_expiry / 30.0, 0.1)

        # Calculate intrinsic values
        if position.spread_type == CalendarSpreadType.CALL_CALENDAR:
            short_intrinsic = max(current_price - position.strike_price, 0)
            long_intrinsic = max(current_price - position.strike_price, 0)
        else:
            short_intrinsic = max(position.strike_price - current_price, 0)
            long_intrinsic = max(position.strike_price - current_price, 0)

        # Estimate option prices
        iv_factor = position.iv_at_entry
        estimated_short_price = short_intrinsic + (iv_factor * short_time_value * 2)
        estimated_long_price = long_intrinsic + (iv_factor * long_time_value * 3)

        # Calendar spread value is long - short
        position.current_value = estimated_long_price - estimated_short_price
        position.unrealized_pnl = position.current_value - position.net_debit

        # Update Greeks (simplified)
        distance_from_strike = abs(current_price - position.strike_price)

        position.delta = 0.0  # Calendar spreads are typically delta neutral
        position.gamma = -0.01 if distance_from_strike < 2 else -0.005

        # Theta should be positive as short option decays faster
        position.theta = 0.05 * (position.net_debit / 100) * (30 / max(position.short_days_to_expiry, 1))
        position.vega = 0.08 * (position.net_debit / 100)

    async def add_position(self, symbol: str, spread_type: CalendarSpreadType,
                          strike_price: float, short_contract: OptionContract,
                          long_contract: OptionContract, net_debit: float,
                          iv_at_entry: float) -> None:
        """Add new Calendar Spread position to tracking"""
        position = CalendarSpreadPosition(
            symbol=symbol,
            spread_type=spread_type,
            strike_price=strike_price,
            short_option_contract=short_contract,
            long_option_contract=long_contract,
            entry_date=datetime.now(),
            net_debit=net_debit,
            max_loss=net_debit,
            optimal_expiry_price=strike_price,
            iv_at_entry=iv_at_entry
        )

        if symbol not in self.active_positions:
            self.active_positions[symbol] = []

        self.active_positions[symbol].append(position)
        logger.info("Added Calendar Spread position", symbol=symbol,
                   spread_type=spread_type.value, strike=strike_price, debit=net_debit)

    def get_positions(self, symbol: str) -> List[CalendarSpreadPosition]:
        """Get calendar spread positions for symbol"""
        return self.active_positions.get(symbol, [])

    async def remove_position(self, symbol: str, position_index: int = 0) -> None:
        """Remove calendar spread position from tracking"""
        if symbol in self.active_positions and len(self.active_positions[symbol]) > position_index:
            removed_position = self.active_positions[symbol].pop(position_index)
            if not self.active_positions[symbol]:  # Remove empty list
                del self.active_positions[symbol]
            logger.info("Removed Calendar Spread position", symbol=symbol,
                       strike=removed_position.strike_price)

    def get_all_positions(self) -> Dict[str, List[CalendarSpreadPosition]]:
        """Get all active calendar spread positions"""
        return self.active_positions.copy()