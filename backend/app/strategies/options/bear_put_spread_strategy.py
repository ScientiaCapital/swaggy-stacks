"""
Bear Put Spread Options Strategy

Limited risk directional strategy for bearish outlook. Involves buying a higher
strike put and selling a lower strike put with the same expiration.
Provides defined maximum profit and maximum loss with lower cost than buying
a put outright.
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


class BearPutSpreadPhase(Enum):
    """Bear Put Spread phases"""
    ENTRY = "entry"                    # Looking for entry opportunity
    MONITORING = "monitoring"          # Monitoring existing position
    PROFIT_TAKING = "profit_taking"    # Taking profits
    TREND_FOLLOWING = "trend_following" # Following bearish trend
    EXPIRATION_MGMT = "expiration_mgmt" # Managing near expiration


@dataclass
class BearPutSpreadConfig:
    """Configuration for Bear Put Spread strategy"""

    # Market outlook criteria
    min_bearish_score: float = 0.60    # Minimum bearish sentiment score
    min_price_decline: float = 0.02    # Minimum 2% recent downward momentum
    max_volatility: float = 0.45       # Maximum IV for cost control

    # Technical criteria
    rsi_min: float = 30.0              # Don't enter if RSI < 30 (oversold)
    rsi_max: float = 60.0              # Prefer RSI < 60 (not overbought)
    macd_bearish: bool = True          # Prefer bearish MACD signal

    # Strike selection
    long_strike_delta_min: float = 0.45  # Long put delta range (absolute)
    long_strike_delta_max: float = 0.75
    spread_width_min: float = 2.50     # Minimum spread width
    spread_width_max: float = 10.0     # Maximum spread width
    target_distance_pct: float = 5.0   # Target 5% move to short strike

    # Entry criteria
    min_days_to_expiry: int = 14       # Minimum 2 weeks
    max_days_to_expiry: int = 60       # Maximum 60 days
    min_open_interest: int = 100       # Minimum open interest per leg
    max_bid_ask_spread_pct: float = 8.0  # Max 8% spread

    # Position management
    profit_target_pct: float = 75.0    # Target 75% of max profit
    stop_loss_pct: float = 50.0        # Stop loss at 50% of debit paid
    breakeven_exit: bool = True        # Exit at breakeven if trend fails
    time_decay_exit_days: int = 7      # Exit 7 days before expiration

    # Risk management
    max_position_size_pct: float = 3.0  # Max 3% of portfolio per spread
    min_risk_reward_ratio: float = 1.5  # Minimum 1.5:1 risk/reward
    max_cost_pct: float = 2.0          # Max 2% of underlying price for debit

    # Greeks thresholds
    min_delta_long: float = 0.40       # Minimum delta for long put (absolute)
    max_theta_decay: float = -0.10     # Maximum acceptable theta decay


@dataclass
class BearPutSpreadPosition:
    """Represents a Bear Put Spread position"""
    symbol: str
    long_put_contract: OptionContract   # Buy higher strike put
    short_put_contract: OptionContract  # Sell lower strike put
    entry_date: datetime
    entry_price: float                  # Underlying price at entry
    net_debit: float                    # Net debit paid
    spread_width: float                 # Difference between strikes
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    phase: BearPutSpreadPhase = BearPutSpreadPhase.ENTRY

    # Greeks for the combined position
    delta: float = 0.0                  # Should be negative (bearish)
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    # Strategy metrics
    max_profit: float = 0.0             # Spread width - debit paid
    max_loss: float = 0.0               # Debit paid
    breakeven_price: float = 0.0        # Long strike - debit paid
    target_price: float = 0.0           # Short strike (max profit)
    profit_zone_end: float = 0.0        # Price where position becomes profitable

    # Performance tracking
    days_held: int = 0
    highest_profit: float = 0.0
    current_profit_pct: float = 0.0
    time_decay_impact: float = 0.0
    trend_direction_score: float = 0.0


class BearPutSpreadStrategy:
    """
    Bear Put Spread options strategy implementation

    This strategy is used when moderately bearish on an underlying.
    It limits both profit potential and risk compared to buying puts outright.
    """

    def __init__(self, alpaca_client: AlpacaClient, config: Optional[BearPutSpreadConfig] = None):
        """Initialize Bear Put Spread strategy"""
        self.alpaca_client = alpaca_client
        self.config = config or BearPutSpreadConfig()
        self.active_positions: Dict[str, BearPutSpreadPosition] = {}
        self.monitoring_task: Optional[asyncio.Task] = None

        logger.info("Bear Put Spread strategy initialized", config=self.config)

    async def analyze_symbol(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """
        Analyze symbol for Bear Put Spread opportunities

        Args:
            symbol: Stock symbol to analyze
            market_data: Current market data including price, indicators, sentiment

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
            logger.error("Error analyzing bear put spread opportunity", symbol=symbol, error=str(e))
            return None

    async def _analyze_new_position(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze for new Bear Put Spread entry opportunity"""
        current_price = market_data["price"]

        # Check bearish criteria
        if not await self._check_bearish_conditions(market_data):
            return None

        # Check technical indicators
        if not self._check_technical_conditions(market_data):
            return None

        # Get option chain
        option_chain = await self._get_spread_option_chain(symbol, current_price)
        if not option_chain:
            return None

        # Find best bear put spread opportunity
        best_spread = await self._find_best_bear_put_spread(symbol, current_price, option_chain)
        if not best_spread:
            return None

        return await self._create_bear_put_signal(symbol, best_spread, market_data)

    async def _analyze_existing_position(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze existing Bear Put Spread position for management actions"""
        position = self.active_positions[symbol]
        current_price = market_data["price"]

        # Update position metrics
        await self._update_position_metrics(position, market_data)

        # Check for profit taking
        if position.current_profit_pct >= self.config.profit_target_pct:
            return await self._create_exit_signal(symbol, position, "PROFIT_TARGET")

        # Check for stop loss
        if position.current_profit_pct <= -self.config.stop_loss_pct:
            return await self._create_exit_signal(symbol, position, "STOP_LOSS")

        # Check breakeven exit if trend failing
        if (self.config.breakeven_exit and
            current_price > position.breakeven_price and
            position.days_held > 14):
            trend_score = market_data.get("trend_score", 0.5)
            if trend_score > 0.6:  # Trend turning bullish
                return await self._create_exit_signal(symbol, position, "TREND_FAILURE")

        # Check time decay exit
        days_to_expiry = (position.long_put_contract.expiration - datetime.now()).days
        if days_to_expiry <= self.config.time_decay_exit_days:
            return await self._create_exit_signal(symbol, position, "TIME_DECAY")

        # Check for rolling opportunity if profitable and trend continues
        if (position.current_profit_pct > 30 and
            current_price < position.target_price * 1.05 and
            days_to_expiry <= 21):
            return await self._create_roll_signal(symbol, position, market_data)

        return None

    async def _check_bearish_conditions(self, market_data: Dict[str, Any]) -> bool:
        """Check if market conditions are bearish enough for bear put spread"""
        # Check sentiment score
        bearish_score = market_data.get("bearish_sentiment", 0.5)
        if bearish_score < self.config.min_bearish_score:
            return False

        # Check price momentum (negative for bearish)
        price_momentum = market_data.get("price_momentum_5d", 0.0)
        if price_momentum > -self.config.min_price_decline:
            return False

        # Check volatility (prefer reasonable IV for spreads)
        iv = market_data.get("implied_volatility", 0.3)
        if iv > self.config.max_volatility:
            return False

        return True

    def _check_technical_conditions(self, market_data: Dict[str, Any]) -> bool:
        """Check technical indicators for bear put spread suitability"""
        # Check RSI
        rsi = market_data.get("rsi", 50.0)
        if not (self.config.rsi_min <= rsi <= self.config.rsi_max):
            return False

        # Check MACD if required
        if self.config.macd_bearish:
            macd_signal = market_data.get("macd_signal", "neutral")
            if macd_signal != "bearish":
                return False

        # Check resistance levels
        distance_from_resistance = market_data.get("distance_from_resistance", 0.0)
        if distance_from_resistance < -0.10:  # More than 10% below resistance might be risky
            return False

        return True

    async def _get_spread_option_chain(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """Get filtered option chain for bear put spread analysis"""
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

            # Filter for puts only with good liquidity
            filtered_chain = []
            for option in option_chain:
                if option.get("type") != "put":
                    continue

                strike = option.get("strike_price", 0)
                open_interest = option.get("open_interest", 0)
                bid_ask_spread = option.get("ask", 0) - option.get("bid", 0)
                bid_ask_spread_pct = (bid_ask_spread / option.get("ask", 1)) * 100 if option.get("ask", 0) > 0 else 100

                # Filter for reasonable strikes and liquidity
                if (strike >= current_price * 0.85 and strike <= current_price * 1.05 and
                    open_interest >= self.config.min_open_interest and
                    bid_ask_spread_pct <= self.config.max_bid_ask_spread_pct):
                    filtered_chain.append(option)

            return filtered_chain

        except Exception as e:
            logger.error("Error filtering option chain for bear put spread", symbol=symbol, error=str(e))
            return []

    async def _find_best_bear_put_spread(self, symbol: str, current_price: float,
                                       option_chain: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the best bear put spread opportunity from option chain"""
        best_spread = None
        best_score = 0

        # Group options by expiration
        expirations = {}
        for option in option_chain:
            exp_date = option.get("expiration")
            if exp_date not in expirations:
                expirations[exp_date] = []
            expirations[exp_date].append(option)

        # Evaluate spreads for each expiration
        for exp_date, options in expirations.items():
            options.sort(key=lambda x: x.get("strike_price", 0), reverse=True)  # High to low for puts

            # Try different long/short combinations
            for i, long_option in enumerate(options):
                long_strike = long_option.get("strike_price")
                long_delta = await self._estimate_delta(long_option, current_price)

                # Check if long put meets delta criteria (absolute value)
                if not (self.config.long_strike_delta_min <= abs(long_delta) <= self.config.long_strike_delta_max):
                    continue

                # Look for short strikes (lower strikes)
                for j, short_option in enumerate(options[i+1:], i+1):
                    short_strike = short_option.get("strike_price")
                    spread_width = long_strike - short_strike

                    # Check spread width
                    if not (self.config.spread_width_min <= spread_width <= self.config.spread_width_max):
                        continue

                    # Check target distance
                    target_distance_pct = ((current_price - short_strike) / current_price) * 100
                    if target_distance_pct > self.config.target_distance_pct * 2:  # Too far OTM
                        continue

                    # Calculate spread metrics
                    spread_data = await self._calculate_spread_metrics(
                        long_option, short_option, current_price, spread_width
                    )

                    if spread_data and self._passes_spread_filters(spread_data):
                        score = await self._score_bear_put_spread(spread_data)
                        if score > best_score:
                            best_score = score
                            best_spread = spread_data

        return best_spread

    async def _calculate_spread_metrics(self, long_option: Dict, short_option: Dict,
                                      current_price: float, spread_width: float) -> Optional[Dict[str, Any]]:
        """Calculate key metrics for a bear put spread"""
        try:
            long_price = (long_option.get("bid", 0) + long_option.get("ask", 0)) / 2
            short_price = (short_option.get("bid", 0) + short_option.get("ask", 0)) / 2

            net_debit = long_price - short_price
            if net_debit <= 0:  # Must be a debit spread
                return None

            long_strike = long_option.get("strike_price")
            short_strike = short_option.get("strike_price")

            max_profit = spread_width - net_debit
            max_loss = net_debit
            breakeven_price = long_strike - net_debit

            # Check cost as percentage of underlying
            cost_pct = (net_debit / current_price) * 100
            if cost_pct > self.config.max_cost_pct:
                return None

            # Check risk/reward ratio
            risk_reward_ratio = max_profit / max_loss if max_loss > 0 else 0
            if risk_reward_ratio < self.config.min_risk_reward_ratio:
                return None

            # Calculate Greeks
            long_greeks = await self._calculate_greeks(long_option, current_price)
            short_greeks = await self._calculate_greeks(short_option, current_price)

            combined_delta = long_greeks["delta"] - short_greeks["delta"]
            combined_gamma = long_greeks["gamma"] - short_greeks["gamma"]
            combined_theta = long_greeks["theta"] - short_greeks["theta"]
            combined_vega = long_greeks["vega"] - short_greeks["vega"]
            combined_rho = long_greeks["rho"] - short_greeks["rho"]

            return {
                "long_option": long_option,
                "short_option": short_option,
                "long_strike": long_strike,
                "short_strike": short_strike,
                "long_price": long_price,
                "short_price": short_price,
                "net_debit": net_debit,
                "spread_width": spread_width,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "breakeven_price": breakeven_price,
                "cost_pct": cost_pct,
                "risk_reward_ratio": risk_reward_ratio,
                "delta": combined_delta,
                "gamma": combined_gamma,
                "theta": combined_theta,
                "vega": combined_vega,
                "rho": combined_rho,
                "long_delta": long_greeks["delta"],
                "expiration": long_option.get("expiration"),
                "days_to_expiry": (datetime.strptime(long_option.get("expiration"), "%Y-%m-%d") - datetime.now()).days
            }

        except Exception as e:
            logger.error("Error calculating bear put spread metrics", error=str(e))
            return None

    def _passes_spread_filters(self, spread_data: Dict[str, Any]) -> bool:
        """Check if bear put spread passes all filters"""
        # Check minimum delta for long put (absolute value)
        if abs(spread_data["long_delta"]) < self.config.min_delta_long:
            return False

        # Check theta decay
        if spread_data["theta"] < self.config.max_theta_decay:
            return False

        # Check negative delta (bearish position)
        if spread_data["delta"] >= 0:
            return False

        return True

    async def _score_bear_put_spread(self, spread_data: Dict[str, Any]) -> float:
        """Score bear put spread opportunity based on multiple factors"""
        score = 0.0

        # Reward good risk/reward ratio
        risk_reward = min(spread_data["risk_reward_ratio"], 4.0)
        score += risk_reward * 20.0  # Up to 80 points

        # Reward good delta exposure (want significant negative delta)
        delta_score = 25.0 * min(abs(spread_data["delta"]), 0.8)  # Up to 20 points
        score += delta_score

        # Reward reasonable cost
        cost_pct = spread_data["cost_pct"]
        if cost_pct <= 1.0:
            score += 20.0 * (1 - cost_pct)

        # Reward optimal time to expiry (sweet spot 30-45 days)
        days_to_expiry = spread_data["days_to_expiry"]
        if 25 <= days_to_expiry <= 50:
            score += 15.0 * (1 - abs(days_to_expiry - 37.5) / 25)

        # Reward good long put delta
        long_delta = abs(spread_data["long_delta"])
        if 0.50 <= long_delta <= 0.70:
            score += 10.0 * (1 - abs(long_delta - 0.60) / 0.20)

        # Small penalty for theta decay
        theta_penalty = abs(spread_data["theta"]) * 10
        score -= min(theta_penalty, 5.0)

        return max(score, 0.0)

    async def _create_bear_put_signal(self, symbol: str, spread_data: Dict[str, Any],
                                    market_data: Dict[str, Any]) -> StrategySignal:
        """Create strategy signal for Bear Put Spread entry"""
        long_contract = await self._convert_to_option_contract(spread_data["long_option"])
        short_contract = await self._convert_to_option_contract(spread_data["short_option"])

        confidence = min(await self._score_bear_put_spread(spread_data) / 100.0, 0.95)

        signal = StrategySignal(
            strategy_name="bear_put_spread",
            symbol=symbol,
            action="OPEN_BEAR_PUT_SPREAD",
            confidence=confidence,
            reasoning=f"Bear Put Spread entry: {spread_data['long_strike']:.2f}/{spread_data['short_strike']:.2f} "
                     f"for ${spread_data['net_debit']:.2f} debit, max profit ${spread_data['max_profit']:.2f}, "
                     f"breakeven ${spread_data['breakeven_price']:.2f}, "
                     f"R/R ratio {spread_data['risk_reward_ratio']:.1f}:1",
            metadata={
                "strategy_type": "bear_put_spread",
                "long_contract": long_contract.__dict__,
                "short_contract": short_contract.__dict__,
                "long_strike": spread_data["long_strike"],
                "short_strike": spread_data["short_strike"],
                "net_debit": spread_data["net_debit"],
                "spread_width": spread_data["spread_width"],
                "max_profit": spread_data["max_profit"],
                "max_loss": spread_data["max_loss"],
                "breakeven_price": spread_data["breakeven_price"],
                "risk_reward_ratio": spread_data["risk_reward_ratio"],
                "greeks": {
                    "delta": spread_data["delta"],
                    "gamma": spread_data["gamma"],
                    "theta": spread_data["theta"],
                    "vega": spread_data["vega"],
                    "rho": spread_data["rho"]
                },
                "days_to_expiry": spread_data["days_to_expiry"],
                "entry_price": market_data["price"],
                "bearish_score": market_data.get("bearish_sentiment", 0.6)
            }
        )

        return signal

    async def _create_exit_signal(self, symbol: str, position: BearPutSpreadPosition,
                                reason: str) -> StrategySignal:
        """Create exit signal for Bear Put Spread position"""
        return StrategySignal(
            strategy_name="bear_put_spread",
            symbol=symbol,
            action="CLOSE_BEAR_PUT_SPREAD",
            confidence=0.90,
            reasoning=f"Exit Bear Put Spread: {reason}, P&L: {position.unrealized_pnl:.2f}, "
                     f"profit: {position.current_profit_pct:.1f}%",
            metadata={
                "strategy_type": "bear_put_spread",
                "exit_reason": reason,
                "net_debit": position.net_debit,
                "current_value": position.current_value,
                "unrealized_pnl": position.unrealized_pnl,
                "current_profit_pct": position.current_profit_pct,
                "days_held": position.days_held,
                "highest_profit": position.highest_profit,
                "long_strike": position.long_put_contract.strike_price,
                "short_strike": position.short_put_contract.strike_price
            }
        )

    async def _create_roll_signal(self, symbol: str, position: BearPutSpreadPosition,
                                market_data: Dict[str, Any]) -> StrategySignal:
        """Create roll signal for profitable Bear Put Spread"""
        return StrategySignal(
            strategy_name="bear_put_spread",
            symbol=symbol,
            action="ROLL_BEAR_PUT_SPREAD",
            confidence=0.75,
            reasoning=f"Roll Bear Put Spread: profitable position with continued bearish trend, "
                     f"current profit: {position.current_profit_pct:.1f}%",
            metadata={
                "strategy_type": "bear_put_spread_roll",
                "current_profit_pct": position.current_profit_pct,
                "current_price": market_data["price"],
                "long_strike": position.long_put_contract.strike_price,
                "short_strike": position.short_put_contract.strike_price,
                "days_to_expiry": (position.long_put_contract.expiration - datetime.now()).days
            }
        )

    async def _estimate_delta(self, option_data: Dict[str, Any], underlying_price: float) -> float:
        """Estimate delta for option"""
        strike = option_data.get("strike_price", underlying_price)
        option_type = option_data.get("type", "put")

        # Simple delta estimation based on moneyness
        if option_type == "call":
            if strike <= underlying_price:
                return 0.6 + (underlying_price - strike) / underlying_price * 0.3
            else:
                return 0.4 - (strike - underlying_price) / underlying_price * 0.3
        else:  # put
            if strike >= underlying_price:
                return -0.6 - (strike - underlying_price) / underlying_price * 0.3
            else:
                return -0.4 + (underlying_price - strike) / underlying_price * 0.3

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
        option_type = option_data.get("type", "put")
        strike = option_data.get("strike_price", underlying_price)
        days_to_expiry = (datetime.strptime(option_data.get("expiration"), "%Y-%m-%d") - datetime.now()).days

        # Simple approximations
        moneyness = underlying_price / strike if strike > 0 else 1.0
        time_factor = math.sqrt(days_to_expiry / 365.0)

        if option_type == "call":
            delta = 0.5 if moneyness == 1.0 else (0.8 if moneyness > 1.0 else 0.2)
        else:  # put
            delta = -0.5 if moneyness == 1.0 else (-0.2 if moneyness > 1.0 else -0.8)

        gamma = 0.02 / max(time_factor, 0.1)
        theta = -0.03 * (30 / max(days_to_expiry, 1)) * time_factor
        vega = 0.1 * time_factor
        rho = 0.01 * time_factor * (-1 if option_type == "put" else 1)

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho
        }

    async def _update_position_metrics(self, position: BearPutSpreadPosition,
                                     market_data: Dict[str, Any]) -> None:
        """Update position metrics with current market data"""
        current_price = market_data["price"]
        now = datetime.now()

        # Update basic metrics
        position.days_held = (now - position.entry_date).days
        days_to_expiry = (position.long_put_contract.expiration - now).days

        # Estimate current spread value
        long_intrinsic = max(position.long_put_contract.strike_price - current_price, 0)
        short_intrinsic = max(position.short_put_contract.strike_price - current_price, 0)

        # Add time value estimates
        time_factor = max(days_to_expiry / 30.0, 0.05)
        iv_factor = position.long_put_contract.implied_volatility

        long_time_value = iv_factor * time_factor * 2 if long_intrinsic == 0 else iv_factor * time_factor
        short_time_value = iv_factor * time_factor * 2 if short_intrinsic == 0 else iv_factor * time_factor

        estimated_long_price = long_intrinsic + long_time_value
        estimated_short_price = short_intrinsic + short_time_value

        position.current_value = estimated_long_price - estimated_short_price
        position.unrealized_pnl = position.current_value - position.net_debit
        position.current_profit_pct = (position.unrealized_pnl / position.net_debit) * 100

        # Track highest profit
        if position.unrealized_pnl > position.highest_profit:
            position.highest_profit = position.unrealized_pnl

        # Update Greeks (simplified)
        distance_from_long = position.long_put_contract.strike_price - current_price
        distance_from_short = position.short_put_contract.strike_price - current_price

        if distance_from_long > 0 and distance_from_short < 0:
            # Between strikes - maximum delta
            position.delta = -0.6
        elif distance_from_short > 0:
            # Below short strike - delta approaches zero
            position.delta = -0.2
        else:
            # Above long strike - delta approaches long put delta
            position.delta = -0.4

        position.gamma = 0.03 if abs(distance_from_long) < 2 else 0.01
        position.theta = -0.05 * (position.net_debit / 100) * (30 / max(days_to_expiry, 1))
        position.vega = 0.08 * (position.net_debit / 100) * time_factor

        # Calculate trend direction score (negative for bearish)
        price_change = (current_price - position.entry_price) / position.entry_price
        position.trend_direction_score = min(max(-price_change * 10, -1.0), 1.0)

    async def add_position(self, symbol: str, long_put_contract: OptionContract,
                          short_put_contract: OptionContract, net_debit: float,
                          entry_price: float) -> None:
        """Add new Bear Put Spread position to tracking"""
        spread_width = long_put_contract.strike_price - short_put_contract.strike_price
        max_profit = spread_width - net_debit
        breakeven_price = long_put_contract.strike_price - net_debit

        position = BearPutSpreadPosition(
            symbol=symbol,
            long_put_contract=long_put_contract,
            short_put_contract=short_put_contract,
            entry_date=datetime.now(),
            entry_price=entry_price,
            net_debit=net_debit,
            spread_width=spread_width,
            max_profit=max_profit,
            max_loss=net_debit,
            breakeven_price=breakeven_price,
            target_price=short_put_contract.strike_price,
            profit_zone_end=breakeven_price
        )

        self.active_positions[symbol] = position
        logger.info("Added Bear Put Spread position", symbol=symbol,
                   long_strike=long_put_contract.strike_price,
                   short_strike=short_put_contract.strike_price,
                   debit=net_debit)

    def get_position(self, symbol: str) -> Optional[BearPutSpreadPosition]:
        """Get Bear Put Spread position for symbol"""
        return self.active_positions.get(symbol)

    async def remove_position(self, symbol: str) -> None:
        """Remove Bear Put Spread position from tracking"""
        if symbol in self.active_positions:
            del self.active_positions[symbol]
            logger.info("Removed Bear Put Spread position", symbol=symbol)

    def get_all_positions(self) -> Dict[str, BearPutSpreadPosition]:
        """Get all active Bear Put Spread positions"""
        return self.active_positions.copy()