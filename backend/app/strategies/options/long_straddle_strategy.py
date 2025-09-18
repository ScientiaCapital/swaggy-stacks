"""
Long Straddle Options Strategy

Volatility strategy that profits from large price movements in either direction.
Involves buying both a call and put at the same strike price and expiration.
Maximum profit is unlimited on upside, substantial on downside.
Maximum loss is limited to premium paid.
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


class StraddlePhase(Enum):
    """Long Straddle phases"""
    ENTRY = "entry"           # Looking for entry opportunity
    MONITORING = "monitoring" # Monitoring existing position
    PROFIT_TAKING = "profit_taking"  # Taking profits
    LOSS_CUTTING = "loss_cutting"    # Cutting losses


@dataclass
class LongStraddleConfig:
    """Configuration for Long Straddle strategy"""

    # Volatility criteria
    min_iv: float = 0.15  # 15% minimum implied volatility
    max_iv: float = 0.45  # 45% maximum implied volatility
    iv_rank_min: float = 0.20  # Minimum IV rank (20th percentile)
    iv_rank_max: float = 0.80  # Maximum IV rank (80th percentile)

    # Entry criteria
    min_days_to_expiry: int = 14    # Minimum 2 weeks
    max_days_to_expiry: int = 45    # Maximum 45 days
    min_open_interest: int = 100    # Minimum open interest per leg
    max_bid_ask_spread_pct: float = 8.0  # Max 8% spread for straddles

    # Strike selection
    atm_threshold: float = 0.02  # Within 2% of ATM for strike selection

    # Position management
    profit_target_pct: float = 100.0   # Target 100% profit (double money)
    stop_loss_pct: float = 50.0        # Stop loss at 50% of premium paid
    time_decay_exit_days: int = 7       # Exit 7 days before expiration

    # Risk management
    max_position_size_pct: float = 5.0  # Max 5% of portfolio per straddle
    min_breakeven_range_pct: float = 8.0  # Minimum 8% breakeven range

    # Greeks thresholds
    max_theta_per_day: float = -5.0     # Maximum theta decay per day
    min_vega_exposure: float = 0.10     # Minimum vega for volatility exposure


@dataclass
class StraddlePosition:
    """Represents a Long Straddle position"""
    symbol: str
    call_contract: OptionContract
    put_contract: OptionContract
    entry_date: datetime
    entry_cost: float  # Total premium paid
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    phase: StraddlePhase = StraddlePhase.ENTRY

    # Greeks for the combined position
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    # Breakeven points
    upside_breakeven: float = 0.0
    downside_breakeven: float = 0.0

    # Risk metrics
    max_loss: float = 0.0  # Limited to premium paid
    days_held: int = 0
    iv_at_entry: float = 0.0


class LongStraddleStrategy:
    """
    Long Straddle options strategy implementation

    This strategy is designed for high volatility environments where
    large price movements are expected. It's market neutral at initiation
    but profits from volatility expansion.
    """

    def __init__(self, alpaca_client: AlpacaClient, config: Optional[LongStraddleConfig] = None):
        """Initialize Long Straddle strategy"""
        self.alpaca_client = alpaca_client
        self.config = config or LongStraddleConfig()
        self.active_positions: Dict[str, StraddlePosition] = {}
        self.monitoring_task: Optional[asyncio.Task] = None

        logger.info("Long Straddle strategy initialized", config=self.config)

    async def analyze_symbol(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """
        Analyze symbol for Long Straddle opportunities

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
            logger.error("Error analyzing straddle opportunity", symbol=symbol, error=str(e))
            return None

    async def _analyze_new_position(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze for new Long Straddle entry opportunity"""
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

        # Get option chain
        option_chain = await self._get_filtered_option_chain(symbol, current_price)
        if not option_chain:
            return None

        # Find best straddle opportunity
        best_straddle = await self._find_best_straddle_opportunity(symbol, current_price, option_chain)
        if not best_straddle:
            return None

        return await self._create_straddle_signal(symbol, best_straddle, market_data)

    async def _analyze_existing_position(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze existing Long Straddle position for management actions"""
        position = self.active_positions[symbol]
        current_price = market_data["price"]

        # Update position metrics
        await self._update_position_metrics(position, market_data)

        # Check for profit taking
        profit_pct = (position.unrealized_pnl / position.entry_cost) * 100
        if profit_pct >= self.config.profit_target_pct:
            return await self._create_exit_signal(symbol, position, "PROFIT_TARGET")

        # Check for stop loss
        if profit_pct <= -self.config.stop_loss_pct:
            return await self._create_exit_signal(symbol, position, "STOP_LOSS")

        # Check time decay exit
        days_to_expiry = (position.call_contract.expiration - datetime.now()).days
        if days_to_expiry <= self.config.time_decay_exit_days:
            return await self._create_exit_signal(symbol, position, "TIME_DECAY")

        # Check for adjustments if needed
        if abs(position.delta) > 0.15:  # Position getting too directional
            return await self._create_adjustment_signal(symbol, position, market_data)

        return None

    async def _get_filtered_option_chain(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """Get filtered option chain for straddle analysis"""
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

            # Filter for ATM strikes and liquidity
            filtered_chain = []
            for option in option_chain:
                strike = option.get("strike_price", 0)
                open_interest = option.get("open_interest", 0)
                bid_ask_spread = option.get("ask", 0) - option.get("bid", 0)
                bid_ask_spread_pct = (bid_ask_spread / option.get("ask", 1)) * 100 if option.get("ask", 0) > 0 else 100

                # Check if strike is near ATM
                atm_distance = abs(strike - current_price) / current_price
                if atm_distance <= self.config.atm_threshold:
                    # Check liquidity criteria
                    if (open_interest >= self.config.min_open_interest and
                        bid_ask_spread_pct <= self.config.max_bid_ask_spread_pct):
                        filtered_chain.append(option)

            return filtered_chain

        except Exception as e:
            logger.error("Error filtering option chain", symbol=symbol, error=str(e))
            return []

    async def _find_best_straddle_opportunity(self, symbol: str, current_price: float,
                                           option_chain: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the best straddle opportunity from option chain"""
        best_straddle = None
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

        # Evaluate each potential straddle
        for exp_date, strikes in expirations.items():
            for strike, options in strikes.items():
                if "call" in options and "put" in options:
                    call_option = options["call"]
                    put_option = options["put"]

                    # Calculate straddle metrics
                    straddle_data = await self._calculate_straddle_metrics(
                        call_option, put_option, current_price
                    )

                    if straddle_data and self._passes_straddle_filters(straddle_data):
                        score = await self._score_straddle_opportunity(straddle_data)
                        if score > best_score:
                            best_score = score
                            best_straddle = straddle_data

        return best_straddle

    async def _calculate_straddle_metrics(self, call_option: Dict, put_option: Dict,
                                        current_price: float) -> Optional[Dict[str, Any]]:
        """Calculate key metrics for a straddle"""
        try:
            call_price = (call_option.get("bid", 0) + call_option.get("ask", 0)) / 2
            put_price = (put_option.get("bid", 0) + put_option.get("ask", 0)) / 2
            strike = call_option.get("strike_price")

            total_cost = call_price + put_price
            upside_breakeven = strike + total_cost
            downside_breakeven = strike - total_cost
            breakeven_range_pct = ((upside_breakeven - downside_breakeven) / current_price) * 100

            # Calculate Greeks for both legs
            call_greeks = await self._calculate_greeks(call_option, current_price)
            put_greeks = await self._calculate_greeks(put_option, current_price)

            # Combined Greeks
            combined_delta = call_greeks["delta"] + put_greeks["delta"]
            combined_gamma = call_greeks["gamma"] + put_greeks["gamma"]
            combined_theta = call_greeks["theta"] + put_greeks["theta"]
            combined_vega = call_greeks["vega"] + put_greeks["vega"]
            combined_rho = call_greeks["rho"] + put_greeks["rho"]

            return {
                "call_option": call_option,
                "put_option": put_option,
                "call_price": call_price,
                "put_price": put_price,
                "total_cost": total_cost,
                "strike": strike,
                "upside_breakeven": upside_breakeven,
                "downside_breakeven": downside_breakeven,
                "breakeven_range_pct": breakeven_range_pct,
                "delta": combined_delta,
                "gamma": combined_gamma,
                "theta": combined_theta,
                "vega": combined_vega,
                "rho": combined_rho,
                "expiration": call_option.get("expiration"),
                "days_to_expiry": (datetime.strptime(call_option.get("expiration"), "%Y-%m-%d") - datetime.now()).days
            }

        except Exception as e:
            logger.error("Error calculating straddle metrics", error=str(e))
            return None

    def _passes_straddle_filters(self, straddle_data: Dict[str, Any]) -> bool:
        """Check if straddle passes all filters"""
        # Check breakeven range
        if straddle_data["breakeven_range_pct"] < self.config.min_breakeven_range_pct:
            return False

        # Check theta decay
        if straddle_data["theta"] < self.config.max_theta_per_day:
            return False

        # Check vega exposure
        if straddle_data["vega"] < self.config.min_vega_exposure:
            return False

        return True

    async def _score_straddle_opportunity(self, straddle_data: Dict[str, Any]) -> float:
        """Score straddle opportunity based on multiple factors"""
        score = 0.0

        # Reward good breakeven range (but not too wide)
        breakeven_pct = straddle_data["breakeven_range_pct"]
        if 8.0 <= breakeven_pct <= 15.0:
            score += 30.0 * (1 - abs(breakeven_pct - 11.5) / 7.5)

        # Reward high vega (volatility exposure)
        vega_score = min(straddle_data["vega"] * 20, 25.0)
        score += vega_score

        # Reward reasonable time to expiry (sweet spot around 30-35 days)
        days_to_expiry = straddle_data["days_to_expiry"]
        if 25 <= days_to_expiry <= 40:
            score += 20.0 * (1 - abs(days_to_expiry - 32.5) / 15)

        # Penalize excessive theta decay
        theta_penalty = abs(straddle_data["theta"]) * 2
        score -= min(theta_penalty, 15.0)

        # Reward delta neutrality
        delta_neutrality = 10.0 * (1 - abs(straddle_data["delta"]) * 10)
        score += max(delta_neutrality, 0)

        return max(score, 0.0)

    async def _create_straddle_signal(self, symbol: str, straddle_data: Dict[str, Any],
                                    market_data: Dict[str, Any]) -> StrategySignal:
        """Create strategy signal for straddle entry"""
        call_contract = await self._convert_to_option_contract(straddle_data["call_option"])
        put_contract = await self._convert_to_option_contract(straddle_data["put_option"])

        signal = StrategySignal(
            strategy_name="long_straddle",
            symbol=symbol,
            action="BUY_STRADDLE",
            confidence=min(await self._score_straddle_opportunity(straddle_data) / 100.0, 0.95),
            reasoning=f"Long straddle entry: IV={market_data.get('implied_volatility', 0):.1%}, "
                     f"breakeven range={straddle_data['breakeven_range_pct']:.1f}%, "
                     f"theta={straddle_data['theta']:.2f}, vega={straddle_data['vega']:.2f}",
            metadata={
                "strategy_type": "long_straddle",
                "call_contract": call_contract.__dict__,
                "put_contract": put_contract.__dict__,
                "total_cost": straddle_data["total_cost"],
                "upside_breakeven": straddle_data["upside_breakeven"],
                "downside_breakeven": straddle_data["downside_breakeven"],
                "breakeven_range_pct": straddle_data["breakeven_range_pct"],
                "greeks": {
                    "delta": straddle_data["delta"],
                    "gamma": straddle_data["gamma"],
                    "theta": straddle_data["theta"],
                    "vega": straddle_data["vega"],
                    "rho": straddle_data["rho"]
                },
                "days_to_expiry": straddle_data["days_to_expiry"],
                "iv_at_entry": market_data.get("implied_volatility", 0)
            }
        )

        return signal

    async def _create_exit_signal(self, symbol: str, position: StraddlePosition,
                                reason: str) -> StrategySignal:
        """Create exit signal for straddle position"""
        return StrategySignal(
            strategy_name="long_straddle",
            symbol=symbol,
            action="SELL_STRADDLE",
            confidence=0.90,
            reasoning=f"Exit straddle position: {reason}, P&L: {position.unrealized_pnl:.2f}",
            metadata={
                "strategy_type": "long_straddle",
                "exit_reason": reason,
                "entry_cost": position.entry_cost,
                "current_value": position.current_value,
                "unrealized_pnl": position.unrealized_pnl,
                "days_held": position.days_held,
                "call_contract": position.call_contract.__dict__,
                "put_contract": position.put_contract.__dict__
            }
        )

    async def _create_adjustment_signal(self, symbol: str, position: StraddlePosition,
                                      market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Create adjustment signal for delta-heavy straddle"""
        # Simple delta hedge by trading underlying
        hedge_shares = int(-position.delta * 100)  # Each option contract represents 100 shares

        if abs(hedge_shares) < 10:  # Not worth hedging small deltas
            return None

        action = "BUY" if hedge_shares > 0 else "SELL"

        return StrategySignal(
            strategy_name="long_straddle",
            symbol=symbol,
            action=f"{action}_HEDGE",
            confidence=0.75,
            reasoning=f"Delta hedge straddle: position delta={position.delta:.3f}, hedge {abs(hedge_shares)} shares",
            metadata={
                "strategy_type": "long_straddle_hedge",
                "hedge_shares": abs(hedge_shares),
                "position_delta": position.delta,
                "current_price": market_data.get("price")
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
        return {
            "delta": option_data.get("delta", 0.5 if option_data.get("type") == "call" else -0.5),
            "gamma": option_data.get("gamma", 0.01),
            "theta": option_data.get("theta", -0.05),
            "vega": option_data.get("vega", 0.1),
            "rho": option_data.get("rho", 0.02)
        }

    async def _update_position_metrics(self, position: StraddlePosition,
                                     market_data: Dict[str, Any]) -> None:
        """Update position metrics with current market data"""
        current_price = market_data["price"]

        # Get current option prices (would typically fetch from market)
        # For now, estimate based on intrinsic + time value
        call_intrinsic = max(current_price - position.call_contract.strike_price, 0)
        put_intrinsic = max(position.put_contract.strike_price - current_price, 0)

        # Simple time value estimation (would be more sophisticated in production)
        days_to_expiry = (position.call_contract.expiration - datetime.now()).days
        time_value_factor = max(days_to_expiry / 30.0, 0.1)

        estimated_call_price = call_intrinsic + (position.call_contract.implied_volatility * time_value_factor * 2)
        estimated_put_price = put_intrinsic + (position.put_contract.implied_volatility * time_value_factor * 2)

        position.current_value = estimated_call_price + estimated_put_price
        position.unrealized_pnl = position.current_value - position.entry_cost
        position.days_held = (datetime.now() - position.entry_date).days

        # Update Greeks (simplified)
        position.delta = 0.5 - 0.5  # Call delta + Put delta ≈ 0 for ATM straddle
        position.gamma = 0.02  # High gamma for ATM options
        position.theta = -0.10 * (position.entry_cost / 100)  # Proportional theta decay
        position.vega = 0.15 * (position.entry_cost / 100)   # Proportional vega exposure

    async def add_position(self, symbol: str, call_contract: OptionContract,
                          put_contract: OptionContract, entry_cost: float,
                          iv_at_entry: float) -> None:
        """Add new straddle position to tracking"""
        position = StraddlePosition(
            symbol=symbol,
            call_contract=call_contract,
            put_contract=put_contract,
            entry_date=datetime.now(),
            entry_cost=entry_cost,
            upside_breakeven=call_contract.strike_price + entry_cost,
            downside_breakeven=call_contract.strike_price - entry_cost,
            max_loss=entry_cost,
            iv_at_entry=iv_at_entry
        )

        self.active_positions[symbol] = position
        logger.info("Added long straddle position", symbol=symbol, entry_cost=entry_cost)

    def get_position(self, symbol: str) -> Optional[StraddlePosition]:
        """Get straddle position for symbol"""
        return self.active_positions.get(symbol)

    async def remove_position(self, symbol: str) -> None:
        """Remove straddle position from tracking"""
        if symbol in self.active_positions:
            del self.active_positions[symbol]
            logger.info("Removed long straddle position", symbol=symbol)

    def get_all_positions(self) -> Dict[str, StraddlePosition]:
        """Get all active straddle positions"""
        return self.active_positions.copy()