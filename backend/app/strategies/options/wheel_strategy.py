"""
Wheel Options Strategy

Income generation strategy that combines cash-secured puts and covered calls.
Uses Bollinger Bands for technical analysis and maintains defined risk through
systematic rotation between two phases.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import structlog
from app.core.exceptions import TradingError
from app.trading.alpaca_client import AlpacaClient
from app.trading.options_trading import OptionContract, GreeksData, OptionType
from app.rag.strategies.strategy_engines import StrategySignal

logger = structlog.get_logger()


class WheelPhase(Enum):
    """Wheel strategy phases"""
    CASH_SECURED_PUT = "CSP"  # Selling cash-secured puts
    COVERED_CALL = "CC"       # Selling covered calls on assigned shares


@dataclass
class WheelConfig:
    """Configuration for Wheel strategy"""

    # Delta thresholds
    csp_delta_min: float = -0.42  # Cash-secured put delta range
    csp_delta_max: float = -0.18
    cc_delta_min: float = 0.18    # Covered call delta range
    cc_delta_max: float = 0.42

    # Expiration range
    min_days_to_expiry: int = 7
    max_days_to_expiry: int = 35

    # Filtering criteria
    min_open_interest: int = 200
    max_bid_ask_spread_pct: float = 5.0  # Max 5% spread

    # Bollinger Band parameters
    bb_period: int = 20
    bb_std_dev: float = 2.0

    # Rolling thresholds
    roll_delta_threshold: float = 0.80  # Roll when delta exceeds this
    roll_profit_threshold: float = 50.0  # Roll when 50%+ profit achieved

    # Risk management
    max_position_size_pct: float = 10.0  # Max 10% of portfolio per position
    assignment_buffer_pct: float = 5.0   # Keep 5% cash buffer for assignments


@dataclass
class BollingerBands:
    """Bollinger Bands calculation result"""
    upper_band: float
    middle_band: float  # Simple moving average
    lower_band: float
    current_price: float
    squeeze_ratio: float  # (upper - lower) / middle


@dataclass
class WheelPosition:
    """Active Wheel strategy position"""

    underlying_symbol: str
    current_phase: WheelPhase

    # Option details
    option_symbol: Optional[str] = None
    option_type: Optional[OptionType] = None
    strike_price: Optional[float] = None
    expiration_date: Optional[datetime] = None

    # Position details
    shares_owned: int = 0
    option_contracts: int = 0  # Number of option contracts
    entry_price: Optional[float] = None
    entry_delta: Optional[float] = None
    entry_time: Optional[datetime] = None

    # Profit tracking
    total_premium_collected: float = 0.0
    unrealized_pnl: float = 0.0

    # Status
    assigned: bool = False
    last_roll_date: Optional[datetime] = None
    bollinger_bands: Optional[BollingerBands] = None


class WheelStrategy:
    """
    Wheel Options Strategy Implementation

    The Wheel strategy is a systematic approach to generating income through:
    1. Selling cash-secured puts to collect premium and potentially acquire shares
    2. If assigned, selling covered calls against the shares to generate additional income
    3. Using Bollinger Bands to optimize strike selection and timing
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        config: WheelConfig = None,
    ):
        self.alpaca_client = alpaca_client
        self.config = config or WheelConfig()
        self.active_positions: Dict[str, WheelPosition] = {}
        self.monitoring_task: Optional[asyncio.Task] = None

        logger.info("Wheel strategy initialized", config=self.config)

    async def analyze_symbol(
        self,
        symbol: str,
        market_data: Dict[str, Any] = None,
    ) -> Optional[StrategySignal]:
        """
        Analyze symbol for Wheel strategy opportunities

        Args:
            symbol: Underlying symbol to analyze
            market_data: Current market data context

        Returns:
            StrategySignal if opportunity found, None otherwise
        """
        try:
            # Check if we already have a position
            existing_position = self.active_positions.get(symbol)

            # Calculate Bollinger Bands
            bollinger_bands = await self._calculate_bollinger_bands(symbol)
            if not bollinger_bands:
                logger.debug("Could not calculate Bollinger Bands", symbol=symbol)
                return None

            # Get current underlying price
            underlying_price = await self.alpaca_client.get_latest_price(symbol)
            if not underlying_price:
                raise TradingError(f"Unable to get current price for {symbol}")

            # Determine strategy based on existing position
            if existing_position:
                return await self._analyze_existing_position(
                    existing_position, bollinger_bands, underlying_price, market_data
                )
            else:
                return await self._analyze_new_position(
                    symbol, bollinger_bands, underlying_price, market_data
                )

        except Exception as e:
            logger.error("Error analyzing symbol for Wheel", symbol=symbol, error=str(e))
            return None

    async def _analyze_new_position(
        self,
        symbol: str,
        bollinger_bands: BollingerBands,
        underlying_price: float,
        market_data: Dict[str, Any],
    ) -> Optional[StrategySignal]:
        """Analyze opportunities for new Wheel position (CSP phase)"""

        # Get options chain for CSPs
        option_chain = await self._get_filtered_option_chain(
            symbol,
            option_type="put",
            underlying_price=underlying_price
        )

        if not option_chain:
            return None

        # Find best CSP opportunity
        best_csp = await self._find_best_csp_opportunity(
            symbol, option_chain, bollinger_bands, underlying_price
        )

        if not best_csp:
            return None

        return await self._create_csp_signal(
            symbol, best_csp, bollinger_bands, underlying_price, market_data
        )

    async def _analyze_existing_position(
        self,
        position: WheelPosition,
        bollinger_bands: BollingerBands,
        underlying_price: float,
        market_data: Dict[str, Any],
    ) -> Optional[StrategySignal]:
        """Analyze existing Wheel position for adjustments or new opportunities"""

        # Update position with current Bollinger Bands
        position.bollinger_bands = bollinger_bands

        # Check if we need to roll the current option
        if position.option_symbol:
            should_roll, roll_reason = await self._should_roll_option(position, underlying_price)
            if should_roll:
                return await self._create_roll_signal(position, roll_reason, underlying_price, market_data)

        # If we're in CSP phase and got assigned, switch to CC phase
        if position.current_phase == WheelPhase.CASH_SECURED_PUT and position.assigned:
            return await self._create_cc_opportunity(position, bollinger_bands, underlying_price, market_data)

        # If we're in CC phase and shares got called away, switch to CSP phase
        if position.current_phase == WheelPhase.COVERED_CALL and position.shares_owned == 0:
            position.current_phase = WheelPhase.CASH_SECURED_PUT
            return await self._analyze_new_position(
                position.underlying_symbol, bollinger_bands, underlying_price, market_data
            )

        return None

    async def _calculate_bollinger_bands(self, symbol: str) -> Optional[BollingerBands]:
        """Calculate Bollinger Bands for the symbol"""
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.bb_period + 10)

            historical_data = await self.alpaca_client.get_historical_data(
                symbol=symbol,
                start=start_date,
                end=end_date,
                timeframe="1Day"
            )

            if not historical_data or len(historical_data) < self.config.bb_period:
                return None

            # Extract closing prices
            closes = [float(bar["close"]) for bar in historical_data[-self.config.bb_period:]]

            # Calculate simple moving average
            sma = sum(closes) / len(closes)

            # Calculate standard deviation
            variance = sum((price - sma) ** 2 for price in closes) / len(closes)
            std_dev = variance ** 0.5

            # Calculate bands
            upper_band = sma + (self.config.bb_std_dev * std_dev)
            lower_band = sma - (self.config.bb_std_dev * std_dev)

            current_price = closes[-1]
            squeeze_ratio = (upper_band - lower_band) / sma if sma > 0 else 0

            return BollingerBands(
                upper_band=upper_band,
                middle_band=sma,
                lower_band=lower_band,
                current_price=current_price,
                squeeze_ratio=squeeze_ratio
            )

        except Exception as e:
            logger.error("Error calculating Bollinger Bands", symbol=symbol, error=str(e))
            return None

    async def _get_filtered_option_chain(
        self,
        symbol: str,
        option_type: str,
        underlying_price: float,
    ) -> List[Dict[str, Any]]:
        """Get filtered options chain for Wheel strategy"""

        # Calculate expiration date range
        min_expiry = datetime.now() + timedelta(days=self.config.min_days_to_expiry)
        max_expiry = datetime.now() + timedelta(days=self.config.max_days_to_expiry)

        # Get options for each expiration in range
        all_options = []
        current_date = min_expiry

        while current_date <= max_expiry:
            try:
                options = await self.alpaca_client.get_option_chain(
                    symbol=symbol,
                    expiration_date=current_date.strftime("%Y-%m-%d"),
                    option_type=option_type,
                    limit=100
                )
                if options:
                    all_options.extend(options)
            except Exception as e:
                logger.warning("Error getting options for date",
                             symbol=symbol, date=current_date, error=str(e))

            current_date += timedelta(days=1)

        # Filter options
        filtered_options = []
        for option in all_options:
            if await self._passes_wheel_filters(option, underlying_price):
                filtered_options.append(option)

        return filtered_options

    async def _passes_wheel_filters(self, option_data: Dict[str, Any], underlying_price: float) -> bool:
        """Apply Wheel strategy filters to option"""

        # Open interest filter
        open_interest = int(option_data.get("open_interest", 0))
        if open_interest < self.config.min_open_interest:
            return False

        # Bid-ask spread filter
        bid = float(option_data.get("bid_price", 0))
        ask = float(option_data.get("ask_price", 0))
        if bid <= 0 or ask <= 0:
            return False

        spread_pct = ((ask - bid) / ((bid + ask) / 2)) * 100
        if spread_pct > self.config.max_bid_ask_spread_pct:
            return False

        # Strike price reasonableness (not too far OTM)
        strike_price = float(option_data.get("strike_price", 0))
        option_type = option_data.get("type", "").upper()

        if option_type == "PUT":
            # For puts, strike should be within reasonable range below current price
            if strike_price > underlying_price * 0.98:  # Not more than 2% OTM
                return False
        elif option_type == "CALL":
            # For calls, strike should be within reasonable range above current price
            if strike_price < underlying_price * 1.02:  # Not more than 2% OTM
                return False

        return True

    async def _find_best_csp_opportunity(
        self,
        symbol: str,
        option_chain: List[Dict[str, Any]],
        bollinger_bands: BollingerBands,
        underlying_price: float,
    ) -> Optional[Tuple[Dict[str, Any], float, GreeksData]]:
        """Find best cash-secured put opportunity"""

        best_option = None
        best_score = 0
        best_greeks = None

        for option_data in option_chain:
            try:
                # Convert to OptionContract for Greeks calculation
                option = await self._convert_to_option_contract(option_data, symbol)

                # Calculate Greeks
                greeks = await self._calculate_greeks(option, underlying_price)
                if not greeks:
                    continue

                # Check delta range for CSP
                if not (self.config.csp_delta_min <= greeks.delta <= self.config.csp_delta_max):
                    continue

                # Score this opportunity
                score = await self._score_csp_opportunity(option, greeks, bollinger_bands, underlying_price)

                if score > best_score:
                    best_score = score
                    best_option = option_data
                    best_greeks = greeks

            except Exception as e:
                logger.warning("Error processing CSP option", option=option_data, error=str(e))
                continue

        return (best_option, best_score, best_greeks) if best_option else None

    async def _score_csp_opportunity(
        self,
        option: OptionContract,
        greeks: GreeksData,
        bollinger_bands: BollingerBands,
        underlying_price: float,
    ) -> float:
        """Score cash-secured put opportunity"""
        score = 0.0

        # Premium yield (annualized)
        option_price = (option.bid + option.ask) / 2
        strike_price = option.strike_price
        days_to_expiry = (option.expiration_date - datetime.now()).days

        if days_to_expiry > 0 and strike_price > 0:
            premium_yield = (option_price / strike_price) * (365 / days_to_expiry) * 100
            score += premium_yield * 5  # Weight premium yield heavily

        # Prefer strikes near lower Bollinger Band (good entry point)
        bb_distance = abs(strike_price - bollinger_bands.lower_band) / bollinger_bands.lower_band
        score += max(0, 10 - bb_distance * 50)  # Closer to lower band = higher score

        # Higher theta (time decay) is better for selling
        if greeks.theta < 0:
            score += abs(greeks.theta) * 20

        # Lower gamma is better (less risk of rapid delta changes)
        if greeks.gamma > 0:
            score += max(0, 10 - greeks.gamma * 100)

        # Liquidity score
        spread_pct = (option.ask - option.bid) / option_price if option_price > 0 else 1.0
        score += max(0, 10 - spread_pct * 20)

        # Open interest bonus
        score += min(option.open_interest / 100, 5)

        return score

    async def _create_csp_signal(
        self,
        symbol: str,
        csp_data: Tuple[Dict[str, Any], float, GreeksData],
        bollinger_bands: BollingerBands,
        underlying_price: float,
        market_data: Dict[str, Any],
    ) -> StrategySignal:
        """Create strategy signal for cash-secured put"""

        option_data, score, greeks = csp_data

        # Calculate entry price and profit targets
        bid = float(option_data.get("bid_price", 0))
        ask = float(option_data.get("ask_price", 0))
        entry_price = bid  # Sell at bid price

        # Profit target: close at 50% of premium
        profit_target = entry_price * 0.5

        # Stop loss: roll if delta exceeds threshold or significant loss
        strike_price = float(option_data.get("strike_price", 0))
        stop_loss = entry_price * 2.0  # Conservative stop loss

        # Calculate confidence
        confidence = min(score / 100, 0.90)  # Cap at 90%

        # Build rationale
        rationale = (
            f"Wheel CSP on {symbol}. Strike ${strike_price:.2f} "
            f"({strike_price/underlying_price:.1%} of current price). "
            f"Delta: {greeks.delta:.3f}, Premium: ${entry_price:.2f}, "
            f"BB Position: {self._get_bb_position(strike_price, bollinger_bands)}"
        )

        return StrategySignal(
            strategy="Wheel-CSP",
            symbol=option_data["symbol"],
            direction="SELL",
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=profit_target,
            rationale=rationale,
            indicators_used=[
                "Delta", "Theta", "Bollinger Bands", "Premium Yield", "Open Interest"
            ],
            market_context={
                "underlying_symbol": symbol,
                "underlying_price": underlying_price,
                "wheel_phase": WheelPhase.CASH_SECURED_PUT.value,
                "strike_price": strike_price,
                "expiration_date": option_data.get("expiration_date"),
                "greeks": {
                    "delta": greeks.delta,
                    "gamma": greeks.gamma,
                    "theta": greeks.theta,
                    "vega": greeks.vega,
                },
                "bollinger_bands": {
                    "upper": bollinger_bands.upper_band,
                    "middle": bollinger_bands.middle_band,
                    "lower": bollinger_bands.lower_band,
                    "squeeze_ratio": bollinger_bands.squeeze_ratio,
                },
                "opportunity_score": score,
                "premium_yield_annualized": self._calculate_premium_yield(
                    entry_price, strike_price, option_data.get("expiration_date")
                ),
            }
        )

    async def _create_cc_opportunity(
        self,
        position: WheelPosition,
        bollinger_bands: BollingerBands,
        underlying_price: float,
        market_data: Dict[str, Any],
    ) -> Optional[StrategySignal]:
        """Create covered call opportunity for assigned shares"""

        if position.shares_owned <= 0:
            return None

        # Get options chain for calls
        option_chain = await self._get_filtered_option_chain(
            position.underlying_symbol,
            option_type="call",
            underlying_price=underlying_price
        )

        if not option_chain:
            return None

        # Find best CC opportunity (prefer strikes above upper Bollinger Band)
        best_cc = await self._find_best_cc_opportunity(
            position.underlying_symbol, option_chain, bollinger_bands, underlying_price
        )

        if not best_cc:
            return None

        return await self._create_cc_signal(
            position, best_cc, bollinger_bands, underlying_price, market_data
        )

    async def _find_best_cc_opportunity(
        self,
        symbol: str,
        option_chain: List[Dict[str, Any]],
        bollinger_bands: BollingerBands,
        underlying_price: float,
    ) -> Optional[Tuple[Dict[str, Any], float, GreeksData]]:
        """Find best covered call opportunity"""

        best_option = None
        best_score = 0
        best_greeks = None

        for option_data in option_chain:
            try:
                option = await self._convert_to_option_contract(option_data, symbol)
                greeks = await self._calculate_greeks(option, underlying_price)

                if not greeks:
                    continue

                # Check delta range for CC
                if not (self.config.cc_delta_min <= greeks.delta <= self.config.cc_delta_max):
                    continue

                # Prefer strikes above upper Bollinger Band
                strike_price = float(option_data.get("strike_price", 0))
                if strike_price <= bollinger_bands.upper_band:
                    continue

                # Score this opportunity
                score = await self._score_cc_opportunity(option, greeks, bollinger_bands, underlying_price)

                if score > best_score:
                    best_score = score
                    best_option = option_data
                    best_greeks = greeks

            except Exception as e:
                logger.warning("Error processing CC option", option=option_data, error=str(e))
                continue

        return (best_option, best_score, best_greeks) if best_option else None

    async def _score_cc_opportunity(
        self,
        option: OptionContract,
        greeks: GreeksData,
        bollinger_bands: BollingerBands,
        underlying_price: float,
    ) -> float:
        """Score covered call opportunity"""
        score = 0.0

        # Premium yield
        option_price = (option.bid + option.ask) / 2
        days_to_expiry = (option.expiration_date - datetime.now()).days

        if days_to_expiry > 0:
            premium_yield = (option_price / underlying_price) * (365 / days_to_expiry) * 100
            score += premium_yield * 5

        # Prefer strikes above upper Bollinger Band (good exit point)
        if option.strike_price > bollinger_bands.upper_band:
            bb_premium = (option.strike_price - bollinger_bands.upper_band) / bollinger_bands.upper_band * 100
            score += bb_premium * 2

        # Higher theta is better
        if greeks.theta < 0:
            score += abs(greeks.theta) * 20

        # Lower gamma is better
        if greeks.gamma > 0:
            score += max(0, 10 - greeks.gamma * 100)

        # Liquidity and open interest
        spread_pct = (option.ask - option.bid) / option_price if option_price > 0 else 1.0
        score += max(0, 10 - spread_pct * 20)
        score += min(option.open_interest / 100, 5)

        return score

    async def _create_cc_signal(
        self,
        position: WheelPosition,
        cc_data: Tuple[Dict[str, Any], float, GreeksData],
        bollinger_bands: BollingerBands,
        underlying_price: float,
        market_data: Dict[str, Any],
    ) -> StrategySignal:
        """Create strategy signal for covered call"""

        option_data, score, greeks = cc_data

        bid = float(option_data.get("bid_price", 0))
        entry_price = bid
        profit_target = entry_price * 0.5
        stop_loss = entry_price * 2.0
        strike_price = float(option_data.get("strike_price", 0))

        confidence = min(score / 100, 0.90)

        rationale = (
            f"Wheel CC on {position.underlying_symbol}. Strike ${strike_price:.2f} "
            f"({strike_price/underlying_price:.1%} above current). "
            f"Delta: {greeks.delta:.3f}, Premium: ${entry_price:.2f}, "
            f"Above BB Upper: ${bollinger_bands.upper_band:.2f}"
        )

        return StrategySignal(
            strategy="Wheel-CC",
            symbol=option_data["symbol"],
            direction="SELL",
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=profit_target,
            rationale=rationale,
            indicators_used=[
                "Delta", "Theta", "Bollinger Bands", "Premium Yield", "Open Interest"
            ],
            market_context={
                "underlying_symbol": position.underlying_symbol,
                "underlying_price": underlying_price,
                "wheel_phase": WheelPhase.COVERED_CALL.value,
                "shares_owned": position.shares_owned,
                "strike_price": strike_price,
                "expiration_date": option_data.get("expiration_date"),
                "greeks": {
                    "delta": greeks.delta,
                    "gamma": greeks.gamma,
                    "theta": greeks.theta,
                    "vega": greeks.vega,
                },
                "bollinger_bands": {
                    "upper": bollinger_bands.upper_band,
                    "middle": bollinger_bands.middle_band,
                    "lower": bollinger_bands.lower_band,
                },
                "total_premium_collected": position.total_premium_collected,
                "opportunity_score": score,
            }
        )

    # Helper methods
    async def _convert_to_option_contract(
        self, option_data: Dict[str, Any], underlying_symbol: str
    ) -> OptionContract:
        """Convert API option data to OptionContract"""
        symbol = option_data.get("symbol", "")
        strike_price = float(option_data.get("strike_price", 0))
        option_type_str = option_data.get("type", "call").upper()
        option_type = OptionType.CALL if option_type_str == "CALL" else OptionType.PUT

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
        """Calculate Greeks using Black-Scholes"""
        try:
            from app.trading.options_trading import get_options_trader

            options_trader = get_options_trader()
            time_to_expiry = (option.expiration_date - datetime.now()).total_seconds() / (365.25 * 24 * 3600)

            greeks = options_trader.bs_calculator.calculate_greeks(
                underlying_price=underlying_price,
                strike_price=option.strike_price,
                time_to_expiry=time_to_expiry,
                risk_free_rate=0.05,
                volatility=option.implied_volatility or 0.25,
                option_type=option.option_type,
            )

            return greeks

        except Exception as e:
            logger.warning("Error calculating Greeks", option=option.symbol, error=str(e))
            return None

    def _get_bb_position(self, price: float, bb: BollingerBands) -> str:
        """Get position relative to Bollinger Bands"""
        if price > bb.upper_band:
            return "Above Upper Band"
        elif price < bb.lower_band:
            return "Below Lower Band"
        elif price > bb.middle_band:
            return "Above Middle"
        else:
            return "Below Middle"

    def _calculate_premium_yield(self, premium: float, strike: float, expiry_str: str) -> float:
        """Calculate annualized premium yield"""
        try:
            expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
            days_to_expiry = (expiry - datetime.now()).days
            if days_to_expiry <= 0:
                return 0.0
            return (premium / strike) * (365 / days_to_expiry) * 100
        except:
            return 0.0

    async def _should_roll_option(self, position: WheelPosition, underlying_price: float) -> Tuple[bool, str]:
        """Check if current option should be rolled"""
        if not position.option_symbol or not position.entry_delta:
            return False, ""

        # Get current option quote
        try:
            quote = await self.alpaca_client.get_option_quote(position.option_symbol)
            if not quote:
                return False, "No quote available"

            current_price = (quote["bid_price"] + quote["ask_price"]) / 2

            # Check profit threshold
            if position.entry_price and current_price <= position.entry_price * (self.config.roll_profit_threshold / 100):
                return True, f"Profit target reached ({self.config.roll_profit_threshold}%)"

            # Check delta threshold (would need current Greeks calculation)
            # This is a simplified check - in practice, you'd recalculate current delta

            # Check time-based rolling (roll with 7 days or less to expiry)
            if position.expiration_date:
                days_to_expiry = (position.expiration_date - datetime.now()).days
                if days_to_expiry <= 7:
                    return True, "Time-based roll (7 days to expiry)"

            return False, ""

        except Exception as e:
            logger.warning("Error checking roll conditions", position=position.option_symbol, error=str(e))
            return False, "Error checking conditions"

    async def _create_roll_signal(
        self,
        position: WheelPosition,
        roll_reason: str,
        underlying_price: float,
        market_data: Dict[str, Any],
    ) -> StrategySignal:
        """Create signal to roll existing option"""

        # This would create a signal to close current option and open new one
        # Simplified implementation
        return StrategySignal(
            strategy=f"Wheel-Roll-{position.current_phase.value}",
            symbol=position.option_symbol,
            direction="BUY",  # Close existing position first
            confidence=0.85,
            entry_price=0.01,  # Placeholder for market order
            stop_loss=999.99,
            take_profit=0.01,
            rationale=f"Rolling {position.current_phase.value} option: {roll_reason}",
            indicators_used=["Delta", "Time to Expiry", "Profit Target"],
            market_context={
                "underlying_symbol": position.underlying_symbol,
                "underlying_price": underlying_price,
                "roll_reason": roll_reason,
                "current_phase": position.current_phase.value,
                "action": "ROLL",
            }
        )

    # Position management methods
    def add_position(
        self,
        symbol: str,
        phase: WheelPhase,
        option_symbol: str = None,
        shares_owned: int = 0,
        option_contracts: int = 0,
        entry_price: float = None,
        strike_price: float = None,
    ):
        """Add new Wheel position"""
        position = WheelPosition(
            underlying_symbol=symbol,
            current_phase=phase,
            option_symbol=option_symbol,
            shares_owned=shares_owned,
            option_contracts=option_contracts,
            entry_price=entry_price,
            strike_price=strike_price,
            entry_time=datetime.now(),
        )

        self.active_positions[symbol] = position
        logger.info("Added Wheel position", symbol=symbol, phase=phase.value)

    def get_position(self, symbol: str) -> Optional[WheelPosition]:
        """Get Wheel position for symbol"""
        return self.active_positions.get(symbol)

    def remove_position(self, symbol: str):
        """Remove Wheel position"""
        if symbol in self.active_positions:
            del self.active_positions[symbol]
            logger.info("Removed Wheel position", symbol=symbol)

    def get_all_positions(self) -> Dict[str, WheelPosition]:
        """Get all active Wheel positions"""
        return self.active_positions.copy()