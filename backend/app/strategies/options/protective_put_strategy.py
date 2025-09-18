"""
Protective Put Strategy Implementation

A protective put strategy involves owning shares of stock and buying put options
to protect against downside risk. This is a portfolio insurance strategy that
provides peace of mind during market volatility.

Strategy Characteristics:
- Long stock position + Long put option
- Downside protection at put strike price
- Unlimited upside potential (minus put premium)
- Insurance cost equals put premium paid
- Best during periods of uncertainty or high volatility
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from decimal import Decimal
import logging
from datetime import datetime, timedelta

from app.core.base_strategy import BaseStrategy, StrategySignal, StrategyConfig
from app.core.market_data import MarketDataService
from app.strategies.options.black_scholes import BlackScholesCalculator, GreeksData

logger = logging.getLogger(__name__)


@dataclass
class ProtectivePutConfig(StrategyConfig):
    """Configuration for Protective Put Strategy"""

    # Stock position requirements
    min_stock_position: int = 100  # Minimum shares (1 contract = 100 shares)
    max_position_size: Decimal = Decimal("50000")  # Maximum position value

    # Put option selection criteria
    min_dte: int = 30  # Minimum days to expiration
    max_dte: int = 90  # Maximum days to expiration
    target_delta: Decimal = Decimal("0.20")  # Target put delta (20 delta puts)
    delta_tolerance: Decimal = Decimal("0.10")  # Delta selection tolerance

    # Protection levels
    min_protection_level: Decimal = Decimal("0.90")  # Minimum 90% protection
    max_protection_level: Decimal = Decimal("0.95")  # Maximum 95% protection
    max_insurance_cost: Decimal = Decimal("0.03")  # Maximum 3% of position value

    # Market volatility triggers
    min_iv_rank: Decimal = Decimal("20")  # Minimum IV rank for cost efficiency
    max_iv_rank: Decimal = Decimal("80")  # Maximum IV rank (too expensive)
    volatility_spike_threshold: Decimal = Decimal("1.5")  # VIX spike multiplier

    # Portfolio risk triggers
    max_portfolio_beta: Decimal = Decimal("1.2")  # Maximum portfolio beta
    max_sector_concentration: Decimal = Decimal("0.30")  # Maximum 30% in one sector

    # Risk management
    profit_target: Decimal = Decimal("0.50")  # Close protection at 50% decay
    roll_threshold: int = 21  # Roll with 21 days or less to expiration
    max_loss_on_protection: Decimal = Decimal("0.75")  # Maximum 75% loss on puts

    # Market conditions
    min_liquidity: int = 50  # Minimum daily volume
    max_bid_ask_spread: Decimal = Decimal("0.15")  # Maximum 15% bid-ask spread


@dataclass
class ProtectivePutPosition:
    """Represents a protective put position"""

    # Stock position
    stock_symbol: str
    stock_shares: int
    stock_avg_cost: Decimal

    # Put option details
    put_symbol: str
    put_strike: Decimal
    put_expiration: datetime
    put_contracts: int
    put_premium_paid: Decimal

    # Position metrics
    entry_date: datetime
    current_stock_price: Decimal
    current_put_price: Decimal
    protection_level: Decimal  # Put strike / Stock cost basis
    insurance_cost_pct: Decimal  # Premium / Stock value

    # Greeks and risk metrics
    position_delta: Decimal
    position_theta: Decimal
    position_gamma: Decimal
    position_vega: Decimal

    # Protection effectiveness
    days_to_expiration: int
    intrinsic_value: Decimal
    time_value: Decimal
    moneyness: Decimal  # Put strike / Current stock price

    def calculate_protected_value(self) -> Decimal:
        """Calculate current protected portfolio value"""
        stock_value = self.current_stock_price * self.stock_shares
        put_value = self.current_put_price * self.put_contracts * 100
        return stock_value + put_value

    def calculate_floor_value(self) -> Decimal:
        """Calculate guaranteed floor value if stock goes to zero"""
        put_intrinsic = max(Decimal("0"), self.put_strike - Decimal("0")) * self.put_contracts * 100
        return put_intrinsic

    def calculate_breakeven_price(self) -> Decimal:
        """Calculate breakeven stock price including put premium"""
        premium_per_share = self.put_premium_paid / self.stock_shares
        return self.stock_avg_cost + premium_per_share

    def calculate_max_loss(self) -> Decimal:
        """Calculate maximum possible loss with protection in place"""
        # Maximum loss occurs if stock falls to put strike
        stock_loss = max(Decimal("0"), self.stock_avg_cost - self.put_strike) * self.stock_shares
        insurance_cost = self.put_premium_paid * self.put_contracts
        return stock_loss + insurance_cost

    def calculate_insurance_effectiveness(self) -> Decimal:
        """Calculate how effectively the insurance is working (0-1 scale)"""
        if self.current_stock_price >= self.stock_avg_cost:
            # Stock is profitable, insurance effectiveness is based on decay rate
            time_decay_rate = (self.put_premium_paid - self.current_put_price) / self.put_premium_paid
            return max(Decimal("0"), 1 - time_decay_rate)
        else:
            # Stock is losing money, effectiveness is based on protection provided
            stock_loss = self.stock_avg_cost - self.current_stock_price
            put_protection = self.current_put_price - (self.put_premium_paid / self.put_contracts)
            protection_ratio = put_protection / stock_loss if stock_loss > 0 else Decimal("0")
            return min(Decimal("1"), protection_ratio)


class ProtectivePutStrategy(BaseStrategy):
    """
    Protective Put Strategy Implementation

    This strategy provides downside protection for stock positions by purchasing
    put options. It's designed for investors who want to maintain upside exposure
    while limiting downside risk during uncertain market conditions.
    """

    def __init__(self, config: ProtectivePutConfig, market_data: MarketDataService):
        self.config = config
        self.market_data = market_data
        self.bs_calculator = BlackScholesCalculator()
        self.positions: List[ProtectivePutPosition] = []

        logger.info(f"Initialized Protective Put Strategy with config: {config}")

    def generate_signals(self, symbol: str, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """Generate protective put strategy signals"""
        signals = []

        try:
            # Check if we have a stock position that needs protection
            stock_position = self._get_stock_position(symbol)

            if stock_position and stock_position >= self.config.min_stock_position:
                # Check if position already has protection
                existing_protection = self._has_existing_protection(symbol)

                if not existing_protection:
                    # Evaluate need for new protection
                    protection_signal = self._evaluate_protection_need(symbol, market_data, stock_position)
                    if protection_signal:
                        signals.append(protection_signal)
                else:
                    # Manage existing protection
                    management_signals = self._evaluate_protection_management(symbol, market_data)
                    signals.extend(management_signals)

            # Check existing positions for management signals
            position_signals = self._evaluate_position_management(symbol, market_data)
            signals.extend(position_signals)

        except Exception as e:
            logger.error(f"Error generating protective put signals for {symbol}: {e}")

        return signals

    def _evaluate_protection_need(self, symbol: str, market_data: Dict[str, Any],
                                 stock_shares: int) -> Optional[StrategySignal]:
        """Evaluate if stock position needs protective put coverage"""

        current_price = Decimal(str(market_data.get('current_price', 0)))
        iv_rank = Decimal(str(market_data.get('iv_rank', 0)))

        # Check market volatility conditions
        vix = Decimal(str(market_data.get('vix', 20)))
        vix_historical_avg = Decimal(str(market_data.get('vix_avg', 20)))
        volatility_spike = vix / vix_historical_avg if vix_historical_avg > 0 else Decimal("1")

        # Check portfolio risk factors
        portfolio_beta = self._calculate_portfolio_beta(symbol, market_data)
        sector_concentration = self._calculate_sector_concentration(symbol, market_data)

        # Determine if protection is warranted
        protection_triggers = 0
        trigger_reasons = []

        # High volatility environment
        if volatility_spike >= self.config.volatility_spike_threshold:
            protection_triggers += 2
            trigger_reasons.append("high_volatility")

        # High IV rank (insurance is relatively expensive but market stress is high)
        if iv_rank >= 60:
            protection_triggers += 1
            trigger_reasons.append("elevated_iv")

        # High portfolio beta (market sensitive position)
        if portfolio_beta >= self.config.max_portfolio_beta:
            protection_triggers += 1
            trigger_reasons.append("high_beta")

        # High sector concentration
        if sector_concentration >= self.config.max_sector_concentration:
            protection_triggers += 1
            trigger_reasons.append("sector_concentration")

        # Technical deterioration
        rsi = Decimal(str(market_data.get('rsi', 50)))
        if rsi <= 30:  # Oversold conditions
            protection_triggers += 1
            trigger_reasons.append("technical_weakness")

        # Require at least 2 triggers for protection signal
        if protection_triggers < 2:
            return None

        # Check IV rank bounds (don't buy insurance when it's too expensive)
        if not (self.config.min_iv_rank <= iv_rank <= self.config.max_iv_rank):
            return None

        # Find optimal put option for protection
        put_option = self._find_optimal_protection_put(symbol, current_price, market_data)
        if not put_option:
            return None

        # Calculate insurance cost
        position_value = current_price * stock_shares
        insurance_cost = put_option['premium'] * (stock_shares // 100)
        insurance_cost_pct = insurance_cost / position_value

        # Check cost threshold
        if insurance_cost_pct > self.config.max_insurance_cost:
            return None

        # Calculate protection metrics
        protection_level = put_option['strike'] / current_price

        return StrategySignal(
            strategy_name="protective_put",
            symbol=symbol,
            signal_type="BUY_TO_OPEN",
            strength=min(95, 60 + (protection_triggers * 10)),
            entry_price=put_option['premium'],
            target_price=put_option['premium'] * (1 - self.config.profit_target),
            stop_price=put_option['premium'] * (1 + self.config.max_loss_on_protection),
            quantity=stock_shares // 100,  # Number of contracts
            metadata={
                "strategy_type": "portfolio_insurance",
                "option_type": "put",
                "strike_price": float(put_option['strike']),
                "expiration_date": put_option['expiration'].isoformat(),
                "protection_level": float(protection_level),
                "insurance_cost_pct": float(insurance_cost_pct),
                "iv_rank": float(iv_rank),
                "volatility_spike": float(volatility_spike),
                "protection_triggers": protection_triggers,
                "trigger_reasons": trigger_reasons,
                "stock_shares": stock_shares,
                "max_loss": float(self._calculate_max_loss(
                    current_price, put_option['strike'], put_option['premium'], stock_shares
                )),
                "floor_value": float(put_option['strike'] * (stock_shares // 100) * 100)
            }
        )

    def _evaluate_protection_management(self, symbol: str, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """Evaluate management of existing protective put positions"""
        signals = []

        for position in self.positions:
            if position.stock_symbol != symbol:
                continue

            current_stock_price = Decimal(str(market_data.get('current_price', 0)))
            current_put_price = self._get_option_price(position.put_symbol, market_data)

            # Update position metrics
            position.current_stock_price = current_stock_price
            position.current_put_price = current_put_price
            position.days_to_expiration = (position.put_expiration - datetime.now()).days
            position.moneyness = position.put_strike / current_stock_price

            # Check for rolling opportunity (near expiration)
            if position.days_to_expiration <= self.config.roll_threshold:
                roll_signal = self._evaluate_roll_opportunity(position, market_data)
                if roll_signal:
                    signals.append(roll_signal)

            # Check for profit taking (put has decayed significantly)
            put_decay = (position.put_premium_paid - current_put_price) / position.put_premium_paid
            if put_decay >= self.config.profit_target:
                # Stock has been stable, insurance has served its purpose
                signals.append(self._create_close_signal(position, "profit_target"))

            # Check for protection adjustment (stock has fallen significantly)
            elif position.moneyness >= 1.05:  # Put is 5% ITM
                # Consider taking profit on protection and re-establishing at higher strike
                signals.append(self._create_adjustment_signal(position, market_data))

            # Check for maximum loss threshold
            elif current_put_price <= position.put_premium_paid * (1 - self.config.max_loss_on_protection):
                # Protection has become too cheap, evaluate if still needed
                if not self._still_needs_protection(position, market_data):
                    signals.append(self._create_close_signal(position, "cost_ineffective"))

        return signals

    def _evaluate_position_management(self, symbol: str, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """Evaluate overall position management across all protective put positions"""
        signals = []

        # This method would implement portfolio-level management
        # For now, return empty list
        return signals

    def _find_optimal_protection_put(self, symbol: str, stock_price: Decimal,
                                   market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the optimal put option for portfolio protection"""

        # Get option chain
        option_chain = market_data.get('option_chain', {})
        puts = option_chain.get('puts', [])

        best_option = None
        best_score = 0

        for put in puts:
            # Filter by DTE
            expiration = datetime.fromisoformat(put['expiration'])
            dte = (expiration - datetime.now()).days

            if not (self.config.min_dte <= dte <= self.config.max_dte):
                continue

            # Calculate protection level
            strike = Decimal(str(put['strike']))
            protection_level = strike / stock_price

            if not (self.config.min_protection_level <= protection_level <= self.config.max_protection_level):
                continue

            # Filter by delta
            delta = abs(Decimal(str(put.get('delta', 0))))
            if abs(delta - self.config.target_delta) > self.config.delta_tolerance:
                continue

            # Check liquidity
            volume = put.get('volume', 0)
            if volume < self.config.min_liquidity:
                continue

            # Check bid-ask spread
            bid = Decimal(str(put.get('bid', 0)))
            ask = Decimal(str(put.get('ask', 0)))
            if ask > 0:
                spread_pct = (ask - bid) / ask
                if spread_pct > self.config.max_bid_ask_spread:
                    continue

            # Calculate premium and insurance cost
            premium = (bid + ask) / 2
            insurance_cost_pct = premium / stock_price

            # Score the option (lower cost and better protection is better)
            cost_score = max(0, 50 - (insurance_cost_pct * 2000))  # Penalty for expensive insurance
            protection_score = protection_level * 30  # Reward for better protection
            delta_score = 20 - abs(delta - self.config.target_delta) * 400  # Closer to target = higher score
            liquidity_score = min(10, volume / 20)  # Bonus for liquidity

            total_score = cost_score + protection_score + delta_score + liquidity_score

            if total_score > best_score:
                best_score = total_score
                best_option = {
                    'symbol': put['symbol'],
                    'strike': strike,
                    'expiration': expiration,
                    'premium': premium,
                    'delta': delta,
                    'volume': volume,
                    'dte': dte,
                    'protection_level': protection_level,
                    'insurance_cost_pct': insurance_cost_pct
                }

        return best_option

    def _calculate_portfolio_beta(self, symbol: str, market_data: Dict[str, Any]) -> Decimal:
        """Calculate portfolio beta for risk assessment"""
        # This would integrate with portfolio management system
        # For now, return a moderate beta
        return Decimal(str(market_data.get('beta', 1.0)))

    def _calculate_sector_concentration(self, symbol: str, market_data: Dict[str, Any]) -> Decimal:
        """Calculate sector concentration risk"""
        # This would integrate with portfolio management system
        # For now, return moderate concentration
        return Decimal("0.25")  # 25% concentration

    def _calculate_max_loss(self, stock_price: Decimal, put_strike: Decimal,
                          put_premium: Decimal, shares: int) -> Decimal:
        """Calculate maximum possible loss with protection"""
        # Stock loss to put strike level
        stock_loss = max(Decimal("0"), stock_price - put_strike) * shares

        # Insurance premium cost
        insurance_cost = put_premium * (shares // 100)

        return stock_loss + insurance_cost

    def _get_stock_position(self, symbol: str) -> int:
        """Get current stock position size"""
        # This would integrate with the portfolio management system
        # For now, return 0 to indicate no position
        return 0

    def _has_existing_protection(self, symbol: str) -> bool:
        """Check if symbol already has protective put coverage"""
        return any(pos.stock_symbol == symbol for pos in self.positions)

    def _get_option_price(self, option_symbol: str, market_data: Dict[str, Any]) -> Decimal:
        """Get current option price from market data"""
        # This would integrate with the options data feed
        # For now, return a placeholder
        return Decimal("0")

    def _create_close_signal(self, position: ProtectivePutPosition, reason: str) -> StrategySignal:
        """Create a signal to close protective put position"""

        return StrategySignal(
            strategy_name="protective_put",
            symbol=position.stock_symbol,
            signal_type="SELL_TO_CLOSE",
            strength=75,
            entry_price=position.current_put_price,
            target_price=Decimal("0"),
            stop_price=position.current_put_price * Decimal("0.80"),
            quantity=position.put_contracts,
            metadata={
                "action": "close_protection",
                "reason": reason,
                "option_symbol": position.put_symbol,
                "days_to_expiration": position.days_to_expiration,
                "protection_effectiveness": float(position.calculate_insurance_effectiveness()),
                "current_protection_level": float(position.protection_level),
                "total_insurance_cost": float(position.put_premium_paid * position.put_contracts)
            }
        )

    def _create_adjustment_signal(self, position: ProtectivePutPosition,
                                market_data: Dict[str, Any]) -> StrategySignal:
        """Create a signal to adjust protection (roll to new strike/expiration)"""

        return StrategySignal(
            strategy_name="protective_put",
            symbol=position.stock_symbol,
            signal_type="ROLL_PROTECTION",
            strength=85,
            entry_price=position.current_put_price,
            target_price=position.current_stock_price * Decimal("0.95"),  # New strike target
            stop_price=position.current_put_price * Decimal("1.20"),
            quantity=position.put_contracts,
            metadata={
                "action": "adjust_protection",
                "current_strike": float(position.put_strike),
                "current_moneyness": float(position.moneyness),
                "suggested_new_strike": float(position.current_stock_price * Decimal("0.95")),
                "reason": "in_the_money_adjustment"
            }
        )

    def _evaluate_roll_opportunity(self, position: ProtectivePutPosition,
                                 market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Evaluate opportunity to roll protection to later expiration"""

        # This would involve finding a suitable put option to roll to
        # For now, return None to indicate no rolling opportunity
        return None

    def _still_needs_protection(self, position: ProtectivePutPosition,
                              market_data: Dict[str, Any]) -> bool:
        """Determine if position still needs protection based on market conditions"""

        # Check current market volatility
        vix = Decimal(str(market_data.get('vix', 20)))
        if vix >= 25:  # High volatility environment
            return True

        # Check technical indicators
        rsi = Decimal(str(market_data.get('rsi', 50)))
        if rsi <= 35:  # Oversold conditions
            return True

        # Check earnings or other catalysts coming up
        days_to_earnings = market_data.get('days_to_earnings', 999)
        if days_to_earnings <= 30:  # Earnings within 30 days
            return True

        return False

    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate overall portfolio metrics for protective put positions"""

        if not self.positions:
            return {"total_positions": 0, "total_insurance_cost": 0, "average_protection": 0}

        total_insurance_cost = sum(pos.put_premium_paid * pos.put_contracts for pos in self.positions)
        total_stock_value = sum(pos.current_stock_price * pos.stock_shares for pos in self.positions)
        total_protected_value = sum(pos.calculate_protected_value() for pos in self.positions)

        average_protection = sum(pos.protection_level for pos in self.positions) / len(self.positions)
        average_effectiveness = sum(pos.calculate_insurance_effectiveness() for pos in self.positions) / len(self.positions)

        return {
            "total_positions": len(self.positions),
            "total_insurance_cost": float(total_insurance_cost),
            "total_stock_value": float(total_stock_value),
            "total_protected_value": float(total_protected_value),
            "average_protection_level": float(average_protection),
            "average_effectiveness": float(average_effectiveness),
            "positions_itm": len([p for p in self.positions if p.moneyness > 1.0]),
            "positions_near_expiry": len([p for p in self.positions if p.days_to_expiration <= 21]),
            "total_floor_value": float(sum(pos.calculate_floor_value() for pos in self.positions))
        }