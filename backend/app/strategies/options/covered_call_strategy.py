"""
Covered Call Strategy Implementation

A covered call strategy involves owning shares of stock and selling call options
against those shares to generate income. This is a conservative income strategy
that works best in neutral to slightly bullish markets.

Strategy Characteristics:
- Long stock position + Short call option
- Income generation from option premium
- Limited upside potential (capped at strike price)
- Downside protection equal to premium received
- Best in low to moderate volatility environments
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
class CoveredCallConfig(StrategyConfig):
    """Configuration for Covered Call Strategy"""

    # Stock position requirements
    min_stock_position: int = 100  # Minimum shares (1 contract = 100 shares)
    max_position_size: Decimal = Decimal("10000")  # Maximum position value

    # Call option selection criteria
    min_dte: int = 30  # Minimum days to expiration
    max_dte: int = 45  # Maximum days to expiration
    target_delta: Decimal = Decimal("0.30")  # Target call delta (30 delta calls)
    delta_tolerance: Decimal = Decimal("0.05")  # Delta selection tolerance

    # Income and volatility criteria
    min_premium_yield: Decimal = Decimal("0.02")  # Minimum 2% monthly yield
    max_iv_rank: Decimal = Decimal("70")  # Maximum IV rank for entry
    min_iv_rank: Decimal = Decimal("30")  # Minimum IV rank for entry

    # Risk management
    profit_target: Decimal = Decimal("0.50")  # Close at 50% profit
    stop_loss: Decimal = Decimal("2.00")  # Stop if option doubles in value
    roll_threshold: Decimal = Decimal("0.85")  # Roll when stock approaches 85% of strike

    # Market conditions
    min_liquidity: int = 100  # Minimum daily volume
    max_bid_ask_spread: Decimal = Decimal("0.10")  # Maximum 10% bid-ask spread


@dataclass
class CoveredCallPosition:
    """Represents a covered call position"""

    # Stock position
    stock_symbol: str
    stock_shares: int
    stock_avg_cost: Decimal

    # Call option details
    call_symbol: str
    call_strike: Decimal
    call_expiration: datetime
    call_contracts: int
    call_premium_received: Decimal

    # Position metrics
    entry_date: datetime
    current_stock_price: Decimal
    current_call_price: Decimal
    total_premium_received: Decimal
    unrealized_pnl: Decimal

    # Greeks and risk metrics
    position_delta: Decimal
    position_theta: Decimal
    position_gamma: Decimal
    position_vega: Decimal

    # Assignment risk
    assignment_probability: Decimal
    days_to_expiration: int
    moneyness: Decimal  # Stock price / Strike price

    def calculate_return_if_assigned(self) -> Decimal:
        """Calculate total return if call is assigned"""
        # Capital gain from stock assignment
        stock_gain = (self.call_strike - self.stock_avg_cost) * self.stock_shares

        # Premium income
        premium_income = self.total_premium_received

        # Total return
        total_return = stock_gain + premium_income
        total_investment = self.stock_avg_cost * self.stock_shares

        return total_return / total_investment if total_investment > 0 else Decimal("0")

    def calculate_current_yield(self) -> Decimal:
        """Calculate current annualized yield"""
        days_held = max(1, (datetime.now() - self.entry_date).days)
        current_return = self.unrealized_pnl / (self.stock_avg_cost * self.stock_shares)

        # Annualize the return
        annualized_return = current_return * (365 / days_held)
        return annualized_return


class CoveredCallStrategy(BaseStrategy):
    """
    Covered Call Strategy Implementation

    This strategy generates income by selling call options against owned stock positions.
    It's designed for conservative investors seeking regular income from their stock holdings.
    """

    def __init__(self, config: CoveredCallConfig, market_data: MarketDataService):
        self.config = config
        self.market_data = market_data
        self.bs_calculator = BlackScholesCalculator()
        self.positions: List[CoveredCallPosition] = []

        logger.info(f"Initialized Covered Call Strategy with config: {config}")

    def generate_signals(self, symbol: str, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """Generate covered call strategy signals"""
        signals = []

        try:
            # Check if we already have a stock position
            stock_position = self._get_stock_position(symbol)

            if stock_position and stock_position >= self.config.min_stock_position:
                # We own the stock, check for covered call opportunities
                call_signal = self._evaluate_call_writing_opportunity(symbol, market_data, stock_position)
                if call_signal:
                    signals.append(call_signal)
            else:
                # No stock position, check for stock purchase + call writing opportunity
                combo_signal = self._evaluate_covered_call_initiation(symbol, market_data)
                if combo_signal:
                    signals.append(combo_signal)

            # Check existing positions for management signals
            management_signals = self._evaluate_position_management(symbol, market_data)
            signals.extend(management_signals)

        except Exception as e:
            logger.error(f"Error generating covered call signals for {symbol}: {e}")

        return signals

    def _evaluate_call_writing_opportunity(self, symbol: str, market_data: Dict[str, Any],
                                         stock_shares: int) -> Optional[StrategySignal]:
        """Evaluate opportunity to write calls against existing stock position"""

        current_price = Decimal(str(market_data.get('current_price', 0)))
        iv_rank = Decimal(str(market_data.get('iv_rank', 0)))

        # Check IV rank conditions
        if not (self.config.min_iv_rank <= iv_rank <= self.config.max_iv_rank):
            return None

        # Find optimal call option to sell
        call_option = self._find_optimal_call_option(symbol, current_price, market_data)
        if not call_option:
            return None

        # Calculate position metrics
        premium_yield = self._calculate_monthly_yield(call_option['premium'], current_price)

        if premium_yield < self.config.min_premium_yield:
            return None

        # Calculate Greeks for the short call position
        greeks = self._calculate_option_greeks(call_option, current_price, market_data)

        return StrategySignal(
            strategy_name="covered_call",
            symbol=symbol,
            signal_type="SELL_TO_OPEN",
            strength=min(95, 50 + int(premium_yield * 1000)),  # Higher yield = stronger signal
            entry_price=call_option['premium'],
            target_price=call_option['premium'] * (1 - self.config.profit_target),
            stop_price=call_option['premium'] * (1 + self.config.stop_loss),
            quantity=stock_shares // 100,  # Number of contracts
            metadata={
                "strategy_type": "income_generation",
                "option_type": "call",
                "strike_price": call_option['strike'],
                "expiration_date": call_option['expiration'].isoformat(),
                "premium_yield": float(premium_yield),
                "iv_rank": float(iv_rank),
                "delta": float(greeks.delta),
                "theta": float(greeks.theta),
                "assignment_risk": "moderate" if call_option['strike'] < current_price * Decimal("1.05") else "low",
                "stock_shares": stock_shares,
                "max_profit": float(call_option['premium'] * (stock_shares // 100)),
                "return_if_assigned": float(self._calculate_return_if_assigned(
                    current_price, call_option['strike'], call_option['premium'], stock_shares
                ))
            }
        )

    def _evaluate_covered_call_initiation(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Evaluate opportunity to initiate new covered call position (buy stock + sell call)"""

        current_price = Decimal(str(market_data.get('current_price', 0)))

        # Check if position size is within limits
        position_value = current_price * self.config.min_stock_position
        if position_value > self.config.max_position_size:
            return None

        # Check market conditions
        iv_rank = Decimal(str(market_data.get('iv_rank', 0)))
        if not (self.config.min_iv_rank <= iv_rank <= self.config.max_iv_rank):
            return None

        # Find optimal call option to sell
        call_option = self._find_optimal_call_option(symbol, current_price, market_data)
        if not call_option:
            return None

        # Calculate net investment and returns
        stock_cost = current_price * self.config.min_stock_position
        call_premium = call_option['premium'] * (self.config.min_stock_position // 100)
        net_investment = stock_cost - call_premium

        # Calculate premium yield on net investment
        monthly_yield = call_premium / net_investment / (call_option['dte'] / 30)

        if monthly_yield < self.config.min_premium_yield:
            return None

        return StrategySignal(
            strategy_name="covered_call",
            symbol=symbol,
            signal_type="BUY_STOCK_SELL_CALL",
            strength=min(95, 40 + int(monthly_yield * 1000)),
            entry_price=current_price,
            target_price=call_option['strike'],  # Target if assigned
            stop_price=current_price * Decimal("0.90"),  # 10% stop on stock
            quantity=self.config.min_stock_position,
            metadata={
                "strategy_type": "covered_call_initiation",
                "stock_price": float(current_price),
                "call_strike": float(call_option['strike']),
                "call_premium": float(call_option['premium']),
                "net_investment": float(net_investment),
                "monthly_yield": float(monthly_yield),
                "iv_rank": float(iv_rank),
                "expiration_date": call_option['expiration'].isoformat(),
                "max_profit_if_assigned": float(
                    (call_option['strike'] - current_price) * self.config.min_stock_position + call_premium
                )
            }
        )

    def _evaluate_position_management(self, symbol: str, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """Evaluate management of existing covered call positions"""
        signals = []

        for position in self.positions:
            if position.stock_symbol != symbol:
                continue

            current_stock_price = Decimal(str(market_data.get('current_price', 0)))
            current_call_price = self._get_option_price(position.call_symbol, market_data)

            # Update position metrics
            position.current_stock_price = current_stock_price
            position.current_call_price = current_call_price
            position.days_to_expiration = (position.call_expiration - datetime.now()).days
            position.moneyness = current_stock_price / position.call_strike

            # Check for profit target
            profit_pct = (position.call_premium_received - current_call_price) / position.call_premium_received
            if profit_pct >= self.config.profit_target:
                signals.append(self._create_close_signal(position, "profit_target"))

            # Check for stop loss
            elif current_call_price >= position.call_premium_received * (1 + self.config.stop_loss):
                signals.append(self._create_close_signal(position, "stop_loss"))

            # Check for rolling opportunity
            elif position.moneyness >= self.config.roll_threshold:
                roll_signal = self._evaluate_roll_opportunity(position, market_data)
                if roll_signal:
                    signals.append(roll_signal)

            # Check for early assignment risk (ITM with < 7 days to expiration)
            elif position.moneyness > 1.0 and position.days_to_expiration <= 7:
                signals.append(self._create_close_signal(position, "assignment_risk"))

        return signals

    def _find_optimal_call_option(self, symbol: str, stock_price: Decimal,
                                market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the optimal call option to sell based on strategy criteria"""

        # Get option chain
        option_chain = market_data.get('option_chain', {})
        calls = option_chain.get('calls', [])

        best_option = None
        best_score = 0

        for call in calls:
            # Filter by DTE
            expiration = datetime.fromisoformat(call['expiration'])
            dte = (expiration - datetime.now()).days

            if not (self.config.min_dte <= dte <= self.config.max_dte):
                continue

            # Filter by delta
            delta = abs(Decimal(str(call.get('delta', 0))))
            if abs(delta - self.config.target_delta) > self.config.delta_tolerance:
                continue

            # Check liquidity
            volume = call.get('volume', 0)
            if volume < self.config.min_liquidity:
                continue

            # Check bid-ask spread
            bid = Decimal(str(call.get('bid', 0)))
            ask = Decimal(str(call.get('ask', 0)))
            if ask > 0:
                spread_pct = (ask - bid) / ask
                if spread_pct > self.config.max_bid_ask_spread:
                    continue

            # Calculate premium yield
            premium = (bid + ask) / 2
            monthly_yield = self._calculate_monthly_yield(premium, stock_price)

            # Score the option (higher yield and closer to target delta is better)
            yield_score = min(50, monthly_yield * 1000)  # Normalize yield to 0-50
            delta_score = 50 - abs(delta - self.config.target_delta) * 1000  # Closer to target = higher score
            liquidity_score = min(20, volume / 100)  # Bonus for liquidity

            total_score = yield_score + delta_score + liquidity_score

            if total_score > best_score:
                best_score = total_score
                best_option = {
                    'symbol': call['symbol'],
                    'strike': Decimal(str(call['strike'])),
                    'expiration': expiration,
                    'premium': premium,
                    'delta': delta,
                    'volume': volume,
                    'dte': dte,
                    'monthly_yield': monthly_yield
                }

        return best_option

    def _calculate_monthly_yield(self, premium: Decimal, stock_price: Decimal) -> Decimal:
        """Calculate monthly yield from option premium"""
        if stock_price == 0:
            return Decimal("0")

        return (premium / stock_price) * 12  # Approximate monthly yield

    def _calculate_option_greeks(self, option: Dict[str, Any], stock_price: Decimal,
                               market_data: Dict[str, Any]) -> GreeksData:
        """Calculate option Greeks using Black-Scholes"""

        strike = option['strike']
        time_to_expiry = option['dte'] / 365.0
        risk_free_rate = Decimal(str(market_data.get('risk_free_rate', 0.05)))
        volatility = Decimal(str(market_data.get('implied_volatility', 0.20)))

        return self.bs_calculator.calculate_greeks(
            stock_price=stock_price,
            strike_price=strike,
            time_to_expiry=Decimal(str(time_to_expiry)),
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type='call'
        )

    def _calculate_return_if_assigned(self, current_price: Decimal, strike: Decimal,
                                    premium: Decimal, shares: int) -> Decimal:
        """Calculate total return if call option is assigned"""

        # Capital appreciation (if any)
        capital_gain = max(Decimal("0"), strike - current_price) * shares

        # Premium income
        premium_income = premium * (shares // 100)

        # Total return as percentage of current investment
        total_investment = current_price * shares
        return (capital_gain + premium_income) / total_investment if total_investment > 0 else Decimal("0")

    def _get_stock_position(self, symbol: str) -> int:
        """Get current stock position size"""
        # This would integrate with the portfolio management system
        # For now, return 0 to indicate no position
        return 0

    def _get_option_price(self, option_symbol: str, market_data: Dict[str, Any]) -> Decimal:
        """Get current option price from market data"""
        # This would integrate with the options data feed
        # For now, return a placeholder
        return Decimal("0")

    def _create_close_signal(self, position: CoveredCallPosition, reason: str) -> StrategySignal:
        """Create a signal to close an existing covered call position"""

        return StrategySignal(
            strategy_name="covered_call",
            symbol=position.stock_symbol,
            signal_type="BUY_TO_CLOSE",
            strength=85,  # High priority for risk management
            entry_price=position.current_call_price,
            target_price=Decimal("0"),  # Close completely
            stop_price=position.current_call_price * Decimal("1.20"),  # Emergency stop
            quantity=position.call_contracts,
            metadata={
                "action": "close_position",
                "reason": reason,
                "option_symbol": position.call_symbol,
                "days_to_expiration": position.days_to_expiration,
                "current_yield": float(position.calculate_current_yield()),
                "moneyness": float(position.moneyness),
                "unrealized_pnl": float(position.unrealized_pnl)
            }
        )

    def _evaluate_roll_opportunity(self, position: CoveredCallPosition,
                                 market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Evaluate opportunity to roll the call option to a later expiration"""

        # This would involve complex logic to find a suitable option to roll to
        # For now, return None to indicate no rolling opportunity
        return None

    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate overall portfolio metrics for covered call positions"""

        if not self.positions:
            return {"total_positions": 0, "total_income": 0, "average_yield": 0}

        total_premium = sum(pos.total_premium_received for pos in self.positions)
        total_investment = sum(pos.stock_avg_cost * pos.stock_shares for pos in self.positions)
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions)

        average_yield = sum(pos.calculate_current_yield() for pos in self.positions) / len(self.positions)

        return {
            "total_positions": len(self.positions),
            "total_premium_income": float(total_premium),
            "total_investment": float(total_investment),
            "total_unrealized_pnl": float(total_unrealized_pnl),
            "average_annualized_yield": float(average_yield),
            "positions_near_assignment": len([p for p in self.positions if p.moneyness > 0.95]),
            "positions_profitable": len([p for p in self.positions if p.unrealized_pnl > 0])
        }