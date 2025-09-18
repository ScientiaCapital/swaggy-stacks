"""
Comprehensive Options Backtesting Framework

This framework provides sophisticated backtesting capabilities specifically designed
for options trading strategies. It handles complex scenarios like:
- Options expiration and time decay
- Volatility surface modeling
- Greeks calculation and tracking
- Multi-leg position management
- Assignment risk simulation
- Transaction costs and slippage modeling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from decimal import Decimal
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
from collections import defaultdict
import json

from app.core.base_strategy import BaseStrategy, StrategySignal
from app.strategies.options.black_scholes import BlackScholesCalculator, GreeksData
from app.strategies.options.options_strategy_factory import OptionsStrategyFactory, StrategyType

logger = logging.getLogger(__name__)


@dataclass
class OptionsMarketData:
    """Market data snapshot for options backtesting"""

    timestamp: datetime
    underlying_price: Decimal
    underlying_volume: int

    # Options chain data
    option_chain: Dict[str, Any]

    # Market indicators
    implied_volatility: Decimal
    iv_rank: Decimal
    vix: Decimal
    risk_free_rate: Decimal

    # Technical indicators
    rsi: Decimal
    bollinger_position: Decimal
    trend_strength: Decimal

    # Greeks for existing positions
    position_greeks: Dict[str, GreeksData] = field(default_factory=dict)


@dataclass
class OptionsPosition:
    """Represents an options position in the backtest"""

    symbol: str
    option_type: str  # 'call' or 'put'
    strike: Decimal
    expiration: datetime
    quantity: int  # Negative for short positions
    entry_price: Decimal
    entry_date: datetime

    # Position tracking
    current_price: Decimal = Decimal("0")
    current_greeks: Optional[GreeksData] = None

    # P&L tracking
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    # Strategy metadata
    strategy_name: str = ""
    leg_name: str = ""  # For multi-leg strategies

    @property
    def is_expired(self) -> bool:
        """Check if option has expired"""
        return datetime.now() >= self.expiration

    @property
    def days_to_expiration(self) -> int:
        """Calculate days to expiration"""
        return max(0, (self.expiration - datetime.now()).days)

    @property
    def market_value(self) -> Decimal:
        """Calculate current market value of position"""
        return self.current_price * abs(self.quantity) * 100  # Options are in lots of 100

    def calculate_pnl(self) -> Decimal:
        """Calculate current P&L"""
        entry_value = self.entry_price * abs(self.quantity) * 100
        current_value = self.current_price * abs(self.quantity) * 100

        # For short positions, profit when price decreases
        if self.quantity < 0:
            return entry_value - current_value
        else:
            return current_value - entry_value


@dataclass
class BacktestTransaction:
    """Represents a trading transaction in the backtest"""

    timestamp: datetime
    symbol: str
    action: str  # 'BUY_TO_OPEN', 'SELL_TO_CLOSE', etc.
    quantity: int
    price: Decimal
    commission: Decimal
    strategy_name: str
    signal_strength: int

    # Transaction costs
    bid_ask_spread: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")

    @property
    def total_cost(self) -> Decimal:
        """Calculate total transaction cost including fees"""
        base_cost = self.price * abs(self.quantity) * 100
        return base_cost + self.commission + self.slippage


@dataclass
class BacktestMetrics:
    """Comprehensive backtesting metrics for options strategies"""

    # Basic performance metrics
    total_return: Decimal
    annualized_return: Decimal
    max_drawdown: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal

    # Options-specific metrics
    win_rate: Decimal
    profit_factor: Decimal
    average_winner: Decimal
    average_loser: Decimal
    max_winner: Decimal
    max_loser: Decimal

    # Greeks metrics
    average_delta: Decimal
    average_gamma: Decimal
    average_theta: Decimal
    average_vega: Decimal

    # Transaction metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_commissions: Decimal

    # Strategy-specific metrics
    strategy_returns: Dict[str, Decimal] = field(default_factory=dict)
    monthly_returns: List[Decimal] = field(default_factory=list)

    # Risk metrics
    var_95: Decimal = Decimal("0")  # Value at Risk
    expected_shortfall: Decimal = Decimal("0")
    max_position_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization"""
        return {
            "total_return": float(self.total_return),
            "annualized_return": float(self.annualized_return),
            "max_drawdown": float(self.max_drawdown),
            "sharpe_ratio": float(self.sharpe_ratio),
            "win_rate": float(self.win_rate),
            "profit_factor": float(self.profit_factor),
            "total_trades": self.total_trades,
            "strategy_returns": {k: float(v) for k, v in self.strategy_returns.items()}
        }


class OptionsBacktester:
    """
    Comprehensive Options Backtesting Engine

    This backtester provides sophisticated simulation capabilities for options
    trading strategies with realistic modeling of:
    - Time decay effects
    - Volatility surface changes
    - Greeks evolution
    - Assignment risk
    - Transaction costs and slippage
    """

    def __init__(
        self,
        initial_capital: Decimal = Decimal("100000"),
        commission_per_contract: Decimal = Decimal("1.0"),
        max_slippage_pct: Decimal = Decimal("0.002"),  # 0.2% max slippage
        assignment_probability_threshold: Decimal = Decimal("0.85")  # 85% ITM for assignment
    ):
        """Initialize the options backtester"""

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_per_contract = commission_per_contract
        self.max_slippage_pct = max_slippage_pct
        self.assignment_threshold = assignment_probability_threshold

        # Position tracking
        self.positions: List[OptionsPosition] = []
        self.closed_positions: List[OptionsPosition] = []
        self.transactions: List[BacktestTransaction] = []

        # Strategy management
        self.active_strategies: Dict[str, BaseStrategy] = {}
        self.strategy_factory = OptionsStrategyFactory()

        # Market data and calculations
        self.market_data_history: List[OptionsMarketData] = []
        self.bs_calculator = BlackScholesCalculator()

        # Performance tracking
        self.daily_pnl: List[Tuple[datetime, Decimal]] = []
        self.daily_portfolio_value: List[Tuple[datetime, Decimal]] = []

        logger.info(f"Initialized OptionsBacktester with ${initial_capital} capital")

    def add_strategy(self, strategy_name: str, strategy_type: StrategyType, **config_kwargs) -> None:
        """Add a strategy to the backtest"""

        try:
            # Create mock market data service (would be injected in practice)
            from app.core.market_data import MarketDataService
            market_data_service = MarketDataService()

            strategy = self.strategy_factory.create_strategy(
                strategy_type=strategy_type,
                market_data=market_data_service,
                **config_kwargs
            )

            self.active_strategies[strategy_name] = strategy
            logger.info(f"Added {strategy_type.value} strategy as '{strategy_name}'")

        except Exception as e:
            logger.error(f"Failed to add strategy {strategy_name}: {e}")

    def run_backtest(
        self,
        market_data: List[OptionsMarketData],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestMetrics:
        """
        Run comprehensive options backtest

        Args:
            market_data: Historical market data with options chains
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Comprehensive backtest metrics
        """

        logger.info(f"Starting options backtest with {len(market_data)} data points")

        # Filter data by date range if specified
        if start_date or end_date:
            market_data = self._filter_data_by_date(market_data, start_date, end_date)

        # Process each market data point
        for i, data_point in enumerate(market_data):
            self._process_market_update(data_point)

            # Log progress periodically
            if i % 100 == 0:
                logger.debug(f"Processed {i}/{len(market_data)} data points")

        # Calculate final metrics
        metrics = self._calculate_metrics()

        logger.info(f"Backtest completed. Total return: {metrics.total_return:.2%}, "
                   f"Win rate: {metrics.win_rate:.2%}, Total trades: {metrics.total_trades}")

        return metrics

    def _process_market_update(self, market_data: OptionsMarketData) -> None:
        """Process a single market data update"""

        self.market_data_history.append(market_data)

        # Update existing positions
        self._update_positions(market_data)

        # Check for expiration and assignment
        self._handle_expirations(market_data)
        self._check_assignment_risk(market_data)

        # Generate new signals from strategies
        self._generate_strategy_signals(market_data)

        # Record daily portfolio value
        portfolio_value = self._calculate_portfolio_value(market_data)
        self.daily_portfolio_value.append((market_data.timestamp, portfolio_value))

    def _update_positions(self, market_data: OptionsMarketData) -> None:
        """Update current prices and Greeks for all positions"""

        underlying_price = market_data.underlying_price

        for position in self.positions:
            # Update option price using Black-Scholes or market data
            option_price = self._get_option_price(position, market_data)
            position.current_price = option_price

            # Calculate Greeks
            if not position.is_expired:
                position.current_greeks = self._calculate_position_greeks(position, market_data)

            # Update P&L
            position.unrealized_pnl = position.calculate_pnl()

    def _get_option_price(self, position: OptionsPosition, market_data: OptionsMarketData) -> Decimal:
        """Get option price from market data or calculate using Black-Scholes"""

        # Try to get from market data first
        option_key = f"{position.symbol}_{position.strike}_{position.option_type}_{position.expiration.strftime('%Y%m%d')}"

        if option_key in market_data.option_chain:
            return Decimal(str(market_data.option_chain[option_key].get('price', 0)))

        # Fall back to Black-Scholes calculation
        time_to_expiry = max(0.001, position.days_to_expiration / 365.0)

        bs_result = self.bs_calculator.calculate_option_price(
            stock_price=market_data.underlying_price,
            strike_price=position.strike,
            time_to_expiry=Decimal(str(time_to_expiry)),
            risk_free_rate=market_data.risk_free_rate,
            volatility=market_data.implied_volatility,
            option_type=position.option_type
        )

        return bs_result.option_price

    def _calculate_position_greeks(self, position: OptionsPosition, market_data: OptionsMarketData) -> GreeksData:
        """Calculate Greeks for a position"""

        time_to_expiry = max(0.001, position.days_to_expiration / 365.0)

        return self.bs_calculator.calculate_greeks(
            stock_price=market_data.underlying_price,
            strike_price=position.strike,
            time_to_expiry=Decimal(str(time_to_expiry)),
            risk_free_rate=market_data.risk_free_rate,
            volatility=market_data.implied_volatility,
            option_type=position.option_type
        )

    def _handle_expirations(self, market_data: OptionsMarketData) -> None:
        """Handle option expirations"""

        expired_positions = [pos for pos in self.positions if pos.is_expired]

        for position in expired_positions:
            # Calculate intrinsic value at expiration
            intrinsic_value = self._calculate_intrinsic_value(position, market_data.underlying_price)

            # Settle the position
            if intrinsic_value > 0:
                # Option has value, simulate exercise/assignment
                self._settle_expired_position(position, intrinsic_value, market_data)
            else:
                # Option expires worthless
                position.current_price = Decimal("0")
                position.realized_pnl = position.calculate_pnl()

            # Move to closed positions
            self.closed_positions.append(position)
            self.positions.remove(position)

    def _calculate_intrinsic_value(self, position: OptionsPosition, underlying_price: Decimal) -> Decimal:
        """Calculate intrinsic value of option at expiration"""

        if position.option_type == 'call':
            return max(Decimal("0"), underlying_price - position.strike)
        else:  # put
            return max(Decimal("0"), position.strike - underlying_price)

    def _settle_expired_position(self, position: OptionsPosition, intrinsic_value: Decimal, market_data: OptionsMarketData) -> None:
        """Settle an expired position with intrinsic value"""

        # For long positions, we receive the intrinsic value
        # For short positions, we pay the intrinsic value
        if position.quantity > 0:
            position.current_price = intrinsic_value
        else:
            position.current_price = intrinsic_value

        position.realized_pnl = position.calculate_pnl()

        # Record settlement transaction
        settlement_transaction = BacktestTransaction(
            timestamp=market_data.timestamp,
            symbol=position.symbol,
            action="SETTLEMENT",
            quantity=abs(position.quantity),
            price=intrinsic_value,
            commission=Decimal("0"),  # No commission on settlement
            strategy_name=position.strategy_name
        )

        self.transactions.append(settlement_transaction)

    def _check_assignment_risk(self, market_data: OptionsMarketData) -> None:
        """Check for early assignment risk on short positions"""

        short_positions = [pos for pos in self.positions if pos.quantity < 0]

        for position in short_positions:
            if position.days_to_expiration <= 7:  # Only check near expiration
                moneyness = self._calculate_moneyness(position, market_data.underlying_price)

                # High probability of assignment for deep ITM options
                if moneyness > self.assignment_threshold:
                    self._handle_assignment(position, market_data)

    def _calculate_moneyness(self, position: OptionsPosition, underlying_price: Decimal) -> Decimal:
        """Calculate option moneyness (S/K for calls, K/S for puts)"""

        if position.option_type == 'call':
            return underlying_price / position.strike
        else:  # put
            return position.strike / underlying_price

    def _handle_assignment(self, position: OptionsPosition, market_data: OptionsMarketData) -> None:
        """Handle early assignment of short options"""

        # Calculate assignment value
        intrinsic_value = self._calculate_intrinsic_value(position, market_data.underlying_price)

        # Force settlement
        position.current_price = intrinsic_value
        position.realized_pnl = position.calculate_pnl()

        # Record assignment transaction
        assignment_transaction = BacktestTransaction(
            timestamp=market_data.timestamp,
            symbol=position.symbol,
            action="ASSIGNMENT",
            quantity=abs(position.quantity),
            price=intrinsic_value,
            commission=Decimal("0"),
            strategy_name=position.strategy_name
        )

        self.transactions.append(assignment_transaction)

        # Move to closed positions
        self.closed_positions.append(position)
        self.positions.remove(position)

        logger.info(f"Early assignment: {position.symbol} at ${intrinsic_value}")

    def _generate_strategy_signals(self, market_data: OptionsMarketData) -> None:
        """Generate trading signals from active strategies"""

        # Convert market data to format expected by strategies
        strategy_market_data = {
            'current_price': float(market_data.underlying_price),
            'iv_rank': float(market_data.iv_rank),
            'vix': float(market_data.vix),
            'rsi': float(market_data.rsi),
            'bollinger_position': float(market_data.bollinger_position),
            'trend_strength': float(market_data.trend_strength),
            'option_chain': market_data.option_chain,
            'implied_volatility': float(market_data.implied_volatility),
            'risk_free_rate': float(market_data.risk_free_rate)
        }

        for strategy_name, strategy in self.active_strategies.items():
            try:
                signals = strategy.generate_signals("SPY", strategy_market_data)  # Using SPY as default symbol

                for signal in signals:
                    self._execute_signal(signal, market_data, strategy_name)

            except Exception as e:
                logger.error(f"Error generating signals for {strategy_name}: {e}")

    def _execute_signal(self, signal: StrategySignal, market_data: OptionsMarketData, strategy_name: str) -> None:
        """Execute a trading signal"""

        try:
            # Calculate transaction costs
            commission = self.commission_per_contract * signal.quantity
            slippage = self._calculate_slippage(signal.entry_price, market_data)

            # Create transaction record
            transaction = BacktestTransaction(
                timestamp=market_data.timestamp,
                symbol=signal.symbol,
                action=signal.signal_type,
                quantity=signal.quantity,
                price=signal.entry_price,
                commission=commission,
                strategy_name=strategy_name,
                signal_strength=signal.strength,
                slippage=slippage
            )

            # Check if we have sufficient capital
            if not self._has_sufficient_capital(transaction):
                logger.warning(f"Insufficient capital for {signal.signal_type} {signal.quantity} {signal.symbol}")
                return

            # Execute the trade
            if signal.signal_type in ["BUY_TO_OPEN", "SELL_TO_OPEN"]:
                self._open_position(signal, market_data, strategy_name, transaction)
            elif signal.signal_type in ["BUY_TO_CLOSE", "SELL_TO_CLOSE"]:
                self._close_position(signal, market_data, strategy_name, transaction)

            self.transactions.append(transaction)

        except Exception as e:
            logger.error(f"Failed to execute signal {signal.signal_type} for {signal.symbol}: {e}")

    def _calculate_slippage(self, entry_price: Decimal, market_data: OptionsMarketData) -> Decimal:
        """Calculate realistic slippage based on market conditions"""

        # Higher slippage for higher volatility
        vol_multiplier = min(2.0, float(market_data.iv_rank) / 50.0)
        base_slippage = entry_price * self.max_slippage_pct * Decimal(str(vol_multiplier))

        return base_slippage

    def _has_sufficient_capital(self, transaction: BacktestTransaction) -> bool:
        """Check if sufficient capital is available for transaction"""

        required_capital = transaction.total_cost
        available_capital = self.current_capital

        return available_capital >= required_capital

    def _open_position(self, signal: StrategySignal, market_data: OptionsMarketData,
                      strategy_name: str, transaction: BacktestTransaction) -> None:
        """Open a new options position"""

        # Extract option details from signal metadata
        option_type = signal.metadata.get('option_type', 'call')
        strike_price = Decimal(str(signal.metadata.get('strike_price', signal.entry_price)))
        expiration_str = signal.metadata.get('expiration_date', '')

        try:
            expiration = datetime.fromisoformat(expiration_str) if expiration_str else market_data.timestamp + timedelta(days=30)
        except:
            expiration = market_data.timestamp + timedelta(days=30)

        # Determine position quantity (negative for short positions)
        position_quantity = signal.quantity if signal.signal_type == "BUY_TO_OPEN" else -signal.quantity

        # Create position
        position = OptionsPosition(
            symbol=signal.symbol,
            option_type=option_type,
            strike=strike_price,
            expiration=expiration,
            quantity=position_quantity,
            entry_price=signal.entry_price,
            entry_date=market_data.timestamp,
            current_price=signal.entry_price,
            strategy_name=strategy_name,
            leg_name=signal.metadata.get('leg_name', '')
        )

        self.positions.append(position)

        # Update capital
        self.current_capital -= transaction.total_cost

        logger.debug(f"Opened {option_type} position: {signal.quantity} {signal.symbol} ${strike_price} @ ${signal.entry_price}")

    def _close_position(self, signal: StrategySignal, market_data: OptionsMarketData,
                       strategy_name: str, transaction: BacktestTransaction) -> None:
        """Close an existing options position"""

        # Find matching position to close
        matching_positions = [
            pos for pos in self.positions
            if pos.symbol == signal.symbol and pos.strategy_name == strategy_name
        ]

        if not matching_positions:
            logger.warning(f"No matching position found to close for {signal.symbol}")
            return

        # Close the first matching position (could be enhanced with better matching logic)
        position = matching_positions[0]

        # Calculate final P&L
        position.current_price = signal.entry_price
        position.realized_pnl = position.calculate_pnl()

        # Update capital
        self.current_capital += transaction.total_cost - transaction.commission

        # Move to closed positions
        self.closed_positions.append(position)
        self.positions.remove(position)

        logger.debug(f"Closed position: {position.symbol} P&L: ${position.realized_pnl}")

    def _calculate_portfolio_value(self, market_data: OptionsMarketData) -> Decimal:
        """Calculate current portfolio value"""

        # Cash + unrealized P&L from open positions
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions)
        return self.current_capital + unrealized_pnl

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics"""

        # Basic return calculations
        final_value = self.daily_portfolio_value[-1][1] if self.daily_portfolio_value else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Calculate other metrics
        win_rate = self._calculate_win_rate()
        profit_factor = self._calculate_profit_factor()
        max_drawdown = self._calculate_max_drawdown()
        sharpe_ratio = self._calculate_sharpe_ratio()

        # Greeks statistics
        avg_greeks = self._calculate_average_greeks()

        # Transaction statistics
        total_trades = len(self.transactions)
        total_commissions = sum(t.commission for t in self.transactions)

        # Strategy-specific returns
        strategy_returns = self._calculate_strategy_returns()

        return BacktestMetrics(
            total_return=total_return,
            annualized_return=self._annualize_return(total_return),
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=Decimal("0"),  # Would calculate if needed
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_winner=self._calculate_average_winner(),
            average_loser=self._calculate_average_loser(),
            max_winner=self._calculate_max_winner(),
            max_loser=self._calculate_max_loser(),
            average_delta=avg_greeks.get('delta', Decimal("0")),
            average_gamma=avg_greeks.get('gamma', Decimal("0")),
            average_theta=avg_greeks.get('theta', Decimal("0")),
            average_vega=avg_greeks.get('vega', Decimal("0")),
            total_trades=total_trades,
            winning_trades=len([p for p in self.closed_positions if p.realized_pnl > 0]),
            losing_trades=len([p for p in self.closed_positions if p.realized_pnl < 0]),
            total_commissions=total_commissions,
            strategy_returns=strategy_returns,
            max_position_count=max(len(self.positions), len(self.closed_positions))
        )

    def _calculate_win_rate(self) -> Decimal:
        """Calculate win rate from closed positions"""
        if not self.closed_positions:
            return Decimal("0")

        winners = len([p for p in self.closed_positions if p.realized_pnl > 0])
        return Decimal(str(winners)) / Decimal(str(len(self.closed_positions)))

    def _calculate_profit_factor(self) -> Decimal:
        """Calculate profit factor (gross profits / gross losses)"""
        gross_profits = sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0)
        gross_losses = abs(sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl < 0))

        return gross_profits / gross_losses if gross_losses > 0 else Decimal("0")

    def _calculate_max_drawdown(self) -> Decimal:
        """Calculate maximum drawdown from portfolio value history"""
        if len(self.daily_portfolio_value) < 2:
            return Decimal("0")

        values = [value for _, value in self.daily_portfolio_value]
        peak = values[0]
        max_dd = Decimal("0")

        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)

        return max_dd

    def _calculate_sharpe_ratio(self) -> Decimal:
        """Calculate Sharpe ratio from daily returns"""
        if len(self.daily_portfolio_value) < 2:
            return Decimal("0")

        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(self.daily_portfolio_value)):
            prev_value = self.daily_portfolio_value[i-1][1]
            curr_value = self.daily_portfolio_value[i][1]
            daily_return = (curr_value - prev_value) / prev_value
            daily_returns.append(daily_return)

        if not daily_returns:
            return Decimal("0")

        # Calculate mean and std of returns
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
        std_return = variance ** Decimal("0.5")

        # Annualize and calculate Sharpe (assuming 0% risk-free rate)
        annual_return = mean_return * 252
        annual_std = std_return * (252 ** Decimal("0.5"))

        return annual_return / annual_std if annual_std > 0 else Decimal("0")

    def _calculate_average_greeks(self) -> Dict[str, Decimal]:
        """Calculate average Greeks across all positions"""
        if not self.closed_positions:
            return {}

        greeks_sum = defaultdict(Decimal)
        count = 0

        for position in self.closed_positions:
            if position.current_greeks:
                greeks_sum['delta'] += position.current_greeks.delta
                greeks_sum['gamma'] += position.current_greeks.gamma
                greeks_sum['theta'] += position.current_greeks.theta
                greeks_sum['vega'] += position.current_greeks.vega
                count += 1

        if count == 0:
            return {}

        return {greek: total / count for greek, total in greeks_sum.items()}

    def _calculate_average_winner(self) -> Decimal:
        """Calculate average winning trade"""
        winners = [p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0]
        return sum(winners) / len(winners) if winners else Decimal("0")

    def _calculate_average_loser(self) -> Decimal:
        """Calculate average losing trade"""
        losers = [p.realized_pnl for p in self.closed_positions if p.realized_pnl < 0]
        return sum(losers) / len(losers) if losers else Decimal("0")

    def _calculate_max_winner(self) -> Decimal:
        """Calculate maximum winning trade"""
        winners = [p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0]
        return max(winners) if winners else Decimal("0")

    def _calculate_max_loser(self) -> Decimal:
        """Calculate maximum losing trade"""
        losers = [p.realized_pnl for p in self.closed_positions if p.realized_pnl < 0]
        return min(losers) if losers else Decimal("0")

    def _calculate_strategy_returns(self) -> Dict[str, Decimal]:
        """Calculate returns by strategy"""
        strategy_pnl = defaultdict(Decimal)

        for position in self.closed_positions:
            strategy_pnl[position.strategy_name] += position.realized_pnl

        return dict(strategy_pnl)

    def _annualize_return(self, total_return: Decimal) -> Decimal:
        """Annualize a total return"""
        if not self.daily_portfolio_value:
            return Decimal("0")

        start_date = self.daily_portfolio_value[0][0]
        end_date = self.daily_portfolio_value[-1][0]
        days = (end_date - start_date).days

        if days <= 0:
            return Decimal("0")

        years = Decimal(str(days)) / Decimal("365")
        return ((1 + total_return) ** (1 / years)) - 1

    def _filter_data_by_date(
        self,
        market_data: List[OptionsMarketData],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[OptionsMarketData]:
        """Filter market data by date range"""

        filtered_data = market_data

        if start_date:
            filtered_data = [d for d in filtered_data if d.timestamp >= start_date]

        if end_date:
            filtered_data = [d for d in filtered_data if d.timestamp <= end_date]

        return filtered_data

    def export_results(self, filepath: str) -> None:
        """Export backtest results to file"""

        results = {
            "backtest_config": {
                "initial_capital": float(self.initial_capital),
                "commission_per_contract": float(self.commission_per_contract),
                "max_slippage_pct": float(self.max_slippage_pct)
            },
            "metrics": self._calculate_metrics().to_dict(),
            "transactions": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "symbol": t.symbol,
                    "action": t.action,
                    "quantity": t.quantity,
                    "price": float(t.price),
                    "commission": float(t.commission),
                    "strategy": t.strategy_name
                }
                for t in self.transactions
            ],
            "daily_portfolio_value": [
                {
                    "date": date.isoformat(),
                    "value": float(value)
                }
                for date, value in self.daily_portfolio_value
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Backtest results exported to {filepath}")

    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of current positions"""

        return {
            "open_positions": len(self.positions),
            "closed_positions": len(self.closed_positions),
            "total_unrealized_pnl": float(sum(p.unrealized_pnl for p in self.positions)),
            "total_realized_pnl": float(sum(p.realized_pnl for p in self.closed_positions)),
            "current_capital": float(self.current_capital),
            "portfolio_value": float(self._calculate_portfolio_value(self.market_data_history[-1]) if self.market_data_history else self.current_capital)
        }