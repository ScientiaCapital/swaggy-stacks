"""
Portfolio Management Service - Extracted from BacktestEngine

Handles portfolio tracking, position management, and trade execution for backtesting
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single trade execution"""

    trade_id: str
    symbol: str
    action: str  # BUY, SELL
    quantity: float
    price: float
    timestamp: datetime
    strategy: str
    confidence: float
    transaction_cost: float


class PortfolioManager:
    """
    Portfolio management for backtesting

    Handles:
    - Position tracking
    - Trade execution
    - Cash management
    - Portfolio value calculation
    - Transaction cost handling
    """

    def __init__(
        self, initial_capital: float = 100000, transaction_cost: float = 0.001
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

        # Portfolio state
        self.cash = initial_capital
        self.positions = defaultdict(float)  # symbol -> quantity
        self.avg_entry_prices = defaultdict(float)  # symbol -> avg_price

        # Trade tracking
        self.trades = []
        self.portfolio_history = []

        logger.info(
            f"PortfolioManager initialized with ${initial_capital:,.2f} capital"
        )

    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions.clear()
        self.avg_entry_prices.clear()
        self.trades.clear()
        self.portfolio_history.clear()
        logger.info("Portfolio reset to initial state")

    def execute_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        date: datetime,
        position_multiplier: float = 1.0,
        strategy: str = "default",
        confidence: float = 0.5,
    ) -> bool:
        """
        Execute a trade and update portfolio state

        Args:
            symbol: Trading symbol
            action: BUY or SELL
            price: Execution price
            date: Trade date
            position_multiplier: Position size multiplier
            strategy: Strategy name for tracking
            confidence: Signal confidence

        Returns:
            True if trade was executed successfully
        """
        try:
            if action == "BUY":
                return self._execute_buy(
                    symbol, price, date, position_multiplier, strategy, confidence
                )
            elif action == "SELL":
                return self._execute_sell(
                    symbol, price, date, position_multiplier, strategy, confidence
                )
            else:
                logger.warning(f"Unknown action: {action}")
                return False

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False

    def _execute_buy(
        self,
        symbol: str,
        price: float,
        date: datetime,
        position_multiplier: float,
        strategy: str,
        confidence: float,
    ) -> bool:
        """Execute buy order"""
        # Calculate position size based on available cash
        max_position_value = self.cash * 0.95  # Leave 5% cash buffer
        target_position_value = (
            max_position_value * position_multiplier * 0.1
        )  # Default 10% of portfolio per position

        if target_position_value < self.cash * 0.001:  # Minimum 0.1% position
            logger.debug(
                f"Position too small for {symbol}: ${target_position_value:.2f}"
            )
            return False

        quantity = target_position_value / price
        total_cost = quantity * price * (1 + self.transaction_cost)

        if total_cost > self.cash:
            # Use available cash
            quantity = (self.cash * 0.95) / (price * (1 + self.transaction_cost))
            total_cost = quantity * price * (1 + self.transaction_cost)

        if quantity <= 0:
            logger.debug(f"Insufficient cash for {symbol} buy")
            return False

        # Update portfolio
        old_quantity = self.positions[symbol]
        old_avg_price = self.avg_entry_prices[symbol]

        # Calculate new average entry price
        if old_quantity > 0:
            total_value = old_quantity * old_avg_price + quantity * price
            total_quantity = old_quantity + quantity
            self.avg_entry_prices[symbol] = total_value / total_quantity
        else:
            self.avg_entry_prices[symbol] = price

        self.positions[symbol] += quantity
        self.cash -= total_cost

        # Record trade
        trade_record = TradeRecord(
            trade_id=f"{symbol}_{date.isoformat()}_{len(self.trades)}",
            symbol=symbol,
            action="BUY",
            quantity=quantity,
            price=price,
            timestamp=date,
            strategy=strategy,
            confidence=confidence,
            transaction_cost=quantity * price * self.transaction_cost,
        )
        self.trades.append(trade_record)

        logger.debug(
            f"BUY {quantity:.2f} {symbol} @ ${price:.2f} (total: ${total_cost:.2f})"
        )
        return True

    def _execute_sell(
        self,
        symbol: str,
        price: float,
        date: datetime,
        position_multiplier: float,
        strategy: str,
        confidence: float,
    ) -> bool:
        """Execute sell order"""
        current_position = self.positions[symbol]

        if current_position <= 0:
            logger.debug(f"No position to sell for {symbol}")
            return False

        # Calculate quantity to sell
        quantity_to_sell = current_position * position_multiplier
        quantity_to_sell = min(quantity_to_sell, current_position)

        if quantity_to_sell <= 0:
            return False

        # Execute sale
        gross_proceeds = quantity_to_sell * price
        transaction_cost = gross_proceeds * self.transaction_cost
        net_proceeds = gross_proceeds - transaction_cost

        self.positions[symbol] -= quantity_to_sell
        self.cash += net_proceeds

        # Clean up zero positions
        if abs(self.positions[symbol]) < 1e-8:
            del self.positions[symbol]
            if symbol in self.avg_entry_prices:
                del self.avg_entry_prices[symbol]

        # Record trade
        trade_record = TradeRecord(
            trade_id=f"{symbol}_{date.isoformat()}_{len(self.trades)}",
            symbol=symbol,
            action="SELL",
            quantity=quantity_to_sell,
            price=price,
            timestamp=date,
            strategy=strategy,
            confidence=confidence,
            transaction_cost=transaction_cost,
        )
        self.trades.append(trade_record)

        logger.debug(
            f"SELL {quantity_to_sell:.2f} {symbol} @ ${price:.2f} (proceeds: ${net_proceeds:.2f})"
        )
        return True

    def calculate_current_portfolio_value(
        self, market_data: Dict[str, pd.DataFrame], current_date: datetime
    ) -> float:
        """Calculate total portfolio value at current date"""
        try:
            total_value = self.cash

            for symbol, quantity in self.positions.items():
                if symbol in market_data and current_date in market_data[symbol].index:
                    current_price = market_data[symbol].loc[current_date, "Close"]
                    position_value = quantity * current_price
                    total_value += position_value
                else:
                    # Use last known price or entry price
                    position_value = quantity * self.avg_entry_prices.get(symbol, 0)
                    total_value += position_value

            return total_value

        except Exception as e:
            logger.error(f"Portfolio value calculation failed: {e}")
            return self.cash

    def calculate_correlation_adjustment(
        self, symbol: str, current_price: float
    ) -> float:
        """Calculate position sizing adjustment based on portfolio correlation"""
        try:
            # Simple correlation adjustment based on existing positions
            if len(self.positions) == 0:
                return 1.0

            # Reduce position size if already holding many positions (diversification)
            position_count = len(self.positions)
            if position_count > 5:
                correlation_penalty = 0.8  # Reduce by 20%
            elif position_count > 10:
                correlation_penalty = 0.6  # Reduce by 40%
            else:
                correlation_penalty = 1.0

            # Check if already holding this symbol
            if symbol in self.positions:
                correlation_penalty *= 0.5  # Reduce additional positions in same symbol

            return correlation_penalty

        except Exception as e:
            logger.warning(f"Correlation adjustment failed: {e}")
            return 1.0

    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of current positions"""
        return {
            "cash": self.cash,
            "positions": dict(self.positions),
            "avg_entry_prices": dict(self.avg_entry_prices),
            "position_count": len(self.positions),
            "total_trades": len(self.trades),
        }

    def get_trade_history(self) -> List[TradeRecord]:
        """Get complete trade history"""
        return self.trades.copy()

    def get_portfolio_history(self) -> List[Dict[str, Any]]:
        """Get portfolio value history"""
        return self.portfolio_history.copy()

    def record_portfolio_snapshot(
        self,
        date: datetime,
        market_data: Dict[str, pd.DataFrame],
        strategy: str = "",
        additional_data: Dict[str, Any] = None,
    ):
        """Record current portfolio state for history tracking"""
        try:
            portfolio_value = self.calculate_current_portfolio_value(market_data, date)

            snapshot = {
                "date": date,
                "total_value": portfolio_value,
                "cash": self.cash,
                "positions": dict(self.positions),
                "strategy": strategy,
                "cash_percentage": (
                    (self.cash / portfolio_value) * 100 if portfolio_value > 0 else 100
                ),
            }

            if additional_data:
                snapshot.update(additional_data)

            self.portfolio_history.append(snapshot)

        except Exception as e:
            logger.error(f"Portfolio snapshot recording failed: {e}")

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate basic portfolio performance metrics"""
        try:
            if not self.portfolio_history:
                return {"error": "No portfolio history available"}

            # Calculate returns
            values = [snapshot["total_value"] for snapshot in self.portfolio_history]
            if len(values) < 2:
                return {"error": "Insufficient data for metrics"}

            total_return = (values[-1] - values[0]) / values[0]

            # Calculate trade metrics
            winning_trades = 0
            losing_trades = 0
            total_trades = len(self.trades)

            buy_trades = [t for t in self.trades if t.action == "BUY"]
            sell_trades = [t for t in self.trades if t.action == "SELL"]

            # Simple P&L calculation
            for sell_trade in sell_trades:
                # Find corresponding buy for this symbol
                symbol_buys = [
                    t
                    for t in buy_trades
                    if t.symbol == sell_trade.symbol
                    and t.timestamp <= sell_trade.timestamp
                ]
                if symbol_buys:
                    avg_buy_price = sum(t.price for t in symbol_buys) / len(symbol_buys)
                    trade_pnl = (sell_trade.price - avg_buy_price) * sell_trade.quantity
                    if trade_pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1

            win_rate = winning_trades / max(winning_trades + losing_trades, 1)

            return {
                "total_return": total_return,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "final_portfolio_value": values[-1],
                "cash_remaining": self.cash,
                "positions_held": len(self.positions),
            }

        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {"error": str(e)}
