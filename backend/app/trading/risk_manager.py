"""
Risk management system for the trading engine
"""

from typing import Dict, List, Optional, Tuple

import structlog

from app.core.config import settings

logger = structlog.get_logger()


class RiskManager:
    """Risk management system for trading operations"""

    def __init__(self, user_id: int, user_risk_params: Optional[Dict] = None):
        self.user_id = user_id
        self.risk_params = user_risk_params or {}

        # Default risk parameters
        self.max_position_size = self.risk_params.get(
            "max_position_size", settings.MAX_POSITION_SIZE
        )
        self.max_daily_loss = self.risk_params.get(
            "max_daily_loss", settings.MAX_DAILY_LOSS
        )
        self.max_portfolio_exposure = self.risk_params.get(
            "max_portfolio_exposure", 0.95
        )  # 95% max exposure
        self.max_single_stock_exposure = self.risk_params.get(
            "max_single_stock_exposure", 0.20
        )  # 20% max per stock
        self.stop_loss_percentage = self.risk_params.get(
            "stop_loss_percentage", 0.05
        )  # 5% stop loss
        self.take_profit_percentage = self.risk_params.get(
            "take_profit_percentage", 0.15
        )  # 15% take profit

        logger.info(
            "Risk manager initialized", user_id=user_id, risk_params=self.risk_params
        )

    def validate_order(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        current_positions: List[Dict],
        account_value: float,
        daily_pnl: float,
    ) -> Tuple[bool, str]:
        """
        Validate if an order meets risk management criteria

        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        try:
            # Calculate order value
            order_value = quantity * price

            # Check maximum position size
            if order_value > self.max_position_size:
                return (
                    False,
                    f"Order value ${order_value:.2f} exceeds maximum position size ${self.max_position_size:.2f}",
                )

            # Check daily loss limit
            if daily_pnl < -self.max_daily_loss:
                return (
                    False,
                    f"Daily loss ${abs(daily_pnl):.2f} exceeds limit ${self.max_daily_loss:.2f}",
                )

            # Check portfolio exposure
            current_exposure = self._calculate_portfolio_exposure(
                current_positions, account_value
            )
            new_exposure = current_exposure + (order_value / account_value)

            if new_exposure > self.max_portfolio_exposure:
                return (
                    False,
                    f"Portfolio exposure {new_exposure:.2%} would exceed limit {self.max_portfolio_exposure:.2%}",
                )

            # Check single stock exposure
            current_stock_exposure = self._calculate_stock_exposure(
                symbol, current_positions, account_value
            )
            new_stock_exposure = current_stock_exposure + (order_value / account_value)

            if new_stock_exposure > self.max_single_stock_exposure:
                return (
                    False,
                    f"Stock exposure {new_stock_exposure:.2%} would exceed limit {self.max_single_stock_exposure:.2%}",
                )

            # Check if we have enough buying power
            if (
                side.upper() == "BUY" and order_value > account_value * 0.95
            ):  # Leave 5% buffer
                return False, "Insufficient buying power for order"

            logger.info(
                "Order validation passed",
                symbol=symbol,
                quantity=quantity,
                price=price,
                side=side,
                order_value=order_value,
            )

            return True, "Order passes risk management checks"

        except Exception as e:
            logger.error("Error validating order", error=str(e), symbol=symbol)
            return False, f"Risk validation error: {str(e)}"

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        account_value: float,
        volatility: Optional[float] = None,
        confidence: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        use_optimizer: bool = True,
    ) -> float:
        """
        Calculate optimal position size based on risk parameters

        Args:
            symbol: Stock symbol
            price: Current price
            account_value: Total account value
            volatility: Stock volatility (optional)
            confidence: Strategy confidence (optional)
            stop_loss_price: Stop loss price for risk calculation
            use_optimizer: Whether to use advanced position optimizer

        Returns:
            float: Recommended position size in dollars
        """
        try:
            if use_optimizer and stop_loss_price:
                # Use advanced position optimizer if available
                try:
                    from app.trading.position_optimizer import PositionOptimizer

                    optimizer = PositionOptimizer(initial_capital=account_value)

                    # Get historical performance (simplified)
                    historical_performance = {
                        "win_rate": 0.6,  # From backtest results
                        "avg_win": 0.08,
                        "avg_loss": 0.04,
                    }

                    position_size, details = optimizer.calculate_optimal_position_size(
                        symbol=symbol,
                        current_price=price,
                        account_value=account_value,
                        signal_confidence=confidence or 0.7,
                        stop_loss_price=stop_loss_price,
                        symbol_volatility=volatility,
                        historical_performance=historical_performance,
                    )

                    logger.info(
                        "Advanced position sizing used",
                        symbol=symbol,
                        position_size=position_size,
                        kelly_fraction=details.get("kelly_fraction"),
                        position_heat=details.get("position_heat"),
                    )

                    return position_size

                except ImportError:
                    logger.warning(
                        "Position optimizer not available, using basic sizing"
                    )
                except Exception as e:
                    logger.error("Error with position optimizer", error=str(e))

            # Fallback to basic position sizing
            # Base position size (2% of account value)
            base_size = account_value * 0.02

            # Adjust for volatility if provided
            if volatility:
                # Reduce position size for high volatility stocks
                volatility_adjustment = max(0.5, 1.0 - (volatility - 0.2) * 2)
                base_size *= volatility_adjustment

            # Adjust for confidence if provided
            if confidence:
                # Increase position size for high confidence trades
                confidence_adjustment = 0.5 + (
                    confidence * 0.5
                )  # 0.5x to 1.0x multiplier
                base_size *= confidence_adjustment

            # Apply stop-loss based risk sizing if available
            if stop_loss_price:
                risk_per_share = abs(price - stop_loss_price)
                max_risk_amount = (
                    account_value * self.max_single_stock_exposure * 0.2
                )  # 20% of max exposure as risk
                risk_based_size = (
                    max_risk_amount / (risk_per_share / price)
                    if risk_per_share > 0
                    else base_size
                )
                base_size = min(base_size, risk_based_size)

            # Ensure position size doesn't exceed limits
            max_size = min(
                self.max_position_size, account_value * self.max_single_stock_exposure
            )

            position_size = min(base_size, max_size)

            # Calculate number of shares
            shares = int(position_size / price)
            final_position_size = shares * price

            logger.info(
                "Basic position size calculated",
                symbol=symbol,
                price=price,
                base_size=base_size,
                final_size=final_position_size,
                shares=shares,
                method="basic",
            )

            return final_position_size

        except Exception as e:
            logger.error("Error calculating position size", error=str(e), symbol=symbol)
            return 0.0

    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price"""
        if side.upper() == "BUY":
            return entry_price * (1 - self.stop_loss_percentage)
        else:  # SELL
            return entry_price * (1 + self.stop_loss_percentage)

    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price"""
        if side.upper() == "BUY":
            return entry_price * (1 + self.take_profit_percentage)
        else:  # SELL
            return entry_price * (1 - self.take_profit_percentage)

    def _calculate_portfolio_exposure(
        self, positions: List[Dict], account_value: float
    ) -> float:
        """Calculate current portfolio exposure"""
        if not positions or account_value <= 0:
            return 0.0

        total_exposure = sum(abs(pos.get("market_value", 0)) for pos in positions)
        return total_exposure / account_value

    def _calculate_stock_exposure(
        self, symbol: str, positions: List[Dict], account_value: float
    ) -> float:
        """Calculate exposure to a specific stock"""
        if not positions or account_value <= 0:
            return 0.0

        stock_exposure = 0.0
        for pos in positions:
            if pos.get("symbol") == symbol:
                stock_exposure += abs(pos.get("market_value", 0))

        return stock_exposure / account_value

    def check_risk_limits(
        self, positions: List[Dict], account_value: float, daily_pnl: float
    ) -> List[str]:
        """
        Check if any risk limits are breached

        Returns:
            List[str]: List of risk limit violations
        """
        violations = []

        try:
            # Check daily loss limit
            if daily_pnl < -self.max_daily_loss:
                violations.append(
                    f"Daily loss ${abs(daily_pnl):.2f} exceeds limit ${self.max_daily_loss:.2f}"
                )

            # Check portfolio exposure
            portfolio_exposure = self._calculate_portfolio_exposure(
                positions, account_value
            )
            if portfolio_exposure > self.max_portfolio_exposure:
                violations.append(
                    f"Portfolio exposure {portfolio_exposure:.2%} exceeds limit {self.max_portfolio_exposure:.2%}"
                )

            # Check individual stock exposure
            for pos in positions:
                symbol = pos.get("symbol")
                stock_exposure = self._calculate_stock_exposure(
                    symbol, positions, account_value
                )
                if stock_exposure > self.max_single_stock_exposure:
                    violations.append(
                        f"Stock {symbol} exposure {stock_exposure:.2%} exceeds limit {self.max_single_stock_exposure:.2%}"
                    )

            if violations:
                logger.warning(
                    "Risk limit violations detected",
                    violations=violations,
                    user_id=self.user_id,
                )

            return violations

        except Exception as e:
            logger.error(
                "Error checking risk limits", error=str(e), user_id=self.user_id
            )
            return [f"Error checking risk limits: {str(e)}"]

    def get_risk_summary(
        self, positions: List[Dict], account_value: float, daily_pnl: float
    ) -> Dict:
        """Get comprehensive risk summary"""
        try:
            portfolio_exposure = self._calculate_portfolio_exposure(
                positions, account_value
            )

            # Calculate exposure by symbol
            symbol_exposures = {}
            for pos in positions:
                symbol = pos.get("symbol")
                if symbol:
                    symbol_exposures[symbol] = self._calculate_stock_exposure(
                        symbol, positions, account_value
                    )

            return {
                "daily_pnl": daily_pnl,
                "daily_loss_limit": self.max_daily_loss,
                "daily_loss_utilization": (
                    abs(daily_pnl) / self.max_daily_loss if daily_pnl < 0 else 0
                ),
                "portfolio_exposure": portfolio_exposure,
                "max_portfolio_exposure": self.max_portfolio_exposure,
                "portfolio_exposure_utilization": portfolio_exposure
                / self.max_portfolio_exposure,
                "symbol_exposures": symbol_exposures,
                "max_single_stock_exposure": self.max_single_stock_exposure,
                "risk_violations": self.check_risk_limits(
                    positions, account_value, daily_pnl
                ),
                "risk_parameters": {
                    "max_position_size": self.max_position_size,
                    "max_daily_loss": self.max_daily_loss,
                    "max_portfolio_exposure": self.max_portfolio_exposure,
                    "max_single_stock_exposure": self.max_single_stock_exposure,
                    "stop_loss_percentage": self.stop_loss_percentage,
                    "take_profit_percentage": self.take_profit_percentage,
                },
            }

        except Exception as e:
            logger.error(
                "Error generating risk summary", error=str(e), user_id=self.user_id
            )
            return {"error": f"Error generating risk summary: {str(e)}"}
