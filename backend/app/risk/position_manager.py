"""
Comprehensive Risk Management Module
Integrates with existing TradingManager and risk_manager.py patterns
"""

import math
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import structlog
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.exceptions import RiskManagementError
from app.models.trade import Trade
from app.monitoring.metrics import PrometheusMetrics
from app.trading.risk_manager import RiskManager

logger = structlog.get_logger()


class PositionSizer:
    """Advanced position sizing algorithms including Kelly Criterion"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_position_size = self.config.get("min_position_size", 100.0)
        self.max_position_size = self.config.get("max_position_size", 50000.0)
        self.kelly_lookback_trades = self.config.get("kelly_lookback_trades", 50)
        self.kelly_max_fraction = self.config.get("kelly_max_fraction", 0.25)  # 25% max
        self.fixed_fraction_default = self.config.get("fixed_fraction_default", 0.02)  # 2%

    def kelly_criterion(
        self,
        account_value: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        confidence_multiplier: float = 1.0
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate position size using Kelly Criterion

        Formula: f = (bp - q) / b
        where:
        - f = fraction of capital to wager
        - b = odds received on the wager (avg_win/avg_loss)
        - p = probability of winning (win_rate)
        - q = probability of losing (1 - win_rate)
        """
        try:
            if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
                raise ValueError("Invalid parameters for Kelly criterion")

            # Calculate Kelly fraction
            b = avg_win / abs(avg_loss)  # Odds ratio
            p = win_rate  # Win probability
            q = 1 - win_rate  # Loss probability

            kelly_fraction = (b * p - q) / b

            # Apply safety constraints
            kelly_fraction = max(0, kelly_fraction)  # No negative positions
            kelly_fraction = min(kelly_fraction, self.kelly_max_fraction)  # Cap at 25%

            # Apply confidence multiplier
            adjusted_fraction = kelly_fraction * confidence_multiplier

            # Calculate position size
            position_size = account_value * adjusted_fraction

            # Apply min/max constraints
            position_size = max(self.min_position_size, position_size)
            position_size = min(self.max_position_size, position_size)

            details = {
                "kelly_fraction": kelly_fraction,
                "adjusted_fraction": adjusted_fraction,
                "odds_ratio": b,
                "win_rate": win_rate,
                "confidence_multiplier": confidence_multiplier,
                "method": "kelly_criterion"
            }

            logger.info(
                "Kelly criterion position sizing",
                position_size=position_size,
                kelly_fraction=kelly_fraction,
                win_rate=win_rate,
                odds_ratio=b
            )

            return position_size, details

        except Exception as e:
            logger.error("Error in Kelly criterion calculation", error=str(e))
            # Fallback to fixed fraction
            return self.fixed_fractional(account_value, self.fixed_fraction_default)

    def fixed_fractional(
        self,
        account_value: float,
        fraction: float = None,
        confidence_multiplier: float = 1.0
    ) -> Tuple[float, Dict[str, Any]]:
        """Fixed fractional position sizing"""
        try:
            fraction = fraction or self.fixed_fraction_default
            adjusted_fraction = fraction * confidence_multiplier

            position_size = account_value * adjusted_fraction

            # Apply constraints
            position_size = max(self.min_position_size, position_size)
            position_size = min(self.max_position_size, position_size)

            details = {
                "fraction": fraction,
                "adjusted_fraction": adjusted_fraction,
                "confidence_multiplier": confidence_multiplier,
                "method": "fixed_fractional"
            }

            logger.info(
                "Fixed fractional position sizing",
                position_size=position_size,
                fraction=fraction
            )

            return position_size, details

        except Exception as e:
            logger.error("Error in fixed fractional calculation", error=str(e))
            return self.min_position_size, {"method": "fallback", "error": str(e)}

    def volatility_adjusted(
        self,
        account_value: float,
        base_fraction: float,
        volatility: float,
        target_volatility: float = 0.20
    ) -> Tuple[float, Dict[str, Any]]:
        """Volatility-adjusted position sizing"""
        try:
            # Adjust position size inversely to volatility
            volatility_multiplier = target_volatility / max(volatility, 0.01)

            # Cap the multiplier to prevent extreme positions
            volatility_multiplier = min(volatility_multiplier, 3.0)
            volatility_multiplier = max(volatility_multiplier, 0.1)

            adjusted_fraction = base_fraction * volatility_multiplier
            position_size = account_value * adjusted_fraction

            # Apply constraints
            position_size = max(self.min_position_size, position_size)
            position_size = min(self.max_position_size, position_size)

            details = {
                "base_fraction": base_fraction,
                "volatility": volatility,
                "target_volatility": target_volatility,
                "volatility_multiplier": volatility_multiplier,
                "adjusted_fraction": adjusted_fraction,
                "method": "volatility_adjusted"
            }

            logger.info(
                "Volatility-adjusted position sizing",
                position_size=position_size,
                volatility=volatility,
                multiplier=volatility_multiplier
            )

            return position_size, details

        except Exception as e:
            logger.error("Error in volatility-adjusted calculation", error=str(e))
            return self.fixed_fractional(account_value, base_fraction)


class StopLossManager:
    """Advanced stop loss management with multiple methodologies"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.atr_periods = self.config.get("atr_periods", 14)
        self.atr_multiplier = self.config.get("atr_multiplier", 2.0)
        self.min_stop_distance = self.config.get("min_stop_distance", 0.02)  # 2%
        self.max_stop_distance = self.config.get("max_stop_distance", 0.10)  # 10%

    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float]) -> float:
        """Calculate Average True Range"""
        try:
            if len(highs) < self.atr_periods or len(lows) < self.atr_periods or len(closes) < self.atr_periods:
                return 0.0

            true_ranges = []

            for i in range(1, len(closes)):
                high_low = highs[i] - lows[i]
                high_close = abs(highs[i] - closes[i-1])
                low_close = abs(lows[i] - closes[i-1])

                true_range = max(high_low, high_close, low_close)
                true_ranges.append(true_range)

            # Calculate ATR as simple moving average of true ranges
            atr = sum(true_ranges[-self.atr_periods:]) / min(len(true_ranges), self.atr_periods)

            logger.debug("ATR calculated", atr=atr, periods=self.atr_periods)
            return atr

        except Exception as e:
            logger.error("Error calculating ATR", error=str(e))
            return 0.0

    def atr_based_stop_loss(
        self,
        entry_price: float,
        side: str,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        multiplier: float = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate ATR-based stop loss"""
        try:
            atr = self.calculate_atr(highs, lows, closes)
            multiplier = multiplier or self.atr_multiplier

            if atr == 0:
                # Fallback to percentage-based
                return self.percentage_based_stop_loss(entry_price, side, self.min_stop_distance)

            stop_distance = atr * multiplier

            # Apply min/max constraints
            price_pct_distance = stop_distance / entry_price
            price_pct_distance = max(self.min_stop_distance, price_pct_distance)
            price_pct_distance = min(self.max_stop_distance, price_pct_distance)

            if side.upper() == "BUY":
                stop_price = entry_price * (1 - price_pct_distance)
            else:  # SELL
                stop_price = entry_price * (1 + price_pct_distance)

            details = {
                "method": "atr_based",
                "atr": atr,
                "multiplier": multiplier,
                "stop_distance": stop_distance,
                "price_pct_distance": price_pct_distance
            }

            logger.info(
                "ATR-based stop loss calculated",
                entry_price=entry_price,
                stop_price=stop_price,
                atr=atr,
                side=side
            )

            return stop_price, details

        except Exception as e:
            logger.error("Error calculating ATR-based stop loss", error=str(e))
            return self.percentage_based_stop_loss(entry_price, side, self.min_stop_distance)

    def percentage_based_stop_loss(
        self,
        entry_price: float,
        side: str,
        percentage: float
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate percentage-based stop loss"""
        try:
            if side.upper() == "BUY":
                stop_price = entry_price * (1 - percentage)
            else:  # SELL
                stop_price = entry_price * (1 + percentage)

            details = {
                "method": "percentage_based",
                "percentage": percentage
            }

            logger.info(
                "Percentage-based stop loss calculated",
                entry_price=entry_price,
                stop_price=stop_price,
                percentage=percentage,
                side=side
            )

            return stop_price, details

        except Exception as e:
            logger.error("Error calculating percentage-based stop loss", error=str(e))
            return entry_price * (0.95 if side.upper() == "BUY" else 1.05), {"method": "fallback"}

    def pattern_based_stop_loss(
        self,
        entry_price: float,
        side: str,
        support_resistance_levels: List[float]
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate pattern-based stop loss using support/resistance levels"""
        try:
            if not support_resistance_levels:
                return self.percentage_based_stop_loss(entry_price, side, self.min_stop_distance)

            if side.upper() == "BUY":
                # Find nearest support level below entry price
                valid_levels = [level for level in support_resistance_levels if level < entry_price]
                if valid_levels:
                    stop_price = max(valid_levels) - (entry_price * 0.005)  # Small buffer
                else:
                    return self.percentage_based_stop_loss(entry_price, side, self.min_stop_distance)
            else:  # SELL
                # Find nearest resistance level above entry price
                valid_levels = [level for level in support_resistance_levels if level > entry_price]
                if valid_levels:
                    stop_price = min(valid_levels) + (entry_price * 0.005)  # Small buffer
                else:
                    return self.percentage_based_stop_loss(entry_price, side, self.min_stop_distance)

            # Validate stop distance
            stop_distance_pct = abs(stop_price - entry_price) / entry_price
            if stop_distance_pct < self.min_stop_distance or stop_distance_pct > self.max_stop_distance:
                return self.percentage_based_stop_loss(entry_price, side, self.min_stop_distance)

            details = {
                "method": "pattern_based",
                "support_resistance_levels": support_resistance_levels,
                "stop_distance_pct": stop_distance_pct
            }

            logger.info(
                "Pattern-based stop loss calculated",
                entry_price=entry_price,
                stop_price=stop_price,
                side=side
            )

            return stop_price, details

        except Exception as e:
            logger.error("Error calculating pattern-based stop loss", error=str(e))
            return self.percentage_based_stop_loss(entry_price, side, self.min_stop_distance)


class PortfolioRiskManager:
    """Portfolio-level risk management with VaR calculations and correlation limits"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.var_confidence_level = self.config.get("var_confidence_level", 0.95)
        self.var_lookback_days = self.config.get("var_lookback_days", 252)  # 1 year
        self.max_correlation = self.config.get("max_correlation", 0.7)
        self.max_sector_exposure = self.config.get("max_sector_exposure", 0.30)  # 30%

    def calculate_portfolio_var(
        self,
        positions: List[Dict[str, Any]],
        price_history: Dict[str, List[float]],
        portfolio_value: float
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate Value at Risk using historical simulation"""
        try:
            if not positions or not price_history:
                return 0.0, {"method": "historical_simulation", "error": "insufficient_data"}

            # Calculate position weights
            weights = {}
            total_value = sum(pos.get("market_value", 0) for pos in positions)

            if total_value == 0:
                return 0.0, {"method": "historical_simulation", "error": "zero_portfolio_value"}

            for pos in positions:
                symbol = pos.get("symbol")
                if symbol and symbol in price_history:
                    weights[symbol] = pos.get("market_value", 0) / total_value

            # Calculate daily returns for each position
            returns_matrix = []
            symbols = list(weights.keys())

            min_length = min(len(price_history[symbol]) for symbol in symbols)
            if min_length < 30:  # Need at least 30 days
                return 0.0, {"method": "historical_simulation", "error": "insufficient_history"}

            for i in range(1, min(min_length, self.var_lookback_days + 1)):
                daily_returns = []
                for symbol in symbols:
                    prices = price_history[symbol]
                    if len(prices) > i:
                        daily_return = (prices[-i] - prices[-i-1]) / prices[-i-1]
                        daily_returns.append(daily_return * weights[symbol])

                if daily_returns:
                    portfolio_return = sum(daily_returns)
                    returns_matrix.append(portfolio_return)

            if len(returns_matrix) < 10:
                return 0.0, {"method": "historical_simulation", "error": "insufficient_return_data"}

            # Calculate VaR
            returns_matrix.sort()
            var_index = int((1 - self.var_confidence_level) * len(returns_matrix))
            var_return = returns_matrix[var_index] if var_index < len(returns_matrix) else returns_matrix[0]

            # Convert to dollar amount
            var_dollar = abs(var_return * portfolio_value)

            details = {
                "method": "historical_simulation",
                "confidence_level": self.var_confidence_level,
                "lookback_days": len(returns_matrix),
                "var_return_pct": var_return,
                "var_dollar": var_dollar,
                "portfolio_positions": len(positions)
            }

            logger.info(
                "Portfolio VaR calculated",
                var_dollar=var_dollar,
                var_pct=var_return,
                confidence=self.var_confidence_level
            )

            return var_dollar, details

        except Exception as e:
            logger.error("Error calculating portfolio VaR", error=str(e))
            return 0.0, {"method": "historical_simulation", "error": str(e)}

    def calculate_correlation_matrix(
        self,
        price_history: Dict[str, List[float]],
        symbols: List[str]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calculate correlation matrix for portfolio positions"""
        try:
            if len(symbols) < 2:
                return np.array([]), {"error": "insufficient_symbols"}

            # Prepare returns data
            returns_data = {}
            min_length = float('inf')

            for symbol in symbols:
                if symbol in price_history and len(price_history[symbol]) > 1:
                    prices = price_history[symbol]
                    returns = [(prices[i] - prices[i-1]) / prices[i-1]
                             for i in range(1, len(prices))]
                    returns_data[symbol] = returns
                    min_length = min(min_length, len(returns))

            if min_length < 30 or len(returns_data) < 2:
                return np.array([]), {"error": "insufficient_data"}

            # Create returns matrix
            returns_matrix = []
            valid_symbols = list(returns_data.keys())

            for symbol in valid_symbols:
                returns_matrix.append(returns_data[symbol][-min_length:])

            returns_matrix = np.array(returns_matrix)

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(returns_matrix)

            details = {
                "symbols": valid_symbols,
                "correlation_periods": min_length,
                "max_correlation": float(np.max(correlation_matrix[correlation_matrix < 1.0])),
                "min_correlation": float(np.min(correlation_matrix)),
                "avg_correlation": float(np.mean(correlation_matrix[correlation_matrix < 1.0]))
            }

            logger.info(
                "Correlation matrix calculated",
                symbols_count=len(valid_symbols),
                max_correlation=details["max_correlation"]
            )

            return correlation_matrix, details

        except Exception as e:
            logger.error("Error calculating correlation matrix", error=str(e))
            return np.array([]), {"error": str(e)}

    def check_correlation_limits(
        self,
        positions: List[Dict[str, Any]],
        price_history: Dict[str, List[float]],
        new_symbol: str = None
    ) -> Tuple[bool, List[str]]:
        """Check if adding a new position would violate correlation limits"""
        try:
            symbols = [pos.get("symbol") for pos in positions if pos.get("symbol")]
            if new_symbol:
                symbols.append(new_symbol)

            if len(symbols) < 2:
                return True, []

            correlation_matrix, details = self.calculate_correlation_matrix(price_history, symbols)

            if correlation_matrix.size == 0:
                return True, ["Unable to calculate correlations - insufficient data"]

            violations = []

            # Check pairwise correlations
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    correlation = correlation_matrix[i, j]
                    if abs(correlation) > self.max_correlation:
                        violations.append(
                            f"High correlation ({correlation:.3f}) between {symbols[i]} and {symbols[j]}"
                        )

            is_valid = len(violations) == 0

            logger.info(
                "Correlation limits check",
                is_valid=is_valid,
                violations_count=len(violations),
                max_correlation=details.get("max_correlation", 0)
            )

            return is_valid, violations

        except Exception as e:
            logger.error("Error checking correlation limits", error=str(e))
            return True, [f"Correlation check error: {str(e)}"]

    def calculate_sector_exposure(
        self,
        positions: List[Dict[str, Any]],
        sector_mappings: Dict[str, str]
    ) -> Dict[str, float]:
        """Calculate exposure by sector"""
        try:
            sector_exposures = {}
            total_value = sum(pos.get("market_value", 0) for pos in positions)

            if total_value == 0:
                return {}

            for pos in positions:
                symbol = pos.get("symbol")
                market_value = pos.get("market_value", 0)

                if symbol and symbol in sector_mappings:
                    sector = sector_mappings[symbol]
                    if sector not in sector_exposures:
                        sector_exposures[sector] = 0
                    sector_exposures[sector] += market_value / total_value

            logger.info(
                "Sector exposure calculated",
                sectors=list(sector_exposures.keys()),
                exposures=sector_exposures
            )

            return sector_exposures

        except Exception as e:
            logger.error("Error calculating sector exposure", error=str(e))
            return {}


class IntegratedRiskManager:
    """Integrated risk manager combining all risk management components"""

    def __init__(self, user_id: int, config: Dict[str, Any] = None):
        self.user_id = user_id
        self.config = config or {}

        # Initialize components
        self.base_risk_manager = RiskManager(user_id, config.get("base_risk", {}))
        self.position_sizer = PositionSizer(config.get("position_sizing", {}))
        self.stop_loss_manager = StopLossManager(config.get("stop_loss", {}))
        self.portfolio_risk_manager = PortfolioRiskManager(config.get("portfolio_risk", {}))

        # Metrics
        self.metrics = PrometheusMetrics()

    def calculate_optimal_position_size(
        self,
        symbol: str,
        entry_price: float,
        account_value: float,
        strategy_confidence: float = 0.7,
        historical_performance: Dict[str, float] = None,
        volatility: float = None,
        method: str = "kelly"
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate optimal position size using specified method"""
        try:
            start_time = datetime.now()

            if method == "kelly" and historical_performance:
                win_rate = historical_performance.get("win_rate", 0.6)
                avg_win = historical_performance.get("avg_win", 0.05)
                avg_loss = historical_performance.get("avg_loss", 0.03)

                position_size, details = self.position_sizer.kelly_criterion(
                    account_value=account_value,
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=abs(avg_loss),
                    confidence_multiplier=strategy_confidence
                )

            elif method == "volatility" and volatility:
                base_fraction = self.position_sizer.fixed_fraction_default
                position_size, details = self.position_sizer.volatility_adjusted(
                    account_value=account_value,
                    base_fraction=base_fraction,
                    volatility=volatility
                )

            else:  # Default to fixed fractional
                position_size, details = self.position_sizer.fixed_fractional(
                    account_value=account_value,
                    confidence_multiplier=strategy_confidence
                )

            # Apply base risk manager constraints
            base_limit = self.base_risk_manager.max_position_size
            final_position_size = min(position_size, base_limit)

            # Calculate shares
            shares = int(final_position_size / entry_price)
            final_dollar_amount = shares * entry_price

            details.update({
                "symbol": symbol,
                "entry_price": entry_price,
                "shares": shares,
                "final_dollar_amount": final_dollar_amount,
                "base_limit_applied": final_position_size != position_size
            })

            # Record metrics
            calculation_time = (datetime.now() - start_time).total_seconds()
            self.metrics.record_position_sizing_calculation(
                calculation_time, method, symbol
            )

            logger.info(
                "Optimal position size calculated",
                symbol=symbol,
                position_size=final_dollar_amount,
                shares=shares,
                method=method,
                user_id=self.user_id
            )

            return final_dollar_amount, details

        except Exception as e:
            logger.error(
                "Error calculating optimal position size",
                error=str(e),
                symbol=symbol,
                user_id=self.user_id
            )
            # Fallback to base risk manager
            fallback_size = self.base_risk_manager.calculate_position_size(
                symbol, entry_price, account_value, volatility, strategy_confidence
            )
            return fallback_size, {"method": "fallback", "error": str(e)}

    def calculate_comprehensive_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        market_data: Dict[str, Any] = None,
        method: str = "atr"
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate stop loss using specified method"""
        try:
            if method == "atr" and market_data:
                highs = market_data.get("highs", [])
                lows = market_data.get("lows", [])
                closes = market_data.get("closes", [])

                if len(highs) >= 14 and len(lows) >= 14 and len(closes) >= 14:
                    return self.stop_loss_manager.atr_based_stop_loss(
                        entry_price, side, highs, lows, closes
                    )

            elif method == "pattern" and market_data:
                support_resistance = market_data.get("support_resistance_levels", [])
                if support_resistance:
                    return self.stop_loss_manager.pattern_based_stop_loss(
                        entry_price, side, support_resistance
                    )

            # Default to percentage-based
            stop_percentage = self.base_risk_manager.stop_loss_percentage
            return self.stop_loss_manager.percentage_based_stop_loss(
                entry_price, side, stop_percentage
            )

        except Exception as e:
            logger.error(
                "Error calculating comprehensive stop loss",
                error=str(e),
                symbol=symbol,
                user_id=self.user_id
            )
            # Fallback to base risk manager
            fallback_stop = self.base_risk_manager.calculate_stop_loss(entry_price, side)
            return fallback_stop, {"method": "fallback", "error": str(e)}

    def comprehensive_risk_assessment(
        self,
        positions: List[Dict[str, Any]],
        account_value: float,
        daily_pnl: float,
        price_history: Dict[str, List[float]] = None,
        sector_mappings: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive portfolio risk assessment"""
        try:
            start_time = datetime.now()

            # Base risk assessment
            base_risk = self.base_risk_manager.get_risk_summary(
                positions, account_value, daily_pnl
            )

            # Portfolio VaR calculation
            var_dollar = 0.0
            var_details = {}
            if price_history:
                var_dollar, var_details = self.portfolio_risk_manager.calculate_portfolio_var(
                    positions, price_history, account_value
                )

            # Correlation analysis
            correlation_valid = True
            correlation_violations = []
            if price_history and len(positions) > 1:
                symbols = [pos.get("symbol") for pos in positions if pos.get("symbol")]
                correlation_valid, correlation_violations = \
                    self.portfolio_risk_manager.check_correlation_limits(
                        positions, price_history
                    )

            # Sector exposure analysis
            sector_exposures = {}
            sector_violations = []
            if sector_mappings:
                sector_exposures = self.portfolio_risk_manager.calculate_sector_exposure(
                    positions, sector_mappings
                )

                # Check sector limits
                max_sector_exposure = self.portfolio_risk_manager.max_sector_exposure
                for sector, exposure in sector_exposures.items():
                    if exposure > max_sector_exposure:
                        sector_violations.append(
                            f"Sector {sector} exposure {exposure:.2%} exceeds limit {max_sector_exposure:.2%}"
                        )

            # Compile comprehensive assessment
            comprehensive_assessment = {
                **base_risk,
                "var_analysis": {
                    "var_dollar": var_dollar,
                    "var_pct_of_portfolio": var_dollar / account_value if account_value > 0 else 0,
                    **var_details
                },
                "correlation_analysis": {
                    "correlation_compliant": correlation_valid,
                    "correlation_violations": correlation_violations
                },
                "sector_analysis": {
                    "sector_exposures": sector_exposures,
                    "sector_violations": sector_violations
                },
                "risk_score": self._calculate_risk_score(
                    base_risk, var_dollar, account_value,
                    correlation_violations, sector_violations
                ),
                "assessment_timestamp": datetime.now().isoformat()
            }

            # Record metrics
            assessment_time = (datetime.now() - start_time).total_seconds()
            self.metrics.record_risk_assessment_latency(
                assessment_time, "comprehensive", str(self.user_id)
            )

            logger.info(
                "Comprehensive risk assessment completed",
                risk_score=comprehensive_assessment["risk_score"],
                var_dollar=var_dollar,
                user_id=self.user_id
            )

            return comprehensive_assessment

        except Exception as e:
            logger.error(
                "Error in comprehensive risk assessment",
                error=str(e),
                user_id=self.user_id
            )
            # Fallback to base risk assessment
            return {
                **self.base_risk_manager.get_risk_summary(positions, account_value, daily_pnl),
                "error": f"Comprehensive assessment failed: {str(e)}"
            }

    def _calculate_risk_score(
        self,
        base_risk: Dict[str, Any],
        var_dollar: float,
        account_value: float,
        correlation_violations: List[str],
        sector_violations: List[str]
    ) -> float:
        """Calculate overall portfolio risk score (0-100, higher = riskier)"""
        try:
            score = 0.0

            # Base risk violations (0-30 points)
            risk_violations = base_risk.get("risk_violations", [])
            score += min(len(risk_violations) * 10, 30)

            # VaR impact (0-25 points)
            if account_value > 0:
                var_pct = var_dollar / account_value
                score += min(var_pct * 100, 25)

            # Correlation violations (0-25 points)
            score += min(len(correlation_violations) * 5, 25)

            # Sector violations (0-20 points)
            score += min(len(sector_violations) * 10, 20)

            return min(score, 100.0)

        except Exception as e:
            logger.error("Error calculating risk score", error=str(e))
            return 50.0  # Default moderate risk