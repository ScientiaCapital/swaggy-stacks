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
        self.anomaly_detector = None

    def set_anomaly_detector(self, anomaly_detector) -> None:
        """
        Set anomaly detector for risk adjustment integration.
        
        Args:
            anomaly_detector: Instance of AnomalyDetector
        """
        self.anomaly_detector = anomaly_detector
        logger.info(
            "Anomaly detector integrated with risk manager",
            user_id=self.user_id,
            detector_fitted=getattr(anomaly_detector, 'is_fitted', False)
        )

    def update_risk_from_anomalies(self, market_data: Optional[Union[pd.DataFrame, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Update risk parameters based on current anomaly detection.
        
        Args:
            market_data: Recent market data for anomaly analysis
            
        Returns:
            Dictionary with anomaly risk assessment and adjustments made
        """
        if not self.anomaly_detector or not self.anomaly_risk_adjustment_enabled:
            return {
                'anomaly_system_active': False,
                'risk_adjustments_made': False,
                'message': 'Anomaly detection not available or disabled'
            }

        try:
            # Get anomaly risk integration data
            anomaly_data = self.anomaly_detector.get_risk_integration_data()
            
            if not anomaly_data.get('anomaly_system_active', False):
                return {
                    'anomaly_system_active': False,
                    'risk_adjustments_made': False,
                    'message': 'Anomaly detector not fitted or active'
                }

            # Update current anomaly level
            new_anomaly_level = anomaly_data.get('alert_level', 'low')
            previous_level = self.current_anomaly_level
            self.current_anomaly_level = new_anomaly_level
            
            # Calculate risk multiplier
            risk_multiplier = self.anomaly_risk_multipliers.get(new_anomaly_level, 1.0)
            
            # Apply risk adjustments
            adjustments_made = self._apply_anomaly_risk_adjustments(
                risk_multiplier, 
                anomaly_data.get('recommended_position_adjustment', 0.0)
            )
            
            # Update tracking
            self.risk_adjustment_active = risk_multiplier > 1.0
            self.last_anomaly_check = pd.Timestamp.now()
            
            result = {
                'anomaly_system_active': True,
                'risk_adjustments_made': adjustments_made,
                'previous_anomaly_level': previous_level,
                'current_anomaly_level': new_anomaly_level,
                'risk_multiplier': risk_multiplier,
                'recommended_position_adjustment': anomaly_data.get('recommended_position_adjustment', 0.0),
                'recent_anomaly_rate': anomaly_data.get('recent_anomaly_rate', 0.0),
                'active_alerts': anomaly_data.get('active_alerts', 0),
                'last_detection_time': anomaly_data.get('last_detection_time'),
                'adjustments': self._get_current_risk_adjustments()
            }
            
            if adjustments_made or new_anomaly_level != previous_level:
                logger.info(
                    "Risk parameters updated from anomaly detection",
                    user_id=self.user_id,
                    previous_level=previous_level,
                    new_level=new_anomaly_level,
                    risk_multiplier=risk_multiplier,
                    adjustments_made=adjustments_made
                )
            
            return result
            
        except Exception as e:
            logger.error(
                "Error updating risk from anomaly detection",
                user_id=self.user_id,
                error=str(e)
            )
            return {
                'anomaly_system_active': False,
                'risk_adjustments_made': False,
                'error': str(e)
            }

    def _apply_anomaly_risk_adjustments(self, risk_multiplier: float, position_adjustment: float) -> bool:
        """
        Apply risk parameter adjustments based on anomaly level.
        
        Args:
            risk_multiplier: Multiplier for risk constraints (>1.0 = more conservative)
            position_adjustment: Recommended position size adjustment (-1.0 to 1.0)
            
        Returns:
            bool: True if any adjustments were made
        """
        adjustments_made = False
        
        # Calculate new risk parameters
        new_max_position_size = self.base_max_position_size / risk_multiplier
        new_max_portfolio_exposure = min(
            self.base_max_portfolio_exposure / risk_multiplier,
            0.95  # Never exceed 95% even with adjustments
        )
        new_max_single_stock_exposure = min(
            self.base_max_single_stock_exposure / risk_multiplier,
            0.30  # Never exceed 30% per stock
        )
        new_max_daily_loss = self.base_max_daily_loss / risk_multiplier
        
        # Apply position adjustment
        if position_adjustment < 0:  # Reduce positions
            adjustment_factor = 1.0 + position_adjustment  # position_adjustment is negative
            new_max_position_size *= adjustment_factor
            new_max_portfolio_exposure *= adjustment_factor
            new_max_single_stock_exposure *= adjustment_factor
        
        # Update parameters if they've changed significantly
        tolerance = 0.01  # 1% tolerance
        
        if abs(self.max_position_size - new_max_position_size) / self.base_max_position_size > tolerance:
            self.max_position_size = new_max_position_size
            adjustments_made = True
            
        if abs(self.max_portfolio_exposure - new_max_portfolio_exposure) / self.base_max_portfolio_exposure > tolerance:
            self.max_portfolio_exposure = new_max_portfolio_exposure
            adjustments_made = True
            
        if abs(self.max_single_stock_exposure - new_max_single_stock_exposure) / self.base_max_single_stock_exposure > tolerance:
            self.max_single_stock_exposure = new_max_single_stock_exposure
            adjustments_made = True
            
        if abs(self.max_daily_loss - new_max_daily_loss) / self.base_max_daily_loss > tolerance:
            self.max_daily_loss = new_max_daily_loss
            adjustments_made = True
        
        return adjustments_made

    def _get_current_risk_adjustments(self) -> Dict[str, Dict[str, float]]:
        """Get current risk parameter adjustments vs base values."""
        return {
            'max_position_size': {
                'base': self.base_max_position_size,
                'current': self.max_position_size,
                'adjustment_factor': self.max_position_size / self.base_max_position_size
            },
            'max_portfolio_exposure': {
                'base': self.base_max_portfolio_exposure,
                'current': self.max_portfolio_exposure,
                'adjustment_factor': self.max_portfolio_exposure / self.base_max_portfolio_exposure
            },
            'max_single_stock_exposure': {
                'base': self.base_max_single_stock_exposure,
                'current': self.max_single_stock_exposure,
                'adjustment_factor': self.max_single_stock_exposure / self.base_max_single_stock_exposure
            },
            'max_daily_loss': {
                'base': self.base_max_daily_loss,
                'current': self.max_daily_loss,
                'adjustment_factor': self.max_daily_loss / self.base_max_daily_loss
            }
        }

    def reset_risk_parameters(self) -> None:
        """Reset risk parameters to their base values."""
        self.max_position_size = self.base_max_position_size
        self.max_daily_loss = self.base_max_daily_loss
        self.max_portfolio_exposure = self.base_max_portfolio_exposure
        self.max_single_stock_exposure = self.base_max_single_stock_exposure
        
        self.current_anomaly_level = 'low'
        self.risk_adjustment_active = False
        
        logger.info(
            "Risk parameters reset to base values",
            user_id=self.user_id
        )
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

    # Store original risk parameters for dynamic adjustment
    self.base_max_position_size = self.max_position_size
    self.base_max_daily_loss = self.max_daily_loss
    self.base_max_portfolio_exposure = self.max_portfolio_exposure
    self.base_max_single_stock_exposure = self.max_single_stock_exposure

    # Anomaly detection integration
    self.anomaly_detector = None
    self.anomaly_risk_adjustment_enabled = self.risk_params.get(
        "anomaly_risk_adjustment_enabled", True
    )
    self.anomaly_risk_multipliers = {
        'low': 1.0,
        'medium': 1.2,
        'high': 1.5,
        'critical': 2.0
    }
    
    # Risk state tracking
    self.current_anomaly_level = 'low'
    self.risk_adjustment_active = False
    self.last_anomaly_check = None

    logger.info(
        "Risk manager initialized", 
        user_id=user_id, 
        risk_params=self.risk_params,
        anomaly_adjustment_enabled=self.anomaly_risk_adjustment_enabled
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
    market_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
) -> Tuple[bool, str]:
    """
    Validate if an order meets risk management criteria

    Args:
        symbol: Stock symbol
        quantity: Order quantity
        price: Order price
        side: Order side (BUY/SELL)
        current_positions: Current portfolio positions
        account_value: Total account value
        daily_pnl: Daily P&L
        market_data: Recent market data for anomaly analysis

    Returns:
        Tuple[bool, str]: (is_valid, reason)
    """
        try:
            # Update risk parameters based on current anomaly conditions
            anomaly_update = self.update_risk_from_anomalies(market_data)

            # Calculate order value
            order_value = quantity * price

            # Check maximum position size (now adjusted for anomalies)
            if order_value > self.max_position_size:
            reason = f"Order value ${order_value:.2f} exceeds maximum position size ${self.max_position_size:.2f}"
            if self.risk_adjustment_active:
                reason += f" (adjusted for {self.current_anomaly_level} anomaly level)"
            return False, reason

        # Check daily loss limit (now adjusted for anomalies)
        if daily_pnl < -self.max_daily_loss:
            reason = f"Daily loss ${abs(daily_pnl):.2f} exceeds limit ${self.max_daily_loss:.2f}"
            if self.risk_adjustment_active:
                reason += f" (tightened due to {self.current_anomaly_level} anomaly level)"
            return False, reason

        # Check portfolio exposure (now adjusted for anomalies)
        current_exposure = self._calculate_portfolio_exposure(
            current_positions, account_value
        )
        new_exposure = current_exposure + (order_value / account_value)

        if new_exposure > self.max_portfolio_exposure:
            reason = f"Portfolio exposure {new_exposure:.2%} would exceed limit {self.max_portfolio_exposure:.2%}"
            if self.risk_adjustment_active:
                reason += f" (reduced due to {self.current_anomaly_level} anomaly level)"
            return False, reason

        # Check single stock exposure (now adjusted for anomalies)
        current_stock_exposure = self._calculate_stock_exposure(
            symbol, current_positions, account_value
        )
        new_stock_exposure = current_stock_exposure + (order_value / account_value)

        if new_stock_exposure > self.max_single_stock_exposure:
            reason = f"Stock exposure {new_stock_exposure:.2%} would exceed limit {self.max_single_stock_exposure:.2%}"
            if self.risk_adjustment_active:
                reason += f" (reduced due to {self.current_anomaly_level} anomaly level)"
            return False, reason

        # Additional anomaly-specific checks
        if anomaly_update.get('anomaly_system_active', False):
            # Block new positions during critical anomalies
            if self.current_anomaly_level == 'critical' and side.upper() == 'BUY':
                return False, "New buy orders blocked due to critical market anomalies detected"
            
            # Reduce position sizes during high anomaly periods
            if self.current_anomaly_level in ['high', 'critical']:
                recommended_adjustment = anomaly_update.get('recommended_position_adjustment', 0.0)
                if recommended_adjustment < -0.1:  # More than 10% reduction recommended
                    max_recommended_value = order_value * (1 + recommended_adjustment)
                    if order_value > max_recommended_value:
                        return False, f"Order size should be reduced by {abs(recommended_adjustment)*100:.0f}% due to detected market anomalies"

        # Check if we have enough buying power
        if (
            side.upper() == "BUY" and order_value > account_value * 0.95
        ):  # Leave 5% buffer
            return False, "Insufficient buying power for order"

        # Log successful validation with anomaly context
        logger.info(
            "Order validation passed",
            symbol=symbol,
            quantity=quantity,
            price=price,
            side=side,
            order_value=order_value,
            anomaly_level=self.current_anomaly_level,
            risk_adjustment_active=self.risk_adjustment_active,
            anomaly_adjustments=anomaly_update.get('risk_adjustments_made', False)
        )

        success_message = "Order passes risk management checks"
        if self.risk_adjustment_active:
            success_message += f" (risk parameters adjusted for {self.current_anomaly_level} anomaly conditions)"

        return True, success_message

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
        """Get comprehensive risk summary including anomaly detection status"""
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

            # Get anomaly detection status if available
            anomaly_status = {}
            if self.anomaly_detector and self.anomaly_risk_adjustment_enabled:
                try:
                    anomaly_data = self.anomaly_detector.get_risk_integration_data()
                    anomaly_status = {
                        'anomaly_system_active': anomaly_data.get('anomaly_system_active', False),
                        'current_anomaly_level': self.current_anomaly_level,
                        'risk_adjustment_active': self.risk_adjustment_active,
                        'recent_anomaly_rate': anomaly_data.get('recent_anomaly_rate', 0.0),
                        'active_alerts': anomaly_data.get('active_alerts', 0),
                        'last_detection_time': anomaly_data.get('last_detection_time'),
                        'recommended_position_adjustment': anomaly_data.get('recommended_position_adjustment', 0.0),
                        'current_risk_multiplier': anomaly_data.get('current_risk_multiplier', 1.0),
                        'last_anomaly_check': self.last_anomaly_check.isoformat() if self.last_anomaly_check else None
                    }

                    # Add risk parameter adjustments if active
                    if self.risk_adjustment_active:
                        anomaly_status['risk_adjustments'] = self._get_current_risk_adjustments()

                except Exception as e:
                    logger.warning("Error getting anomaly status for risk summary", error=str(e))
                    anomaly_status = {
                        'anomaly_system_active': False,
                        'error': str(e)
                    }
            else:
                anomaly_status = {
                    'anomaly_system_active': False,
                    'reason': 'Anomaly detection not configured or disabled'
                }

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
                "base_risk_parameters": {
                    "max_position_size": self.base_max_position_size,
                    "max_daily_loss": self.base_max_daily_loss,
                    "max_portfolio_exposure": self.base_max_portfolio_exposure,
                    "max_single_stock_exposure": self.base_max_single_stock_exposure,
                },
                "anomaly_detection": anomaly_status,
            }

        except Exception as e:
            logger.error(
                "Error generating risk summary", error=str(e), user_id=self.user_id
            )
            return {"error": f"Error generating risk summary: {str(e)}"}
