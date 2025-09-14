"""
Risk Assessment Tool for evaluating trading risks
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from app.core.exceptions import TradingError
from app.trading.risk_manager import RiskManager

from .base_tool import AgentTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class RiskAssessmentTool(AgentTool):
    """Tool for assessing trading risks using risk management patterns"""

    def __init__(self):
        super().__init__(
            name="risk_assessment",
            description="Assess trading risks including position sizing, exposure, and risk limits",
        )
        self.category = "risk_management"

    def get_parameters(self) -> List[ToolParameter]:
        """Get tool parameter definitions"""
        return [
            ToolParameter(
                name="action",
                type="str",
                description="Action to perform: 'validate_order', 'calculate_position_size', 'check_limits', 'risk_summary'",
                required=True,
            ),
            ToolParameter(
                name="user_id",
                type="int",
                description="User ID for risk assessment",
                required=True,
            ),
            ToolParameter(
                name="symbol",
                type="str",
                description="Stock symbol (required for order validation and position sizing)",
                required=False,
            ),
            ToolParameter(
                name="quantity",
                type="float",
                description="Order quantity (required for order validation)",
                required=False,
            ),
            ToolParameter(
                name="price",
                type="float",
                description="Stock price (required for order validation and position sizing)",
                required=False,
            ),
            ToolParameter(
                name="side",
                type="str",
                description="Order side: 'BUY' or 'SELL' (required for order validation)",
                required=False,
            ),
            ToolParameter(
                name="current_positions",
                type="list",
                description="List of current positions with market_value and symbol",
                required=False,
                default=[],
            ),
            ToolParameter(
                name="account_value",
                type="float",
                description="Total account value",
                required=False,
            ),
            ToolParameter(
                name="daily_pnl",
                type="float",
                description="Today's profit/loss",
                required=False,
                default=0.0,
            ),
            ToolParameter(
                name="volatility",
                type="float",
                description="Stock volatility for position sizing",
                required=False,
            ),
            ToolParameter(
                name="confidence",
                type="float",
                description="Strategy confidence (0.0 to 1.0) for position sizing",
                required=False,
            ),
            ToolParameter(
                name="stop_loss_price",
                type="float",
                description="Stop loss price for risk-based position sizing",
                required=False,
            ),
            ToolParameter(
                name="risk_params",
                type="dict",
                description="Custom risk parameters (optional)",
                required=False,
            ),
        ]

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute risk assessment"""
        try:
            action = parameters["action"].lower()
            user_id = parameters["user_id"]
            risk_params = parameters.get("risk_params", {})

            # Initialize risk manager
            risk_manager = RiskManager(user_id, risk_params)

            if action == "validate_order":
                return await self._validate_order(risk_manager, parameters)
            elif action == "calculate_position_size":
                return await self._calculate_position_size(risk_manager, parameters)
            elif action == "check_limits":
                return await self._check_risk_limits(risk_manager, parameters)
            elif action == "risk_summary":
                return await self._get_risk_summary(risk_manager, parameters)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown action: {action}. Supported: 'validate_order', 'calculate_position_size', 'check_limits', 'risk_summary'",
                )

        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return ToolResult(
                success=False, data=None, error=f"Risk assessment failed: {str(e)}"
            )

    async def _validate_order(
        self, risk_manager: RiskManager, parameters: Dict[str, Any]
    ) -> ToolResult:
        """Validate order against risk limits"""
        try:
            # Check required parameters
            required_params = ["symbol", "quantity", "price", "side", "account_value"]
            missing_params = [
                p
                for p in required_params
                if p not in parameters or parameters[p] is None
            ]
            if missing_params:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Missing required parameters for order validation: {missing_params}",
                )

            symbol = parameters["symbol"]
            quantity = float(parameters["quantity"])
            price = float(parameters["price"])
            side = parameters["side"].upper()
            current_positions = parameters.get("current_positions", [])
            account_value = float(parameters["account_value"])
            daily_pnl = float(parameters.get("daily_pnl", 0.0))

            # Validate order
            is_valid, reason = risk_manager.validate_order(
                symbol=symbol,
                quantity=quantity,
                price=price,
                side=side,
                current_positions=current_positions,
                account_value=account_value,
                daily_pnl=daily_pnl,
            )

            result_data = {
                "is_valid": is_valid,
                "reason": reason,
                "order_details": {
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": price,
                    "side": side,
                    "order_value": quantity * price,
                },
            }

            metadata = {
                "action": "validate_order",
                "user_id": risk_manager.user_id,
                "validated_at": datetime.now().isoformat(),
            }

            return ToolResult(success=True, data=result_data, metadata=metadata)

        except Exception as e:
            raise TradingError(f"Order validation failed: {str(e)}")

    async def _calculate_position_size(
        self, risk_manager: RiskManager, parameters: Dict[str, Any]
    ) -> ToolResult:
        """Calculate optimal position size"""
        try:
            # Check required parameters
            required_params = ["symbol", "price", "account_value"]
            missing_params = [
                p
                for p in required_params
                if p not in parameters or parameters[p] is None
            ]
            if missing_params:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Missing required parameters for position sizing: {missing_params}",
                )

            symbol = parameters["symbol"]
            price = float(parameters["price"])
            account_value = float(parameters["account_value"])
            volatility = parameters.get("volatility")
            confidence = parameters.get("confidence")
            stop_loss_price = parameters.get("stop_loss_price")

            if volatility is not None:
                volatility = float(volatility)
            if confidence is not None:
                confidence = float(confidence)
            if stop_loss_price is not None:
                stop_loss_price = float(stop_loss_price)

            # Calculate position size
            position_size = risk_manager.calculate_position_size(
                symbol=symbol,
                price=price,
                account_value=account_value,
                volatility=volatility,
                confidence=confidence,
                stop_loss_price=stop_loss_price,
            )

            # Calculate number of shares
            shares = int(position_size / price) if position_size > 0 else 0
            actual_position_size = shares * price

            # Calculate suggested stop loss and take profit
            suggested_stop_loss = risk_manager.calculate_stop_loss(price, "BUY")
            suggested_take_profit = risk_manager.calculate_take_profit(price, "BUY")

            result_data = {
                "recommended_position_size": position_size,
                "recommended_shares": shares,
                "actual_position_size": actual_position_size,
                "price_per_share": price,
                "account_percentage": (
                    (actual_position_size / account_value) * 100
                    if account_value > 0
                    else 0
                ),
                "suggested_stop_loss": suggested_stop_loss,
                "suggested_take_profit": suggested_take_profit,
                "max_risk_amount": actual_position_size
                * risk_manager.stop_loss_percentage,
                "inputs": {
                    "symbol": symbol,
                    "price": price,
                    "account_value": account_value,
                    "volatility": volatility,
                    "confidence": confidence,
                    "stop_loss_price": stop_loss_price,
                },
            }

            metadata = {
                "action": "calculate_position_size",
                "user_id": risk_manager.user_id,
                "calculated_at": datetime.now().isoformat(),
            }

            return ToolResult(success=True, data=result_data, metadata=metadata)

        except Exception as e:
            raise TradingError(f"Position size calculation failed: {str(e)}")

    async def _check_risk_limits(
        self, risk_manager: RiskManager, parameters: Dict[str, Any]
    ) -> ToolResult:
        """Check current risk limit violations"""
        try:
            current_positions = parameters.get("current_positions", [])
            account_value = float(
                parameters.get("account_value", 100000)
            )  # Default value
            daily_pnl = float(parameters.get("daily_pnl", 0.0))

            violations = risk_manager.check_risk_limits(
                current_positions, account_value, daily_pnl
            )

            result_data = {
                "violations": violations,
                "has_violations": len(violations) > 0,
                "violation_count": len(violations),
                "status": "RISK_VIOLATION" if violations else "OK",
            }

            metadata = {
                "action": "check_risk_limits",
                "user_id": risk_manager.user_id,
                "checked_at": datetime.now().isoformat(),
            }

            return ToolResult(success=True, data=result_data, metadata=metadata)

        except Exception as e:
            raise TradingError(f"Risk limit check failed: {str(e)}")

    async def _get_risk_summary(
        self, risk_manager: RiskManager, parameters: Dict[str, Any]
    ) -> ToolResult:
        """Get comprehensive risk summary"""
        try:
            current_positions = parameters.get("current_positions", [])
            account_value = float(
                parameters.get("account_value", 100000)
            )  # Default value
            daily_pnl = float(parameters.get("daily_pnl", 0.0))

            risk_summary = risk_manager.get_risk_summary(
                current_positions, account_value, daily_pnl
            )

            metadata = {
                "action": "risk_summary",
                "user_id": risk_manager.user_id,
                "generated_at": datetime.now().isoformat(),
            }

            return ToolResult(success=True, data=risk_summary, metadata=metadata)

        except Exception as e:
            raise TradingError(f"Risk summary generation failed: {str(e)}")
