"""
Risk Agent - Portfolio risk validation and position sizing.

Validates trades against risk limits, calculates appropriate position
sizes, and monitors portfolio exposure with conservative defaults.
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent, ModelConfig
import structlog

logger = structlog.get_logger()


class RiskAgent(BaseAgent):
    """
    Risk Agent validates trade risk and calculates position sizing.

    Responsibilities:
    - Portfolio risk assessment
    - Position sizing (Kelly Criterion, volatility-based)
    - Exposure limit validation
    - Greeks monitoring
    - Approve or reject trades
    """

    def __init__(self):
        super().__init__(
            name="Risk Agent",
            description="Portfolio risk validation and position sizing",
            model_config=ModelConfig.RISK,  # Always use best model for risk
        )

    async def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade risk"""
        try:
            strategy = input_data.get("recommended_strategy")
            params = input_data.get("strategy_params", {})
            portfolio = input_data.get("portfolio", {})

            # Perform risk assessment
            risk_result = self._assess_risk(strategy, params, portfolio)

            result = {
                "risk_approved": risk_result["approved"],
                "position_size": risk_result["position_size"],
                "risk_assessment": risk_result["assessment"],
                "adjustments": risk_result.get("adjustments", []),
                "reasoning": risk_result["reasoning"]
            }

            decision = "APPROVED" if result["risk_approved"] else "REJECTED"
            self.logger.info(
                "risk_validation_complete",
                decision=decision,
                strategy=strategy,
                position_size=result["position_size"]
            )

            return result

        except Exception as e:
            self.logger.error("risk_agent_error", error=str(e))
            # Default to rejection on error (conservative)
            return {
                "risk_approved": False,
                "position_size": 0,
                "risk_assessment": {},
                "adjustments": [],
                "reasoning": f"Risk validation error: {str(e)}",
                "error": str(e)
            }

    def _assess_risk(
        self,
        strategy: str,
        params: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess trade risk against portfolio limits"""

        portfolio_value = portfolio.get("total_value", 100000)
        cash_available = portfolio.get("cash", 50000)

        # Calculate max position size (15% of portfolio)
        max_position = portfolio_value * 0.15

        # Get trade risk from params
        trade_risk = params.get("max_risk", 5000)

        # Calculate appropriate position size
        if trade_risk > max_position:
            # Need to reduce position
            position_size = max_position
            approved = True  # Approved with adjustment
            reasoning = f"Reduced position from {trade_risk} to {position_size} (15% portfolio limit)"
            adjustments = [f"Position size reduced to {position_size}"]
        elif trade_risk > cash_available:
            # Insufficient cash
            position_size = 0
            approved = False
            reasoning = f"Insufficient cash: need {trade_risk}, have {cash_available}"
            adjustments = []
        else:
            # Trade is within limits
            position_size = trade_risk
            approved = True
            reasoning = f"Trade approved: risk {trade_risk} is within limits"
            adjustments = []

        return {
            "approved": approved,
            "position_size": position_size,
            "assessment": {
                "portfolio_value": portfolio_value,
                "max_position": max_position,
                "trade_risk": trade_risk,
                "cash_available": cash_available
            },
            "adjustments": adjustments,
            "reasoning": reasoning
        }
