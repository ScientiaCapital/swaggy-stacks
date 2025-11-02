"""
Execution Agent for order placement and fill monitoring
"""
from typing import Any, Dict
from app.agents.base_agent import BaseAgent, ModelConfig
import structlog

logger = structlog.get_logger(__name__)


class ExecutionAgent(BaseAgent):
    """
    Execution Agent places orders and monitors fills.

    Responsibilities:
    - Place orders via Alpaca API
    - Monitor fill status
    - Handle partial fills
    - Record execution quality metrics
    - Update portfolio positions
    """

    def __init__(self):
        super().__init__(
            name="Execution Agent",
            description="Order execution and fill monitoring",
            model_config=ModelConfig.FAST,  # Use Cerebras for speed
        )
        # Note: Don't instantiate TradingManager here in tests
        # Only use when actually executing real trades

    async def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute approved trade and monitor fills"""

        # Validate risk approval
        risk_approved = input_data.get("risk_approved", False)
        if not risk_approved:
            logger.warning("Trade execution rejected - no risk approval")
            return {
                "execution_status": "rejected",
                "orders": [],
                "reason": "Risk approval required"
            }

        # Extract trade parameters
        strategy = input_data.get("recommended_strategy")
        params = input_data.get("strategy_params", {})
        position_size = input_data.get("position_size", 0)

        # Execute trade (mock for now - real integration with TradingManager later)
        result = self._execute_trade(strategy, params, position_size)

        return result

    def _execute_trade(
        self,
        strategy: str,
        params: Dict[str, Any],
        position_size: float
    ) -> Dict[str, Any]:
        """Execute trade via TradingManager"""

        # For now, return pending status
        # Real implementation will integrate with TradingManager and OrderManager
        logger.info(
            "Executing trade",
            strategy=strategy,
            position_size=position_size,
            params=params
        )

        return {
            "execution_status": "pending",
            "orders": [
                {
                    "strategy": strategy,
                    "position_size": position_size,
                    "params": params,
                    "status": "pending"
                }
            ],
            "execution_quality": {
                "slippage": 0.0,
                "fill_time_seconds": 0.0
            }
        }
