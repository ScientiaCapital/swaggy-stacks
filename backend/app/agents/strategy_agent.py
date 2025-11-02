"""
Strategy Agent - Options strategy selection based on market conditions.

Selects optimal options strategies by matching market regimes to
appropriate trading strategies with confidence scoring.
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent, ModelConfig
import structlog

logger = structlog.get_logger()


class StrategyAgent(BaseAgent):
    """
    Strategy Agent selects optimal options strategies based on market regime.

    Responsibilities:
    - Match strategy to market regime
    - Calculate optimal parameters (strikes, expiration)
    - Provide confidence score and rationale
    - Consider historical performance
    """

    def __init__(self):
        super().__init__(
            name="Strategy Agent",
            description="Options strategy selection based on market conditions",
            model_config=ModelConfig.PRIMARY,  # Claude Sonnet for strategy selection
        )

    async def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal options strategy"""
        try:
            market_regime = input_data.get("market_regime", "unknown")
            regime_confidence = input_data.get("regime_confidence", 0.0)
            signals = input_data.get("signals", [])

            # Select strategy based on regime
            strategy_selection = self._select_strategy(
                market_regime,
                regime_confidence,
                signals
            )

            result = {
                "recommended_strategy": strategy_selection["strategy"],
                "strategy_params": strategy_selection["params"],
                "confidence_score": strategy_selection["confidence"],
                "reasoning": strategy_selection["reasoning"],
                "alternatives": strategy_selection.get("alternatives", [])
            }

            self.logger.info(
                "strategy_selection_complete",
                strategy=result["recommended_strategy"],
                confidence=result["confidence_score"],
                regime=market_regime
            )

            return result

        except Exception as e:
            self.logger.error("strategy_agent_error", error=str(e))
            return {
                "recommended_strategy": None,
                "strategy_params": {},
                "confidence_score": 0.0,
                "reasoning": f"Error: {str(e)}",
                "error": str(e)
            }

    def _select_strategy(
        self,
        regime: str,
        confidence: float,
        signals: list
    ) -> Dict[str, Any]:
        """Select strategy based on market regime"""

        # Regime-strategy mapping with base confidence
        regime_strategies = {
            "bull": ("bull_call_spread", 0.8),
            "bear": ("bear_put_spread", 0.8),
            "volatile": ("long_straddle", 0.7),
            "sideways": ("calendar_spread", 0.75)
        }

        strategy, base_confidence = regime_strategies.get(
            regime,
            ("covered_call", 0.5)  # Default fallback
        )

        # Adjust confidence based on regime confidence
        final_confidence = base_confidence * confidence

        return {
            "strategy": strategy,
            "params": self._get_default_params(strategy),
            "confidence": final_confidence,
            "reasoning": f"Selected {strategy} for {regime} regime (confidence: {final_confidence:.2f})",
            "alternatives": []
        }

    def _get_default_params(self, strategy: str) -> Dict[str, Any]:
        """Get default parameters for strategy"""
        return {
            "symbol": "SPY",
            "quantity": 1,
            "expiration_days": 30,
            "notes": f"Default parameters for {strategy}"
        }
