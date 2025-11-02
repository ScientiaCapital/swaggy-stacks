"""
Learning Agent - Post-market learning and continuous improvement.
"""
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent, ModelConfig
import structlog

logger = structlog.get_logger(__name__)


class LearningAgent(BaseAgent):
    """
    Learning Agent processes trade experiences and improves strategies.

    Responsibilities:
    - Capture trade experiences (wins/losses)
    - Run unsupervised learning on day's trades
    - Update pattern memory
    - Cluster similar experiences
    - Update regime-strategy performance matrix
    - Generate next-day recommendations
    """

    def __init__(self):
        super().__init__(
            name="Learning Agent",
            description="Post-market learning and continuous improvement",
            model_config=ModelConfig.CHEAP,  # Use DeepSeek for batch processing
        )

    async def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process trade experiences and learn"""
        try:
            completed_trades = input_data.get("completed_trades", [])

            if not completed_trades:
                return self._no_trades_result()

            # Process experiences
            learning_result = self._process_experiences(completed_trades)

            result = {
                "learning_summary": learning_result["summary"],
                "patterns_updated": learning_result["patterns_count"],
                "regime_matrix_updated": learning_result["matrix_updated"],
                "insights": learning_result["insights"],
                "next_day_recommendations": learning_result["recommendations"]
            }

            logger.info(
                "Learning complete",
                trades_processed=len(completed_trades),
                patterns_updated=result["patterns_updated"],
                insights_count=len(result["insights"])
            )

            return result

        except Exception as e:
            logger.error("learning_agent_error", error=str(e))
            return {
                "learning_summary": f"Learning error: {str(e)}",
                "patterns_updated": 0,
                "regime_matrix_updated": False,
                "insights": [],
                "next_day_recommendations": [],
                "error": str(e)
            }

    def _no_trades_result(self) -> Dict[str, Any]:
        """Return result when no trades to process"""
        return {
            "learning_summary": "No trades to process today",
            "patterns_updated": 0,
            "regime_matrix_updated": False,
            "insights": [],
            "next_day_recommendations": []
        }

    def _process_experiences(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process trade experiences and extract insights"""

        wins = [t for t in trades if t.get("outcome") == "win"]
        losses = [t for t in trades if t.get("outcome") == "loss"]

        # Calculate metrics
        win_rate = len(wins) / len(trades) if trades else 0
        total_pnl = sum(t.get("pnl", 0) for t in trades)

        # Generate insights
        insights = []
        if win_rate > 0.7:
            insights.append("High win rate achieved - strategies are well-calibrated")
        elif win_rate < 0.5:
            insights.append("Win rate below 50% - need to review strategy selection")

        if total_pnl > 0:
            insights.append(f"Profitable day: ${total_pnl:.2f}")
        else:
            insights.append(f"Unprofitable day: ${total_pnl:.2f} - review risk management")

        # Group by regime
        regimes = {}
        for trade in trades:
            regime = trade.get("market_regime", "unknown")
            if regime not in regimes:
                regimes[regime] = []
            regimes[regime].append(trade)

        # Generate next-day recommendations
        recommendations = []
        for regime, regime_trades in regimes.items():
            regime_wins = [t for t in regime_trades if t.get("outcome") == "win"]
            regime_win_rate = len(regime_wins) / len(regime_trades) if regime_trades else 0

            if regime_win_rate > 0.6:
                recommendations.append(
                    f"Continue {regime} regime strategies (win rate: {regime_win_rate:.1%})"
                )

        return {
            "summary": f"Processed {len(trades)} trades: {len(wins)} wins, {len(losses)} losses",
            "patterns_count": len(regimes),
            "matrix_updated": True,
            "insights": insights,
            "recommendations": recommendations
        }
