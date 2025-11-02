"""
Learning Workflow for overnight trade analysis.
"""
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from app.agents.workflows.state_schemas import LearningState
from app.agents.learning_agent import LearningAgent
import structlog

logger = structlog.get_logger(__name__)


class LearningWorkflow:
    """
    Learning workflow processes completed trades overnight.

    Simpler than trading workflow - just calls Learning Agent
    to analyze trades and extract insights.
    """

    def __init__(self):
        """Initialize learning workflow."""
        self.learning_agent = LearningAgent()
        self.graph = self._build_graph()

        logger.info("LearningWorkflow initialized")

    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine."""
        workflow = StateGraph(LearningState)

        # Add single node for learning
        workflow.add_node("learning", self._learning_node)

        # Simple linear flow
        workflow.add_edge(START, "learning")
        workflow.add_edge("learning", END)

        # Compile (no checkpointer needed for batch job)
        return workflow.compile()

    async def _learning_node(self, state: LearningState) -> Dict[str, Any]:
        """Call Learning Agent."""
        try:
            logger.info(
                "Calling Learning Agent",
                trades_count=len(state["completed_trades"])
            )

            result = await self.learning_agent.invoke({
                "completed_trades": state["completed_trades"]
            })

            # Mark as completed
            result["completed"] = True

            return result

        except Exception as e:
            logger.error("Learning Agent failed", error=str(e))
            # Return error state instead of raising
            return {
                "learning_summary": f"Learning failed: {str(e)}",
                "patterns_updated": 0,
                "regime_matrix_updated": False,
                "insights": [],
                "next_day_recommendations": [],
                "completed": True
            }

    async def run(self, initial_state: LearningState) -> Dict[str, Any]:
        """
        Run learning workflow.

        Args:
            initial_state: Initial state with completed_trades

        Returns:
            Final state with learning insights
        """
        logger.info(
            "Starting learning workflow",
            trades_count=len(initial_state["completed_trades"])
        )

        result = await self.graph.ainvoke(initial_state)

        logger.info(
            "Learning workflow complete",
            patterns_updated=result["patterns_updated"],
            insights_count=len(result.get("insights", []))
        )

        return result
