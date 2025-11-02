"""
Trading Workflow using LangGraph supervisor pattern.
"""
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from app.agents.workflows.state_schemas import TradingState
from app.agents.workflows.supervisor import TradingSupervisor
from app.agents.research_agent import ResearchAgent
from app.agents.strategy_agent import StrategyAgent
from app.agents.risk_agent import RiskAgent
from app.agents.execution_agent import ExecutionAgent
import structlog

logger = structlog.get_logger(__name__)


class WorkflowError(Exception):
    """Base exception for workflow failures."""
    pass


class AgentFailureError(WorkflowError):
    """Raised when an agent fails."""
    def __init__(self, agent_name: str, error: str):
        self.agent_name = agent_name
        self.error = error
        super().__init__(f"{agent_name} failed: {error}")


class TradingWorkflow:
    """
    Trading workflow orchestrates Research → Strategy → Risk → Execution.

    Uses supervisor pattern where supervisor agent coordinates
    specialized agents as tools.
    """

    def __init__(self):
        """Initialize trading workflow with supervisor and agents."""
        self.supervisor = TradingSupervisor()

        # Initialize agents
        self.research_agent = ResearchAgent()
        self.strategy_agent = StrategyAgent()
        self.risk_agent = RiskAgent()
        self.execution_agent = ExecutionAgent()

        # Build state graph
        self.graph = self._build_graph()

        logger.info("TradingWorkflow initialized")

    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine."""
        workflow = StateGraph(TradingState)

        # Add nodes
        workflow.add_node("research", self._research_node)
        workflow.add_node("strategy", self._strategy_node)
        workflow.add_node("risk", self._risk_node)
        workflow.add_node("execution", self._execution_node)

        # Add edges
        workflow.add_edge(START, "research")
        workflow.add_edge("research", "strategy")
        workflow.add_edge("strategy", "risk")
        workflow.add_conditional_edges(
            "risk",
            self._should_execute,
            {
                "execute": "execution",
                "skip": END
            }
        )
        workflow.add_edge("execution", END)

        # Compile without checkpointer for now (checkpointer requires context manager)
        # TODO: Add proper checkpointing support with context manager
        return workflow.compile()

    async def _research_node(self, state: TradingState) -> Dict[str, Any]:
        """Call Research Agent."""
        try:
            logger.info("Calling Research Agent", symbol=state["symbol"])
            result = await self.research_agent.invoke({
                "market_data": state["market_data"]
            })
            return result
        except Exception as e:
            logger.error("Research Agent failed", error=str(e))
            raise AgentFailureError("Research Agent", str(e))

    async def _strategy_node(self, state: TradingState) -> Dict[str, Any]:
        """Call Strategy Agent."""
        try:
            logger.info("Calling Strategy Agent", regime=state.get("market_regime"))
            result = await self.strategy_agent.invoke({
                "market_regime": state["market_regime"],
                "regime_confidence": state["regime_confidence"],
                "signals": state["signals"]
            })
            return result
        except Exception as e:
            logger.error("Strategy Agent failed", error=str(e))
            raise AgentFailureError("Strategy Agent", str(e))

    async def _risk_node(self, state: TradingState) -> Dict[str, Any]:
        """Call Risk Agent."""
        try:
            logger.info("Calling Risk Agent", strategy=state.get("recommended_strategy"))
            result = await self.risk_agent.invoke({
                "recommended_strategy": state["recommended_strategy"],
                "strategy_params": state["strategy_params"],
                "portfolio": {
                    "total_value": 100000,  # Mock portfolio
                    "cash": 50000
                }
            })
            return result
        except Exception as e:
            logger.error("Risk Agent failed", error=str(e))
            raise AgentFailureError("Risk Agent", str(e))

    async def _execution_node(self, state: TradingState) -> Dict[str, Any]:
        """Call Execution Agent."""
        try:
            logger.info("Calling Execution Agent", approved=state.get("risk_approved"))
            result = await self.execution_agent.invoke({
                "risk_approved": state["risk_approved"],
                "recommended_strategy": state["recommended_strategy"],
                "strategy_params": state["strategy_params"],
                "position_size": state["position_size"]
            })
            result["completed"] = True
            return result
        except Exception as e:
            logger.error("Execution Agent failed", error=str(e))
            raise AgentFailureError("Execution Agent", str(e))

    def _should_execute(self, state: TradingState) -> str:
        """Determine if execution should proceed."""
        if state.get("risk_approved"):
            logger.info("Risk approved - proceeding to execution")
            return "execute"
        else:
            logger.info("Risk rejected - skipping execution")
            # Mark as completed even though we're skipping execution
            state["completed"] = True
            return "skip"

    async def run(self, initial_state: TradingState) -> Dict[str, Any]:
        """
        Run trading workflow.

        Args:
            initial_state: Initial workflow state

        Returns:
            Final workflow state after all agents execute
        """
        logger.info(
            "Starting trading workflow",
            symbol=initial_state.get("symbol")
        )

        try:
            # Run the graph
            result = await self.graph.ainvoke(initial_state)

            logger.info(
                "Trading workflow complete",
                approved=result.get("risk_approved"),
                executed=result.get("execution_status")
            )

            return result

        except AgentFailureError as e:
            logger.error(
                "Trading workflow failed",
                agent=e.agent_name,
                error=e.error
            )
            raise
