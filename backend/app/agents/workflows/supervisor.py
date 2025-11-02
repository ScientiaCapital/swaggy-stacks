"""
Trading Supervisor for LangGraph workflows.
"""
from typing import List, Dict, Any
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import Tool
from langgraph.checkpoint.postgres import PostgresSaver
from app.agents.research_agent import ResearchAgent
from app.agents.strategy_agent import StrategyAgent
from app.agents.risk_agent import RiskAgent
from app.agents.execution_agent import ExecutionAgent
from app.agents.base_agent import BaseAgent
from app.core.config import settings
import structlog

logger = structlog.get_logger(__name__)


class TradingSupervisor:
    """
    Supervisor coordinates specialized trading agents.

    The supervisor uses Claude Sonnet 4.5 to make routing decisions,
    calling specialized agents as tools based on workflow state.
    """

    def __init__(self):
        """Initialize supervisor with model and agent tools."""
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.model = ChatAnthropic(
            model="claude-sonnet-4.5",
            api_key=anthropic_key
        )

        # Setup PostgreSQL checkpointing if URI configured
        self.checkpointer = None
        if hasattr(settings, 'LANGGRAPH_DB_URI') and settings.LANGGRAPH_DB_URI:
            self.checkpointer = PostgresSaver.from_conn_string(
                settings.LANGGRAPH_DB_URI
            )
            logger.info("PostgreSQL checkpointing enabled")
        else:
            logger.warning("LANGGRAPH_DB_URI not configured, using in-memory state")

        # Convert agents to tools
        self.tools = self._create_agent_tools()

        logger.info(
            "TradingSupervisor initialized",
            tools_count=len(self.tools),
            checkpointing=self.checkpointer is not None
        )

    def _create_agent_tools(self) -> List[Tool]:
        """Convert specialized agents to tools."""
        agents = [
            ResearchAgent(),
            StrategyAgent(),
            RiskAgent(),
            ExecutionAgent()
        ]

        return [self._agent_to_tool(agent) for agent in agents]

    def _agent_to_tool(self, agent: BaseAgent) -> Tool:
        """
        Convert agent into LangGraph tool.

        Args:
            agent: BaseAgent instance to convert

        Returns:
            Tool that can be called by supervisor
        """
        tool_name = agent.name.lower().replace(" ", "_")

        return Tool(
            name=tool_name,
            description=agent.description,
            func=agent.invoke
        )

    def get_system_prompt(self) -> str:
        """Get supervisor system prompt for routing decisions."""
        return """You are the Trading Supervisor coordinating specialized agents.

Your responsibilities:
1. Analyze current workflow state
2. Decide which agent to call next
3. Pass appropriate data to agents
4. Determine when workflow is complete

Agent call order:
1. research_agent - Analyzes market conditions, detects regime
2. strategy_agent - Selects options strategy for regime
3. risk_agent - Validates portfolio risk and position sizing
4. execution_agent - Executes approved trades (ONLY if risk_approved=True)

Rules:
- Call agents in order: research → strategy → risk → execution
- Skip execution if risk_approved=False
- Return "FINISH" when workflow complete
- Fail immediately if any agent returns error

Output format:
- Agent name to call next: "research_agent", "strategy_agent", etc.
- Or "FINISH" if workflow complete
"""
