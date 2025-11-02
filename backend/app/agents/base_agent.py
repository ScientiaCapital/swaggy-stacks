"""
Base agent configuration for DeepAgents + LangGraph system.

This module provides the foundation for all specialized trading agents:
- Research Agent: Market analysis and opportunity detection
- Strategy Agent: Options strategy selection
- Risk Agent: Portfolio risk validation
- Execution Agent: Trade execution and monitoring
- Learning Agent: Post-market learning and improvement
"""
from typing import Optional, Dict, Any, List
from enum import Enum
import os
from anthropic import Anthropic
from openai import OpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
import structlog

logger = structlog.get_logger()


class ModelProvider(Enum):
    """AI model providers"""
    CLAUDE = "claude"
    OPENROUTER = "openrouter"
    CEREBRAS = "cerebras"


class ModelConfig:
    """Model configuration for different use cases"""

    # Primary reasoning (complex analysis)
    PRIMARY = os.getenv("AGENT_PRIMARY_MODEL", "claude-sonnet-4.5")

    # Fast inference (real-time decisions)
    FAST = os.getenv("AGENT_FAST_MODEL", "cerebras/llama-3.3-70b")

    # Cost-optimized (batch processing)
    CHEAP = os.getenv("AGENT_CHEAP_MODEL", "deepseek/deepseek-chat")

    # Risk validation (critical decisions)
    RISK = os.getenv("AGENT_RISK_MODEL", "claude-sonnet-4.5")


class BaseAgent:
    """
    Base agent class providing common functionality for all trading agents.

    Features:
    - Multi-model AI support (Claude, OpenRouter, Cerebras)
    - LangGraph state machine integration
    - PostgreSQL checkpoint persistence
    - Structured logging with context
    - Model selection based on use case
    """

    def __init__(
        self,
        name: str,
        description: str,
        model_config: str = ModelConfig.PRIMARY,
        enable_checkpoints: bool = True,
    ):
        """
        Initialize base agent.

        Args:
            name: Agent name (e.g., "research", "strategy")
            description: Agent purpose and capabilities
            model_config: Model to use (PRIMARY, FAST, CHEAP, RISK)
            enable_checkpoints: Enable state persistence to PostgreSQL
        """
        self.name = name
        self.description = description
        self.model_config = model_config
        self.enable_checkpoints = enable_checkpoints

        # Initialize logger with agent context
        self.logger = logger.bind(agent=name)

        # Initialize AI clients
        self._init_ai_clients()

        # Initialize LangGraph checkpointing if enabled
        self.checkpoint = None
        if enable_checkpoints:
            self._init_checkpointing()

    def _init_ai_clients(self):
        """Initialize AI provider clients based on model config."""
        self.logger.info("Initializing AI clients", model=self.model_config)

        # Always init Anthropic (primary provider)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.anthropic = Anthropic(api_key=anthropic_key)
        self.claude_chat = ChatAnthropic(
            model="claude-sonnet-4.5",
            api_key=anthropic_key,
            max_tokens=4096,
        )

        # Init OpenRouter for multi-model access
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            self.openrouter = OpenAI(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
            )
            self.logger.info("OpenRouter client initialized")
        else:
            self.openrouter = None
            self.logger.warning("OpenRouter API key not found, multi-model access disabled")

    def _init_checkpointing(self):
        """Initialize PostgreSQL-backed state persistence."""
        db_uri = os.getenv("LANGGRAPH_DB_URI")
        if not db_uri:
            self.logger.warning("LANGGRAPH_DB_URI not set, checkpointing disabled")
            return

        checkpoint_table = os.getenv("LANGGRAPH_CHECKPOINT_TABLE", "agent_checkpoints")

        self.checkpoint = PostgresSaver(
            conn_string=db_uri,
        )

        self.logger.info(
            "Checkpoint persistence initialized",
            table=checkpoint_table,
            namespace=f"{self.name}_agent"
        )

    async def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke agent with input data.

        Args:
            input_data: Agent-specific input parameters

        Returns:
            Agent-specific output results
        """
        raise NotImplementedError("Subclasses must implement invoke()")

    def get_model_client(self, use_case: str = "default"):
        """
        Get appropriate AI client based on use case.

        Args:
            use_case: "reasoning", "fast", "cheap", "risk"

        Returns:
            Appropriate AI client for the use case
        """
        if use_case == "fast" and self.openrouter:
            return self.openrouter
        elif use_case == "cheap" and self.openrouter:
            return self.openrouter
        else:
            return self.claude_chat
