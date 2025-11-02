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
        self.checkpoint_uri = None
        self.checkpoint_table = None
        self.checkpoint_ns = None
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
            model=self.model_config,  # Use instance configuration, not hardcoded
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
        """Store checkpoint configuration for graph creation."""
        self.checkpoint_uri = os.getenv("LANGGRAPH_DB_URI")
        if not self.checkpoint_uri:
            self.logger.warning("LANGGRAPH_DB_URI not set, checkpointing disabled")
            self.checkpoint_uri = None
            return

        self.checkpoint_table = os.getenv("LANGGRAPH_CHECKPOINT_TABLE", "agent_checkpoints")
        self.checkpoint_ns = f"{self.name}_agent"

        self.logger.info(
            "Checkpoint persistence configured",
            table=self.checkpoint_table,
            namespace=self.checkpoint_ns,
            uri_set=True
        )

    def get_checkpointer(self):
        """
        Create PostgresSaver checkpointer for graph compilation.

        Usage:
            with agent.get_checkpointer() as checkpointer:
                graph = workflow.compile(checkpointer=checkpointer)

        Returns:
            PostgresSaver context manager or None if checkpointing disabled
        """
        if not self.checkpoint_uri:
            return None

        return PostgresSaver.from_conn_string(self.checkpoint_uri)

    async def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke agent with input data.

        Args:
            input_data: Agent-specific input parameters

        Returns:
            Agent-specific output results
        """
        raise NotImplementedError("Subclasses must implement invoke()")

    def get_model_client(self, use_case: str = "default") -> Any:
        """
        Get appropriate AI client based on use case.

        Args:
            use_case: "reasoning", "fast", "cheap", "risk", or "default"

        Returns:
            Configured AI client for the use case
        """
        # Map use case to model configuration
        model_map = {
            "reasoning": ModelConfig.PRIMARY,
            "fast": ModelConfig.FAST,
            "cheap": ModelConfig.CHEAP,
            "risk": ModelConfig.RISK,
            "default": self.model_config,
        }

        model = model_map.get(use_case, self.model_config)

        # Route to appropriate provider based on model prefix
        if model.startswith("cerebras/") or model.startswith("deepseek/"):
            if not self.openrouter:
                self.logger.warning(
                    "OpenRouter unavailable, falling back to Claude",
                    requested_model=model
                )
                return self.claude_chat

            # Return OpenRouter client configured for specific model
            # Note: Model selection happens at API call time via model parameter
            return self.openrouter
        else:
            # Claude models use direct Anthropic client
            return self.claude_chat
