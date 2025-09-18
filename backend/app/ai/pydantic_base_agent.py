"""
PydanticAI Base Agent Framework
==============================

Type-safe, validated agent framework using PydanticAI for enhanced reliability
and performance in high-frequency trading operations.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass

from pydantic import BaseModel, Field, validator
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model, OpenAIModel
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)

# Type variable for agent response types
T = TypeVar('T', bound=BaseModel)


class AgentContext(BaseModel):
    """Structured context for agent execution"""

    agent_id: str = Field(..., description="Unique agent identifier")
    symbol: str = Field(..., description="Trading symbol being analyzed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    risk_tolerance: float = Field(default=0.02, ge=0.0, le=1.0, description="Risk tolerance (0-1)")
    max_position_size: float = Field(default=10000.0, gt=0, description="Maximum position size")

    class Config:
        arbitrary_types_allowed = True


class AgentResponse(BaseModel, Generic[T]):
    """Standardized agent response with metadata"""

    data: T = Field(..., description="Typed response data")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    execution_time_ms: float = Field(..., gt=0, description="Execution time in milliseconds")
    agent_id: str = Field(..., description="Agent identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is within valid range"""
        return max(0.0, min(1.0, v))


class AgentExecutionStats(BaseModel):
    """Agent execution statistics and performance metrics"""

    total_executions: int = Field(default=0, ge=0)
    successful_executions: int = Field(default=0, ge=0)
    failed_executions: int = Field(default=0, ge=0)
    average_execution_time_ms: float = Field(default=0.0, ge=0.0)
    average_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    last_execution: Optional[datetime] = Field(default=None)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    def update_stats(self, execution_time_ms: float, confidence: float, success: bool = True):
        """Update execution statistics"""
        self.total_executions += 1
        self.last_execution = datetime.now()

        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        # Update averages using exponential moving average
        alpha = 0.1  # Smoothing factor
        self.average_execution_time_ms = (
            alpha * execution_time_ms + (1 - alpha) * self.average_execution_time_ms
        )
        self.average_confidence = (
            alpha * confidence + (1 - alpha) * self.average_confidence
        )

        # Update error rate
        self.error_rate = self.failed_executions / self.total_executions if self.total_executions > 0 else 0.0


class TradingToolResult(BaseModel):
    """Structured result from trading tool execution"""

    tool_name: str = Field(..., description="Name of executed tool")
    success: bool = Field(..., description="Whether tool execution succeeded")
    result: Any = Field(default=None, description="Tool execution result")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time_ms: float = Field(..., gt=0, description="Tool execution time")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PydanticBaseAgent(ABC, Generic[T]):
    """
    Type-safe base agent using PydanticAI framework

    Provides:
    - Structured, validated inputs/outputs
    - Type-safe tool calling
    - Comprehensive error handling
    - Performance monitoring
    - Dependency injection support
    """

    def __init__(
        self,
        agent_type: str,
        model_name: str = "claude-3-5-sonnet-20241022",
        system_prompt: Optional[str] = None,
        enable_tools: bool = True,
        max_retries: int = 3,
    ):
        self.agent_type = agent_type
        self.model_name = model_name
        self.enable_tools = enable_tools
        self.max_retries = max_retries
        self.stats = AgentExecutionStats()

        # Initialize PydanticAI model
        self.model = self._create_model(model_name)

        # Create PydanticAI agent with structured response type
        self.agent = Agent(
            model=self.model,
            system_prompt=system_prompt or self._get_default_system_prompt(),
            result_type=self._get_response_type(),
        )

        # Register trading tools if enabled
        if enable_tools:
            self._register_tools()

        logger.info(
            "PydanticAI agent initialized",
            agent_type=agent_type,
            model=model_name,
            tools_enabled=enable_tools,
        )

    def _create_model(self, model_name: str) -> Model:
        """Create appropriate model instance based on configuration"""
        if model_name.startswith("claude"):
            # Use Anthropic Claude models
            return OpenAIModel(
                model_name,
                base_url="https://api.anthropic.com",
                api_key=settings.ANTHROPIC_API_KEY if hasattr(settings, 'ANTHROPIC_API_KEY') else None,
            )
        elif model_name.startswith("gpt"):
            # Use OpenAI models
            return OpenAIModel(model_name)
        else:
            # Default to Claude
            return OpenAIModel("claude-3-5-sonnet-20241022")

    @abstractmethod
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for this agent type"""
        pass

    @abstractmethod
    def _get_response_type(self) -> type[BaseModel]:
        """Get the expected response type for this agent"""
        pass

    def _register_tools(self):
        """Register trading-specific tools with the agent"""

        @self.agent.tool
        async def get_market_data(ctx: RunContext[AgentContext], symbol: str) -> Dict[str, Any]:
            """Fetch current market data for a symbol"""
            start_time = time.time()

            try:
                # Mock market data for now - would integrate with real data source
                market_data = {
                    "symbol": symbol,
                    "price": 150.25,
                    "volume": 1000000,
                    "price_change": 2.34,
                    "price_change_pct": 1.58,
                    "timestamp": datetime.now().isoformat(),
                }

                execution_time_ms = (time.time() - start_time) * 1000

                logger.info(
                    "Market data retrieved",
                    symbol=symbol,
                    execution_time_ms=execution_time_ms,
                    agent_id=ctx.deps.agent_id,
                )

                return market_data

            except Exception as e:
                execution_time_ms = (time.time() - start_time) * 1000
                logger.error(
                    "Market data retrieval failed",
                    symbol=symbol,
                    error=str(e),
                    execution_time_ms=execution_time_ms,
                )
                raise

        @self.agent.tool
        async def calculate_technical_indicators(
            ctx: RunContext[AgentContext],
            symbol: str,
            period: int = 14
        ) -> Dict[str, float]:
            """Calculate technical indicators for analysis"""
            start_time = time.time()

            try:
                # Mock technical indicators - would calculate from real data
                indicators = {
                    "rsi": 65.23,
                    "macd": 1.45,
                    "macd_signal": 1.12,
                    "bb_upper": 155.67,
                    "bb_lower": 144.83,
                    "bb_position": 0.62,
                    "atr": 2.45,
                    "volume_sma": 950000,
                }

                execution_time_ms = (time.time() - start_time) * 1000

                logger.info(
                    "Technical indicators calculated",
                    symbol=symbol,
                    period=period,
                    execution_time_ms=execution_time_ms,
                    agent_id=ctx.deps.agent_id,
                )

                return indicators

            except Exception as e:
                execution_time_ms = (time.time() - start_time) * 1000
                logger.error(
                    "Technical indicator calculation failed",
                    symbol=symbol,
                    error=str(e),
                    execution_time_ms=execution_time_ms,
                )
                raise

        @self.agent.tool
        async def assess_portfolio_risk(
            ctx: RunContext[AgentContext],
            position_size: float,
            account_value: float
        ) -> Dict[str, float]:
            """Assess risk for proposed position"""
            start_time = time.time()

            try:
                position_percentage = (position_size / account_value) * 100
                risk_level = "low" if position_percentage < 2 else ("medium" if position_percentage < 5 else "high")

                risk_assessment = {
                    "position_percentage": position_percentage,
                    "risk_level": risk_level,
                    "recommended_size": min(position_size, account_value * ctx.deps.risk_tolerance),
                    "max_loss_estimate": position_size * 0.05,  # 5% potential loss
                    "portfolio_heat": position_percentage / 100,
                }

                execution_time_ms = (time.time() - start_time) * 1000

                logger.info(
                    "Portfolio risk assessed",
                    position_size=position_size,
                    risk_level=risk_level,
                    execution_time_ms=execution_time_ms,
                    agent_id=ctx.deps.agent_id,
                )

                return risk_assessment

            except Exception as e:
                execution_time_ms = (time.time() - start_time) * 1000
                logger.error(
                    "Portfolio risk assessment failed",
                    position_size=position_size,
                    error=str(e),
                    execution_time_ms=execution_time_ms,
                )
                raise

    async def execute(
        self,
        prompt: str,
        context: AgentContext,
        **kwargs
    ) -> AgentResponse[T]:
        """Execute agent with structured context and response"""
        start_time = time.time()
        execution_attempts = 0

        while execution_attempts < self.max_retries:
            try:
                execution_attempts += 1

                logger.info(
                    "Executing PydanticAI agent",
                    agent_type=self.agent_type,
                    agent_id=context.agent_id,
                    symbol=context.symbol,
                    attempt=execution_attempts,
                )

                # Run agent with context as dependencies
                result = await self.agent.run(prompt, deps=context, **kwargs)

                execution_time_ms = (time.time() - start_time) * 1000

                # Extract confidence from result if available
                confidence = getattr(result.data, 'confidence', 0.8)

                # Update statistics
                self.stats.update_stats(execution_time_ms, confidence, success=True)

                # Create structured response
                response = AgentResponse[T](
                    data=result.data,
                    confidence=confidence,
                    execution_time_ms=execution_time_ms,
                    agent_id=context.agent_id,
                    metadata={
                        "model": self.model_name,
                        "attempts": execution_attempts,
                        "tools_used": len(result.all_messages()) - 1,  # Exclude initial prompt
                    }
                )

                logger.info(
                    "Agent execution completed successfully",
                    agent_type=self.agent_type,
                    execution_time_ms=execution_time_ms,
                    confidence=confidence,
                    attempts=execution_attempts,
                )

                return response

            except Exception as e:
                execution_time_ms = (time.time() - start_time) * 1000

                logger.warning(
                    "Agent execution attempt failed",
                    agent_type=self.agent_type,
                    attempt=execution_attempts,
                    error=str(e),
                    execution_time_ms=execution_time_ms,
                )

                # If this was the last attempt, update stats and raise
                if execution_attempts >= self.max_retries:
                    self.stats.update_stats(execution_time_ms, 0.0, success=False)

                    logger.error(
                        "Agent execution failed after all retries",
                        agent_type=self.agent_type,
                        attempts=execution_attempts,
                        final_error=str(e),
                    )
                    raise

                # Brief delay before retry
                await asyncio.sleep(0.1 * execution_attempts)

    async def health_check(self) -> Dict[str, Any]:
        """Check agent health and performance"""
        return {
            "agent_type": self.agent_type,
            "model": self.model_name,
            "status": "healthy" if self.stats.error_rate < 0.1 else "degraded",
            "stats": self.stats.dict(),
            "tools_enabled": self.enable_tools,
            "max_retries": self.max_retries,
        }

    def get_stats(self) -> AgentExecutionStats:
        """Get current execution statistics"""
        return self.stats