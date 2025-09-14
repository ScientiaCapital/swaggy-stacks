"""
LangChain Tool Registry for Trading System

This module provides a centralized tool management system that wraps trading operations
as LangChain Tools, enabling sophisticated multi-agent workflows with proper access control.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from langchain.agents import Tool

from app.monitoring.metrics import PrometheusMetrics
from app.trading.trading_manager import get_trading_manager

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories for organizing trading tools"""

    MARKET_DATA = "market_data"
    INDICATORS = "indicators"
    RISK = "risk"
    EXECUTION = "execution"
    PATTERNS = "patterns"
    ANALYSIS = "analysis"
    MEMORY = "memory"
    PORTFOLIO = "portfolio"


class PermissionLevel(Enum):
    """Permission levels for tool access control"""

    READ_ONLY = "read_only"  # Market data, analysis only
    ANALYSIS = "analysis"  # + indicators, patterns
    RISK_ASSESSMENT = "risk_assess"  # + risk calculations
    EXECUTION = "execution"  # + order placement/modification
    ADMIN = "admin"  # All tools


@dataclass
class ToolMetadata:
    """Metadata for a registered tool"""

    name: str
    description: str
    category: ToolCategory
    permission_level: PermissionLevel
    version: str = "1.0"
    tags: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)
    fallback_tool: Optional[str] = None
    usage_count: int = 0
    last_used: Optional[datetime] = None
    success_rate: float = 1.0
    avg_execution_time: float = 0.0


@dataclass
class ToolExecutionResult:
    """Result of a tool execution"""

    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTradingToolRegistry(ABC):
    """Base class for tool registries"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.metadata: Dict[str, ToolMetadata] = {}
        self.metrics = PrometheusMetrics()

    @abstractmethod
    async def register_tool(self, tool: Tool, metadata: ToolMetadata) -> None:
        """Register a new tool"""

    @abstractmethod
    def get_tools_by_permission(self, permission: PermissionLevel) -> List[Tool]:
        """Get tools available for a permission level"""


class LangChainToolRegistry(BaseTradingToolRegistry):
    """
    Advanced LangChain Tool Registry with permission management,
    metrics tracking, and tool composition capabilities
    """

    def __init__(self):
        super().__init__()
        self.permission_hierarchy = {
            PermissionLevel.READ_ONLY: [PermissionLevel.READ_ONLY],
            PermissionLevel.ANALYSIS: [
                PermissionLevel.READ_ONLY,
                PermissionLevel.ANALYSIS,
            ],
            PermissionLevel.RISK_ASSESSMENT: [
                PermissionLevel.READ_ONLY,
                PermissionLevel.ANALYSIS,
                PermissionLevel.RISK_ASSESSMENT,
            ],
            PermissionLevel.EXECUTION: [
                PermissionLevel.READ_ONLY,
                PermissionLevel.ANALYSIS,
                PermissionLevel.RISK_ASSESSMENT,
                PermissionLevel.EXECUTION,
            ],
            PermissionLevel.ADMIN: list(PermissionLevel),
        }
        self.trading_manager = None

    async def initialize(self) -> None:
        """Initialize the registry with trading manager"""
        self.trading_manager = get_trading_manager()
        await self.trading_manager.initialize()

        # Register core trading tools
        await self._register_core_tools()

        logger.info(f"LangChain Tool Registry initialized with {len(self.tools)} tools")

    async def register_tool(self, tool: Tool, metadata: ToolMetadata) -> None:
        """Register a new tool with metadata"""
        # Wrap the tool function to add metrics and error handling
        original_func = tool.func

        async def wrapped_func(*args, **kwargs) -> str:
            start_time = datetime.now()

            try:
                # Execute the tool
                if asyncio.iscoroutinefunction(original_func):
                    result = await original_func(*args, **kwargs)
                else:
                    result = original_func(*args, **kwargs)

                # Update metrics
                execution_time = (datetime.now() - start_time).total_seconds()
                await self._update_tool_metrics(metadata.name, True, execution_time)

                return str(result)

            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                await self._update_tool_metrics(metadata.name, False, execution_time)

                error_msg = f"Tool {metadata.name} failed: {str(e)}"
                logger.error(error_msg)

                # Try fallback tool if available
                if metadata.fallback_tool and metadata.fallback_tool in self.tools:
                    logger.info(f"Trying fallback tool: {metadata.fallback_tool}")
                    self.tools[metadata.fallback_tool]
                    return await wrapped_func(*args, **kwargs)

                return error_msg

        # Create wrapped tool
        wrapped_tool = Tool(
            name=tool.name, description=tool.description, func=wrapped_func
        )

        self.tools[metadata.name] = wrapped_tool
        self.metadata[metadata.name] = metadata

        logger.info(f"Registered tool: {metadata.name} ({metadata.category.value})")

    def get_tools_by_permission(self, permission: PermissionLevel) -> List[Tool]:
        """Get tools available for a permission level"""
        allowed_permissions = self.permission_hierarchy.get(permission, [])
        return [
            tool
            for name, tool in self.tools.items()
            if self.metadata[name].permission_level in allowed_permissions
        ]

    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get tools by category"""
        return [
            tool
            for name, tool in self.tools.items()
            if self.metadata[name].category == category
        ]

    def get_tool_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """Get metadata for a specific tool"""
        return self.metadata.get(tool_name)

    def compose_tools(self, tool_names: List[str], composition_name: str) -> Tool:
        """Compose multiple tools into a single tool"""

        def composed_func(input_data: str) -> str:
            results = []
            for tool_name in tool_names:
                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    result = tool.func(input_data)
                    results.append(f"{tool_name}: {result}")
            return "; ".join(results)

        return Tool(
            name=composition_name,
            description=f"Composed tool using: {', '.join(tool_names)}",
            func=composed_func,
        )

    async def _register_core_tools(self) -> None:
        """Register core trading tools from TradingManager"""
        if not self.trading_manager:
            logger.warning("TradingManager not available for tool registration")
            return

        # Market Data Tools
        await self.register_tool(
            Tool(
                name="get_current_price",
                description="Get current price for a symbol",
                func=self._get_current_price_wrapper,
            ),
            ToolMetadata(
                name="get_current_price",
                description="Retrieve real-time price data for any symbol",
                category=ToolCategory.MARKET_DATA,
                permission_level=PermissionLevel.READ_ONLY,
                tags={"price", "market_data", "real_time"},
            ),
        )

        # Portfolio Tools
        await self.register_tool(
            Tool(
                name="get_portfolio_status",
                description="Get current portfolio status and positions",
                func=self._get_portfolio_status_wrapper,
            ),
            ToolMetadata(
                name="get_portfolio_status",
                description="Retrieve comprehensive portfolio information",
                category=ToolCategory.PORTFOLIO,
                permission_level=PermissionLevel.READ_ONLY,
                tags={"portfolio", "positions", "performance"},
            ),
        )

        # Analysis Tools
        await self.register_tool(
            Tool(
                name="get_market_analysis",
                description="Get comprehensive market analysis for a symbol",
                func=self._get_market_analysis_wrapper,
            ),
            ToolMetadata(
                name="get_market_analysis",
                description="Perform technical and fundamental analysis",
                category=ToolCategory.ANALYSIS,
                permission_level=PermissionLevel.ANALYSIS,
                tags={"analysis", "technical", "fundamental"},
            ),
        )

        # Risk Tools
        await self.register_tool(
            Tool(
                name="assess_trade_risk",
                description="Assess risk for a potential trade",
                func=self._assess_trade_risk_wrapper,
            ),
            ToolMetadata(
                name="assess_trade_risk",
                description="Evaluate risk metrics for trade decisions",
                category=ToolCategory.RISK,
                permission_level=PermissionLevel.RISK_ASSESSMENT,
                tags={"risk", "assessment", "trade"},
            ),
        )

        # Execution Tools (restricted to execution permission)
        await self.register_tool(
            Tool(
                name="execute_trade",
                description="Execute a trade with risk validation",
                func=self._execute_trade_wrapper,
            ),
            ToolMetadata(
                name="execute_trade",
                description="Place trades with comprehensive risk checks",
                category=ToolCategory.EXECUTION,
                permission_level=PermissionLevel.EXECUTION,
                tags={"execution", "trade", "order"},
            ),
        )

        await self.register_tool(
            Tool(
                name="close_position",
                description="Close an existing position",
                func=self._close_position_wrapper,
            ),
            ToolMetadata(
                name="close_position",
                description="Close positions with execution tracking",
                category=ToolCategory.EXECUTION,
                permission_level=PermissionLevel.EXECUTION,
                tags={"execution", "close", "position"},
            ),
        )

    async def _get_current_price_wrapper(self, symbol: str) -> str:
        """Wrapper for getting current price"""
        try:
            price = await self.trading_manager.get_current_price(symbol.strip())
            if price is None:
                return f"Unable to get price for {symbol}"
            return f"Current price for {symbol}: ${price:.2f}"
        except Exception as e:
            return f"Error getting price for {symbol}: {str(e)}"

    async def _get_portfolio_status_wrapper(self, _: str = "") -> str:
        """Wrapper for getting portfolio status"""
        try:
            status = await self.trading_manager.get_portfolio_status()
            return f"Portfolio Status: {status}"
        except Exception as e:
            return f"Error getting portfolio status: {str(e)}"

    async def _get_market_analysis_wrapper(self, symbol: str) -> str:
        """Wrapper for market analysis"""
        try:
            analysis = await self.trading_manager.get_market_analysis(symbol.strip())
            return f"Market Analysis for {symbol}: {analysis}"
        except Exception as e:
            return f"Error analyzing {symbol}: {str(e)}"

    async def _assess_trade_risk_wrapper(self, trade_data: str) -> str:
        """Wrapper for risk assessment"""
        try:
            # Parse trade data (symbol, action, quantity)
            parts = trade_data.split(",")
            if len(parts) < 3:
                return "Trade data format: symbol,action,quantity"

            symbol, action, quantity_str = (
                parts[0].strip(),
                parts[1].strip(),
                parts[2].strip(),
            )
            float(quantity_str)

            # Use trading manager's risk assessment
            risk_status = await self.trading_manager._get_portfolio_risk_status()
            symbol_risk = await self.trading_manager._assess_symbol_risk(symbol)

            return f"Risk Assessment - Portfolio Risk: {risk_status}, Symbol Risk: {symbol_risk}"
        except Exception as e:
            return f"Error assessing trade risk: {str(e)}"

    async def _execute_trade_wrapper(self, trade_data: str) -> str:
        """Wrapper for trade execution"""
        try:
            # Parse trade data (symbol, action, quantity, order_type)
            parts = trade_data.split(",")
            if len(parts) < 3:
                return "Trade data format: symbol,action,quantity[,order_type]"

            symbol = parts[0].strip()
            action = parts[1].strip().upper()
            quantity = float(parts[2].strip())
            order_type = parts[3].strip().upper() if len(parts) > 3 else "MARKET"

            result = await self.trading_manager.execute_trade(
                symbol=symbol, action=action, quantity=quantity, order_type=order_type
            )

            return f"Trade execution result: {result}"
        except Exception as e:
            return f"Error executing trade: {str(e)}"

    async def _close_position_wrapper(self, symbol: str) -> str:
        """Wrapper for closing positions"""
        try:
            result = await self.trading_manager.close_position(symbol.strip())
            return f"Position close result for {symbol}: {result}"
        except Exception as e:
            return f"Error closing position for {symbol}: {str(e)}"

    async def _update_tool_metrics(
        self, tool_name: str, success: bool, execution_time: float
    ) -> None:
        """Update metrics for tool usage"""
        if tool_name in self.metadata:
            metadata = self.metadata[tool_name]
            metadata.usage_count += 1
            metadata.last_used = datetime.now()

            # Update success rate (exponential moving average)
            alpha = 0.1
            metadata.success_rate = (
                alpha * (1.0 if success else 0.0) + (1 - alpha) * metadata.success_rate
            )

            # Update average execution time
            metadata.avg_execution_time = (
                alpha * execution_time + (1 - alpha) * metadata.avg_execution_time
            )

            # Record metrics
            self.metrics.record_tool_usage(
                tool_name=tool_name,
                category=metadata.category.value,
                success=success,
                execution_time=execution_time,
            )


# Global registry instance
_registry_instance: Optional[LangChainToolRegistry] = None


async def get_tool_registry() -> LangChainToolRegistry:
    """Get the global tool registry instance"""
    global _registry_instance

    if _registry_instance is None:
        _registry_instance = LangChainToolRegistry()
        await _registry_instance.initialize()

    return _registry_instance


def create_agent_tools(permission_level: PermissionLevel) -> List[Tool]:
    """Create tools for an agent with specified permission level"""

    async def get_tools():
        registry = await get_tool_registry()
        return registry.get_tools_by_permission(permission_level)

    # Run in event loop or create new one
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, we need to handle this differently
            return []
        return loop.run_until_complete(get_tools())
    except RuntimeError:
        # Create new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tools = loop.run_until_complete(get_tools())
        loop.close()
        return tools


__all__ = [
    "ToolCategory",
    "PermissionLevel",
    "ToolMetadata",
    "LangChainToolRegistry",
    "get_tool_registry",
    "create_agent_tools",
]
