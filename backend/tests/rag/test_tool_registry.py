"""Comprehensive tests for the Tool Registry component."""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Callable

# Import the components we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from app.rag.services.tool_registry import (
    TradingToolRegistry,
    ToolDefinition,
    ToolCategory,
    PermissionLevel,
    ToolResult,
    ToolExecutionContext
)
from tests.rag.fixtures.market_data_fixtures import (
    sample_tool_definitions,
    mock_db_connection
)


class TestToolDefinition:
    """Test the ToolDefinition data class."""
    
    def test_tool_definition_creation(self):
        """Test creating a ToolDefinition instance."""
        tool_def = ToolDefinition(
            name="get_market_data",
            description="Retrieve current market data for a symbol",
            category=ToolCategory.MARKET_DATA,
            parameters={
                "symbol": {"type": "string", "required": True},
                "fields": {"type": "array", "items": {"type": "string"}}
            },
            permissions=[PermissionLevel.READ_MARKET_DATA],
            async_capable=True,
            implementation=lambda symbol, fields=None: {"price": 150.25}
        )
        
        assert tool_def.name == "get_market_data"
        assert tool_def.category == ToolCategory.MARKET_DATA
        assert tool_def.async_capable is True
        assert PermissionLevel.READ_MARKET_DATA in tool_def.permissions
        assert tool_def.implementation is not None
    
    def test_tool_parameter_validation_schema(self):
        """Test tool parameter validation schema."""
        tool_def = ToolDefinition(
            name="calculate_rsi",
            description="Calculate RSI indicator",
            category=ToolCategory.TECHNICAL_INDICATORS,
            parameters={
                "prices": {"type": "array", "items": {"type": "number"}, "required": True},
                "period": {"type": "integer", "default": 14, "minimum": 1, "maximum": 100}
            },
            implementation=lambda prices, period=14: {"rsi": 65.5}
        )
        
        # Valid parameters should validate
        valid_params = {"prices": [100, 101, 102, 103], "period": 14}
        assert tool_def.validate_parameters(valid_params) is True
        
        # Invalid parameters should fail validation
        invalid_params = {"prices": "not_array", "period": -1}
        assert tool_def.validate_parameters(invalid_params) is False
    
    def test_tool_serialization(self):
        """Test tool definition serialization."""
        tool_def = ToolDefinition(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.RISK_MANAGEMENT,
            parameters={"param1": {"type": "string"}},
            implementation=lambda x: x
        )
        
        serialized = tool_def.to_dict()
        assert serialized["name"] == "test_tool"
        assert serialized["category"] == "risk_management"
        assert "implementation" not in serialized  # Implementation should not be serialized
        
        # Should be able to recreate from dict (without implementation)
        recreated = ToolDefinition.from_dict(serialized)
        assert recreated.name == tool_def.name
        assert recreated.category == tool_def.category


class TestToolResult:
    """Test the ToolResult data class."""
    
    def test_result_creation(self):
        """Test creating a ToolResult instance."""
        result = ToolResult(
            tool_name="get_market_data",
            success=True,
            data={"symbol": "AAPL", "price": 150.25, "volume": 45000000},
            metadata={
                "execution_time_ms": 120,
                "source": "alpaca_api"
            }
        )
        
        assert result.tool_name == "get_market_data"
        assert result.success is True
        assert result.data["price"] == 150.25
        assert result.execution_time is not None
    
    def test_error_result(self):
        """Test creating an error result."""
        result = ToolResult(
            tool_name="failing_tool",
            success=False,
            error="API rate limit exceeded",
            data=None
        )
        
        assert result.success is False
        assert result.error == "API rate limit exceeded"
        assert result.data is None


@pytest.mark.asyncio
class TestTradingToolRegistry:
    """Comprehensive tests for TradingToolRegistry."""
    
    @pytest.fixture
    async def tool_registry(self, mock_db_connection):
        """Create a tool registry instance for testing."""
        with patch('app.rag.services.tool_registry.get_db_connection') as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db_connection)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=None)
            
            registry = TradingToolRegistry()
            await registry.initialize()
            return registry
    
    async def test_initialization(self, tool_registry):
        """Test tool registry initialization."""
        assert tool_registry._initialized is True
        assert len(tool_registry._tools) >= 0  # May have default tools
        assert len(tool_registry._categories) > 0
    
    async def test_register_tool(self, tool_registry):
        """Test registering a new tool."""
        def mock_market_data_tool(symbol: str, fields: List[str] = None) -> Dict[str, Any]:
            return {
                "symbol": symbol,
                "price": 150.25,
                "volume": 45000000,
                "fields": fields or ["price", "volume"]
            }
        
        tool_def = ToolDefinition(
            name="test_market_data",
            description="Test market data retrieval",
            category=ToolCategory.MARKET_DATA,
            parameters={
                "symbol": {"type": "string", "required": True},
                "fields": {"type": "array", "items": {"type": "string"}}
            },
            permissions=[PermissionLevel.READ_MARKET_DATA],
            implementation=mock_market_data_tool,
            async_capable=False
        )
        
        success = await tool_registry.register_tool(tool_def)
        
        assert success is True
        assert "test_market_data" in tool_registry._tools
        assert tool_registry._tools["test_market_data"].name == "test_market_data"
    
    async def test_register_duplicate_tool(self, tool_registry):
        """Test registering a tool with duplicate name."""
        tool_def = ToolDefinition(
            name="duplicate_tool",
            description="First tool",
            category=ToolCategory.MARKET_DATA,
            implementation=lambda: "first"
        )
        
        # First registration should succeed
        success1 = await tool_registry.register_tool(tool_def)
        assert success1 is True
        
        # Second registration with same name should fail
        duplicate_def = ToolDefinition(
            name="duplicate_tool",
            description="Second tool",
            category=ToolCategory.TECHNICAL_INDICATORS,
            implementation=lambda: "second"
        )
        
        success2 = await tool_registry.register_tool(duplicate_def, allow_override=False)
        assert success2 is False
        
        # Should succeed with override allowed
        success3 = await tool_registry.register_tool(duplicate_def, allow_override=True)
        assert success3 is True
    
    async def test_discover_tools(self, tool_registry):
        """Test tool discovery functionality."""
        # Register some test tools
        await tool_registry.register_tool(ToolDefinition(
            name="market_tool_1",
            description="Market data tool",
            category=ToolCategory.MARKET_DATA,
            implementation=lambda: {}
        ))
        
        await tool_registry.register_tool(ToolDefinition(
            name="indicator_tool_1", 
            description="Technical indicator tool",
            category=ToolCategory.TECHNICAL_INDICATORS,
            implementation=lambda: {}
        ))
        
        # Discover all tools
        all_tools = await tool_registry.discover_tools()
        assert len(all_tools) >= 2
        
        # Discover by category
        market_tools = await tool_registry.discover_tools(
            category=ToolCategory.MARKET_DATA
        )
        assert len(market_tools) >= 1
        assert all(tool.category == ToolCategory.MARKET_DATA for tool in market_tools)
        
        # Discover by permissions
        read_tools = await tool_registry.discover_tools(
            required_permissions=[PermissionLevel.READ_MARKET_DATA]
        )
        assert len(read_tools) >= 0
    
    async def test_execute_tool(self, tool_registry):
        """Test tool execution."""
        def mock_calculator(a: int, b: int) -> int:
            return a + b
        
        tool_def = ToolDefinition(
            name="test_calculator",
            description="Test calculator",
            category=ToolCategory.ANALYTICS,
            parameters={
                "a": {"type": "integer", "required": True},
                "b": {"type": "integer", "required": True}
            },
            implementation=mock_calculator
        )
        
        await tool_registry.register_tool(tool_def)
        
        context = ToolExecutionContext(
            agent_id="test_agent",
            session_id="test_session",
            permissions=[PermissionLevel.EXECUTE_CALCULATIONS]
        )
        
        result = await tool_registry.execute_tool(
            tool_name="test_calculator",
            parameters={"a": 5, "b": 3},
            context=context
        )
        
        assert result.success is True
        assert result.data == 8
        assert result.tool_name == "test_calculator"
    
    async def test_async_tool_execution(self, tool_registry):
        """Test execution of async tools."""
        async def mock_async_tool(symbol: str) -> Dict[str, Any]:
            await asyncio.sleep(0.01)  # Simulate async operation
            return {"symbol": symbol, "async_result": True}
        
        tool_def = ToolDefinition(
            name="test_async_tool",
            description="Test async tool",
            category=ToolCategory.MARKET_DATA,
            parameters={"symbol": {"type": "string", "required": True}},
            implementation=mock_async_tool,
            async_capable=True
        )
        
        await tool_registry.register_tool(tool_def)
        
        context = ToolExecutionContext(agent_id="test_agent")
        
        result = await tool_registry.execute_tool(
            tool_name="test_async_tool",
            parameters={"symbol": "AAPL"},
            context=context
        )
        
        assert result.success is True
        assert result.data["symbol"] == "AAPL"
        assert result.data["async_result"] is True
    
    async def test_parameter_validation(self, tool_registry):
        """Test parameter validation during tool execution."""
        tool_def = ToolDefinition(
            name="strict_tool",
            description="Tool with strict parameters",
            category=ToolCategory.TECHNICAL_INDICATORS,
            parameters={
                "required_param": {"type": "string", "required": True},
                "optional_param": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100}
            },
            implementation=lambda required_param, optional_param=10: {
                "result": f"{required_param}_{optional_param}"
            }
        )
        
        await tool_registry.register_tool(tool_def)
        
        context = ToolExecutionContext(agent_id="test_agent")
        
        # Valid parameters should work
        valid_result = await tool_registry.execute_tool(
            tool_name="strict_tool",
            parameters={"required_param": "test", "optional_param": 50},
            context=context
        )
        assert valid_result.success is True
        
        # Missing required parameter should fail
        invalid_result = await tool_registry.execute_tool(
            tool_name="strict_tool",
            parameters={"optional_param": 50},  # Missing required_param
            context=context
        )
        assert invalid_result.success is False
        assert "required_param" in invalid_result.error.lower()
        
        # Out of range parameter should fail
        range_result = await tool_registry.execute_tool(
            tool_name="strict_tool",
            parameters={"required_param": "test", "optional_param": 150},  # Too high
            context=context
        )
        assert range_result.success is False
    
    async def test_permission_enforcement(self, tool_registry):
        """Test permission enforcement for tool access."""
        restricted_tool_def = ToolDefinition(
            name="restricted_tool",
            description="Tool requiring special permissions",
            category=ToolCategory.ORDER_EXECUTION,
            parameters={"action": {"type": "string"}},
            permissions=[PermissionLevel.EXECUTE_TRADES, PermissionLevel.MODIFY_ORDERS],
            implementation=lambda action: {"executed": action}
        )
        
        await tool_registry.register_tool(restricted_tool_def)
        
        # Context with insufficient permissions
        limited_context = ToolExecutionContext(
            agent_id="limited_agent",
            permissions=[PermissionLevel.READ_MARKET_DATA]
        )
        
        # Should fail due to insufficient permissions
        result = await tool_registry.execute_tool(
            tool_name="restricted_tool",
            parameters={"action": "buy"},
            context=limited_context
        )
        
        assert result.success is False
        assert "permission" in result.error.lower()
        
        # Context with sufficient permissions
        authorized_context = ToolExecutionContext(
            agent_id="authorized_agent",
            permissions=[PermissionLevel.EXECUTE_TRADES, PermissionLevel.MODIFY_ORDERS]
        )
        
        # Should succeed with proper permissions
        result = await tool_registry.execute_tool(
            tool_name="restricted_tool",
            parameters={"action": "buy"},
            context=authorized_context
        )
        
        assert result.success is True
        assert result.data["executed"] == "buy"
    
    async def test_tool_execution_logging(self, tool_registry, mock_db_connection):
        """Test logging of tool executions."""
        tool_def = ToolDefinition(
            name="logged_tool",
            description="Tool with execution logging",
            category=ToolCategory.ANALYTICS,
            implementation=lambda x: {"result": x * 2}
        )
        
        await tool_registry.register_tool(tool_def)
        
        context = ToolExecutionContext(
            agent_id="test_agent",
            session_id="test_session"
        )
        
        # Mock successful database logging
        mock_db_connection.return_value = True
        
        result = await tool_registry.execute_tool(
            tool_name="logged_tool",
            parameters={"x": 5},
            context=context,
            log_execution=True
        )
        
        assert result.success is True
        # Verify logging was attempted
        assert len(mock_db_connection.queries_executed) > 0
    
    async def test_batch_tool_registration(self, tool_registry):
        """Test registering multiple tools at once."""
        tool_definitions = [
            ToolDefinition(
                name=f"batch_tool_{i}",
                description=f"Batch tool {i}",
                category=ToolCategory.ANALYTICS,
                implementation=lambda x: {"tool_id": i, "result": x}
            )
            for i in range(3)
        ]
        
        results = await tool_registry.batch_register_tools(tool_definitions)
        
        assert len(results) == 3
        assert all(results)  # All registrations should succeed
        
        # Verify all tools are registered
        for i in range(3):
            assert f"batch_tool_{i}" in tool_registry._tools
    
    async def test_tool_unregistration(self, tool_registry):
        """Test unregistering tools."""
        tool_def = ToolDefinition(
            name="temporary_tool",
            description="Tool to be removed",
            category=ToolCategory.ANALYTICS,
            implementation=lambda: "temp"
        )
        
        # Register tool
        await tool_registry.register_tool(tool_def)
        assert "temporary_tool" in tool_registry._tools
        
        # Unregister tool
        success = await tool_registry.unregister_tool("temporary_tool")
        assert success is True
        assert "temporary_tool" not in tool_registry._tools
        
        # Unregistering non-existent tool should return False
        success = await tool_registry.unregister_tool("non_existent_tool")
        assert success is False
    
    async def test_tool_metadata_and_statistics(self, tool_registry, mock_db_connection):
        """Test tool metadata and usage statistics."""
        # Mock statistics from database
        mock_stats = [
            {
                "tool_name": "popular_tool",
                "execution_count": 100,
                "success_rate": 0.95,
                "avg_execution_time_ms": 150.5,
                "last_used": datetime.now().isoformat()
            }
        ]
        
        with patch.object(mock_db_connection, 'fetch', return_value=mock_stats):
            stats = await tool_registry.get_tool_statistics("popular_tool")
            
            assert stats["execution_count"] == 100
            assert stats["success_rate"] == 0.95
            assert stats["avg_execution_time_ms"] == 150.5
    
    async def test_error_handling_in_tool_execution(self, tool_registry):
        """Test error handling when tools raise exceptions."""
        def failing_tool(should_fail: bool):
            if should_fail:
                raise ValueError("Tool intentionally failed")
            return {"success": True}
        
        tool_def = ToolDefinition(
            name="failing_tool",
            description="Tool that can fail",
            category=ToolCategory.ANALYTICS,
            parameters={"should_fail": {"type": "boolean"}},
            implementation=failing_tool
        )
        
        await tool_registry.register_tool(tool_def)
        
        context = ToolExecutionContext(agent_id="test_agent")
        
        # Execute with failure
        result = await tool_registry.execute_tool(
            tool_name="failing_tool",
            parameters={"should_fail": True},
            context=context
        )
        
        assert result.success is False
        assert "Tool intentionally failed" in result.error
        
        # Execute without failure
        result = await tool_registry.execute_tool(
            tool_name="failing_tool",
            parameters={"should_fail": False},
            context=context
        )
        
        assert result.success is True
        assert result.data["success"] is True
    
    async def test_tool_timeout_handling(self, tool_registry):
        """Test handling of tool execution timeouts."""
        async def slow_tool():
            await asyncio.sleep(1.0)  # Longer than timeout
            return {"completed": True}
        
        tool_def = ToolDefinition(
            name="slow_tool",
            description="Tool that takes time",
            category=ToolCategory.ANALYTICS,
            implementation=slow_tool,
            async_capable=True,
            timeout_seconds=0.1  # Very short timeout
        )
        
        await tool_registry.register_tool(tool_def)
        
        context = ToolExecutionContext(agent_id="test_agent")
        
        result = await tool_registry.execute_tool(
            tool_name="slow_tool",
            parameters={},
            context=context
        )
        
        # Should fail due to timeout
        assert result.success is False
        assert "timeout" in result.error.lower()
    
    async def test_concurrent_tool_executions(self, tool_registry):
        """Test concurrent execution of multiple tools."""
        def concurrent_tool(tool_id: int, delay: float = 0.01):
            import time
            time.sleep(delay)  # Simulate work
            return {"tool_id": tool_id, "timestamp": datetime.now().isoformat()}
        
        tool_def = ToolDefinition(
            name="concurrent_test_tool",
            description="Tool for concurrency testing",
            category=ToolCategory.ANALYTICS,
            parameters={
                "tool_id": {"type": "integer", "required": True},
                "delay": {"type": "number", "default": 0.01}
            },
            implementation=concurrent_tool
        )
        
        await tool_registry.register_tool(tool_def)
        
        context = ToolExecutionContext(agent_id="test_agent")
        
        # Execute multiple tools concurrently
        tasks = [
            tool_registry.execute_tool(
                tool_name="concurrent_test_tool",
                parameters={"tool_id": i, "delay": 0.01},
                context=context
            )
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All executions should succeed
        assert len(results) == 10
        assert all(result.success for result in results)
        
        # Each should have unique tool_id
        tool_ids = {result.data["tool_id"] for result in results}
        assert len(tool_ids) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])