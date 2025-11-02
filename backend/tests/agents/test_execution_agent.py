"""
Tests for Execution Agent
"""
import pytest
from app.agents.execution_agent import ExecutionAgent


def test_execution_agent_creation():
    """Test Execution Agent can be instantiated"""
    agent = ExecutionAgent()
    assert agent.name == "Execution Agent"
    assert agent.description == "Order execution and fill monitoring"


@pytest.mark.asyncio
async def test_execution_agent_execute_trade():
    """Test Execution Agent can execute approved trade"""
    agent = ExecutionAgent()

    state = {
        "risk_approved": True,
        "recommended_strategy": "bull_call_spread",
        "strategy_params": {"symbol": "AAPL", "quantity": 5},
        "position_size": 2500
    }

    result = await agent.invoke(state)

    assert "execution_status" in result
    assert result["execution_status"] in ["pending", "filled", "rejected", "error"]
    assert "orders" in result
    assert isinstance(result["orders"], list)
