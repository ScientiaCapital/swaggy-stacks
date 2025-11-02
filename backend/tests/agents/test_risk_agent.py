"""Tests for Risk Agent"""
import pytest
from app.agents.risk_agent import RiskAgent


@pytest.mark.asyncio
async def test_risk_agent_creation():
    """Test Risk Agent can be instantiated"""
    agent = RiskAgent()
    assert agent.name == "Risk Agent"


@pytest.mark.asyncio
async def test_risk_agent_validate_trade():
    """Test Risk Agent can validate trade risk"""
    agent = RiskAgent()

    state = {
        "recommended_strategy": "bull_call_spread",
        "strategy_params": {
            "symbol": "AAPL",
            "quantity": 10,
            "max_risk": 5000
        },
        "portfolio": {
            "cash": 100000,
            "total_value": 150000
        }
    }

    result = await agent.invoke(state)

    assert "risk_approved" in result
    assert isinstance(result["risk_approved"], bool)
    assert "position_size" in result
    assert "risk_assessment" in result
