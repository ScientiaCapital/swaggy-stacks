"""Tests for Research Agent"""
import pytest
from app.agents.research_agent import ResearchAgent


@pytest.mark.asyncio
async def test_research_agent_creation():
    """Test Research Agent can be instantiated"""
    agent = ResearchAgent()
    assert agent.name == "Research Agent"
    assert agent.model_config is not None


@pytest.mark.asyncio
async def test_research_agent_analyze_market():
    """Test Research Agent can analyze market conditions"""
    agent = ResearchAgent()

    state = {
        "market_data": {
            "SPY": {"price": 450.00, "change": 0.5},
            "VIX": {"value": 15.2}
        }
    }

    result = await agent.invoke(state)

    assert "market_regime" in result
    assert result["market_regime"] in ["bull", "bear", "volatile", "sideways"]
    assert "signals" in result
    assert isinstance(result["signals"], list)
