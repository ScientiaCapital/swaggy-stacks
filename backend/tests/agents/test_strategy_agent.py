"""Tests for Strategy Agent"""
import pytest
from app.agents.strategy_agent import StrategyAgent


@pytest.mark.asyncio
async def test_strategy_agent_creation():
    """Test Strategy Agent can be instantiated"""
    agent = StrategyAgent()
    assert agent.name == "Strategy Agent"


@pytest.mark.asyncio
async def test_strategy_agent_select_strategy():
    """Test Strategy Agent can select options strategy"""
    agent = StrategyAgent()

    state = {
        "market_regime": "bull",
        "regime_confidence": 0.85,
        "signals": [
            {"symbol": "AAPL", "type": "buy", "confidence": 0.8}
        ]
    }

    result = await agent.invoke(state)

    assert "recommended_strategy" in result
    assert result["recommended_strategy"] in [
        "bull_call_spread", "bear_put_spread", "iron_butterfly",
        "long_straddle", "covered_call", "protective_put",
        "calendar_spread"
    ]
    assert "strategy_params" in result
    assert "confidence_score" in result
