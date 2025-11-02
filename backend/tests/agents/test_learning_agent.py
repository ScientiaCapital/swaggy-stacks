"""Tests for Learning Agent"""
import pytest
from app.agents.learning_agent import LearningAgent


@pytest.mark.asyncio
async def test_learning_agent_creation():
    """Test Learning Agent can be instantiated"""
    agent = LearningAgent()
    assert agent.name == "Learning Agent"
    assert agent.description == "Post-market learning and continuous improvement"


@pytest.mark.asyncio
async def test_learning_agent_process_experience():
    """Test Learning Agent can process trade experiences"""
    agent = LearningAgent()

    state = {
        "completed_trades": [
            {
                "strategy": "bull_call_spread",
                "outcome": "win",
                "pnl": 500,
                "market_regime": "bull"
            }
        ]
    }

    result = await agent.invoke(state)

    assert "learning_summary" in result
    assert "patterns_updated" in result
    assert "regime_matrix_updated" in result
    assert "insights" in result
    assert isinstance(result["insights"], list)
    assert "next_day_recommendations" in result
    assert isinstance(result["next_day_recommendations"], list)
