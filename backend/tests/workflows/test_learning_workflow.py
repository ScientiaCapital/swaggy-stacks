"""Tests for Learning Workflow."""
import pytest
from app.agents.workflows.learning_workflow import LearningWorkflow
from app.agents.workflows.state_schemas import LearningState


@pytest.mark.asyncio
async def test_learning_workflow_creation():
    """Test LearningWorkflow can be instantiated."""
    workflow = LearningWorkflow()

    assert workflow is not None
    assert hasattr(workflow, 'learning_agent')
    assert hasattr(workflow, 'graph')


@pytest.mark.asyncio
async def test_learning_workflow_process_trades():
    """Test learning workflow processes completed trades."""
    workflow = LearningWorkflow()

    initial_state: LearningState = {
        "completed_trades": [
            {
                "strategy": "bull_call_spread",
                "outcome": "win",
                "pnl": 500,
                "market_regime": "bull"
            },
            {
                "strategy": "bear_put_spread",
                "outcome": "loss",
                "pnl": -200,
                "market_regime": "bear"
            }
        ],
        "learning_summary": "",
        "patterns_updated": 0,
        "regime_matrix_updated": False,
        "insights": [],
        "next_day_recommendations": [],
        "completed": False
    }

    result = await workflow.run(initial_state)

    # Verify learning agent processed trades
    assert result["learning_summary"]
    assert result["patterns_updated"] > 0
    assert result["completed"] is True


@pytest.mark.asyncio
async def test_learning_workflow_no_trades():
    """Test learning workflow handles empty trade list."""
    workflow = LearningWorkflow()

    initial_state: LearningState = {
        "completed_trades": [],
        "learning_summary": "",
        "patterns_updated": 0,
        "regime_matrix_updated": False,
        "insights": [],
        "next_day_recommendations": [],
        "completed": False
    }

    result = await workflow.run(initial_state)

    # Verify graceful handling
    assert "No trades" in result["learning_summary"]
    assert result["patterns_updated"] == 0
    assert result["completed"] is True
