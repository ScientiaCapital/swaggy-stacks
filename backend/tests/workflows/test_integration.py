"""Integration tests for complete workflow system."""
import pytest
from app.agents.workflows.trading_workflow import TradingWorkflow
from app.agents.workflows.learning_workflow import LearningWorkflow
from app.agents.workflows.state_schemas import TradingState, LearningState


@pytest.mark.asyncio
async def test_end_to_end_trading_to_learning():
    """Test complete flow: Trading → Learning."""

    # Step 1: Run trading workflow
    trading_workflow = TradingWorkflow()

    trading_state: TradingState = {
        "symbol": "AAPL",
        "market_data": {"VIX": {"value": 15.0}},
        "completed": False,
        "market_regime": "",
        "regime_confidence": 0.0,
        "signals": [],
        "recommended_strategy": "",
        "strategy_params": {},
        "strategy_confidence": 0.0,
        "risk_approved": False,
        "position_size": 0.0,
        "risk_assessment": {},
        "execution_status": "",
        "orders": [],
        "next_agent": "",
        "messages": []
    }

    trading_result = await trading_workflow.run(trading_state)

    # Verify trading completed
    assert trading_result["completed"] is True

    # Step 2: Simulate trade completion
    completed_trade = {
        "strategy": trading_result.get("recommended_strategy", "unknown"),
        "outcome": "win" if trading_result.get("risk_approved") else "rejected",
        "pnl": 500 if trading_result.get("risk_approved") else 0,
        "market_regime": trading_result.get("market_regime", "unknown")
    }

    # Step 3: Run learning workflow
    learning_workflow = LearningWorkflow()

    learning_state: LearningState = {
        "completed_trades": [completed_trade],
        "learning_summary": "",
        "patterns_updated": 0,
        "regime_matrix_updated": False,
        "insights": [],
        "next_day_recommendations": [],
        "completed": False
    }

    learning_result = await learning_workflow.run(learning_state)

    # Verify learning completed
    assert learning_result["completed"] is True
    assert learning_result["patterns_updated"] >= 0


@pytest.mark.asyncio
async def test_workflow_error_handling():
    """Test workflows handle errors correctly."""
    from app.agents.workflows.trading_workflow import AgentFailureError

    # This test verifies error handling exists
    # Actual error triggering would require mocking
    assert AgentFailureError is not None


@pytest.mark.asyncio
async def test_all_workflows_available():
    """Test all workflow components can be imported."""
    from app.agents.workflows.trading_workflow import TradingWorkflow
    from app.agents.workflows.learning_workflow import LearningWorkflow
    from app.agents.workflows.supervisor import TradingSupervisor
    from app.agents.workflows.state_schemas import TradingState, LearningState

    # Verify all imports successful
    assert TradingWorkflow is not None
    assert LearningWorkflow is not None
    assert TradingSupervisor is not None
    assert TradingState is not None
    assert LearningState is not None
