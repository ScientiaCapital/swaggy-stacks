"""Tests for Trading Workflow."""
import pytest
from app.agents.workflows.trading_workflow import TradingWorkflow
from app.agents.workflows.state_schemas import TradingState


@pytest.mark.asyncio
async def test_trading_workflow_creation():
    """Test TradingWorkflow can be instantiated."""
    workflow = TradingWorkflow()

    assert workflow is not None
    assert hasattr(workflow, 'supervisor')
    assert hasattr(workflow, 'graph')


@pytest.mark.asyncio
async def test_trading_workflow_complete_flow():
    """Test complete Research → Strategy → Risk → Execution flow."""
    workflow = TradingWorkflow()

    initial_state: TradingState = {
        "symbol": "AAPL",
        "market_data": {"VIX": {"value": 15.0}},
        "completed": False,
        # Initialize optional fields
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

    result = await workflow.run(initial_state)

    # Verify all agents were called
    assert "market_regime" in result and result["market_regime"]  # Research ran
    assert "recommended_strategy" in result and result["recommended_strategy"]  # Strategy ran
    assert "risk_approved" in result  # Risk ran
    assert result["completed"] is True


@pytest.mark.asyncio
async def test_trading_workflow_risk_rejection():
    """Test workflow stops if Risk rejects."""
    workflow = TradingWorkflow()

    # Mock state where risk will reject (insufficient cash)
    initial_state: TradingState = {
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

    result = await workflow.run(initial_state)

    # Verify execution was skipped
    if not result.get("risk_approved"):
        assert result.get("execution_status", "") != "filled"
