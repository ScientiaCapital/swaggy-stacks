"""Tests for Trading Supervisor."""
import pytest
from app.agents.workflows.supervisor import TradingSupervisor


def test_supervisor_creation():
    """Test TradingSupervisor can be instantiated."""
    supervisor = TradingSupervisor()

    assert supervisor is not None
    assert hasattr(supervisor, 'model')
    assert hasattr(supervisor, 'tools')
    assert len(supervisor.tools) == 4  # Research, Strategy, Risk, Execution


def test_supervisor_agent_to_tool_conversion():
    """Test agents are converted to tools correctly."""
    supervisor = TradingSupervisor()

    tool_names = [tool.name for tool in supervisor.tools]

    assert "research_agent" in tool_names
    assert "strategy_agent" in tool_names
    assert "risk_agent" in tool_names
    assert "execution_agent" in tool_names
