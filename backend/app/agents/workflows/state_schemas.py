"""
State schemas for LangGraph workflows.
"""
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage


class TradingState(TypedDict):
    """State for trading workflow."""

    # Input
    symbol: str
    market_data: Dict[str, Any]

    # Research Agent output
    market_regime: str  # "bull", "bear", "volatile", "sideways"
    regime_confidence: float
    signals: List[Dict[str, Any]]

    # Strategy Agent output
    recommended_strategy: str
    strategy_params: Dict[str, Any]
    strategy_confidence: float

    # Risk Agent output
    risk_approved: bool
    position_size: float
    risk_assessment: Dict[str, Any]

    # Execution Agent output
    execution_status: str  # "pending", "filled", "rejected", "error"
    orders: List[Dict[str, Any]]

    # Supervisor control
    next_agent: str
    messages: List[BaseMessage]
    completed: bool


class LearningState(TypedDict):
    """State for learning workflow."""

    completed_trades: List[Dict[str, Any]]
    learning_summary: str
    patterns_updated: int
    regime_matrix_updated: bool
    insights: List[str]
    next_day_recommendations: List[str]
    completed: bool
