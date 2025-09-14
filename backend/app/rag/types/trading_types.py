"""
Common trading data types shared across the RAG system
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TradingSignal:
    """Standard trading signal format across all agents"""

    agent_type: str
    strategy_name: str
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    reasoning: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MarketContext:
    """Market context data for agent decision making"""

    symbol: str
    current_price: float
    volume: float
    timestamp: datetime
    market_session: str  # pre_market, market_hours, after_hours
    volatility: Optional[float] = None
    trend: Optional[str] = None
    sentiment: Optional[str] = None
    technical_indicators: Dict[str, Any] = None

    def __post_init__(self):
        if self.technical_indicators is None:
            self.technical_indicators = {}


@dataclass
class AgentState:
    """Agent state information for context building"""

    agent_id: str
    current_positions: Dict[str, Any]
    available_capital: float
    risk_exposure: float
    last_action_timestamp: Optional[datetime] = None
    performance_metrics: Dict[str, float] = None
    active_strategies: List[str] = None

    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.active_strategies is None:
            self.active_strategies = []
