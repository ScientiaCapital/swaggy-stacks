"""
Context Builder for LangGraph Trading Workflows

This module provides state management and context building for LangGraph trading workflows,
maintaining conversation history, market data, and decision context across workflow steps.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.monitoring.metrics import PrometheusMetrics
from app.rag.services.memory_manager import AgentMemoryManager, MemoryQuery, MemoryType
from app.rag.types import TradingSignal
from app.trading.trading_manager import get_trading_manager

logger = logging.getLogger(__name__)


@dataclass
class MarketDataContext:
    """Market data context for trading decisions"""

    symbol: str
    current_price: Optional[float] = None
    price_history: List[float] = field(default_factory=list)
    volume_history: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    market_hours: bool = True
    volatility: Optional[float] = None
    trend_direction: Optional[str] = None


@dataclass
class RiskMetrics:
    """Risk assessment metrics"""

    portfolio_value: float
    available_cash: float
    total_exposure: float
    position_concentration: Dict[str, float] = field(default_factory=dict)
    var_1d: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    risk_score: float = 0.5  # 0.0 (low risk) to 1.0 (high risk)


@dataclass
class DecisionContext:
    """Trading decision context"""

    primary_strategy: str
    confidence_level: float
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    opportunity_factors: List[str] = field(default_factory=list)
    decision_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionPlan:
    """Execution plan for trading decisions"""

    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: Optional[float] = None
    order_type: str = "MARKET"
    price_limit: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_in_force: str = "DAY"
    execution_priority: str = "NORMAL"  # LOW, NORMAL, HIGH
    risk_validated: bool = False


class TradingState(TypedDict):
    """
    LangGraph state for trading workflows
    All state fields are maintained across workflow nodes
    """

    # Core identifiers
    session_id: str
    symbol: str
    workflow_type: str

    # Market context
    market_data: MarketDataContext

    # Memory and historical context
    memory_context: List[Dict[str, Any]]
    similar_patterns: List[Dict[str, Any]]

    # Strategy analysis
    strategy_signals: List[TradingSignal]
    strategy_consensus: Dict[str, float]  # strategy_name -> confidence

    # Risk assessment
    risk_metrics: RiskMetrics
    risk_approval: bool

    # Decision making
    decision_context: DecisionContext
    final_decision: Optional[str]

    # Execution
    execution_plan: Optional[ExecutionPlan]
    execution_status: Optional[str]

    # Conversation and flow control
    conversation_history: List[BaseMessage]
    current_step: str
    next_steps: List[str]
    workflow_metadata: Dict[str, Any]

    # Error handling
    errors: List[str]
    warnings: List[str]


class TradingContextBuilder:
    """
    Advanced context builder for LangGraph trading workflows
    Manages state transitions, data aggregation, and conversation continuity
    """

    def __init__(self):
        self.trading_manager = None
        self.memory_manager = None
        self.metrics = PrometheusMetrics()

    async def initialize(self) -> None:
        """Initialize context builder with dependencies"""
        self.trading_manager = get_trading_manager()
        await self.trading_manager.initialize()

        self.memory_manager = AgentMemoryManager()
        await self.memory_manager.initialize()

        logger.info("Trading Context Builder initialized")

    async def create_initial_state(
        self,
        symbol: str,
        workflow_type: str = "standard_analysis",
        session_id: Optional[str] = None,
    ) -> TradingState:
        """Create initial state for a trading workflow"""
        if session_id is None:
            session_id = f"trading_{symbol}_{int(datetime.now().timestamp())}"

        # Get initial market data
        market_data = await self._build_market_context(symbol)

        # Get initial risk metrics
        risk_metrics = await self._build_risk_context()

        state: TradingState = {
            "session_id": session_id,
            "symbol": symbol,
            "workflow_type": workflow_type,
            "market_data": market_data,
            "memory_context": [],
            "similar_patterns": [],
            "strategy_signals": [],
            "strategy_consensus": {},
            "risk_metrics": risk_metrics,
            "risk_approval": False,
            "decision_context": DecisionContext(
                primary_strategy="", confidence_level=0.0
            ),
            "final_decision": None,
            "execution_plan": None,
            "execution_status": None,
            "conversation_history": [
                SystemMessage(content=f"Starting trading analysis for {symbol}")
            ],
            "current_step": "initialization",
            "next_steps": ["market_analysis"],
            "workflow_metadata": {
                "created_at": datetime.now().isoformat(),
                "workflow_version": "1.0",
            },
            "errors": [],
            "warnings": [],
        }

        logger.info(
            f"Created initial trading state for {symbol} (session: {session_id})"
        )
        return state

    async def update_market_context(self, state: TradingState) -> TradingState:
        """Update market data context"""
        try:
            market_data = await self._build_market_context(state["symbol"])
            state["market_data"] = market_data

            # Add conversation update
            state["conversation_history"].append(
                AIMessage(content=f"Updated market data for {state['symbol']}")
            )

        except Exception as e:
            error_msg = f"Failed to update market context: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)

        return state

    async def add_memory_context(
        self, state: TradingState, query_text: str
    ) -> TradingState:
        """Add relevant memory context to state"""
        try:
            if not self.memory_manager:
                logger.warning("Memory manager not available")
                return state

            # Query for similar trading patterns
            query = MemoryQuery(
                query_text=query_text,
                memory_types=[MemoryType.TRADING_PATTERN, MemoryType.MARKET_CONDITION],
                limit=5,
                agent_name="context_builder",
            )

            memories = await self.memory_manager.query_memories(query)

            # Convert memories to context format
            memory_context = []
            for memory in memories:
                context_item = {
                    "memory_id": memory.id,
                    "content": memory.content,
                    "similarity": memory.metadata.get("similarity", 0.0),
                    "outcome": memory.metadata.get("outcome", "unknown"),
                    "timestamp": memory.timestamp.isoformat(),
                }
                memory_context.append(context_item)

            state["memory_context"] = memory_context

            # Add to conversation
            if memory_context:
                state["conversation_history"].append(
                    AIMessage(
                        content=f"Found {len(memory_context)} similar historical patterns"
                    )
                )

        except Exception as e:
            error_msg = f"Failed to add memory context: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)

        return state

    async def add_strategy_signal(
        self, state: TradingState, signal: TradingSignal
    ) -> TradingState:
        """Add a strategy signal to the state"""
        state["strategy_signals"].append(signal)

        # Update strategy consensus
        strategy_name = f"{signal.agent_type}_{signal.strategy_name}"
        state["strategy_consensus"][strategy_name] = signal.confidence

        # Add to conversation
        state["conversation_history"].append(
            AIMessage(
                content=f"{strategy_name} signal: {signal.action} ({signal.confidence:.1%} confidence)"
            )
        )

        return state

    async def update_risk_context(self, state: TradingState) -> TradingState:
        """Update risk assessment context"""
        try:
            risk_metrics = await self._build_risk_context()
            state["risk_metrics"] = risk_metrics

            # Check if risk is acceptable
            state["risk_approval"] = await self._assess_risk_approval(
                risk_metrics, state
            )

            # Add to conversation
            risk_status = "approved" if state["risk_approval"] else "requires review"
            state["conversation_history"].append(
                AIMessage(content=f"Risk assessment updated - Status: {risk_status}")
            )

        except Exception as e:
            error_msg = f"Failed to update risk context: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)

        return state

    async def create_decision_context(self, state: TradingState) -> TradingState:
        """Create decision context from all available information"""
        try:
            # Analyze strategy consensus
            if state["strategy_consensus"]:
                primary_strategy = max(
                    state["strategy_consensus"], key=state["strategy_consensus"].get
                )
                avg_confidence = np.mean(list(state["strategy_consensus"].values()))
            else:
                primary_strategy = "no_consensus"
                avg_confidence = 0.0

            # Gather supporting evidence
            supporting_evidence = []
            contradicting_evidence = []

            for signal in state["strategy_signals"]:
                evidence = f"{signal.strategy_name}: {signal.reasoning}"
                if signal.confidence > 0.6:
                    supporting_evidence.append(evidence)
                else:
                    contradicting_evidence.append(evidence)

            # Risk factors
            risk_factors = []
            if state["risk_metrics"].risk_score > 0.7:
                risk_factors.append("High portfolio risk score")
            if (
                state["risk_metrics"].total_exposure
                > state["risk_metrics"].portfolio_value * 0.8
            ):
                risk_factors.append("High portfolio exposure")

            # Opportunity factors
            opportunity_factors = []
            if (
                state["market_data"].volatility
                and state["market_data"].volatility > 0.02
            ):
                opportunity_factors.append("High volatility environment")
            if len([s for s in state["strategy_signals"] if s.confidence > 0.8]) > 1:
                opportunity_factors.append("Multiple high-confidence signals")

            decision_context = DecisionContext(
                primary_strategy=primary_strategy,
                confidence_level=avg_confidence,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                risk_factors=risk_factors,
                opportunity_factors=opportunity_factors,
            )

            state["decision_context"] = decision_context

            # Add to conversation
            state["conversation_history"].append(
                AIMessage(
                    content=f"Decision context created - Primary strategy: {primary_strategy} ({avg_confidence:.1%} confidence)"
                )
            )

        except Exception as e:
            error_msg = f"Failed to create decision context: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)

        return state

    async def create_execution_plan(
        self, state: TradingState, action: str
    ) -> TradingState:
        """Create execution plan for the decided action"""
        try:
            if action.upper() == "HOLD":
                state["execution_plan"] = None
                state["final_decision"] = "HOLD"
                return state

            # Calculate position size based on risk metrics
            portfolio_value = state["risk_metrics"].portfolio_value
            risk_per_trade = 0.02  # 2% risk per trade
            max_position_value = portfolio_value * risk_per_trade

            current_price = state["market_data"].current_price
            if current_price:
                max_quantity = max_position_value / current_price
            else:
                max_quantity = 0

            # Create execution plan
            execution_plan = ExecutionPlan(
                symbol=state["symbol"],
                action=action.upper(),
                quantity=max_quantity,
                order_type="MARKET",
                risk_validated=state["risk_approval"],
            )

            state["execution_plan"] = execution_plan
            state["final_decision"] = action.upper()

            # Add to conversation
            state["conversation_history"].append(
                AIMessage(
                    content=f"Execution plan created: {action} {max_quantity:.2f} shares of {state['symbol']}"
                )
            )

        except Exception as e:
            error_msg = f"Failed to create execution plan: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)

        return state

    async def update_workflow_step(
        self, state: TradingState, current_step: str, next_steps: List[str] = None
    ) -> TradingState:
        """Update workflow progression"""
        state["current_step"] = current_step
        if next_steps:
            state["next_steps"] = next_steps

        # Record metrics
        self.metrics.record_workflow_step(
            workflow_type=state["workflow_type"],
            step=current_step,
            session_id=state["session_id"],
        )

        return state

    async def add_user_message(self, state: TradingState, message: str) -> TradingState:
        """Add user message to conversation history"""
        state["conversation_history"].append(HumanMessage(content=message))
        return state

    async def add_ai_message(self, state: TradingState, message: str) -> TradingState:
        """Add AI message to conversation history"""
        state["conversation_history"].append(AIMessage(content=message))
        return state

    async def _build_market_context(self, symbol: str) -> MarketDataContext:
        """Build market data context"""
        if not self.trading_manager:
            return MarketDataContext(symbol=symbol)

        try:
            # Get current price
            current_price = await self.trading_manager.get_current_price(symbol)

            # Get market analysis for additional context
            analysis = await self.trading_manager.get_market_analysis(symbol)

            # Calculate basic volatility from analysis if available
            volatility = None
            trend_direction = None

            if isinstance(analysis, dict):
                volatility = analysis.get("volatility")
                trend_direction = analysis.get("trend_direction")

            return MarketDataContext(
                symbol=symbol,
                current_price=current_price,
                volatility=volatility,
                trend_direction=trend_direction,
                market_hours=True,  # Simplified for now
            )

        except Exception as e:
            logger.error(f"Error building market context for {symbol}: {e}")
            return MarketDataContext(symbol=symbol)

    async def _build_risk_context(self) -> RiskMetrics:
        """Build risk metrics context"""
        if not self.trading_manager:
            return RiskMetrics(
                portfolio_value=100000, available_cash=100000, total_exposure=0
            )

        try:
            # Get portfolio status
            portfolio_status = await self.trading_manager.get_portfolio_status()

            if isinstance(portfolio_status, dict):
                portfolio_value = float(portfolio_status.get("total_value", 100000))
                available_cash = float(
                    portfolio_status.get("available_cash", portfolio_value)
                )

                # Calculate exposure from positions
                positions = portfolio_status.get("positions", {})
                total_exposure = sum(
                    abs(pos.get("market_value", 0)) for pos in positions.values()
                )

                # Calculate position concentration
                position_concentration = {}
                if total_exposure > 0:
                    for symbol, pos in positions.items():
                        market_value = abs(pos.get("market_value", 0))
                        position_concentration[symbol] = market_value / total_exposure

                return RiskMetrics(
                    portfolio_value=portfolio_value,
                    available_cash=available_cash,
                    total_exposure=total_exposure,
                    position_concentration=position_concentration,
                    risk_score=(
                        min(total_exposure / portfolio_value, 1.0)
                        if portfolio_value > 0
                        else 0.0
                    ),
                )
            else:
                return RiskMetrics(
                    portfolio_value=100000, available_cash=100000, total_exposure=0
                )

        except Exception as e:
            logger.error(f"Error building risk context: {e}")
            return RiskMetrics(
                portfolio_value=100000, available_cash=100000, total_exposure=0
            )

    async def _assess_risk_approval(
        self, risk_metrics: RiskMetrics, state: TradingState
    ) -> bool:
        """Assess if risk is acceptable for execution"""
        try:
            # Check portfolio risk score
            if risk_metrics.risk_score > 0.8:
                state["warnings"].append("High portfolio risk score")
                return False

            # Check available cash
            if risk_metrics.available_cash < risk_metrics.portfolio_value * 0.1:
                state["warnings"].append("Low available cash")
                return False

            # Check position concentration
            max_concentration = (
                max(risk_metrics.position_concentration.values())
                if risk_metrics.position_concentration
                else 0
            )
            if max_concentration > 0.3:
                state["warnings"].append("High position concentration")
                return False

            return True

        except Exception as e:
            logger.error(f"Error assessing risk approval: {e}")
            return False


# Global context builder instance
_context_builder_instance: Optional[TradingContextBuilder] = None


async def get_context_builder() -> TradingContextBuilder:
    """Get the global context builder instance"""
    global _context_builder_instance

    if _context_builder_instance is None:
        _context_builder_instance = TradingContextBuilder()
        await _context_builder_instance.initialize()

    return _context_builder_instance


__all__ = [
    "TradingState",
    "MarketDataContext",
    "RiskMetrics",
    "DecisionContext",
    "ExecutionPlan",
    "TradingContextBuilder",
    "get_context_builder",
]
