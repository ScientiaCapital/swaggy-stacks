"""
WebSocket service for real-time AI agent coordination and decision streaming
Extends existing infrastructure for agent-specific communication patterns
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import structlog
from fastapi import WebSocket

logger = structlog.get_logger(__name__)


@dataclass
class AgentDecisionUpdate:
    """Real-time agent decision stream update"""

    agent_id: str
    agent_type: (
        str  # market_analyst, risk_advisor, strategy_optimizer, performance_coach
    )
    symbol: str
    decision: str  # BUY, SELL, HOLD
    confidence: float
    reasoning: str
    timestamp: str
    metadata: Dict[str, Any] = None
    tool_calls: List[str] = None


@dataclass
class ToolExecutionResult:
    """Tool execution feedback result"""

    agent_id: str
    tool_name: str
    execution_id: str
    status: str  # success, failed, timeout
    result: Any
    execution_time_ms: float
    timestamp: str
    error_message: Optional[str] = None


@dataclass
class AgentCoordinationMessage:
    """Inter-agent coordination message"""

    sender_agent_id: str
    recipient_agent_id: Optional[str]  # None for broadcast
    message_type: str  # consensus_request, disagreement, coordination
    payload: Dict[str, Any]
    timestamp: str
    requires_response: bool = False


@dataclass
class AgentStatusUpdate:
    """Agent status and health update"""

    agent_id: str
    agent_type: str
    status: str  # active, idle, processing, error
    current_task: Optional[str] = None
    queue_size: int = 0
    last_decision_time: Optional[str] = None
    performance_metrics: Dict[str, float] = None
    timestamp: str = ""


class AgentCoordinationManager:
    """Enhanced connection manager for AI agent coordination"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.agent_subscriptions: Dict[WebSocket, Dict[str, Set[str]]] = {}
        self.agent_status_cache: Dict[str, AgentStatusUpdate] = {}
        self.decision_history: Dict[str, List[AgentDecisionUpdate]] = {}
        self.coordination_channels: Dict[str, Set[WebSocket]] = {
            "all_agents": set(),
            "market_analysts": set(),
            "risk_advisors": set(),
            "strategy_optimizers": set(),
            "performance_coaches": set(),
        }

    async def connect(self, websocket: WebSocket, client_type: str = "dashboard"):
        """Accept new WebSocket connection for agent coordination"""
        await websocket.accept()
        self.active_connections.add(websocket)

        # Initialize subscriptions
        self.agent_subscriptions[websocket] = {
            "agent_decisions": set(),  # specific agent IDs
            "tool_feedback": set(),  # specific agent IDs or "all"
            "coordination": set(),  # channel names
            "status_updates": set(),  # agent types or "all"
        }

        logger.info(
            "Agent coordination WebSocket established",
            client_type=client_type,
            connections=len(self.active_connections),
        )

        # Send initial status data
        await self._send_initial_agent_status(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection and cleanup"""
        self.active_connections.discard(websocket)
        self.agent_subscriptions.pop(websocket, None)

        # Remove from coordination channels
        for channel_subscribers in self.coordination_channels.values():
            channel_subscribers.discard(websocket)

        logger.info(
            "Agent coordination WebSocket disconnected",
            connections=len(self.active_connections),
        )

    async def subscribe_to_agent_decisions(
        self, websocket: WebSocket, agent_ids: List[str]
    ):
        """Subscribe to specific agent decision streams"""
        if websocket not in self.agent_subscriptions:
            return

        self.agent_subscriptions[websocket]["agent_decisions"].update(agent_ids)
        logger.info("Subscribed to agent decisions", agent_ids=agent_ids)

    async def subscribe_to_coordination_channel(
        self, websocket: WebSocket, channel: str
    ):
        """Subscribe to inter-agent coordination channel"""
        if channel in self.coordination_channels:
            self.coordination_channels[channel].add(websocket)
            if websocket in self.agent_subscriptions:
                self.agent_subscriptions[websocket]["coordination"].add(channel)
            logger.info("Subscribed to coordination channel", channel=channel)

    async def broadcast_agent_decision(self, decision: AgentDecisionUpdate):
        """Broadcast agent decision to subscribed clients"""
        message = {
            "type": "agent_decision",
            "data": asdict(decision),
            "timestamp": datetime.now().isoformat(),
        }

        # Cache decision
        if decision.symbol not in self.decision_history:
            self.decision_history[decision.symbol] = []
        self.decision_history[decision.symbol].append(decision)

        # Keep only last 100 decisions per symbol
        if len(self.decision_history[decision.symbol]) > 100:
            self.decision_history[decision.symbol] = self.decision_history[
                decision.symbol
            ][-100:]

        await self._broadcast_to_agent_subscribers(
            "agent_decisions", decision.agent_id, message
        )

    async def broadcast_tool_execution_result(self, result: ToolExecutionResult):
        """Broadcast tool execution results for feedback loops"""
        message = {
            "type": "tool_execution",
            "data": asdict(result),
            "timestamp": datetime.now().isoformat(),
        }

        await self._broadcast_to_agent_subscribers(
            "tool_feedback", result.agent_id, message
        )

    async def broadcast_coordination_message(
        self, coordination: AgentCoordinationMessage
    ):
        """Broadcast inter-agent coordination messages"""
        message = {
            "type": "agent_coordination",
            "data": asdict(coordination),
            "timestamp": datetime.now().isoformat(),
        }

        # Send to appropriate coordination channel
        channel_name = coordination.recipient_agent_id or "all_agents"
        await self._broadcast_to_coordination_channel(channel_name, message)

    async def broadcast_agent_status(self, status: AgentStatusUpdate):
        """Broadcast agent status updates"""
        message = {
            "type": "agent_status",
            "data": asdict(status),
            "timestamp": datetime.now().isoformat(),
        }

        # Cache status
        self.agent_status_cache[status.agent_id] = status

        await self._broadcast_to_agent_subscribers(
            "status_updates", status.agent_type, message
        )

    async def send_direct_message(
        self, websocket: WebSocket, message_type: str, data: Any
    ):
        """Send direct message to specific WebSocket connection"""
        message = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.warning("Failed to send direct message", error=str(e))
            self.disconnect(websocket)

    async def _broadcast_to_agent_subscribers(
        self, subscription_type: str, identifier: str, message: dict
    ):
        """Broadcast to clients subscribed to specific agent events"""
        disconnected = []

        for websocket in self.active_connections.copy():
            if websocket in self.agent_subscriptions:
                subscribed_items = self.agent_subscriptions[websocket].get(
                    subscription_type, set()
                )
                if identifier in subscribed_items or "all" in subscribed_items:
                    try:
                        await websocket.send_text(json.dumps(message))
                    except Exception as e:
                        logger.warning(
                            "Failed to broadcast agent message", error=str(e)
                        )
                        disconnected.append(websocket)

        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws)

    async def _broadcast_to_coordination_channel(self, channel: str, message: dict):
        """Broadcast to coordination channel subscribers"""
        if channel not in self.coordination_channels:
            return

        disconnected = []
        for websocket in self.coordination_channels[channel].copy():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(
                    "Failed to broadcast coordination message",
                    channel=channel,
                    error=str(e),
                )
                disconnected.append(websocket)

        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws)

    async def _send_initial_agent_status(self, websocket: WebSocket):
        """Send cached agent status to newly connected client"""
        try:
            for status in self.agent_status_cache.values():
                message = {
                    "type": "agent_status",
                    "data": asdict(status),
                    "timestamp": datetime.now().isoformat(),
                }
                await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.warning("Failed to send initial agent status", error=str(e))

    def get_agent_decision_history(
        self, symbol: str, limit: int = 50
    ) -> List[AgentDecisionUpdate]:
        """Get recent decision history for a symbol"""
        if symbol not in self.decision_history:
            return []
        return self.decision_history[symbol][-limit:]

    def get_active_agents(self) -> List[AgentStatusUpdate]:
        """Get list of currently active agents"""
        return [
            status
            for status in self.agent_status_cache.values()
            if status.status == "active"
        ]

    async def request_agent_consensus(
        self, symbol: str, decision_context: Dict[str, Any]
    ) -> str:
        """Request consensus from all active agents"""
        consensus_id = f"consensus_{symbol}_{datetime.now().timestamp()}"

        coordination_message = AgentCoordinationMessage(
            sender_agent_id="coordination_manager",
            recipient_agent_id=None,  # broadcast
            message_type="consensus_request",
            payload={
                "consensus_id": consensus_id,
                "symbol": symbol,
                "context": decision_context,
            },
            timestamp=datetime.now().isoformat(),
            requires_response=True,
        )

        await self.broadcast_coordination_message(coordination_message)
        return consensus_id


# Global manager instance
agent_coordination_manager = AgentCoordinationManager()
