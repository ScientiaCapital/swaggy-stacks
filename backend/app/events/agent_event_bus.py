"""
Agent Event Bus - Event-driven coordination system for AI agents
Extends existing RabbitMQ infrastructure for agent-specific messaging patterns
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog

from app.messaging.rabbitmq_client import RabbitMQClient, TradingMessage
from app.websockets.agent_coordination_socket import (
    AgentCoordinationMessage,
    AgentDecisionUpdate,
    AgentStatusUpdate,
    ToolExecutionResult,
    agent_coordination_manager,
)

logger = structlog.get_logger(__name__)


class AgentEventType(Enum):
    """Event types for agent coordination"""

    MARKET_UPDATE = "agent.market.update"
    DECISION_REQUEST = "agent.decision.request"
    DECISION_RESPONSE = "agent.decision.response"
    TOOL_EXECUTION = "agent.tool.execution"
    TOOL_FEEDBACK = "agent.tool.feedback"
    COORDINATION = "agent.coordination"
    STATUS_UPDATE = "agent.status.update"
    CONSENSUS_REQUEST = "agent.consensus.request"
    CONSENSUS_RESPONSE = "agent.consensus.response"
    ERROR_EVENT = "agent.error.event"


@dataclass
class AgentEvent:
    """Base event structure for agent communication"""

    event_type: str
    agent_id: str
    timestamp: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    priority: int = 0  # 0=normal, 1=high, 2=urgent

    def to_trading_message(
        self, source_service: str = "agent_event_bus"
    ) -> TradingMessage:
        """Convert to TradingMessage for RabbitMQ publishing"""
        return TradingMessage(
            message_type=self.event_type,
            payload={
                "agent_id": self.agent_id,
                "event_payload": self.payload,
                "correlation_id": self.correlation_id,
                "reply_to": self.reply_to,
            },
            timestamp=self.timestamp,
            source_service=source_service,
            priority=self.priority,
            correlation_id=self.correlation_id
            or f"agent_{self.agent_id}_{datetime.now().timestamp()}",
        )

    @classmethod
    def from_trading_message(cls, msg: TradingMessage) -> "AgentEvent":
        """Create AgentEvent from TradingMessage"""
        return cls(
            event_type=msg.message_type,
            agent_id=msg.payload.get("agent_id", "unknown"),
            timestamp=msg.timestamp,
            payload=msg.payload.get("event_payload", {}),
            correlation_id=msg.correlation_id,
            reply_to=msg.payload.get("reply_to"),
            priority=msg.priority,
        )


class AgentEventBus:
    """Event bus for AI agent coordination and communication"""

    def __init__(self, rabbitmq_client: Optional[RabbitMQClient] = None):
        self.rabbitmq_client = rabbitmq_client or RabbitMQClient(
            exchange_name="agent_coordination_exchange"
        )
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.agent_subscriptions: Dict[str, List[str]] = {}  # agent_id -> event_types
        self.active_agents: Dict[str, AgentStatusUpdate] = {}
        self.decision_cache: Dict[str, List[AgentDecisionUpdate]] = {}
        self.tool_feedback_cache: Dict[str, List[ToolExecutionResult]] = {}

    async def initialize(self) -> bool:
        """Initialize the event bus and establish connections"""
        try:
            # Connect to RabbitMQ
            if not await self.rabbitmq_client.connect():
                logger.error("Failed to connect to RabbitMQ for agent event bus")
                return False

            # Set up agent-specific queues and routing
            await self._setup_agent_queues()

            logger.info("Agent Event Bus initialized successfully")
            return True

        except Exception as e:
            logger.error("Failed to initialize Agent Event Bus", error=str(e))
            return False

    async def _setup_agent_queues(self):
        """Set up RabbitMQ queues for different agent event types"""
        agent_queues = {
            "agent_decisions": ["agent.decision.*"],
            "agent_coordination": ["agent.coordination.*", "agent.consensus.*"],
            "agent_tools": ["agent.tool.*"],
            "agent_status": ["agent.status.*"],
            "agent_market": ["agent.market.*"],
            "agent_errors": ["agent.error.*"],
        }

        for queue_name, routing_patterns in agent_queues.items():
            await self.rabbitmq_client.subscribe_to_messages(
                routing_patterns=routing_patterns,
                queue_name=queue_name,
                handler=self._handle_agent_message,
                durable=True,
            )
            logger.info(
                f"Set up agent queue: {queue_name} with patterns: {routing_patterns}"
            )

    async def _handle_agent_message(self, message: TradingMessage):
        """Handle incoming agent messages from RabbitMQ"""
        try:
            agent_event = AgentEvent.from_trading_message(message)

            # Route to appropriate handlers
            if agent_event.event_type in self.event_handlers:
                for handler in self.event_handlers[agent_event.event_type]:
                    try:
                        await handler(agent_event)
                    except Exception as e:
                        logger.error(
                            "Agent event handler failed",
                            event_type=agent_event.event_type,
                            agent_id=agent_event.agent_id,
                            error=str(e),
                        )

            # Broadcast to WebSocket clients if appropriate
            await self._broadcast_to_websocket(agent_event)

        except Exception as e:
            logger.error("Failed to handle agent message", error=str(e))

    async def _broadcast_to_websocket(self, event: AgentEvent):
        """Broadcast agent events to WebSocket clients"""
        try:
            if event.event_type == AgentEventType.DECISION_RESPONSE.value:
                # Convert to AgentDecisionUpdate and broadcast
                decision_update = AgentDecisionUpdate(
                    agent_id=event.agent_id,
                    agent_type=event.payload.get("agent_type", "unknown"),
                    symbol=event.payload.get("symbol", ""),
                    decision=event.payload.get("decision", "HOLD"),
                    confidence=event.payload.get("confidence", 0.0),
                    reasoning=event.payload.get("reasoning", ""),
                    timestamp=event.timestamp,
                    metadata=event.payload.get("metadata", {}),
                    tool_calls=event.payload.get("tool_calls", []),
                )
                await agent_coordination_manager.broadcast_agent_decision(
                    decision_update
                )

            elif event.event_type == AgentEventType.TOOL_FEEDBACK.value:
                # Convert to ToolExecutionResult and broadcast
                tool_result = ToolExecutionResult(
                    agent_id=event.agent_id,
                    tool_name=event.payload.get("tool_name", ""),
                    execution_id=event.payload.get("execution_id", ""),
                    status=event.payload.get("status", "unknown"),
                    result=event.payload.get("result"),
                    execution_time_ms=event.payload.get("execution_time_ms", 0.0),
                    timestamp=event.timestamp,
                    error_message=event.payload.get("error_message"),
                )
                await agent_coordination_manager.broadcast_tool_execution_result(
                    tool_result
                )

            elif event.event_type in [
                AgentEventType.COORDINATION.value,
                AgentEventType.CONSENSUS_REQUEST.value,
            ]:
                # Convert to AgentCoordinationMessage and broadcast
                coord_message = AgentCoordinationMessage(
                    sender_agent_id=event.agent_id,
                    recipient_agent_id=event.payload.get("recipient_agent_id"),
                    message_type=event.payload.get("message_type", "coordination"),
                    payload=event.payload.get("coordination_payload", {}),
                    timestamp=event.timestamp,
                    requires_response=event.payload.get("requires_response", False),
                )
                await agent_coordination_manager.broadcast_coordination_message(
                    coord_message
                )

            elif event.event_type == AgentEventType.STATUS_UPDATE.value:
                # Convert to AgentStatusUpdate and broadcast
                status_update = AgentStatusUpdate(
                    agent_id=event.agent_id,
                    agent_type=event.payload.get("agent_type", "unknown"),
                    status=event.payload.get("status", "unknown"),
                    current_task=event.payload.get("current_task"),
                    queue_size=event.payload.get("queue_size", 0),
                    last_decision_time=event.payload.get("last_decision_time"),
                    performance_metrics=event.payload.get("performance_metrics", {}),
                    timestamp=event.timestamp,
                )
                await agent_coordination_manager.broadcast_agent_status(status_update)

        except Exception as e:
            logger.warning("Failed to broadcast agent event to WebSocket", error=str(e))

    async def publish_event(self, event: AgentEvent) -> bool:
        """Publish agent event to the event bus"""
        try:
            trading_message = event.to_trading_message()
            success = await self.rabbitmq_client.publish_message(
                trading_message, routing_key=event.event_type
            )

            if success:
                logger.debug(
                    "Published agent event",
                    event_type=event.event_type,
                    agent_id=event.agent_id,
                    correlation_id=event.correlation_id,
                )

            return success

        except Exception as e:
            logger.error(
                "Failed to publish agent event",
                event_type=event.event_type,
                agent_id=event.agent_id,
                error=str(e),
            )
            return False

    async def publish_market_update(
        self, agent_id: str, symbol: str, market_data: Dict[str, Any]
    ) -> bool:
        """Publish market update event for agents"""
        event = AgentEvent(
            event_type=AgentEventType.MARKET_UPDATE.value,
            agent_id=agent_id,
            timestamp=datetime.now().isoformat(),
            payload={"symbol": symbol, "market_data": market_data},
        )
        return await self.publish_event(event)

    async def publish_decision_request(
        self,
        agent_id: str,
        symbol: str,
        market_context: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> bool:
        """Request decision from specific agent"""
        event = AgentEvent(
            event_type=AgentEventType.DECISION_REQUEST.value,
            agent_id=agent_id,
            timestamp=datetime.now().isoformat(),
            payload={
                "symbol": symbol,
                "market_context": market_context,
                "requested_at": datetime.now().isoformat(),
            },
            correlation_id=correlation_id,
            priority=1,
        )
        return await self.publish_event(event)

    async def publish_decision_response(
        self,
        agent_id: str,
        agent_type: str,
        symbol: str,
        decision: str,
        confidence: float,
        reasoning: str,
        metadata: Dict[str, Any] = None,
        tool_calls: List[str] = None,
        correlation_id: Optional[str] = None,
    ) -> bool:
        """Publish agent decision response"""
        event = AgentEvent(
            event_type=AgentEventType.DECISION_RESPONSE.value,
            agent_id=agent_id,
            timestamp=datetime.now().isoformat(),
            payload={
                "agent_type": agent_type,
                "symbol": symbol,
                "decision": decision,
                "confidence": confidence,
                "reasoning": reasoning,
                "metadata": metadata or {},
                "tool_calls": tool_calls or [],
            },
            correlation_id=correlation_id,
            priority=1,
        )
        return await self.publish_event(event)

    async def publish_tool_execution(
        self,
        agent_id: str,
        tool_name: str,
        execution_id: str,
        status: str,
        result: Any,
        execution_time_ms: float,
        error_message: Optional[str] = None,
    ) -> bool:
        """Publish tool execution feedback"""
        event = AgentEvent(
            event_type=AgentEventType.TOOL_FEEDBACK.value,
            agent_id=agent_id,
            timestamp=datetime.now().isoformat(),
            payload={
                "tool_name": tool_name,
                "execution_id": execution_id,
                "status": status,
                "result": result,
                "execution_time_ms": execution_time_ms,
                "error_message": error_message,
            },
            priority=0,
        )
        return await self.publish_event(event)

    async def publish_coordination_message(
        self,
        sender_agent_id: str,
        recipient_agent_id: Optional[str],
        message_type: str,
        payload: Dict[str, Any],
        requires_response: bool = False,
    ) -> bool:
        """Publish inter-agent coordination message"""
        event = AgentEvent(
            event_type=AgentEventType.COORDINATION.value,
            agent_id=sender_agent_id,
            timestamp=datetime.now().isoformat(),
            payload={
                "recipient_agent_id": recipient_agent_id,
                "message_type": message_type,
                "coordination_payload": payload,
                "requires_response": requires_response,
            },
            priority=1 if requires_response else 0,
        )
        return await self.publish_event(event)

    async def publish_status_update(
        self,
        agent_id: str,
        agent_type: str,
        status: str,
        current_task: Optional[str] = None,
        queue_size: int = 0,
        performance_metrics: Dict[str, float] = None,
    ) -> bool:
        """Publish agent status update"""
        event = AgentEvent(
            event_type=AgentEventType.STATUS_UPDATE.value,
            agent_id=agent_id,
            timestamp=datetime.now().isoformat(),
            payload={
                "agent_type": agent_type,
                "status": status,
                "current_task": current_task,
                "queue_size": queue_size,
                "last_decision_time": (
                    datetime.now().isoformat() if status == "active" else None
                ),
                "performance_metrics": performance_metrics or {},
            },
        )
        return await self.publish_event(event)

    async def request_consensus(
        self, requester_agent_id: str, symbol: str, decision_context: Dict[str, Any]
    ) -> str:
        """Request consensus from all active agents"""
        correlation_id = f"consensus_{symbol}_{datetime.now().timestamp()}"

        event = AgentEvent(
            event_type=AgentEventType.CONSENSUS_REQUEST.value,
            agent_id=requester_agent_id,
            timestamp=datetime.now().isoformat(),
            payload={
                "symbol": symbol,
                "decision_context": decision_context,
                "consensus_id": correlation_id,
            },
            correlation_id=correlation_id,
            priority=2,
        )

        await self.publish_event(event)
        return correlation_id

    def subscribe_to_event(
        self, event_type: str, handler: Callable[[AgentEvent], None]
    ):
        """Subscribe to specific agent event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Subscribed to agent event type: {event_type}")

    def unsubscribe_from_event(
        self, event_type: str, handler: Callable[[AgentEvent], None]
    ):
        """Unsubscribe from agent event type"""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
                if not self.event_handlers[event_type]:
                    del self.event_handlers[event_type]
                logger.info(f"Unsubscribed from agent event type: {event_type}")
            except ValueError:
                logger.warning(f"Handler not found for event type: {event_type}")

    async def get_agent_decision_history(
        self, agent_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent decision history for an agent"""
        if agent_id in self.decision_cache:
            decisions = self.decision_cache[agent_id][-limit:]
            return [asdict(decision) for decision in decisions]
        return []

    async def get_active_agents(self) -> List[str]:
        """Get list of currently active agent IDs"""
        return [
            agent_id
            for agent_id, status in self.active_agents.items()
            if status.status == "active"
        ]

    async def health_check(self) -> Dict[str, Any]:
        """Check health of the agent event bus"""
        rabbitmq_health = await self.rabbitmq_client.health_check()

        return {
            "status": (
                "healthy" if rabbitmq_health.get("status") == "healthy" else "degraded"
            ),
            "rabbitmq": rabbitmq_health,
            "active_agents": len(self.active_agents),
            "event_handlers": len(self.event_handlers),
            "timestamp": datetime.now().isoformat(),
        }

    async def shutdown(self):
        """Shutdown the agent event bus"""
        try:
            await self.rabbitmq_client.disconnect()
            logger.info("Agent Event Bus shutdown completed")
        except Exception as e:
            logger.error("Error during Agent Event Bus shutdown", error=str(e))


# Global event bus instance
agent_event_bus = AgentEventBus()
