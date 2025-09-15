"""
Agent Coordination Hub - Real-time Multi-Agent Coordination System
================================================================

Central coordination system for agents to communicate, share insights, and make collective decisions.
Integrates with streaming data, trigger system, and AITradingCoordinator for synchronized trading.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

from .trading_agents import AITradingCoordinator
from ..core.config import settings
from ..events.trigger_engine import get_trigger_engine
from ..trading.alpaca_stream_manager import get_stream_manager

logger = structlog.get_logger(__name__)


class MessageType(Enum):
    """Types of coordination messages between agents"""
    MARKET_EVENT = "market_event"
    TRADE_PROPOSAL = "trade_proposal"
    CONSENSUS_REQUEST = "consensus_request"
    CONSENSUS_RESPONSE = "consensus_response"
    KNOWLEDGE_SHARE = "knowledge_share"
    CONFLICT_RESOLUTION = "conflict_resolution"
    LEARNING_UPDATE = "learning_update"
    AGENT_STATUS = "agent_status"


class ConsensusStatus(Enum):
    """Status of consensus decisions"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CONFLICTED = "conflicted"


@dataclass
class CoordinationMessage:
    """Message format for agent coordination"""
    message_id: str
    message_type: MessageType
    sender_agent: str
    recipient_agents: List[str]
    symbol: str
    content: Dict[str, Any]
    timestamp: str
    correlation_id: Optional[str] = None
    priority: int = 5  # 1=highest, 10=lowest


@dataclass
class ConsensusDecision:
    """Consensus decision tracking"""
    decision_id: str
    symbol: str
    proposal: Dict[str, Any]
    votes: Dict[str, Dict[str, Any]]  # agent_name -> vote_data
    status: ConsensusStatus
    final_decision: Optional[Dict[str, Any]]
    created_at: str
    expires_at: str
    correlation_id: str


@dataclass
class AgentKnowledge:
    """Shared knowledge between agents"""
    agent_name: str
    symbol: str
    knowledge_type: str
    data: Dict[str, Any]
    confidence: float
    timestamp: str
    validity_period: int = 300  # seconds


class AgentCoordinationHub:
    """
    Central coordination system for multi-agent trading decisions

    Features:
    - Real-time message routing between agents
    - Consensus mechanism for trade decisions
    - Knowledge sharing protocol
    - Conflict resolution system
    - Learning feedback loops
    """

    def __init__(self):
        self.ai_coordinator = None
        self.trigger_engine = None
        self.stream_manager = None

        # Message routing and coordination
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.consensus_decisions: Dict[str, ConsensusDecision] = {}
        self.shared_knowledge: Dict[str, List[AgentKnowledge]] = {}

        # Coordination callbacks and subscriptions
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.consensus_callbacks: List[Callable] = []
        self.conflict_resolution_callbacks: List[Callable] = []

        # Performance tracking
        self.coordination_stats = {
            "messages_processed": 0,
            "consensus_decisions": 0,
            "conflicts_resolved": 0,
            "knowledge_shared": 0,
            "agent_coordination_score": 0.0
        }

        # Background tasks
        self.coordination_task: Optional[asyncio.Task] = None
        self.consensus_monitor_task: Optional[asyncio.Task] = None
        self.knowledge_cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False

    async def initialize(self):
        """Initialize the coordination hub with all dependencies"""
        try:
            # Initialize AI coordinator
            self.ai_coordinator = AITradingCoordinator(
                enable_streaming=True,
                enable_unsupervised=True
            )

            # Get trigger engine and stream manager
            self.trigger_engine = await get_trigger_engine()
            self.stream_manager = await get_stream_manager()

            # Register coordination callbacks
            await self._register_coordination_callbacks()

            # Initialize default agents
            await self._initialize_default_agents()

            # Start coordination tasks
            await self.start_coordination()

            logger.info("Agent Coordination Hub initialized successfully",
                       agent_count=len(self.active_agents))

        except Exception as e:
            logger.error("Failed to initialize coordination hub", error=str(e))
            raise

    async def _register_coordination_callbacks(self):
        """Register callbacks with trigger engine and stream manager"""
        # Register decision callbacks with AI coordinator
        self.ai_coordinator.add_decision_callback(self._handle_agent_decision)
        self.ai_coordinator.add_coordination_callback(self._handle_coordination_event)

        # Register market data callback with trigger engine
        await self.trigger_engine.register_event_handler(
            "coordination_trigger",
            self._handle_market_trigger_event
        )

    async def _initialize_default_agents(self):
        """Initialize default trading agents"""
        default_agents = {
            "analyst": {
                "model": "llama3.2:3b",
                "specialization": "market_analysis",
                "decision_weight": 0.25,
                "status": "active"
            },
            "risk": {
                "model": "phi3:mini",
                "specialization": "risk_assessment",
                "decision_weight": 0.30,  # Risk has higher weight
                "status": "active"
            },
            "strategist": {
                "model": "qwen2.5-coder:3b",
                "specialization": "strategy_optimization",
                "decision_weight": 0.25,
                "status": "active"
            },
            "chat": {
                "model": "gemma2:2b",
                "specialization": "coordination",
                "decision_weight": 0.10,
                "status": "active"
            },
            "reasoning": {
                "model": "deepseek-r1:1.5b",
                "specialization": "pattern_analysis",
                "decision_weight": 0.10,
                "status": "active"
            }
        }

        for agent_name, config in default_agents.items():
            await self.register_agent(agent_name, config)

    async def start_coordination(self):
        """Start background coordination tasks"""
        if self.is_running:
            return

        self.is_running = True

        # Start coordination tasks
        self.coordination_task = asyncio.create_task(self._coordination_loop())
        self.consensus_monitor_task = asyncio.create_task(self._consensus_monitor_loop())
        self.knowledge_cleanup_task = asyncio.create_task(self._knowledge_cleanup_loop())

        logger.info("Agent coordination started")

    async def stop_coordination(self):
        """Stop background coordination tasks"""
        self.is_running = False

        # Cancel tasks
        for task in [self.coordination_task, self.consensus_monitor_task, self.knowledge_cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("Agent coordination stopped")

    async def register_agent(self, agent_name: str, config: Dict[str, Any]) -> bool:
        """Register a new agent with the coordination hub"""
        try:
            self.active_agents[agent_name] = {
                **config,
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "last_heartbeat": datetime.now(timezone.utc).isoformat(),
                "decisions_made": 0,
                "consensus_participation": 0,
                "knowledge_contributions": 0
            }

            logger.info("Agent registered", agent=agent_name, config=config)
            return True

        except Exception as e:
            logger.error("Failed to register agent", agent=agent_name, error=str(e))
            return False

    async def unregister_agent(self, agent_name: str):
        """Unregister an agent from coordination"""
        if agent_name in self.active_agents:
            del self.active_agents[agent_name]
            logger.info("Agent unregistered", agent=agent_name)

    async def send_message(self, message: CoordinationMessage):
        """Send a coordination message to specified agents"""
        try:
            # Add to message queue for processing
            await self.message_queue.put(message)
            self.coordination_stats["messages_processed"] += 1

            logger.debug("Message queued",
                        message_id=message.message_id,
                        type=message.message_type.value,
                        sender=message.sender_agent)

        except Exception as e:
            logger.error("Failed to send message", error=str(e))

    async def broadcast_market_event(self, agent_names: List[str], event_data: Dict[str, Any]):
        """Broadcast market event to specified agents for coordinated response"""
        try:
            message = CoordinationMessage(
                message_id=f"market_{int(time.time())}_{event_data.get('symbol', 'unknown')}",
                message_type=MessageType.MARKET_EVENT,
                sender_agent="coordination_hub",
                recipient_agents=agent_names,
                symbol=event_data.get('symbol', 'UNKNOWN'),
                content=event_data,
                timestamp=datetime.now(timezone.utc).isoformat(),
                priority=1  # High priority for market events
            )

            await self.send_message(message)

            # Trigger coordinated analysis
            await self._trigger_coordinated_analysis(event_data)

        except Exception as e:
            logger.error("Failed to broadcast market event", error=str(e))

    async def broadcast_scheduled_event(self, agent_names: List[str], event_data: Dict[str, Any]):
        """Broadcast scheduled event to specified agents"""
        try:
            message = CoordinationMessage(
                message_id=f"scheduled_{int(time.time())}_{event_data.get('event_type', 'unknown')}",
                message_type=MessageType.MARKET_EVENT,
                sender_agent="schedule_trigger",
                recipient_agents=agent_names,
                symbol="SCHEDULE",
                content=event_data,
                timestamp=datetime.now(timezone.utc).isoformat(),
                priority=2  # Medium priority for scheduled events
            )

            await self.send_message(message)

        except Exception as e:
            logger.error("Failed to broadcast scheduled event", error=str(e))

    async def request_consensus(self, symbol: str, proposal: Dict[str, Any],
                              requesting_agent: str, timeout: int = 30) -> ConsensusDecision:
        """Request consensus decision from all active agents"""
        try:
            decision_id = f"consensus_{symbol}_{int(time.time())}"
            correlation_id = f"corr_{decision_id}"

            # Create consensus decision
            consensus_decision = ConsensusDecision(
                decision_id=decision_id,
                symbol=symbol,
                proposal=proposal,
                votes={},
                status=ConsensusStatus.PENDING,
                final_decision=None,
                created_at=datetime.now(timezone.utc).isoformat(),
                expires_at=datetime.fromtimestamp(time.time() + timeout).isoformat(),
                correlation_id=correlation_id
            )

            self.consensus_decisions[decision_id] = consensus_decision

            # Send consensus request to all agents
            message = CoordinationMessage(
                message_id=f"consensus_request_{decision_id}",
                message_type=MessageType.CONSENSUS_REQUEST,
                sender_agent=requesting_agent,
                recipient_agents=list(self.active_agents.keys()),
                symbol=symbol,
                content={
                    "decision_id": decision_id,
                    "proposal": proposal,
                    "timeout": timeout,
                    "correlation_id": correlation_id
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
                correlation_id=correlation_id,
                priority=1
            )

            await self.send_message(message)

            logger.info("Consensus requested",
                       decision_id=decision_id,
                       symbol=symbol,
                       requesting_agent=requesting_agent)

            return consensus_decision

        except Exception as e:
            logger.error("Failed to request consensus", error=str(e))
            raise

    async def submit_vote(self, decision_id: str, agent_name: str, vote_data: Dict[str, Any]):
        """Submit vote for consensus decision"""
        try:
            if decision_id not in self.consensus_decisions:
                logger.warning("Vote submitted for unknown decision",
                              decision_id=decision_id, agent=agent_name)
                return

            consensus = self.consensus_decisions[decision_id]

            # Check if decision is still pending
            if consensus.status != ConsensusStatus.PENDING:
                logger.warning("Vote submitted for closed decision",
                              decision_id=decision_id, agent=agent_name, status=consensus.status.value)
                return

            # Record vote
            consensus.votes[agent_name] = {
                **vote_data,
                "voted_at": datetime.now(timezone.utc).isoformat()
            }

            # Update agent stats
            if agent_name in self.active_agents:
                self.active_agents[agent_name]["consensus_participation"] += 1

            logger.debug("Vote recorded",
                        decision_id=decision_id,
                        agent=agent_name,
                        vote_count=len(consensus.votes))

            # Check if we have all votes
            if len(consensus.votes) >= len(self.active_agents):
                await self._finalize_consensus(decision_id)

        except Exception as e:
            logger.error("Failed to submit vote", error=str(e))

    async def share_knowledge(self, agent_name: str, symbol: str,
                            knowledge_type: str, data: Dict[str, Any], confidence: float):
        """Share knowledge between agents"""
        try:
            knowledge = AgentKnowledge(
                agent_name=agent_name,
                symbol=symbol,
                knowledge_type=knowledge_type,
                data=data,
                confidence=confidence,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

            # Store knowledge
            if symbol not in self.shared_knowledge:
                self.shared_knowledge[symbol] = []
            self.shared_knowledge[symbol].append(knowledge)

            # Update agent stats
            if agent_name in self.active_agents:
                self.active_agents[agent_name]["knowledge_contributions"] += 1

            # Broadcast knowledge to other agents
            message = CoordinationMessage(
                message_id=f"knowledge_{symbol}_{int(time.time())}",
                message_type=MessageType.KNOWLEDGE_SHARE,
                sender_agent=agent_name,
                recipient_agents=[name for name in self.active_agents.keys() if name != agent_name],
                symbol=symbol,
                content=asdict(knowledge),
                timestamp=datetime.now(timezone.utc).isoformat(),
                priority=3
            )

            await self.send_message(message)
            self.coordination_stats["knowledge_shared"] += 1

            logger.debug("Knowledge shared",
                        agent=agent_name,
                        symbol=symbol,
                        type=knowledge_type,
                        confidence=confidence)

        except Exception as e:
            logger.error("Failed to share knowledge", error=str(e))

    async def get_shared_knowledge(self, symbol: str, knowledge_type: Optional[str] = None) -> List[AgentKnowledge]:
        """Get shared knowledge for a symbol"""
        if symbol not in self.shared_knowledge:
            return []

        knowledge_list = self.shared_knowledge[symbol]

        if knowledge_type:
            knowledge_list = [k for k in knowledge_list if k.knowledge_type == knowledge_type]

        # Filter expired knowledge
        current_time = time.time()
        valid_knowledge = []

        for knowledge in knowledge_list:
            knowledge_time = datetime.fromisoformat(knowledge.timestamp.replace('Z', '+00:00')).timestamp()
            if current_time - knowledge_time < knowledge.validity_period:
                valid_knowledge.append(knowledge)

        return valid_knowledge

    # Background coordination loops
    async def _coordination_loop(self):
        """Main coordination message processing loop"""
        logger.info("Coordination message processing started")

        while self.is_running:
            try:
                # Process message queue
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                    await self._process_message(message)
                except asyncio.TimeoutError:
                    continue

                # Update coordination score
                await self._update_coordination_score()

            except Exception as e:
                logger.error("Error in coordination loop", error=str(e))
                await asyncio.sleep(1)

    async def _consensus_monitor_loop(self):
        """Monitor consensus decisions and handle timeouts"""
        logger.info("Consensus monitoring started")

        while self.is_running:
            try:
                current_time = time.time()
                expired_decisions = []

                for decision_id, consensus in self.consensus_decisions.items():
                    if consensus.status == ConsensusStatus.PENDING:
                        expires_at = datetime.fromisoformat(consensus.expires_at).timestamp()

                        if current_time > expires_at:
                            expired_decisions.append(decision_id)

                # Handle expired decisions
                for decision_id in expired_decisions:
                    await self._handle_consensus_timeout(decision_id)

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error("Error in consensus monitoring", error=str(e))
                await asyncio.sleep(5)

    async def _knowledge_cleanup_loop(self):
        """Clean up expired knowledge"""
        logger.info("Knowledge cleanup started")

        while self.is_running:
            try:
                current_time = time.time()

                for symbol, knowledge_list in self.shared_knowledge.items():
                    valid_knowledge = []

                    for knowledge in knowledge_list:
                        knowledge_time = datetime.fromisoformat(knowledge.timestamp.replace('Z', '+00:00')).timestamp()
                        if current_time - knowledge_time < knowledge.validity_period:
                            valid_knowledge.append(knowledge)

                    self.shared_knowledge[symbol] = valid_knowledge

                await asyncio.sleep(60)  # Cleanup every minute

            except Exception as e:
                logger.error("Error in knowledge cleanup", error=str(e))
                await asyncio.sleep(60)

    # Message processing handlers
    async def _process_message(self, message: CoordinationMessage):
        """Process coordination message"""
        try:
            # Call registered handlers
            handlers = self.message_handlers.get(message.message_type, [])
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error("Message handler failed",
                                handler=handler.__name__,
                                error=str(e))

            # Default processing based on message type
            if message.message_type == MessageType.CONSENSUS_REQUEST:
                await self._handle_consensus_request(message)
            elif message.message_type == MessageType.CONSENSUS_RESPONSE:
                await self._handle_consensus_response(message)
            elif message.message_type == MessageType.MARKET_EVENT:
                await self._handle_market_event_message(message)

        except Exception as e:
            logger.error("Failed to process message", error=str(e))

    async def _handle_agent_decision(self, decision_data: Dict[str, Any]):
        """Handle agent decision from AI coordinator"""
        try:
            agent_type = decision_data.get("agent_type", "unknown")
            symbol = decision_data.get("symbol", "UNKNOWN")

            # Update agent stats
            if agent_type in self.active_agents:
                self.active_agents[agent_type]["decisions_made"] += 1
                self.active_agents[agent_type]["last_heartbeat"] = datetime.now(timezone.utc).isoformat()

            # Share decision as knowledge
            await self.share_knowledge(
                agent_name=agent_type,
                symbol=symbol,
                knowledge_type="decision",
                data=decision_data,
                confidence=decision_data.get("confidence", 0.5)
            )

        except Exception as e:
            logger.error("Failed to handle agent decision", error=str(e))

    async def _handle_coordination_event(self, coordination_data: Dict[str, Any]):
        """Handle coordination event from AI coordinator"""
        try:
            # Process final coordinated decisions
            if coordination_data.get("message_type") == "final_decision":
                await self._process_final_decision(coordination_data)

        except Exception as e:
            logger.error("Failed to handle coordination event", error=str(e))

    async def _handle_market_trigger_event(self, trigger_event):
        """Handle market trigger event"""
        try:
            symbol = trigger_event.trigger_condition.symbol
            event_data = {
                "symbol": symbol,
                "trigger_type": trigger_event.trigger_condition.trigger_type.value,
                "current_value": trigger_event.current_value,
                "threshold": trigger_event.trigger_condition.threshold,
                "market_data": trigger_event.market_data,
                "priority": trigger_event.trigger_condition.priority.value
            }

            # Broadcast to target agents
            await self.broadcast_market_event(
                trigger_event.trigger_condition.target_agents,
                event_data
            )

        except Exception as e:
            logger.error("Failed to handle market trigger event", error=str(e))

    async def _trigger_coordinated_analysis(self, event_data: Dict[str, Any]):
        """Trigger coordinated analysis using AI coordinator"""
        try:
            symbol = event_data.get('symbol', 'UNKNOWN')

            if symbol == 'UNKNOWN' or not self.ai_coordinator:
                return

            # Create mock data for comprehensive analysis
            # In production, this would come from real market data
            market_data = event_data.get('market_data', {})
            if not market_data:
                market_data = {
                    "price": event_data.get('current_value', 100.0),
                    "volume": 1000000,
                    "price_change_pct": 0.0,
                    "volume_ratio": 1.0
                }

            technical_indicators = {
                "rsi": 50.0,
                "macd": 0.0,
                "bb_position": 0.5,
                "atr": 1.0
            }

            account_info = {
                "equity": 100000,
                "buying_power": 50000
            }

            markov_analysis = {
                "current_state": "trending",
                "transition_probability": 0.7
            }

            # Run comprehensive analysis
            result = await self.ai_coordinator.comprehensive_analysis(
                symbol=symbol,
                market_data=market_data,
                technical_indicators=technical_indicators,
                account_info=account_info,
                current_positions=[],
                markov_analysis=markov_analysis
            )

            logger.info("Coordinated analysis completed",
                       symbol=symbol,
                       recommendation=result.get("final_recommendation"))

        except Exception as e:
            logger.error("Failed to trigger coordinated analysis", error=str(e))

    async def _finalize_consensus(self, decision_id: str):
        """Finalize consensus decision"""
        try:
            consensus = self.consensus_decisions[decision_id]

            # Calculate consensus
            total_weight = 0
            weighted_approval = 0

            for agent_name, vote in consensus.votes.items():
                if agent_name in self.active_agents:
                    weight = self.active_agents[agent_name].get("decision_weight", 0.2)
                    total_weight += weight

                    if vote.get("approval", False):
                        weighted_approval += weight

            # Determine consensus
            if total_weight > 0:
                approval_ratio = weighted_approval / total_weight

                if approval_ratio >= 0.6:  # 60% weighted approval
                    consensus.status = ConsensusStatus.APPROVED
                    consensus.final_decision = {
                        "approved": True,
                        "approval_ratio": approval_ratio,
                        "implementing_agents": [name for name, vote in consensus.votes.items()
                                              if vote.get("approval", False)]
                    }
                else:
                    consensus.status = ConsensusStatus.REJECTED
                    consensus.final_decision = {
                        "approved": False,
                        "approval_ratio": approval_ratio,
                        "rejecting_agents": [name for name, vote in consensus.votes.items()
                                           if not vote.get("approval", True)]
                    }
            else:
                consensus.status = ConsensusStatus.CONFLICTED
                consensus.final_decision = {"approved": False, "reason": "No valid votes"}

            self.coordination_stats["consensus_decisions"] += 1

            logger.info("Consensus finalized",
                       decision_id=decision_id,
                       status=consensus.status.value,
                       approval_ratio=consensus.final_decision.get("approval_ratio", 0))

            # Notify callbacks
            for callback in self.consensus_callbacks:
                try:
                    await callback(consensus)
                except Exception as e:
                    logger.error("Consensus callback failed", error=str(e))

        except Exception as e:
            logger.error("Failed to finalize consensus", error=str(e))

    async def _handle_consensus_timeout(self, decision_id: str):
        """Handle consensus decision timeout"""
        try:
            consensus = self.consensus_decisions[decision_id]
            consensus.status = ConsensusStatus.CONFLICTED
            consensus.final_decision = {
                "approved": False,
                "reason": "timeout",
                "votes_received": len(consensus.votes),
                "agents_available": len(self.active_agents)
            }

            logger.warning("Consensus decision timed out",
                          decision_id=decision_id,
                          votes_received=len(consensus.votes))

        except Exception as e:
            logger.error("Failed to handle consensus timeout", error=str(e))

    async def _update_coordination_score(self):
        """Update overall coordination effectiveness score"""
        try:
            if not self.active_agents:
                self.coordination_stats["agent_coordination_score"] = 0.0
                return

            # Calculate coordination metrics
            total_decisions = sum(agent.get("decisions_made", 0) for agent in self.active_agents.values())
            total_participation = sum(agent.get("consensus_participation", 0) for agent in self.active_agents.values())
            total_knowledge = sum(agent.get("knowledge_contributions", 0) for agent in self.active_agents.values())

            # Normalize scores
            avg_decisions = total_decisions / len(self.active_agents) if self.active_agents else 0
            avg_participation = total_participation / len(self.active_agents) if self.active_agents else 0
            avg_knowledge = total_knowledge / len(self.active_agents) if self.active_agents else 0

            # Calculate coordination score (0-1)
            coordination_score = min(1.0, (avg_decisions * 0.4 + avg_participation * 0.4 + avg_knowledge * 0.2) / 10)

            self.coordination_stats["agent_coordination_score"] = coordination_score

        except Exception as e:
            logger.error("Failed to update coordination score", error=str(e))

    # Message handler registration
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register message handler for specific message type"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)

    def register_consensus_callback(self, callback: Callable):
        """Register callback for consensus decisions"""
        self.consensus_callbacks.append(callback)

    def register_conflict_resolution_callback(self, callback: Callable):
        """Register callback for conflict resolution"""
        self.conflict_resolution_callbacks.append(callback)

    # Status and monitoring
    async def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination hub status"""
        return {
            "active_agents": len(self.active_agents),
            "agents": self.active_agents,
            "pending_consensus": len([c for c in self.consensus_decisions.values()
                                    if c.status == ConsensusStatus.PENDING]),
            "shared_knowledge_symbols": len(self.shared_knowledge),
            "coordination_stats": self.coordination_stats,
            "message_queue_size": self.message_queue.qsize(),
            "is_running": self.is_running,
            "ai_coordinator_available": self.ai_coordinator is not None,
            "trigger_engine_available": self.trigger_engine is not None,
            "stream_manager_available": self.stream_manager is not None
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for coordination hub"""
        try:
            # Check AI coordinator health
            ai_health = {}
            if self.ai_coordinator:
                ai_health = await self.ai_coordinator.health_check()

            return {
                "coordination_hub": "healthy" if self.is_running else "stopped",
                "agents_active": len(self.active_agents),
                "coordination_score": self.coordination_stats["agent_coordination_score"],
                "ai_coordinator": ai_health,
                "system_integration": {
                    "trigger_engine": self.trigger_engine is not None,
                    "stream_manager": self.stream_manager is not None,
                    "ai_coordinator": self.ai_coordinator is not None
                }
            }

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {"coordination_hub": "error", "error": str(e)}


# Global coordination hub instance
coordination_hub: Optional[AgentCoordinationHub] = None


async def get_coordination_hub() -> AgentCoordinationHub:
    """Get or create global coordination hub instance"""
    global coordination_hub

    if coordination_hub is None:
        coordination_hub = AgentCoordinationHub()
        await coordination_hub.initialize()

    return coordination_hub