"""
Multi-Agent Coordination System - Advanced consensus mechanisms for collaborative decision making
Implements voting, consensus building, and conflict resolution for multiple AI agents
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, Counter
import statistics
import uuid

import structlog

from .agent_event_bus import AgentEventBus, AgentEvent, AgentEventType, agent_event_bus
from app.websockets.agent_coordination_socket import AgentCoordinationMessage, agent_coordination_manager

logger = structlog.get_logger(__name__)


class ConsensusMethod(Enum):
    """Consensus building methods"""
    SIMPLE_MAJORITY = "simple_majority"
    WEIGHTED_CONFIDENCE = "weighted_confidence"
    UNANIMOUS = "unanimous"
    EXPERT_OVERRIDE = "expert_override"
    RISK_WEIGHTED = "risk_weighted"


class ConflictResolution(Enum):
    """Conflict resolution strategies"""
    HIGHEST_CONFIDENCE = "highest_confidence"
    CONSERVATIVE_BIAS = "conservative_bias"
    RISK_MANAGER_OVERRIDE = "risk_manager_override"
    WEIGHTED_AVERAGE = "weighted_average"
    ESCALATE_TO_HUMAN = "escalate_to_human"


@dataclass
class AgentVote:
    """Individual agent vote in consensus process"""
    agent_id: str
    agent_type: str
    decision: str
    confidence: float
    reasoning: str
    weight: float = 1.0
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ConsensusRequest:
    """Request for multi-agent consensus"""
    consensus_id: str
    symbol: str
    context: Dict[str, Any]
    required_agents: List[str]
    consensus_method: ConsensusMethod
    conflict_resolution: ConflictResolution
    timeout_seconds: int
    min_confidence: float
    requester_id: str
    created_at: datetime
    expires_at: datetime


@dataclass
class ConsensusResult:
    """Result of multi-agent consensus process"""
    consensus_id: str
    symbol: str
    final_decision: str
    confidence: float
    consensus_achieved: bool
    participating_agents: List[str]
    votes: List[AgentVote]
    reasoning: str
    conflict_resolution_applied: Optional[str]
    processing_time_ms: float
    timestamp: datetime
    metadata: Dict[str, Any]


class MultiAgentCoordinator:
    """Coordinates multiple AI agents for collaborative decision making"""
    
    def __init__(self, event_bus: Optional[AgentEventBus] = None):
        self.event_bus = event_bus or agent_event_bus
        
        # Consensus tracking
        self.active_consensus: Dict[str, ConsensusRequest] = {}
        self.consensus_votes: Dict[str, List[AgentVote]] = {}
        self.consensus_results: Dict[str, ConsensusResult] = {}
        
        # Agent capabilities and weights
        self.agent_weights: Dict[str, Dict[str, float]] = {
            "market_analyst": {
                "market_analysis": 1.5,
                "trend_identification": 1.3,
                "technical_analysis": 1.4,
                "default": 1.0
            },
            "risk_advisor": {
                "risk_assessment": 2.0,
                "portfolio_management": 1.8,
                "volatility_analysis": 1.5,
                "default": 1.0
            },
            "strategy_optimizer": {
                "signal_generation": 1.5,
                "backtesting": 1.3,
                "optimization": 1.4,
                "default": 1.0
            },
            "performance_coach": {
                "performance_analysis": 1.2,
                "improvement_recommendations": 1.3,
                "default": 1.0
            }
        }
        
        # Consensus callbacks
        self.consensus_callbacks: List[Callable[[ConsensusResult], None]] = []
        self.conflict_callbacks: List[Callable[[str, List[AgentVote]], None]] = []
        
        # Performance tracking
        self.consensus_history: List[ConsensusResult] = []
        self.agent_performance: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_votes": 0,
            "consensus_contributions": 0,
            "average_confidence": 0.0,
            "decision_accuracy": 0.0
        })
    
    def add_consensus_callback(self, callback: Callable[[ConsensusResult], None]):
        """Add callback for consensus completion events"""
        self.consensus_callbacks.append(callback)
    
    def add_conflict_callback(self, callback: Callable[[str, List[AgentVote]], None]):
        """Add callback for conflict detection events"""
        self.conflict_callbacks.append(callback)
    
    async def initialize(self):
        """Initialize the multi-agent coordinator"""
        # Subscribe to consensus-related events
        self.event_bus.subscribe_to_event(
            AgentEventType.CONSENSUS_RESPONSE.value,
            self._handle_consensus_response
        )
        
        self.event_bus.subscribe_to_event(
            AgentEventType.COORDINATION.value,
            self._handle_coordination_message
        )
        
        logger.info("Multi-agent coordinator initialized")
    
    async def request_consensus(
        self,
        symbol: str,
        context: Dict[str, Any],
        required_agents: List[str],
        consensus_method: ConsensusMethod = ConsensusMethod.WEIGHTED_CONFIDENCE,
        conflict_resolution: ConflictResolution = ConflictResolution.HIGHEST_CONFIDENCE,
        timeout_seconds: int = 30,
        min_confidence: float = 0.6,
        requester_id: str = "coordinator"
    ) -> str:
        """Request consensus from multiple agents"""
        
        consensus_id = str(uuid.uuid4())
        created_at = datetime.now()
        expires_at = created_at + timedelta(seconds=timeout_seconds)
        
        # Create consensus request
        request = ConsensusRequest(
            consensus_id=consensus_id,
            symbol=symbol,
            context=context,
            required_agents=required_agents,
            consensus_method=consensus_method,
            conflict_resolution=conflict_resolution,
            timeout_seconds=timeout_seconds,
            min_confidence=min_confidence,
            requester_id=requester_id,
            created_at=created_at,
            expires_at=expires_at
        )
        
        # Store request
        self.active_consensus[consensus_id] = request
        self.consensus_votes[consensus_id] = []
        
        # Send consensus request to all required agents
        await self._broadcast_consensus_request(request)
        
        # Start timeout monitoring
        asyncio.create_task(self._monitor_consensus_timeout(consensus_id))
        
        logger.info("Consensus requested",
                   consensus_id=consensus_id,
                   symbol=symbol,
                   required_agents=required_agents,
                   method=consensus_method.value)
        
        return consensus_id
    
    async def _broadcast_consensus_request(self, request: ConsensusRequest):
        """Broadcast consensus request to required agents"""
        
        for agent_id in request.required_agents:
            # Send via event bus
            await self.event_bus.publish_event(AgentEvent(
                event_type=AgentEventType.CONSENSUS_REQUEST.value,
                agent_id=agent_id,
                timestamp=datetime.now().isoformat(),
                payload={
                    "consensus_id": request.consensus_id,
                    "symbol": request.symbol,
                    "context": request.context,
                    "method": request.consensus_method.value,
                    "timeout_seconds": request.timeout_seconds,
                    "min_confidence": request.min_confidence
                },
                correlation_id=request.consensus_id,
                priority=2
            ))
        
        # Also broadcast via WebSocket coordination
        coordination_message = AgentCoordinationMessage(
            sender_agent_id="multi_agent_coordinator",
            recipient_agent_id=None,  # Broadcast to all
            message_type="consensus_request",
            payload={
                "consensus_id": request.consensus_id,
                "symbol": request.symbol,
                "required_agents": request.required_agents,
                "context": request.context
            },
            timestamp=datetime.now().isoformat(),
            requires_response=True
        )
        
        await agent_coordination_manager.broadcast_coordination_message(coordination_message)
    
    async def _handle_consensus_response(self, event: AgentEvent):
        """Handle agent responses to consensus requests"""
        
        consensus_id = event.payload.get("consensus_id")
        if not consensus_id or consensus_id not in self.active_consensus:
            logger.warning("Received consensus response for unknown request", 
                          consensus_id=consensus_id)
            return
        
        # Create agent vote
        vote = AgentVote(
            agent_id=event.agent_id,
            agent_type=event.payload.get("agent_type", "unknown"),
            decision=event.payload.get("decision", "HOLD"),
            confidence=event.payload.get("confidence", 0.0),
            reasoning=event.payload.get("reasoning", ""),
            weight=self._get_agent_weight(event.payload.get("agent_type"), "default"),
            timestamp=datetime.now(),
            metadata=event.payload.get("metadata", {})
        )
        
        # Add vote to consensus
        self.consensus_votes[consensus_id].append(vote)
        
        logger.info("Consensus vote received",
                   consensus_id=consensus_id,
                   agent_id=event.agent_id,
                   decision=vote.decision,
                   confidence=vote.confidence)
        
        # Check if we have enough votes
        request = self.active_consensus[consensus_id]
        if len(self.consensus_votes[consensus_id]) >= len(request.required_agents):
            # All agents have voted
            await self._process_consensus(consensus_id)
    
    async def _handle_coordination_message(self, event: AgentEvent):
        """Handle general coordination messages between agents"""
        
        message_type = event.payload.get("message_type")
        
        if message_type == "disagreement":
            # Handle agent disagreement
            await self._handle_agent_disagreement(event)
        elif message_type == "collaboration_request":
            # Handle collaboration request
            await self._handle_collaboration_request(event)
        elif message_type == "performance_feedback":
            # Handle performance feedback
            await self._handle_performance_feedback(event)
    
    async def _handle_agent_disagreement(self, event: AgentEvent):
        """Handle disagreement between agents"""
        
        disagreement_data = event.payload.get("disagreement_data", {})
        conflicting_agents = disagreement_data.get("conflicting_agents", [])
        
        logger.info("Agent disagreement detected",
                   reporting_agent=event.agent_id,
                   conflicting_agents=conflicting_agents)
        
        # Trigger conflict callbacks
        for callback in self.conflict_callbacks:
            try:
                await callback(event.agent_id, conflicting_agents)
            except Exception as e:
                logger.warning("Conflict callback failed", error=str(e))
    
    async def _monitor_consensus_timeout(self, consensus_id: str):
        """Monitor consensus request timeout"""
        
        if consensus_id not in self.active_consensus:
            return
        
        request = self.active_consensus[consensus_id]
        
        # Wait for timeout
        await asyncio.sleep(request.timeout_seconds)
        
        # Check if consensus still active
        if consensus_id in self.active_consensus:
            logger.warning("Consensus timeout",
                          consensus_id=consensus_id,
                          received_votes=len(self.consensus_votes.get(consensus_id, [])),
                          required_agents=len(request.required_agents))
            
            # Process with available votes
            await self._process_consensus(consensus_id, timeout=True)
    
    async def _process_consensus(self, consensus_id: str, timeout: bool = False):
        """Process consensus votes and determine final decision"""
        
        if consensus_id not in self.active_consensus:
            return
        
        start_time = datetime.now()
        request = self.active_consensus[consensus_id]
        votes = self.consensus_votes[consensus_id]
        
        if not votes:
            # No votes received
            result = ConsensusResult(
                consensus_id=consensus_id,
                symbol=request.symbol,
                final_decision="HOLD",
                confidence=0.0,
                consensus_achieved=False,
                participating_agents=[],
                votes=[],
                reasoning="No agent votes received",
                conflict_resolution_applied=None,
                processing_time_ms=0.0,
                timestamp=datetime.now(),
                metadata={"timeout": timeout}
            )
        else:
            # Process votes according to consensus method
            result = await self._calculate_consensus(request, votes, timeout)
        
        # Store result
        self.consensus_results[consensus_id] = result
        self.consensus_history.append(result)
        
        # Clean up active consensus
        self.active_consensus.pop(consensus_id, None)
        self.consensus_votes.pop(consensus_id, None)
        
        # Update agent performance metrics
        self._update_agent_performance(votes, result)
        
        # Trigger callbacks
        for callback in self.consensus_callbacks:
            try:
                await callback(result)
            except Exception as e:
                logger.warning("Consensus callback failed", error=str(e))
        
        # Broadcast result
        await self._broadcast_consensus_result(result)
        
        logger.info("Consensus processed",
                   consensus_id=consensus_id,
                   final_decision=result.final_decision,
                   confidence=result.confidence,
                   consensus_achieved=result.consensus_achieved)
    
    async def _calculate_consensus(self, request: ConsensusRequest, votes: List[AgentVote], timeout: bool) -> ConsensusResult:
        """Calculate consensus based on specified method"""
        
        start_time = datetime.now()
        participating_agents = [vote.agent_id for vote in votes]
        
        # Check for conflicts
        decisions = [vote.decision for vote in votes]
        decision_counts = Counter(decisions)
        has_conflict = len(decision_counts) > 1
        
        final_decision = "HOLD"
        confidence = 0.0
        reasoning = ""
        conflict_resolution_applied = None
        consensus_achieved = True
        
        try:
            if request.consensus_method == ConsensusMethod.SIMPLE_MAJORITY:
                # Simple majority vote
                most_common_decision = decision_counts.most_common(1)[0]
                final_decision = most_common_decision[0]
                
                # Confidence is average of votes for winning decision
                winning_votes = [vote for vote in votes if vote.decision == final_decision]
                confidence = statistics.mean([vote.confidence for vote in winning_votes])
                reasoning = f"Majority decision: {most_common_decision[1]}/{len(votes)} agents voted {final_decision}"
                
                consensus_achieved = most_common_decision[1] > len(votes) / 2
                
            elif request.consensus_method == ConsensusMethod.WEIGHTED_CONFIDENCE:
                # Weight votes by confidence and agent weight
                decision_scores = defaultdict(float)
                decision_weights = defaultdict(float)
                
                for vote in votes:
                    agent_weight = self._get_agent_weight(vote.agent_type, "default")
                    weighted_score = vote.confidence * agent_weight
                    decision_scores[vote.decision] += weighted_score
                    decision_weights[vote.decision] += agent_weight
                
                # Find highest weighted decision
                if decision_scores:
                    final_decision = max(decision_scores.items(), key=lambda x: x[1])[0]
                    confidence = decision_scores[final_decision] / decision_weights[final_decision]
                    reasoning = f"Weighted confidence decision: {final_decision} (score: {decision_scores[final_decision]:.2f})"
                    
                    # Consensus achieved if winning decision has >50% of total weight
                    total_weight = sum(decision_weights.values())
                    consensus_achieved = decision_weights[final_decision] > total_weight * 0.5
                
            elif request.consensus_method == ConsensusMethod.UNANIMOUS:
                # Require unanimous decision
                if len(decision_counts) == 1:
                    final_decision = list(decision_counts.keys())[0]
                    confidence = statistics.mean([vote.confidence for vote in votes])
                    reasoning = f"Unanimous decision: all {len(votes)} agents agreed on {final_decision}"
                else:
                    # No consensus, apply conflict resolution
                    final_decision, confidence, conflict_resolution_applied = await self._resolve_conflict(
                        request.conflict_resolution, votes
                    )
                    reasoning = f"No unanimous consensus, applied {conflict_resolution_applied}: {final_decision}"
                    consensus_achieved = False
                    
            elif request.consensus_method == ConsensusMethod.RISK_WEIGHTED:
                # Weight risk advisor votes higher for risk-related decisions
                decision_scores = defaultdict(float)
                
                for vote in votes:
                    weight = 2.0 if vote.agent_type == "risk_advisor" else 1.0
                    decision_scores[vote.decision] += vote.confidence * weight
                
                if decision_scores:
                    final_decision = max(decision_scores.items(), key=lambda x: x[1])[0]
                    
                    # Calculate weighted confidence
                    weighted_votes = [vote for vote in votes if vote.decision == final_decision]
                    total_weight = sum(2.0 if vote.agent_type == "risk_advisor" else 1.0 for vote in weighted_votes)
                    weighted_confidence = sum(
                        (2.0 if vote.agent_type == "risk_advisor" else 1.0) * vote.confidence 
                        for vote in weighted_votes
                    ) / total_weight
                    
                    confidence = weighted_confidence
                    reasoning = f"Risk-weighted decision: {final_decision} (risk advisor emphasis)"
                    consensus_achieved = True
            
            # Apply conflict resolution if needed
            if has_conflict and request.consensus_method not in [ConsensusMethod.UNANIMOUS]:
                # Check if confidence meets minimum threshold
                if confidence < request.min_confidence:
                    final_decision, confidence, conflict_resolution_applied = await self._resolve_conflict(
                        request.conflict_resolution, votes
                    )
                    consensus_achieved = False
        
        except Exception as e:
            logger.error("Error calculating consensus", error=str(e))
            final_decision = "HOLD"
            confidence = 0.0
            reasoning = f"Consensus calculation failed: {str(e)}"
            consensus_achieved = False
        
        processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return ConsensusResult(
            consensus_id=request.consensus_id,
            symbol=request.symbol,
            final_decision=final_decision,
            confidence=confidence,
            consensus_achieved=consensus_achieved,
            participating_agents=participating_agents,
            votes=votes,
            reasoning=reasoning,
            conflict_resolution_applied=conflict_resolution_applied,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now(),
            metadata={
                "timeout": timeout,
                "method": request.consensus_method.value,
                "has_conflict": has_conflict,
                "decision_distribution": dict(decision_counts)
            }
        )
    
    async def _resolve_conflict(self, resolution_method: ConflictResolution, votes: List[AgentVote]) -> Tuple[str, float, str]:
        """Resolve conflicts between agent votes"""
        
        if resolution_method == ConflictResolution.HIGHEST_CONFIDENCE:
            # Choose vote with highest confidence
            highest_confidence_vote = max(votes, key=lambda v: v.confidence)
            return highest_confidence_vote.decision, highest_confidence_vote.confidence, "highest_confidence"
            
        elif resolution_method == ConflictResolution.CONSERVATIVE_BIAS:
            # Prefer HOLD, then SELL, then BUY
            decision_priority = {"HOLD": 3, "SELL": 2, "BUY": 1}
            conservative_votes = sorted(votes, key=lambda v: decision_priority.get(v.decision, 0), reverse=True)
            best_vote = conservative_votes[0]
            return best_vote.decision, best_vote.confidence * 0.8, "conservative_bias"  # Reduce confidence due to conflict
            
        elif resolution_method == ConflictResolution.RISK_MANAGER_OVERRIDE:
            # Risk advisor vote takes precedence
            risk_votes = [vote for vote in votes if vote.agent_type == "risk_advisor"]
            if risk_votes:
                risk_vote = max(risk_votes, key=lambda v: v.confidence)
                return risk_vote.decision, risk_vote.confidence, "risk_manager_override"
            else:
                # Fallback to highest confidence
                highest_confidence_vote = max(votes, key=lambda v: v.confidence)
                return highest_confidence_vote.decision, highest_confidence_vote.confidence, "risk_manager_override_fallback"
                
        elif resolution_method == ConflictResolution.WEIGHTED_AVERAGE:
            # Weight decisions by confidence and calculate average
            decision_values = {"BUY": 1, "HOLD": 0, "SELL": -1}
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for vote in votes:
                decision_value = decision_values.get(vote.decision, 0)
                weight = vote.confidence
                weighted_sum += decision_value * weight
                total_weight += weight
            
            if total_weight > 0:
                average_value = weighted_sum / total_weight
                
                if average_value > 0.3:
                    final_decision = "BUY"
                elif average_value < -0.3:
                    final_decision = "SELL"
                else:
                    final_decision = "HOLD"
                    
                confidence = statistics.mean([vote.confidence for vote in votes]) * 0.9  # Reduce due to conflict
                return final_decision, confidence, "weighted_average"
        
        # Default fallback
        return "HOLD", 0.5, "default_fallback"
    
    def _get_agent_weight(self, agent_type: str, context: str = "default") -> float:
        """Get agent weight for specific context"""
        agent_weights = self.agent_weights.get(agent_type, {})
        return agent_weights.get(context, agent_weights.get("default", 1.0))
    
    def _update_agent_performance(self, votes: List[AgentVote], result: ConsensusResult):
        """Update agent performance metrics"""
        
        for vote in votes:
            agent_id = vote.agent_id
            performance = self.agent_performance[agent_id]
            
            performance["total_votes"] += 1
            
            if result.consensus_achieved:
                performance["consensus_contributions"] += 1
            
            # Update running average of confidence
            old_confidence = performance["average_confidence"]
            total_votes = performance["total_votes"]
            new_confidence = (old_confidence * (total_votes - 1) + vote.confidence) / total_votes
            performance["average_confidence"] = new_confidence
            
            # Decision accuracy (if decision matches final consensus)
            if vote.decision == result.final_decision:
                accuracy_contributions = performance.get("accurate_decisions", 0) + 1
                performance["accurate_decisions"] = accuracy_contributions
                performance["decision_accuracy"] = accuracy_contributions / total_votes
    
    async def _broadcast_consensus_result(self, result: ConsensusResult):
        """Broadcast consensus result to all participants and interested parties"""
        
        # Send via event bus
        for agent_id in result.participating_agents:
            await self.event_bus.publish_event(AgentEvent(
                event_type=AgentEventType.CONSENSUS_RESPONSE.value,
                agent_id=agent_id,
                timestamp=datetime.now().isoformat(),
                payload={
                    "consensus_id": result.consensus_id,
                    "final_decision": result.final_decision,
                    "confidence": result.confidence,
                    "consensus_achieved": result.consensus_achieved,
                    "your_vote_included": True
                },
                correlation_id=result.consensus_id
            ))
        
        # Broadcast via WebSocket
        coordination_message = AgentCoordinationMessage(
            sender_agent_id="multi_agent_coordinator",
            recipient_agent_id=None,  # Broadcast
            message_type="consensus_result",
            payload={
                "consensus_id": result.consensus_id,
                "symbol": result.symbol,
                "final_decision": result.final_decision,
                "confidence": result.confidence,
                "consensus_achieved": result.consensus_achieved,
                "participating_agents": result.participating_agents
            },
            timestamp=datetime.now().isoformat()
        )
        
        await agent_coordination_manager.broadcast_coordination_message(coordination_message)
    
    def get_consensus_result(self, consensus_id: str) -> Optional[ConsensusResult]:
        """Get consensus result by ID"""
        return self.consensus_results.get(consensus_id)
    
    def get_agent_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summary for all agents"""
        return dict(self.agent_performance)
    
    def get_consensus_history(self, limit: int = 50) -> List[ConsensusResult]:
        """Get recent consensus history"""
        return self.consensus_history[-limit:]
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for multi-agent coordinator"""
        return {
            "status": "healthy",
            "active_consensus_requests": len(self.active_consensus),
            "total_consensus_processed": len(self.consensus_history),
            "tracked_agents": len(self.agent_performance),
            "average_consensus_time_ms": statistics.mean([
                r.processing_time_ms for r in self.consensus_history[-10:]
            ]) if self.consensus_history else 0.0,
            "consensus_success_rate": sum(
                1 for r in self.consensus_history[-20:] if r.consensus_achieved
            ) / min(20, len(self.consensus_history)) if self.consensus_history else 0.0,
            "timestamp": datetime.now().isoformat()
        }


# Global multi-agent coordinator instance
multi_agent_coordinator = MultiAgentCoordinator()