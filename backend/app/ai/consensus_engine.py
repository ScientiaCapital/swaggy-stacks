"""
Consensus Engine - Multi-Agent Decision Making System
===================================================

Advanced consensus mechanism for coordinating trading decisions across multiple AI agents.
Handles weighted voting, conflict resolution, and confidence-based trade execution.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class VoteType(Enum):
    """Types of agent votes"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    ABSTAIN = "abstain"


class ConsensusMode(Enum):
    """Consensus calculation modes"""
    SIMPLE_MAJORITY = "simple_majority"
    WEIGHTED_MAJORITY = "weighted_majority"
    UNANIMOUS = "unanimous"
    SUPER_MAJORITY = "super_majority"  # 2/3 threshold


@dataclass
class AgentVote:
    """Individual agent vote data"""
    agent_name: str
    vote: VoteType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    risk_assessment: Dict[str, Any]
    position_size_suggestion: Optional[float] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class ConsensusResult:
    """Result of consensus calculation"""
    decision_id: str
    symbol: str
    final_vote: VoteType
    confidence_score: float
    participating_agents: List[str]
    vote_breakdown: Dict[str, int]
    weighted_score: float
    execution_recommended: bool
    risk_score: float
    suggested_position_size: float
    consensus_reasoning: str
    timestamp: str
    execution_conditions: Dict[str, Any]


class ConsensusEngine:
    """
    Advanced consensus engine for multi-agent trading decisions

    Features:
    - Weighted voting based on agent expertise and historical performance
    - Confidence-based decision making
    - Risk assessment integration
    - Position sizing consensus
    - Conflict resolution mechanisms
    """

    def __init__(
        self,
        consensus_mode: ConsensusMode = ConsensusMode.WEIGHTED_MAJORITY,
        minimum_agents: int = 3,
        confidence_threshold: float = 0.6,
        risk_threshold: float = 0.7
    ):
        self.consensus_mode = consensus_mode
        self.minimum_agents = minimum_agents
        self.confidence_threshold = confidence_threshold
        self.risk_threshold = risk_threshold

        # Agent weights and performance tracking
        self.agent_weights: Dict[str, float] = {
            "analyst": 0.25,     # Market analysis specialist
            "risk": 0.30,        # Risk management (highest weight)
            "strategist": 0.25,  # Strategy optimization
            "chat": 0.10,        # Coordination specialist
            "reasoning": 0.10    # Pattern analysis
        }

        self.agent_performance: Dict[str, Dict[str, Any]] = {}

        # Consensus callbacks
        self.decision_callbacks: List[Callable] = []
        self.conflict_callbacks: List[Callable] = []

        # Statistics
        self.consensus_stats = {
            "decisions_made": 0,
            "successful_consensus": 0,
            "conflicts_resolved": 0,
            "avg_confidence": 0.0,
            "avg_execution_rate": 0.0
        }

        logger.info("Consensus engine initialized",
                   mode=consensus_mode.value,
                   min_agents=minimum_agents)

    async def calculate_consensus(
        self,
        decision_id: str,
        symbol: str,
        votes: List[AgentVote],
        timeout: int = 30
    ) -> ConsensusResult:
        """Calculate consensus from agent votes"""
        try:
            if len(votes) < self.minimum_agents:
                raise ValueError(f"Insufficient votes: {len(votes)} < {self.minimum_agents}")

            # Validate votes
            validated_votes = await self._validate_votes(votes)

            # Calculate vote breakdown
            vote_breakdown = self._calculate_vote_breakdown(validated_votes)

            # Calculate weighted scores
            weighted_scores = await self._calculate_weighted_scores(validated_votes)

            # Determine final decision
            final_vote, weighted_score = self._determine_final_vote(weighted_scores, vote_breakdown)

            # Calculate confidence score
            confidence_score = await self._calculate_confidence_score(validated_votes, final_vote)

            # Assess overall risk
            risk_score = await self._assess_consensus_risk(validated_votes, final_vote)

            # Determine position sizing
            suggested_position_size = await self._calculate_position_size(validated_votes, confidence_score, risk_score)

            # Generate reasoning
            consensus_reasoning = await self._generate_consensus_reasoning(validated_votes, final_vote, confidence_score)

            # Determine execution recommendation
            execution_recommended = await self._should_execute_decision(
                final_vote, confidence_score, risk_score, validated_votes
            )

            # Create execution conditions
            execution_conditions = await self._create_execution_conditions(
                validated_votes, confidence_score, risk_score
            )

            # Build consensus result
            result = ConsensusResult(
                decision_id=decision_id,
                symbol=symbol,
                final_vote=final_vote,
                confidence_score=confidence_score,
                participating_agents=[vote.agent_name for vote in validated_votes],
                vote_breakdown=vote_breakdown,
                weighted_score=weighted_score,
                execution_recommended=execution_recommended,
                risk_score=risk_score,
                suggested_position_size=suggested_position_size,
                consensus_reasoning=consensus_reasoning,
                timestamp=datetime.now(timezone.utc).isoformat(),
                execution_conditions=execution_conditions
            )

            # Update statistics
            self.consensus_stats["decisions_made"] += 1
            if execution_recommended:
                self.consensus_stats["successful_consensus"] += 1

            # Update running averages
            self._update_running_stats(confidence_score, execution_recommended)

            # Notify callbacks
            await self._notify_decision_callbacks(result)

            logger.info("Consensus calculated",
                       decision_id=decision_id,
                       symbol=symbol,
                       final_vote=final_vote.value,
                       confidence=confidence_score,
                       execution=execution_recommended)

            return result

        except Exception as e:
            logger.error("Failed to calculate consensus",
                        decision_id=decision_id,
                        error=str(e))
            raise

    async def _validate_votes(self, votes: List[AgentVote]) -> List[AgentVote]:
        """Validate and filter agent votes"""
        validated_votes = []

        for vote in votes:
            # Check confidence bounds
            if not 0.0 <= vote.confidence <= 1.0:
                logger.warning("Invalid confidence score",
                              agent=vote.agent_name,
                              confidence=vote.confidence)
                continue

            # Check if agent is known
            if vote.agent_name not in self.agent_weights:
                logger.warning("Unknown agent vote", agent=vote.agent_name)
                continue

            # Check vote type
            if vote.vote not in VoteType:
                logger.warning("Invalid vote type",
                              agent=vote.agent_name,
                              vote=vote.vote)
                continue

            validated_votes.append(vote)

        return validated_votes

    def _calculate_vote_breakdown(self, votes: List[AgentVote]) -> Dict[str, int]:
        """Calculate vote breakdown by type"""
        breakdown = {vote_type.value: 0 for vote_type in VoteType}

        for vote in votes:
            breakdown[vote.vote.value] += 1

        return breakdown

    async def _calculate_weighted_scores(self, votes: List[AgentVote]) -> Dict[str, float]:
        """Calculate weighted scores for each vote type"""
        scores = {vote_type.value: 0.0 for vote_type in VoteType}
        total_weight = 0.0

        for vote in votes:
            agent_weight = self.agent_weights.get(vote.agent_name, 0.1)

            # Adjust weight by confidence and historical performance
            performance_multiplier = await self._get_agent_performance_multiplier(vote.agent_name)
            adjusted_weight = agent_weight * vote.confidence * performance_multiplier

            scores[vote.vote.value] += adjusted_weight
            total_weight += adjusted_weight

        # Normalize scores
        if total_weight > 0:
            for vote_type in scores:
                scores[vote_type] /= total_weight

        return scores

    def _determine_final_vote(self, weighted_scores: Dict[str, float], vote_breakdown: Dict[str, int]) -> tuple:
        """Determine final consensus vote"""
        # Remove abstain votes from consideration
        active_scores = {k: v for k, v in weighted_scores.items() if k != VoteType.ABSTAIN.value}

        if not active_scores:
            return VoteType.HOLD, 0.0

        # Find highest scoring vote
        final_vote_str = max(active_scores, key=active_scores.get)
        final_vote = VoteType(final_vote_str)
        weighted_score = active_scores[final_vote_str]

        # Apply consensus mode rules
        if self.consensus_mode == ConsensusMode.UNANIMOUS:
            if len(set(vote.value for vote in active_scores.keys())) > 1:
                return VoteType.HOLD, weighted_score  # No consensus

        elif self.consensus_mode == ConsensusMode.SUPER_MAJORITY:
            if weighted_score < 0.67:  # 2/3 threshold
                return VoteType.HOLD, weighted_score

        return final_vote, weighted_score

    async def _calculate_confidence_score(self, votes: List[AgentVote], final_vote: VoteType) -> float:
        """Calculate overall confidence in the decision"""
        if not votes:
            return 0.0

        # Confidence of agents voting for final decision
        supporting_confidences = []
        for vote in votes:
            if vote.vote == final_vote:
                supporting_confidences.append(vote.confidence)

        if not supporting_confidences:
            return 0.0

        # Weighted average confidence
        total_weight = 0.0
        weighted_confidence = 0.0

        for vote in votes:
            if vote.vote == final_vote:
                agent_weight = self.agent_weights.get(vote.agent_name, 0.1)
                weighted_confidence += vote.confidence * agent_weight
                total_weight += agent_weight

        return weighted_confidence / total_weight if total_weight > 0 else 0.0

    async def _assess_consensus_risk(self, votes: List[AgentVote], final_vote: VoteType) -> float:
        """Assess overall risk of the consensus decision"""
        if not votes or final_vote == VoteType.HOLD:
            return 0.0

        risk_scores = []
        for vote in votes:
            if vote.vote == final_vote:
                # Extract risk score from agent's risk assessment
                risk_assessment = vote.risk_assessment
                agent_risk = risk_assessment.get('risk_score', 0.5)
                risk_scores.append(agent_risk)

        return sum(risk_scores) / len(risk_scores) if risk_scores else 0.5

    async def _calculate_position_size(
        self,
        votes: List[AgentVote],
        confidence_score: float,
        risk_score: float
    ) -> float:
        """Calculate suggested position size based on consensus"""
        if not votes:
            return 0.0

        # Get position size suggestions from agents
        position_suggestions = []
        for vote in votes:
            if vote.position_size_suggestion is not None:
                position_suggestions.append(vote.position_size_suggestion)

        if not position_suggestions:
            # Default position sizing based on confidence and risk
            base_size = 1000.0  # Base position size
            confidence_factor = confidence_score
            risk_factor = max(0.1, 1.0 - risk_score)  # Lower risk = larger position

            return base_size * confidence_factor * risk_factor

        # Average of agent suggestions, adjusted for confidence and risk
        avg_suggestion = sum(position_suggestions) / len(position_suggestions)
        confidence_factor = confidence_score
        risk_factor = max(0.1, 1.0 - risk_score)

        return avg_suggestion * confidence_factor * risk_factor

    async def _generate_consensus_reasoning(
        self,
        votes: List[AgentVote],
        final_vote: VoteType,
        confidence_score: float
    ) -> str:
        """Generate human-readable consensus reasoning"""
        if not votes:
            return "No valid votes received"

        # Count supporting and opposing votes
        supporting_votes = [v for v in votes if v.vote == final_vote]
        total_votes = len(votes)

        reasoning_parts = []

        # Overall decision summary
        reasoning_parts.append(
            f"{len(supporting_votes)}/{total_votes} agents support {final_vote.value.upper()} "
            f"with {confidence_score:.1%} confidence"
        )

        # Key reasoning from supporting agents
        key_reasoning = []
        for vote in supporting_votes[:3]:  # Top 3 supporting reasons
            if vote.reasoning:
                key_reasoning.append(f"{vote.agent_name}: {vote.reasoning}")

        if key_reasoning:
            reasoning_parts.append("Key reasoning: " + "; ".join(key_reasoning))

        return ". ".join(reasoning_parts)

    async def _should_execute_decision(
        self,
        final_vote: VoteType,
        confidence_score: float,
        risk_score: float,
        votes: List[AgentVote]
    ) -> bool:
        """Determine if decision should be executed"""
        # Don't execute HOLD or ABSTAIN
        if final_vote in [VoteType.HOLD, VoteType.ABSTAIN]:
            return False

        # Check confidence threshold
        if confidence_score < self.confidence_threshold:
            return False

        # Check risk threshold
        if risk_score > self.risk_threshold:
            return False

        # Check if risk agent approves (critical veto power)
        risk_agent_vote = next((v for v in votes if v.agent_name == "risk"), None)
        if risk_agent_vote and risk_agent_vote.vote == VoteType.HOLD:
            return False  # Risk agent has veto power

        return True

    async def _create_execution_conditions(
        self,
        votes: List[AgentVote],
        confidence_score: float,
        risk_score: float
    ) -> Dict[str, Any]:
        """Create conditions for trade execution"""
        return {
            "min_confidence": confidence_score,
            "max_risk": risk_score,
            "risk_agent_approval": any(
                v.agent_name == "risk" and v.vote != VoteType.HOLD
                for v in votes
            ),
            "execution_timeout": 300,  # 5 minutes
            "market_conditions": {
                "max_spread_pct": 0.5,  # Max 0.5% spread
                "min_volume": 100000    # Minimum volume requirement
            }
        }

    async def _get_agent_performance_multiplier(self, agent_name: str) -> float:
        """Get performance-based weight multiplier for agent"""
        if agent_name not in self.agent_performance:
            return 1.0  # Default multiplier

        perf = self.agent_performance[agent_name]

        # Simple performance multiplier based on historical accuracy
        accuracy = perf.get('accuracy', 0.5)
        return max(0.5, min(1.5, accuracy * 2.0))  # Range: 0.5x to 1.5x

    def _update_running_stats(self, confidence_score: float, execution_recommended: bool):
        """Update running statistics"""
        total_decisions = self.consensus_stats["decisions_made"]

        # Update average confidence
        current_avg_conf = self.consensus_stats["avg_confidence"]
        self.consensus_stats["avg_confidence"] = (
            (current_avg_conf * (total_decisions - 1) + confidence_score) / total_decisions
        )

        # Update execution rate
        execution_count = self.consensus_stats["successful_consensus"]
        self.consensus_stats["avg_execution_rate"] = execution_count / total_decisions

    async def _notify_decision_callbacks(self, result: ConsensusResult):
        """Notify registered callbacks of consensus decision"""
        for callback in self.decision_callbacks:
            try:
                await callback(result)
            except Exception as e:
                logger.error("Consensus decision callback failed", error=str(e))

    # Public API
    def add_decision_callback(self, callback: Callable):
        """Register callback for consensus decisions"""
        self.decision_callbacks.append(callback)

    def add_conflict_callback(self, callback: Callable):
        """Register callback for conflict resolution"""
        self.conflict_callbacks.append(callback)

    def update_agent_weight(self, agent_name: str, weight: float):
        """Update agent voting weight"""
        if 0.0 <= weight <= 1.0:
            self.agent_weights[agent_name] = weight
            logger.info("Agent weight updated", agent=agent_name, weight=weight)

    def update_agent_performance(self, agent_name: str, performance_data: Dict[str, Any]):
        """Update agent performance metrics"""
        self.agent_performance[agent_name] = performance_data
        logger.info("Agent performance updated", agent=agent_name)

    def get_consensus_stats(self) -> Dict[str, Any]:
        """Get consensus engine statistics"""
        return {
            **self.consensus_stats,
            "agent_weights": self.agent_weights.copy(),
            "configuration": {
                "consensus_mode": self.consensus_mode.value,
                "minimum_agents": self.minimum_agents,
                "confidence_threshold": self.confidence_threshold,
                "risk_threshold": self.risk_threshold
            }
        }


# Global consensus engine instance
consensus_engine: Optional[ConsensusEngine] = None


def get_consensus_engine() -> ConsensusEngine:
    """Get or create global consensus engine instance"""
    global consensus_engine

    if consensus_engine is None:
        consensus_engine = ConsensusEngine()

    return consensus_engine