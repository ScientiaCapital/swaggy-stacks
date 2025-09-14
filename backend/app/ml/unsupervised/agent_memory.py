"""
Agent Memory - Experience clustering and replay for autonomous learning

Provides long-term memory capabilities for AI agents through experience clustering,
prioritized experience replay, and adaptive learning mechanisms.
"""

import asyncio
import pickle
import random
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import structlog
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)


@dataclass
class Experience:
    """Individual experience record for agent learning"""

    experience_id: str
    agent_id: str
    agent_type: str
    state: Dict[str, Any]  # Market state, context
    action: Dict[str, Any]  # Action taken
    result: Dict[str, Any]  # Outcome
    reward: float  # Success metric
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class ExperienceCluster:
    """Clustered experiences for pattern learning"""

    cluster_id: int
    cluster_type: str  # high_reward, low_reward, mixed
    experience_count: int
    avg_reward: float
    common_patterns: Dict[str, Any]
    representative_experience: Experience
    cluster_features: List[float]
    confidence_score: float
    last_updated: datetime


@dataclass
class LearningInsight:
    """Learning insights extracted from experience clusters"""

    agent_type: str
    insight_type: str  # pattern, timing, context
    description: str
    confidence: float
    impact_score: float
    supporting_experiences: int
    actionable_recommendations: List[str]
    validation_metrics: Dict[str, float]
    created_at: datetime


class AgentMemory:
    """Enhanced agent memory with experience clustering and replay"""

    def __init__(
        self,
        max_experiences: int = 50000,
        cluster_update_interval: int = 100,
        memory_persistence_path: Optional[str] = None
    ):
        self.max_experiences = max_experiences
        self.cluster_update_interval = cluster_update_interval
        self.memory_path = Path(memory_persistence_path) if memory_persistence_path else None

        # Experience storage
        self.experiences: deque[Experience] = deque(maxlen=max_experiences)
        self.experiences_by_agent: Dict[str, deque[Experience]] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self.experiences_by_type: Dict[str, deque[Experience]] = defaultdict(
            lambda: deque(maxlen=10000)
        )

        # Clustering components
        self.experience_clusters: Dict[str, List[ExperienceCluster]] = {}
        self.scaler = StandardScaler()
        self.clusterer = DBSCAN(eps=0.3, min_samples=5)

        # Learning components
        self.learning_insights: Dict[str, List[LearningInsight]] = {}
        self.replay_priorities: Dict[str, float] = {}

        # Performance tracking
        self.experience_count = 0
        self.cluster_update_count = 0
        self.last_cluster_update: Optional[datetime] = None

        # Load persisted memory if available
        if self.memory_path and self.memory_path.exists():
            self._load_memory()

    async def store_experience(
        self,
        agent_id: str,
        agent_type: str,
        state: Dict[str, Any],
        action: Dict[str, Any],
        result: Dict[str, Any],
        reward: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a new experience in agent memory"""

        experience_id = f"{agent_id}_{self.experience_count}_{datetime.now().timestamp()}"

        experience = Experience(
            experience_id=experience_id,
            agent_id=agent_id,
            agent_type=agent_type,
            state=state,
            action=action,
            result=result,
            reward=reward,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        # Store experience
        self.experiences.append(experience)
        self.experiences_by_agent[agent_id].append(experience)
        self.experiences_by_type[agent_type].append(experience)

        self.experience_count += 1

        # Update replay priorities based on reward
        self.replay_priorities[experience_id] = self._calculate_replay_priority(experience)

        # Trigger clustering update if needed
        if self.experience_count % self.cluster_update_interval == 0:
            await self._update_clusters()

        logger.debug(
            "Experience stored",
            agent_id=agent_id,
            experience_id=experience_id,
            reward=reward
        )

        return experience_id

    def _calculate_replay_priority(self, experience: Experience) -> float:
        """Calculate replay priority based on experience characteristics"""

        # Base priority from reward
        reward_priority = abs(experience.reward)

        # Recency bonus (more recent experiences have higher priority)
        age_hours = (datetime.now() - experience.timestamp).total_seconds() / 3600
        recency_bonus = max(0.1, 1.0 - (age_hours / 168.0))  # Decay over a week

        # Novelty bonus (experiences from less common states have higher priority)
        state_hash = hash(str(sorted(experience.state.items())))
        state_frequency = sum(1 for exp in self.experiences if hash(str(sorted(exp.state.items()))) == state_hash)
        novelty_bonus = max(0.5, 2.0 - (state_frequency / 100.0))

        # Final priority
        priority = reward_priority * recency_bonus * novelty_bonus

        return min(10.0, max(0.1, priority))  # Clamp between 0.1 and 10.0

    async def sample_experiences(
        self,
        agent_type: str,
        batch_size: int = 32,
        prioritized: bool = True,
        min_reward: Optional[float] = None
    ) -> List[Experience]:
        """Sample experiences for training/learning"""

        # Get experiences for agent type
        type_experiences = list(self.experiences_by_type.get(agent_type, []))

        if not type_experiences:
            return []

        # Filter by minimum reward if specified
        if min_reward is not None:
            type_experiences = [exp for exp in type_experiences if exp.reward >= min_reward]

        if len(type_experiences) <= batch_size:
            return type_experiences

        if prioritized:
            # Prioritized sampling based on replay priorities
            priorities = [
                self.replay_priorities.get(exp.experience_id, 1.0)
                for exp in type_experiences
            ]

            # Convert to probabilities
            total_priority = sum(priorities)
            if total_priority > 0:
                probabilities = [p / total_priority for p in priorities]

                # Sample with replacement
                indices = np.random.choice(
                    len(type_experiences),
                    size=min(batch_size, len(type_experiences)),
                    replace=False,
                    p=probabilities
                )

                return [type_experiences[i] for i in indices]

        # Random sampling fallback
        return random.sample(type_experiences, min(batch_size, len(type_experiences)))

    async def _update_clusters(self):
        """Update experience clusters for pattern learning"""

        try:
            for agent_type in self.experiences_by_type.keys():
                await self._cluster_agent_experiences(agent_type)

            self.cluster_update_count += 1
            self.last_cluster_update = datetime.now()

            logger.info("Experience clusters updated",
                       agent_types=len(self.experiences_by_type))

        except Exception as e:
            logger.error("Failed to update experience clusters", error=str(e))

    async def _cluster_agent_experiences(self, agent_type: str):
        """Cluster experiences for a specific agent type"""

        experiences = list(self.experiences_by_type[agent_type])

        if len(experiences) < 10:  # Need minimum experiences for clustering
            return

        try:
            # Extract features from experiences
            features = []
            for exp in experiences:
                feature_vector = self._extract_experience_features(exp)
                features.append(feature_vector)

            if not features:
                return

            features_array = np.array(features)

            # Normalize features
            features_scaled = self.scaler.fit_transform(features_array)

            # Perform DBSCAN clustering
            cluster_labels = self.clusterer.fit_predict(features_scaled)

            # Create cluster objects
            clusters = []
            unique_labels = set(cluster_labels)

            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue

                cluster_experiences = [
                    exp for i, exp in enumerate(experiences)
                    if cluster_labels[i] == label
                ]

                if len(cluster_experiences) < 3:  # Skip very small clusters
                    continue

                # Calculate cluster statistics
                rewards = [exp.reward for exp in cluster_experiences]
                avg_reward = np.mean(rewards)

                # Determine cluster type
                if avg_reward >= 0.8:
                    cluster_type = "high_reward"
                elif avg_reward <= 0.2:
                    cluster_type = "low_reward"
                else:
                    cluster_type = "mixed"

                # Extract common patterns
                common_patterns = self._extract_common_experience_patterns(cluster_experiences)

                # Find representative experience (closest to centroid)
                cluster_indices = [i for i, label_val in enumerate(cluster_labels) if label_val == label]
                cluster_features = features_scaled[cluster_indices]
                centroid = np.mean(cluster_features, axis=0)

                distances = [np.linalg.norm(features_scaled[i] - centroid) for i in cluster_indices]
                representative_idx = cluster_indices[np.argmin(distances)]
                representative_experience = experiences[representative_idx]

                # Calculate confidence score
                cluster_variance = np.var(cluster_features, axis=0)
                confidence_score = max(0.1, 1.0 - np.mean(cluster_variance))

                cluster = ExperienceCluster(
                    cluster_id=label,
                    cluster_type=cluster_type,
                    experience_count=len(cluster_experiences),
                    avg_reward=avg_reward,
                    common_patterns=common_patterns,
                    representative_experience=representative_experience,
                    cluster_features=np.mean(cluster_features, axis=0).tolist(),
                    confidence_score=confidence_score,
                    last_updated=datetime.now()
                )

                clusters.append(cluster)

            # Store clusters
            self.experience_clusters[agent_type] = clusters

            # Generate learning insights
            await self._generate_learning_insights(agent_type, clusters)

            logger.info(f"Clustered {len(experiences)} experiences into {len(clusters)} patterns for {agent_type}")

        except Exception as e:
            logger.error(f"Failed to cluster experiences for {agent_type}", error=str(e))

    def _extract_experience_features(self, experience: Experience) -> List[float]:
        """Extract numerical features from experience for clustering"""

        features = []

        # Basic features
        features.append(experience.reward)
        features.append(experience.timestamp.hour)  # Hour of day
        features.append(experience.timestamp.weekday())  # Day of week

        # State features
        state_features = []
        for key, value in experience.state.items():
            if isinstance(value, (int, float)):
                state_features.append(float(value))
            elif isinstance(value, bool):
                state_features.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # Use hash for categorical values
                state_features.append(float(hash(value) % 1000))

        # Pad or truncate to fixed size
        max_state_features = 15
        if len(state_features) > max_state_features:
            state_features = state_features[:max_state_features]
        else:
            state_features.extend([0.0] * (max_state_features - len(state_features)))

        features.extend(state_features)

        # Action features
        action_features = []
        for key, value in experience.action.items():
            if isinstance(value, (int, float)):
                action_features.append(float(value))
            elif isinstance(value, bool):
                action_features.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                action_features.append(float(hash(value) % 1000))

        # Pad or truncate to fixed size
        max_action_features = 10
        if len(action_features) > max_action_features:
            action_features = action_features[:max_action_features]
        else:
            action_features.extend([0.0] * (max_action_features - len(action_features)))

        features.extend(action_features)

        return features

    def _extract_common_experience_patterns(self, experiences: List[Experience]) -> Dict[str, Any]:
        """Extract common patterns from a list of experiences"""

        if not experiences:
            return {}

        patterns = {}

        # Common state patterns
        state_patterns = {}
        all_state_keys = set()
        for exp in experiences:
            all_state_keys.update(exp.state.keys())

        for key in all_state_keys:
            values = [exp.state.get(key) for exp in experiences if key in exp.state]
            if len(values) >= len(experiences) * 0.6:  # Appears in 60% of experiences
                if all(isinstance(v, (int, float)) for v in values):
                    state_patterns[key] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "frequency": len(values) / len(experiences)
                    }
                elif all(isinstance(v, str) for v in values):
                    # Most common categorical value
                    value_counts = {}
                    for v in values:
                        value_counts[v] = value_counts.get(v, 0) + 1
                    most_common = max(value_counts.items(), key=lambda x: x[1])
                    if most_common[1] >= len(values) * 0.5:
                        state_patterns[key] = most_common[0]

        patterns["state_patterns"] = state_patterns

        # Common action patterns
        action_patterns = {}
        all_action_keys = set()
        for exp in experiences:
            all_action_keys.update(exp.action.keys())

        for key in all_action_keys:
            values = [exp.action.get(key) for exp in experiences if key in exp.action]
            if len(values) >= len(experiences) * 0.6:
                if all(isinstance(v, (int, float)) for v in values):
                    action_patterns[key] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "frequency": len(values) / len(experiences)
                    }
                elif all(isinstance(v, str) for v in values):
                    value_counts = {}
                    for v in values:
                        value_counts[v] = value_counts.get(v, 0) + 1
                    most_common = max(value_counts.items(), key=lambda x: x[1])
                    if most_common[1] >= len(values) * 0.5:
                        action_patterns[key] = most_common[0]

        patterns["action_patterns"] = action_patterns

        # Reward statistics
        rewards = [exp.reward for exp in experiences]
        patterns["reward_stats"] = {
            "mean": np.mean(rewards),
            "std": np.std(rewards),
            "min": np.min(rewards),
            "max": np.max(rewards)
        }

        return patterns

    async def _generate_learning_insights(self, agent_type: str, clusters: List[ExperienceCluster]):
        """Generate learning insights from experience clusters"""

        insights = []

        try:
            # Find high-performing patterns
            high_reward_clusters = [c for c in clusters if c.cluster_type == "high_reward"]

            for cluster in high_reward_clusters:
                # Pattern insight
                pattern_insight = LearningInsight(
                    agent_type=agent_type,
                    insight_type="pattern",
                    description=f"High-reward pattern identified with {cluster.experience_count} experiences, "
                              f"average reward {cluster.avg_reward:.3f}",
                    confidence=cluster.confidence_score,
                    impact_score=cluster.avg_reward,
                    supporting_experiences=cluster.experience_count,
                    actionable_recommendations=self._generate_pattern_recommendations(cluster),
                    validation_metrics={
                        "avg_reward": cluster.avg_reward,
                        "experience_count": cluster.experience_count,
                        "confidence": cluster.confidence_score
                    },
                    created_at=datetime.now()
                )

                insights.append(pattern_insight)

            # Timing insights
            timing_insight = self._analyze_timing_patterns(clusters)
            if timing_insight:
                insights.append(timing_insight)

            # Context insights
            context_insight = self._analyze_context_patterns(clusters)
            if context_insight:
                insights.append(context_insight)

            # Store insights
            self.learning_insights[agent_type] = insights

            logger.info(f"Generated {len(insights)} learning insights for {agent_type}")

        except Exception as e:
            logger.error(f"Failed to generate learning insights for {agent_type}", error=str(e))

    def _generate_pattern_recommendations(self, cluster: ExperienceCluster) -> List[str]:
        """Generate actionable recommendations from cluster patterns"""

        recommendations = []

        patterns = cluster.common_patterns

        # State-based recommendations
        if "state_patterns" in patterns:
            for key, pattern in patterns["state_patterns"].items():
                if isinstance(pattern, dict) and "mean" in pattern:
                    recommendations.append(
                        f"Optimize {key} around {pattern['mean']:.3f} "
                        f"(±{pattern['std']:.3f}) for better results"
                    )
                elif isinstance(pattern, str):
                    recommendations.append(f"Prefer {key}='{pattern}' for higher success rates")

        # Action-based recommendations
        if "action_patterns" in patterns:
            for key, pattern in patterns["action_patterns"].items():
                if isinstance(pattern, dict) and "mean" in pattern:
                    recommendations.append(
                        f"Set {key} to approximately {pattern['mean']:.3f} "
                        f"for optimal performance"
                    )
                elif isinstance(pattern, str):
                    recommendations.append(f"Use {key}='{pattern}' for best outcomes")

        return recommendations

    def _analyze_timing_patterns(self, clusters: List[ExperienceCluster]) -> Optional[LearningInsight]:
        """Analyze timing patterns across clusters"""

        try:
            # Analyze hour-of-day patterns
            hour_rewards = defaultdict(list)

            for cluster in clusters:
                exp = cluster.representative_experience
                hour_rewards[exp.timestamp.hour].append(cluster.avg_reward)

            if len(hour_rewards) > 3:  # Need sufficient time diversity
                hour_avg_rewards = {
                    hour: np.mean(rewards)
                    for hour, rewards in hour_rewards.items()
                }

                best_hour = max(hour_avg_rewards.items(), key=lambda x: x[1])
                worst_hour = min(hour_avg_rewards.items(), key=lambda x: x[1])

                if best_hour[1] > worst_hour[1] * 1.2:  # Significant difference
                    return LearningInsight(
                        agent_type=clusters[0].representative_experience.agent_type,
                        insight_type="timing",
                        description=f"Performance varies by time: "
                                  f"best at hour {best_hour[0]} (reward: {best_hour[1]:.3f}), "
                                  f"worst at hour {worst_hour[0]} (reward: {worst_hour[1]:.3f})",
                        confidence=0.8,
                        impact_score=best_hour[1] - worst_hour[1],
                        supporting_experiences=sum(len(hour_rewards[h]) for h in [best_hour[0], worst_hour[0]]),
                        actionable_recommendations=[
                            f"Schedule high-priority actions around hour {best_hour[0]}",
                            f"Avoid or use reduced confidence during hour {worst_hour[0]}"
                        ],
                        validation_metrics={
                            "best_hour": best_hour[0],
                            "best_reward": best_hour[1],
                            "worst_hour": worst_hour[0],
                            "worst_reward": worst_hour[1],
                            "improvement_potential": best_hour[1] - worst_hour[1]
                        },
                        created_at=datetime.now()
                    )

        except Exception as e:
            logger.error("Failed to analyze timing patterns", error=str(e))

        return None

    def _analyze_context_patterns(self, clusters: List[ExperienceCluster]) -> Optional[LearningInsight]:
        """Analyze context patterns across clusters"""

        try:
            # Analyze common context factors that lead to high rewards
            high_reward_contexts = []
            low_reward_contexts = []

            for cluster in clusters:
                if cluster.cluster_type == "high_reward":
                    high_reward_contexts.append(cluster.common_patterns.get("state_patterns", {}))
                elif cluster.cluster_type == "low_reward":
                    low_reward_contexts.append(cluster.common_patterns.get("state_patterns", {}))

            if high_reward_contexts:
                # Find context factors that appear frequently in high-reward clusters
                context_factors = defaultdict(int)
                for context in high_reward_contexts:
                    for key in context.keys():
                        context_factors[key] += 1

                important_factors = [
                    key for key, count in context_factors.items()
                    if count >= len(high_reward_contexts) * 0.6
                ]

                if important_factors:
                    return LearningInsight(
                        agent_type=clusters[0].representative_experience.agent_type,
                        insight_type="context",
                        description=f"Key context factors for success: {', '.join(important_factors)}",
                        confidence=0.7,
                        impact_score=len(important_factors) / 10.0,
                        supporting_experiences=len(high_reward_contexts),
                        actionable_recommendations=[
                            f"Monitor and optimize {factor}" for factor in important_factors[:3]
                        ],
                        validation_metrics={
                            "important_factors": important_factors,
                            "factor_frequency": dict(context_factors)
                        },
                        created_at=datetime.now()
                    )

        except Exception as e:
            logger.error("Failed to analyze context patterns", error=str(e))

        return None

    def get_learning_insights(self, agent_type: str) -> List[LearningInsight]:
        """Get learning insights for an agent type"""
        return self.learning_insights.get(agent_type, [])

    def get_experience_clusters(self, agent_type: str) -> List[ExperienceCluster]:
        """Get experience clusters for an agent type"""
        return self.experience_clusters.get(agent_type, [])

    def _save_memory(self):
        """Save memory to disk for persistence"""
        if not self.memory_path:
            return

        try:
            memory_data = {
                'experiences': list(self.experiences),
                'experience_clusters': self.experience_clusters,
                'learning_insights': self.learning_insights,
                'replay_priorities': self.replay_priorities,
                'experience_count': self.experience_count,
                'cluster_update_count': self.cluster_update_count,
                'last_cluster_update': self.last_cluster_update
            }

            with open(self.memory_path, 'wb') as f:
                pickle.dump(memory_data, f)

            logger.info("Agent memory saved to disk", path=str(self.memory_path))

        except Exception as e:
            logger.error("Failed to save agent memory", error=str(e))

    def _load_memory(self):
        """Load memory from disk"""
        try:
            with open(self.memory_path, 'rb') as f:
                memory_data = pickle.load(f)

            # Restore memory state
            self.experiences.extend(memory_data.get('experiences', []))
            self.experience_clusters = memory_data.get('experience_clusters', {})
            self.learning_insights = memory_data.get('learning_insights', {})
            self.replay_priorities = memory_data.get('replay_priorities', {})
            self.experience_count = memory_data.get('experience_count', 0)
            self.cluster_update_count = memory_data.get('cluster_update_count', 0)
            self.last_cluster_update = memory_data.get('last_cluster_update')

            # Rebuild auxiliary indexes
            for exp in self.experiences:
                self.experiences_by_agent[exp.agent_id].append(exp)
                self.experiences_by_type[exp.agent_type].append(exp)

            logger.info("Agent memory loaded from disk",
                       experiences=len(self.experiences),
                       clusters=len(self.experience_clusters))

        except Exception as e:
            logger.error("Failed to load agent memory", error=str(e))

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "total_experiences": len(self.experiences),
            "experiences_by_agent": {k: len(v) for k, v in self.experiences_by_agent.items()},
            "experiences_by_type": {k: len(v) for k, v in self.experiences_by_type.items()},
            "experience_clusters": {k: len(v) for k, v in self.experience_clusters.items()},
            "learning_insights": {k: len(v) for k, v in self.learning_insights.items()},
            "cluster_update_count": self.cluster_update_count,
            "last_cluster_update": self.last_cluster_update.isoformat() if self.last_cluster_update else None,
            "memory_persistence_enabled": self.memory_path is not None
        }


# Global agent memory instance
agent_memory = AgentMemory(
    memory_persistence_path="./data/agent_memory.pkl"
)