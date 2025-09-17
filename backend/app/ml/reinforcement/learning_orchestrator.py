"""
Learning Orchestrator - Unified coordination of all learning systems

Manages and coordinates:
- Reinforcement Learning (Q-learning, Policy Gradient)
- Unsupervised Learning (Clustering, Pattern Mining)
- Strategy Evolution (A/B Testing, Mutations)
- Real-time adaptation and continuous improvement

Optimized for M1 MacBook with efficient resource management.
"""

import asyncio
import json
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np

import structlog
import torch
from app.core.exceptions import TradingError

# Import RL components
from .q_learning import TradingQLearningAgent, QLearningConfig
from .policy_gradient import TradingPolicyGradientAgent, PolicyGradientConfig
from .value_functions import TemporalDifferenceLearner, ValueFunctionConfig
from .enhanced_memory import EnhancedAgentMemory

# Import unsupervised learning components
from app.ml.unsupervised.strategy_evolution import StrategyEvolution
from app.ml.unsupervised.pattern_mining import PatternMining
from app.ml.unsupervised.clustering import MarketRegimeDetector
from app.ml.unsupervised.anomaly_detector import AnomalyDetector

logger = structlog.get_logger()


class LearningMode(Enum):
    """Learning modes for different market conditions"""
    EXPLORATION = "exploration"  # High exploration for new markets
    EXPLOITATION = "exploitation"  # Low exploration for known patterns
    BALANCED = "balanced"  # Balance exploration and exploitation
    SAFE = "safe"  # Conservative learning with constraints
    AGGRESSIVE = "aggressive"  # High-risk high-reward learning


class AlgorithmType(Enum):
    """Available learning algorithms"""
    DQN = "dqn"
    PPO = "ppo"
    A2C = "a2c"
    SARSA = "sarsa"
    TD_LAMBDA = "td_lambda"
    HYBRID = "hybrid"


@dataclass
class LearningSchedule:
    """Schedule for learning updates"""
    algorithm: AlgorithmType
    update_frequency: int  # Steps between updates
    batch_size: int
    priority: float  # Resource allocation priority
    enabled: bool


@dataclass
class LearningMetrics:
    """Comprehensive learning metrics"""
    total_updates: int
    avg_reward: float
    avg_loss: float
    exploration_rate: float
    learning_rate: float
    success_rate: float
    market_regime: str
    algorithm_performance: Dict[str, float]
    resource_usage: Dict[str, float]
    timestamp: datetime


class LearningOrchestrator:
    """
    Unified orchestrator for all learning systems.

    Coordinates reinforcement learning, unsupervised learning, and strategy evolution
    to create a comprehensive continuous learning system.
    """

    def __init__(
        self,
        enable_q_learning: bool = True,
        enable_policy_gradient: bool = True,
        enable_td_learning: bool = True,
        enable_unsupervised: bool = True,
        enable_evolution: bool = True,
        memory_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize the learning orchestrator.

        Args:
            enable_q_learning: Enable Q-learning algorithms
            enable_policy_gradient: Enable policy gradient methods
            enable_td_learning: Enable TD learning
            enable_unsupervised: Enable unsupervised learning
            enable_evolution: Enable strategy evolution
            memory_path: Path for memory persistence
            checkpoint_path: Path for model checkpoints
        """

        self.memory_path = Path(memory_path) if memory_path else None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        # Initialize enhanced memory system
        self.memory = EnhancedAgentMemory(
            max_experiences=100000,
            memory_persistence_path=memory_path
        )

        # Initialize RL agents
        self.agents = {}

        if enable_q_learning:
            self.agents['dqn'] = TradingQLearningAgent(
                QLearningConfig(
                    state_dim=128,
                    action_dim=5,
                    use_double_dqn=True,
                    use_dueling_dqn=True
                )
            )

        if enable_policy_gradient:
            self.agents['ppo'] = TradingPolicyGradientAgent(
                PolicyGradientConfig(
                    state_dim=128,
                    action_dim=5,
                    algorithm="ppo"
                )
            )
            self.agents['a2c'] = TradingPolicyGradientAgent(
                PolicyGradientConfig(
                    state_dim=128,
                    action_dim=5,
                    algorithm="a2c"
                )
            )

        if enable_td_learning:
            self.td_learner = TemporalDifferenceLearner(
                ValueFunctionConfig(
                    state_dim=128,
                    method="td_lambda"
                )
            )

        # Initialize unsupervised learning
        if enable_unsupervised:
            self.regime_detector = MarketRegimeDetector()
            self.anomaly_detector = AnomalyDetector()
            self.pattern_miner = PatternMining()

        # Initialize strategy evolution
        if enable_evolution:
            self.strategy_evolution = StrategyEvolution()

        # Learning schedules for each algorithm
        self.learning_schedules = {
            AlgorithmType.DQN: LearningSchedule(
                algorithm=AlgorithmType.DQN,
                update_frequency=4,
                batch_size=64,
                priority=0.3,
                enabled=enable_q_learning
            ),
            AlgorithmType.PPO: LearningSchedule(
                algorithm=AlgorithmType.PPO,
                update_frequency=2048,
                batch_size=64,
                priority=0.3,
                enabled=enable_policy_gradient
            ),
            AlgorithmType.A2C: LearningSchedule(
                algorithm=AlgorithmType.A2C,
                update_frequency=32,
                batch_size=32,
                priority=0.2,
                enabled=enable_policy_gradient
            ),
            AlgorithmType.TD_LAMBDA: LearningSchedule(
                algorithm=AlgorithmType.TD_LAMBDA,
                update_frequency=1,
                batch_size=1,
                priority=0.2,
                enabled=enable_td_learning
            )
        }

        # Learning mode and state
        self.current_mode = LearningMode.BALANCED
        self.current_regime = "unknown"
        self.active_algorithm = AlgorithmType.HYBRID

        # Metrics tracking
        self.step_count = 0
        self.episode_count = 0
        self.learning_metrics = deque(maxlen=1000)
        self.algorithm_performance = defaultdict(lambda: deque(maxlen=100))

        # Resource management
        self.cpu_limit = 0.3  # 30% CPU limit for learning
        self.memory_limit = 2048  # 2GB memory limit
        self.update_queue = asyncio.Queue()

        logger.info(
            "Learning Orchestrator initialized",
            enabled_algorithms=[
                name for name, agent in self.agents.items()
            ],
            memory_path=memory_path,
            checkpoint_path=checkpoint_path
        )

    async def process_experience(
        self,
        agent_id: str,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None
    ):
        """
        Process new experience through all learning systems.

        Args:
            agent_id: Agent identifier
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
            info: Additional information
        """

        self.step_count += 1

        # Store in enhanced memory
        experience_id = self.memory.store_sarsa(
            agent_id=agent_id,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info
        )

        # Store in individual agent buffers
        for agent_name, agent in self.agents.items():
            if agent_name == 'dqn':
                agent.store_experience(state, action, reward, next_state, done)
            elif agent_name in ['ppo', 'a2c']:
                # Policy gradient agents need additional info
                if 'log_prob' in info and 'value' in info:
                    agent.store_transition(
                        state, action, reward,
                        info['log_prob'], info.get('value'), done
                    )

        # Update TD learner
        if hasattr(self, 'td_learner'):
            await self._update_td_learner(state, reward, next_state, done)

        # Detect anomalies
        if hasattr(self, 'anomaly_detector'):
            is_anomaly = await self._check_anomaly(state)
            if is_anomaly:
                await self._handle_anomaly(state, action, reward)

        # Check for learning updates
        await self._check_learning_updates()

        # Episode completion
        if done:
            self.episode_count += 1
            await self._on_episode_complete(agent_id)

    async def _update_td_learner(
        self,
        state: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Update TD learner with new experience"""

        if self.td_learner:
            td_error = self.td_learner.td_lambda_update(state, reward, next_state, done)

            # Use TD error to update priorities in memory
            if abs(td_error) > 0:
                self.memory.update_td_errors([self.step_count - 1], [td_error])

    async def _check_anomaly(self, state: np.ndarray) -> bool:
        """Check if current state is anomalous"""

        if self.anomaly_detector:
            anomaly_score = self.anomaly_detector.detect_anomaly(state.reshape(1, -1))
            return anomaly_score > 0.8

        return False

    async def _handle_anomaly(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float
    ):
        """Handle detected anomaly"""

        logger.warning(
            "Anomaly detected",
            state_summary=state[:5].tolist(),
            action=action,
            reward=reward
        )

        # Switch to safe mode
        self.current_mode = LearningMode.SAFE

        # Reduce exploration
        for agent in self.agents.values():
            if hasattr(agent, 'epsilon'):
                agent.epsilon = max(0.01, agent.epsilon * 0.5)

    async def _check_learning_updates(self):
        """Check and trigger learning updates based on schedules"""

        for algorithm, schedule in self.learning_schedules.items():
            if not schedule.enabled:
                continue

            if self.step_count % schedule.update_frequency == 0:
                await self.update_queue.put({
                    'algorithm': algorithm,
                    'batch_size': schedule.batch_size,
                    'priority': schedule.priority
                })

        # Process updates with resource constraints
        await self._process_update_queue()

    async def _process_update_queue(self):
        """Process pending updates with resource management"""

        # Check resource usage
        if not self._check_resources():
            return

        updates_processed = 0
        max_updates = 3  # Process at most 3 updates per step

        while not self.update_queue.empty() and updates_processed < max_updates:
            update = await self.update_queue.get()
            algorithm = update['algorithm']

            try:
                loss = await self._perform_update(algorithm, update['batch_size'])

                # Track performance
                self.algorithm_performance[algorithm].append(loss)
                updates_processed += 1

            except Exception as e:
                logger.error(f"Update failed for {algorithm}: {e}")

    async def _perform_update(
        self,
        algorithm: AlgorithmType,
        batch_size: int
    ) -> float:
        """Perform learning update for specific algorithm"""

        loss = 0

        if algorithm == AlgorithmType.DQN and 'dqn' in self.agents:
            # Sample batch from memory
            batch = self.memory.sample_batch(batch_size, prioritized=True)

            # Convert to DQN format and train
            if batch:
                loss = self.agents['dqn'].train_step() or 0

        elif algorithm == AlgorithmType.PPO and 'ppo' in self.agents:
            # PPO updates when trajectory buffer is full
            self.agents['ppo'].train()
            loss = np.mean(self.agents['ppo'].losses) if self.agents['ppo'].losses else 0

        elif algorithm == AlgorithmType.A2C and 'a2c' in self.agents:
            # A2C updates more frequently
            self.agents['a2c'].train()
            loss = np.mean(self.agents['a2c'].losses) if self.agents['a2c'].losses else 0

        elif algorithm == AlgorithmType.TD_LAMBDA and hasattr(self, 'td_learner'):
            # TD learning is updated continuously
            loss = np.mean(self.td_learner.td_errors) if self.td_learner.td_errors else 0

        return loss

    async def _on_episode_complete(self, agent_id: str):
        """Handle episode completion"""

        # Get episode statistics
        episode_stats = self.memory.get_episode_statistics(agent_id, last_n=10)

        # Update learning mode based on performance
        await self._adapt_learning_mode(episode_stats)

        # Trigger strategy evolution
        if hasattr(self, 'strategy_evolution') and self.episode_count % 10 == 0:
            await self._evolve_strategies(agent_id, episode_stats)

        # Update metrics
        self._update_metrics(episode_stats)

        # Checkpoint models periodically
        if self.episode_count % 100 == 0:
            await self.save_checkpoint()

    async def _adapt_learning_mode(self, episode_stats: Dict[str, Any]):
        """Adapt learning mode based on performance"""

        if not episode_stats:
            return

        success_rate = episode_stats.get('success_rate', 0)
        avg_reward = episode_stats.get('avg_reward', 0)

        # Determine new mode
        if success_rate > 0.8:
            # High success - exploit more
            self.current_mode = LearningMode.EXPLOITATION
        elif success_rate < 0.3:
            # Low success - explore more
            self.current_mode = LearningMode.EXPLORATION
        elif avg_reward < -10:
            # Poor performance - safe mode
            self.current_mode = LearningMode.SAFE
        else:
            # Normal performance - balanced
            self.current_mode = LearningMode.BALANCED

        # Adjust exploration parameters
        await self._adjust_exploration()

    async def _adjust_exploration(self):
        """Adjust exploration based on current mode"""

        exploration_rates = {
            LearningMode.EXPLORATION: 0.3,
            LearningMode.EXPLOITATION: 0.01,
            LearningMode.BALANCED: 0.1,
            LearningMode.SAFE: 0.05,
            LearningMode.AGGRESSIVE: 0.5
        }

        target_rate = exploration_rates[self.current_mode]

        # Adjust DQN epsilon
        if 'dqn' in self.agents:
            self.agents['dqn'].epsilon = target_rate

        # Adjust policy gradient temperature
        if 'ppo' in self.agents:
            self.agents['ppo'].temperature = target_rate * 2

    async def _evolve_strategies(
        self,
        agent_id: str,
        episode_stats: Dict[str, Any]
    ):
        """Evolve strategies based on performance"""

        if not self.strategy_evolution:
            return

        recent_performance = [{
            'reward': episode_stats.get('avg_reward', 0),
            'success_rate': episode_stats.get('success_rate', 0),
            'episode_length': episode_stats.get('avg_length', 0)
        }]

        # Get current parameters from best performing agent
        best_agent = self._get_best_agent()
        if best_agent:
            base_parameters = self._extract_agent_parameters(best_agent)

            # Evolve strategy
            new_variant = await self.strategy_evolution.evolve_strategy(
                agent_type=agent_id,
                base_parameters=base_parameters,
                recent_performance=recent_performance
            )

            # Apply evolved parameters
            if new_variant:
                await self._apply_evolved_parameters(new_variant)

    def _get_best_agent(self) -> Optional[Any]:
        """Get best performing agent"""

        best_agent = None
        best_performance = float('-inf')

        for name, agent in self.agents.items():
            if hasattr(agent, 'rewards') and agent.rewards:
                avg_reward = np.mean(agent.rewards)
                if avg_reward > best_performance:
                    best_performance = avg_reward
                    best_agent = agent

        return best_agent

    def _extract_agent_parameters(self, agent: Any) -> Dict[str, Any]:
        """Extract parameters from agent"""

        params = {}

        if hasattr(agent, 'config'):
            params = asdict(agent.config)
        elif hasattr(agent, 'get_params'):
            params = agent.get_params()

        return params

    async def _apply_evolved_parameters(self, variant: Dict[str, Any]):
        """Apply evolved parameters to agents"""

        # Apply to appropriate agent based on variant type
        if 'learning_rate' in variant:
            for agent in self.agents.values():
                if hasattr(agent, 'optimizer'):
                    for param_group in agent.optimizer.param_groups:
                        param_group['lr'] = variant['learning_rate']

    def _check_resources(self) -> bool:
        """Check if resources are available for learning"""

        # Simple resource check - can be enhanced with actual system monitoring
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent

        return cpu_percent < 80 and memory_percent < 90

    def _update_metrics(self, episode_stats: Dict[str, Any]):
        """Update learning metrics"""

        metrics = LearningMetrics(
            total_updates=self.step_count,
            avg_reward=episode_stats.get('avg_reward', 0),
            avg_loss=np.mean([
                np.mean(losses) for losses in self.algorithm_performance.values()
                if losses
            ]) if self.algorithm_performance else 0,
            exploration_rate=self.agents['dqn'].epsilon if 'dqn' in self.agents else 0,
            learning_rate=self.agents['dqn'].config.learning_rate if 'dqn' in self.agents else 0,
            success_rate=episode_stats.get('success_rate', 0),
            market_regime=self.current_regime,
            algorithm_performance={
                str(algo): float(np.mean(perfs)) if perfs else 0
                for algo, perfs in self.algorithm_performance.items()
            },
            resource_usage={
                'cpu': psutil.cpu_percent(),
                'memory': psutil.virtual_memory().percent
            } if 'psutil' in globals() else {},
            timestamp=datetime.now()
        )

        self.learning_metrics.append(metrics)

    async def make_trading_decision(
        self,
        market_data: Dict[str, Any],
        algorithm: Optional[AlgorithmType] = None
    ) -> Dict[str, Any]:
        """
        Make trading decision using learned policies.

        Args:
            market_data: Current market state
            algorithm: Specific algorithm to use (or hybrid)

        Returns:
            Trading decision with confidence and reasoning
        """

        # Detect market regime
        if hasattr(self, 'regime_detector'):
            self.current_regime = await self._detect_market_regime(market_data)

        # Select algorithm based on regime or use specified
        if algorithm is None:
            algorithm = self._select_algorithm(self.current_regime)

        # Get decisions from selected algorithm(s)
        if algorithm == AlgorithmType.HYBRID:
            # Ensemble decision from multiple algorithms
            decisions = []

            for agent_name, agent in self.agents.items():
                if hasattr(agent, 'make_trading_decision'):
                    decision = agent.make_trading_decision(market_data)
                    decisions.append(decision)

            # Combine decisions
            return self._ensemble_decision(decisions)

        else:
            # Single algorithm decision
            agent_map = {
                AlgorithmType.DQN: 'dqn',
                AlgorithmType.PPO: 'ppo',
                AlgorithmType.A2C: 'a2c'
            }

            agent_name = agent_map.get(algorithm)
            if agent_name and agent_name in self.agents:
                return self.agents[agent_name].make_trading_decision(market_data)

        # Fallback decision
        return {
            'action': 'HOLD',
            'confidence': 0.5,
            'reasoning': 'No trained model available'
        }

    async def _detect_market_regime(self, market_data: Dict[str, Any]) -> str:
        """Detect current market regime"""

        if self.regime_detector:
            # Extract features for regime detection
            features = np.array([
                market_data.get('price_change_24h', 0),
                market_data.get('volume_ratio', 1),
                market_data.get('volatility', 0),
                market_data.get('rsi', 50),
            ]).reshape(1, -1)

            regime = self.regime_detector.detect_regime(features)
            return regime

        return "unknown"

    def _select_algorithm(self, market_regime: str) -> AlgorithmType:
        """Select best algorithm for current market regime"""

        # Algorithm selection based on regime
        regime_algorithms = {
            'bull': AlgorithmType.PPO,  # Policy gradient for trending
            'bear': AlgorithmType.DQN,  # Q-learning for defensive
            'sideways': AlgorithmType.A2C,  # Actor-critic for ranging
            'volatile': AlgorithmType.DQN,  # Q-learning for safety
            'unknown': AlgorithmType.HYBRID  # Ensemble for uncertainty
        }

        return regime_algorithms.get(market_regime, AlgorithmType.HYBRID)

    def _ensemble_decision(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple decisions into ensemble"""

        if not decisions:
            return {'action': 'HOLD', 'confidence': 0}

        # Count action votes
        action_votes = defaultdict(float)
        total_confidence = 0

        for decision in decisions:
            action = decision.get('action', 'HOLD')
            confidence = decision.get('confidence', 0.5)
            action_votes[action] += confidence
            total_confidence += confidence

        # Select action with highest weighted votes
        best_action = max(action_votes.items(), key=lambda x: x[1])

        return {
            'action': best_action[0],
            'confidence': best_action[1] / max(total_confidence, 1),
            'ensemble_size': len(decisions),
            'algorithm': 'hybrid',
            'reasoning': f"Ensemble decision from {len(decisions)} models"
        }

    async def save_checkpoint(self):
        """Save all models and memory"""

        if not self.checkpoint_path:
            return

        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save agents
        for name, agent in self.agents.items():
            if hasattr(agent, 'save'):
                filepath = self.checkpoint_path / f"{name}_{timestamp}.pt"
                agent.save(str(filepath))

        # Save TD learner
        if hasattr(self, 'td_learner'):
            filepath = self.checkpoint_path / f"td_learner_{timestamp}.pt"
            self.td_learner.save(str(filepath))

        # Save memory
        if self.memory_path:
            filepath = self.memory_path / f"memory_{timestamp}.pkl"
            self.memory.save_memory(str(filepath))

        # Save orchestrator state
        state = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'current_mode': self.current_mode.value,
            'current_regime': self.current_regime,
            'learning_schedules': {
                k: asdict(v) for k, v in self.learning_schedules.items()
            }
        }

        filepath = self.checkpoint_path / f"orchestrator_{timestamp}.json"
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Checkpoint saved at {timestamp}")

    async def load_checkpoint(self, timestamp: str):
        """Load models and memory from checkpoint"""

        if not self.checkpoint_path:
            return

        # Load agents
        for name, agent in self.agents.items():
            filepath = self.checkpoint_path / f"{name}_{timestamp}.pt"
            if filepath.exists() and hasattr(agent, 'load'):
                agent.load(str(filepath))

        # Load TD learner
        if hasattr(self, 'td_learner'):
            filepath = self.checkpoint_path / f"td_learner_{timestamp}.pt"
            if filepath.exists():
                self.td_learner.load(str(filepath))

        # Load memory
        if self.memory_path:
            filepath = self.memory_path / f"memory_{timestamp}.pkl"
            if filepath.exists():
                self.memory.load_memory(str(filepath))

        # Load orchestrator state
        filepath = self.checkpoint_path / f"orchestrator_{timestamp}.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.step_count = state['step_count']
            self.episode_count = state['episode_count']
            self.current_mode = LearningMode(state['current_mode'])
            self.current_regime = state['current_regime']

        logger.info(f"Checkpoint loaded from {timestamp}")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""

        return {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'current_mode': self.current_mode.value,
            'current_regime': self.current_regime,
            'active_algorithms': list(self.agents.keys()),
            'memory_stats': self.memory.get_memory_metrics(),
            'recent_metrics': [
                asdict(m) for m in list(self.learning_metrics)[-10:]
            ] if self.learning_metrics else [],
            'algorithm_performance': {
                str(algo): float(np.mean(list(perfs)[-10:])) if perfs else 0
                for algo, perfs in self.algorithm_performance.items()
            }
        }