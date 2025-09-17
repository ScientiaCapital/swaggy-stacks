"""
Enhanced Agent Memory with SARSA Support

Extends the existing agent memory system with reinforcement learning capabilities:
- SARSA tuple storage for temporal difference learning
- N-step returns computation
- Importance sampling weights
- Trajectory management for episodic learning
"""

import json
import pickle
from collections import deque, namedtuple, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import numpy as np

import structlog
from app.ml.unsupervised.agent_memory import AgentMemory, Experience, ExperienceCluster

logger = structlog.get_logger()

# SARSA tuple for RL algorithms
SARSATuple = namedtuple('SARSATuple',
                        ['state', 'action', 'reward', 'next_state', 'next_action', 'done',
                         'info', 'timestamp'])

# N-step trajectory
Trajectory = namedtuple('Trajectory',
                        ['states', 'actions', 'rewards', 'next_states', 'dones',
                         'values', 'log_probs'])


@dataclass
class RLExperience(Experience):
    """Extended experience with RL-specific fields"""

    next_state: Optional[Dict[str, Any]] = None
    done: bool = False
    value_estimate: Optional[float] = None
    advantage: Optional[float] = None
    return_value: Optional[float] = None
    td_error: Optional[float] = None
    importance_weight: float = 1.0


@dataclass
class EpisodeBuffer:
    """Buffer for storing complete episodes"""

    episode_id: str
    agent_id: str
    sarsa_tuples: List[SARSATuple]
    total_reward: float
    episode_length: int
    success: bool
    start_time: datetime
    end_time: datetime
    metadata: Dict[str, Any]


class EnhancedAgentMemory(AgentMemory):
    """Enhanced memory system with RL capabilities"""

    def __init__(
        self,
        max_experiences: int = 100000,
        max_episodes: int = 1000,
        n_step: int = 5,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
        **kwargs
    ):
        """
        Initialize enhanced memory with RL parameters.

        Args:
            max_experiences: Maximum experiences to store
            max_episodes: Maximum episodes to store
            n_step: N-step returns lookahead
            gamma: Discount factor
            lambda_gae: GAE lambda parameter
            priority_alpha: Prioritization exponent
            priority_beta: Importance sampling exponent
        """
        super().__init__(max_experiences=max_experiences, **kwargs)

        # RL-specific parameters
        self.n_step = n_step
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta

        # SARSA storage
        self.sarsa_buffer = deque(maxlen=max_experiences)
        self.sarsa_by_agent: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )

        # Episode management
        self.episode_buffer = deque(maxlen=max_episodes)
        self.current_episodes: Dict[str, List[SARSATuple]] = {}

        # N-step buffers
        self.n_step_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=n_step)
        )

        # Priority replay components
        self.priorities = np.ones(max_experiences) * 1e-6
        self.priority_tree = None  # Sum tree for efficient sampling

        # Trajectory storage for policy gradient methods
        self.trajectories: deque[Trajectory] = deque(maxlen=100)

        # Value function estimates
        self.value_estimates: Dict[str, float] = {}

        logger.info(
            "Enhanced Agent Memory initialized",
            n_step=n_step,
            gamma=gamma,
            priority_alpha=priority_alpha
        )

    def store_sarsa(
        self,
        agent_id: str,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        next_action: Optional[Union[int, np.ndarray]] = None,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store SARSA tuple for RL learning.

        Args:
            agent_id: Agent identifier
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (for SARSA, optional for Q-learning)
            done: Episode termination flag
            info: Additional information

        Returns:
            Experience ID
        """

        # Create SARSA tuple
        sarsa_tuple = SARSATuple(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_action=next_action,
            done=done,
            info=info or {},
            timestamp=datetime.now()
        )

        # Add to buffers
        self.sarsa_buffer.append(sarsa_tuple)
        self.sarsa_by_agent[agent_id].append(sarsa_tuple)

        # Add to current episode
        if agent_id not in self.current_episodes:
            self.current_episodes[agent_id] = []
        self.current_episodes[agent_id].append(sarsa_tuple)

        # Update n-step buffer
        self.n_step_buffers[agent_id].append(sarsa_tuple)

        # Calculate priority based on TD error
        priority = self._calculate_priority(sarsa_tuple)
        self._update_priority(len(self.sarsa_buffer) - 1, priority)

        # End episode if done
        if done:
            self._finalize_episode(agent_id)

        # Also store as regular experience for clustering
        experience_id = self.store_experience(
            agent_id=agent_id,
            state={'array': state.tolist()},
            action={'value': action if isinstance(action, int) else action.tolist()},
            result={'next_state': next_state.tolist(), 'done': done},
            reward=reward,
            metadata=info
        )

        return experience_id

    def _calculate_priority(self, sarsa_tuple: SARSATuple) -> float:
        """Calculate priority for experience replay"""

        # Use absolute TD error as priority
        if sarsa_tuple.done:
            td_error = abs(sarsa_tuple.reward)  # Terminal state
        else:
            # Estimate TD error (would need value function for accurate calculation)
            # Using reward magnitude as proxy for now
            td_error = abs(sarsa_tuple.reward) + 0.1

        return (td_error + 1e-6) ** self.priority_alpha

    def _update_priority(self, idx: int, priority: float):
        """Update priority for experience at index"""
        if idx < len(self.priorities):
            self.priorities[idx] = priority

    def _finalize_episode(self, agent_id: str):
        """Finalize episode and compute returns"""

        if agent_id not in self.current_episodes:
            return

        episode_tuples = self.current_episodes[agent_id]
        if not episode_tuples:
            return

        # Calculate episode statistics
        total_reward = sum(t.reward for t in episode_tuples)
        episode_length = len(episode_tuples)
        success = total_reward > 0  # Simple success metric

        # Create episode buffer
        episode = EpisodeBuffer(
            episode_id=f"{agent_id}_{datetime.now().timestamp()}",
            agent_id=agent_id,
            sarsa_tuples=episode_tuples,
            total_reward=total_reward,
            episode_length=episode_length,
            success=success,
            start_time=episode_tuples[0].timestamp,
            end_time=episode_tuples[-1].timestamp,
            metadata={
                'final_state': episode_tuples[-1].next_state.tolist()
            }
        )

        self.episode_buffer.append(episode)

        # Compute n-step returns for the episode
        self._compute_episode_returns(episode_tuples)

        # Clear current episode
        del self.current_episodes[agent_id]
        self.n_step_buffers[agent_id].clear()

        logger.info(
            "Episode finalized",
            agent_id=agent_id,
            total_reward=total_reward,
            length=episode_length,
            success=success
        )

    def _compute_episode_returns(self, episode_tuples: List[SARSATuple]):
        """Compute n-step returns and GAE for episode"""

        returns = []
        advantages = []
        values = []

        # Calculate returns backward
        G = 0
        for t in reversed(episode_tuples):
            G = t.reward + self.gamma * G * (1 - t.done)
            returns.append(G)

        returns.reverse()

        # Calculate GAE if value estimates available
        if len(values) == len(episode_tuples):
            gae = 0
            for t in reversed(range(len(episode_tuples))):
                if t == len(episode_tuples) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]

                delta = (
                    episode_tuples[t].reward +
                    self.gamma * next_value * (1 - episode_tuples[t].done) -
                    values[t]
                )
                gae = delta + self.gamma * self.lambda_gae * gae * (1 - episode_tuples[t].done)
                advantages.append(gae)

            advantages.reverse()

        # Store computed values
        for i, (sarsa_tuple, G) in enumerate(zip(episode_tuples, returns)):
            state_key = str(sarsa_tuple.state.tobytes())
            self.value_estimates[state_key] = G

    def sample_batch(
        self,
        batch_size: int,
        agent_id: Optional[str] = None,
        prioritized: bool = True
    ) -> List[SARSATuple]:
        """
        Sample batch of experiences for training.

        Args:
            batch_size: Number of experiences to sample
            agent_id: Optional agent filter
            prioritized: Use prioritized replay

        Returns:
            List of SARSA tuples
        """

        # Select buffer
        if agent_id and agent_id in self.sarsa_by_agent:
            buffer = self.sarsa_by_agent[agent_id]
        else:
            buffer = self.sarsa_buffer

        if len(buffer) < batch_size:
            return list(buffer)

        if prioritized:
            # Prioritized sampling
            priorities = self.priorities[:len(buffer)]
            probabilities = priorities / priorities.sum()

            indices = np.random.choice(
                len(buffer),
                size=batch_size,
                p=probabilities,
                replace=False
            )

            # Calculate importance sampling weights
            weights = (len(buffer) * probabilities[indices]) ** (-self.priority_beta)
            weights /= weights.max()

            batch = [buffer[i] for i in indices]

            # Attach weights to batch
            for i, sarsa_tuple in enumerate(batch):
                sarsa_tuple.info['importance_weight'] = weights[i]

            return batch
        else:
            # Uniform sampling
            return random.sample(list(buffer), batch_size)

    def compute_n_step_return(
        self,
        sarsa_tuples: List[SARSATuple],
        n: Optional[int] = None
    ) -> float:
        """
        Compute n-step return from sequence of SARSA tuples.

        Args:
            sarsa_tuples: Sequence of experiences
            n: Number of steps (defaults to self.n_step)

        Returns:
            N-step return value
        """

        n = n or self.n_step
        n = min(n, len(sarsa_tuples))

        G = 0
        for i in range(n):
            G += (self.gamma ** i) * sarsa_tuples[i].reward
            if sarsa_tuples[i].done:
                break

        # Bootstrap from final state if not terminal
        if i < n - 1 and not sarsa_tuples[i].done:
            # Would need value function to estimate final state value
            # Using 0 for now
            pass

        return G

    def store_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        values: Optional[np.ndarray] = None,
        log_probs: Optional[np.ndarray] = None
    ):
        """
        Store complete trajectory for policy gradient methods.

        Args:
            states: State sequence
            actions: Action sequence
            rewards: Reward sequence
            next_states: Next state sequence
            dones: Done flags
            values: Value estimates
            log_probs: Action log probabilities
        """

        trajectory = Trajectory(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            values=values,
            log_probs=log_probs
        )

        self.trajectories.append(trajectory)

    def get_latest_trajectories(self, n: int = 10) -> List[Trajectory]:
        """Get latest n trajectories"""
        return list(self.trajectories)[-n:]

    def update_td_errors(self, indices: List[int], td_errors: np.ndarray):
        """
        Update TD errors for prioritized replay.

        Args:
            indices: Experience indices
            td_errors: New TD errors
        """

        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.priority_alpha
            self._update_priority(idx, priority)

    def get_episode_statistics(
        self,
        agent_id: Optional[str] = None,
        last_n: int = 100
    ) -> Dict[str, Any]:
        """
        Get episode statistics.

        Args:
            agent_id: Optional agent filter
            last_n: Number of recent episodes to consider

        Returns:
            Statistics dictionary
        """

        episodes = list(self.episode_buffer)[-last_n:]

        if agent_id:
            episodes = [e for e in episodes if e.agent_id == agent_id]

        if not episodes:
            return {}

        rewards = [e.total_reward for e in episodes]
        lengths = [e.episode_length for e in episodes]
        success_rate = sum(e.success for e in episodes) / len(episodes)

        return {
            'num_episodes': len(episodes),
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'avg_length': np.mean(lengths),
            'success_rate': success_rate,
            'total_experiences': len(self.sarsa_buffer)
        }

    def clear_agent_memory(self, agent_id: str):
        """Clear all memory for specific agent"""

        # Clear SARSA buffers
        if agent_id in self.sarsa_by_agent:
            del self.sarsa_by_agent[agent_id]

        if agent_id in self.current_episodes:
            del self.current_episodes[agent_id]

        if agent_id in self.n_step_buffers:
            del self.n_step_buffers[agent_id]

        # Clear from parent class
        if agent_id in self.experiences_by_agent:
            del self.experiences_by_agent[agent_id]

        logger.info(f"Cleared memory for agent {agent_id}")

    def save_memory(self, filepath: str):
        """Save memory to disk"""

        memory_state = {
            'sarsa_buffer': list(self.sarsa_buffer),
            'episode_buffer': list(self.episode_buffer),
            'value_estimates': self.value_estimates,
            'priorities': self.priorities.tolist(),
            'trajectories': list(self.trajectories),
            'parameters': {
                'n_step': self.n_step,
                'gamma': self.gamma,
                'lambda_gae': self.lambda_gae,
                'priority_alpha': self.priority_alpha,
                'priority_beta': self.priority_beta
            }
        }

        # Save parent class data
        memory_state['experiences'] = list(self.experiences)
        memory_state['clusters'] = self.experience_clusters
        memory_state['insights'] = self.learning_insights

        with open(filepath, 'wb') as f:
            pickle.dump(memory_state, f)

        logger.info(f"Memory saved to {filepath}")

    def load_memory(self, filepath: str):
        """Load memory from disk"""

        with open(filepath, 'rb') as f:
            memory_state = pickle.load(f)

        # Restore SARSA components
        self.sarsa_buffer = deque(memory_state['sarsa_buffer'], maxlen=self.max_experiences)
        self.episode_buffer = deque(memory_state['episode_buffer'], maxlen=1000)
        self.value_estimates = memory_state['value_estimates']
        self.priorities = np.array(memory_state['priorities'])
        self.trajectories = deque(memory_state['trajectories'], maxlen=100)

        # Restore parameters
        params = memory_state['parameters']
        self.n_step = params['n_step']
        self.gamma = params['gamma']
        self.lambda_gae = params['lambda_gae']
        self.priority_alpha = params['priority_alpha']
        self.priority_beta = params['priority_beta']

        # Restore parent class data
        self.experiences = deque(memory_state['experiences'], maxlen=self.max_experiences)
        self.experience_clusters = memory_state.get('clusters', {})
        self.learning_insights = memory_state.get('insights', {})

        logger.info(f"Memory loaded from {filepath}")

    def get_memory_metrics(self) -> Dict[str, Any]:
        """Get comprehensive memory metrics"""

        base_metrics = {
            'total_experiences': len(self.experiences),
            'total_sarsa_tuples': len(self.sarsa_buffer),
            'total_episodes': len(self.episode_buffer),
            'total_trajectories': len(self.trajectories),
            'active_episodes': len(self.current_episodes),
            'num_agents': len(self.sarsa_by_agent),
            'value_estimates': len(self.value_estimates)
        }

        # Add episode statistics
        if self.episode_buffer:
            episode_stats = self.get_episode_statistics()
            base_metrics.update(episode_stats)

        # Add clustering metrics from parent
        base_metrics['num_clusters'] = sum(
            len(clusters) for clusters in self.experience_clusters.values()
        )
        base_metrics['num_insights'] = sum(
            len(insights) for insights in self.learning_insights.values()
        )

        return base_metrics