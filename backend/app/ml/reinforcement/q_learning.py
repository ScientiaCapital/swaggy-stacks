"""
Deep Q-Learning Implementation for Trading System

Implements DQN, Double DQN, and Dueling DQN architectures with:
- Prioritized experience replay
- Target network for stability
- Multiple exploration strategies (epsilon-greedy, UCB, Boltzmann)
- Optimized for M1 MacBook with PyTorch Metal acceleration
"""

import json
import pickle
from collections import deque, namedtuple
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import structlog
from app.core.exceptions import TradingError

logger = structlog.get_logger()

# Check for M1 Mac Metal acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Experience tuple for replay buffer
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done', 'priority'])

@dataclass
class QLearningConfig:
    """Configuration for Q-Learning agent"""
    state_dim: int = 128  # Dimension of state representation
    action_dim: int = 5  # Number of discrete actions
    hidden_dim: int = 256  # Hidden layer dimension
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update parameter for target network

    # Replay buffer settings
    buffer_size: int = 100000
    batch_size: int = 64
    min_buffer_size: int = 1000  # Minimum experiences before training

    # Prioritized replay settings
    priority_alpha: float = 0.6  # How much prioritization to use (0 = uniform)
    priority_beta: float = 0.4  # Importance sampling weight (increases to 1)
    priority_epsilon: float = 0.01  # Small constant to ensure non-zero priorities

    # Exploration settings
    exploration_strategy: str = "epsilon_greedy"  # epsilon_greedy, ucb, boltzmann
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    temperature: float = 1.0  # For Boltzmann exploration
    ucb_c: float = 2.0  # UCB exploration constant

    # Double DQN and Dueling settings
    use_double_dqn: bool = True
    use_dueling_dqn: bool = True
    use_noisy_nets: bool = False  # Noisy networks for exploration

    # Training settings
    update_target_every: int = 100  # Steps between target network updates
    train_every: int = 4  # Steps between training updates
    gradient_clip: float = 1.0  # Gradient clipping value


class DQN(nn.Module):
    """Deep Q-Network with optional dueling architecture"""

    def __init__(self, config: QLearningConfig):
        super(DQN, self).__init__()
        self.config = config

        # Feature extraction layers
        self.fc1 = nn.Linear(config.state_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)

        if config.use_dueling_dqn:
            # Dueling DQN: separate value and advantage streams
            self.value_stream = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1)
            )

            self.advantage_stream = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, config.action_dim)
            )
        else:
            # Standard DQN: single output layer
            self.fc3 = nn.Linear(config.hidden_dim, config.action_dim)

        # Noisy networks for exploration
        if config.use_noisy_nets:
            self._add_noise_to_weights()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        if self.config.use_dueling_dqn:
            # Combine value and advantage streams
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            # Q(s,a) = V(s) + A(s,a) - mean(A(s,*))
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.fc3(x)

        return q_values

    def _add_noise_to_weights(self):
        """Add learnable noise parameters for noisy networks"""
        def noisy_linear(in_features, out_features):
            layer = nn.Linear(in_features, out_features)
            layer.noise_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.017)
            layer.noise_bias = nn.Parameter(torch.randn(out_features) * 0.017)
            return layer

        # Replace linear layers with noisy versions
        # Implementation details omitted for brevity


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta  # Importance sampling exponent
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def add(self, experience: Experience):
        """Add experience with maximum priority"""
        max_priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with prioritized replay"""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])

        # Calculate sampling probabilities
        priorities = np.array(self.priorities) ** self.alpha
        probabilities = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            min(batch_size, len(self.buffer)),
            p=probabilities,
            replace=False
        )

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]

        return experiences, weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small constant to ensure non-zero

    def __len__(self):
        return len(self.buffer)


class QLearningAgent:
    """Deep Q-Learning agent for trading decisions"""

    def __init__(self, config: QLearningConfig):
        self.config = config

        # Initialize Q-networks
        self.q_network = DQN(config).to(device)
        self.target_network = DQN(config).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            config.buffer_size,
            config.priority_alpha,
            config.priority_beta
        )

        # Exploration parameters
        self.epsilon = config.epsilon_start
        self.temperature = config.temperature
        self.action_counts = np.zeros(config.action_dim)  # For UCB
        self.total_steps = 0

        # Training metrics
        self.losses = deque(maxlen=1000)
        self.rewards = deque(maxlen=1000)

        logger.info(
            "Q-Learning agent initialized",
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            double_dqn=config.use_double_dqn,
            dueling_dqn=config.use_dueling_dqn
        )

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using configured exploration strategy"""

        if not training:
            # Greedy action selection during evaluation
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()

        if self.config.exploration_strategy == "epsilon_greedy":
            return self._epsilon_greedy_action(state)
        elif self.config.exploration_strategy == "ucb":
            return self._ucb_action(state)
        elif self.config.exploration_strategy == "boltzmann":
            return self._boltzmann_action(state)
        else:
            raise ValueError(f"Unknown exploration strategy: {self.config.exploration_strategy}")

    def _epsilon_greedy_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy exploration"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.config.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def _ucb_action(self, state: np.ndarray) -> int:
        """Upper Confidence Bound exploration"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor).squeeze().cpu().numpy()

        # Add UCB bonus
        ucb_values = q_values + self.config.ucb_c * np.sqrt(
            np.log(self.total_steps + 1) / (self.action_counts + 1)
        )

        action = ucb_values.argmax()
        self.action_counts[action] += 1
        return action

    def _boltzmann_action(self, state: np.ndarray) -> int:
        """Boltzmann (softmax) exploration"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor).squeeze()

            # Apply temperature scaling
            probabilities = F.softmax(q_values / self.temperature, dim=0)

            # Sample action from distribution
            action_dist = Categorical(probabilities)
            return action_dist.sample().item()

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""

        # Calculate TD error for prioritized replay
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

            current_q = self.q_network(state_tensor)[0, action]

            if self.config.use_double_dqn:
                # Double DQN: use online network to select action, target to evaluate
                next_action = self.q_network(next_state_tensor).argmax(dim=1)
                next_q = self.target_network(next_state_tensor)[0, next_action]
            else:
                # Standard DQN
                next_q = self.target_network(next_state_tensor).max(dim=1)[0]

            td_error = abs((reward + self.config.gamma * next_q * (1 - done)) - current_q)
            priority = td_error.item() + self.config.priority_epsilon

        experience = Experience(state, action, reward, next_state, done, priority)
        self.replay_buffer.add(experience)
        self.rewards.append(reward)

    def train_step(self) -> Optional[float]:
        """Perform one training step"""

        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None

        # Sample batch with prioritized replay
        experiences, weights, indices = self.replay_buffer.sample(self.config.batch_size)

        if not experiences:
            return None

        # Prepare batch tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(device)
        actions = torch.LongTensor([e.action for e in experiences]).to(device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(device)
        weights = torch.FloatTensor(weights).to(device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN target
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q_values = self.target_network(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze()
            else:
                # Standard DQN target
                next_q_values = self.target_network(next_states).max(dim=1)[0]

            target_q_values = rewards + self.config.gamma * next_q_values * (1 - dones)

        # Calculate loss with importance sampling weights
        td_errors = target_q_values - current_q_values
        loss = (weights * td_errors.pow(2)).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            self.config.gradient_clip
        )

        self.optimizer.step()

        # Update priorities in replay buffer
        new_priorities = td_errors.detach().abs().cpu().numpy()
        self.replay_buffer.update_priorities(indices, new_priorities)

        # Update exploration parameters
        self._update_exploration()

        # Update target network periodically
        if self.total_steps % self.config.update_target_every == 0:
            self._update_target_network()

        self.total_steps += 1
        self.losses.append(loss.item())

        return loss.item()

    def _update_target_network(self):
        """Soft update of target network"""
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data +
                (1 - self.config.tau) * target_param.data
            )

    def _update_exploration(self):
        """Update exploration parameters"""
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )

        # Decay temperature for Boltzmann exploration
        self.temperature = max(0.1, self.temperature * 0.999)

        # Increase importance sampling beta towards 1
        self.replay_buffer.beta = min(
            1.0,
            self.replay_buffer.beta + (1.0 - self.config.priority_beta) / 100000
        )

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions in a state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor).squeeze().cpu().numpy()
        return q_values

    def save(self, filepath: str):
        """Save agent state"""
        state = {
            'config': asdict(self.config),
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'temperature': self.temperature,
            'action_counts': self.action_counts,
            'total_steps': self.total_steps,
            'losses': list(self.losses),
            'rewards': list(self.rewards)
        }
        torch.save(state, filepath)
        logger.info(f"Q-Learning agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent state"""
        state = torch.load(filepath, map_location=device)

        self.config = QLearningConfig(**state['config'])
        self.q_network.load_state_dict(state['q_network_state'])
        self.target_network.load_state_dict(state['target_network_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.epsilon = state['epsilon']
        self.temperature = state['temperature']
        self.action_counts = state['action_counts']
        self.total_steps = state['total_steps']
        self.losses = deque(state['losses'], maxlen=1000)
        self.rewards = deque(state['rewards'], maxlen=1000)

        logger.info(f"Q-Learning agent loaded from {filepath}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        return {
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses) if self.losses else 0,
            'avg_reward': np.mean(self.rewards) if self.rewards else 0,
            'q_network_params': sum(p.numel() for p in self.q_network.parameters()),
            'exploration_strategy': self.config.exploration_strategy
        }


class TradingQLearningAgent(QLearningAgent):
    """Q-Learning agent specialized for trading decisions"""

    # Trading actions
    ACTIONS = {
        0: "STRONG_SELL",  # Sell with high confidence
        1: "SELL",         # Normal sell
        2: "HOLD",         # Do nothing
        3: "BUY",          # Normal buy
        4: "STRONG_BUY"    # Buy with high confidence
    }

    def __init__(self, config: QLearningConfig):
        super().__init__(config)
        self.position = 0  # Current position: -1 (short), 0 (neutral), 1 (long)
        self.entry_price = 0

    def create_state_representation(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Convert market data to state representation for Q-learning"""

        features = []

        # Price features
        features.extend([
            market_data.get('price_change_1h', 0),
            market_data.get('price_change_24h', 0),
            market_data.get('price_change_7d', 0),
            market_data.get('volume_ratio', 1),
            market_data.get('volatility', 0),
        ])

        # Technical indicators
        features.extend([
            market_data.get('rsi', 50) / 100,
            market_data.get('macd_signal', 0),
            (market_data.get('bb_position', 0) + 1) / 2,  # Normalize to [0,1]
        ])

        # Position information
        features.extend([
            self.position,
            market_data.get('unrealized_pnl', 0) / 100,  # Normalize PnL
            market_data.get('position_duration', 0) / 1440,  # Minutes to days
        ])

        # Market regime
        regime_encoding = {
            'bull': [1, 0, 0, 0],
            'bear': [0, 1, 0, 0],
            'sideways': [0, 0, 1, 0],
            'volatile': [0, 0, 0, 1]
        }
        features.extend(regime_encoding.get(market_data.get('market_regime', 'sideways')))

        # Pad or truncate to match state_dim
        if len(features) < self.config.state_dim:
            features.extend([0] * (self.config.state_dim - len(features)))
        else:
            features = features[:self.config.state_dim]

        return np.array(features, dtype=np.float32)

    def calculate_reward(self, action: int, market_data: Dict[str, Any],
                        next_market_data: Dict[str, Any]) -> float:
        """Calculate reward for trading action"""

        reward = 0
        price_change = (next_market_data['price'] - market_data['price']) / market_data['price']

        # Position-based rewards
        if self.position == 1:  # Long position
            reward += price_change * 100  # Scale for meaningful gradients
        elif self.position == -1:  # Short position
            reward -= price_change * 100

        # Action-based rewards/penalties
        action_name = self.ACTIONS[action]

        if action_name in ["BUY", "STRONG_BUY"] and self.position < 1:
            # Reward for taking position when not already long
            self.position = 1
            self.entry_price = market_data['price']
            reward -= 0.1  # Transaction cost

        elif action_name in ["SELL", "STRONG_SELL"] and self.position > -1:
            # Reward for taking position when not already short
            if self.position == 1:
                # Close long position
                profit = (market_data['price'] - self.entry_price) / self.entry_price
                reward += profit * 100
            self.position = -1
            self.entry_price = market_data['price']
            reward -= 0.1  # Transaction cost

        elif action_name == "HOLD":
            # Small penalty for inaction to encourage decisiveness
            reward -= 0.01

        # Risk-adjusted rewards
        if market_data.get('volatility', 0) > 0.03:  # High volatility
            reward *= 0.8  # Reduce rewards in high-risk environments

        # Clip rewards to prevent instability
        reward = np.clip(reward, -10, 10)

        return reward

    def make_trading_decision(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make trading decision based on Q-values"""

        state = self.create_state_representation(market_data)
        action = self.select_action(state, training=False)
        q_values = self.get_q_values(state)

        # Calculate confidence based on Q-value distribution
        q_softmax = np.exp(q_values) / np.sum(np.exp(q_values))
        confidence = float(np.max(q_softmax))

        return {
            'action': self.ACTIONS[action],
            'action_index': action,
            'confidence': confidence,
            'q_values': q_values.tolist(),
            'position': self.position,
            'reasoning': f"Q-value for {self.ACTIONS[action]}: {q_values[action]:.3f}"
        }