"""
Policy Gradient Methods for Trading System

Implements REINFORCE, Actor-Critic (A2C), and Proximal Policy Optimization (PPO)
with advanced features for stable and efficient learning in trading environments.

Optimized for M1 MacBook with PyTorch Metal acceleration.
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
from torch.distributions import Categorical, Normal

import structlog
from app.core.exceptions import TradingError

logger = structlog.get_logger()

# Check for M1 Mac Metal acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Policy Gradient using device: {device}")

# Trajectory for policy gradient methods
Trajectory = namedtuple('Trajectory',
                        ['states', 'actions', 'rewards', 'log_probs', 'values', 'dones'])


@dataclass
class PolicyGradientConfig:
    """Configuration for Policy Gradient agents"""
    state_dim: int = 128
    action_dim: int = 5  # Discrete actions for classification
    hidden_dim: int = 256
    learning_rate_actor: float = 0.0003
    learning_rate_critic: float = 0.001
    gamma: float = 0.99
    gae_lambda: float = 0.95  # GAE parameter

    # Algorithm selection
    algorithm: str = "ppo"  # "reinforce", "a2c", "ppo"

    # PPO specific
    ppo_epochs: int = 10
    ppo_clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Training settings
    batch_size: int = 64
    trajectory_length: int = 2048  # Steps before update
    normalize_advantages: bool = True
    use_gae: bool = True  # Generalized Advantage Estimation

    # Continuous action space settings
    continuous_actions: bool = False
    action_std_init: float = 0.5
    action_std_min: float = 0.1
    action_std_decay: float = 0.99995


class PolicyNetwork(nn.Module):
    """Policy network for action selection"""

    def __init__(self, config: PolicyGradientConfig):
        super(PolicyNetwork, self).__init__()
        self.config = config

        # Shared layers
        self.fc1 = nn.Linear(config.state_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)

        if config.continuous_actions:
            # Continuous action space: output mean and std
            self.mean_head = nn.Linear(config.hidden_dim, config.action_dim)
            self.log_std = nn.Parameter(
                torch.ones(1, config.action_dim) * np.log(config.action_std_init)
            )
        else:
            # Discrete action space: output logits
            self.action_head = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(self, state: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through policy network"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        if self.config.continuous_actions:
            mean = self.mean_head(x)
            std = self.log_std.exp().expand_as(mean)
            return mean, std
        else:
            return self.action_head(x)


class ValueNetwork(nn.Module):
    """Value network for state value estimation"""

    def __init__(self, config: PolicyGradientConfig):
        super(ValueNetwork, self).__init__()

        self.fc1 = nn.Linear(config.state_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through value network"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value_head(x)


class ActorCriticNetwork(nn.Module):
    """Combined Actor-Critic network with shared backbone"""

    def __init__(self, config: PolicyGradientConfig):
        super(ActorCriticNetwork, self).__init__()
        self.config = config

        # Shared backbone
        self.shared_fc1 = nn.Linear(config.state_dim, config.hidden_dim)
        self.shared_fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Actor head
        if config.continuous_actions:
            self.actor_mean = nn.Linear(config.hidden_dim, config.action_dim)
            self.actor_log_std = nn.Parameter(
                torch.ones(1, config.action_dim) * np.log(config.action_std_init)
            )
        else:
            self.actor_head = nn.Linear(config.hidden_dim, config.action_dim)

        # Critic head
        self.critic_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple:
        """Forward pass returning both policy and value"""
        # Shared layers
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))

        # Actor output
        if self.config.continuous_actions:
            mean = self.actor_mean(x)
            std = self.actor_log_std.exp().expand_as(mean)
            policy_output = (mean, std)
        else:
            policy_output = self.actor_head(x)

        # Critic output
        value = self.critic_head(x)

        return policy_output, value


class PolicyGradientAgent:
    """Base class for policy gradient agents"""

    def __init__(self, config: PolicyGradientConfig):
        self.config = config

        # Initialize networks based on algorithm
        if config.algorithm == "reinforce":
            self.policy_net = PolicyNetwork(config).to(device)
            self.policy_optimizer = optim.Adam(
                self.policy_net.parameters(),
                lr=config.learning_rate_actor
            )
            self.value_net = None
        elif config.algorithm in ["a2c", "ppo"]:
            self.actor_critic = ActorCriticNetwork(config).to(device)
            self.optimizer = optim.Adam(
                self.actor_critic.parameters(),
                lr=config.learning_rate_actor
            )
            if config.algorithm == "ppo":
                # PPO uses old policy for importance sampling
                self.old_actor_critic = ActorCriticNetwork(config).to(device)
                self.old_actor_critic.load_state_dict(self.actor_critic.state_dict())

        # Trajectory buffer
        self.trajectory_buffer = []
        self.current_trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }

        # Training metrics
        self.episode_rewards = deque(maxlen=100)
        self.losses = deque(maxlen=1000)
        self.entropy_values = deque(maxlen=1000)

        logger.info(
            "Policy Gradient agent initialized",
            algorithm=config.algorithm,
            state_dim=config.state_dim,
            action_dim=config.action_dim
        )

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, Optional[float]]:
        """Select action using current policy"""

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad() if not training else torch.enable_grad():
            if self.config.algorithm == "reinforce":
                if self.config.continuous_actions:
                    mean, std = self.policy_net(state_tensor)
                    dist = Normal(mean, std)
                else:
                    logits = self.policy_net(state_tensor)
                    dist = Categorical(logits=logits)

                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
                value = None

            else:  # a2c or ppo
                policy_output, value = self.actor_critic(state_tensor)

                if self.config.continuous_actions:
                    mean, std = policy_output
                    dist = Normal(mean, std)
                else:
                    dist = Categorical(logits=policy_output)

                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
                value = value.squeeze()

        if self.config.continuous_actions:
            return action.squeeze().cpu().numpy(), log_prob.item(), value.item() if value is not None else None
        else:
            return action.item(), log_prob.item(), value.item() if value is not None else None

    def store_transition(self, state: np.ndarray, action: Union[int, np.ndarray],
                        reward: float, log_prob: float, value: Optional[float], done: bool):
        """Store transition in trajectory buffer"""

        self.current_trajectory['states'].append(state)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['rewards'].append(reward)
        self.current_trajectory['log_probs'].append(log_prob)
        self.current_trajectory['values'].append(value if value is not None else 0)
        self.current_trajectory['dones'].append(done)

        # Check if trajectory is complete
        if len(self.current_trajectory['states']) >= self.config.trajectory_length or done:
            self.trajectory_buffer.append(Trajectory(**{
                k: torch.FloatTensor(v) if k != 'actions' else torch.LongTensor(v)
                for k, v in self.current_trajectory.items()
            }))

            # Reset current trajectory
            self.current_trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'log_probs': [],
                'values': [],
                'dones': []
            }

            # Trigger training if buffer is full
            if len(self.trajectory_buffer) >= self.config.batch_size:
                self.train()

    def compute_returns_and_advantages(self, trajectory: Trajectory) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute discounted returns and advantages using GAE"""

        rewards = trajectory.rewards
        values = trajectory.values
        dones = trajectory.dones

        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        if self.config.use_gae and values is not None:
            # Generalized Advantage Estimation
            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0  # Bootstrap from learned value
                else:
                    next_value = values[t + 1]

                delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = gae + values[t]
        else:
            # Simple discounted returns
            running_return = 0
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + self.config.gamma * running_return * (1 - dones[t])
                returns[t] = running_return

            if values is not None:
                advantages = returns - values

        if self.config.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns.to(device), advantages.to(device)

    def train(self):
        """Train the policy based on collected trajectories"""

        if not self.trajectory_buffer:
            return

        if self.config.algorithm == "reinforce":
            self._train_reinforce()
        elif self.config.algorithm == "a2c":
            self._train_a2c()
        elif self.config.algorithm == "ppo":
            self._train_ppo()

        # Clear trajectory buffer after training
        self.trajectory_buffer.clear()

    def _train_reinforce(self):
        """REINFORCE algorithm training"""

        policy_losses = []

        for trajectory in self.trajectory_buffer:
            states = trajectory.states.to(device)
            actions = trajectory.actions.to(device)
            returns, _ = self.compute_returns_and_advantages(trajectory)

            # Recalculate log probabilities
            if self.config.continuous_actions:
                mean, std = self.policy_net(states)
                dist = Normal(mean, std)
            else:
                logits = self.policy_net(states)
                dist = Categorical(logits=logits)

            log_probs = dist.log_prob(actions).sum(dim=-1)

            # Policy gradient loss
            policy_loss = -(log_probs * returns).mean()
            policy_losses.append(policy_loss)

        # Optimize
        total_loss = torch.stack(policy_losses).mean()
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
        self.policy_optimizer.step()

        self.losses.append(total_loss.item())

    def _train_a2c(self):
        """Advantage Actor-Critic training"""

        policy_losses = []
        value_losses = []
        entropy_losses = []

        for trajectory in self.trajectory_buffer:
            states = trajectory.states.to(device)
            actions = trajectory.actions.to(device)
            returns, advantages = self.compute_returns_and_advantages(trajectory)

            # Forward pass
            policy_output, values = self.actor_critic(states)
            values = values.squeeze()

            # Calculate policy loss
            if self.config.continuous_actions:
                mean, std = policy_output
                dist = Normal(mean, std)
            else:
                dist = Categorical(logits=policy_output)

            log_probs = dist.log_prob(actions).sum(dim=-1)
            policy_loss = -(log_probs * advantages.detach()).mean()

            # Calculate value loss
            value_loss = F.mse_loss(values, returns.detach())

            # Calculate entropy bonus for exploration
            entropy = dist.entropy().mean()

            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropy_losses.append(entropy)

        # Total loss
        total_policy_loss = torch.stack(policy_losses).mean()
        total_value_loss = torch.stack(value_losses).mean()
        total_entropy = torch.stack(entropy_losses).mean()

        total_loss = (
            total_policy_loss +
            self.config.value_loss_coef * total_value_loss -
            self.config.entropy_coef * total_entropy
        )

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.losses.append(total_loss.item())
        self.entropy_values.append(total_entropy.item())

    def _train_ppo(self):
        """Proximal Policy Optimization training"""

        # Prepare data from all trajectories
        all_states = []
        all_actions = []
        all_returns = []
        all_advantages = []
        all_old_log_probs = []

        for trajectory in self.trajectory_buffer:
            returns, advantages = self.compute_returns_and_advantages(trajectory)
            all_states.append(trajectory.states)
            all_actions.append(trajectory.actions)
            all_returns.append(returns)
            all_advantages.append(advantages)
            all_old_log_probs.append(trajectory.log_probs)

        # Concatenate all data
        states = torch.cat(all_states).to(device)
        actions = torch.cat(all_actions).to(device)
        returns = torch.cat(all_returns)
        advantages = torch.cat(all_advantages)
        old_log_probs = torch.cat(all_old_log_probs).to(device)

        # PPO update for multiple epochs
        for _ in range(self.config.ppo_epochs):
            # Random sampling for mini-batch updates
            indices = torch.randperm(len(states))

            for start_idx in range(0, len(states), self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # Forward pass
                policy_output, values = self.actor_critic(batch_states)
                values = values.squeeze()

                # Calculate current log probabilities
                if self.config.continuous_actions:
                    mean, std = policy_output
                    dist = Normal(mean, std)
                else:
                    dist = Categorical(logits=policy_output)

                log_probs = dist.log_prob(batch_actions).sum(dim=-1)

                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.config.ppo_clip_ratio, 1 + self.config.ppo_clip_ratio)
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()

                # Value loss (clipped)
                value_pred_clipped = batch_returns + torch.clamp(
                    values - batch_returns,
                    -self.config.ppo_clip_ratio,
                    self.config.ppo_clip_ratio
                )
                value_loss = torch.max(
                    F.mse_loss(values, batch_returns),
                    F.mse_loss(value_pred_clipped, batch_returns)
                )

                # Entropy bonus
                entropy = dist.entropy().mean()

                # Total loss
                total_loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss -
                    self.config.entropy_coef * entropy
                )

                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                self.losses.append(total_loss.item())
                self.entropy_values.append(entropy.item())

        # Update old policy for next PPO iteration
        if self.config.algorithm == "ppo":
            self.old_actor_critic.load_state_dict(self.actor_critic.state_dict())

    def save(self, filepath: str):
        """Save agent state"""
        state = {
            'config': asdict(self.config),
            'episode_rewards': list(self.episode_rewards),
            'losses': list(self.losses),
            'entropy_values': list(self.entropy_values)
        }

        if self.config.algorithm == "reinforce":
            state['policy_net_state'] = self.policy_net.state_dict()
            state['policy_optimizer_state'] = self.policy_optimizer.state_dict()
        else:
            state['actor_critic_state'] = self.actor_critic.state_dict()
            state['optimizer_state'] = self.optimizer.state_dict()
            if self.config.algorithm == "ppo":
                state['old_actor_critic_state'] = self.old_actor_critic.state_dict()

        torch.save(state, filepath)
        logger.info(f"Policy Gradient agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent state"""
        state = torch.load(filepath, map_location=device)
        self.config = PolicyGradientConfig(**state['config'])

        if self.config.algorithm == "reinforce":
            self.policy_net.load_state_dict(state['policy_net_state'])
            self.policy_optimizer.load_state_dict(state['policy_optimizer_state'])
        else:
            self.actor_critic.load_state_dict(state['actor_critic_state'])
            self.optimizer.load_state_dict(state['optimizer_state'])
            if self.config.algorithm == "ppo" and 'old_actor_critic_state' in state:
                self.old_actor_critic.load_state_dict(state['old_actor_critic_state'])

        self.episode_rewards = deque(state['episode_rewards'], maxlen=100)
        self.losses = deque(state['losses'], maxlen=1000)
        self.entropy_values = deque(state.get('entropy_values', []), maxlen=1000)

        logger.info(f"Policy Gradient agent loaded from {filepath}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        return {
            'algorithm': self.config.algorithm,
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'avg_loss': np.mean(self.losses) if self.losses else 0,
            'avg_entropy': np.mean(self.entropy_values) if self.entropy_values else 0,
            'trajectory_buffer_size': len(self.trajectory_buffer),
            'total_parameters': sum(
                p.numel() for p in (
                    self.policy_net.parameters() if self.config.algorithm == "reinforce"
                    else self.actor_critic.parameters()
                )
            )
        }


class TradingPolicyGradientAgent(PolicyGradientAgent):
    """Policy Gradient agent specialized for trading"""

    # Trading actions (same as Q-learning for consistency)
    ACTIONS = {
        0: "STRONG_SELL",
        1: "SELL",
        2: "HOLD",
        3: "BUY",
        4: "STRONG_BUY"
    }

    def __init__(self, config: PolicyGradientConfig):
        super().__init__(config)
        self.position = 0  # Current position
        self.entry_price = 0
        self.episode_pnl = 0

    def create_state_representation(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Convert market data to state representation"""

        features = []

        # Price and volume features
        features.extend([
            market_data.get('price_change_1h', 0),
            market_data.get('price_change_24h', 0),
            market_data.get('volume_ratio', 1),
            market_data.get('volatility', 0),
            market_data.get('bid_ask_spread', 0),
        ])

        # Technical indicators
        features.extend([
            market_data.get('rsi', 50) / 100,
            market_data.get('macd_signal', 0),
            (market_data.get('bb_position', 0) + 1) / 2,
            market_data.get('momentum', 0),
            market_data.get('vwap_deviation', 0),
        ])

        # Position and P&L information
        features.extend([
            self.position,
            market_data.get('unrealized_pnl', 0) / 100,
            market_data.get('position_duration', 0) / 1440,
            self.episode_pnl / 100,
        ])

        # Market microstructure
        features.extend([
            market_data.get('order_flow_imbalance', 0),
            market_data.get('trade_intensity', 0),
        ])

        # Pad to state_dim
        if len(features) < self.config.state_dim:
            features.extend([0] * (self.config.state_dim - len(features)))
        else:
            features = features[:self.config.state_dim]

        return np.array(features, dtype=np.float32)

    def calculate_reward(self, action: int, market_data: Dict[str, Any],
                        next_market_data: Dict[str, Any]) -> float:
        """Calculate reward with risk-adjusted returns"""

        reward = 0
        price_change = (next_market_data['price'] - market_data['price']) / market_data['price']

        # Position-based rewards
        if self.position == 1:  # Long
            pnl = price_change * 100
            reward += pnl
            self.episode_pnl += pnl
        elif self.position == -1:  # Short
            pnl = -price_change * 100
            reward += pnl
            self.episode_pnl += pnl

        # Action costs and bonuses
        action_name = self.ACTIONS[action]

        if action_name in ["BUY", "STRONG_BUY"] and self.position != 1:
            self.position = 1
            self.entry_price = market_data['price']
            reward -= 0.1  # Transaction cost

        elif action_name in ["SELL", "STRONG_SELL"] and self.position != -1:
            if self.position == 1:  # Closing long
                profit = (market_data['price'] - self.entry_price) / self.entry_price * 100
                reward += profit * 0.5  # Realize profit bonus
            self.position = -1
            self.entry_price = market_data['price']
            reward -= 0.1  # Transaction cost

        elif action_name == "HOLD" and self.position == 0:
            reward -= 0.02  # Small penalty for inaction when flat

        # Risk adjustment
        sharpe_adjustment = reward / (market_data.get('volatility', 0.01) + 0.01)
        reward = sharpe_adjustment * 0.1  # Scale down for stable learning

        return np.clip(reward, -10, 10)

    def make_trading_decision(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make trading decision using policy"""

        state = self.create_state_representation(market_data)
        action, log_prob, value = self.select_action(state, training=False)

        # Get action probabilities for confidence
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            if self.config.algorithm == "reinforce":
                logits = self.policy_net(state_tensor)
            else:
                policy_output, _ = self.actor_critic(state_tensor)
                logits = policy_output

            action_probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

        confidence = float(action_probs[action])

        return {
            'action': self.ACTIONS[action],
            'action_index': action,
            'confidence': confidence,
            'action_probabilities': action_probs.tolist(),
            'value_estimate': value if value is not None else 0,
            'position': self.position,
            'episode_pnl': self.episode_pnl,
            'reasoning': f"Policy selected {self.ACTIONS[action]} with {confidence:.1%} confidence"
        }