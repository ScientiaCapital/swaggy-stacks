"""
Value Functions and Temporal Difference Learning

Implements various value function approximation methods including:
- TD(0), TD(λ) with eligibility traces
- SARSA and Expected SARSA
- Monte Carlo value estimation
- Bellman equation solvers
- N-step returns

Optimized for trading environments with continuous state spaces.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Union
import structlog

logger = structlog.get_logger()

# Check for M1 Mac Metal acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class ValueFunctionConfig:
    """Configuration for value function learning"""
    state_dim: int = 128
    hidden_dim: int = 256
    learning_rate: float = 0.001

    # TD Learning parameters
    gamma: float = 0.99  # Discount factor
    lambda_param: float = 0.95  # TD(λ) trace decay
    n_step: int = 5  # N-step returns

    # Algorithm selection
    method: str = "td_lambda"  # "td0", "td_lambda", "sarsa", "expected_sarsa", "monte_carlo"

    # Eligibility traces
    use_eligibility_traces: bool = True
    trace_decay: float = 0.95
    trace_max: float = 1.0

    # Function approximation
    use_neural_network: bool = True
    use_tile_coding: bool = False  # Alternative to neural networks
    num_tilings: int = 8
    tile_size: int = 8

    # Training settings
    batch_size: int = 32
    update_frequency: int = 1
    target_update_frequency: int = 100


class ValueFunction(nn.Module):
    """Neural network for value function approximation"""

    def __init__(self, config: ValueFunctionConfig):
        super(ValueFunction, self).__init__()
        self.config = config

        self.fc1 = nn.Linear(config.state_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc3 = nn.Linear(config.hidden_dim, 1)

        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(config.hidden_dim)
        self.bn2 = nn.BatchNorm1d(config.hidden_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through value network"""
        # Handle both batched and single states
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = F.relu(self.bn1(self.fc1(state)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        value = self.fc3(x)

        return value.squeeze()


class ActionValueFunction(nn.Module):
    """Neural network for action-value (Q) function approximation"""

    def __init__(self, config: ValueFunctionConfig, action_dim: int):
        super(ActionValueFunction, self).__init__()
        self.config = config
        self.action_dim = action_dim

        # State processing
        self.state_fc = nn.Linear(config.state_dim, config.hidden_dim)

        # Action processing (for continuous actions)
        self.action_fc = nn.Linear(action_dim, config.hidden_dim)

        # Combined processing
        self.fc1 = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.q_value = nn.Linear(config.hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass for Q(s,a)"""
        state_features = F.relu(self.state_fc(state))
        action_features = F.relu(self.action_fc(action))

        combined = torch.cat([state_features, action_features], dim=-1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        q = self.q_value(x)

        return q.squeeze()


class TileCoding:
    """Tile coding for linear function approximation"""

    def __init__(self, state_dim: int, num_tilings: int = 8, tile_size: int = 8):
        self.state_dim = state_dim
        self.num_tilings = num_tilings
        self.tile_size = tile_size

        # Initialize tile offsets for each tiling
        self.offsets = np.random.uniform(0, tile_size, (num_tilings, state_dim))

        # Feature weights
        self.weights = defaultdict(float)

    def get_features(self, state: np.ndarray) -> List[Tuple[int, ...]]:
        """Get active tile indices for a state"""
        features = []

        for tiling_idx in range(self.num_tilings):
            # Apply offset and discretize
            offset_state = state + self.offsets[tiling_idx]
            tile_indices = tuple((offset_state / self.tile_size).astype(int))
            features.append((tiling_idx,) + tile_indices)

        return features

    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for a state"""
        features = self.get_features(state)
        return sum(self.weights[f] for f in features) / self.num_tilings

    def update(self, state: np.ndarray, target: float, learning_rate: float):
        """Update tile coding weights"""
        features = self.get_features(state)
        current_value = self.get_value(state)
        error = target - current_value

        for feature in features:
            self.weights[feature] += learning_rate * error / self.num_tilings


class EligibilityTraces:
    """Manages eligibility traces for TD(λ) learning"""

    def __init__(self, trace_decay: float = 0.95, trace_max: float = 1.0):
        self.trace_decay = trace_decay
        self.trace_max = trace_max
        self.traces = {}

    def update(self, state_key: Any, value: float = 1.0):
        """Update trace for a state"""
        if state_key not in self.traces:
            self.traces[state_key] = 0
        self.traces[state_key] = min(self.traces[state_key] + value, self.trace_max)

    def decay(self, gamma: float):
        """Decay all traces"""
        for key in list(self.traces.keys()):
            self.traces[key] *= gamma * self.trace_decay
            if self.traces[key] < 1e-4:
                del self.traces[key]

    def get(self, state_key: Any) -> float:
        """Get trace value for a state"""
        return self.traces.get(state_key, 0.0)

    def reset(self):
        """Reset all traces"""
        self.traces.clear()


class TemporalDifferenceLearner:
    """Temporal Difference learning algorithms"""

    def __init__(self, config: ValueFunctionConfig):
        self.config = config

        # Initialize value function
        if config.use_neural_network:
            self.value_function = ValueFunction(config).to(device)
            self.optimizer = optim.Adam(
                self.value_function.parameters(),
                lr=config.learning_rate
            )

            # Target network for stability
            self.target_value_function = ValueFunction(config).to(device)
            self.target_value_function.load_state_dict(self.value_function.state_dict())
        else:
            self.value_function = TileCoding(
                config.state_dim,
                config.num_tilings,
                config.tile_size
            )

        # Eligibility traces
        if config.use_eligibility_traces:
            self.eligibility_traces = EligibilityTraces(
                config.trace_decay,
                config.trace_max
            )

        # Experience buffer for n-step returns
        self.n_step_buffer = deque(maxlen=config.n_step)

        # Metrics
        self.td_errors = deque(maxlen=1000)
        self.value_estimates = deque(maxlen=1000)

        logger.info(
            "TD Learner initialized",
            method=config.method,
            use_neural_network=config.use_neural_network,
            use_eligibility_traces=config.use_eligibility_traces
        )

    def td0_update(self, state: np.ndarray, reward: float, next_state: np.ndarray,
                   done: bool) -> float:
        """TD(0) update - single step temporal difference"""

        if self.config.use_neural_network:
            return self._td0_neural_update(state, reward, next_state, done)
        else:
            return self._td0_linear_update(state, reward, next_state, done)

    def _td0_neural_update(self, state: np.ndarray, reward: float,
                           next_state: np.ndarray, done: bool) -> float:
        """TD(0) with neural network"""

        state_tensor = torch.FloatTensor(state).to(device)
        next_state_tensor = torch.FloatTensor(next_state).to(device)

        # Current value estimate
        value = self.value_function(state_tensor)

        # Target value (using target network)
        with torch.no_grad():
            next_value = self.target_value_function(next_state_tensor) if not done else 0
            target = reward + self.config.gamma * next_value

        # TD error
        td_error = target - value

        # Update
        loss = td_error.pow(2)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), 1.0)
        self.optimizer.step()

        self.td_errors.append(td_error.item())
        self.value_estimates.append(value.item())

        return td_error.item()

    def _td0_linear_update(self, state: np.ndarray, reward: float,
                           next_state: np.ndarray, done: bool) -> float:
        """TD(0) with tile coding"""

        current_value = self.value_function.get_value(state)
        next_value = 0 if done else self.value_function.get_value(next_state)

        target = reward + self.config.gamma * next_value
        td_error = target - current_value

        self.value_function.update(state, target, self.config.learning_rate)

        self.td_errors.append(td_error)
        self.value_estimates.append(current_value)

        return td_error

    def td_lambda_update(self, state: np.ndarray, reward: float,
                         next_state: np.ndarray, done: bool) -> float:
        """TD(λ) update with eligibility traces"""

        if not self.config.use_eligibility_traces:
            return self.td0_update(state, reward, next_state, done)

        # Compute TD error
        if self.config.use_neural_network:
            state_tensor = torch.FloatTensor(state).to(device)
            next_state_tensor = torch.FloatTensor(next_state).to(device)

            with torch.no_grad():
                value = self.value_function(state_tensor).item()
                next_value = 0 if done else self.value_function(next_state_tensor).item()
        else:
            value = self.value_function.get_value(state)
            next_value = 0 if done else self.value_function.get_value(next_state)

        target = reward + self.config.gamma * next_value
        td_error = target - value

        # Update eligibility trace for current state
        state_key = tuple(state) if isinstance(state, np.ndarray) else state
        self.eligibility_traces.update(state_key)

        # Update all states according to their traces
        if self.config.use_neural_network:
            # Accumulate gradients weighted by eligibility traces
            accumulated_loss = 0

            for trace_state_key, trace_value in self.eligibility_traces.traces.items():
                trace_state = np.array(trace_state_key)
                trace_state_tensor = torch.FloatTensor(trace_state).to(device)

                trace_value_estimate = self.value_function(trace_state_tensor)
                trace_target = trace_value_estimate.detach() + td_error * trace_value

                loss = F.mse_loss(trace_value_estimate, trace_target)
                accumulated_loss += loss

            if accumulated_loss > 0:
                self.optimizer.zero_grad()
                accumulated_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), 1.0)
                self.optimizer.step()
        else:
            # Update tile coding weights
            for trace_state_key, trace_value in self.eligibility_traces.traces.items():
                trace_state = np.array(trace_state_key)
                update_target = self.value_function.get_value(trace_state) + td_error * trace_value
                self.value_function.update(
                    trace_state,
                    update_target,
                    self.config.learning_rate * trace_value
                )

        # Decay traces
        self.eligibility_traces.decay(self.config.gamma)

        # Reset traces on episode end
        if done:
            self.eligibility_traces.reset()

        self.td_errors.append(td_error)
        self.value_estimates.append(value)

        return td_error

    def n_step_update(self, state: np.ndarray, action: int, reward: float,
                      next_state: np.ndarray, done: bool) -> Optional[float]:
        """N-step TD update"""

        # Add to buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # Not enough steps yet
        if len(self.n_step_buffer) < self.config.n_step and not done:
            return None

        # Calculate n-step return
        n_step_return = 0
        gamma_power = 1.0

        for i, (s, a, r, ns, d) in enumerate(self.n_step_buffer):
            n_step_return += gamma_power * r
            gamma_power *= self.config.gamma
            if d:
                break

        # Bootstrap from final state if not terminal
        if not self.n_step_buffer[-1][-1]:  # Not done
            final_state = self.n_step_buffer[-1][3]  # next_state of last transition

            if self.config.use_neural_network:
                final_state_tensor = torch.FloatTensor(final_state).to(device)
                with torch.no_grad():
                    final_value = self.value_function(final_state_tensor).item()
            else:
                final_value = self.value_function.get_value(final_state)

            n_step_return += gamma_power * final_value

        # Update value function for first state in buffer
        first_state = self.n_step_buffer[0][0]

        if self.config.use_neural_network:
            state_tensor = torch.FloatTensor(first_state).to(device)
            value = self.value_function(state_tensor)
            target = torch.tensor(n_step_return, device=device)

            loss = F.mse_loss(value, target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), 1.0)
            self.optimizer.step()

            td_error = (target - value).item()
        else:
            current_value = self.value_function.get_value(first_state)
            td_error = n_step_return - current_value
            self.value_function.update(first_state, n_step_return, self.config.learning_rate)

        # Clear buffer if episode ended
        if done:
            self.n_step_buffer.clear()

        self.td_errors.append(td_error)
        self.value_estimates.append(current_value if not self.config.use_neural_network else value.item())

        return td_error

    def sarsa_update(self, state: np.ndarray, action: int, reward: float,
                     next_state: np.ndarray, next_action: int, done: bool) -> float:
        """SARSA: On-policy TD control"""

        # For discrete actions, we maintain separate value for each state-action pair
        state_action_key = (tuple(state), action)
        next_state_action_key = (tuple(next_state), next_action)

        # This would typically use an action-value function Q(s,a)
        # For simplicity, using tabular representation here
        if not hasattr(self, 'q_values'):
            self.q_values = defaultdict(float)

        current_q = self.q_values[state_action_key]
        next_q = 0 if done else self.q_values[next_state_action_key]

        target = reward + self.config.gamma * next_q
        td_error = target - current_q

        # Update Q-value
        self.q_values[state_action_key] += self.config.learning_rate * td_error

        self.td_errors.append(td_error)
        return td_error

    def expected_sarsa_update(self, state: np.ndarray, action: int, reward: float,
                             next_state: np.ndarray, action_probs: np.ndarray,
                             done: bool) -> float:
        """Expected SARSA: Uses expected value over next actions"""

        if not hasattr(self, 'q_values'):
            self.q_values = defaultdict(float)

        state_action_key = (tuple(state), action)
        current_q = self.q_values[state_action_key]

        # Calculate expected value of next state
        expected_next_q = 0
        if not done:
            for next_action, prob in enumerate(action_probs):
                next_state_action_key = (tuple(next_state), next_action)
                expected_next_q += prob * self.q_values[next_state_action_key]

        target = reward + self.config.gamma * expected_next_q
        td_error = target - current_q

        # Update Q-value
        self.q_values[state_action_key] += self.config.learning_rate * td_error

        self.td_errors.append(td_error)
        return td_error

    def monte_carlo_update(self, episode: List[Tuple[np.ndarray, float]]) -> float:
        """Monte Carlo value estimation using complete episodes"""

        if not episode:
            return 0

        # Calculate returns for each state in episode
        returns = []
        G = 0

        for state, reward in reversed(episode):
            G = reward + self.config.gamma * G
            returns.append((state, G))

        returns.reverse()

        # Update value function with returns
        total_error = 0

        for state, G in returns:
            if self.config.use_neural_network:
                state_tensor = torch.FloatTensor(state).to(device)
                value = self.value_function(state_tensor)
                target = torch.tensor(G, device=device)

                loss = F.mse_loss(value, target)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), 1.0)
                self.optimizer.step()

                error = (target - value).item()
            else:
                current_value = self.value_function.get_value(state)
                error = G - current_value
                self.value_function.update(state, G, self.config.learning_rate)

            total_error += abs(error)
            self.td_errors.append(error)
            self.value_estimates.append(current_value if not self.config.use_neural_network else value.item())

        return total_error / len(returns)

    def bellman_backup(self, states: List[np.ndarray], rewards: List[float],
                       transition_probs: Dict) -> np.ndarray:
        """Bellman backup for value iteration (requires model)"""

        num_states = len(states)
        values = np.zeros(num_states)

        # Iterative value iteration
        for iteration in range(100):  # Max iterations
            old_values = values.copy()

            for s_idx, state in enumerate(states):
                # Bellman equation: V(s) = max_a sum_s' P(s'|s,a)[R(s,a,s') + gamma * V(s')]
                max_value = float('-inf')

                for action in transition_probs.get(s_idx, {}).keys():
                    action_value = 0

                    for next_s_idx, prob in transition_probs[s_idx][action].items():
                        immediate_reward = rewards[s_idx]  # Simplified reward structure
                        future_value = self.config.gamma * old_values[next_s_idx]
                        action_value += prob * (immediate_reward + future_value)

                    max_value = max(max_value, action_value)

                values[s_idx] = max_value if max_value > float('-inf') else 0

            # Check convergence
            if np.max(np.abs(values - old_values)) < 1e-6:
                logger.info(f"Bellman backup converged in {iteration + 1} iterations")
                break

        return values

    def update_target_network(self):
        """Update target network for stability"""
        if self.config.use_neural_network and hasattr(self, 'target_value_function'):
            self.target_value_function.load_state_dict(self.value_function.state_dict())

    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for a state"""

        if self.config.use_neural_network:
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                value = self.value_function(state_tensor).item()
        else:
            value = self.value_function.get_value(state)

        return value

    def get_action_values(self, state: np.ndarray, num_actions: int) -> np.ndarray:
        """Get Q-values for all actions in a state"""

        if hasattr(self, 'q_values'):
            q_array = np.zeros(num_actions)
            for action in range(num_actions):
                state_action_key = (tuple(state), action)
                q_array[action] = self.q_values.get(state_action_key, 0)
            return q_array
        else:
            return np.zeros(num_actions)

    def save(self, filepath: str):
        """Save learner state"""
        state = {
            'config': asdict(self.config),
            'td_errors': list(self.td_errors),
            'value_estimates': list(self.value_estimates)
        }

        if self.config.use_neural_network:
            state['value_function_state'] = self.value_function.state_dict()
            state['optimizer_state'] = self.optimizer.state_dict()
            if hasattr(self, 'target_value_function'):
                state['target_value_function_state'] = self.target_value_function.state_dict()
        else:
            state['tile_weights'] = dict(self.value_function.weights)

        if hasattr(self, 'q_values'):
            # Convert keys to strings for JSON serialization
            state['q_values'] = {str(k): v for k, v in self.q_values.items()}

        torch.save(state, filepath)
        logger.info(f"TD Learner saved to {filepath}")

    def load(self, filepath: str):
        """Load learner state"""
        state = torch.load(filepath, map_location=device)
        self.config = ValueFunctionConfig(**state['config'])

        if self.config.use_neural_network:
            self.value_function.load_state_dict(state['value_function_state'])
            self.optimizer.load_state_dict(state['optimizer_state'])
            if 'target_value_function_state' in state:
                self.target_value_function.load_state_dict(state['target_value_function_state'])
        else:
            self.value_function.weights = defaultdict(float, state.get('tile_weights', {}))

        if 'q_values' in state:
            # Convert string keys back to tuples
            self.q_values = defaultdict(float)
            for k, v in state['q_values'].items():
                self.q_values[eval(k)] = v

        self.td_errors = deque(state['td_errors'], maxlen=1000)
        self.value_estimates = deque(state['value_estimates'], maxlen=1000)

        logger.info(f"TD Learner loaded from {filepath}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get learning metrics"""
        return {
            'method': self.config.method,
            'avg_td_error': np.mean(self.td_errors) if self.td_errors else 0,
            'avg_value_estimate': np.mean(self.value_estimates) if self.value_estimates else 0,
            'num_states_visited': len(self.eligibility_traces.traces) if hasattr(self, 'eligibility_traces') else 0,
            'use_neural_network': self.config.use_neural_network,
            'lambda': self.config.lambda_param,
            'gamma': self.config.gamma
        }