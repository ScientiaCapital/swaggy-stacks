"""
Enhanced DQN Brain with LSTM and Dueling Architecture
Advanced neural network for temporal pattern recognition and value estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Tuple, Optional, Dict, Any

class EnhancedDQNBrain(nn.Module):
    """
    Enhanced DQN with LSTM for temporal patterns and dueling architecture
    
    Features:
    - LSTM layers for temporal pattern recognition
    - Dueling architecture for better value estimation
    - Proper weight initialization for better convergence
    - Support for sequence-based trading decisions
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128, 
                 num_lstm_layers: int = 1, dropout_rate: float = 0.2, 
                 device: str = 'cpu'):
        """
        Initialize Enhanced DQN Brain
        
        Args:
            state_size: Number of features in the state
            action_size: Number of possible actions
            hidden_size: Size of hidden layers
            num_lstm_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
            device: Device to run the model on ('cpu' or 'cuda')
        """
        super(EnhancedDQNBrain, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.device = device
        
        # LSTM layer for temporal pattern recognition
        self.lstm = nn.LSTM(
            input_size=state_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0
        )
        
        # Dueling architecture: separate value and advantage streams
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.to(device)
        
    def _initialize_weights(self):
        """Initialize weights for better convergence"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Orthogonal initialization for LSTM
                    nn.init.orthogonal_(param.data)
                else:
                    # Xavier initialization for linear layers
                    nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
                
    def forward(self, x: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network
        
        Args:
            x: Input state tensor of shape (batch_size, sequence_length, state_size)
            hidden_state: Previous hidden state for LSTM
            
        Returns:
            q_values: Q-values for each action
            hidden_out: New hidden state for LSTM
        """
        batch_size = x.size(0)
        sequence_length = x.size(1) if len(x.shape) > 2 else 1
        
        # If input is 2D, add sequence dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Initialize hidden state if not provided
        if hidden_state is None:
            h0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size).to(self.device)
            c0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size).to(self.device)
            hidden_state = (h0, c0)
        
        # LSTM for temporal patterns
        lstm_out, hidden_out = self.lstm(x, hidden_state)
        
        # Use the last output in the sequence
        lstm_out = lstm_out[:, -1, :]
        
        # Calculate value and advantage
        value = self.value_stream(lstm_out)
        advantage = self.advantage_stream(lstm_out)
        
        # Combine value and advantage (dueling architecture)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values, hidden_out
    
    def get_action(self, state: np.ndarray, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                   epsilon: float = 0.0) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            hidden_state: Previous hidden state for LSTM
            epsilon: Exploration rate
            
        Returns:
            action: Selected action
            hidden_out: New hidden state
        """
        if np.random.random() < epsilon:
            # Random action for exploration
            return np.random.randint(self.action_size), hidden_state
        
        # Get Q-values from network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if hidden_state is not None:
                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
                
            q_values, hidden_out = self.forward(state_tensor, hidden_state)
            
            # Select action with highest Q-value
            action = q_values.argmax().item()
            
        return action, hidden_out
    
    def get_q_values(self, state: np.ndarray, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get Q-values for all actions
        
        Args:
            state: Current state
            hidden_state: Previous hidden state for LSTM
            
        Returns:
            q_values: Q-values for all actions
            hidden_out: New hidden state
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if hidden_state is not None:
                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
                
            q_values, hidden_out = self.forward(state_tensor, hidden_state)
            
        return q_values, hidden_out
    
    def save_model(self, filepath: str):
        """Save the model to a file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_size': self.hidden_size,
            'num_lstm_layers': self.num_lstm_layers
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the model from a file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")


class ReplayBuffer:
    """
    Experience replay buffer for DQN training
    Stores experiences with LSTM hidden states
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool, hidden_state: Optional[Tuple] = None,
             next_hidden_state: Optional[Tuple] = None):
        """
        Add experience to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            hidden_state: LSTM hidden state
            next_hidden_state: Next LSTM hidden state
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'hidden_state': hidden_state,
            'next_hidden_state': next_hidden_state
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """
        Sample a batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary containing batched experiences
        """
        batch = random.sample(self.buffer, batch_size)
        
        # Separate components
        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])
        
        # Handle hidden states (they might be None)
        hidden_states = [exp['hidden_state'] for exp in batch]
        next_hidden_states = [exp['next_hidden_state'] for exp in batch]
        
        return {
            'states': torch.FloatTensor(states),
            'actions': torch.LongTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'next_states': torch.FloatTensor(next_states),
            'dones': torch.BoolTensor(dones),
            'hidden_states': hidden_states,
            'next_hidden_states': next_hidden_states
        }
    
    def __len__(self) -> int:
        """Return current size of buffer"""
        return len(self.buffer)


class DQNTrainer:
    """
    Trainer for Enhanced DQN with LSTM
    Handles training loop, target network updates, and experience replay
    """
    
    def __init__(self, model: EnhancedDQNBrain, target_model: EnhancedDQNBrain,
                 replay_buffer: ReplayBuffer, learning_rate: float = 0.001,
                 gamma: float = 0.99, target_update_frequency: int = 100,
                 batch_size: int = 32, device: str = 'cpu'):
        """
        Initialize DQN trainer
        
        Args:
            model: Main DQN model
            target_model: Target DQN model for stable training
            replay_buffer: Experience replay buffer
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            target_update_frequency: How often to update target network
            batch_size: Batch size for training
            device: Device to run training on
        """
        self.model = model
        self.target_model = target_model
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.batch_size = batch_size
        self.device = device
        
        # Initialize target network with same weights as main network
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Training statistics
        self.training_step = 0
        self.losses = []
        
    def train_step(self) -> float:
        """
        Perform one training step
        
        Returns:
            Loss value for this training step
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Get current Q-values
        current_q_values, _ = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Get next Q-values from target network
        with torch.no_grad():
            next_q_values, _ = self.target_model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            
            # Calculate target Q-values
            target_q_values = rewards + (self.gamma * max_next_q_values * ~dones)
        
        # Calculate loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Record loss
        self.losses.append(loss.item())
        
        return loss.item()
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get training statistics"""
        if not self.losses:
            return {'avg_loss': 0.0, 'recent_loss': 0.0}
        
        recent_losses = self.losses[-100:] if len(self.losses) > 100 else self.losses
        
        return {
            'avg_loss': np.mean(self.losses),
            'recent_loss': np.mean(recent_losses),
            'training_steps': self.training_step
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the Enhanced DQN Brain
    print("Testing Enhanced DQN Brain...")
    
    # Create model
    state_size = 20  # Example: 20 features
    action_size = 3  # Example: 3 actions (buy, sell, hold)
    model = EnhancedDQNBrain(state_size, action_size, hidden_size=128, num_lstm_layers=2)
    
    # Test forward pass
    batch_size = 4
    sequence_length = 10
    test_input = torch.randn(batch_size, sequence_length, state_size)
    
    q_values, hidden_state = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Q-values shape: {q_values.shape}")
    print(f"Hidden state shape: {hidden_state[0].shape}")
    
    # Test action selection
    test_state = np.random.randn(state_size)
    action, new_hidden = model.get_action(test_state, epsilon=0.1)
    print(f"Selected action: {action}")
    
    # Test replay buffer
    buffer = ReplayBuffer(capacity=1000)
    for i in range(100):
        state = np.random.randn(state_size)
        action = np.random.randint(action_size)
        reward = np.random.randn()
        next_state = np.random.randn(state_size)
        done = np.random.choice([True, False])
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Test batch sampling
    batch = buffer.sample(32)
    print(f"Batch states shape: {batch['states'].shape}")
    print(f"Batch actions shape: {batch['actions'].shape}")
    
    print("Enhanced DQN Brain test completed successfully!")
