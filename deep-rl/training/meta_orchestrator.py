"""
Meta-Orchestrator for Combining Specialized Trading Agents
Learns to weight and combine multiple specialized RL agents for optimal performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class MetaRLTradingOrchestrator:
    """
    Meta-orchestrator that learns to combine specialized trading agents
    Uses a neural network to learn optimal weighting of agent predictions
    """
    
    def __init__(self, specialized_agents: Dict[str, Any], state_size: int, action_size: int, 
                 learning_rate: float = 0.001, device: str = 'cpu'):
        """
        Initialize Meta-Orchestrator
        
        Args:
            specialized_agents: Dictionary of specialized agents
            state_size: Size of the state space
            action_size: Size of the action space
            learning_rate: Learning rate for the meta-network
            device: Device to run the model on
        """
        self.agents = specialized_agents
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Meta-network that learns to weight agent opinions
        self.meta_network = nn.Sequential(
            nn.Linear(state_size * (len(specialized_agents) + 1), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(specialized_agents)),
            nn.Softmax(dim=1)  # Output weights sum to 1
        ).to(device)
        
        self.optimizer = optim.Adam(self.meta_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Track agent performance
        self.agent_performance = {name: [] for name in specialized_agents.keys()}
        self.agent_confidence = {name: 0.5 for name in specialized_agents.keys()}
        self.agent_weights_history = {name: [] for name in specialized_agents.keys()}
        
        # Training statistics
        self.training_losses = []
        self.episode_rewards = []
        self.weight_entropy = []
        
        # Performance tracking
        self.performance_window = 100
        self.agent_recent_performance = {name: [] for name in specialized_agents.keys()}
        
    def get_action(self, state: np.ndarray, hidden_states: Optional[Dict[str, Any]] = None, 
                   epsilon: float = 0.0) -> Tuple[int, Dict[str, Any], np.ndarray]:
        """
        Get action by combining specialized agent predictions
        
        Args:
            state: Current state
            hidden_states: Hidden states for each agent's LSTM
            epsilon: Exploration rate
            
        Returns:
            final_action: Combined action
            new_hidden_states: Updated hidden states for all agents
            weights: Agent weights used for this decision
        """
        if hidden_states is None:
            hidden_states = {name: None for name in self.agents.keys()}
            
        # Get predictions from all agents
        agent_predictions = {}
        new_hidden_states = {}
        
        for name, agent in self.agents.items():
            hidden_state = hidden_states.get(name, None)
            action, new_hidden = agent.get_action(state, hidden_state, epsilon=0.0)
            agent_predictions[name] = action
            new_hidden_states[name] = new_hidden
        
        # Prepare input for meta-network
        meta_input = self._prepare_meta_input(state, agent_predictions)
        
        # Get weights from meta-network
        with torch.no_grad():
            weights_tensor = self.meta_network(meta_input)
            weights = weights_tensor.squeeze().cpu().numpy()
        
        # Store weights for analysis
        for i, name in enumerate(self.agents.keys()):
            self.agent_weights_history[name].append(weights[i])
        
        # Weighted action selection
        weighted_actions = np.zeros(self.action_size)
        for i, (name, action) in enumerate(agent_predictions.items()):
            weighted_actions[action] += weights[i]
        
        # Select action with highest weighted vote
        final_action = np.argmax(weighted_actions)
        
        # Occasionally explore random action
        if np.random.random() < epsilon:
            final_action = np.random.randint(self.action_size)
            
        return final_action, new_hidden_states, weights
    
    def _prepare_meta_input(self, state: np.ndarray, agent_predictions: Dict[str, int]) -> torch.Tensor:
        """
        Prepare input for the meta-network
        Includes original state and each agent's prediction
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Convert agent predictions to one-hot encoding
        agent_inputs = []
        for name, action in agent_predictions.items():
            one_hot = np.zeros(self.action_size)
            one_hot[action] = 1
            agent_inputs.append(one_hot)
        
        # Concatenate all inputs
        agent_tensor = torch.FloatTensor(np.concatenate(agent_inputs)).unsqueeze(0).to(self.device)
        meta_input = torch.cat([state_tensor, agent_tensor], dim=1)
        
        return meta_input
    
    def update_agent_performance(self, agent_name: str, reward: float):
        """Update performance tracking for a specific agent"""
        self.agent_performance[agent_name].append(reward)
        self.agent_recent_performance[agent_name].append(reward)
        
        # Keep only recent performance for rolling average
        if len(self.agent_recent_performance[agent_name]) > self.performance_window:
            self.agent_recent_performance[agent_name].pop(0)
        
        # Update confidence based on recent performance
        if len(self.agent_recent_performance[agent_name]) > 0:
            recent_performance = np.mean(self.agent_recent_performance[agent_name])
            self.agent_confidence[agent_name] = 0.9 * self.agent_confidence[agent_name] + 0.1 * recent_performance
    
    def train_meta_network(self, states: List[np.ndarray], actions: List[int], 
                          rewards: List[float], next_states: List[np.ndarray], 
                          dones: List[bool]) -> float:
        """
        Train the meta-network to improve agent weighting
        
        Args:
            states: List of states
            actions: List of actions taken
            rewards: List of rewards received
            next_states: List of next states
            dones: List of done flags
            
        Returns:
            Loss value for this training step
        """
        if len(states) < 2:
            return 0.0
        
        # Calculate advantages for each agent
        agent_advantages = self._calculate_agent_advantages(states, actions, rewards, next_states, dones)
        
        # Prepare training data
        meta_inputs = []
        target_weights = []
        
        for i in range(len(states)):
            # Get agent predictions for this state
            agent_predictions = {}
            for name, agent in self.agents.items():
                action, _ = agent.get_action(states[i], epsilon=0.0)
                agent_predictions[name] = action
            
            # Prepare meta input
            meta_input = self._prepare_meta_input(states[i], agent_predictions)
            meta_inputs.append(meta_input)
            
            # Calculate target weights based on advantages
            target_weight = self._calculate_target_weights(agent_advantages, i)
            target_weights.append(target_weight)
        
        # Convert to tensors
        meta_inputs = torch.cat(meta_inputs, dim=0)
        target_weights = torch.FloatTensor(target_weights).to(self.device)
        
        # Forward pass
        predicted_weights = self.meta_network(meta_inputs)
        
        # Calculate loss
        loss = self.loss_fn(predicted_weights, target_weights)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.meta_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Record loss
        self.training_losses.append(loss.item())
        
        return loss.item()
    
    def _calculate_agent_advantages(self, states: List[np.ndarray], actions: List[int], 
                                   rewards: List[float], next_states: List[np.ndarray], 
                                   dones: List[bool]) -> Dict[str, List[float]]:
        """Calculate advantages for each agent based on their performance"""
        agent_advantages = {name: [] for name in self.agents.keys()}
        
        # Calculate discounted returns
        discounted_returns = []
        gamma = 0.99
        
        for i in range(len(rewards)):
            if dones[i]:
                discounted_returns.append(rewards[i])
            else:
                # Calculate future discounted return
                future_return = 0
                discount = 1
                for j in range(i + 1, len(rewards)):
                    future_return += discount * rewards[j]
                    discount *= gamma
                    if dones[j]:
                        break
                discounted_returns.append(rewards[i] + gamma * future_return)
        
        # Calculate advantages for each agent
        for name in self.agents.keys():
            agent_rewards = []
            
            # Get rewards for this agent's actions
            for i in range(len(states)):
                # Get what this agent would have done
                agent_action, _ = self.agents[name].get_action(states[i], epsilon=0.0)
                
                # If agent's action matches the taken action, use the reward
                if agent_action == actions[i]:
                    agent_rewards.append(discounted_returns[i])
                else:
                    # Use a baseline reward (could be improved with value function)
                    agent_rewards.append(0.0)
            
            # Calculate advantages (reward - baseline)
            baseline = np.mean(agent_rewards) if agent_rewards else 0
            advantages = [r - baseline for r in agent_rewards]
            agent_advantages[name] = advantages
        
        return agent_advantages
    
    def _calculate_target_weights(self, agent_advantages: Dict[str, List[float]], 
                                 step: int) -> np.ndarray:
        """Calculate target weights based on agent advantages"""
        weights = np.zeros(len(self.agents))
        
        # Get advantages for this step
        step_advantages = {}
        for name, advantages in agent_advantages.items():
            if step < len(advantages):
                step_advantages[name] = advantages[step]
            else:
                step_advantages[name] = 0.0
        
        # Convert advantages to weights using softmax
        advantage_values = list(step_advantages.values())
        
        # Add temperature to control sharpness
        temperature = 2.0
        exp_advantages = np.exp(np.array(advantage_values) / temperature)
        weights = exp_advantages / np.sum(exp_advantages)
        
        return weights
    
    def get_agent_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive statistics for all agents"""
        stats = {}
        
        for name in self.agents.keys():
            performance = self.agent_performance[name]
            recent_performance = self.agent_recent_performance[name]
            weights = self.agent_weights_history[name]
            
            stats[name] = {
                'total_episodes': len(performance),
                'avg_performance': np.mean(performance) if performance else 0,
                'recent_avg_performance': np.mean(recent_performance) if recent_performance else 0,
                'confidence': self.agent_confidence[name],
                'avg_weight': np.mean(weights) if weights else 0,
                'weight_std': np.std(weights) if weights else 0,
                'best_performance': max(performance) if performance else 0,
                'worst_performance': min(performance) if performance else 0
            }
        
        return stats
    
    def plot_agent_performance(self, save_path: Optional[str] = None):
        """Plot agent performance and weight evolution"""
        if not any(self.agent_performance.values()):
            print("No performance data available for plotting.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Meta-Orchestrator Agent Performance', fontsize=16)
        
        # Plot 1: Agent performance over time
        for name, performance in self.agent_performance.items():
            if performance:
                # Calculate rolling average
                window = min(50, len(performance))
                rolling_avg = np.convolve(performance, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(rolling_avg, label=name, alpha=0.7)
        
        axes[0, 0].set_title('Agent Performance Over Time')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Performance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Agent weights over time
        for name, weights in self.agent_weights_history.items():
            if weights:
                # Calculate rolling average
                window = min(50, len(weights))
                rolling_avg = np.convolve(weights, np.ones(window)/window, mode='valid')
                axes[0, 1].plot(rolling_avg, label=name, alpha=0.7)
        
        axes[0, 1].set_title('Agent Weights Over Time')
        axes[0, 1].set_xlabel('Decision')
        axes[0, 1].set_ylabel('Weight')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Current agent confidence
        names = list(self.agent_confidence.keys())
        confidences = list(self.agent_confidence.values())
        axes[1, 0].bar(names, confidences, alpha=0.7)
        axes[1, 0].set_title('Current Agent Confidence')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Training loss
        if self.training_losses:
            axes[1, 1].plot(self.training_losses, alpha=0.7)
            axes[1, 1].set_title('Meta-Network Training Loss')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Agent performance plot saved to {save_path}")
        
        plt.show()
    
    def save_orchestrator(self, filepath: str):
        """Save the orchestrator state"""
        state = {
            'meta_network_state_dict': self.meta_network.state_dict(),
            'agent_performance': self.agent_performance,
            'agent_confidence': self.agent_confidence,
            'agent_weights_history': self.agent_weights_history,
            'training_losses': self.training_losses,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'agent_names': list(self.agents.keys())
        }
        
        torch.save(state, filepath)
        print(f"Meta-orchestrator saved to {filepath}")
    
    def load_orchestrator(self, filepath: str):
        """Load the orchestrator state"""
        state = torch.load(filepath, map_location=self.device)
        
        self.meta_network.load_state_dict(state['meta_network_state_dict'])
        self.agent_performance = state['agent_performance']
        self.agent_confidence = state['agent_confidence']
        self.agent_weights_history = state['agent_weights_history']
        self.training_losses = state['training_losses']
        
        print(f"Meta-orchestrator loaded from {filepath}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of orchestrator performance"""
        stats = self.get_agent_statistics()
        
        report = []
        report.append("=" * 60)
        report.append("META-ORCHESTRATOR PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of Agents: {len(self.agents)}")
        report.append(f"Total Training Steps: {len(self.training_losses)}")
        report.append("")
        
        # Individual agent statistics
        for name, stat in stats.items():
            report.append(f"{name.upper()} AGENT:")
            report.append(f"  Total Episodes: {stat['total_episodes']}")
            report.append(f"  Average Performance: {stat['avg_performance']:.4f}")
            report.append(f"  Recent Performance: {stat['recent_avg_performance']:.4f}")
            report.append(f"  Confidence: {stat['confidence']:.4f}")
            report.append(f"  Average Weight: {stat['avg_weight']:.4f}")
            report.append(f"  Weight Std Dev: {stat['weight_std']:.4f}")
            report.append(f"  Best Performance: {stat['best_performance']:.4f}")
            report.append(f"  Worst Performance: {stat['worst_performance']:.4f}")
            report.append("")
        
        # Overall statistics
        if self.training_losses:
            avg_loss = np.mean(self.training_losses)
            recent_loss = np.mean(self.training_losses[-100:]) if len(self.training_losses) > 100 else avg_loss
            report.append("META-NETWORK TRAINING:")
            report.append(f"  Average Loss: {avg_loss:.6f}")
            report.append(f"  Recent Loss: {recent_loss:.6f}")
            report.append("")
        
        # Best performing agent
        best_agent = max(stats.keys(), key=lambda x: stats[x]['recent_avg_performance'])
        report.append(f"BEST PERFORMING AGENT: {best_agent}")
        report.append(f"Recent Performance: {stats[best_agent]['recent_avg_performance']:.4f}")
        
        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    # Mock specialized agents for testing
    class MockAgent:
        def __init__(self, name, bias=0.0):
            self.name = name
            self.bias = bias
        
        def get_action(self, state, hidden_state=None, epsilon=0.0):
            # Simulate different agent behaviors
            if self.name == 'trend_follower':
                action = 0 if np.random.random() < 0.6 + self.bias else np.random.randint(3)
            elif self.name == 'mean_reversion':
                action = 1 if np.random.random() < 0.6 + self.bias else np.random.randint(3)
            elif self.name == 'volatility':
                action = 2 if np.random.random() < 0.6 + self.bias else np.random.randint(3)
            else:
                action = np.random.randint(3)
            
            return action, None
    
    # Test the meta-orchestrator
    print("Testing Meta-Orchestrator...")
    
    # Create specialized agents
    specialized_agents = {
        'trend_follower': MockAgent('trend_follower', 0.1),
        'mean_reversion': MockAgent('mean_reversion', -0.1),
        'volatility': MockAgent('volatility', 0.05)
    }
    
    # Create meta-orchestrator
    state_size = 20
    action_size = 3
    orchestrator = MetaRLTradingOrchestrator(specialized_agents, state_size, action_size)
    
    # Simulate training episodes
    for episode in range(100):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        # Generate episode data
        for step in range(50):
            state = np.random.randn(state_size)
            action, hidden_states, weights = orchestrator.get_action(state, epsilon=0.1)
            reward = np.random.randn() * 0.1
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.random.randn(state_size))
            dones.append(step == 49)
            
            # Update agent performance
            for name in specialized_agents.keys():
                orchestrator.update_agent_performance(name, reward + np.random.randn() * 0.05)
        
        # Train meta-network
        loss = orchestrator.train_meta_network(states, actions, rewards, next_states, dones)
        
        if episode % 20 == 0:
            print(f"Episode {episode}, Loss: {loss:.6f}")
    
    # Get statistics
    stats = orchestrator.get_agent_statistics()
    print("\nAgent Statistics:")
    for name, stat in stats.items():
        print(f"{name}: {stat['recent_avg_performance']:.4f} (confidence: {stat['confidence']:.4f})")
    
    # Generate report
    report = orchestrator.generate_report()
    print("\n" + report)
    
    print("Meta-Orchestrator test completed successfully!")
