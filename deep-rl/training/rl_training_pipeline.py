"""
Complete RL Training Pipeline for Swaggy Stacks
Integrates Enhanced DQN Brain, Validation Framework, Meta-Orchestrator, and Dashboard
"""

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enhanced_dqn_brain import EnhancedDQNBrain, ReplayBuffer, DQNTrainer
from validation.trading_validation_framework import TradingValidationFramework
from training.meta_orchestrator import MetaRLTradingOrchestrator
from monitoring.trading_dashboard import TradingDashboard

class RLTradingTrainer:
    """
    Complete RL training pipeline for trading systems
    Integrates all components for end-to-end training and validation
    """
    
    def __init__(self, config: Dict[str, Any], env_class: Callable, data_sources: Dict[str, Any]):
        """
        Initialize RL training pipeline
        
        Args:
            config: Training configuration dictionary
            env_class: Environment class for creating training environments
            data_sources: Dictionary of data sources for different market regimes
        """
        self.config = config
        self.env_class = env_class
        self.data_sources = data_sources
        
        # Training parameters
        self.state_size = config.get('state_size', 20)
        self.action_size = config.get('action_size', 3)
        self.hidden_size = config.get('hidden_size', 128)
        self.num_lstm_layers = config.get('num_lstm_layers', 2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 32)
        self.target_update_frequency = config.get('target_update_frequency', 100)
        self.memory_size = config.get('memory_size', 10000)
        self.device = config.get('device', 'cpu')
        
        # Training state
        self.epsilon = self.epsilon_start
        self.episode = 0
        self.training_step = 0
        self.best_performance = float('-inf')
        
        # Initialize components
        self._initialize_components()
        
        # Training statistics
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'portfolio_values': [],
            'epsilon_values': [],
            'losses': [],
            'validation_results': []
        }
        
    def _initialize_components(self):
        """Initialize all training components"""
        print("Initializing RL Training Pipeline...")
        
        # Initialize Enhanced DQN Brain
        self.brain = EnhancedDQNBrain(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_size=self.hidden_size,
            num_lstm_layers=self.num_lstm_layers,
            device=self.device
        )
        
        # Initialize target network
        self.target_brain = EnhancedDQNBrain(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_size=self.hidden_size,
            num_lstm_layers=self.num_lstm_layers,
            device=self.device
        )
        
        # Copy weights to target network
        self.target_brain.load_state_dict(self.brain.state_dict())
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=self.memory_size)
        
        # Initialize DQN trainer
        self.dqn_trainer = DQNTrainer(
            model=self.brain,
            target_model=self.target_brain,
            replay_buffer=self.replay_buffer,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            target_update_frequency=self.target_update_frequency,
            batch_size=self.batch_size,
            device=self.device
        )
        
        # Initialize validation framework
        self.validation_framework = TradingValidationFramework(
            model=self.brain,
            env_class=self.env_class,
            data_sources=self.data_sources,
            initial_balance=self.config.get('initial_balance', 10000)
        )
        
        # Initialize dashboard
        self.dashboard = TradingDashboard(
            update_interval=self.config.get('dashboard_update_interval', 10),
            save_interval=self.config.get('dashboard_save_interval', 100)
        )
        
        # Initialize meta-orchestrator (optional)
        if self.config.get('use_meta_orchestrator', False):
            self._initialize_meta_orchestrator()
        
        print("RL Training Pipeline initialized successfully!")
    
    def _initialize_meta_orchestrator(self):
        """Initialize meta-orchestrator with specialized agents"""
        print("Initializing Meta-Orchestrator...")
        
        # Create specialized agents
        specialized_agents = {}
        
        # Trend following agent
        specialized_agents['trend_follower'] = EnhancedDQNBrain(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_size=self.hidden_size,
            num_lstm_layers=self.num_lstm_layers,
            device=self.device
        )
        
        # Mean reversion agent
        specialized_agents['mean_reversion'] = EnhancedDQNBrain(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_size=self.hidden_size,
            num_lstm_layers=self.num_lstm_layers,
            device=self.device
        )
        
        # Volatility agent
        specialized_agents['volatility'] = EnhancedDQNBrain(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_size=self.hidden_size,
            num_lstm_layers=self.num_lstm_layers,
            device=self.device
        )
        
        # Initialize meta-orchestrator
        self.meta_orchestrator = MetaRLTradingOrchestrator(
            specialized_agents=specialized_agents,
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=self.learning_rate,
            device=self.device
        )
        
        print("Meta-Orchestrator initialized successfully!")
    
    def train(self, num_episodes: int, validation_interval: int = 100, 
              save_interval: int = 500, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the RL trading system
        
        Args:
            num_episodes: Number of training episodes
            validation_interval: How often to run validation
            save_interval: How often to save models
            verbose: Whether to print training progress
            
        Returns:
            Training results dictionary
        """
        print(f"Starting RL training for {num_episodes} episodes...")
        print(f"Device: {self.device}")
        print(f"State size: {self.state_size}, Action size: {self.action_size}")
        print(f"Hidden size: {self.hidden_size}, LSTM layers: {self.num_lstm_layers}")
        print("=" * 60)
        
        # Training loop
        for episode in range(num_episodes):
            self.episode = episode
            
            # Run training episode
            episode_result = self._run_training_episode()
            
            # Update training statistics
            self._update_training_stats(episode_result)
            
            # Update dashboard
            self._update_dashboard(episode_result)
            
            # Run validation
            if episode % validation_interval == 0 and episode > 0:
                validation_results = self._run_validation()
                self.training_stats['validation_results'].append({
                    'episode': episode,
                    'results': validation_results
                })
            
            # Save models
            if episode % save_interval == 0 and episode > 0:
                self._save_models(episode)
            
            # Print progress
            if verbose and episode % 10 == 0:
                self._print_training_progress(episode, episode_result)
            
            # Decay epsilon
            self._decay_epsilon()
        
        # Final validation and save
        print("\nTraining completed! Running final validation...")
        final_validation = self._run_validation()
        self._save_models(num_episodes, is_final=True)
        
        # Generate final report
        final_report = self._generate_final_report()
        
        return {
            'training_stats': self.training_stats,
            'final_validation': final_validation,
            'final_report': final_report,
            'best_performance': self.best_performance
        }
    
    def _run_training_episode(self) -> Dict[str, Any]:
        """Run a single training episode"""
        # Create environment
        env = self.env_class(
            data=self.data_sources['training'],  # Use training data
            initial_balance=self.config.get('initial_balance', 10000)
        )
        
        # Initialize episode
        state = env.reset()
        hidden_state = None
        done = False
        episode_reward = 0
        episode_actions = []
        episode_losses = []
        
        while not done:
            # Get action from model
            if hasattr(self, 'meta_orchestrator') and self.config.get('use_meta_orchestrator', False):
                action, hidden_states, weights = self.meta_orchestrator.get_action(
                    state, {name: hidden_state for name in self.meta_orchestrator.agents.keys()}, 
                    epsilon=self.epsilon
                )
                # Use the main brain's hidden state
                hidden_state = hidden_states.get('trend_follower', hidden_state)
            else:
                action, hidden_state = self.brain.get_action(state, hidden_state, epsilon=self.epsilon)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            self.replay_buffer.push(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                hidden_state=hidden_state,
                next_hidden_state=hidden_state  # Simplified for now
            )
            
            # Train the model
            if len(self.replay_buffer) >= self.batch_size:
                loss = self.dqn_trainer.train_step()
                episode_losses.append(loss)
                self.training_step += 1
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_actions.append(action)
        
        # Calculate episode metrics
        final_return = (env.portfolio_value - env.initial_balance) / env.initial_balance * 100
        sharpe_ratio = env.calculate_sharpe_ratio() if hasattr(env, 'calculate_sharpe_ratio') else 0
        win_rate = 1 if final_return > 0 else 0
        max_drawdown = env.calculate_max_drawdown() if hasattr(env, 'calculate_max_drawdown') else 0
        profit_factor = env.calculate_profit_factor() if hasattr(env, 'calculate_profit_factor') else 0
        calmar_ratio = final_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'episode': self.episode,
            'reward': episode_reward,
            'portfolio_value': env.portfolio_value,
            'final_return': final_return,
            'sharpe_ratio': sharpe_ratio,
            'actions': episode_actions,
            'epsilon': self.epsilon,
            'losses': episode_losses,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio
        }
    
    def _update_training_stats(self, episode_result: Dict[str, Any]):
        """Update training statistics"""
        self.training_stats['episodes'].append(episode_result['episode'])
        self.training_stats['rewards'].append(episode_result['reward'])
        self.training_stats['portfolio_values'].append(episode_result['portfolio_value'])
        self.training_stats['epsilon_values'].append(episode_result['epsilon'])
        
        if episode_result['losses']:
            self.training_stats['losses'].extend(episode_result['losses'])
    
    def _update_dashboard(self, episode_result: Dict[str, Any]):
        """Update the monitoring dashboard"""
        self.dashboard.update(
            episode=episode_result['episode'],
            portfolio_value=episode_result['portfolio_value'],
            returns=episode_result['final_return'],
            sharpe_ratio=episode_result['sharpe_ratio'],
            actions=episode_result['actions'],
            epsilon=episode_result['epsilon'],
            loss=np.mean(episode_result['losses']) if episode_result['losses'] else 0,
            win_rate=episode_result['win_rate'],
            max_drawdown=episode_result['max_drawdown'],
            profit_factor=episode_result['profit_factor'],
            calmar_ratio=episode_result['calmar_ratio']
        )
    
    def _run_validation(self) -> Dict[str, Dict[str, float]]:
        """Run validation on out-of-sample data"""
        print(f"\nRunning validation at episode {self.episode}...")
        
        validation_results = self.validation_framework.run_validation(
            num_episodes=self.config.get('validation_episodes', 10),
            verbose=False
        )
        
        # Update best performance
        overall_return = np.mean([results['avg_return'] for results in validation_results.values()])
        if overall_return > self.best_performance:
            self.best_performance = overall_return
            print(f"New best performance: {overall_return:.2f}%")
        
        return validation_results
    
    def _save_models(self, episode: int, is_final: bool = False):
        """Save trained models"""
        suffix = 'final' if is_final else f'episode_{episode}'
        
        # Save main brain
        self.brain.save_model(f'models/brain_{suffix}.pth')
        
        # Save target brain
        self.target_brain.save_model(f'models/target_brain_{suffix}.pth')
        
        # Save meta-orchestrator if used
        if hasattr(self, 'meta_orchestrator'):
            self.meta_orchestrator.save_orchestrator(f'models/meta_orchestrator_{suffix}.pth')
        
        # Save training statistics
        with open(f'models/training_stats_{suffix}.json', 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        print(f"Models saved with suffix: {suffix}")
    
    def _print_training_progress(self, episode: int, episode_result: Dict[str, Any]):
        """Print training progress"""
        print(f"Episode {episode:4d} | "
              f"Reward: {episode_result['reward']:8.2f} | "
              f"Return: {episode_result['final_return']:6.2f}% | "
              f"Portfolio: ${episode_result['portfolio_value']:8.2f} | "
              f"Sharpe: {episode_result['sharpe_ratio']:5.2f} | "
              f"Epsilon: {episode_result['epsilon']:5.3f}")
    
    def _decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _generate_final_report(self) -> str:
        """Generate final training report"""
        report = []
        report.append("=" * 80)
        report.append("RL TRADING SYSTEM TRAINING REPORT")
        report.append("=" * 80)
        report.append(f"Training completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total episodes: {len(self.training_stats['episodes'])}")
        report.append(f"Total training steps: {self.training_step}")
        report.append(f"Best performance: {self.best_performance:.2f}%")
        report.append("")
        
        # Training statistics
        if self.training_stats['rewards']:
            avg_reward = np.mean(self.training_stats['rewards'])
            best_reward = max(self.training_stats['rewards'])
            worst_reward = min(self.training_stats['rewards'])
            
            report.append("TRAINING STATISTICS:")
            report.append(f"  Average reward: {avg_reward:.4f}")
            report.append(f"  Best reward: {best_reward:.4f}")
            report.append(f"  Worst reward: {worst_reward:.4f}")
            report.append("")
        
        # Portfolio performance
        if self.training_stats['portfolio_values']:
            initial_value = self.training_stats['portfolio_values'][0]
            final_value = self.training_stats['portfolio_values'][-1]
            total_return = (final_value - initial_value) / initial_value * 100
            
            report.append("PORTFOLIO PERFORMANCE:")
            report.append(f"  Initial value: ${initial_value:,.2f}")
            report.append(f"  Final value: ${final_value:,.2f}")
            report.append(f"  Total return: {total_return:.2f}%")
            report.append("")
        
        # Validation results
        if self.training_stats['validation_results']:
            report.append("VALIDATION RESULTS:")
            for val_result in self.training_stats['validation_results']:
                episode = val_result['episode']
                results = val_result['results']
                overall_return = np.mean([r['avg_return'] for r in results.values()])
                report.append(f"  Episode {episode}: {overall_return:.2f}% overall return")
            report.append("")
        
        # Model architecture
        report.append("MODEL ARCHITECTURE:")
        report.append(f"  State size: {self.state_size}")
        report.append(f"  Action size: {self.action_size}")
        report.append(f"  Hidden size: {self.hidden_size}")
        report.append(f"  LSTM layers: {self.num_lstm_layers}")
        report.append(f"  Device: {self.device}")
        report.append("")
        
        # Training parameters
        report.append("TRAINING PARAMETERS:")
        report.append(f"  Learning rate: {self.learning_rate}")
        report.append(f"  Gamma: {self.gamma}")
        report.append(f"  Epsilon start: {self.epsilon_start}")
        report.append(f"  Epsilon end: {self.epsilon_end}")
        report.append(f"  Batch size: {self.batch_size}")
        report.append(f"  Memory size: {self.memory_size}")
        
        return "\n".join(report)
    
    def load_models(self, model_path: str):
        """Load trained models"""
        self.brain.load_model(f'{model_path}/brain_final.pth')
        self.target_brain.load_model(f'{model_path}/target_brain_final.pth')
        
        if hasattr(self, 'meta_orchestrator'):
            self.meta_orchestrator.load_orchestrator(f'{model_path}/meta_orchestrator_final.pth')
        
        print(f"Models loaded from {model_path}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the trained model"""
        print(f"Evaluating model for {num_episodes} episodes...")
        
        # Set epsilon to 0 for evaluation
        original_epsilon = self.epsilon
        self.epsilon = 0.0
        
        evaluation_results = []
        
        for episode in range(num_episodes):
            episode_result = self._run_training_episode()
            evaluation_results.append(episode_result)
            
            print(f"Evaluation episode {episode + 1}: Return = {episode_result['final_return']:.2f}%")
        
        # Restore original epsilon
        self.epsilon = original_epsilon
        
        # Calculate evaluation statistics
        returns = [result['final_return'] for result in evaluation_results]
        portfolio_values = [result['portfolio_value'] for result in evaluation_results]
        sharpe_ratios = [result['sharpe_ratio'] for result in evaluation_results]
        
        evaluation_stats = {
            'num_episodes': num_episodes,
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'best_return': max(returns),
            'worst_return': min(returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'win_rate': np.mean([1 if r > 0 else 0 for r in returns]) * 100,
            'avg_portfolio_value': np.mean(portfolio_values)
        }
        
        print("\nEvaluation Results:")
        for key, value in evaluation_stats.items():
            print(f"  {key}: {value:.2f}")
        
        return {
            'episode_results': evaluation_results,
            'statistics': evaluation_stats
        }


# Example usage and testing
if __name__ == "__main__":
    # Mock environment class for testing
    class MockTradingEnvironment:
        def __init__(self, data, initial_balance=10000):
            self.initial_balance = initial_balance
            self.portfolio_value = initial_balance
            self.step_count = 0
            self.max_steps = 100
            self.data = data
        
        def reset(self):
            self.step_count = 0
            self.portfolio_value = self.initial_balance
            return np.random.randn(20)
        
        def step(self, action):
            self.step_count += 1
            
            # Simulate portfolio value changes
            change = np.random.randn() * 0.01
            self.portfolio_value *= (1 + change)
            
            reward = change * 1000
            done = self.step_count >= self.max_steps
            
            return np.random.randn(20), reward, done, {}
        
        def calculate_sharpe_ratio(self):
            return np.random.uniform(0.5, 2.0)
        
        def calculate_max_drawdown(self):
            return np.random.uniform(5, 25)
        
        def calculate_profit_factor(self):
            return np.random.uniform(0.8, 2.0)
    
    # Test the RL training pipeline
    print("Testing RL Training Pipeline...")
    
    # Configuration
    config = {
        'state_size': 20,
        'action_size': 3,
        'hidden_size': 128,
        'num_lstm_layers': 2,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 32,
        'target_update_frequency': 100,
        'memory_size': 10000,
        'device': 'cpu',
        'initial_balance': 10000,
        'dashboard_update_interval': 5,
        'dashboard_save_interval': 20,
        'use_meta_orchestrator': False,
        'validation_episodes': 5
    }
    
    # Mock data sources
    data_sources = {
        'training': np.random.randn(1000),
        'validation_bull': np.random.randn(500),
        'validation_bear': np.random.randn(500) * 0.5,
        'validation_sideways': np.random.randn(500) * 0.1
    }
    
    # Create training pipeline
    trainer = RLTradingTrainer(config, MockTradingEnvironment, data_sources)
    
    # Run training
    results = trainer.train(num_episodes=50, validation_interval=20, save_interval=50)
    
    # Print final report
    print("\n" + results['final_report'])
    
    # Run evaluation
    evaluation = trainer.evaluate(num_episodes=5)
    
    print("RL Training Pipeline test completed successfully!")
