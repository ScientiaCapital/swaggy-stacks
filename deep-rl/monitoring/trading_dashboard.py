"""
Trading Dashboard for Real-time Performance Visualization
Comprehensive monitoring system for Deep RL trading performance
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import threading
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TradingDashboard:
    """
    Dashboard for monitoring trading system performance in real-time
    Provides comprehensive visualization of training progress and performance metrics
    """
    
    def __init__(self, update_interval: int = 10, save_interval: int = 100, 
                 max_episodes: int = 1000, figsize: Tuple[int, int] = (20, 12)):
        """
        Initialize trading dashboard
        
        Args:
            update_interval: How often to update the dashboard (in episodes)
            save_interval: How often to save dashboard images
            max_episodes: Maximum number of episodes to display
            figsize: Figure size for the dashboard
        """
        self.update_interval = update_interval
        self.save_interval = save_interval
        self.max_episodes = max_episodes
        self.figsize = figsize
        
        # Performance metrics storage
        self.episode_data = []
        self.performance_metrics = {
            'episodes': [],
            'returns': [],
            'sharpe_ratios': [],
            'portfolio_values': [],
            'actions': [],
            'epsilon_values': [],
            'losses': [],
            'win_rates': [],
            'max_drawdowns': [],
            'profit_factors': [],
            'calmar_ratios': []
        }
        
        # Real-time data
        self.current_episode = 0
        self.live_portfolio_value = 10000
        self.live_returns = []
        self.live_actions = []
        
        # Create figure and subplots
        self._setup_dashboard()
        
        # Animation and threading
        self.animation = None
        self.is_running = False
        self.update_thread = None
        
    def _setup_dashboard(self):
        """Set up the dashboard layout and subplots"""
        self.fig = plt.figure(figsize=self.figsize)
        self.gs = GridSpec(4, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(self.gs[0, 0])  # Portfolio value
        self.ax2 = self.fig.add_subplot(self.gs[0, 1])  # Returns distribution
        self.ax3 = self.fig.add_subplot(self.gs[0, 2])  # Sharpe ratio
        self.ax4 = self.fig.add_subplot(self.gs[1, 0])  # Actions distribution
        self.ax5 = self.fig.add_subplot(self.gs[1, 1])  # Training loss
        self.ax6 = self.fig.add_subplot(self.gs[1, 2])  # Win rate
        self.ax7 = self.fig.add_subplot(self.gs[2, 0])  # Max drawdown
        self.ax8 = self.fig.add_subplot(self.gs[2, 1])  # Profit factor
        self.ax9 = self.fig.add_subplot(self.gs[2, 2])  # Calmar ratio
        self.ax10 = self.fig.add_subplot(self.gs[3, :])  # Exploration rate
        
        # Format figure
        self.fig.suptitle('Deep RL Trading System Dashboard', fontsize=16, fontweight='bold')
        
        # Initialize empty plots
        self._initialize_plots()
        
    def _initialize_plots(self):
        """Initialize empty plots with proper formatting"""
        # Portfolio value
        self.ax1.set_title('Portfolio Value Over Time', fontweight='bold')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Portfolio Value ($)')
        self.ax1.grid(True, alpha=0.3)
        
        # Returns distribution
        self.ax2.set_title('Returns Distribution', fontweight='bold')
        self.ax2.set_xlabel('Return (%)')
        self.ax2.set_ylabel('Frequency')
        self.ax2.grid(True, alpha=0.3)
        
        # Sharpe ratio
        self.ax3.set_title('Sharpe Ratio Over Time', fontweight='bold')
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Sharpe Ratio')
        self.ax3.grid(True, alpha=0.3)
        
        # Actions distribution
        self.ax4.set_title('Action Distribution', fontweight='bold')
        self.ax4.set_xlabel('Action')
        self.ax4.set_ylabel('Count')
        self.ax4.grid(True, alpha=0.3)
        
        # Training loss
        self.ax5.set_title('Training Loss', fontweight='bold')
        self.ax5.set_xlabel('Episode')
        self.ax5.set_ylabel('Loss')
        self.ax5.grid(True, alpha=0.3)
        
        # Win rate
        self.ax6.set_title('Win Rate Over Time', fontweight='bold')
        self.ax6.set_xlabel('Episode')
        self.ax6.set_ylabel('Win Rate (%)')
        self.ax6.grid(True, alpha=0.3)
        
        # Max drawdown
        self.ax7.set_title('Max Drawdown Over Time', fontweight='bold')
        self.ax7.set_xlabel('Episode')
        self.ax7.set_ylabel('Drawdown (%)')
        self.ax7.grid(True, alpha=0.3)
        
        # Profit factor
        self.ax8.set_title('Profit Factor Over Time', fontweight='bold')
        self.ax8.set_xlabel('Episode')
        self.ax8.set_ylabel('Profit Factor')
        self.ax8.grid(True, alpha=0.3)
        
        # Calmar ratio
        self.ax9.set_title('Calmar Ratio Over Time', fontweight='bold')
        self.ax9.set_xlabel('Episode')
        self.ax9.set_ylabel('Calmar Ratio')
        self.ax9.grid(True, alpha=0.3)
        
        # Exploration rate
        self.ax10.set_title('Exploration Rate (Epsilon)', fontweight='bold')
        self.ax10.set_xlabel('Episode')
        self.ax10.set_ylabel('Epsilon')
        self.ax10.grid(True, alpha=0.3)
        
    def update(self, episode: int, portfolio_value: float, returns: float, 
               sharpe_ratio: float, actions: List[int], epsilon: float,
               loss: float = 0.0, win_rate: float = 0.0, max_drawdown: float = 0.0,
               profit_factor: float = 0.0, calmar_ratio: float = 0.0):
        """
        Update dashboard with new data
        
        Args:
            episode: Current episode number
            portfolio_value: Current portfolio value
            returns: Episode returns
            sharpe_ratio: Current Sharpe ratio
            actions: List of actions taken in episode
            epsilon: Current exploration rate
            loss: Training loss
            win_rate: Win rate
            max_drawdown: Maximum drawdown
            profit_factor: Profit factor
            calmar_ratio: Calmar ratio
        """
        # Store data
        self.performance_metrics['episodes'].append(episode)
        self.performance_metrics['returns'].append(returns)
        self.performance_metrics['sharpe_ratios'].append(sharpe_ratio)
        self.performance_metrics['portfolio_values'].append(portfolio_value)
        self.performance_metrics['actions'].extend(actions)
        self.performance_metrics['epsilon_values'].append(epsilon)
        self.performance_metrics['losses'].append(loss)
        self.performance_metrics['win_rates'].append(win_rate)
        self.performance_metrics['max_drawdowns'].append(max_drawdown)
        self.performance_metrics['profit_factors'].append(profit_factor)
        self.performance_metrics['calmar_ratios'].append(calmar_ratio)
        
        # Update current episode
        self.current_episode = episode
        self.live_portfolio_value = portfolio_value
        self.live_returns.append(returns)
        self.live_actions.extend(actions)
        
        # Limit data size
        for key in self.performance_metrics:
            if len(self.performance_metrics[key]) > self.max_episodes:
                self.performance_metrics[key] = self.performance_metrics[key][-self.max_episodes:]
        
        # Update display at specified intervals
        if episode % self.update_interval == 0:
            self._update_display(episode)
        
        # Save dashboard at specified intervals
        if episode % self.save_interval == 0 and episode > 0:
            self.save_dashboard(f'trading_dashboard_episode_{episode}.png')
    
    def _update_display(self, episode: int):
        """Update the dashboard display"""
        # Clear previous plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, 
                   self.ax6, self.ax7, self.ax8, self.ax9, self.ax10]:
            ax.clear()
        
        # Re-initialize plot formatting
        self._initialize_plots()
        
        # Plot portfolio value over time
        if self.performance_metrics['portfolio_values']:
            episodes = self.performance_metrics['episodes']
            portfolio_values = self.performance_metrics['portfolio_values']
            self.ax1.plot(episodes, portfolio_values, 'b-', linewidth=2, alpha=0.8)
            
            # Add trend line
            if len(portfolio_values) > 10:
                z = np.polyfit(episodes, portfolio_values, 1)
                p = np.poly1d(z)
                self.ax1.plot(episodes, p(episodes), 'r--', alpha=0.7, label='Trend')
                self.ax1.legend()
        
        # Plot returns distribution
        if self.performance_metrics['returns']:
            returns = self.performance_metrics['returns']
            self.ax2.hist(returns, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            self.ax2.axvline(np.mean(returns), color='r', linestyle='--', 
                           label=f'Mean: {np.mean(returns):.2f}%')
            self.ax2.axvline(np.median(returns), color='g', linestyle='--', 
                           label=f'Median: {np.median(returns):.2f}%')
            self.ax2.legend()
        
        # Plot Sharpe ratio over time
        if self.performance_metrics['sharpe_ratios']:
            episodes = self.performance_metrics['episodes']
            sharpe_ratios = self.performance_metrics['sharpe_ratios']
            self.ax3.plot(episodes, sharpe_ratios, 'g-', linewidth=2, alpha=0.8)
            
            # Add horizontal line at 1.0 (good Sharpe ratio)
            self.ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Good Sharpe (1.0)')
            self.ax3.legend()
        
        # Plot action distribution
        if self.performance_metrics['actions']:
            actions = self.performance_metrics['actions']
            action_counts = pd.Series(actions).value_counts().sort_index()
            bars = self.ax4.bar(action_counts.index.astype(str), action_counts.values, 
                               alpha=0.7, color=['red', 'green', 'blue'][:len(action_counts)])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                self.ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')
        
        # Plot training loss
        if self.performance_metrics['losses']:
            episodes = self.performance_metrics['episodes']
            losses = self.performance_metrics['losses']
            # Filter out zero losses for better visualization
            non_zero_losses = [(ep, loss) for ep, loss in zip(episodes, losses) if loss > 0]
            if non_zero_losses:
                ep_losses, loss_values = zip(*non_zero_losses)
                self.ax5.plot(ep_losses, loss_values, 'r-', linewidth=2, alpha=0.8)
        
        # Plot win rate over time
        if self.performance_metrics['win_rates']:
            episodes = self.performance_metrics['episodes']
            win_rates = self.performance_metrics['win_rates']
            self.ax6.plot(episodes, win_rates, 'purple', linewidth=2, alpha=0.8)
            self.ax6.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='50% Win Rate')
            self.ax6.legend()
        
        # Plot max drawdown over time
        if self.performance_metrics['max_drawdowns']:
            episodes = self.performance_metrics['episodes']
            drawdowns = self.performance_metrics['max_drawdowns']
            self.ax7.plot(episodes, drawdowns, 'orange', linewidth=2, alpha=0.8)
            self.ax7.axhline(y=20, color='r', linestyle='--', alpha=0.7, label='20% Drawdown Limit')
            self.ax7.legend()
        
        # Plot profit factor over time
        if self.performance_metrics['profit_factors']:
            episodes = self.performance_metrics['episodes']
            profit_factors = self.performance_metrics['profit_factors']
            self.ax8.plot(episodes, profit_factors, 'brown', linewidth=2, alpha=0.8)
            self.ax8.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Break-even (1.0)')
            self.ax8.legend()
        
        # Plot Calmar ratio over time
        if self.performance_metrics['calmar_ratios']:
            episodes = self.performance_metrics['episodes']
            calmar_ratios = self.performance_metrics['calmar_ratios']
            self.ax9.plot(episodes, calmar_ratios, 'pink', linewidth=2, alpha=0.8)
            self.ax9.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Good Calmar (1.0)')
            self.ax9.legend()
        
        # Plot exploration rate
        if self.performance_metrics['epsilon_values']:
            episodes = self.performance_metrics['episodes']
            epsilon_values = self.performance_metrics['epsilon_values']
            self.ax10.plot(episodes, epsilon_values, 'gray', linewidth=2, alpha=0.8)
            self.ax10.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='10% Exploration')
            self.ax10.legend()
        
        # Add episode information
        self.fig.suptitle(f'Deep RL Trading System Dashboard - Episode {episode}', 
                         fontsize=16, fontweight='bold')
        
        # Update layout and display
        self.fig.tight_layout()
        plt.pause(0.01)
    
    def start_live_monitoring(self):
        """Start live monitoring with automatic updates"""
        self.is_running = True
        self.update_thread = threading.Thread(target=self._live_update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, self._animate, interval=1000, blit=False
        )
        
        plt.show()
    
    def stop_live_monitoring(self):
        """Stop live monitoring"""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
        if self.update_thread:
            self.update_thread.join()
    
    def _live_update_loop(self):
        """Live update loop for real-time monitoring"""
        while self.is_running:
            time.sleep(1)  # Update every second
            # This would typically receive data from the training process
    
    def _animate(self, frame):
        """Animation function for live updates"""
        if self.is_running:
            # Update live data visualization
            pass
    
    def save_dashboard(self, filename: str = 'trading_dashboard.png'):
        """Save the dashboard to a file"""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved as {filename}")
    
    def add_validation_results(self, validation_results: Dict[str, Dict[str, float]]):
        """Add validation results to the dashboard"""
        # Create a new figure for validation results
        val_fig, val_axes = plt.subplots(2, 2, figsize=(15, 10))
        val_fig.suptitle('Validation Results Dashboard', fontsize=16, fontweight='bold')
        
        # Extract data from validation results
        regimes = list(validation_results.keys())
        returns = [results['avg_return'] for results in validation_results.values()]
        sharpe_ratios = [results['avg_sharpe'] for results in validation_results.values()]
        win_rates = [results['win_rate'] for results in validation_results.values()]
        drawdowns = [results['max_drawdown'] for results in validation_results.values()]
        
        # Plot returns by regime
        bars1 = val_axes[0, 0].bar(regimes, returns, alpha=0.7, color='skyblue')
        val_axes[0, 0].set_title('Average Returns by Market Regime', fontweight='bold')
        val_axes[0, 0].set_ylabel('Return (%)')
        val_axes[0, 0].tick_params(axis='x', rotation=45)
        val_axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            val_axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom')
        
        # Plot Sharpe ratios by regime
        bars2 = val_axes[0, 1].bar(regimes, sharpe_ratios, alpha=0.7, color='lightgreen')
        val_axes[0, 1].set_title('Sharpe Ratios by Market Regime', fontweight='bold')
        val_axes[0, 1].set_ylabel('Sharpe Ratio')
        val_axes[0, 1].tick_params(axis='x', rotation=45)
        val_axes[0, 1].grid(True, alpha=0.3)
        val_axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Good Sharpe (1.0)')
        val_axes[0, 1].legend()
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            val_axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}', ha='center', va='bottom')
        
        # Plot win rates by regime
        bars3 = val_axes[1, 0].bar(regimes, win_rates, alpha=0.7, color='lightcoral')
        val_axes[1, 0].set_title('Win Rates by Market Regime', fontweight='bold')
        val_axes[1, 0].set_ylabel('Win Rate (%)')
        val_axes[1, 0].tick_params(axis='x', rotation=45)
        val_axes[1, 0].grid(True, alpha=0.3)
        val_axes[1, 0].axhline(y=50, color='r', linestyle='--', alpha=0.7, label='50% Win Rate')
        val_axes[1, 0].legend()
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            val_axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom')
        
        # Plot drawdowns by regime
        bars4 = val_axes[1, 1].bar(regimes, drawdowns, alpha=0.7, color='orange')
        val_axes[1, 1].set_title('Max Drawdown by Market Regime', fontweight='bold')
        val_axes[1, 1].set_ylabel('Drawdown (%)')
        val_axes[1, 1].tick_params(axis='x', rotation=45)
        val_axes[1, 1].grid(True, alpha=0.3)
        val_axes[1, 1].axhline(y=20, color='r', linestyle='--', alpha=0.7, label='20% Drawdown Limit')
        val_axes[1, 1].legend()
        
        # Add value labels
        for bar in bars4:
            height = bar.get_height()
            val_axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom')
        
        val_fig.tight_layout()
        val_fig.savefig('validation_results_dashboard.png', dpi=300, bbox_inches='tight')
        print("Validation results dashboard saved as validation_results_dashboard.png")
        plt.show()
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get a summary of current performance metrics"""
        if not self.performance_metrics['returns']:
            return {}
        
        returns = self.performance_metrics['returns']
        portfolio_values = self.performance_metrics['portfolio_values']
        sharpe_ratios = self.performance_metrics['sharpe_ratios']
        
        summary = {
            'total_episodes': len(returns),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'best_return': max(returns),
            'worst_return': min(returns),
            'current_portfolio_value': portfolio_values[-1] if portfolio_values else 0,
            'total_return': ((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100) if len(portfolio_values) > 1 else 0,
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'current_epsilon': self.performance_metrics['epsilon_values'][-1] if self.performance_metrics['epsilon_values'] else 0,
            'win_rate': np.mean(self.performance_metrics['win_rates']) if self.performance_metrics['win_rates'] else 0
        }
        
        return summary
    
    def export_data(self, filename: str = 'trading_data.csv'):
        """Export performance data to CSV"""
        df = pd.DataFrame(self.performance_metrics)
        df.to_csv(filename, index=False)
        print(f"Performance data exported to {filename}")
    
    def clear_data(self):
        """Clear all stored data"""
        for key in self.performance_metrics:
            self.performance_metrics[key] = []
        self.current_episode = 0
        self.live_portfolio_value = 10000
        self.live_returns = []
        self.live_actions = []
        print("Dashboard data cleared")


# Example usage and testing
if __name__ == "__main__":
    # Test the trading dashboard
    print("Testing Trading Dashboard...")
    
    # Create dashboard
    dashboard = TradingDashboard(update_interval=5, save_interval=20)
    
    # Simulate training episodes
    initial_portfolio = 10000
    portfolio_value = initial_portfolio
    
    for episode in range(50):
        # Simulate episode data
        returns = np.random.normal(0.5, 2.0)  # Random returns
        portfolio_value *= (1 + returns / 100)
        
        sharpe_ratio = np.random.uniform(0.5, 2.0)
        actions = [np.random.randint(3) for _ in range(20)]  # Random actions
        epsilon = max(0.1, 1.0 - episode * 0.02)  # Decaying epsilon
        loss = np.random.exponential(0.1)
        win_rate = np.random.uniform(40, 80)
        max_drawdown = np.random.uniform(5, 25)
        profit_factor = np.random.uniform(0.8, 2.0)
        calmar_ratio = np.random.uniform(0.5, 3.0)
        
        # Update dashboard
        dashboard.update(
            episode=episode,
            portfolio_value=portfolio_value,
            returns=returns,
            sharpe_ratio=sharpe_ratio,
            actions=actions,
            epsilon=epsilon,
            loss=loss,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio
        )
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}: Portfolio Value = ${portfolio_value:.2f}, Return = {returns:.2f}%")
    
    # Get performance summary
    summary = dashboard.get_performance_summary()
    print("\nPerformance Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")
    
    # Save dashboard
    dashboard.save_dashboard('test_trading_dashboard.png')
    
    # Export data
    dashboard.export_data('test_trading_data.csv')
    
    print("Trading Dashboard test completed successfully!")
