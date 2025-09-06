"""
Trading Validation Framework
Robust validation system for testing trading models on out-of-sample data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradingValidationFramework:
    """
    Framework for validating trading models on out-of-sample data
    Supports multiple market regimes and comprehensive performance metrics
    """
    
    def __init__(self, model: Any, env_class: Callable, data_sources: Dict[str, Any], 
                 initial_balance: float = 10000, commission: float = 0.001):
        """
        Initialize validation framework
        
        Args:
            model: Trained model to validate
            env_class: Environment class for creating validation environments
            data_sources: Dictionary of data sources for different market regimes
            initial_balance: Initial portfolio balance
            commission: Trading commission rate
        """
        self.model = model
        self.env_class = env_class
        self.data_sources = data_sources
        self.initial_balance = initial_balance
        self.commission = commission
        self.validation_results = {}
        
        # Performance metrics to track
        self.metrics = [
            'avg_return', 'std_return', 'avg_sharpe', 'win_rate', 'max_drawdown',
            'profit_factor', 'calmar_ratio', 'sortino_ratio', 'var_95', 'cvar_95'
        ]
        
    def run_validation(self, num_episodes: int = 10, render: bool = False, 
                      verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Run validation across different market regimes
        
        Args:
            num_episodes: Number of episodes to run for each regime
            render: Whether to render the environment during validation
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing validation results for each regime
        """
        results = {}
        
        if verbose:
            print("=" * 60)
            print("TRADING MODEL VALIDATION")
            print("=" * 60)
            print(f"Initial Balance: ${self.initial_balance:,.2f}")
            print(f"Commission Rate: {self.commission:.3f}")
            print(f"Episodes per Regime: {num_episodes}")
            print()
        
        for regime_name, data in self.data_sources.items():
            if verbose:
                print(f"Validating on {regime_name.upper()} regime...")
                
            regime_results = self._validate_on_regime(data, num_episodes, render, verbose)
            results[regime_name] = regime_results
            
            if verbose:
                self._print_regime_summary(regime_name, regime_results)
                print()
            
        self.validation_results = results
        
        if verbose:
            self._print_overall_summary(results)
            
        return results
    
    def _validate_on_regime(self, data: Any, num_episodes: int, render: bool, 
                           verbose: bool) -> Dict[str, float]:
        """Validate model on a specific market regime"""
        episode_returns = []
        sharpe_ratios = []
        win_rates = []
        max_drawdowns = []
        profit_factors = []
        calmar_ratios = []
        sortino_ratios = []
        var_95_values = []
        cvar_95_values = []
        
        for episode in range(num_episodes):
            if verbose and episode % max(1, num_episodes // 10) == 0:
                print(f"  Episode {episode + 1}/{num_episodes}")
                
            # Create environment with this regime's data
            env = self.env_class(data, initial_balance=self.initial_balance, 
                               commission=self.commission)
            
            # Run episode
            episode_result = self._run_episode(env, render)
            
            # Extract metrics
            final_return = episode_result['final_return']
            episode_returns.append(final_return)
            sharpe_ratios.append(episode_result['sharpe_ratio'])
            win_rates.append(1 if final_return > 0 else 0)
            max_drawdowns.append(episode_result['max_drawdown'])
            profit_factors.append(episode_result['profit_factor'])
            calmar_ratios.append(episode_result['calmar_ratio'])
            sortino_ratios.append(episode_result['sortino_ratio'])
            var_95_values.append(episode_result['var_95'])
            cvar_95_values.append(episode_result['cvar_95'])
        
        # Calculate statistics
        return {
            'avg_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'win_rate': np.mean(win_rates) * 100,
            'max_drawdown': np.mean(max_drawdowns),
            'profit_factor': np.mean(profit_factors),
            'calmar_ratio': np.mean(calmar_ratios),
            'sortino_ratio': np.mean(sortino_ratios),
            'var_95': np.mean(var_95_values),
            'cvar_95': np.mean(cvar_95_values),
            'all_returns': episode_returns,
            'all_sharpe_ratios': sharpe_ratios,
            'all_drawdowns': max_drawdowns
        }
    
    def _run_episode(self, env: Any, render: bool) -> Dict[str, float]:
        """Run a single episode and return performance metrics"""
        state = env.reset()
        hidden_state = None
        done = False
        
        # Track portfolio values for metrics calculation
        portfolio_values = [env.portfolio_value]
        returns = []
        
        while not done:
            # Get action from model
            action, hidden_state = self.model.get_action(state, hidden_state, epsilon=0.0)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            state = next_state
            
            # Track portfolio value
            portfolio_values.append(env.portfolio_value)
            
            if render:
                env.render()
        
        # Calculate performance metrics
        final_return = (env.portfolio_value - self.initial_balance) / self.initial_balance * 100
        
        # Calculate daily returns
        portfolio_values = np.array(portfolio_values)
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # Profit factor
        profit_factor = self._calculate_profit_factor(daily_returns)
        
        # Calmar ratio
        calmar_ratio = final_return / max_drawdown if max_drawdown > 0 else 0
        
        # Sortino ratio
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        
        # Value at Risk (95%)
        var_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
        
        # Conditional Value at Risk (95%)
        cvar_95 = np.mean(daily_returns[daily_returns <= var_95]) if len(daily_returns) > 0 else 0
        
        return {
            'final_return': final_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return abs(np.min(drawdown)) * 100
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor"""
        if len(returns) == 0:
            return 0.0
        
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    
    def _print_regime_summary(self, regime_name: str, results: Dict[str, float]):
        """Print summary for a specific regime"""
        print(f"  Average Return: {results['avg_return']:.2f}%")
        print(f"  Return Std Dev: {results['std_return']:.2f}%")
        print(f"  Sharpe Ratio: {results['avg_sharpe']:.2f}")
        print(f"  Win Rate: {results['win_rate']:.2f}%")
        print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print(f"  Calmar Ratio: {results['calmar_ratio']:.2f}")
        print(f"  Sortino Ratio: {results['sortino_ratio']:.2f}")
        print(f"  VaR (95%): {results['var_95']:.2f}%")
        print(f"  CVaR (95%): {results['cvar_95']:.2f}%")
    
    def _print_overall_summary(self, results: Dict[str, Dict[str, float]]):
        """Print overall validation summary"""
        print("=" * 60)
        print("OVERALL VALIDATION SUMMARY")
        print("=" * 60)
        
        # Aggregate all returns
        all_returns = []
        for regime_results in results.values():
            all_returns.extend(regime_results['all_returns'])
        
        if all_returns:
            overall_return = np.mean(all_returns)
            overall_std = np.std(all_returns)
            overall_sharpe = overall_return / overall_std if overall_std > 0 else 0
            overall_win_rate = np.mean([1 if r > 0 else 0 for r in all_returns]) * 100
            
            print(f"Overall Average Return: {overall_return:.2f}%")
            print(f"Overall Return Std Dev: {overall_std:.2f}%")
            print(f"Overall Sharpe Ratio: {overall_sharpe:.2f}")
            print(f"Overall Win Rate: {overall_win_rate:.2f}%")
            
            # Risk-adjusted return
            if overall_std > 0:
                risk_adjusted = overall_return / overall_std
                print(f"Risk-Adjusted Return: {risk_adjusted:.2f}")
        
        # Best and worst performing regimes
        regime_returns = {name: results['avg_return'] for name, results in results.items()}
        best_regime = max(regime_returns, key=regime_returns.get)
        worst_regime = min(regime_returns, key=regime_returns.get)
        
        print(f"\nBest Performing Regime: {best_regime} ({regime_returns[best_regime]:.2f}%)")
        print(f"Worst Performing Regime: {worst_regime} ({regime_returns[worst_regime]:.2f}%)")
    
    def generate_validation_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive validation report
        
        Args:
            save_path: Path to save the report (optional)
            
        Returns:
            Report text
        """
        if not self.validation_results:
            return "No validation results available. Run validation first."
        
        report = []
        report.append("=" * 80)
        report.append("TRADING MODEL VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Initial Balance: ${self.initial_balance:,.2f}")
        report.append(f"Commission Rate: {self.commission:.3f}")
        report.append("")
        
        # Individual regime results
        for regime, results in self.validation_results.items():
            report.append(f"{regime.upper()} REGIME:")
            report.append("-" * 40)
            report.append(f"  Average Return: {results['avg_return']:.2f}%")
            report.append(f"  Return Std Dev: {results['std_return']:.2f}%")
            report.append(f"  Sharpe Ratio: {results['avg_sharpe']:.2f}")
            report.append(f"  Win Rate: {results['win_rate']:.2f}%")
            report.append(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
            report.append(f"  Profit Factor: {results['profit_factor']:.2f}")
            report.append(f"  Calmar Ratio: {results['calmar_ratio']:.2f}")
            report.append(f"  Sortino Ratio: {results['sortino_ratio']:.2f}")
            report.append(f"  VaR (95%): {results['var_95']:.2f}%")
            report.append(f"  CVaR (95%): {results['cvar_95']:.2f}%")
            report.append("")
        
        # Overall performance
        all_returns = []
        for results in self.validation_results.values():
            all_returns.extend(results['all_returns'])
            
        if all_returns:
            overall_return = np.mean(all_returns)
            overall_std = np.std(all_returns)
            overall_sharpe = overall_return / overall_std if overall_std > 0 else 0
            overall_win_rate = np.mean([1 if r > 0 else 0 for r in all_returns]) * 100
            
            report.append("OVERALL PERFORMANCE:")
            report.append("-" * 40)
            report.append(f"  Average Return: {overall_return:.2f}%")
            report.append(f"  Return Std Dev: {overall_std:.2f}%")
            report.append(f"  Sharpe Ratio: {overall_sharpe:.2f}")
            report.append(f"  Win Rate: {overall_win_rate:.2f}%")
            
            if overall_std > 0:
                risk_adjusted = overall_return / overall_std
                report.append(f"  Risk-Adjusted Return: {risk_adjusted:.2f}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Validation report saved to {save_path}")
        
        return report_text
    
    def plot_validation_results(self, save_path: Optional[str] = None):
        """
        Create visualization of validation results
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.validation_results:
            print("No validation results available. Run validation first.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Trading Model Validation Results', fontsize=16)
        
        # Extract data
        regimes = list(self.validation_results.keys())
        returns = [results['avg_return'] for results in self.validation_results.values()]
        sharpe_ratios = [results['avg_sharpe'] for results in self.validation_results.values()]
        win_rates = [results['win_rate'] for results in self.validation_results.values()]
        drawdowns = [results['max_drawdown'] for results in self.validation_results.values()]
        profit_factors = [results['profit_factor'] for results in self.validation_results.values()]
        calmar_ratios = [results['calmar_ratio'] for results in self.validation_results.values()]
        
        # Plot returns by regime
        axes[0, 0].bar(regimes, returns, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Average Returns by Market Regime')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot Sharpe ratios by regime
        axes[0, 1].bar(regimes, sharpe_ratios, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Sharpe Ratios by Market Regime')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot win rates by regime
        axes[0, 2].bar(regimes, win_rates, color='lightcoral', alpha=0.7)
        axes[0, 2].set_title('Win Rates by Market Regime')
        axes[0, 2].set_ylabel('Win Rate (%)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot drawdowns by regime
        axes[1, 0].bar(regimes, drawdowns, color='orange', alpha=0.7)
        axes[1, 0].set_title('Max Drawdown by Market Regime')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot profit factors by regime
        axes[1, 1].bar(regimes, profit_factors, color='purple', alpha=0.7)
        axes[1, 1].set_title('Profit Factors by Market Regime')
        axes[1, 1].set_ylabel('Profit Factor')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot Calmar ratios by regime
        axes[1, 2].bar(regimes, calmar_ratios, color='brown', alpha=0.7)
        axes[1, 2].set_title('Calmar Ratios by Market Regime')
        axes[1, 2].set_ylabel('Calmar Ratio')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Validation plot saved to {save_path}")
        
        plt.show()
    
    def get_best_regime(self) -> Tuple[str, Dict[str, float]]:
        """Get the best performing regime based on Sharpe ratio"""
        if not self.validation_results:
            return None, None
        
        best_regime = max(self.validation_results.keys(), 
                         key=lambda x: self.validation_results[x]['avg_sharpe'])
        
        return best_regime, self.validation_results[best_regime]
    
    def get_worst_regime(self) -> Tuple[str, Dict[str, float]]:
        """Get the worst performing regime based on Sharpe ratio"""
        if not self.validation_results:
            return None, None
        
        worst_regime = min(self.validation_results.keys(), 
                          key=lambda x: self.validation_results[x]['avg_sharpe'])
        
        return worst_regime, self.validation_results[worst_regime]


# Example usage and testing
if __name__ == "__main__":
    # Mock model and environment for testing
    class MockModel:
        def get_action(self, state, hidden_state=None, epsilon=0.0):
            return np.random.randint(3), None
    
    class MockEnvironment:
        def __init__(self, data, initial_balance=10000, commission=0.001):
            self.portfolio_value = initial_balance
            self.initial_balance = initial_balance
            self.commission = commission
            self.step_count = 0
            self.max_steps = 100
        
        def reset(self):
            self.step_count = 0
            return np.random.randn(20)
        
        def step(self, action):
            self.step_count += 1
            
            # Simulate portfolio value changes
            change = np.random.randn() * 0.01
            self.portfolio_value *= (1 + change)
            
            reward = change * 1000
            done = self.step_count >= self.max_steps
            
            return np.random.randn(20), reward, done, {}
    
    # Test the validation framework
    print("Testing Trading Validation Framework...")
    
    # Create mock data sources
    data_sources = {
        'bull_market': np.random.randn(1000),
        'bear_market': np.random.randn(1000) * 0.5,
        'sideways_market': np.random.randn(1000) * 0.1,
        'volatile_market': np.random.randn(1000) * 2.0
    }
    
    # Create validation framework
    model = MockModel()
    framework = TradingValidationFramework(model, MockEnvironment, data_sources)
    
    # Run validation
    results = framework.run_validation(num_episodes=5, verbose=True)
    
    # Generate report
    report = framework.generate_validation_report()
    print("\n" + report)
    
    # Get best and worst regimes
    best_regime, best_results = framework.get_best_regime()
    worst_regime, worst_results = framework.get_worst_regime()
    
    print(f"\nBest Regime: {best_regime}")
    print(f"Worst Regime: {worst_regime}")
    
    print("Trading Validation Framework test completed successfully!")
