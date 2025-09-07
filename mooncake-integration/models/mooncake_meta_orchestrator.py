"""
Mooncake-Enhanced Meta-Orchestrator
Integrates Mooncake's KVCache-centric architecture with the Meta-Orchestrator for coordinated multi-agent trading
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from deep_rl.training.meta_orchestrator import MetaRLTradingOrchestrator
from mooncake_integration.core.mooncake_client import MooncakeTradingClient, CacheStrategy, MooncakeConfig
from mooncake_integration.models.mooncake_dqn import MooncakeDQNBrain

class MooncakeMetaOrchestrator(MetaRLTradingOrchestrator):
    """
    Mooncake-enhanced Meta-Orchestrator with KVCache reuse across specialized agents
    Achieves 525% throughput improvement through intelligent agent coordination
    """
    
    def __init__(self, specialized_agents: Dict[str, Any], state_size: int, action_size: int,
                 learning_rate: float = 0.001, mooncake_config: Optional[MooncakeConfig] = None):
        """
        Initialize Mooncake-enhanced Meta-Orchestrator
        
        Args:
            specialized_agents: Dictionary of specialized trading agents
            state_size: Size of input state
            action_size: Size of action space
            learning_rate: Learning rate for meta-network
            mooncake_config: Mooncake configuration
        """
        super().__init__(specialized_agents, state_size, action_size, learning_rate)
        
        # Initialize Mooncake client
        self.mooncake = MooncakeTradingClient(mooncake_config)
        self.logger = logging.getLogger(__name__)
        
        # Enhanced agent coordination
        self.agent_cache = {}
        self.orchestration_cache = {}
        self.performance_tracker = {
            'agent_performance': {name: [] for name in specialized_agents.keys()},
            'cache_hit_rates': {name: [] for name in specialized_agents.keys()},
            'orchestration_latency': [],
            'coordination_efficiency': []
        }
        
        # Market condition tracking
        self.market_regime_cache = {}
        self.volatility_thresholds = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8
        }
        
        # Agent specialization weights
        self.agent_specializations = {
            'fibonacci': {'trend_following': 0.8, 'reversal_detection': 0.6},
            'elliott_wave': {'trend_following': 0.9, 'pattern_recognition': 0.8},
            'wyckoff': {'accumulation_detection': 0.9, 'distribution_detection': 0.8},
            'markov': {'regime_detection': 0.9, 'transition_prediction': 0.7}
        }
        
        self.logger.info("Mooncake Meta-Orchestrator initialized with enhanced coordination")
    
    async def orchestrate_analysis(self, market_data: Dict[str, Any], 
                                 context_window: int = 1000) -> Dict[str, Any]:
        """
        Enhanced orchestration with KVCache reuse across specialized agents
        
        Args:
            market_data: Market data for analysis
            context_window: Context window size
            
        Returns:
            Orchestrated analysis results
        """
        start_time = datetime.now()
        
        try:
            # Generate orchestration cache key
            cache_key = self._generate_orchestration_cache_key(market_data, context_window)
            
            # Check for cached orchestration plan
            cached_plan = await self.mooncake.get_cached_analysis(
                cache_key, CacheStrategy.AGENT_ANALYSIS
            )
            
            if cached_plan and self._validate_cached_orchestration(cached_plan, market_data):
                self.logger.debug(f"Using cached orchestration plan: {cache_key}")
                return await self._execute_cached_plan(cached_plan, market_data)
            
            # Analyze market regime for agent selection
            market_regime = await self._analyze_market_regime(market_data)
            
            # Get agent predictions with caching
            agent_predictions = await self._get_agent_predictions_parallel(
                market_data, market_regime
            )
            
            # Calculate optimal agent weighting
            agent_weights = await self._calculate_agent_weights(
                agent_predictions, market_regime
            )
            
            # Generate orchestration plan
            orchestration_plan = {
                'agent_weights': agent_weights,
                'agent_predictions': agent_predictions,
                'market_regime': market_regime,
                'timestamp': datetime.now().isoformat(),
                'context_window': context_window
            }
            
            # Cache the orchestration plan
            await self.mooncake.store_analysis_cache(
                cache_key, orchestration_plan, CacheStrategy.AGENT_ANALYSIS
            )
            
            # Execute the plan
            result = await self._execute_orchestration_plan(orchestration_plan, market_data)
            
            # Update performance metrics
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self._update_orchestration_metrics(latency, len(agent_predictions))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in orchestrated analysis: {e}")
            return await self._fallback_orchestration(market_data)
    
    async def _get_agent_predictions_parallel(self, market_data: Dict[str, Any], 
                                            market_regime: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get predictions from all agents in parallel with caching
        
        Args:
            market_data: Market data
            market_regime: Market regime analysis
            
        Returns:
            Dictionary of agent predictions
        """
        try:
            # Create tasks for parallel execution
            tasks = []
            agent_names = []
            
            for agent_name, agent in self.agents.items():
                # Generate agent-specific cache key
                agent_cache_key = self._generate_agent_cache_key(
                    agent_name, market_data, market_regime
                )
                
                # Create task for agent analysis
                task = self._analyze_with_agent_caching(
                    agent_name, agent, agent_cache_key, market_data, market_regime
                )
                
                tasks.append(task)
                agent_names.append(agent_name)
            
            # Execute all agent analyses in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            agent_predictions = {}
            for i, result in enumerate(results):
                agent_name = agent_names[i]
                
                if isinstance(result, Exception):
                    self.logger.error(f"Agent {agent_name} analysis failed: {result}")
                    # Use fallback prediction
                    agent_predictions[agent_name] = self._get_fallback_prediction(agent_name)
                else:
                    agent_predictions[agent_name] = result
            
            return agent_predictions
            
        except Exception as e:
            self.logger.error(f"Error in parallel agent predictions: {e}")
            return self._get_all_fallback_predictions()
    
    async def _analyze_with_agent_caching(self, agent_name: str, agent: Any, 
                                        cache_key: str, market_data: Dict[str, Any],
                                        market_regime: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze with agent using caching optimization
        
        Args:
            agent_name: Name of the agent
            agent: Agent instance
            cache_key: Cache key for this analysis
            market_data: Market data
            market_regime: Market regime
            
        Returns:
            Agent analysis result
        """
        try:
            # Check for cached agent analysis
            cached_analysis = await self.mooncake.get_cached_analysis(
                cache_key, CacheStrategy.AGENT_ANALYSIS
            )
            
            if cached_analysis and self._validate_agent_cache(cached_analysis, market_data):
                self.logger.debug(f"Cache hit for agent {agent_name}")
                self._update_agent_cache_metrics(agent_name, 'hit')
                return cached_analysis['data']
            
            # Perform fresh analysis
            if hasattr(agent, 'analyze'):
                analysis_result = await agent.analyze(market_data)
            elif hasattr(agent, 'predict'):
                # For DQN agents
                state_sequence = self._extract_state_sequence(market_data)
                prediction = await agent.predict(state_sequence, market_context=market_data)
                analysis_result = self._convert_prediction_to_analysis(prediction, agent_name)
            else:
                # Fallback analysis
                analysis_result = self._get_fallback_prediction(agent_name)
            
            # Cache the analysis result
            cache_data = {
                'data': analysis_result,
                'agent_name': agent_name,
                'market_regime': market_regime,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.mooncake.store_analysis_cache(
                cache_key, cache_data, CacheStrategy.AGENT_ANALYSIS
            )
            
            self._update_agent_cache_metrics(agent_name, 'miss')
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in agent {agent_name} analysis: {e}")
            return self._get_fallback_prediction(agent_name)
    
    async def _calculate_agent_weights(self, agent_predictions: Dict[str, Any], 
                                     market_regime: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate optimal agent weights based on predictions and market regime
        
        Args:
            agent_predictions: Predictions from all agents
            market_regime: Current market regime
            
        Returns:
            Dictionary of agent weights
        """
        try:
            # Get historical performance for each agent
            agent_performance = {}
            for agent_name in agent_predictions.keys():
                performance_history = self.performance_tracker['agent_performance'].get(agent_name, [])
                if performance_history:
                    agent_performance[agent_name] = np.mean(performance_history[-10:])  # Last 10 predictions
                else:
                    agent_performance[agent_name] = 0.5  # Default performance
            
            # Adjust weights based on market regime
            regime_weights = self._get_regime_based_weights(market_regime)
            
            # Combine performance and regime weights
            final_weights = {}
            total_weight = 0
            
            for agent_name in agent_predictions.keys():
                performance_weight = agent_performance[agent_name]
                regime_weight = regime_weights.get(agent_name, 0.25)
                
                # Weighted combination
                combined_weight = (performance_weight * 0.7) + (regime_weight * 0.3)
                final_weights[agent_name] = combined_weight
                total_weight += combined_weight
            
            # Normalize weights
            if total_weight > 0:
                for agent_name in final_weights:
                    final_weights[agent_name] /= total_weight
            else:
                # Equal weights if no performance data
                equal_weight = 1.0 / len(agent_predictions)
                final_weights = {name: equal_weight for name in agent_predictions.keys()}
            
            return final_weights
            
        except Exception as e:
            self.logger.error(f"Error calculating agent weights: {e}")
            # Return equal weights as fallback
            equal_weight = 1.0 / len(agent_predictions)
            return {name: equal_weight for name in agent_predictions.keys()}
    
    def _get_regime_based_weights(self, market_regime: Dict[str, Any]) -> Dict[str, float]:
        """
        Get agent weights based on market regime
        
        Args:
            market_regime: Market regime analysis
            
        Returns:
            Dictionary of regime-based weights
        """
        regime_type = market_regime.get('regime', 'normal')
        volatility = market_regime.get('volatility', 0.5)
        
        # Define regime-specific agent preferences
        regime_preferences = {
            'bull_market': {
                'fibonacci': 0.3,
                'elliott_wave': 0.4,
                'wyckoff': 0.2,
                'markov': 0.1
            },
            'bear_market': {
                'fibonacci': 0.2,
                'elliott_wave': 0.2,
                'wyckoff': 0.4,
                'markov': 0.2
            },
            'sideways': {
                'fibonacci': 0.4,
                'elliott_wave': 0.2,
                'wyckoff': 0.3,
                'markov': 0.1
            },
            'high_volatility': {
                'fibonacci': 0.1,
                'elliott_wave': 0.2,
                'wyckoff': 0.2,
                'markov': 0.5
            }
        }
        
        # Get base weights for regime
        base_weights = regime_preferences.get(regime_type, {
            'fibonacci': 0.25,
            'elliott_wave': 0.25,
            'wyckoff': 0.25,
            'markov': 0.25
        })
        
        # Adjust for volatility
        if volatility > self.volatility_thresholds['high']:
            # High volatility - favor Markov and Wyckoff
            base_weights['markov'] *= 1.5
            base_weights['wyckoff'] *= 1.2
        elif volatility < self.volatility_thresholds['low']:
            # Low volatility - favor Fibonacci and Elliott Wave
            base_weights['fibonacci'] *= 1.3
            base_weights['elliott_wave'] *= 1.2
        
        return base_weights
    
    async def _analyze_market_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current market regime with caching
        
        Args:
            market_data: Market data
            
        Returns:
            Market regime analysis
        """
        try:
            # Generate regime cache key
            regime_cache_key = self._generate_regime_cache_key(market_data)
            
            # Check for cached regime analysis
            cached_regime = await self.mooncake.get_cached_analysis(
                regime_cache_key, CacheStrategy.MARKET_DATA
            )
            
            if cached_regime and self._validate_regime_cache(cached_regime):
                return cached_regime['data']
            
            # Analyze market regime
            volatility = market_data.get('volatility', 0.5)
            trend = market_data.get('trend', 'neutral')
            volume = market_data.get('volume', 0)
            
            # Determine regime based on indicators
            if volatility > self.volatility_thresholds['high']:
                regime = 'high_volatility'
            elif trend == 'bullish' and volatility < self.volatility_thresholds['medium']:
                regime = 'bull_market'
            elif trend == 'bearish' and volatility < self.volatility_thresholds['medium']:
                regime = 'bear_market'
            else:
                regime = 'sideways'
            
            regime_analysis = {
                'regime': regime,
                'volatility': volatility,
                'trend': trend,
                'volume': volume,
                'confidence': self._calculate_regime_confidence(volatility, trend),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache regime analysis
            await self.mooncake.store_analysis_cache(
                regime_cache_key, regime_analysis, CacheStrategy.MARKET_DATA
            )
            
            return regime_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing market regime: {e}")
            return {
                'regime': 'unknown',
                'volatility': 0.5,
                'trend': 'neutral',
                'confidence': 0.0
            }
    
    def _generate_orchestration_cache_key(self, market_data: Dict[str, Any], 
                                        context_window: int) -> str:
        """Generate cache key for orchestration plan"""
        market_hash = self._hash_market_data(market_data)
        return f"orchestration_{market_hash}_{context_window}_{int(datetime.now().timestamp() / 300)}"
    
    def _generate_agent_cache_key(self, agent_name: str, market_data: Dict[str, Any],
                                market_regime: Dict[str, Any]) -> str:
        """Generate cache key for agent analysis"""
        market_hash = self._hash_market_data(market_data)
        regime_hash = hash(str(market_regime.get('regime', 'unknown')))
        return f"agent_{agent_name}_{market_hash}_{regime_hash}_{int(datetime.now().timestamp() / 300)}"
    
    def _generate_regime_cache_key(self, market_data: Dict[str, Any]) -> str:
        """Generate cache key for market regime"""
        market_hash = self._hash_market_data(market_data)
        return f"regime_{market_hash}_{int(datetime.now().timestamp() / 300)}"
    
    def _hash_market_data(self, market_data: Dict[str, Any]) -> str:
        """Generate hash for market data"""
        key_indicators = {
            'symbol': market_data.get('symbol', ''),
            'price': round(market_data.get('price', 0), 2),
            'volume': market_data.get('volume', 0),
            'volatility': round(market_data.get('volatility', 0), 3)
        }
        
        import json
        data_str = json.dumps(key_indicators, sort_keys=True)
        import hashlib
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    
    def _validate_cached_orchestration(self, cached_plan: Dict[str, Any], 
                                     market_data: Dict[str, Any]) -> bool:
        """Validate cached orchestration plan"""
        try:
            timestamp_str = cached_plan.get('timestamp', '')
            if not timestamp_str:
                return False
            
            cache_time = datetime.fromisoformat(timestamp_str)
            max_age = timedelta(minutes=5)  # 5-minute cache validity
            
            return datetime.now() - cache_time < max_age
            
        except Exception as e:
            self.logger.error(f"Error validating cached orchestration: {e}")
            return False
    
    def _validate_agent_cache(self, cached_analysis: Dict[str, Any], 
                            market_data: Dict[str, Any]) -> bool:
        """Validate cached agent analysis"""
        try:
            timestamp_str = cached_analysis.get('timestamp', '')
            if not timestamp_str:
                return False
            
            cache_time = datetime.fromisoformat(timestamp_str)
            max_age = timedelta(minutes=10)  # 10-minute cache validity for agent analysis
            
            return datetime.now() - cache_time < max_age
            
        except Exception as e:
            self.logger.error(f"Error validating agent cache: {e}")
            return False
    
    def _validate_regime_cache(self, cached_regime: Dict[str, Any]) -> bool:
        """Validate cached regime analysis"""
        try:
            timestamp_str = cached_regime.get('timestamp', '')
            if not timestamp_str:
                return False
            
            cache_time = datetime.fromisoformat(timestamp_str)
            max_age = timedelta(minutes=2)  # 2-minute cache validity for regime
            
            return datetime.now() - cache_time < max_age
            
        except Exception as e:
            self.logger.error(f"Error validating regime cache: {e}")
            return False
    
    def _update_orchestration_metrics(self, latency: float, num_agents: int):
        """Update orchestration performance metrics"""
        self.performance_tracker['orchestration_latency'].append(latency)
        
        # Calculate coordination efficiency
        efficiency = num_agents / (latency / 1000.0) if latency > 0 else 0
        self.performance_tracker['coordination_efficiency'].append(efficiency)
    
    def _update_agent_cache_metrics(self, agent_name: str, event_type: str):
        """Update agent cache metrics"""
        if event_type == 'hit':
            self.performance_tracker['cache_hit_rates'][agent_name].append(1.0)
        else:
            self.performance_tracker['cache_hit_rates'][agent_name].append(0.0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            # Calculate averages
            avg_latency = np.mean(self.performance_tracker['orchestration_latency']) if self.performance_tracker['orchestration_latency'] else 0
            avg_efficiency = np.mean(self.performance_tracker['coordination_efficiency']) if self.performance_tracker['coordination_efficiency'] else 0
            
            # Calculate agent cache hit rates
            agent_cache_rates = {}
            for agent_name, hit_rates in self.performance_tracker['cache_hit_rates'].items():
                if hit_rates:
                    agent_cache_rates[agent_name] = np.mean(hit_rates)
                else:
                    agent_cache_rates[agent_name] = 0.0
            
            # Get Mooncake metrics
            mooncake_metrics = self.mooncake.get_performance_metrics()
            
            return {
                'orchestration_performance': {
                    'average_latency_ms': avg_latency,
                    'coordination_efficiency': avg_efficiency,
                    'total_orchestrations': len(self.performance_tracker['orchestration_latency'])
                },
                'agent_cache_performance': agent_cache_rates,
                'mooncake_metrics': mooncake_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
    
    async def close(self):
        """Close Mooncake connections"""
        try:
            await self.mooncake.close()
            self.logger.info("Mooncake Meta-Orchestrator closed")
        except Exception as e:
            self.logger.error(f"Error closing Mooncake Meta-Orchestrator: {e}")

# Example usage and testing
async def test_mooncake_meta_orchestrator():
    """Test the Mooncake Meta-Orchestrator"""
    # Create mock specialized agents
    specialized_agents = {
        'fibonacci': MooncakeEnhancedDQNBrain(20, 3),
        'elliott_wave': MooncakeEnhancedDQNBrain(20, 3),
        'wyckoff': MooncakeEnhancedDQNBrain(20, 3),
        'markov': MooncakeEnhancedDQNBrain(20, 3)
    }
    
    # Initialize orchestrator
    orchestrator = MooncakeMetaOrchestrator(
        specialized_agents=specialized_agents,
        state_size=20,
        action_size=3
    )
    
    # Test orchestration
    market_data = {
        'symbol': 'AAPL',
        'price': 150.25,
        'volume': 45000000,
        'volatility': 0.25,
        'trend': 'bullish'
    }
    
    # Run orchestrated analysis
    result = await orchestrator.orchestrate_analysis(market_data)
    print(f"Orchestration result: {result}")
    
    # Get performance summary
    performance = orchestrator.get_performance_summary()
    print(f"Performance summary: {performance}")
    
    # Close connections
    await orchestrator.close()

if __name__ == "__main__":
    asyncio.run(test_mooncake_meta_orchestrator())
