"""
Seven-Model Trading Orchestrator with Mooncake Integration
Revolutionary trading system that coordinates seven specialized AI models using Mooncake's KVCache architecture
"""

import asyncio
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import json
import hashlib
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from mooncake_integration.core.mooncake_client import MooncakeTradingClient, CacheStrategy, MooncakeConfig
from mooncake_integration.enhanced_models.mooncake_enhanced_dqn import MooncakeEnhancedDQNBrain
from mooncake_integration.enhanced_models.mooncake_meta_orchestrator import MooncakeMetaOrchestrator

class ModelSpecialization(Enum):
    """Specialized roles for each of the seven models"""
    DEEPSEEK = "mathematical_analysis"      # Complex calculations, options pricing
    YI = "cultural_sentiment"               # Social media, cultural context
    QWEN = "general_intelligence"           # Synthesis, reasoning
    CHATGLM = "financial_knowledge"         # Market expertise, regulations
    MINIMAX = "voice_generation"            # Audio explanations
    MOONSHOT = "pattern_recognition"        # Technical patterns, chart analysis
    INTERNLM2 = "stream_processing"         # Real-time data processing

@dataclass
class ModelCluster:
    """Configuration for each model cluster"""
    model_name: str
    specialization: ModelSpecialization
    optimization: str
    dedicated_memory_gb: int
    max_context_length: int
    processing_speed: str  # "ultra_fast", "fast", "standard"

@dataclass
class TradingQuery:
    """Structured trading query"""
    query_text: str
    symbol: str
    query_type: str  # "simple", "pattern_based", "mathematical", "comprehensive"
    user_tier: str   # "free", "basic", "premium", "enterprise"
    urgency: str     # "low", "medium", "high", "critical"
    timestamp: datetime

@dataclass
class ModelAnalysis:
    """Analysis result from a specific model"""
    model_name: str
    analysis_type: str
    result: Dict[str, Any]
    confidence: float
    processing_time_ms: float
    cache_hit: bool
    timestamp: datetime

class MooncakeMultiModelTradingSystem:
    """
    Revolutionary trading system that coordinates seven specialized AI models
    using Mooncake's KVCache-centric architecture for 525% throughput improvements
    """
    
    def __init__(self, mooncake_config: Optional[MooncakeConfig] = None):
        """
        Initialize the seven-model trading system
        
        Args:
            mooncake_config: Mooncake configuration
        """
        self.mooncake = MooncakeTradingClient(mooncake_config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model clusters with specialized configurations
        self.model_clusters = self._initialize_model_clusters()
        
        # Shared memory system using Mooncake's KVCache
        self.shared_memory = {
            'pattern_cache': {},
            'analysis_cache': {},
            'stream_cache': {},
            'user_context_cache': {}
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_queries': 0,
            'cache_hit_rate': 0.0,
            'average_response_time': 0.0,
            'model_usage_stats': {model: 0 for model in self.model_clusters.keys()},
            'cost_optimization_savings': 0.0
        }
        
        # Query routing intelligence
        self.query_classifier = QueryComplexityClassifier()
        self.cost_optimizer = CostOptimizationEngine()
        
        self.logger.info("Seven-Model Trading System initialized with Mooncake architecture")
    
    def _initialize_model_clusters(self) -> Dict[str, ModelCluster]:
        """Initialize specialized model clusters"""
        return {
            'deepseek': ModelCluster(
                model_name='deepseek',
                specialization=ModelSpecialization.DEEPSEEK,
                optimization='mathematical_precision',
                dedicated_memory_gb=16,
                max_context_length=128000,
                processing_speed='fast'
            ),
            'yi': ModelCluster(
                model_name='yi',
                specialization=ModelSpecialization.YI,
                optimization='cultural_understanding',
                dedicated_memory_gb=8,
                max_context_length=64000,
                processing_speed='standard'
            ),
            'qwen': ModelCluster(
                model_name='qwen',
                specialization=ModelSpecialization.QWEN,
                optimization='synthesis_reasoning',
                dedicated_memory_gb=12,
                max_context_length=128000,
                processing_speed='fast'
            ),
            'chatglm': ModelCluster(
                model_name='chatglm',
                specialization=ModelSpecialization.CHATGLM,
                optimization='financial_expertise',
                dedicated_memory_gb=10,
                max_context_length=64000,
                processing_speed='standard'
            ),
            'minimax': ModelCluster(
                model_name='minimax',
                specialization=ModelSpecialization.MINIMAX,
                optimization='voice_generation',
                dedicated_memory_gb=8,
                max_context_length=32000,
                processing_speed='standard'
            ),
            'moonshot': ModelCluster(
                model_name='moonshot',
                specialization=ModelSpecialization.MOONSHOT,
                optimization='pattern_recognition',
                dedicated_memory_gb=14,
                max_context_length=128000,
                processing_speed='ultra_fast'
            ),
            'internlm2': ModelCluster(
                model_name='internlm2',
                specialization=ModelSpecialization.INTERNLM2,
                optimization='ultra_low_latency',
                dedicated_memory_gb=16,
                max_context_length=128000,
                processing_speed='ultra_fast'
            )
        }
    
    async def process_trading_query(self, query: TradingQuery) -> Dict[str, Any]:
        """
        Process trading query using intelligent model orchestration
        
        Args:
            query: Trading query to process
            
        Returns:
            Comprehensive analysis result
        """
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._generate_query_cache_key(query)
            cached_result = await self._check_query_cache(cache_key)
            
            if cached_result and self._is_cache_valid(cached_result, query):
                self._update_cache_metrics('hit')
                self.logger.info(f"Cache hit for query: {query.query_text[:50]}...")
                return cached_result
            
            # Classify query complexity
            complexity = await self.query_classifier.classify_complexity(query)
            
            # Route to appropriate models based on complexity and user tier
            model_plan = await self._create_model_execution_plan(query, complexity)
            
            # Execute model analysis
            analysis_results = await self._execute_model_plan(model_plan, query)
            
            # Synthesize results
            final_result = await self._synthesize_analysis(analysis_results, query)
            
            # Cache the result
            await self._cache_query_result(cache_key, final_result, query)
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_metrics(processing_time, len(analysis_results))
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error processing trading query: {e}")
            return await self._fallback_analysis(query)
    
    async def _create_model_execution_plan(self, query: TradingQuery, 
                                         complexity: str) -> Dict[str, Any]:
        """
        Create intelligent execution plan for models based on query characteristics
        
        Args:
            query: Trading query
            complexity: Query complexity classification
            
        Returns:
            Model execution plan
        """
        plan = {
            'primary_models': [],
            'secondary_models': [],
            'parallel_execution': True,
            'cache_strategy': 'aggressive',
            'timeout_ms': 5000
        }
        
        # Route based on complexity and user tier
        if complexity == 'simple':
            plan['primary_models'] = ['qwen']
            plan['parallel_execution'] = False
            
        elif complexity == 'pattern_based':
            plan['primary_models'] = ['moonshot', 'qwen']
            plan['secondary_models'] = ['chatglm']
            
        elif complexity == 'mathematical':
            plan['primary_models'] = ['deepseek', 'qwen']
            plan['secondary_models'] = ['chatglm']
            
        elif complexity == 'comprehensive':
            if query.user_tier in ['premium', 'enterprise']:
                plan['primary_models'] = ['moonshot', 'deepseek', 'qwen']
                plan['secondary_models'] = ['yi', 'chatglm', 'internlm2']
                if query.urgency == 'critical':
                    plan['secondary_models'].append('minimax')
            else:
                plan['primary_models'] = ['moonshot', 'qwen']
                plan['secondary_models'] = ['chatglm']
        
        # Adjust for real-time queries
        if 'real-time' in query.query_text.lower() or query.urgency == 'critical':
            plan['primary_models'].insert(0, 'internlm2')
            plan['timeout_ms'] = 2000
        
        return plan
    
    async def _execute_model_plan(self, plan: Dict[str, Any], 
                                query: TradingQuery) -> List[ModelAnalysis]:
        """
        Execute model analysis plan with parallel processing
        
        Args:
            plan: Model execution plan
            query: Trading query
            
        Returns:
            List of model analysis results
        """
        try:
            # Create tasks for parallel execution
            tasks = []
            
            # Primary models (always executed)
            for model_name in plan['primary_models']:
                task = self._analyze_with_model(model_name, query)
                tasks.append(task)
            
            # Secondary models (executed if time permits)
            if plan['parallel_execution']:
                for model_name in plan['secondary_models']:
                    task = self._analyze_with_model(model_name, query)
                    tasks.append(task)
            
            # Execute with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=plan['timeout_ms'] / 1000.0
            )
            
            # Process results
            analysis_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Model analysis failed: {result}")
                    continue
                
                if result:
                    analysis_results.append(result)
            
            return analysis_results
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Model execution timeout for query: {query.query_text}")
            return []
        except Exception as e:
            self.logger.error(f"Error executing model plan: {e}")
            return []
    
    async def _analyze_with_model(self, model_name: str, 
                                query: TradingQuery) -> Optional[ModelAnalysis]:
        """
        Analyze query with specific model using caching optimization
        
        Args:
            model_name: Name of the model to use
            query: Trading query
            
        Returns:
            Model analysis result
        """
        start_time = datetime.now()
        
        try:
            # Generate model-specific cache key
            cache_key = self._generate_model_cache_key(model_name, query)
            
            # Check model cache
            cached_analysis = await self.mooncake.get_cached_analysis(
                cache_key, CacheStrategy.AGENT_ANALYSIS
            )
            
            if cached_analysis and self._validate_model_cache(cached_analysis):
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                return ModelAnalysis(
                    model_name=model_name,
                    analysis_type=cached_analysis['analysis_type'],
                    result=cached_analysis['result'],
                    confidence=cached_analysis['confidence'],
                    processing_time_ms=processing_time,
                    cache_hit=True,
                    timestamp=datetime.now()
                )
            
            # Perform fresh analysis
            analysis_result = await self._run_model_analysis(model_name, query)
            
            # Cache the result
            cache_data = {
                'analysis_type': analysis_result['type'],
                'result': analysis_result['data'],
                'confidence': analysis_result['confidence'],
                'timestamp': datetime.now().isoformat()
            }
            
            await self.mooncake.store_analysis_cache(
                cache_key, cache_data, CacheStrategy.AGENT_ANALYSIS
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ModelAnalysis(
                model_name=model_name,
                analysis_type=analysis_result['type'],
                result=analysis_result['data'],
                confidence=analysis_result['confidence'],
                processing_time_ms=processing_time,
                cache_hit=False,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing with model {model_name}: {e}")
            return None
    
    async def _run_model_analysis(self, model_name: str, 
                                query: TradingQuery) -> Dict[str, Any]:
        """
        Run analysis with specific model (mock implementation)
        
        Args:
            model_name: Name of the model
            query: Trading query
            
        Returns:
            Analysis result
        """
        # Mock implementation - replace with actual model calls
        model_cluster = self.model_clusters[model_name]
        
        # Simulate model-specific processing
        if model_name == 'deepseek':
            return {
                'type': 'mathematical_analysis',
                'data': {
                    'probability_calculation': 0.75,
                    'risk_metrics': {'var': 0.05, 'expected_return': 0.12},
                    'options_pricing': {'call_value': 2.50, 'put_value': 1.20}
                },
                'confidence': 0.85
            }
        
        elif model_name == 'moonshot':
            return {
                'type': 'pattern_analysis',
                'data': {
                    'pattern_type': 'bullish_flag',
                    'breakout_probability': 0.68,
                    'target_price': 155.00,
                    'support_levels': [148.50, 145.00]
                },
                'confidence': 0.78
            }
        
        elif model_name == 'qwen':
            return {
                'type': 'synthesis',
                'data': {
                    'recommendation': 'buy',
                    'reasoning': 'Strong technical pattern with favorable risk/reward ratio',
                    'confidence_level': 'high',
                    'time_horizon': '1-2 weeks'
                },
                'confidence': 0.82
            }
        
        elif model_name == 'yi':
            return {
                'type': 'sentiment_analysis',
                'data': {
                    'social_sentiment': 0.65,
                    'news_sentiment': 0.72,
                    'viral_potential': 0.45,
                    'cultural_context': 'positive'
                },
                'confidence': 0.70
            }
        
        elif model_name == 'chatglm':
            return {
                'type': 'financial_analysis',
                'data': {
                    'fundamental_rating': 'strong_buy',
                    'sector_outlook': 'positive',
                    'regulatory_environment': 'favorable',
                    'analyst_consensus': 'bullish'
                },
                'confidence': 0.80
            }
        
        elif model_name == 'internlm2':
            return {
                'type': 'stream_analysis',
                'data': {
                    'real_time_trend': 'bullish',
                    'volume_anomaly': 'high',
                    'price_momentum': 0.15,
                    'market_microstructure': 'favorable'
                },
                'confidence': 0.75
            }
        
        elif model_name == 'minimax':
            return {
                'type': 'voice_summary',
                'data': {
                    'audio_url': 'generated_audio_summary.mp3',
                    'transcript': 'Based on technical analysis, this appears to be a strong buying opportunity...',
                    'duration_seconds': 45,
                    'tone': 'professional'
                },
                'confidence': 0.90
            }
        
        else:
            return {
                'type': 'general_analysis',
                'data': {'message': 'Analysis completed'},
                'confidence': 0.50
            }
    
    async def _synthesize_analysis(self, analysis_results: List[ModelAnalysis], 
                                 query: TradingQuery) -> Dict[str, Any]:
        """
        Synthesize results from multiple models into final recommendation
        
        Args:
            analysis_results: List of model analysis results
            query: Original trading query
            
        Returns:
            Synthesized analysis result
        """
        try:
            # Weight results based on model confidence and specialization
            weighted_results = {}
            total_weight = 0
            
            for analysis in analysis_results:
                weight = analysis.confidence
                weighted_results[analysis.model_name] = {
                    'weight': weight,
                    'result': analysis.result,
                    'type': analysis.analysis_type
                }
                total_weight += weight
            
            # Normalize weights
            for model_name in weighted_results:
                weighted_results[model_name]['weight'] /= total_weight
            
            # Extract key insights
            synthesis = {
                'query': query.query_text,
                'symbol': query.symbol,
                'timestamp': datetime.now().isoformat(),
                'model_contributions': weighted_results,
                'final_recommendation': self._generate_final_recommendation(weighted_results),
                'confidence_score': self._calculate_overall_confidence(analysis_results),
                'processing_metadata': {
                    'models_used': len(analysis_results),
                    'cache_hit_rate': sum(1 for a in analysis_results if a.cache_hit) / len(analysis_results),
                    'total_processing_time': sum(a.processing_time_ms for a in analysis_results)
                }
            }
            
            return synthesis
            
        except Exception as e:
            self.logger.error(f"Error synthesizing analysis: {e}")
            return {
                'error': 'Synthesis failed',
                'query': query.query_text,
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_final_recommendation(self, weighted_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final trading recommendation from weighted model results"""
        # Extract recommendations from each model
        recommendations = []
        
        for model_name, data in weighted_results.items():
            result = data['result']
            weight = data['weight']
            
            if 'recommendation' in result:
                recommendations.append({
                    'model': model_name,
                    'recommendation': result['recommendation'],
                    'weight': weight
                })
        
        # Calculate weighted recommendation
        if recommendations:
            buy_weight = sum(r['weight'] for r in recommendations if r['recommendation'] == 'buy')
            sell_weight = sum(r['weight'] for r in recommendations if r['recommendation'] == 'sell')
            hold_weight = sum(r['weight'] for r in recommendations if r['recommendation'] == 'hold')
            
            if buy_weight > sell_weight and buy_weight > hold_weight:
                final_action = 'buy'
                confidence = buy_weight
            elif sell_weight > buy_weight and sell_weight > hold_weight:
                final_action = 'sell'
                confidence = sell_weight
            else:
                final_action = 'hold'
                confidence = hold_weight
        else:
            final_action = 'hold'
            confidence = 0.5
        
        return {
            'action': final_action,
            'confidence': confidence,
            'reasoning': 'Based on multi-model analysis',
            'risk_level': 'medium' if confidence > 0.7 else 'high'
        }
    
    def _calculate_overall_confidence(self, analysis_results: List[ModelAnalysis]) -> float:
        """Calculate overall confidence score from model results"""
        if not analysis_results:
            return 0.0
        
        # Weight by model confidence and processing time
        total_confidence = 0.0
        total_weight = 0.0
        
        for analysis in analysis_results:
            # Weight by confidence and inverse processing time (faster = more weight)
            weight = analysis.confidence * (1000.0 / max(analysis.processing_time_ms, 1.0))
            total_confidence += analysis.confidence * weight
            total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.0
    
    # Cache management methods
    def _generate_query_cache_key(self, query: TradingQuery) -> str:
        """Generate cache key for query"""
        query_hash = hashlib.md5(query.query_text.encode()).hexdigest()[:16]
        return f"query_{query_hash}_{query.symbol}_{query.user_tier}"
    
    def _generate_model_cache_key(self, model_name: str, query: TradingQuery) -> str:
        """Generate cache key for model analysis"""
        query_hash = hashlib.md5(query.query_text.encode()).hexdigest()[:16]
        return f"model_{model_name}_{query_hash}_{query.symbol}"
    
    async def _check_query_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check query cache"""
        return await self.mooncake.get_cached_analysis(cache_key, CacheStrategy.AGENT_ANALYSIS)
    
    def _is_cache_valid(self, cached_result: Dict[str, Any], query: TradingQuery) -> bool:
        """Check if cached result is still valid"""
        try:
            timestamp_str = cached_result.get('timestamp', '')
            if not timestamp_str:
                return False
            
            cache_time = datetime.fromisoformat(timestamp_str)
            
            # Different TTL based on query type
            if query.query_type == 'comprehensive':
                max_age = timedelta(minutes=5)
            elif query.query_type == 'mathematical':
                max_age = timedelta(minutes=2)
            else:
                max_age = timedelta(minutes=10)
            
            return datetime.now() - cache_time < max_age
            
        except Exception as e:
            self.logger.error(f"Error validating cache: {e}")
            return False
    
    async def _cache_query_result(self, cache_key: str, result: Dict[str, Any], 
                                query: TradingQuery):
        """Cache query result"""
        cache_data = {
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'query_type': query.query_type,
            'user_tier': query.user_tier
        }
        
        await self.mooncake.store_analysis_cache(
            cache_key, cache_data, CacheStrategy.AGENT_ANALYSIS
        )
    
    def _validate_model_cache(self, cached_analysis: Dict[str, Any]) -> bool:
        """Validate model cache"""
        try:
            timestamp_str = cached_analysis.get('timestamp', '')
            if not timestamp_str:
                return False
            
            cache_time = datetime.fromisoformat(timestamp_str)
            max_age = timedelta(minutes=15)  # 15-minute validity for model analysis
            
            return datetime.now() - cache_time < max_age
            
        except Exception as e:
            self.logger.error(f"Error validating model cache: {e}")
            return False
    
    # Performance tracking methods
    def _update_cache_metrics(self, event_type: str):
        """Update cache metrics"""
        if event_type == 'hit':
            self.performance_metrics['cache_hit_rate'] = min(1.0, 
                self.performance_metrics['cache_hit_rate'] + 0.01)
    
    def _update_performance_metrics(self, processing_time: float, num_models: int):
        """Update performance metrics"""
        self.performance_metrics['total_queries'] += 1
        
        # Update average response time
        current_avg = self.performance_metrics['average_response_time']
        total_queries = self.performance_metrics['total_queries']
        
        self.performance_metrics['average_response_time'] = (
            (current_avg * (total_queries - 1) + processing_time) / total_queries
        )
    
    async def _fallback_analysis(self, query: TradingQuery) -> Dict[str, Any]:
        """Fallback analysis when main system fails"""
        return {
            'query': query.query_text,
            'symbol': query.symbol,
            'timestamp': datetime.now().isoformat(),
            'final_recommendation': {
                'action': 'hold',
                'confidence': 0.3,
                'reasoning': 'System temporarily unavailable',
                'risk_level': 'high'
            },
            'confidence_score': 0.3,
            'processing_metadata': {
                'models_used': 0,
                'cache_hit_rate': 0.0,
                'total_processing_time': 0.0,
                'fallback_mode': True
            }
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'system_performance': self.performance_metrics,
            'mooncake_metrics': self.mooncake.get_performance_metrics(),
            'model_utilization': {
                model: count / max(self.performance_metrics['total_queries'], 1)
                for model, count in self.performance_metrics['model_usage_stats'].items()
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def close(self):
        """Close system connections"""
        try:
            await self.mooncake.close()
            self.logger.info("Seven-Model Trading System closed")
        except Exception as e:
            self.logger.error(f"Error closing system: {e}")

# Supporting classes
class QueryComplexityClassifier:
    """Classifies query complexity for intelligent routing"""
    
    def __init__(self):
        self.complexity_keywords = {
            'simple': ['price', 'current', 'what is'],
            'pattern_based': ['pattern', 'chart', 'technical', 'support', 'resistance'],
            'mathematical': ['calculate', 'probability', 'options', 'greeks', 'volatility'],
            'comprehensive': ['analyze', 'recommendation', 'should I', 'buy or sell']
        }
    
    async def classify_complexity(self, query: TradingQuery) -> str:
        """Classify query complexity"""
        query_lower = query.query_text.lower()
        
        # Check for comprehensive keywords first
        for keyword in self.complexity_keywords['comprehensive']:
            if keyword in query_lower:
                return 'comprehensive'
        
        # Check for mathematical keywords
        for keyword in self.complexity_keywords['mathematical']:
            if keyword in query_lower:
                return 'mathematical'
        
        # Check for pattern-based keywords
        for keyword in self.complexity_keywords['pattern_based']:
            if keyword in query_lower:
                return 'pattern_based'
        
        # Default to simple
        return 'simple'

class CostOptimizationEngine:
    """Optimizes costs through intelligent model routing"""
    
    def __init__(self):
        self.model_costs = {
            'deepseek': 0.10,  # per request
            'yi': 0.05,
            'qwen': 0.08,
            'chatglm': 0.06,
            'minimax': 0.12,
            'moonshot': 0.07,
            'internlm2': 0.09
        }
    
    def calculate_optimization_savings(self, plan: Dict[str, Any]) -> float:
        """Calculate cost savings from optimization"""
        # Mock implementation
        return 0.0

# Example usage and testing
async def test_seven_model_system():
    """Test the seven-model trading system"""
    # Initialize system
    system = MooncakeMultiModelTradingSystem()
    
    # Create test query
    query = TradingQuery(
        query_text="Should I buy Tesla calls based on the new factory announcement?",
        symbol="TSLA",
        query_type="comprehensive",
        user_tier="premium",
        urgency="high",
        timestamp=datetime.now()
    )
    
    # Process query
    result = await system.process_trading_query(query)
    print(f"Analysis result: {json.dumps(result, indent=2, default=str)}")
    
    # Get performance summary
    performance = system.get_performance_summary()
    print(f"Performance summary: {json.dumps(performance, indent=2, default=str)}")
    
    # Close system
    await system.close()

if __name__ == "__main__":
    asyncio.run(test_seven_model_system())
