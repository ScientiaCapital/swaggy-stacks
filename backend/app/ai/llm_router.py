"""
Intelligent LLM Router for Chinese Trading Models
Routes trading tasks to optimal Chinese LLMs based on performance tracking
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from app.ai.deepseek_trade_orchestrator import TaskType, LLMModel, TaskContext, TradingDecisionResult
from app.ai.ollama_client import OllamaClient


@dataclass
class PerformanceMetric:
    """Performance tracking for LLM-task combinations"""
    llm_model: str
    task_type: str
    success_rate: float
    avg_execution_time: float
    confidence_score: float
    total_executions: int
    last_updated: datetime
    rolling_success: deque  # Last 10 results for trend analysis
    
    def __post_init__(self):
        if isinstance(self.rolling_success, list):
            self.rolling_success = deque(self.rolling_success, maxlen=10)
        elif self.rolling_success is None:
            self.rolling_success = deque(maxlen=10)


@dataclass
class RoutingDecision:
    """Decision made by the router"""
    selected_llm: str
    confidence: float
    reasoning: str
    fallback_options: List[str]
    expected_performance: float
    estimated_time: float


class RoutingStrategy(Enum):
    """Different routing strategies"""
    PERFORMANCE_BASED = "performance"  # Route based on historical success rates
    LOAD_BALANCED = "balanced"        # Balance performance with system load
    ADAPTIVE = "adaptive"             # Learn and adapt routing over time
    ENSEMBLE = "ensemble"             # Use multiple models for critical decisions


class IntelligentLLMRouter:
    """
    Intelligent router that selects optimal Chinese LLMs for trading tasks
    based on historical performance, current system load, and task requirements
    """
    
    def __init__(
        self, 
        ollama_client: OllamaClient,
        performance_db_path: str = "llm_performance.json",
        strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE
    ):
        self.ollama_client = ollama_client
        self.performance_db_path = performance_db_path
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_metrics: Dict[str, PerformanceMetric] = {}
        self.load_balancer = LLMLoadBalancer()
        
        # Routing configuration
        self.routing_config = {
            "min_confidence_threshold": 0.6,
            "performance_weight": 0.6,
            "load_weight": 0.2,
            "freshness_weight": 0.2,
            "ensemble_threshold": 0.9,  # Use ensemble for high-stakes decisions
        }
        
        # Load historical performance data
        self._load_performance_data()
        
        # Initialize Chinese LLM capabilities matrix
        self.llm_capabilities = {
            LLMModel.DEEPSEEK_R1.value: {
                TaskType.HEDGE_FUND_ANALYSIS: 0.92,
                TaskType.PATTERN_RECOGNITION: 0.88,
                TaskType.RISK_ASSESSMENT: 0.85,
                TaskType.STRATEGY_OPTIMIZATION: 0.90,
                TaskType.BACKTEST_ANALYSIS: 0.87,
                TaskType.MARKET_SENTIMENT: 0.83,
                TaskType.PORTFOLIO_ALLOCATION: 0.89
            },
            LLMModel.QWEN_QUANT.value: {
                TaskType.HEDGE_FUND_ANALYSIS: 0.78,
                TaskType.PATTERN_RECOGNITION: 0.85,
                TaskType.RISK_ASSESSMENT: 0.90,
                TaskType.STRATEGY_OPTIMIZATION: 0.88,
                TaskType.BACKTEST_ANALYSIS: 0.92,
                TaskType.MARKET_SENTIMENT: 0.75,
                TaskType.PORTFOLIO_ALLOCATION: 0.87
            },
            LLMModel.YI_TECHNICAL.value: {
                TaskType.HEDGE_FUND_ANALYSIS: 0.72,
                TaskType.PATTERN_RECOGNITION: 0.94,
                TaskType.RISK_ASSESSMENT: 0.76,
                TaskType.STRATEGY_OPTIMIZATION: 0.82,
                TaskType.BACKTEST_ANALYSIS: 0.85,
                TaskType.MARKET_SENTIMENT: 0.88,
                TaskType.PORTFOLIO_ALLOCATION: 0.79
            },
            LLMModel.GLM_RISK.value: {
                TaskType.HEDGE_FUND_ANALYSIS: 0.85,
                TaskType.PATTERN_RECOGNITION: 0.78,
                TaskType.RISK_ASSESSMENT: 0.95,
                TaskType.STRATEGY_OPTIMIZATION: 0.83,
                TaskType.BACKTEST_ANALYSIS: 0.80,
                TaskType.MARKET_SENTIMENT: 0.82,
                TaskType.PORTFOLIO_ALLOCATION: 0.91
            },
            LLMModel.DEEPSEEK_CODER.value: {
                TaskType.HEDGE_FUND_ANALYSIS: 0.70,
                TaskType.PATTERN_RECOGNITION: 0.75,
                TaskType.RISK_ASSESSMENT: 0.72,
                TaskType.STRATEGY_OPTIMIZATION: 0.95,
                TaskType.BACKTEST_ANALYSIS: 0.88,
                TaskType.MARKET_SENTIMENT: 0.68,
                TaskType.PORTFOLIO_ALLOCATION: 0.85
            }
        }

    def route_task(
        self, 
        task_context: TaskContext,
        require_ensemble: bool = False
    ) -> RoutingDecision:
        """
        Route a task to the optimal LLM based on current strategy
        
        Args:
            task_context: The trading task context
            require_ensemble: Force ensemble routing for critical decisions
            
        Returns:
            RoutingDecision with selected LLM and metadata
        """
        if require_ensemble or self._should_use_ensemble(task_context):
            return self._route_ensemble(task_context)
            
        if self.strategy == RoutingStrategy.PERFORMANCE_BASED:
            return self._route_performance_based(task_context)
        elif self.strategy == RoutingStrategy.LOAD_BALANCED:
            return self._route_load_balanced(task_context)
        elif self.strategy == RoutingStrategy.ADAPTIVE:
            return self._route_adaptive(task_context)
        else:
            return self._route_performance_based(task_context)

    def _route_performance_based(self, task_context: TaskContext) -> RoutingDecision:
        """Route based purely on historical performance"""
        task_type = task_context.task_type
        
        # Get performance scores for each LLM
        llm_scores = []
        for llm_model in LLMModel:
            base_score = self.llm_capabilities.get(llm_model.value, {}).get(task_type, 0.5)
            
            # Adjust based on historical performance
            perf_key = f"{llm_model.value}_{task_type.value}"
            if perf_key in self.performance_metrics:
                metric = self.performance_metrics[perf_key]
                # Weighted combination of base capability and historical performance
                adjusted_score = (base_score * 0.4) + (metric.success_rate * 0.6)
                
                # Apply recency bonus for recently successful models
                recency_factor = self._calculate_recency_factor(metric.last_updated)
                adjusted_score *= recency_factor
                
                llm_scores.append((llm_model.value, adjusted_score, metric.avg_execution_time))
            else:
                # Use base score for new LLM-task combinations
                llm_scores.append((llm_model.value, base_score, 30.0))  # Assume 30s default
        
        # Sort by score descending
        llm_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_llm = llm_scores[0][0]
        confidence = llm_scores[0][1]
        expected_time = llm_scores[0][2]
        
        # Build fallback options
        fallback_options = [llm for llm, score, _ in llm_scores[1:4]]
        
        return RoutingDecision(
            selected_llm=selected_llm,
            confidence=confidence,
            reasoning=f"Performance-based routing: {selected_llm} has {confidence:.2f} success rate for {task_type.value}",
            fallback_options=fallback_options,
            expected_performance=confidence,
            estimated_time=expected_time
        )

    def _route_load_balanced(self, task_context: TaskContext) -> RoutingDecision:
        """Route considering both performance and current system load"""
        performance_decision = self._route_performance_based(task_context)
        
        # Check system load for top candidates
        candidates = [performance_decision.selected_llm] + performance_decision.fallback_options[:2]
        
        best_candidate = None
        best_score = 0
        
        for llm in candidates:
            load_factor = self.load_balancer.get_load_factor(llm)
            performance_score = self.llm_capabilities.get(llm, {}).get(task_context.task_type, 0.5)
            
            # Combined score: performance weighted by load
            combined_score = performance_score * (1.0 - load_factor * 0.3)
            
            if combined_score > best_score:
                best_score = combined_score
                best_candidate = llm
        
        # Update reasoning
        reasoning = f"Load-balanced routing: {best_candidate} selected with load factor consideration"
        
        return RoutingDecision(
            selected_llm=best_candidate or performance_decision.selected_llm,
            confidence=performance_decision.confidence * (1.0 - self.load_balancer.get_load_factor(best_candidate) * 0.2),
            reasoning=reasoning,
            fallback_options=performance_decision.fallback_options,
            expected_performance=best_score,
            estimated_time=performance_decision.estimated_time
        )

    def _route_adaptive(self, task_context: TaskContext) -> RoutingDecision:
        """Adaptive routing that learns from recent performance trends"""
        base_decision = self._route_performance_based(task_context)
        
        # Apply adaptive learning from recent trends
        for llm_model in LLMModel:
            perf_key = f"{llm_model.value}_{task_context.task_type.value}"
            if perf_key in self.performance_metrics:
                metric = self.performance_metrics[perf_key]
                
                # Analyze trend from rolling success history
                if len(metric.rolling_success) >= 5:
                    recent_trend = np.mean(list(metric.rolling_success)[-5:])
                    overall_average = metric.success_rate
                    
                    # Boost models showing upward trend
                    if recent_trend > overall_average + 0.1:
                        if llm_model.value == base_decision.selected_llm:
                            base_decision.confidence += 0.05
                        # Consider promoting trending models
                        elif recent_trend > 0.8 and llm_model.value in base_decision.fallback_options:
                            # Swap with selected if trend is very strong
                            if recent_trend > base_decision.confidence + 0.15:
                                old_selected = base_decision.selected_llm
                                base_decision.selected_llm = llm_model.value
                                base_decision.fallback_options[0] = old_selected
                                base_decision.reasoning += f" (Promoted {llm_model.value} due to positive trend)"
        
        base_decision.reasoning = "Adaptive " + base_decision.reasoning
        return base_decision

    def _route_ensemble(self, task_context: TaskContext) -> RoutingDecision:
        """Route to ensemble of top models for critical decisions"""
        performance_decision = self._route_performance_based(task_context)
        
        # Select top 3 models for ensemble
        top_models = [performance_decision.selected_llm] + performance_decision.fallback_options[:2]
        
        return RoutingDecision(
            selected_llm=f"ensemble_{'+'.join(top_models)}",
            confidence=min(performance_decision.confidence + 0.1, 0.95),
            reasoning=f"Ensemble routing: Using {len(top_models)} models for high-stakes decision",
            fallback_options=[performance_decision.selected_llm],
            expected_performance=performance_decision.expected_performance + 0.05,
            estimated_time=performance_decision.estimated_time * 1.5  # Ensemble takes longer
        )

    def update_performance(
        self, 
        llm_model: str,
        task_type: TaskType,
        success: bool,
        execution_time: float,
        confidence_score: float = None
    ):
        """Update performance metrics after task completion"""
        perf_key = f"{llm_model}_{task_type.value}"
        
        if perf_key not in self.performance_metrics:
            self.performance_metrics[perf_key] = PerformanceMetric(
                llm_model=llm_model,
                task_type=task_type.value,
                success_rate=0.0,
                avg_execution_time=0.0,
                confidence_score=0.0,
                total_executions=0,
                last_updated=datetime.utcnow(),
                rolling_success=deque(maxlen=10)
            )
        
        metric = self.performance_metrics[perf_key]
        
        # Update rolling success history
        metric.rolling_success.append(1 if success else 0)
        
        # Update success rate with exponential moving average
        alpha = 0.1  # Learning rate
        if metric.total_executions == 0:
            metric.success_rate = 1.0 if success else 0.0
        else:
            new_success_rate = 1.0 if success else 0.0
            metric.success_rate = (1 - alpha) * metric.success_rate + alpha * new_success_rate
        
        # Update execution time with moving average
        if metric.total_executions == 0:
            metric.avg_execution_time = execution_time
        else:
            metric.avg_execution_time = (1 - alpha) * metric.avg_execution_time + alpha * execution_time
        
        # Update confidence score
        if confidence_score is not None:
            if metric.total_executions == 0:
                metric.confidence_score = confidence_score
            else:
                metric.confidence_score = (1 - alpha) * metric.confidence_score + alpha * confidence_score
        
        metric.total_executions += 1
        metric.last_updated = datetime.utcnow()
        
        # Persist performance data
        self._save_performance_data()
        
        self.logger.info(f"Updated performance for {llm_model} on {task_type.value}: "
                        f"success_rate={metric.success_rate:.3f}, "
                        f"avg_time={metric.avg_execution_time:.1f}s")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_metrics": len(self.performance_metrics),
            "llm_performance": {},
            "task_performance": {},
            "top_performers": {}
        }
        
        # Aggregate by LLM
        llm_stats = defaultdict(list)
        task_stats = defaultdict(list)
        
        for perf_key, metric in self.performance_metrics.items():
            llm_stats[metric.llm_model].append(metric)
            task_stats[metric.task_type].append(metric)
        
        # LLM performance summary
        for llm, metrics in llm_stats.items():
            avg_success = np.mean([m.success_rate for m in metrics])
            avg_time = np.mean([m.avg_execution_time for m in metrics])
            total_executions = sum([m.total_executions for m in metrics])
            
            report["llm_performance"][llm] = {
                "avg_success_rate": round(avg_success, 3),
                "avg_execution_time": round(avg_time, 1),
                "total_executions": total_executions,
                "task_count": len(metrics)
            }
        
        # Task type performance summary
        for task_type, metrics in task_stats.items():
            best_llm = max(metrics, key=lambda m: m.success_rate)
            report["task_performance"][task_type] = {
                "best_llm": best_llm.llm_model,
                "best_success_rate": round(best_llm.success_rate, 3),
                "llm_count": len(metrics)
            }
        
        # Top performers overall
        if self.performance_metrics:
            sorted_metrics = sorted(
                self.performance_metrics.values(),
                key=lambda m: m.success_rate * m.total_executions,
                reverse=True
            )
            
            report["top_performers"] = {
                "overall_best": {
                    "llm": sorted_metrics[0].llm_model,
                    "task": sorted_metrics[0].task_type,
                    "success_rate": round(sorted_metrics[0].success_rate, 3),
                    "executions": sorted_metrics[0].total_executions
                }
            }
        
        return report

    def _should_use_ensemble(self, task_context: TaskContext) -> bool:
        """Determine if ensemble routing should be used"""
        # Use ensemble for high-stakes decisions
        high_stakes_conditions = [
            task_context.task_type in [TaskType.HEDGE_FUND_ANALYSIS, TaskType.RISK_ASSESSMENT],
            task_context.market_conditions and task_context.market_conditions.get("volatility", 0) > 0.8,
            task_context.portfolio_value and task_context.portfolio_value > 1000000,  # Large portfolios
        ]
        
        return any(high_stakes_conditions)

    def _calculate_recency_factor(self, last_updated: datetime) -> float:
        """Calculate factor to boost recently successful models"""
        days_ago = (datetime.utcnow() - last_updated).days
        # Decay factor: 1.0 for today, 0.95 for 1 week old, 0.9 for 2 weeks old
        return max(0.8, 1.0 - (days_ago * 0.01))

    def _load_performance_data(self):
        """Load performance metrics from persistent storage"""
        try:
            with open(self.performance_db_path, 'r') as f:
                data = json.load(f)
                for perf_key, metric_data in data.items():
                    # Convert datetime string back to datetime object
                    metric_data["last_updated"] = datetime.fromisoformat(metric_data["last_updated"])
                    # Convert rolling_success list back to deque
                    metric_data["rolling_success"] = deque(metric_data["rolling_success"], maxlen=10)
                    self.performance_metrics[perf_key] = PerformanceMetric(**metric_data)
            self.logger.info(f"Loaded {len(self.performance_metrics)} performance metrics")
        except FileNotFoundError:
            self.logger.info("No existing performance data found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading performance data: {e}")

    def _save_performance_data(self):
        """Save performance metrics to persistent storage"""
        try:
            data = {}
            for perf_key, metric in self.performance_metrics.items():
                metric_dict = asdict(metric)
                # Convert datetime to string for JSON serialization
                metric_dict["last_updated"] = metric.last_updated.isoformat()
                # Convert deque to list for JSON serialization
                metric_dict["rolling_success"] = list(metric.rolling_success)
                data[perf_key] = metric_dict
            
            with open(self.performance_db_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving performance data: {e}")


class LLMLoadBalancer:
    """Simple load balancer to track LLM usage and prevent overloading"""
    
    def __init__(self):
        self.active_requests: Dict[str, int] = defaultdict(int)
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def get_load_factor(self, llm_model: str) -> float:
        """Get current load factor (0.0 = no load, 1.0 = maximum load)"""
        active = self.active_requests[llm_model]
        recent_requests = len([
            req_time for req_time in self.request_history[llm_model]
            if datetime.utcnow() - req_time < timedelta(minutes=5)
        ])
        
        # Normalize load (assuming max 5 concurrent, 20 requests per 5 min)
        load_factor = min(1.0, (active / 5.0) + (recent_requests / 20.0))
        return load_factor
    
    def start_request(self, llm_model: str):
        """Mark start of request for load tracking"""
        self.active_requests[llm_model] += 1
        self.request_history[llm_model].append(datetime.utcnow())
    
    def end_request(self, llm_model: str):
        """Mark end of request for load tracking"""
        self.active_requests[llm_model] = max(0, self.active_requests[llm_model] - 1)