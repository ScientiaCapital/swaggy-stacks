"""
Trading Intelligence Hub
Central coordinator for Chinese LLM routing and trading decision making
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
import json

from app.ai.deepseek_trade_orchestrator import DeepSeekTradeOrchestrator, TaskType, TaskContext, TradingDecisionResult
from app.ai.llm_router import IntelligentLLMRouter, RoutingStrategy, PerformanceMetric
from app.ai.ollama_client import OllamaClient


@dataclass
class TradingIntelligenceRequest:
    """Request structure for trading intelligence"""
    task_type: TaskType
    symbol: str
    market_data: Dict[str, Any]
    portfolio_context: Optional[Dict[str, Any]] = None
    risk_parameters: Optional[Dict[str, Any]] = None
    time_horizon: Optional[str] = "short"  # short, medium, long
    priority: str = "normal"  # low, normal, high, critical
    custom_context: Optional[Dict[str, Any]] = None


@dataclass
class TradingIntelligenceResponse:
    """Response from trading intelligence system"""
    decision: TradingDecisionResult
    routing_info: Dict[str, Any]
    execution_metrics: Dict[str, Any]
    confidence_factors: Dict[str, float]
    alternative_scenarios: Optional[List[TradingDecisionResult]] = None


class TradingIntelligenceHub:
    """
    Central hub coordinating Chinese LLM routing for trading decisions
    Combines DeepSeek orchestrator with intelligent routing for optimal performance
    """
    
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        routing_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
        enable_ensemble_mode: bool = True,
        performance_db_path: str = "trading_intelligence_performance.json"
    ):
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.ollama_client = OllamaClient(base_url=ollama_base_url)
        self.router = IntelligentLLMRouter(
            ollama_client=self.ollama_client,
            performance_db_path=performance_db_path,
            strategy=routing_strategy
        )
        self.orchestrator = DeepSeekTradeOrchestrator(ollama_base_url=ollama_base_url)
        
        # Configuration
        self.enable_ensemble_mode = enable_ensemble_mode
        self.execution_history: List[Dict[str, Any]] = []
        
        # Performance thresholds
        self.performance_thresholds = {
            "min_confidence": 0.7,
            "ensemble_threshold": 0.9,
            "fallback_threshold": 0.5,
            "critical_task_threshold": 0.85
        }
        
        # Initialize performance tracking
        self._initialize_performance_tracking()

    async def process_trading_request(
        self, 
        request: TradingIntelligenceRequest,
        force_ensemble: bool = False
    ) -> TradingIntelligenceResponse:
        """
        Process a trading intelligence request using optimal LLM routing
        
        Args:
            request: Trading intelligence request
            force_ensemble: Force ensemble mode regardless of routing decision
            
        Returns:
            TradingIntelligenceResponse with decision and metadata
        """
        start_time = time.time()
        
        # Create task context from request
        task_context = self._create_task_context(request)
        
        # Determine routing strategy
        use_ensemble = force_ensemble or (
            self.enable_ensemble_mode and 
            (request.priority == "critical" or self._requires_ensemble(request))
        )
        
        try:
            # Route the task to optimal LLM(s)
            routing_decision = self.router.route_task(task_context, require_ensemble=use_ensemble)
            
            # Execute the trading analysis
            if routing_decision.selected_llm.startswith("ensemble_"):
                decision_result = await self._execute_ensemble_analysis(
                    task_context, routing_decision
                )
            else:
                decision_result = await self._execute_single_llm_analysis(
                    task_context, routing_decision
                )
            
            # Calculate execution metrics
            execution_time = time.time() - start_time
            
            # Update router performance
            success = self._evaluate_decision_success(decision_result)
            await self._update_performance_metrics(
                routing_decision.selected_llm,
                task_context.task_type,
                success,
                execution_time,
                decision_result.confidence
            )
            
            # Build response
            response = TradingIntelligenceResponse(
                decision=decision_result,
                routing_info={
                    "selected_llm": routing_decision.selected_llm,
                    "routing_confidence": routing_decision.confidence,
                    "routing_reasoning": routing_decision.reasoning,
                    "fallback_options": routing_decision.fallback_options,
                    "routing_strategy": self.router.strategy.value
                },
                execution_metrics={
                    "execution_time": execution_time,
                    "expected_time": routing_decision.estimated_time,
                    "efficiency_ratio": routing_decision.estimated_time / execution_time if execution_time > 0 else 1.0
                },
                confidence_factors=self._calculate_confidence_factors(
                    decision_result, routing_decision
                )
            )
            
            # Log successful execution
            self._log_execution(request, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing trading request: {e}")
            # Return fallback decision
            return await self._handle_execution_error(request, task_context, str(e))

    async def _execute_single_llm_analysis(
        self, 
        task_context: TaskContext, 
        routing_decision
    ) -> TradingDecisionResult:
        """Execute analysis using single LLM"""
        selected_llm = routing_decision.selected_llm
        
        # Map routing decision to orchestrator method
        if "deepseek" in selected_llm:
            return await self.orchestrator.analyze_with_deepseek(task_context)
        elif "qwen" in selected_llm:
            return await self.orchestrator.analyze_with_qwen(task_context)
        elif "yi" in selected_llm:
            return await self.orchestrator.analyze_with_yi(task_context)
        elif "glm" in selected_llm:
            return await self.orchestrator.analyze_with_glm(task_context)
        elif "coder" in selected_llm:
            return await self.orchestrator.analyze_with_deepseek_coder(task_context)
        else:
            # Fallback to DeepSeek R1 (primary orchestrator)
            return await self.orchestrator.analyze_with_deepseek(task_context)

    async def _execute_ensemble_analysis(
        self, 
        task_context: TaskContext, 
        routing_decision
    ) -> TradingDecisionResult:
        """Execute analysis using ensemble of LLMs"""
        ensemble_models = routing_decision.selected_llm.replace("ensemble_", "").split("+")
        
        # Execute analysis with multiple models concurrently
        tasks = []
        for model in ensemble_models[:3]:  # Limit to top 3 for performance
            if "deepseek" in model:
                tasks.append(self.orchestrator.analyze_with_deepseek(task_context))
            elif "qwen" in model:
                tasks.append(self.orchestrator.analyze_with_qwen(task_context))
            elif "yi" in model:
                tasks.append(self.orchestrator.analyze_with_yi(task_context))
            elif "glm" in model:
                tasks.append(self.orchestrator.analyze_with_glm(task_context))
        
        # Wait for all analyses to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and combine results
        valid_results = [r for r in results if isinstance(r, TradingDecisionResult)]
        
        if not valid_results:
            # All ensemble models failed, use fallback
            return await self.orchestrator.analyze_with_deepseek(task_context)
        
        # Combine ensemble results using weighted voting
        return self._combine_ensemble_results(valid_results, ensemble_models)

    def _combine_ensemble_results(
        self, 
        results: List[TradingDecisionResult], 
        model_names: List[str]
    ) -> TradingDecisionResult:
        """Combine multiple LLM results into consensus decision"""
        if len(results) == 1:
            return results[0]
        
        # Weight votes by model performance for this task type
        action_votes = {}
        confidence_sum = 0
        reasoning_parts = []
        
        for i, result in enumerate(results):
            model_name = model_names[i] if i < len(model_names) else "unknown"
            weight = 1.0  # Could be adjusted based on historical performance
            
            # Vote on action
            if result.action in action_votes:
                action_votes[result.action] += weight * result.confidence
            else:
                action_votes[result.action] = weight * result.confidence
            
            confidence_sum += result.confidence * weight
            reasoning_parts.append(f"{model_name}: {result.reasoning[:200]}...")
        
        # Determine consensus action
        consensus_action = max(action_votes.keys(), key=lambda k: action_votes[k])
        consensus_confidence = min(0.95, confidence_sum / len(results))
        
        # Combine reasoning
        consensus_reasoning = "Ensemble Decision - " + " | ".join(reasoning_parts)
        
        # Use the first result as template and update key fields
        consensus_result = results[0]
        consensus_result.action = consensus_action
        consensus_result.confidence = consensus_confidence
        consensus_result.reasoning = consensus_reasoning
        consensus_result.metadata["ensemble_size"] = len(results)
        consensus_result.metadata["vote_distribution"] = action_votes
        
        return consensus_result

    def _create_task_context(self, request: TradingIntelligenceRequest) -> TaskContext:
        """Convert request to TaskContext for orchestrator"""
        return TaskContext(
            task_type=request.task_type,
            symbol=request.symbol,
            market_data=request.market_data,
            portfolio_value=request.portfolio_context.get("total_value") if request.portfolio_context else None,
            risk_tolerance=request.risk_parameters.get("tolerance") if request.risk_parameters else "moderate",
            time_horizon=request.time_horizon,
            market_conditions=request.market_data.get("conditions"),
            additional_context=request.custom_context or {}
        )

    def _requires_ensemble(self, request: TradingIntelligenceRequest) -> bool:
        """Determine if request requires ensemble processing"""
        ensemble_triggers = [
            request.task_type in [TaskType.HEDGE_FUND_ANALYSIS, TaskType.RISK_ASSESSMENT],
            request.portfolio_context and request.portfolio_context.get("total_value", 0) > 500000,
            request.market_data.get("volatility", 0) > 0.8,
            request.priority == "critical"
        ]
        
        return any(ensemble_triggers)

    def _evaluate_decision_success(self, decision: TradingDecisionResult) -> bool:
        """Evaluate if decision meets success criteria"""
        # Simple success criteria - could be enhanced with actual trading results
        success_factors = [
            decision.confidence >= self.performance_thresholds["min_confidence"],
            decision.action != "hold" or decision.confidence >= 0.8,  # High confidence for hold decisions
            len(decision.reasoning) >= 50,  # Adequate reasoning provided
        ]
        
        return sum(success_factors) >= 2  # At least 2 out of 3 criteria

    async def _update_performance_metrics(
        self,
        llm_model: str,
        task_type: TaskType,
        success: bool,
        execution_time: float,
        confidence: float
    ):
        """Update performance metrics asynchronously"""
        try:
            # Update router performance
            self.router.update_performance(
                llm_model=llm_model,
                task_type=task_type,
                success=success,
                execution_time=execution_time,
                confidence_score=confidence
            )
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    def _calculate_confidence_factors(
        self, 
        decision: TradingDecisionResult, 
        routing_decision
    ) -> Dict[str, float]:
        """Calculate various confidence factors for the decision"""
        return {
            "llm_confidence": decision.confidence,
            "routing_confidence": routing_decision.confidence,
            "model_specialization": self._get_model_specialization_score(
                routing_decision.selected_llm, decision.action
            ),
            "consensus_factor": self._calculate_consensus_factor(decision),
            "historical_accuracy": self._get_historical_accuracy(
                routing_decision.selected_llm, decision.action
            )
        }

    def _get_model_specialization_score(self, llm_model: str, action: str) -> float:
        """Get specialization score for model-action combination"""
        # Simplified scoring - could be enhanced with detailed mappings
        specialization_map = {
            "deepseek": {"buy": 0.9, "sell": 0.85, "hold": 0.8},
            "qwen": {"buy": 0.85, "sell": 0.9, "hold": 0.75},
            "yi": {"buy": 0.8, "sell": 0.8, "hold": 0.9},
            "glm": {"buy": 0.75, "sell": 0.85, "hold": 0.95},
        }
        
        for model_key, actions in specialization_map.items():
            if model_key in llm_model.lower():
                return actions.get(action, 0.7)
        
        return 0.7  # Default score

    def _calculate_consensus_factor(self, decision: TradingDecisionResult) -> float:
        """Calculate consensus factor from ensemble metadata"""
        if "ensemble_size" not in decision.metadata:
            return 1.0
        
        ensemble_size = decision.metadata["ensemble_size"]
        vote_distribution = decision.metadata.get("vote_distribution", {})
        
        if not vote_distribution:
            return 1.0
        
        # Higher consensus factor for more unanimous decisions
        max_votes = max(vote_distribution.values())
        total_votes = sum(vote_distribution.values())
        
        return max_votes / total_votes if total_votes > 0 else 1.0

    def _get_historical_accuracy(self, llm_model: str, action: str) -> float:
        """Get historical accuracy for model-action combination"""
        # Simplified implementation - would use actual historical data
        if len(self.execution_history) < 10:
            return 0.8  # Default for insufficient history
        
        # Filter recent executions for this model and action
        relevant_executions = [
            exec_record for exec_record in self.execution_history[-50:]
            if (exec_record.get("routing_info", {}).get("selected_llm") == llm_model and
                exec_record.get("decision", {}).get("action") == action)
        ]
        
        if not relevant_executions:
            return 0.8
        
        # Calculate success rate (simplified)
        success_count = sum(1 for exec_record in relevant_executions 
                          if exec_record.get("success", False))
        
        return success_count / len(relevant_executions)

    async def _handle_execution_error(
        self, 
        request: TradingIntelligenceRequest, 
        task_context: TaskContext, 
        error_msg: str
    ) -> TradingIntelligenceResponse:
        """Handle execution errors with fallback strategy"""
        self.logger.warning(f"Falling back due to error: {error_msg}")
        
        # Use simple DeepSeek analysis as fallback
        try:
            fallback_decision = await self.orchestrator.analyze_with_deepseek(task_context)
            
            return TradingIntelligenceResponse(
                decision=fallback_decision,
                routing_info={
                    "selected_llm": "fallback_deepseek",
                    "routing_confidence": 0.5,
                    "routing_reasoning": f"Fallback due to error: {error_msg}",
                    "fallback_options": [],
                    "routing_strategy": "fallback"
                },
                execution_metrics={
                    "execution_time": 30.0,
                    "expected_time": 30.0,
                    "efficiency_ratio": 1.0
                },
                confidence_factors={
                    "llm_confidence": fallback_decision.confidence,
                    "routing_confidence": 0.5,
                    "model_specialization": 0.7,
                    "consensus_factor": 1.0,
                    "historical_accuracy": 0.7
                }
            )
            
        except Exception as fallback_error:
            self.logger.error(f"Fallback also failed: {fallback_error}")
            raise Exception(f"Both primary and fallback execution failed: {error_msg}, {fallback_error}")

    def _log_execution(self, request: TradingIntelligenceRequest, response: TradingIntelligenceResponse):
        """Log execution for performance tracking"""
        execution_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "request": {
                "task_type": request.task_type.value,
                "symbol": request.symbol,
                "priority": request.priority
            },
            "routing_info": response.routing_info,
            "decision": {
                "action": response.decision.action,
                "confidence": response.decision.confidence
            },
            "execution_metrics": response.execution_metrics,
            "success": response.decision.confidence >= self.performance_thresholds["min_confidence"]
        }
        
        self.execution_history.append(execution_record)
        
        # Keep only recent history to manage memory
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-500:]

    def _initialize_performance_tracking(self):
        """Initialize performance tracking components"""
        self.logger.info("Trading Intelligence Hub initialized successfully")
        self.logger.info(f"Routing Strategy: {self.router.strategy.value}")
        self.logger.info(f"Ensemble Mode: {'Enabled' if self.enable_ensemble_mode else 'Disabled'}")

    # Public API methods
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and performance"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "router_strategy": self.router.strategy.value,
            "ensemble_enabled": self.enable_ensemble_mode,
            "execution_history_size": len(self.execution_history),
            "performance_report": self.router.get_performance_report(),
            "available_models": list(self.router.llm_capabilities.keys())
        }

    def update_performance_thresholds(self, thresholds: Dict[str, float]):
        """Update performance thresholds"""
        self.performance_thresholds.update(thresholds)
        self.logger.info(f"Updated performance thresholds: {thresholds}")

    def switch_routing_strategy(self, strategy: RoutingStrategy):
        """Switch routing strategy"""
        self.router.strategy = strategy
        self.logger.info(f"Switched to routing strategy: {strategy.value}")

    async def warm_up_models(self, models: List[str] = None):
        """Warm up specified models or all available models"""
        if models is None:
            models = list(self.router.llm_capabilities.keys())
        
        self.logger.info(f"Warming up models: {models}")
        
        # Create simple warm-up context
        warm_up_context = TaskContext(
            task_type=TaskType.PATTERN_RECOGNITION,
            symbol="TEST",
            market_data={"price": 100.0},
            time_horizon="short"
        )
        
        for model in models:
            try:
                await self._execute_single_llm_analysis(warm_up_context, 
                    type('obj', (object,), {'selected_llm': model})())
                self.logger.info(f"Successfully warmed up {model}")
            except Exception as e:
                self.logger.warning(f"Failed to warm up {model}: {e}")