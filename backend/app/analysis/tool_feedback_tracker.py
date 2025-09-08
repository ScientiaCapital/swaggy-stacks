"""
Tool Feedback Tracker - Monitor and analyze agent tool execution for performance optimization
Provides learning mechanisms to improve agent decision-making through tool execution feedback
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import json

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ToolExecution:
    """Individual tool execution record"""
    execution_id: str
    agent_id: str
    agent_type: str
    tool_name: str
    input_params: Dict[str, Any]
    output_result: Any
    execution_time_ms: float
    success: bool
    error_message: Optional[str]
    timestamp: datetime
    context: Dict[str, Any]  # Market conditions, symbol, etc.
    

@dataclass
class ToolPerformanceMetrics:
    """Performance metrics for tool execution analysis"""
    tool_name: str
    agent_type: str
    total_executions: int
    success_rate: float
    average_execution_time_ms: float
    median_execution_time_ms: float
    p95_execution_time_ms: float
    error_patterns: Dict[str, int]
    performance_trend: str  # improving, declining, stable
    last_updated: datetime


@dataclass
class FeedbackLearning:
    """Learning insights from tool feedback analysis"""
    tool_name: str
    agent_type: str
    optimization_suggestions: List[str]
    parameter_insights: Dict[str, Any]
    context_patterns: Dict[str, Any]
    performance_predictors: Dict[str, float]
    confidence_score: float


class ToolFeedbackTracker:
    """Track and analyze tool execution feedback for agent performance optimization"""
    
    def __init__(self, max_history_size: int = 10000, analysis_window_hours: int = 24):
        self.max_history_size = max_history_size
        self.analysis_window_hours = analysis_window_hours
        
        # Tool execution history
        self.execution_history: deque[ToolExecution] = deque(maxlen=max_history_size)
        self.executions_by_tool: Dict[str, deque[ToolExecution]] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.executions_by_agent: Dict[str, deque[ToolExecution]] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        # Performance metrics cache
        self.performance_metrics: Dict[str, ToolPerformanceMetrics] = {}
        self.learning_insights: Dict[str, FeedbackLearning] = {}
        
        # Feedback callbacks
        self.feedback_callbacks: List[Callable[[ToolExecution], None]] = []
        self.performance_callbacks: List[Callable[[ToolPerformanceMetrics], None]] = []
        
        # Analysis state
        self.last_analysis_time: Optional[datetime] = None
        self.analysis_task: Optional[asyncio.Task] = None
        
    def add_feedback_callback(self, callback: Callable[[ToolExecution], None]):
        """Add callback for real-time tool feedback events"""
        self.feedback_callbacks.append(callback)
    
    def add_performance_callback(self, callback: Callable[[ToolPerformanceMetrics], None]):
        """Add callback for performance metrics updates"""
        self.performance_callbacks.append(callback)
    
    async def record_tool_execution(
        self,
        agent_id: str,
        agent_type: str,
        tool_name: str,
        input_params: Dict[str, Any],
        output_result: Any,
        execution_time_ms: float,
        success: bool,
        error_message: Optional[str] = None,
        context: Dict[str, Any] = None
    ) -> str:
        """Record a tool execution for feedback analysis"""
        
        execution_id = f"{agent_id}_{tool_name}_{datetime.now().timestamp()}"
        
        execution = ToolExecution(
            execution_id=execution_id,
            agent_id=agent_id,
            agent_type=agent_type,
            tool_name=tool_name,
            input_params=input_params,
            output_result=output_result,
            execution_time_ms=execution_time_ms,
            success=success,
            error_message=error_message,
            timestamp=datetime.now(),
            context=context or {}
        )
        
        # Store execution
        self.execution_history.append(execution)
        self.executions_by_tool[f"{agent_type}_{tool_name}"].append(execution)
        self.executions_by_agent[agent_id].append(execution)
        
        # Trigger real-time callbacks
        for callback in self.feedback_callbacks:
            try:
                await callback(execution)
            except Exception as e:
                logger.warning("Feedback callback failed", error=str(e))
        
        # Log execution
        logger.info("Tool execution recorded",
                   agent_id=agent_id,
                   tool_name=tool_name,
                   success=success,
                   execution_time_ms=execution_time_ms)
        
        return execution_id
    
    def get_tool_performance_metrics(self, agent_type: str, tool_name: str) -> Optional[ToolPerformanceMetrics]:
        """Get performance metrics for a specific agent tool"""
        key = f"{agent_type}_{tool_name}"
        return self.performance_metrics.get(key)
    
    def get_agent_performance_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get performance summary for a specific agent"""
        agent_executions = list(self.executions_by_agent.get(agent_id, []))
        
        if not agent_executions:
            return {"agent_id": agent_id, "total_executions": 0}
        
        # Calculate basic metrics
        total_executions = len(agent_executions)
        successful_executions = sum(1 for exec in agent_executions if exec.success)
        success_rate = successful_executions / total_executions if total_executions > 0 else 0.0
        
        execution_times = [exec.execution_time_ms for exec in agent_executions]
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0.0
        
        # Tool usage breakdown
        tool_usage = defaultdict(int)
        for exec in agent_executions:
            tool_usage[exec.tool_name] += 1
        
        # Recent performance trend (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_executions = [exec for exec in agent_executions if exec.timestamp >= recent_cutoff]
        recent_success_rate = (
            sum(1 for exec in recent_executions if exec.success) / len(recent_executions) 
            if recent_executions else 0.0
        )
        
        return {
            "agent_id": agent_id,
            "total_executions": total_executions,
            "success_rate": success_rate,
            "recent_success_rate": recent_success_rate,
            "average_execution_time_ms": avg_execution_time,
            "tool_usage": dict(tool_usage),
            "last_execution": agent_executions[-1].timestamp.isoformat() if agent_executions else None
        }
    
    async def analyze_tool_performance(self) -> Dict[str, ToolPerformanceMetrics]:
        """Analyze tool performance across all agents"""
        analysis_start = datetime.now()
        
        # Group executions by agent_type + tool_name
        tool_groups = defaultdict(list)
        cutoff_time = datetime.now() - timedelta(hours=self.analysis_window_hours)
        
        for execution in self.execution_history:
            if execution.timestamp >= cutoff_time:
                key = f"{execution.agent_type}_{execution.tool_name}"
                tool_groups[key].append(execution)
        
        # Calculate metrics for each tool
        for tool_key, executions in tool_groups.items():
            if not executions:
                continue
                
            agent_type, tool_name = tool_key.split("_", 1)
            
            # Basic metrics
            total_executions = len(executions)
            successful_executions = sum(1 for exec in executions if exec.success)
            success_rate = successful_executions / total_executions if total_executions > 0 else 0.0
            
            # Execution time metrics
            execution_times = [exec.execution_time_ms for exec in executions]
            avg_time = statistics.mean(execution_times) if execution_times else 0.0
            median_time = statistics.median(execution_times) if execution_times else 0.0
            
            # P95 calculation
            if len(execution_times) >= 20:  # Need sufficient data for percentile
                p95_time = sorted(execution_times)[int(0.95 * len(execution_times))]
            else:
                p95_time = max(execution_times) if execution_times else 0.0
            
            # Error pattern analysis
            error_patterns = defaultdict(int)
            for exec in executions:
                if not exec.success and exec.error_message:
                    # Simplify error message to pattern
                    error_pattern = exec.error_message[:100]  # First 100 chars
                    error_patterns[error_pattern] += 1
            
            # Performance trend analysis
            if len(executions) >= 10:
                mid_point = len(executions) // 2
                first_half_success = sum(1 for exec in executions[:mid_point] if exec.success) / mid_point
                second_half_success = sum(1 for exec in executions[mid_point:] if exec.success) / (len(executions) - mid_point)
                
                if second_half_success > first_half_success * 1.1:
                    trend = "improving"
                elif second_half_success < first_half_success * 0.9:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
            
            # Create metrics object
            metrics = ToolPerformanceMetrics(
                tool_name=tool_name,
                agent_type=agent_type,
                total_executions=total_executions,
                success_rate=success_rate,
                average_execution_time_ms=avg_time,
                median_execution_time_ms=median_time,
                p95_execution_time_ms=p95_time,
                error_patterns=dict(error_patterns),
                performance_trend=trend,
                last_updated=analysis_start
            )
            
            self.performance_metrics[tool_key] = metrics
            
            # Trigger performance callbacks
            for callback in self.performance_callbacks:
                try:
                    await callback(metrics)
                except Exception as e:
                    logger.warning("Performance callback failed", error=str(e))
        
        self.last_analysis_time = analysis_start
        
        logger.info("Tool performance analysis completed",
                   tools_analyzed=len(tool_groups),
                   analysis_duration_ms=(datetime.now() - analysis_start).total_seconds() * 1000)
        
        return self.performance_metrics
    
    async def generate_learning_insights(self) -> Dict[str, FeedbackLearning]:
        """Generate learning insights for agent optimization"""
        insights = {}
        
        for tool_key, metrics in self.performance_metrics.items():
            agent_type, tool_name = tool_key.split("_", 1)
            
            # Get recent executions for analysis
            recent_executions = list(self.executions_by_tool[tool_key])[-100:]  # Last 100 executions
            
            if len(recent_executions) < 10:
                continue  # Need sufficient data for learning
            
            # Analyze parameter patterns
            parameter_insights = self._analyze_parameter_patterns(recent_executions)
            
            # Analyze context patterns  
            context_patterns = self._analyze_context_patterns(recent_executions)
            
            # Generate optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(metrics, recent_executions)
            
            # Calculate performance predictors
            performance_predictors = self._calculate_performance_predictors(recent_executions)
            
            # Calculate confidence score based on data quality
            confidence_score = min(1.0, len(recent_executions) / 100.0)
            if metrics.success_rate < 0.5:
                confidence_score *= 0.8  # Reduce confidence for low success rates
            
            learning = FeedbackLearning(
                tool_name=tool_name,
                agent_type=agent_type,
                optimization_suggestions=optimization_suggestions,
                parameter_insights=parameter_insights,
                context_patterns=context_patterns,
                performance_predictors=performance_predictors,
                confidence_score=confidence_score
            )
            
            insights[tool_key] = learning
        
        self.learning_insights = insights
        
        logger.info("Learning insights generated", 
                   tools_analyzed=len(insights))
        
        return insights
    
    def _analyze_parameter_patterns(self, executions: List[ToolExecution]) -> Dict[str, Any]:
        """Analyze input parameter patterns for successful vs failed executions"""
        successful_params = []
        failed_params = []
        
        for exec in executions:
            if exec.success:
                successful_params.append(exec.input_params)
            else:
                failed_params.append(exec.input_params)
        
        # Find common parameters in successful executions
        param_analysis = {
            "successful_patterns": {},
            "failure_patterns": {},
            "parameter_correlation": {}
        }
        
        # Simple analysis - could be enhanced with ML techniques
        if successful_params:
            # Find most common parameter values in successful executions
            param_keys = set()
            for params in successful_params:
                param_keys.update(params.keys())
            
            for key in param_keys:
                values = [params.get(key) for params in successful_params if key in params]
                if values:
                    # For numeric values, calculate statistics
                    numeric_values = [v for v in values if isinstance(v, (int, float))]
                    if numeric_values:
                        param_analysis["successful_patterns"][key] = {
                            "mean": statistics.mean(numeric_values),
                            "median": statistics.median(numeric_values),
                            "count": len(numeric_values)
                        }
        
        return param_analysis
    
    def _analyze_context_patterns(self, executions: List[ToolExecution]) -> Dict[str, Any]:
        """Analyze context patterns that correlate with success/failure"""
        context_analysis = {
            "success_contexts": defaultdict(int),
            "failure_contexts": defaultdict(int),
            "context_success_rates": {}
        }
        
        for exec in executions:
            context = exec.context
            
            # Analyze symbol context
            if "symbol" in context:
                symbol = context["symbol"]
                if exec.success:
                    context_analysis["success_contexts"][f"symbol_{symbol}"] += 1
                else:
                    context_analysis["failure_contexts"][f"symbol_{symbol}"] += 1
            
            # Analyze market condition context
            if "market_condition" in context:
                condition = context["market_condition"]
                if exec.success:
                    context_analysis["success_contexts"][f"market_{condition}"] += 1
                else:
                    context_analysis["failure_contexts"][f"market_{condition}"] += 1
        
        # Calculate success rates for different contexts
        all_contexts = set(context_analysis["success_contexts"].keys()) | set(context_analysis["failure_contexts"].keys())
        for context in all_contexts:
            successes = context_analysis["success_contexts"][context]
            failures = context_analysis["failure_contexts"][context]
            total = successes + failures
            if total > 0:
                context_analysis["context_success_rates"][context] = successes / total
        
        return dict(context_analysis)
    
    def _generate_optimization_suggestions(self, metrics: ToolPerformanceMetrics, 
                                         executions: List[ToolExecution]) -> List[str]:
        """Generate optimization suggestions based on performance analysis"""
        suggestions = []
        
        # Performance-based suggestions
        if metrics.success_rate < 0.8:
            suggestions.append(f"Low success rate ({metrics.success_rate:.2%}) - investigate error patterns")
        
        if metrics.average_execution_time_ms > 5000:  # 5 seconds
            suggestions.append("High average execution time - consider optimization or caching")
        
        if metrics.p95_execution_time_ms > metrics.average_execution_time_ms * 3:
            suggestions.append("High P95 latency variance - investigate performance outliers")
        
        # Error pattern suggestions
        if metrics.error_patterns:
            most_common_error = max(metrics.error_patterns.items(), key=lambda x: x[1])
            suggestions.append(f"Most common error pattern: '{most_common_error[0][:50]}...'")
        
        # Trend-based suggestions
        if metrics.performance_trend == "declining":
            suggestions.append("Performance is declining - investigate recent changes")
        elif metrics.performance_trend == "improving":
            suggestions.append("Performance is improving - continue current optimization efforts")
        
        return suggestions
    
    def _calculate_performance_predictors(self, executions: List[ToolExecution]) -> Dict[str, float]:
        """Calculate predictors of tool execution performance"""
        predictors = {}
        
        if not executions:
            return predictors
        
        # Time-based patterns
        hour_success_rates = defaultdict(list)
        for exec in executions:
            hour = exec.timestamp.hour
            hour_success_rates[hour].append(1.0 if exec.success else 0.0)
        
        # Find best and worst hours
        hour_rates = {hour: statistics.mean(rates) for hour, rates in hour_success_rates.items() if len(rates) >= 3}
        if hour_rates:
            best_hour = max(hour_rates.items(), key=lambda x: x[1])
            worst_hour = min(hour_rates.items(), key=lambda x: x[1])
            
            predictors["best_hour"] = best_hour[0]
            predictors["best_hour_success_rate"] = best_hour[1]
            predictors["worst_hour"] = worst_hour[0]
            predictors["worst_hour_success_rate"] = worst_hour[1]
        
        # Execution time predictors
        execution_times = [exec.execution_time_ms for exec in executions]
        if execution_times:
            predictors["execution_time_variance"] = statistics.variance(execution_times)
            predictors["execution_time_stability"] = 1.0 / (1.0 + predictors["execution_time_variance"] / 1000.0)
        
        return predictors
    
    async def start_continuous_analysis(self, interval_minutes: int = 30):
        """Start continuous performance analysis"""
        if self.analysis_task and not self.analysis_task.done():
            logger.warning("Continuous analysis already running")
            return
        
        async def analysis_loop():
            while True:
                try:
                    await self.analyze_tool_performance()
                    await self.generate_learning_insights()
                    await asyncio.sleep(interval_minutes * 60)
                except asyncio.CancelledError:
                    logger.info("Continuous analysis stopped")
                    break
                except Exception as e:
                    logger.error("Error in continuous analysis", error=str(e))
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
        
        self.analysis_task = asyncio.create_task(analysis_loop())
        logger.info("Started continuous tool feedback analysis", interval_minutes=interval_minutes)
    
    async def stop_continuous_analysis(self):
        """Stop continuous performance analysis"""
        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped continuous tool feedback analysis")
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get health status of the feedback tracker"""
        return {
            "status": "healthy",
            "total_executions": len(self.execution_history),
            "tools_tracked": len(self.executions_by_tool),
            "agents_tracked": len(self.executions_by_agent),
            "performance_metrics_available": len(self.performance_metrics),
            "learning_insights_available": len(self.learning_insights),
            "last_analysis": self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            "continuous_analysis_running": self.analysis_task is not None and not self.analysis_task.done(),
            "timestamp": datetime.now().isoformat()
        }


# Global feedback tracker instance
tool_feedback_tracker = ToolFeedbackTracker()