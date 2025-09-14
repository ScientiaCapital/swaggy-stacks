"""
Core Feedback - Essential feedback tracking without clustering complexity

Simplified feedback system focused on core metrics and learning
for personal trading use.
"""

import asyncio
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

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
class SimplePerformanceMetrics:
    """Simplified performance metrics for personal trading"""

    tool_name: str
    agent_type: str
    total_executions: int
    success_rate: float
    average_execution_time_ms: float
    recent_trend: str  # improving, declining, stable
    last_updated: datetime

    # Personal trading focused metrics
    profitable_decisions: int
    average_confidence: float
    best_performing_context: Optional[str]


@dataclass
class PersonalLearningInsight:
    """Simple learning insights for personal review"""

    insight_type: str  # timing, context, parameter
    description: str
    confidence: float
    actionable_suggestion: str
    supporting_data_points: int
    created_at: datetime


class CoreFeedbackTracker:
    """Simplified feedback tracker optimized for personal trading"""

    def __init__(self, max_history_size: int = 5000):
        self.max_history_size = max_history_size

        # Simplified storage
        self.execution_history: deque[ToolExecution] = deque(maxlen=max_history_size)
        self.executions_by_tool: Dict[str, deque[ToolExecution]] = defaultdict(
            lambda: deque(maxlen=500)
        )

        # Performance tracking
        self.performance_metrics: Dict[str, SimplePerformanceMetrics] = {}
        self.learning_insights: List[PersonalLearningInsight] = []

        # Callbacks for real-time updates
        self.feedback_callbacks: List[Callable[[ToolExecution], None]] = []

        # Analysis state
        self.last_analysis_time: Optional[datetime] = None

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
        context: Dict[str, Any] = None,
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
            context=context or {},
        )

        # Store execution
        self.execution_history.append(execution)
        tool_key = f"{agent_type}_{tool_name}"
        self.executions_by_tool[tool_key].append(execution)

        # Trigger real-time callbacks
        for callback in self.feedback_callbacks:
            try:
                await callback(execution)
            except Exception as e:
                logger.warning("Feedback callback failed", error=str(e))

        # Log execution
        logger.debug(
            "Tool execution recorded",
            agent_id=agent_id,
            tool_name=tool_name,
            success=success,
            execution_time_ms=execution_time_ms,
        )

        return execution_id

    async def analyze_performance(self) -> Dict[str, SimplePerformanceMetrics]:
        """Analyze tool performance with simplified metrics"""

        analysis_start = datetime.now()
        cutoff_time = datetime.now() - timedelta(hours=24)  # Last 24 hours

        # Group executions by tool
        tool_groups = defaultdict(list)
        for execution in self.execution_history:
            if execution.timestamp >= cutoff_time:
                key = f"{execution.agent_type}_{execution.tool_name}"
                tool_groups[key].append(execution)

        # Calculate simplified metrics for each tool
        for tool_key, executions in tool_groups.items():
            if not executions:
                continue

            agent_type, tool_name = tool_key.split("_", 1)

            # Basic metrics
            total_executions = len(executions)
            successful_executions = sum(1 for exec in executions if exec.success)
            success_rate = successful_executions / total_executions if total_executions > 0 else 0.0

            # Execution time
            execution_times = [exec.execution_time_ms for exec in executions]
            avg_time = statistics.mean(execution_times) if execution_times else 0.0

            # Personal trading metrics
            profitable_decisions = 0
            confidence_scores = []

            for exec in executions:
                # Extract profitability from context if available
                if exec.context.get('profitable', False):
                    profitable_decisions += 1

                # Extract confidence from output if available
                if isinstance(exec.output_result, dict):
                    confidence = exec.output_result.get('confidence', 0.5)
                    confidence_scores.append(confidence)

            avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.5

            # Simple trend analysis
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

            # Find best performing context
            context_performance = defaultdict(list)
            for exec in executions:
                if 'symbol' in exec.context:
                    symbol = exec.context['symbol']
                    context_performance[f"symbol_{symbol}"].append(exec.success)
                if 'market_condition' in exec.context:
                    condition = exec.context['market_condition']
                    context_performance[f"market_{condition}"].append(exec.success)

            best_context = None
            best_success_rate = 0
            for context, successes in context_performance.items():
                if len(successes) >= 3:  # Minimum sample size
                    success_rate_context = sum(successes) / len(successes)
                    if success_rate_context > best_success_rate:
                        best_success_rate = success_rate_context
                        best_context = context

            # Create simplified metrics
            metrics = SimplePerformanceMetrics(
                tool_name=tool_name,
                agent_type=agent_type,
                total_executions=total_executions,
                success_rate=success_rate,
                average_execution_time_ms=avg_time,
                recent_trend=trend,
                last_updated=analysis_start,
                profitable_decisions=profitable_decisions,
                average_confidence=avg_confidence,
                best_performing_context=best_context,
            )

            self.performance_metrics[tool_key] = metrics

        self.last_analysis_time = analysis_start

        logger.info(
            "Performance analysis completed",
            tools_analyzed=len(tool_groups),
            analysis_duration_ms=(datetime.now() - analysis_start).total_seconds() * 1000,
        )

        return self.performance_metrics

    async def generate_personal_insights(self) -> List[PersonalLearningInsight]:
        """Generate simple, actionable insights for personal trading"""

        insights = []

        try:
            # Timing insights
            timing_insight = self._analyze_timing_patterns()
            if timing_insight:
                insights.append(timing_insight)

            # Context insights
            context_insight = self._analyze_context_patterns()
            if context_insight:
                insights.append(context_insight)

            # Performance insights
            performance_insight = self._analyze_performance_patterns()
            if performance_insight:
                insights.append(performance_insight)

            # Store insights
            self.learning_insights = insights[-10:]  # Keep last 10 insights

            logger.info(f"Generated {len(insights)} personal insights")

        except Exception as e:
            logger.error("Failed to generate personal insights", error=str(e))

        return insights

    def _analyze_timing_patterns(self) -> Optional[PersonalLearningInsight]:
        """Analyze timing patterns for personal insights"""

        try:
            # Analyze hour-of-day performance
            hour_performance = defaultdict(list)

            for execution in self.execution_history:
                hour = execution.timestamp.hour
                hour_performance[hour].append(execution.success)

            if len(hour_performance) < 3:
                return None

            # Find best and worst hours
            hour_success_rates = {}
            for hour, successes in hour_performance.items():
                if len(successes) >= 3:  # Minimum sample
                    hour_success_rates[hour] = sum(successes) / len(successes)

            if not hour_success_rates:
                return None

            best_hour = max(hour_success_rates.items(), key=lambda x: x[1])
            worst_hour = min(hour_success_rates.items(), key=lambda x: x[1])

            if best_hour[1] > worst_hour[1] * 1.2:  # Significant difference
                return PersonalLearningInsight(
                    insight_type="timing",
                    description=f"Performance varies by time: "
                              f"best at {best_hour[0]}:00 ({best_hour[1]:.1%} success), "
                              f"worst at {worst_hour[0]}:00 ({worst_hour[1]:.1%} success)",
                    confidence=0.8,
                    actionable_suggestion=f"Focus trading activity around {best_hour[0]}:00 for better results",
                    supporting_data_points=len(hour_performance[best_hour[0]]) + len(hour_performance[worst_hour[0]]),
                    created_at=datetime.now()
                )

        except Exception as e:
            logger.error("Failed to analyze timing patterns", error=str(e))

        return None

    def _analyze_context_patterns(self) -> Optional[PersonalLearningInsight]:
        """Analyze context patterns for personal insights"""

        try:
            # Analyze symbol performance
            symbol_performance = defaultdict(list)

            for execution in self.execution_history:
                if 'symbol' in execution.context:
                    symbol = execution.context['symbol']
                    symbol_performance[symbol].append(execution.success)

            # Find consistently successful symbols
            successful_symbols = []
            for symbol, successes in symbol_performance.items():
                if len(successes) >= 5:  # Minimum sample
                    success_rate = sum(successes) / len(successes)
                    if success_rate >= 0.8:
                        successful_symbols.append((symbol, success_rate))

            if successful_symbols:
                successful_symbols.sort(key=lambda x: x[1], reverse=True)
                top_symbol = successful_symbols[0]

                return PersonalLearningInsight(
                    insight_type="context",
                    description=f"Strong performance with {top_symbol[0]} ({top_symbol[1]:.1%} success rate)",
                    confidence=0.7,
                    actionable_suggestion=f"Consider focusing more trades on {top_symbol[0]} based on historical success",
                    supporting_data_points=len(symbol_performance[top_symbol[0]]),
                    created_at=datetime.now()
                )

        except Exception as e:
            logger.error("Failed to analyze context patterns", error=str(e))

        return None

    def _analyze_performance_patterns(self) -> Optional[PersonalLearningInsight]:
        """Analyze overall performance patterns"""

        try:
            if len(self.execution_history) < 20:
                return None

            recent_executions = list(self.execution_history)[-20:]
            older_executions = list(self.execution_history)[-40:-20] if len(self.execution_history) >= 40 else []

            if not older_executions:
                return None

            recent_success_rate = sum(1 for e in recent_executions if e.success) / len(recent_executions)
            older_success_rate = sum(1 for e in older_executions if e.success) / len(older_executions)

            improvement = recent_success_rate - older_success_rate

            if abs(improvement) >= 0.1:  # 10% change
                trend = "improving" if improvement > 0 else "declining"
                description = f"Performance is {trend}: recent success rate {recent_success_rate:.1%} vs previous {older_success_rate:.1%}"

                if trend == "improving":
                    suggestion = "Continue current approach - performance is improving"
                else:
                    suggestion = "Consider reviewing recent strategy changes - performance has declined"

                return PersonalLearningInsight(
                    insight_type="performance",
                    description=description,
                    confidence=0.6,
                    actionable_suggestion=suggestion,
                    supporting_data_points=len(recent_executions) + len(older_executions),
                    created_at=datetime.now()
                )

        except Exception as e:
            logger.error("Failed to analyze performance patterns", error=str(e))

        return None

    def get_personal_summary(self) -> Dict[str, Any]:
        """Get simplified summary for personal review"""

        total_executions = len(self.execution_history)

        if total_executions == 0:
            return {
                "status": "no_data",
                "total_executions": 0,
                "message": "No trading decisions recorded yet"
            }

        recent_executions = [e for e in self.execution_history
                           if (datetime.now() - e.timestamp).days <= 7]

        success_rate = sum(1 for e in self.execution_history if e.success) / total_executions
        recent_success_rate = (sum(1 for e in recent_executions if e.success) / len(recent_executions)
                             if recent_executions else 0)

        # Tool usage summary
        tool_usage = defaultdict(int)
        for execution in self.execution_history:
            tool_usage[execution.tool_name] += 1

        return {
            "status": "active",
            "total_executions": total_executions,
            "overall_success_rate": success_rate,
            "recent_success_rate": recent_success_rate,
            "recent_activity": len(recent_executions),
            "most_used_tools": dict(sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:5]),
            "learning_insights_available": len(self.learning_insights),
            "last_analysis": self.last_analysis_time.isoformat() if self.last_analysis_time else None
        }

    def add_feedback_callback(self, callback: Callable[[ToolExecution], None]):
        """Add callback for real-time feedback events"""
        self.feedback_callbacks.append(callback)

    def get_health_status(self) -> Dict[str, Any]:
        """Get simplified health status"""
        return {
            "status": "healthy",
            "total_executions": len(self.execution_history),
            "tools_tracked": len(self.executions_by_tool),
            "performance_metrics_available": len(self.performance_metrics),
            "learning_insights_available": len(self.learning_insights),
            "last_analysis": self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            "memory_usage": "optimized"
        }


# Global simplified feedback tracker
core_feedback_tracker = CoreFeedbackTracker()