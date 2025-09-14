"""
Strategy Evolution - Autonomous strategy improvement for AI agents

Lightweight focused module for autonomous parameter tuning and A/B testing
of trading strategies based on clustered execution patterns.
"""

import asyncio
import random
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from scipy import stats

logger = structlog.get_logger(__name__)


@dataclass
class StrategyVariant:
    """Individual strategy variant for A/B testing"""

    variant_id: str
    base_strategy: str
    parameters: Dict[str, Any]
    confidence_multiplier: float
    timing_preferences: Dict[str, Any]
    context_filters: Dict[str, Any]
    creation_time: datetime

    # Performance tracking
    trial_count: int = 0
    success_count: int = 0
    total_reward: float = 0.0
    avg_execution_time: float = 0.0
    last_used: Optional[datetime] = None


@dataclass
class ABTestResult:
    """Result of A/B testing between strategy variants"""

    test_id: str
    variant_a_id: str
    variant_b_id: str
    test_duration_hours: float

    # Statistical results
    variant_a_performance: Dict[str, float]
    variant_b_performance: Dict[str, float]
    statistical_significance: float
    confidence_interval: Tuple[float, float]

    # Recommendation
    winning_variant: str
    performance_improvement: float
    recommendation: str
    test_completed: datetime


@dataclass
class EvolutionMetrics:
    """Metrics tracking strategy evolution progress"""

    agent_type: str
    evolution_generation: int
    active_variants: int
    completed_tests: int

    # Performance progression
    baseline_performance: float
    current_best_performance: float
    improvement_rate_per_week: float

    # Learning metrics
    successful_mutations: int
    failed_mutations: int
    exploration_vs_exploitation_ratio: float

    last_updated: datetime


class StrategyEvolution:
    """Autonomous strategy improvement through evolutionary optimization"""

    def __init__(self,
                 mutation_rate: float = 0.1,
                 max_variants_per_strategy: int = 5,
                 ab_test_duration_hours: int = 24,
                 min_trials_for_significance: int = 30):

        self.mutation_rate = mutation_rate
        self.max_variants_per_strategy = max_variants_per_strategy
        self.ab_test_duration_hours = ab_test_duration_hours
        self.min_trials_for_significance = min_trials_for_significance

        # Strategy tracking
        self.strategy_variants: Dict[str, List[StrategyVariant]] = defaultdict(list)
        self.active_ab_tests: Dict[str, ABTestResult] = {}
        self.evolution_metrics: Dict[str, EvolutionMetrics] = {}

        # Performance history
        self.performance_history: deque = deque(maxlen=1000)
        self.test_results: deque[ABTestResult] = deque(maxlen=100)

        # Evolution state
        self.generation_count = 0
        self.last_evolution_time: Optional[datetime] = None

    async def evolve_strategy(self,
                            agent_type: str,
                            base_parameters: Dict[str, Any],
                            recent_performance: List[Dict[str, Any]]) -> Optional[StrategyVariant]:
        """Evolve a strategy based on recent performance data"""

        try:
            # Initialize base strategy if needed
            strategy_key = f"{agent_type}_base"
            if not self.strategy_variants[strategy_key]:
                base_variant = self._create_base_variant(agent_type, base_parameters)
                self.strategy_variants[strategy_key].append(base_variant)

            # Analyze recent performance for mutation direction
            performance_insights = self._analyze_performance_trends(recent_performance)

            # Generate new variant through guided mutation
            new_variant = await self._mutate_strategy(
                agent_type,
                base_parameters,
                performance_insights
            )

            if new_variant:
                # Add to variants list (maintain max limit)
                variants = self.strategy_variants[strategy_key]
                if len(variants) >= self.max_variants_per_strategy:
                    # Remove worst performing variant
                    worst_variant = min(variants, key=lambda v: v.total_reward / max(1, v.trial_count))
                    variants.remove(worst_variant)

                variants.append(new_variant)

                # Start A/B test if we have multiple variants
                if len(variants) >= 2:
                    await self._start_ab_test(strategy_key, variants[-2], new_variant)

                logger.info(f"Evolved new strategy variant for {agent_type}",
                           variant_id=new_variant.variant_id,
                           total_variants=len(variants))

                return new_variant

        except Exception as e:
            logger.error(f"Failed to evolve strategy for {agent_type}", error=str(e))

        return None

    def _create_base_variant(self, agent_type: str, parameters: Dict[str, Any]) -> StrategyVariant:
        """Create base strategy variant"""

        variant_id = f"{agent_type}_base_{datetime.now().timestamp()}"

        return StrategyVariant(
            variant_id=variant_id,
            base_strategy=agent_type,
            parameters=parameters.copy(),
            confidence_multiplier=1.0,
            timing_preferences={},
            context_filters={},
            creation_time=datetime.now()
        )

    def _analyze_performance_trends(self, recent_performance: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze recent performance to guide mutation direction"""

        if not recent_performance:
            return {}

        insights = {}

        # Reward trend analysis
        rewards = [p.get('reward', 0.0) for p in recent_performance]
        if len(rewards) >= 5:
            # Calculate trend using linear regression
            x = np.arange(len(rewards))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, rewards)

            insights['reward_trend'] = {
                'slope': slope,
                'r_squared': r_value ** 2,
                'is_improving': slope > 0 and p_value < 0.1
            }

        # Execution time analysis
        exec_times = [p.get('execution_time_ms', 0) for p in recent_performance]
        if exec_times:
            insights['execution_time'] = {
                'mean': np.mean(exec_times),
                'std': np.std(exec_times),
                'is_stable': np.std(exec_times) < np.mean(exec_times) * 0.3
            }

        # Context success patterns
        context_success = defaultdict(list)
        for p in recent_performance:
            context = p.get('context', {})
            success = p.get('success', False)

            for key, value in context.items():
                context_success[f"{key}_{value}"].append(success)

        # Find context factors with high success rates
        successful_contexts = {}
        for context_key, successes in context_success.items():
            if len(successes) >= 3:  # Minimum sample size
                success_rate = sum(successes) / len(successes)
                if success_rate >= 0.8:
                    successful_contexts[context_key] = success_rate

        insights['successful_contexts'] = successful_contexts

        return insights

    async def _mutate_strategy(self,
                             agent_type: str,
                             base_parameters: Dict[str, Any],
                             performance_insights: Dict[str, Any]) -> Optional[StrategyVariant]:
        """Create mutated strategy variant based on performance insights"""

        variant_id = f"{agent_type}_mut_{self.generation_count}_{datetime.now().timestamp()}"

        # Start with base parameters
        mutated_params = base_parameters.copy()
        confidence_multiplier = 1.0
        timing_preferences = {}
        context_filters = {}

        # Apply mutations based on performance insights

        # 1. Parameter mutations based on reward trends
        if 'reward_trend' in performance_insights:
            trend = performance_insights['reward_trend']

            if not trend['is_improving']:
                # Performance declining - try more aggressive mutations
                for key, value in mutated_params.items():
                    if isinstance(value, (int, float)) and random.random() < self.mutation_rate * 2:
                        # Larger mutations for declining performance
                        mutation_factor = random.uniform(0.7, 1.3)
                        mutated_params[key] = value * mutation_factor
            else:
                # Performance improving - smaller, refined mutations
                for key, value in mutated_params.items():
                    if isinstance(value, (int, float)) and random.random() < self.mutation_rate:
                        # Smaller mutations for improving performance
                        mutation_factor = random.uniform(0.95, 1.05)
                        mutated_params[key] = value * mutation_factor

        # 2. Confidence adjustments based on execution stability
        if 'execution_time' in performance_insights:
            exec_info = performance_insights['execution_time']

            if exec_info['is_stable']:
                # Stable execution - can be more confident
                confidence_multiplier = random.uniform(1.05, 1.15)
            else:
                # Unstable execution - be more conservative
                confidence_multiplier = random.uniform(0.85, 0.95)

        # 3. Context filters based on successful patterns
        if 'successful_contexts' in performance_insights:
            successful_contexts = performance_insights['successful_contexts']

            # Add filters for highly successful contexts
            for context_key, success_rate in successful_contexts.items():
                if success_rate >= 0.9:
                    # Parse context key back to filter
                    parts = context_key.rsplit('_', 1)
                    if len(parts) == 2:
                        filter_key, filter_value = parts
                        context_filters[filter_key] = filter_value

        # 4. Timing mutations
        current_hour = datetime.now().hour
        if random.random() < 0.3:  # 30% chance of timing preference
            # Prefer current successful time or random exploration
            preferred_hours = list(range(max(0, current_hour - 2), min(24, current_hour + 3)))
            timing_preferences['preferred_hours'] = preferred_hours

        # Create variant
        variant = StrategyVariant(
            variant_id=variant_id,
            base_strategy=agent_type,
            parameters=mutated_params,
            confidence_multiplier=confidence_multiplier,
            timing_preferences=timing_preferences,
            context_filters=context_filters,
            creation_time=datetime.now()
        )

        self.generation_count += 1

        return variant

    async def _start_ab_test(self, strategy_key: str, variant_a: StrategyVariant, variant_b: StrategyVariant):
        """Start A/B test between two strategy variants"""

        test_id = f"ab_test_{strategy_key}_{datetime.now().timestamp()}"

        ab_test = ABTestResult(
            test_id=test_id,
            variant_a_id=variant_a.variant_id,
            variant_b_id=variant_b.variant_id,
            test_duration_hours=self.ab_test_duration_hours,
            variant_a_performance={},
            variant_b_performance={},
            statistical_significance=0.0,
            confidence_interval=(0.0, 0.0),
            winning_variant="",
            performance_improvement=0.0,
            recommendation="testing_in_progress",
            test_completed=datetime.now() + timedelta(hours=self.ab_test_duration_hours)
        )

        self.active_ab_tests[test_id] = ab_test

        logger.info(f"Started A/B test {test_id}",
                   variant_a=variant_a.variant_id,
                   variant_b=variant_b.variant_id)

    async def record_variant_performance(self,
                                       variant_id: str,
                                       success: bool,
                                       reward: float,
                                       execution_time_ms: float):
        """Record performance for a strategy variant"""

        # Find the variant
        variant = None
        for strategy_variants in self.strategy_variants.values():
            for v in strategy_variants:
                if v.variant_id == variant_id:
                    variant = v
                    break
            if variant:
                break

        if not variant:
            logger.warning(f"Variant {variant_id} not found for performance recording")
            return

        # Update variant performance
        variant.trial_count += 1
        if success:
            variant.success_count += 1
        variant.total_reward += reward
        variant.avg_execution_time = (
            (variant.avg_execution_time * (variant.trial_count - 1) + execution_time_ms) /
            variant.trial_count
        )
        variant.last_used = datetime.now()

        # Check if variant is part of active A/B test
        await self._update_ab_test_results(variant_id, success, reward, execution_time_ms)

    async def _update_ab_test_results(self, variant_id: str, success: bool, reward: float, execution_time_ms: float):
        """Update A/B test results with new performance data"""

        for test_id, ab_test in self.active_ab_tests.items():
            if variant_id in [ab_test.variant_a_id, ab_test.variant_b_id]:

                # Check if test duration has elapsed
                if datetime.now() >= ab_test.test_completed:
                    await self._complete_ab_test(test_id)
                    break

    async def _complete_ab_test(self, test_id: str):
        """Complete an A/B test and determine the winner"""

        ab_test = self.active_ab_tests.get(test_id)
        if not ab_test:
            return

        try:
            # Get variant performance data
            variant_a = self._get_variant_by_id(ab_test.variant_a_id)
            variant_b = self._get_variant_by_id(ab_test.variant_b_id)

            if not variant_a or not variant_b:
                logger.warning(f"Could not find variants for A/B test {test_id}")
                return

            # Calculate performance metrics
            variant_a_perf = self._calculate_variant_performance(variant_a)
            variant_b_perf = self._calculate_variant_performance(variant_b)

            ab_test.variant_a_performance = variant_a_perf
            ab_test.variant_b_performance = variant_b_perf

            # Statistical significance test
            if (variant_a.trial_count >= self.min_trials_for_significance and
                variant_b.trial_count >= self.min_trials_for_significance):

                # Use reward as primary metric
                a_rewards = [variant_a_perf['avg_reward']] * variant_a.trial_count
                b_rewards = [variant_b_perf['avg_reward']] * variant_b.trial_count

                # Welch's t-test for unequal variances
                t_stat, p_value = stats.ttest_ind(a_rewards, b_rewards, equal_var=False)

                ab_test.statistical_significance = 1.0 - p_value

                # Determine winner
                if p_value < 0.05:  # Statistically significant
                    if variant_a_perf['avg_reward'] > variant_b_perf['avg_reward']:
                        ab_test.winning_variant = ab_test.variant_a_id
                        ab_test.performance_improvement = (
                            variant_a_perf['avg_reward'] - variant_b_perf['avg_reward']
                        )
                    else:
                        ab_test.winning_variant = ab_test.variant_b_id
                        ab_test.performance_improvement = (
                            variant_b_perf['avg_reward'] - variant_a_perf['avg_reward']
                        )

                    ab_test.recommendation = f"Use {ab_test.winning_variant} for {ab_test.performance_improvement:.1%} improvement"
                else:
                    ab_test.recommendation = "No significant difference - continue with either variant"
            else:
                ab_test.recommendation = "Insufficient data for statistical significance"

            # Mark test as completed
            ab_test.test_completed = datetime.now()

            # Move to completed tests
            self.test_results.append(ab_test)
            del self.active_ab_tests[test_id]

            logger.info(f"Completed A/B test {test_id}",
                       winner=ab_test.winning_variant,
                       improvement=ab_test.performance_improvement,
                       significance=ab_test.statistical_significance)

        except Exception as e:
            logger.error(f"Failed to complete A/B test {test_id}", error=str(e))

    def _get_variant_by_id(self, variant_id: str) -> Optional[StrategyVariant]:
        """Get variant by ID"""
        for strategy_variants in self.strategy_variants.values():
            for variant in strategy_variants:
                if variant.variant_id == variant_id:
                    return variant
        return None

    def _calculate_variant_performance(self, variant: StrategyVariant) -> Dict[str, float]:
        """Calculate performance metrics for a variant"""

        if variant.trial_count == 0:
            return {
                'success_rate': 0.0,
                'avg_reward': 0.0,
                'avg_execution_time': 0.0,
                'trial_count': 0
            }

        return {
            'success_rate': variant.success_count / variant.trial_count,
            'avg_reward': variant.total_reward / variant.trial_count,
            'avg_execution_time': variant.avg_execution_time,
            'trial_count': variant.trial_count
        }

    def get_best_variant(self, agent_type: str) -> Optional[StrategyVariant]:
        """Get the best performing variant for an agent type"""

        strategy_key = f"{agent_type}_base"
        variants = self.strategy_variants.get(strategy_key, [])

        if not variants:
            return None

        # Filter variants with sufficient trials
        viable_variants = [v for v in variants if v.trial_count >= 10]

        if not viable_variants:
            viable_variants = variants

        # Return variant with highest average reward
        return max(viable_variants, key=lambda v: v.total_reward / max(1, v.trial_count))

    def get_evolution_metrics(self, agent_type: str) -> Optional[EvolutionMetrics]:
        """Get evolution metrics for an agent type"""
        return self.evolution_metrics.get(agent_type)

    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get overall evolution statistics"""

        total_variants = sum(len(variants) for variants in self.strategy_variants.values())
        completed_tests = len(self.test_results)
        active_tests = len(self.active_ab_tests)

        # Performance improvement calculation
        improvements = []
        for test in self.test_results:
            if test.performance_improvement > 0:
                improvements.append(test.performance_improvement)

        avg_improvement = np.mean(improvements) if improvements else 0.0

        return {
            "total_strategy_variants": total_variants,
            "active_strategies": len(self.strategy_variants),
            "completed_ab_tests": completed_tests,
            "active_ab_tests": active_tests,
            "evolution_generations": self.generation_count,
            "average_improvement_per_test": avg_improvement,
            "successful_mutations": len([t for t in self.test_results if t.performance_improvement > 0]),
            "last_evolution_time": self.last_evolution_time.isoformat() if self.last_evolution_time else None
        }


# Global strategy evolution instance
strategy_evolution = StrategyEvolution()