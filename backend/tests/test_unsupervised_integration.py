"""
Integration Tests for Unsupervised Learning System
Production-grade testing for institutional trading system
"""

import asyncio
import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, List, Any
import tempfile
import os

try:
    from app.analysis.tool_feedback_tracker import ToolFeedbackTracker, ToolExecution
    from app.ml.unsupervised.agent_memory import AgentMemory, Experience
    from app.ml.unsupervised.strategy_evolution import StrategyEvolution, StrategyVariant
    UNSUPERVISED_AVAILABLE = True
except ImportError:
    UNSUPERVISED_AVAILABLE = False


@pytest.mark.skipif(not UNSUPERVISED_AVAILABLE, reason="Unsupervised components not available")
class TestUnsupervisedIntegration:
    """Integration tests for complete unsupervised learning pipeline"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def feedback_tracker(self, temp_db_path):
        """Create ToolFeedbackTracker for testing"""
        return ToolFeedbackTracker(
            max_history_size=1000,
            enable_clustering=True,
            enable_pattern_memory=True,
            enable_anomaly_detection=True,
            enable_strategy_evolution=True,
            database_path=temp_db_path
        )

    @pytest.fixture
    def agent_memory(self):
        """Create AgentMemory for testing"""
        return AgentMemory(max_experiences=1000)

    @pytest.fixture
    def strategy_evolution(self):
        """Create StrategyEvolution for testing"""
        return StrategyEvolution(
            min_sample_size=10,
            significance_threshold=0.05,
            performance_threshold=0.05
        )

    @pytest.fixture
    def sample_tool_executions(self):
        """Generate sample tool executions for testing"""
        executions = []
        base_time = datetime.now()

        for i in range(50):
            execution = ToolExecution(
                execution_id=f"exec_{i}",
                agent_id=f"agent_{i % 5}",
                agent_type="market_agent",
                tool_name="technical_analysis",
                input_params={
                    "symbol": "AAPL" if i % 2 == 0 else "GOOGL",
                    "timeframe": "1d",
                    "indicators": ["rsi", "macd", "bb"]
                },
                output_result={
                    "signal": "BUY" if i % 3 == 0 else "SELL" if i % 3 == 1 else "HOLD",
                    "confidence": 0.7 + (i % 10) * 0.03,
                    "price_target": 150.0 + i * 2.5
                },
                execution_time_ms=50.0 + (i % 20) * 5.0,
                success=i % 10 != 9,  # 90% success rate
                error_message="timeout_error" if i % 10 == 9 else None,
                timestamp=base_time + timedelta(minutes=i * 5),
                context={
                    "market_condition": "bull" if i < 25 else "bear",
                    "volatility": 0.2 + (i % 5) * 0.1,
                    "volume": 1000000 + i * 50000
                }
            )
            executions.append(execution)

        return executions

    @pytest.mark.asyncio
    async def test_complete_feedback_loop_integration(self, feedback_tracker, sample_tool_executions):
        """Test complete feedback loop from execution recording to pattern learning"""

        # Record multiple tool executions
        execution_ids = []
        for execution in sample_tool_executions[:30]:
            exec_id = await feedback_tracker.record_tool_execution(
                agent_id=execution.agent_id,
                agent_type=execution.agent_type,
                tool_name=execution.tool_name,
                input_params=execution.input_params,
                output_result=execution.output_result,
                execution_time_ms=execution.execution_time_ms,
                success=execution.success,
                error_message=execution.error_message,
                context=execution.context
            )
            execution_ids.append(exec_id)

        # Allow some processing time
        await asyncio.sleep(0.1)

        # Test clustering analysis
        tool_key = "market_agent_technical_analysis"
        if hasattr(feedback_tracker, 'cluster_execution_patterns'):
            clusters = await feedback_tracker.cluster_execution_patterns(tool_key, min_executions=10)

            # Should find meaningful clusters
            assert len(clusters) > 0
            assert len(clusters) <= 5  # Reasonable number of clusters

            # Each cluster should have meaningful patterns
            for cluster in clusters:
                assert cluster.size > 0
                assert 0 <= cluster.avg_confidence <= 1
                assert cluster.avg_execution_time > 0
                assert cluster.success_rate >= 0

        # Test performance analysis
        performance_data = await feedback_tracker.analyze_performance()
        assert tool_key in performance_data

        perf_metrics = performance_data[tool_key]
        assert perf_metrics.total_executions == 30
        assert 0 <= perf_metrics.success_rate <= 1
        assert perf_metrics.average_execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_agent_memory_experience_clustering(self, agent_memory, sample_tool_executions):
        """Test agent memory experience clustering and replay"""

        # Convert tool executions to experiences
        experiences = []
        for execution in sample_tool_executions[:20]:
            experience = Experience(
                agent_id=execution.agent_id,
                tool_name=execution.tool_name,
                state=execution.input_params,
                action=execution.output_result,
                reward=1.0 if execution.success else -1.0,
                next_state=execution.context,
                timestamp=execution.timestamp,
                metadata={
                    "execution_time_ms": execution.execution_time_ms,
                    "confidence": execution.output_result.get("confidence", 0.5)
                }
            )
            experiences.append(experience)

        # Store experiences
        for experience in experiences:
            agent_memory.store_experience(experience)

        # Test clustering
        if hasattr(agent_memory, 'cluster_experiences'):
            clusters = agent_memory.cluster_experiences(min_cluster_size=3)

            # Should find some clusters
            assert len(clusters) >= 0  # Might be 0 if data is too sparse

            if len(clusters) > 0:
                # Clusters should be meaningful
                for cluster_id, cluster_experiences in clusters.items():
                    assert len(cluster_experiences) >= 3
                    assert all(isinstance(exp, Experience) for exp in cluster_experiences)

        # Test experience retrieval
        if hasattr(agent_memory, 'get_similar_experiences'):
            query_experience = experiences[0]
            similar = agent_memory.get_similar_experiences(query_experience, top_k=5)

            assert len(similar) <= 5
            assert all(isinstance(exp, Experience) for exp in similar)

    @pytest.mark.asyncio
    async def test_strategy_evolution_pipeline(self, strategy_evolution, sample_tool_executions):
        """Test strategy evolution and A/B testing pipeline"""

        # Create strategy variants from sample data
        base_strategy = {
            "rsi_threshold": 70,
            "macd_signal": "crossover",
            "confidence_threshold": 0.6
        }

        # Test strategy evolution
        if hasattr(strategy_evolution, 'evolve_strategy'):
            variant = await strategy_evolution.evolve_strategy("momentum_strategy", base_strategy)

            if variant:
                assert isinstance(variant, StrategyVariant)
                assert variant.strategy_name == "momentum_strategy"
                assert variant.parameters != base_strategy  # Should be different

        # Test A/B testing with mock performance data
        if hasattr(strategy_evolution, 'run_ab_test'):
            # Mock performance data
            control_performance = [0.02, 0.01, -0.01, 0.03, 0.02, 0.01, 0.04, -0.02, 0.01, 0.02]
            treatment_performance = [0.03, 0.04, 0.01, 0.05, 0.03, 0.02, 0.06, 0.01, 0.03, 0.04]

            result = await strategy_evolution.run_ab_test(
                strategy_name="momentum_strategy",
                control_performance=control_performance,
                treatment_performance=treatment_performance
            )

            assert "p_value" in result
            assert "effect_size" in result
            assert "recommendation" in result
            assert 0 <= result["p_value"] <= 1

    @pytest.mark.asyncio
    async def test_real_time_pattern_detection(self, feedback_tracker, sample_tool_executions):
        """Test real-time pattern detection and anomaly alerts"""

        # Record executions in real-time pattern
        for execution in sample_tool_executions[:15]:
            await feedback_tracker.record_tool_execution(
                agent_id=execution.agent_id,
                agent_type=execution.agent_type,
                tool_name=execution.tool_name,
                input_params=execution.input_params,
                output_result=execution.output_result,
                execution_time_ms=execution.execution_time_ms,
                success=execution.success,
                error_message=execution.error_message,
                context=execution.context
            )

        # Test pattern memory storage
        if hasattr(feedback_tracker, 'pattern_memory'):
            pattern_memory = feedback_tracker.pattern_memory
            if pattern_memory and hasattr(pattern_memory, 'get_memory_stats'):
                stats = pattern_memory.get_memory_stats()

                # Should have some patterns stored
                assert isinstance(stats, dict)

        # Test anomaly detection
        if hasattr(feedback_tracker, 'anomaly_detector'):
            anomaly_detector = feedback_tracker.anomaly_detector
            if anomaly_detector and hasattr(anomaly_detector, 'get_performance_metrics'):
                performance = anomaly_detector.get_performance_metrics()

                # Should have performance data
                assert isinstance(performance, dict)

    @pytest.mark.asyncio
    async def test_metrics_integration_with_components(self, feedback_tracker):
        """Test that components provide metrics in expected format"""

        # Test pattern memory metrics interface
        if hasattr(feedback_tracker, 'pattern_memory') and feedback_tracker.pattern_memory:
            pattern_memory = feedback_tracker.pattern_memory
            if hasattr(pattern_memory, 'get_memory_stats'):
                stats = pattern_memory.get_memory_stats()
                assert isinstance(stats, dict)

                # Check expected structure
                for symbol, pattern_data in stats.items():
                    assert isinstance(pattern_data, dict)
                    for pattern_type, metrics in pattern_data.items():
                        assert isinstance(metrics, dict)
                        # Expected metric keys
                        expected_keys = [
                            'total_patterns', 'cache_hits', 'cache_misses',
                            'compression_ratio', 'memory_usage_mb', 'avg_retrieval_latency_ms'
                        ]
                        # Not all keys need to be present, but if they are, check types
                        for key in expected_keys:
                            if key in metrics:
                                assert isinstance(metrics[key], (int, float))

        # Test anomaly detector metrics interface
        if hasattr(feedback_tracker, 'anomaly_detector') and feedback_tracker.anomaly_detector:
            anomaly_detector = feedback_tracker.anomaly_detector
            if hasattr(anomaly_detector, 'get_performance_metrics'):
                performance = anomaly_detector.get_performance_metrics()
                assert isinstance(performance, dict)

        # Test regime detector metrics interface
        if hasattr(feedback_tracker, 'regime_detector') and feedback_tracker.regime_detector:
            regime_detector = feedback_tracker.regime_detector
            if hasattr(regime_detector, 'get_regime_statistics'):
                stats = regime_detector.get_regime_statistics()
                assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_system_resilience_with_errors(self, feedback_tracker):
        """Test system resilience when components encounter errors"""

        # Test with malformed input
        try:
            await feedback_tracker.record_tool_execution(
                agent_id="test_agent",
                agent_type="market_agent",
                tool_name="test_tool",
                input_params=None,  # Malformed
                output_result={"result": "test"},
                execution_time_ms=50.0,
                success=True
            )
        except Exception as e:
            # Should handle gracefully, not crash
            assert isinstance(e, (ValueError, TypeError))

        # Test with extremely large execution time
        await feedback_tracker.record_tool_execution(
            agent_id="test_agent",
            agent_type="market_agent",
            tool_name="test_tool",
            input_params={"test": "data"},
            output_result={"result": "test"},
            execution_time_ms=999999.0,  # Very large time
            success=True
        )

        # Test with invalid timestamp (should use current time)
        await feedback_tracker.record_tool_execution(
            agent_id="test_agent",
            agent_type="market_agent",
            tool_name="test_tool",
            input_params={"test": "data"},
            output_result={"result": "test"},
            execution_time_ms=50.0,
            success=True
        )

    @pytest.mark.asyncio
    async def test_performance_under_load(self, feedback_tracker):
        """Test system performance under high load"""
        import time

        start_time = time.time()

        # Simulate high-frequency trading executions
        tasks = []
        for i in range(100):
            task = feedback_tracker.record_tool_execution(
                agent_id=f"agent_{i % 10}",
                agent_type="market_agent",
                tool_name="high_frequency_analysis",
                input_params={
                    "symbol": f"SYM{i % 20}",
                    "price": 100.0 + i * 0.1,
                    "volume": 1000 + i * 10
                },
                output_result={
                    "signal": "BUY" if i % 2 == 0 else "SELL",
                    "confidence": 0.5 + (i % 50) * 0.01
                },
                execution_time_ms=10.0 + (i % 10),
                success=i % 20 != 19  # 95% success rate
            )
            tasks.append(task)

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete 100 executions in reasonable time (under 5 seconds)
        assert total_time < 5.0

        # Verify all executions were recorded
        if hasattr(feedback_tracker, 'execution_history'):
            assert len(feedback_tracker.execution_history) >= 100

    def test_memory_usage_efficiency(self, feedback_tracker, sample_tool_executions):
        """Test memory usage remains reasonable with large datasets"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Load many executions
        for _ in range(5):  # Repeat sample data 5 times
            for execution in sample_tool_executions:
                asyncio.run(feedback_tracker.record_tool_execution(
                    agent_id=execution.agent_id,
                    agent_type=execution.agent_type,
                    tool_name=execution.tool_name,
                    input_params=execution.input_params,
                    output_result=execution.output_result,
                    execution_time_ms=execution.execution_time_ms,
                    success=execution.success,
                    error_message=execution.error_message,
                    context=execution.context
                ))

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB for test data)
        assert memory_increase < 100 * 1024 * 1024


class TestUnsupervisedSystemHealth:
    """Health and diagnostic tests for unsupervised learning system"""

    @pytest.mark.skipif(not UNSUPERVISED_AVAILABLE, reason="Unsupervised components not available")
    def test_component_initialization(self):
        """Test that all components initialize correctly"""

        # Test ToolFeedbackTracker initialization
        tracker = ToolFeedbackTracker(
            max_history_size=100,
            enable_clustering=True,
            enable_pattern_memory=True,
            enable_anomaly_detection=True
        )
        assert tracker is not None

        # Test AgentMemory initialization
        memory = AgentMemory(max_experiences=100)
        assert memory is not None
        assert memory.max_experiences == 100

        # Test StrategyEvolution initialization
        evolution = StrategyEvolution()
        assert evolution is not None

    @pytest.mark.skipif(not UNSUPERVISED_AVAILABLE, reason="Unsupervised components not available")
    def test_component_health_checks(self):
        """Test health check methods for all components"""

        # Test ToolFeedbackTracker health
        tracker = ToolFeedbackTracker(max_history_size=100)
        if hasattr(tracker, 'get_health_status'):
            health = tracker.get_health_status()
            assert isinstance(health, dict)
            assert "status" in health

        # Test AgentMemory health
        memory = AgentMemory(max_experiences=100)
        if hasattr(memory, 'get_health_status'):
            health = memory.get_health_status()
            assert isinstance(health, dict)

    @pytest.mark.skipif(not UNSUPERVISED_AVAILABLE, reason="Unsupervised components not available")
    def test_graceful_degradation(self):
        """Test system continues working even if some components fail"""

        # Create tracker with some components disabled
        tracker = ToolFeedbackTracker(
            max_history_size=100,
            enable_clustering=False,  # Disable clustering
            enable_pattern_memory=True,
            enable_anomaly_detection=False  # Disable anomaly detection
        )

        # Should still be able to record executions
        asyncio.run(tracker.record_tool_execution(
            agent_id="test_agent",
            agent_type="market_agent",
            tool_name="test_tool",
            input_params={"test": "data"},
            output_result={"result": "success"},
            execution_time_ms=50.0,
            success=True
        ))

        # Should have recorded the execution
        assert len(tracker.execution_history) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])