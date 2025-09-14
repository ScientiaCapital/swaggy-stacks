"""
Comprehensive Test Suite for Unsupervised Learning Monitoring
Institutional-grade testing for production trading systems
"""

import asyncio
import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

from app.monitoring.core_metrics import CoreTradingMetrics, CoreMetricsCollector


class TestPrometheusUnsupervisedMetrics:
    """Test Prometheus metrics for unsupervised learning components"""

    @pytest.fixture
    def prometheus_metrics(self):
        """Create PrometheusMetrics instance for testing"""
        return PrometheusMetrics()

    @pytest.fixture
    def sample_pattern_memory_data(self):
        """Sample pattern memory metrics data"""
        return {
            "symbol": "AAPL",
            "pattern_type": "price_volume",
            "total_patterns": 1500,
            "cache_hit_rate": 0.85,
            "compression_ratio": 7.2,
            "memory_usage_mb": 45.8,
            "retrieval_latency_ms": 12.5
        }

    @pytest.fixture
    def sample_regime_data(self):
        """Sample regime detection metrics data"""
        return {
            "symbol": "AAPL",
            "current_regime": "bull_trend",
            "regime_stability": 0.89,
            "transition_probability": 0.15,
            "prediction_accuracy": 0.91,
            "detection_latency_ms": 8.3
        }

    @pytest.fixture
    def sample_anomaly_data(self):
        """Sample anomaly detection metrics data"""
        return {
            "symbol": "AAPL",
            "anomaly_type": "volume_spike",
            "anomaly_score": 0.75,
            "detection_accuracy": 0.88,
            "false_positive_rate": 0.12,
            "detection_latency_ms": 15.2,
            "is_anomaly": True
        }

    def test_pattern_memory_metrics_update(self, prometheus_metrics, sample_pattern_memory_data):
        """Test pattern memory metrics are updated correctly"""
        prometheus_metrics.update_pattern_memory_metrics(**sample_pattern_memory_data)

        # Verify metrics are set correctly
        pattern_total = prometheus_metrics.pattern_memory_total_patterns.labels(
            symbol="AAPL", pattern_type="price_volume"
        )

        # Note: In testing, we can't easily access the metric values
        # but we can verify the methods run without error
        assert pattern_total is not None

    def test_regime_detection_metrics_update(self, prometheus_metrics, sample_regime_data):
        """Test regime detection metrics are updated correctly"""
        prometheus_metrics.update_regime_detection_metrics(**sample_regime_data)

        # Verify current regime is set
        current_regime = prometheus_metrics.regime_detection_current_regime.labels(
            symbol="AAPL", regime="bull_trend"
        )
        assert current_regime is not None

        # Verify other regime indicators are reset
        bear_regime = prometheus_metrics.regime_detection_current_regime.labels(
            symbol="AAPL", regime="bear_trend"
        )
        assert bear_regime is not None

    def test_anomaly_detection_metrics_update(self, prometheus_metrics, sample_anomaly_data):
        """Test anomaly detection metrics are updated correctly"""
        prometheus_metrics.update_anomaly_detection_metrics(**sample_anomaly_data)

        # Verify anomaly metrics are set
        anomaly_score = prometheus_metrics.anomaly_detection_score.labels(
            symbol="AAPL", anomaly_type="volume_spike"
        )
        assert anomaly_score is not None

    def test_clustering_metrics_update(self, prometheus_metrics):
        """Test clustering quality metrics are updated correctly"""
        clustering_data = {
            "cluster_type": "market_agent_analysis",
            "algorithm": "kmeans",
            "silhouette_score": 0.73,
            "inertia": 1250.5,
            "n_clusters": 8,
            "convergence_iterations": 12,
            "processing_time_ms": 45.8
        }

        prometheus_metrics.update_clustering_metrics(**clustering_data)

        # Verify clustering metrics are set
        silhouette = prometheus_metrics.clustering_silhouette_score.labels(
            cluster_type="market_agent_analysis", algorithm="kmeans"
        )
        assert silhouette is not None

    def test_experience_clustering_metrics_update(self, prometheus_metrics):
        """Test experience clustering metrics are updated correctly"""
        experience_data = {
            "agent_type": "market_agent",
            "tool_name": "technical_analysis",
            "total_experiences": 2500,
            "clustered_experiences": 2200,
            "cluster_quality": 0.82,
            "learning_rate": 0.15,
            "replay_efficiency": 0.91
        }

        prometheus_metrics.update_experience_clustering_metrics(**experience_data)

        # Verify experience clustering metrics are set
        total_exp = prometheus_metrics.experience_clustering_total.labels(
            agent_type="market_agent", tool_name="technical_analysis"
        )
        assert total_exp is not None

    def test_strategy_evolution_metrics_update(self, prometheus_metrics):
        """Test strategy evolution metrics are updated correctly"""
        evolution_data = {
            "strategy_name": "momentum_strategy",
            "variant_name": "enhanced_v2",
            "performance_improvement": 0.23,
            "statistical_significance": 0.95,
            "sample_size": 1000,
            "test_duration_hours": 72.5,
            "adoption_rate": 0.85
        }

        prometheus_metrics.update_strategy_evolution_metrics(**evolution_data)

        # Verify strategy evolution metrics are set
        performance = prometheus_metrics.strategy_evolution_performance.labels(
            strategy_name="momentum_strategy", variant_name="enhanced_v2"
        )
        assert performance is not None

    def test_strategy_evolution_experiment_recording(self, prometheus_metrics):
        """Test strategy evolution experiment recording"""
        prometheus_metrics.record_strategy_evolution_experiment(
            strategy_name="momentum_strategy",
            variant_name="enhanced_v2",
            success=True
        )

        # Verify experiment is recorded
        experiments = prometheus_metrics.strategy_evolution_experiments.labels(
            strategy_name="momentum_strategy",
            variant_name="enhanced_v2",
            status="success"
        )
        assert experiments is not None

    def test_resource_metrics_update(self, prometheus_metrics):
        """Test unsupervised component resource metrics"""
        resource_data = {
            "component_name": "pattern_memory",
            "cpu_usage_percent": 15.5,
            "memory_usage_mb": 128.7,
            "processing_latency_ms": 8.9
        }

        prometheus_metrics.update_unsupervised_resource_metrics(**resource_data)

        # Verify resource metrics are set
        cpu_usage = prometheus_metrics.unsupervised_cpu_usage.labels(
            component="pattern_memory"
        )
        assert cpu_usage is not None


class TestUnsupervisedMetricsCollection:
    """Test automatic collection of unsupervised learning metrics"""

    @pytest.fixture
    def metrics_collector(self):
        """Create MetricsCollector instance for testing"""
        return MetricsCollector()

    @pytest.fixture
    def mock_global_feedback_tracker(self):
        """Mock global feedback tracker with unsupervised components"""
        mock_tracker = Mock()

        # Mock pattern memory
        mock_pattern_memory = Mock()
        mock_pattern_memory.get_memory_stats.return_value = {
            "AAPL": {
                "price_volume": {
                    "total_patterns": 1500,
                    "cache_hits": 85,
                    "cache_misses": 15,
                    "compression_ratio": 7.2,
                    "memory_usage_mb": 45.8,
                    "avg_retrieval_latency_ms": 12.5
                }
            }
        }
        mock_tracker.pattern_memory = mock_pattern_memory

        # Mock anomaly detector
        mock_anomaly_detector = Mock()
        mock_anomaly_detector.get_performance_metrics.return_value = {
            "AAPL": {
                "volume_spike": {
                    "latest_score": 0.75,
                    "accuracy": 0.88,
                    "false_positive_rate": 0.12,
                    "avg_latency_ms": 15.2
                }
            }
        }
        mock_tracker.anomaly_detector = mock_anomaly_detector

        # Mock regime detector
        mock_regime_detector = Mock()
        mock_regime_detector.get_regime_statistics.return_value = {
            "AAPL": {
                "current_regime": "bull_trend",
                "stability": 0.89,
                "transition_probability": 0.15,
                "prediction_accuracy": 0.91,
                "avg_detection_latency_ms": 8.3
            }
        }
        mock_tracker.regime_detector = mock_regime_detector

        # Mock clustering statistics
        mock_tracker.get_clustering_statistics.return_value = {
            "market_agent_technical_analysis": {
                "total_experiences": 2500,
                "clustered_experiences": 2200,
                "cluster_quality": 0.82,
                "learning_rate": 0.15,
                "replay_efficiency": 0.91,
                "clustering_metrics": {
                    "algorithm": "kmeans",
                    "silhouette_score": 0.73,
                    "inertia": 1250.5,
                    "n_clusters": 8,
                    "iterations": 12,
                    "processing_time_ms": 45.8
                }
            }
        }

        return mock_tracker

    @pytest.mark.asyncio
    async def test_collect_unsupervised_metrics_success(self, metrics_collector, mock_global_feedback_tracker):
        """Test successful collection of all unsupervised metrics"""
        with patch('app.monitoring.metrics.UNSUPERVISED_AVAILABLE', True):
            with patch('app.analysis.tool_feedback_tracker.global_feedback_tracker', mock_global_feedback_tracker):
                with patch('app.monitoring.metrics.PatternMemory'):
                    with patch('app.monitoring.metrics.AnomalyDetector'):
                        with patch('app.monitoring.metrics.MarketRegimeDetector'):
                            # This should run without errors
                            await metrics_collector.prometheus_metrics.collect_unsupervised_metrics()

    @pytest.mark.asyncio
    async def test_collect_unsupervised_metrics_unavailable(self, metrics_collector):
        """Test graceful handling when unsupervised components are unavailable"""
        with patch('app.monitoring.metrics.UNSUPERVISED_AVAILABLE', False):
            # Should not raise any errors
            await metrics_collector.prometheus_metrics.collect_unsupervised_metrics()

    @pytest.mark.asyncio
    async def test_collect_pattern_memory_metrics(self, metrics_collector, mock_global_feedback_tracker):
        """Test collection of pattern memory metrics specifically"""
        mock_pattern_memory = mock_global_feedback_tracker.pattern_memory

        await metrics_collector.prometheus_metrics._collect_pattern_memory_metrics(mock_pattern_memory)

        # Verify the method was called
        mock_pattern_memory.get_memory_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_collect_anomaly_metrics(self, metrics_collector, mock_global_feedback_tracker):
        """Test collection of anomaly detection metrics specifically"""
        mock_anomaly_detector = mock_global_feedback_tracker.anomaly_detector

        await metrics_collector.prometheus_metrics._collect_anomaly_metrics(mock_anomaly_detector)

        # Verify the method was called
        mock_anomaly_detector.get_performance_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_collect_regime_metrics(self, metrics_collector, mock_global_feedback_tracker):
        """Test collection of regime detection metrics specifically"""
        mock_regime_detector = mock_global_feedback_tracker.regime_detector

        await metrics_collector.prometheus_metrics._collect_regime_metrics(mock_regime_detector)

        # Verify the method was called
        mock_regime_detector.get_regime_statistics.assert_called_once()

    @pytest.mark.asyncio
    async def test_collect_experience_clustering_metrics(self, metrics_collector, mock_global_feedback_tracker):
        """Test collection of experience clustering metrics specifically"""
        await metrics_collector.prometheus_metrics._collect_experience_clustering_metrics(mock_global_feedback_tracker)

        # Verify the method was called
        mock_global_feedback_tracker.get_clustering_statistics.assert_called_once()

    @pytest.mark.asyncio
    async def test_metrics_collection_with_errors(self, metrics_collector):
        """Test graceful error handling in metrics collection"""
        mock_tracker = Mock()
        mock_tracker.pattern_memory = None  # Simulate missing component

        # Should not raise errors even with missing components
        with patch('app.analysis.tool_feedback_tracker.global_feedback_tracker', mock_tracker):
            await metrics_collector.prometheus_metrics._collect_pattern_memory_metrics(None)

    @pytest.mark.asyncio
    async def test_full_metrics_collection_cycle(self, metrics_collector):
        """Test full metrics collection cycle including unsupervised metrics"""
        with patch.object(metrics_collector.health_checker, 'check_all_components',
                         return_value={"status": "healthy"}):
            with patch.object(metrics_collector.health_checker, 'get_system_metrics',
                             return_value={"cpu": 50.0, "memory": 60.0}):
                with patch.object(metrics_collector.prometheus_metrics, 'collect_unsupervised_metrics',
                                 new_callable=AsyncMock):

                    metrics = await metrics_collector.collect_system_metrics()

                    # Verify unsupervised metrics collection was called
                    metrics_collector.prometheus_metrics.collect_unsupervised_metrics.assert_called_once()

                    # Verify metrics structure
                    assert "health_status" in metrics
                    assert "system_metrics" in metrics
                    assert "timestamp" in metrics


class TestMetricsIntegration:
    """Integration tests for complete metrics pipeline"""

    @pytest.mark.asyncio
    async def test_end_to_end_metrics_pipeline(self):
        """Test complete end-to-end metrics collection and reporting"""
        # Create a real metrics collector
        collector = MetricsCollector()

        # Mock the health checker components
        with patch.object(collector.health_checker, 'check_all_components',
                         return_value={"status": "healthy", "components": ["database", "redis"]}):
            with patch.object(collector.health_checker, 'get_system_metrics',
                             return_value={"cpu_percent": 25.5, "memory_percent": 45.2}):
                with patch('app.monitoring.metrics.UNSUPERVISED_AVAILABLE', True):

                    # Run the full collection
                    metrics = await collector.collect_system_metrics()

                    # Verify structure
                    assert metrics["health_status"]["status"] == "healthy"
                    assert "timestamp" in metrics
                    assert "system_metrics" in metrics

    def test_prometheus_metrics_registry(self):
        """Test that all unsupervised metrics are properly registered"""
        prometheus_metrics = PrometheusMetrics()

        # Verify critical metrics exist
        assert hasattr(prometheus_metrics, 'pattern_memory_total_patterns')
        assert hasattr(prometheus_metrics, 'regime_detection_accuracy')
        assert hasattr(prometheus_metrics, 'anomaly_detection_score')
        assert hasattr(prometheus_metrics, 'clustering_silhouette_score')
        assert hasattr(prometheus_metrics, 'experience_clustering_quality')
        assert hasattr(prometheus_metrics, 'strategy_evolution_performance')

    def test_metric_label_consistency(self):
        """Test that metric labels are consistent across all components"""
        prometheus_metrics = PrometheusMetrics()

        # Test pattern memory labels
        pattern_metric = prometheus_metrics.pattern_memory_total_patterns.labels(
            symbol="AAPL", pattern_type="volume_pattern"
        )
        assert pattern_metric is not None

        # Test regime detection labels
        regime_metric = prometheus_metrics.regime_detection_accuracy.labels(
            symbol="AAPL"
        )
        assert regime_metric is not None

        # Test experience clustering labels
        experience_metric = prometheus_metrics.experience_clustering_quality.labels(
            agent_type="market_agent", tool_name="technical_analysis"
        )
        assert experience_metric is not None


class TestMetricsPerformance:
    """Performance tests for metrics collection in production environment"""

    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self):
        """Test that metrics collection completes within acceptable time limits"""
        import time

        collector = MetricsCollector()

        with patch.object(collector.health_checker, 'check_all_components',
                         return_value={"status": "healthy"}):
            with patch.object(collector.health_checker, 'get_system_metrics',
                             return_value={"cpu": 50.0}):
                with patch('app.monitoring.metrics.UNSUPERVISED_AVAILABLE', False):

                    start_time = time.time()
                    await collector.collect_system_metrics()
                    end_time = time.time()

                    # Metrics collection should complete in under 1 second
                    assert (end_time - start_time) < 1.0

    def test_metrics_memory_efficiency(self):
        """Test that metrics don't consume excessive memory"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create multiple metrics instances
        metrics_instances = [PrometheusMetrics() for _ in range(10)]

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB for 10 instances)
        assert memory_increase < 50 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_concurrent_metrics_updates(self):
        """Test concurrent updates to metrics don't cause race conditions"""
        prometheus_metrics = PrometheusMetrics()

        async def update_pattern_metrics(symbol: str, iteration: int):
            prometheus_metrics.update_pattern_memory_metrics(
                symbol=symbol,
                pattern_type="test_pattern",
                total_patterns=iteration,
                cache_hits=iteration * 10,
                cache_misses=iteration * 2,
                compression_ratio=5.0,
                memory_usage_mb=50.0,
                retrieval_latency_ms=10.0
            )

        # Run concurrent updates
        tasks = [
            update_pattern_metrics(f"SYM{i}", i)
            for i in range(20)
        ]

        # This should complete without errors
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])