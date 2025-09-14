"""
Comprehensive Performance Benchmarks for Unsupervised Learning System
Production readiness validation with real-time performance requirements
"""

import pytest
import numpy as np
import pandas as pd
import time
import asyncio
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch

from app.ml.unsupervised.clustering import MarketDataClusterer
from app.ml.unsupervised.market_regime import MarketRegimeDetector
from app.ml.unsupervised.anomaly_detector import EnsembleAnomalyDetector
from app.ml.unsupervised.pattern_memory import PatternMemorySystem
from app.analysis.backtesting_framework import (
    BacktestingEngine, BacktestConfig, StrategyType, run_comprehensive_validation
)
from app.monitoring.metrics import PrometheusMetrics


class TestPerformanceBenchmarks:
    """Comprehensive performance benchmark tests"""

    @pytest.fixture
    def benchmark_config(self):
        """Configuration for performance benchmarks"""
        return {
            'max_processing_time_1k_points': 1.0,      # 1 second for 1000 data points
            'max_memory_usage_mb': 200,                # 200MB max memory usage
            'min_throughput_points_per_second': 500,   # 500 data points/second
            'max_real_time_latency_ms': 100,           # 100ms for real-time processing
            'parallel_processing_scalability': 0.8     # 80% efficiency with parallel processing
        }

    @pytest.fixture
    def large_market_dataset(self):
        """Generate large market dataset for performance testing"""
        np.random.seed(42)

        # 30 days of 1-minute data (43,200 data points)
        n_points = 30 * 24 * 60
        timestamps = pd.date_range('2024-01-01', periods=n_points, freq='1min')

        # Realistic market data with various patterns
        returns = np.random.normal(0.0001, 0.01, n_points)

        # Add market events
        crash_start = 15000
        for i in range(100):  # Flash crash
            if crash_start + i < n_points:
                returns[crash_start + i] = np.random.normal(-0.02, 0.01)

        bubble_start = 25000
        for i in range(1000):  # Market bubble
            if bubble_start + i < n_points:
                returns[bubble_start + i] = np.random.normal(0.005, 0.015)

        prices = 100 * np.exp(np.cumsum(returns))
        volumes = np.random.lognormal(7, 0.5, n_points)
        volatility = np.abs(returns) + np.random.gamma(1, 0.005, n_points)

        return pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes,
            'volatility': volatility,
            'returns': returns
        })

    def test_clustering_performance_benchmark(self, benchmark_config, large_market_dataset):
        """Test clustering performance with large datasets"""
        clusterer = MarketDataClusterer()

        # Test different data sizes
        data_sizes = [100, 500, 1000, 2000, 5000]
        processing_times = []
        memory_usage = []

        process = psutil.Process(os.getpid())

        for size in data_sizes:
            data_subset = large_market_dataset.head(size)

            # Measure memory before
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Measure processing time
            start_time = time.time()
            result = clusterer.cluster_market_data(f'PERF_TEST_{size}', data_subset)
            processing_time = time.time() - start_time

            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before

            processing_times.append(processing_time)
            memory_usage.append(memory_increase)

            # Validate result
            assert 'kmeans_clusters' in result
            assert len(result['kmeans_clusters']) == size

            # Performance requirements
            max_time = size * 0.002  # 2ms per data point
            assert processing_time < max_time, f"Clustering {size} points took {processing_time:.3f}s, expected < {max_time:.3f}s"

        # Test 1000-point performance specifically
        idx_1000 = data_sizes.index(1000)
        assert processing_times[idx_1000] < benchmark_config['max_processing_time_1k_points']

        # Memory usage should be reasonable
        max_memory = max(memory_usage)
        assert max_memory < benchmark_config['max_memory_usage_mb'], f"Memory usage {max_memory:.1f}MB exceeds limit"

        # Calculate throughput
        throughput = 1000 / processing_times[idx_1000]  # points per second
        assert throughput > benchmark_config['min_throughput_points_per_second']

    def test_regime_detection_performance_benchmark(self, benchmark_config):
        """Test regime detection performance requirements"""
        regime_detector = MarketRegimeDetector()

        # Real-time processing simulation
        batch_sizes = [10, 50, 100, 200, 500]
        processing_times = []

        for batch_size in batch_sizes:
            # Generate batch data
            returns = np.random.normal(0.0001, 0.01, batch_size)
            prices = 100 * np.exp(np.cumsum(returns))
            data = pd.DataFrame({
                'price': prices,
                'volume': np.random.lognormal(7, 0.3, batch_size),
                'volatility': np.abs(returns),
                'returns': returns
            })

            # Measure processing time
            start_time = time.time()
            result = regime_detector.detect_regime(data)
            processing_time = time.time() - start_time

            processing_times.append(processing_time)

            # Validate result
            assert 'regime' in result
            assert 'confidence' in result

            # Real-time requirement: process within 100ms for trading decisions
            if batch_size <= 100:  # Typical real-time batch size
                assert processing_time * 1000 < benchmark_config['max_real_time_latency_ms'], \
                    f"Regime detection took {processing_time*1000:.1f}ms, expected < {benchmark_config['max_real_time_latency_ms']}ms"

        # Performance scaling check
        for i in range(1, len(batch_sizes)):
            scaling_ratio = processing_times[i] / processing_times[i-1]
            size_ratio = batch_sizes[i] / batch_sizes[i-1]
            efficiency = size_ratio / scaling_ratio

            # Should scale reasonably well
            assert efficiency > 0.5, f"Poor scaling efficiency: {efficiency:.2f}"

    def test_anomaly_detection_performance_benchmark(self, benchmark_config):
        """Test anomaly detection performance under load"""
        anomaly_detector = EnsembleAnomalyDetector()

        # High-frequency data simulation
        data_size = 2000
        data = pd.DataFrame({
            'price': 100 + np.cumsum(np.random.randn(data_size) * 0.01),
            'volume': np.random.lognormal(7, 0.5, data_size),
            'volatility': np.random.gamma(1, 0.01, data_size)
        })

        # Add known anomalies
        anomaly_indices = [500, 1000, 1500]
        for idx in anomaly_indices:
            data.loc[idx, 'volume'] *= 15  # Volume spike

        # Performance test
        start_time = time.time()
        result = anomaly_detector.detect_anomalies(data)
        processing_time = time.time() - start_time

        # Validate result
        assert 'anomaly_scores' in result
        assert 'anomaly_flags' in result
        assert len(result['anomaly_scores']) == data_size

        # Performance requirements
        throughput = data_size / processing_time
        assert throughput > benchmark_config['min_throughput_points_per_second']

        # Should detect injected anomalies
        detected_anomalies = np.where(result['anomaly_flags'])[0]
        detection_success = sum(1 for idx in anomaly_indices
                              if any(abs(detected - idx) <= 5 for detected in detected_anomalies))
        detection_rate = detection_success / len(anomaly_indices)
        assert detection_rate >= 0.67, f"Low anomaly detection rate: {detection_rate}"

    def test_parallel_processing_performance(self, benchmark_config):
        """Test parallel processing capabilities"""
        clusterer = MarketDataClusterer()
        regime_detector = MarketRegimeDetector()

        # Generate multiple datasets
        datasets = []
        for i in range(4):  # 4 parallel tasks
            data = pd.DataFrame({
                'price': 100 + np.cumsum(np.random.randn(500) * 0.01),
                'volume': np.random.lognormal(7, 0.3, 500),
                'volatility': np.random.gamma(1, 0.01, 500)
            })
            datasets.append(data)

        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for i, data in enumerate(datasets):
            clustering_result = clusterer.cluster_market_data(f'SEQ_{i}', data)
            regime_result = regime_detector.detect_regime(data)
            sequential_results.append((clustering_result, regime_result))
        sequential_time = time.time() - start_time

        # Parallel processing simulation (using concurrent execution)
        import concurrent.futures

        def process_dataset(args):
            i, data = args
            clustering_result = clusterer.cluster_market_data(f'PAR_{i}', data)
            regime_result = regime_detector.detect_regime(data)
            return clustering_result, regime_result

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            parallel_results = list(executor.map(process_dataset, enumerate(datasets)))
        parallel_time = time.time() - start_time

        # Validate results
        assert len(sequential_results) == len(parallel_results) == 4

        # Performance improvement with parallelization
        speedup = sequential_time / parallel_time
        efficiency = speedup / 4  # 4 workers

        assert efficiency > benchmark_config['parallel_processing_scalability'], \
            f"Parallel efficiency {efficiency:.2f} below threshold"

    def test_memory_efficiency_benchmark(self, benchmark_config, large_market_dataset):
        """Test memory efficiency under sustained load"""
        clusterer = MarketDataClusterer()
        process = psutil.Process(os.getpid())

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process data in batches to simulate sustained operation
        batch_size = 1000
        num_batches = 20
        max_memory_observed = initial_memory

        for batch_num in range(num_batches):
            start_idx = (batch_num * batch_size) % (len(large_market_dataset) - batch_size)
            batch_data = large_market_dataset.iloc[start_idx:start_idx + batch_size].copy()

            # Process batch
            result = clusterer.cluster_market_data(f'MEMORY_TEST_{batch_num}', batch_data)

            # Monitor memory
            current_memory = process.memory_info().rss / 1024 / 1024
            max_memory_observed = max(max_memory_observed, current_memory)

            # Validate result
            assert 'kmeans_clusters' in result

        memory_increase = max_memory_observed - initial_memory

        # Memory should not grow excessively
        assert memory_increase < benchmark_config['max_memory_usage_mb'], \
            f"Memory increased by {memory_increase:.1f}MB, exceeds limit"

    def test_real_time_streaming_performance(self, benchmark_config):
        """Test real-time streaming data processing performance"""
        anomaly_detector = EnsembleAnomalyDetector()

        # Simulate real-time data stream
        stream_duration = 30  # seconds
        data_frequency = 10   # data points per second
        total_points = stream_duration * data_frequency

        processing_times = []
        queue_depths = []

        # Generate streaming data
        for point_num in range(total_points):
            # Generate single data point
            data_point = pd.DataFrame({
                'price': [100 + np.random.randn() * 0.01],
                'volume': [np.random.lognormal(7, 0.3)],
                'volatility': [np.random.gamma(1, 0.01)]
            })

            # Measure processing time for single point
            start_time = time.time()
            result = anomaly_detector.detect_anomalies(data_point)
            processing_time = time.time() - start_time

            processing_times.append(processing_time)

            # Simulate queue depth (processing time vs arrival rate)
            arrival_interval = 1.0 / data_frequency  # 0.1 seconds
            queue_depth = max(0, processing_time - arrival_interval)
            queue_depths.append(queue_depth)

            # Real-time constraint: each point must be processed within arrival interval
            assert processing_time < arrival_interval * 2, \
                f"Processing time {processing_time:.4f}s exceeds 2x arrival interval"

        # Overall streaming performance metrics
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        avg_queue_depth = np.mean(queue_depths)

        # Performance requirements
        assert avg_processing_time * 1000 < benchmark_config['max_real_time_latency_ms']
        assert max_processing_time < 0.2  # No single processing should take more than 200ms
        assert avg_queue_depth < 0.01  # Queue should stay small

    def test_backtesting_performance_benchmark(self, benchmark_config):
        """Test backtesting framework performance"""
        # Generate backtesting data
        n_points = 10000  # Large dataset for performance testing
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='5min'),
            'price': 100 * np.exp(np.cumsum(np.random.randn(n_points) * 0.005)),
            'volume': np.random.lognormal(7, 0.5, n_points),
            'volatility': np.random.gamma(1, 0.01, n_points)
        })

        config = BacktestConfig(
            start_date=data['timestamp'].min(),
            end_date=data['timestamp'].max(),
            initial_capital=100000
        )

        engine = BacktestingEngine(config)

        # Measure backtesting performance
        start_time = time.time()

        # Run baseline strategy only for performance test
        result = asyncio.run(engine._run_single_strategy_backtest(StrategyType.BASELINE, data))

        processing_time = time.time() - start_time

        # Validate result
        assert result.total_trades > 0
        assert result.equity_curve is not None

        # Performance requirements for backtesting
        throughput = n_points / processing_time
        assert throughput > 1000, f"Backtesting throughput {throughput:.0f} points/s below 1000 points/s"

        # Should complete large backtest within reasonable time
        assert processing_time < 30, f"Backtesting took {processing_time:.1f}s, expected < 30s"

    def test_comprehensive_system_load_test(self, benchmark_config):
        """Comprehensive system load test with all components"""
        # Initialize all components
        clusterer = MarketDataClusterer()
        regime_detector = MarketRegimeDetector()
        anomaly_detector = EnsembleAnomalyDetector()
        pattern_memory = PatternMemorySystem()

        # Generate test data
        n_points = 2000
        data = pd.DataFrame({
            'price': 100 + np.cumsum(np.random.randn(n_points) * 0.01),
            'volume': np.random.lognormal(7, 0.5, n_points),
            'volatility': np.random.gamma(1, 0.01, n_points),
            'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='1min')
        })

        # Measure comprehensive processing
        start_time = time.time()

        # Process through all components
        clustering_result = clusterer.cluster_market_data('LOAD_TEST', data)
        regime_result = regime_detector.detect_regime(data)
        anomaly_result = anomaly_detector.detect_anomalies(data)

        # Store patterns
        patterns = {
            'clusters': clustering_result['kmeans_clusters'],
            'regime': regime_result['regime'],
            'anomalies': anomaly_result['anomaly_flags']
        }
        pattern_storage = pattern_memory.store_patterns('LOAD_TEST', patterns, data)

        total_processing_time = time.time() - start_time

        # Validate all results
        assert 'kmeans_clusters' in clustering_result
        assert 'regime' in regime_result
        assert 'anomaly_scores' in anomaly_result
        assert 'pattern_id' in pattern_storage

        # Overall system performance
        system_throughput = n_points / total_processing_time
        assert system_throughput > 200, f"System throughput {system_throughput:.0f} points/s below 200 points/s"

        # Individual component performance
        assert len(clustering_result['kmeans_clusters']) == n_points
        assert len(anomaly_result['anomaly_scores']) == n_points

    @pytest.mark.asyncio
    async def test_concurrent_processing_benchmark(self, benchmark_config):
        """Test concurrent processing of multiple data streams"""
        anomaly_detector = EnsembleAnomalyDetector()

        # Simulate multiple concurrent data streams
        num_streams = 5
        points_per_stream = 500

        async def process_stream(stream_id):
            """Process a single data stream"""
            processing_times = []

            for point_num in range(points_per_stream):
                data = pd.DataFrame({
                    'price': [100 + np.random.randn() * 0.01],
                    'volume': [np.random.lognormal(7, 0.3)],
                    'volatility': [np.random.gamma(1, 0.01)]
                })

                start_time = time.time()
                result = anomaly_detector.detect_anomalies(data)
                processing_time = time.time() - start_time

                processing_times.append(processing_time)

                # Small delay to simulate real-time data arrival
                await asyncio.sleep(0.001)

            return stream_id, processing_times

        # Run concurrent streams
        start_time = time.time()
        tasks = [process_stream(i) for i in range(num_streams)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Validate results
        for stream_id, processing_times in results:
            assert len(processing_times) == points_per_stream
            avg_processing_time = np.mean(processing_times)
            assert avg_processing_time < 0.01, f"Stream {stream_id} avg processing time too high: {avg_processing_time:.4f}s"

        # Concurrent processing efficiency
        total_points = num_streams * points_per_stream
        overall_throughput = total_points / total_time

        # Should handle concurrent processing efficiently
        assert overall_throughput > 1000, f"Concurrent throughput {overall_throughput:.0f} points/s below 1000 points/s"

    def test_stress_test_edge_cases(self, benchmark_config):
        """Stress test with edge cases and challenging data"""
        clusterer = MarketDataClusterer()
        regime_detector = MarketRegimeDetector()

        # Edge case scenarios
        edge_cases = [
            # Very high volatility
            {
                'name': 'high_volatility',
                'data': pd.DataFrame({
                    'price': 100 + np.cumsum(np.random.randn(1000) * 0.05),
                    'volume': np.random.lognormal(8, 1.0, 1000),
                    'volatility': np.random.gamma(3, 0.03, 1000)
                })
            },
            # Very low volatility
            {
                'name': 'low_volatility',
                'data': pd.DataFrame({
                    'price': 100 + np.cumsum(np.random.randn(1000) * 0.001),
                    'volume': np.random.lognormal(6, 0.1, 1000),
                    'volatility': np.random.gamma(0.5, 0.001, 1000)
                })
            },
            # Extreme values
            {
                'name': 'extreme_values',
                'data': pd.DataFrame({
                    'price': [100] + [100 * (1.1 ** i) for i in range(999)],  # Exponential growth
                    'volume': np.random.lognormal(7, 2.0, 1000),  # High variance
                    'volatility': np.random.gamma(1, 0.01, 1000)
                })
            }
        ]

        for case in edge_cases:
            data = case['data']
            case_name = case['name']

            # Test clustering under stress
            start_time = time.time()
            try:
                clustering_result = clusterer.cluster_market_data(f'STRESS_{case_name}', data)
                clustering_time = time.time() - start_time

                # Should complete within reasonable time even for edge cases
                assert clustering_time < 5.0, f"Clustering stress test {case_name} took {clustering_time:.2f}s"

                # Should produce valid results
                assert 'kmeans_clusters' in clustering_result
                assert len(clustering_result['kmeans_clusters']) == len(data)

            except Exception as e:
                pytest.fail(f"Clustering failed for stress test {case_name}: {e}")

            # Test regime detection under stress
            start_time = time.time()
            try:
                regime_result = regime_detector.detect_regime(data)
                regime_time = time.time() - start_time

                # Should complete within reasonable time
                assert regime_time < 2.0, f"Regime detection stress test {case_name} took {regime_time:.2f}s"

                # Should produce valid results
                assert 'regime' in regime_result
                assert 'confidence' in regime_result

            except Exception as e:
                pytest.fail(f"Regime detection failed for stress test {case_name}: {e}")


class TestProductionReadinessBenchmarks:
    """Production readiness validation benchmarks"""

    def test_30_percent_improvement_validation(self):
        """Validate the 30%+ improvement requirement through backtesting"""
        # Generate realistic market data for validation
        np.random.seed(42)  # Consistent results
        n_points = 5000

        # Create market data with clear patterns that unsupervised learning can exploit
        returns = np.random.normal(0.0005, 0.015, n_points)

        # Add regime changes
        bull_periods = [(0, 1500), (3000, 4500)]
        bear_periods = [(1500, 2500)]
        volatile_periods = [(2500, 3000), (4500, 5000)]

        for start, end in bull_periods:
            returns[start:end] = np.random.normal(0.002, 0.01, end - start)

        for start, end in bear_periods:
            returns[start:end] = np.random.normal(-0.001, 0.02, end - start)

        for start, end in volatile_periods:
            returns[start:end] = np.random.normal(0, 0.03, end - start)

        prices = 100 * np.exp(np.cumsum(returns))
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='5min'),
            'price': prices,
            'volume': np.random.lognormal(7, 0.5, n_points),
            'volatility': np.abs(returns) + np.random.gamma(1, 0.005, n_points)
        })

        # Run comprehensive validation
        validation_result = asyncio.run(run_comprehensive_validation(data))

        # Validate 30% improvement requirement
        assert validation_result['validation_passed'], "Failed to meet 30% improvement requirement"

        improvement_pct = validation_result['comparison_report']['improvement_analysis']['return_improvement_pct']
        assert improvement_pct >= 30, f"Improvement {improvement_pct:.1f}% below 30% requirement"

        # Additional quality checks
        comparison = validation_result['comparison_report']['comparative_metrics']
        baseline_sharpe = comparison['baseline']['sharpe_ratio']
        enhanced_sharpe = comparison['enhanced']['sharpe_ratio']

        # Enhanced strategy should also improve risk-adjusted returns
        assert enhanced_sharpe > baseline_sharpe, "Enhanced strategy should improve Sharpe ratio"

    def test_production_performance_requirements(self):
        """Test all production performance requirements are met"""
        requirements = {
            'clustering_1k_points_max_time': 1.0,
            'regime_detection_100_points_max_time': 0.1,
            'anomaly_detection_1k_points_max_time': 0.5,
            'memory_usage_max_mb': 200,
            'concurrent_streams': 5
        }

        # Test clustering performance
        clusterer = MarketDataClusterer()
        data_1k = pd.DataFrame({
            'price': 100 + np.cumsum(np.random.randn(1000) * 0.01),
            'volume': np.random.lognormal(7, 0.5, 1000),
            'volatility': np.random.gamma(1, 0.01, 1000)
        })

        start_time = time.time()
        clustering_result = clusterer.cluster_market_data('PROD_REQ_TEST', data_1k)
        clustering_time = time.time() - start_time

        assert clustering_time < requirements['clustering_1k_points_max_time']
        assert 'kmeans_clusters' in clustering_result

        # Test regime detection performance
        regime_detector = MarketRegimeDetector()
        data_100 = data_1k.head(100)

        start_time = time.time()
        regime_result = regime_detector.detect_regime(data_100)
        regime_time = time.time() - start_time

        assert regime_time < requirements['regime_detection_100_points_max_time']
        assert 'regime' in regime_result

        # Test anomaly detection performance
        anomaly_detector = EnsembleAnomalyDetector()

        start_time = time.time()
        anomaly_result = anomaly_detector.detect_anomalies(data_1k)
        anomaly_time = time.time() - start_time

        assert anomaly_time < requirements['anomaly_detection_1k_points_max_time']
        assert 'anomaly_scores' in anomaly_result

        # Memory usage validation
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB

        # This is a rough check - in production, dedicated monitoring would track this
        assert memory_usage < requirements['memory_usage_max_mb'] * 2  # Allow some buffer for test overhead

    def test_comprehensive_production_validation(self):
        """Comprehensive production readiness validation"""
        validation_results = {
            'performance_benchmarks': True,
            'accuracy_requirements': True,
            'scalability_requirements': True,
            'reliability_requirements': True
        }

        # Performance validation
        try:
            # Test suite of performance requirements
            clusterer = MarketDataClusterer()
            regime_detector = MarketRegimeDetector()
            anomaly_detector = EnsembleAnomalyDetector()

            test_data = pd.DataFrame({
                'price': 100 + np.cumsum(np.random.randn(500) * 0.01),
                'volume': np.random.lognormal(7, 0.3, 500),
                'volatility': np.random.gamma(1, 0.01, 500)
            })

            # All components should process successfully
            clustering_result = clusterer.cluster_market_data('VALIDATION', test_data)
            regime_result = regime_detector.detect_regime(test_data)
            anomaly_result = anomaly_detector.detect_anomalies(test_data)

            assert 'kmeans_clusters' in clustering_result
            assert 'regime' in regime_result
            assert 'anomaly_scores' in anomaly_result

        except Exception as e:
            validation_results['performance_benchmarks'] = False
            pytest.fail(f"Performance benchmark validation failed: {e}")

        # Accuracy validation (basic checks)
        try:
            # Clustering should produce reasonable number of clusters
            n_clusters = len(set(clustering_result['kmeans_clusters']))
            assert 2 <= n_clusters <= 10, f"Unreasonable number of clusters: {n_clusters}"

            # Regime detection should have reasonable confidence
            assert 0.3 <= regime_result['confidence'] <= 1.0, f"Unreasonable regime confidence: {regime_result['confidence']}"

            # Anomaly detection should have reasonable distribution
            anomaly_rate = np.mean(anomaly_result['anomaly_flags'])
            assert 0.01 <= anomaly_rate <= 0.3, f"Unreasonable anomaly rate: {anomaly_rate}"

        except Exception as e:
            validation_results['accuracy_requirements'] = False
            pytest.fail(f"Accuracy validation failed: {e}")

        # Scalability validation
        try:
            # Test with multiple data sizes
            for size in [100, 300, 500]:
                test_subset = test_data.head(size)
                result = clusterer.cluster_market_data(f'SCALE_{size}', test_subset)
                assert len(result['kmeans_clusters']) == size

        except Exception as e:
            validation_results['scalability_requirements'] = False
            pytest.fail(f"Scalability validation failed: {e}")

        # Reliability validation
        try:
            # Multiple runs should produce consistent results
            results = []
            for _ in range(3):
                result = regime_detector.detect_regime(test_data)
                results.append(result['regime'])

            # Should be consistent
            unique_regimes = set(results)
            assert len(unique_regimes) <= 2, f"Inconsistent regime detection: {unique_regimes}"

        except Exception as e:
            validation_results['reliability_requirements'] = False
            pytest.fail(f"Reliability validation failed: {e}")

        # All validations should pass
        assert all(validation_results.values()), f"Production validation failed: {validation_results}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])