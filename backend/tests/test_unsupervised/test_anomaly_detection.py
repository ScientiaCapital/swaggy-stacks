"""
Comprehensive Unit Tests for Anomaly Detection System
Critical for detecting black swan events and market irregularities
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Tuple

from app.ml.unsupervised.anomaly_detector import (
    MarketAnomalyDetector,
    DBSCANAnomalyDetector,
    IsolationForestAnomalyDetector,
    EnsembleAnomalyDetector
)
from app.monitoring.metrics import PrometheusMetrics


class TestMarketAnomalyDetector:
    """Test core market anomaly detection functionality"""

    @pytest.fixture
    def anomaly_detector(self):
        """Create MarketAnomalyDetector instance"""
        return MarketAnomalyDetector(
            contamination=0.1,
            sensitivity_threshold=0.8,
            alert_threshold=0.9
        )

    @pytest.fixture
    def normal_market_data(self):
        """Generate normal market behavior data"""
        np.random.seed(42)
        n_points = 1000
        dates = pd.date_range('2024-01-01', periods=n_points, freq='5min')

        # Normal market: stable returns, moderate volume
        returns = np.random.normal(0.0001, 0.005, n_points)
        prices = 100 * np.exp(np.cumsum(returns))
        volumes = np.random.lognormal(7, 0.3, n_points)
        volatility = np.random.gamma(1, 0.005, n_points)

        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': volumes,
            'volatility': volatility,
            'returns': returns
        })

    @pytest.fixture
    def anomalous_market_data(self, normal_market_data):
        """Generate market data with known anomalies"""
        data = normal_market_data.copy()

        # Inject specific anomalies
        anomaly_indices = [100, 300, 500, 750]

        for idx in anomaly_indices:
            if idx < len(data):
                # Flash crash: sudden large drop
                if idx == 100:
                    data.loc[idx, 'returns'] = -0.05  # 5% drop
                    data.loc[idx, 'volume'] = data.loc[idx, 'volume'] * 10  # 10x volume spike

                # Volume spike anomaly
                elif idx == 300:
                    data.loc[idx, 'volume'] = data.loc[idx, 'volume'] * 20

                # Volatility spike
                elif idx == 500:
                    data.loc[idx, 'volatility'] = data.loc[idx, 'volatility'] * 15

                # Price gap
                elif idx == 750:
                    data.loc[idx, 'returns'] = 0.03  # 3% gap up
                    data.loc[idx, 'volume'] = data.loc[idx, 'volume'] * 5

        # Recalculate prices after anomalies
        data['price'] = 100 * np.exp(data['returns'].cumsum())

        return data, anomaly_indices

    @pytest.fixture
    def black_swan_event_data(self, normal_market_data):
        """Generate data with black swan event"""
        data = normal_market_data.copy()

        # Black swan: massive coordinated drop
        black_swan_start = 400
        black_swan_duration = 20

        for i in range(black_swan_duration):
            idx = black_swan_start + i
            if idx < len(data):
                data.loc[idx, 'returns'] = np.random.normal(-0.02, 0.01)  # 2% average drop
                data.loc[idx, 'volume'] = data.loc[idx, 'volume'] * np.random.uniform(5, 15)
                data.loc[idx, 'volatility'] = data.loc[idx, 'volatility'] * np.random.uniform(8, 20)

        # Recalculate prices
        data['price'] = 100 * np.exp(data['returns'].cumsum())

        return data, (black_swan_start, black_swan_start + black_swan_duration)

    def test_anomaly_detector_initialization(self, anomaly_detector):
        """Test anomaly detector initialization"""
        assert anomaly_detector.contamination == 0.1
        assert anomaly_detector.sensitivity_threshold == 0.8
        assert anomaly_detector.alert_threshold == 0.9
        assert anomaly_detector._fitted is False

    def test_feature_engineering_for_anomaly_detection(self, anomaly_detector, normal_market_data):
        """Test feature engineering for anomaly detection"""
        features = anomaly_detector._engineer_anomaly_features(normal_market_data)

        # Check feature structure
        expected_features = [
            'return_magnitude', 'volume_zscore', 'volatility_spike',
            'price_gap', 'volume_return_correlation', 'momentum_divergence'
        ]

        for feature in expected_features:
            assert feature in features.columns, f"Missing feature: {feature}"

        # Validate feature quality
        assert not features.isnull().any().any(), "Features contain NaN values"
        assert features.shape[0] == len(normal_market_data), "Feature count mismatch"

        # Features should be normalized
        for col in features.columns:
            assert features[col].std() > 0, f"Zero variance in feature {col}"

    def test_normal_market_anomaly_detection(self, anomaly_detector, normal_market_data):
        """Test anomaly detection on normal market data"""
        start_time = time.time()
        result = anomaly_detector.detect_anomalies(normal_market_data)
        detection_time = time.time() - start_time

        # Validate result structure
        assert 'anomaly_scores' in result
        assert 'anomaly_flags' in result
        assert 'severity_levels' in result
        assert 'anomaly_types' in result

        # Check dimensions
        assert len(result['anomaly_scores']) == len(normal_market_data)
        assert len(result['anomaly_flags']) == len(normal_market_data)

        # Normal data should have few anomalies
        anomaly_rate = np.mean(result['anomaly_flags'])
        assert anomaly_rate < 0.15, f"High anomaly rate {anomaly_rate} in normal data"

        # Performance requirement
        assert detection_time < 0.5, f"Anomaly detection took {detection_time:.3f}s, expected < 0.5s"

    def test_anomalous_market_detection(self, anomaly_detector, anomalous_market_data):
        """Test detection of known anomalies"""
        data, anomaly_indices = anomalous_market_data

        result = anomaly_detector.detect_anomalies(data)

        # Should detect injected anomalies
        detected_anomalies = np.where(result['anomaly_flags'])[0]
        anomaly_scores = result['anomaly_scores']

        # Check detection of known anomalies
        detection_success = 0
        for idx in anomaly_indices:
            # Allow detection within small window (±2 points)
            if any(abs(detected - idx) <= 2 for detected in detected_anomalies):
                detection_success += 1

        detection_rate = detection_success / len(anomaly_indices)
        assert detection_rate >= 0.75, f"Low anomaly detection rate: {detection_rate}"

        # Scores at anomaly points should be high
        anomaly_point_scores = [anomaly_scores[idx] for idx in anomaly_indices if idx < len(anomaly_scores)]
        avg_anomaly_score = np.mean(anomaly_point_scores)
        assert avg_anomaly_score > 0.7, f"Low anomaly scores at known anomaly points: {avg_anomaly_score}"

    def test_black_swan_event_detection(self, anomaly_detector, black_swan_event_data):
        """Test detection of black swan events"""
        data, (start_idx, end_idx) = black_swan_event_data

        result = anomaly_detector.detect_anomalies(data)

        # Should detect black swan period
        black_swan_region = result['anomaly_flags'][start_idx:end_idx]
        detection_rate_in_region = np.mean(black_swan_region)

        assert detection_rate_in_region > 0.6, f"Low black swan detection rate: {detection_rate_in_region}"

        # Scores should be very high during black swan
        black_swan_scores = result['anomaly_scores'][start_idx:end_idx]
        avg_black_swan_score = np.mean(black_swan_scores)
        assert avg_black_swan_score > 0.8, f"Low scores during black swan: {avg_black_swan_score}"

        # Should trigger high severity alerts
        severity_levels = result['severity_levels'][start_idx:end_idx]
        high_severity_count = np.sum(np.array(severity_levels) >= 2)  # Assuming 0=low, 1=medium, 2=high
        assert high_severity_count > 0, "No high severity alerts during black swan"

    def test_anomaly_type_classification(self, anomaly_detector, anomalous_market_data):
        """Test classification of anomaly types"""
        data, anomaly_indices = anomalous_market_data

        result = anomaly_detector.detect_anomalies(data)
        anomaly_types = result['anomaly_types']

        # Should classify different types of anomalies
        unique_types = set(type_name for type_name in anomaly_types if type_name != 'normal')
        assert len(unique_types) > 1, f"Only detected one anomaly type: {unique_types}"

        # Expected anomaly types
        expected_types = ['price_anomaly', 'volume_anomaly', 'volatility_anomaly']
        detected_types = list(unique_types)

        # Should detect at least some expected types
        type_overlap = len(set(expected_types) & set(detected_types))
        assert type_overlap > 0, f"No expected anomaly types detected. Got: {detected_types}"

    def test_anomaly_detection_consistency(self, anomaly_detector, normal_market_data):
        """Test consistency of anomaly detection across multiple runs"""
        results = []

        # Run detection multiple times
        for _ in range(5):
            result = anomaly_detector.detect_anomalies(normal_market_data)
            results.append(result['anomaly_flags'])

        # Check consistency
        all_flags = np.array(results)
        consistency_score = np.mean(np.std(all_flags.astype(int), axis=0))

        # Should be consistent (low standard deviation)
        assert consistency_score < 0.2, f"High inconsistency in anomaly detection: {consistency_score}"

    def test_real_time_anomaly_detection(self, anomaly_detector):
        """Test real-time anomaly detection performance"""
        # Simulate real-time data stream
        batch_size = 50
        num_batches = 20

        processing_times = []

        for batch_num in range(num_batches):
            # Generate new batch
            returns = np.random.normal(0.0001, 0.005, batch_size)
            prices = 100 * np.exp(np.cumsum(returns))
            volumes = np.random.lognormal(7, 0.3, batch_size)

            batch_data = pd.DataFrame({
                'price': prices,
                'volume': volumes,
                'returns': returns,
                'volatility': np.abs(returns)
            })

            # Measure processing time
            start_time = time.time()
            result = anomaly_detector.detect_anomalies(batch_data)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Validate result
            assert 'anomaly_scores' in result
            assert len(result['anomaly_scores']) == batch_size

        # Performance requirements for real-time trading
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)

        assert avg_processing_time < 0.1, f"Average processing time {avg_processing_time:.3f}s too high"
        assert max_processing_time < 0.2, f"Max processing time {max_processing_time:.3f}s too high"

    @patch('app.monitoring.metrics.PrometheusMetrics')
    def test_anomaly_detection_metrics_integration(self, mock_metrics, anomaly_detector, anomalous_market_data):
        """Test integration with Prometheus metrics"""
        metrics_instance = Mock()
        mock_metrics.return_value = metrics_instance

        anomaly_detector.prometheus_metrics = metrics_instance

        data, anomaly_indices = anomalous_market_data
        result = anomaly_detector.detect_anomalies(data)

        # Verify metrics data is available
        assert 'anomaly_scores' in result
        assert 'anomaly_flags' in result
        assert 'severity_levels' in result

        # Calculate metrics that would be reported
        anomaly_rate = np.mean(result['anomaly_flags'])
        avg_anomaly_score = np.mean(result['anomaly_scores'])
        high_severity_count = np.sum(np.array(result['severity_levels']) >= 2)

        assert 0 <= anomaly_rate <= 1
        assert 0 <= avg_anomaly_score <= 1
        assert high_severity_count >= 0


class TestDBSCANAnomalyDetector:
    """Test DBSCAN-based anomaly detection"""

    @pytest.fixture
    def dbscan_detector(self):
        """Create DBSCAN anomaly detector"""
        return DBSCANAnomalyDetector(eps=0.5, min_samples=5)

    @pytest.fixture
    def clustered_data_with_outliers(self):
        """Generate clustered data with clear outliers"""
        np.random.seed(42)

        # Main cluster
        cluster1 = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], 200)
        cluster2 = np.random.multivariate_normal([5, 5], [[1, -0.2], [-0.2, 1]], 150)

        # Outliers
        outliers = np.array([
            [10, 10], [-5, -5], [15, -3], [-2, 12], [8, -8]
        ])

        # Combine
        all_data = np.vstack([cluster1, cluster2, outliers])
        labels = ['normal'] * 350 + ['outlier'] * 5

        return pd.DataFrame(all_data, columns=['feature1', 'feature2']), labels

    def test_dbscan_detector_initialization(self, dbscan_detector):
        """Test DBSCAN detector initialization"""
        assert dbscan_detector.eps == 0.5
        assert dbscan_detector.min_samples == 5

    def test_dbscan_outlier_detection(self, dbscan_detector, clustered_data_with_outliers):
        """Test DBSCAN outlier detection capability"""
        data, true_labels = clustered_data_with_outliers

        result = dbscan_detector.detect_anomalies(data)

        # Should detect outliers
        detected_outliers = result['anomaly_flags']
        num_detected = np.sum(detected_outliers)

        assert num_detected > 0, "No outliers detected"
        assert num_detected < len(data) * 0.2, "Too many points classified as outliers"

        # Check detection of true outliers (last 5 points)
        true_outlier_indices = list(range(len(data) - 5, len(data)))
        detected_true_outliers = sum(detected_outliers[idx] for idx in true_outlier_indices)

        detection_rate = detected_true_outliers / 5
        assert detection_rate >= 0.6, f"Low true outlier detection rate: {detection_rate}"

    def test_dbscan_parameter_sensitivity(self, dbscan_detector):
        """Test sensitivity to DBSCAN parameters"""
        # Generate test data
        data = np.random.randn(200, 3)

        # Test different eps values
        eps_values = [0.3, 0.5, 0.8, 1.2]
        outlier_counts = []

        for eps in eps_values:
            detector = DBSCANAnomalyDetector(eps=eps, min_samples=5)
            result = detector.detect_anomalies(pd.DataFrame(data))
            outlier_count = np.sum(result['anomaly_flags'])
            outlier_counts.append(outlier_count)

        # Should show variation with parameter changes
        assert len(set(outlier_counts)) > 1, "DBSCAN not sensitive to eps parameter"


class TestIsolationForestAnomalyDetector:
    """Test Isolation Forest anomaly detection"""

    @pytest.fixture
    def isolation_forest_detector(self):
        """Create Isolation Forest detector"""
        return IsolationForestAnomalyDetector(
            contamination=0.1,
            n_estimators=100,
            random_state=42
        )

    def test_isolation_forest_initialization(self, isolation_forest_detector):
        """Test Isolation Forest initialization"""
        assert isolation_forest_detector.contamination == 0.1
        assert isolation_forest_detector.n_estimators == 100
        assert isolation_forest_detector.random_state == 42

    def test_isolation_forest_anomaly_detection(self, isolation_forest_detector, normal_market_data):
        """Test Isolation Forest anomaly detection"""
        result = isolation_forest_detector.detect_anomalies(normal_market_data)

        # Validate result structure
        assert 'anomaly_scores' in result
        assert 'anomaly_flags' in result

        # Check contamination rate
        anomaly_rate = np.mean(result['anomaly_flags'])
        expected_rate = isolation_forest_detector.contamination

        # Should be close to expected contamination rate
        assert abs(anomaly_rate - expected_rate) < 0.05, f"Anomaly rate {anomaly_rate} far from expected {expected_rate}"

    def test_isolation_forest_score_distribution(self, isolation_forest_detector):
        """Test score distribution properties"""
        # Generate mixed normal and anomalous data
        normal_data = np.random.randn(800, 5)
        anomalous_data = np.random.randn(200, 5) * 3 + 5  # Shifted and scaled

        combined_data = pd.DataFrame(np.vstack([normal_data, anomalous_data]))
        result = isolation_forest_detector.detect_anomalies(combined_data)

        scores = result['anomaly_scores']

        # Scores should use full range
        assert np.min(scores) < 0.3, "Minimum score too high"
        assert np.max(scores) > 0.7, "Maximum score too low"

        # Anomalous data (last 200 points) should have higher scores on average
        normal_scores = scores[:800]
        anomalous_scores = scores[800:]

        assert np.mean(anomalous_scores) > np.mean(normal_scores), "Anomalous data doesn't have higher scores"


class TestEnsembleAnomalyDetector:
    """Test ensemble anomaly detection combining multiple methods"""

    @pytest.fixture
    def ensemble_detector(self):
        """Create ensemble anomaly detector"""
        return EnsembleAnomalyDetector(
            methods=['dbscan', 'isolation_forest', 'statistical'],
            voting_strategy='soft',
            confidence_threshold=0.7
        )

    def test_ensemble_initialization(self, ensemble_detector):
        """Test ensemble detector initialization"""
        assert 'dbscan' in ensemble_detector.methods
        assert 'isolation_forest' in ensemble_detector.methods
        assert ensemble_detector.voting_strategy == 'soft'
        assert ensemble_detector.confidence_threshold == 0.7

    def test_ensemble_anomaly_detection(self, ensemble_detector, anomalous_market_data):
        """Test ensemble anomaly detection"""
        data, anomaly_indices = anomalous_market_data

        result = ensemble_detector.detect_anomalies(data)

        # Validate ensemble result structure
        assert 'anomaly_scores' in result
        assert 'anomaly_flags' in result
        assert 'method_votes' in result
        assert 'confidence_scores' in result

        # Check method contributions
        method_votes = result['method_votes']
        assert isinstance(method_votes, dict)
        assert len(method_votes) == len(ensemble_detector.methods)

        # Should detect known anomalies with high confidence
        confidence_scores = result['confidence_scores']
        anomaly_flags = result['anomaly_flags']

        detected_anomalies = np.where(anomaly_flags)[0]
        for detected_idx in detected_anomalies:
            confidence = confidence_scores[detected_idx]
            assert confidence >= ensemble_detector.confidence_threshold, f"Low confidence {confidence} for detected anomaly"

    def test_ensemble_consensus_accuracy(self, ensemble_detector):
        """Test ensemble consensus improves accuracy"""
        # Generate data with clear anomalies
        np.random.seed(42)
        normal_data = np.random.randn(500, 4)
        anomaly_data = np.random.randn(50, 4) * 4 + 6  # Clear anomalies

        combined_data = pd.DataFrame(np.vstack([normal_data, anomaly_data]))
        true_anomalies = np.zeros(550)
        true_anomalies[500:] = 1  # Last 50 are anomalies

        result = ensemble_detector.detect_anomalies(combined_data)
        predicted_anomalies = result['anomaly_flags'].astype(int)

        # Calculate precision and recall
        true_positives = np.sum((true_anomalies == 1) & (predicted_anomalies == 1))
        false_positives = np.sum((true_anomalies == 0) & (predicted_anomalies == 1))
        false_negatives = np.sum((true_anomalies == 1) & (predicted_anomalies == 0))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        # Ensemble should achieve reasonable performance
        assert precision > 0.5, f"Low precision: {precision}"
        assert recall > 0.4, f"Low recall: {recall}"

    def test_ensemble_method_weights(self, ensemble_detector, normal_market_data):
        """Test ensemble method weighting and voting"""
        result = ensemble_detector.detect_anomalies(normal_market_data)

        method_votes = result['method_votes']
        confidence_scores = result['confidence_scores']

        # All methods should contribute
        for method, votes in method_votes.items():
            assert len(votes) == len(normal_market_data), f"Missing votes from {method}"
            assert not np.isnan(votes).any(), f"NaN votes from {method}"

        # Confidence scores should be reasonable
        assert np.all((confidence_scores >= 0) & (confidence_scores <= 1)), "Confidence scores out of range"

    def test_ensemble_performance_scaling(self, ensemble_detector):
        """Test ensemble performance with different data sizes"""
        data_sizes = [100, 500, 1000, 2000]
        processing_times = []

        for size in data_sizes:
            # Generate test data
            data = pd.DataFrame(np.random.randn(size, 5))

            start_time = time.time()
            result = ensemble_detector.detect_anomalies(data)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Validate result
            assert len(result['anomaly_scores']) == size

        # Check performance scaling
        for i, (size, time_taken) in enumerate(zip(data_sizes, processing_times)):
            max_allowed_time = size * 0.002  # 2ms per data point
            assert time_taken < max_allowed_time, f"Processing {size} points took {time_taken:.3f}s"


class TestAnomalyDetectionIntegration:
    """Integration tests for complete anomaly detection system"""

    def test_multi_symbol_anomaly_detection(self):
        """Test anomaly detection across multiple symbols"""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        detector = EnsembleAnomalyDetector()

        symbol_results = {}

        for symbol in symbols:
            # Generate symbol-specific data
            np.random.seed(ord(symbol[0]))  # Different seed per symbol
            data = pd.DataFrame({
                'price': 100 + np.cumsum(np.random.randn(200) * 0.01),
                'volume': np.random.lognormal(7, 0.5, 200),
                'volatility': np.random.gamma(1, 0.01, 200)
            })

            # Add symbol-specific anomaly
            anomaly_idx = 100 + ord(symbol[0]) % 50
            data.loc[anomaly_idx, 'volume'] *= 10

            result = detector.detect_anomalies(data)
            symbol_results[symbol] = result

        # Validate results for all symbols
        for symbol, result in symbol_results.items():
            assert 'anomaly_scores' in result
            assert len(result['anomaly_scores']) == 200

            # Should detect at least one anomaly per symbol
            anomaly_count = np.sum(result['anomaly_flags'])
            assert anomaly_count > 0, f"No anomalies detected for {symbol}"

    def test_anomaly_detection_pipeline_stress_test(self):
        """Stress test anomaly detection pipeline"""
        detector = EnsembleAnomalyDetector()

        # Large dataset stress test
        large_data = pd.DataFrame({
            'price': 100 + np.cumsum(np.random.randn(10000) * 0.01),
            'volume': np.random.lognormal(7, 0.5, 10000),
            'volatility': np.random.gamma(1, 0.01, 10000),
            'bid_ask_spread': np.random.gamma(0.5, 0.001, 10000)
        })

        # Add random anomalies
        anomaly_indices = np.random.choice(10000, 100, replace=False)
        for idx in anomaly_indices:
            large_data.loc[idx, 'volume'] *= np.random.uniform(5, 20)

        start_time = time.time()
        result = detector.detect_anomalies(large_data)
        processing_time = time.time() - start_time

        # Should handle large datasets efficiently
        assert processing_time < 10.0, f"Large dataset processing took {processing_time:.2f}s"

        # Should detect reasonable number of anomalies
        detected_count = np.sum(result['anomaly_flags'])
        detection_rate = detected_count / len(anomaly_indices)
        assert detection_rate > 0.3, f"Low detection rate on large dataset: {detection_rate}"

    def test_anomaly_detection_memory_efficiency(self):
        """Test memory efficiency of anomaly detection"""
        import psutil
        import os

        detector = EnsembleAnomalyDetector()
        process = psutil.Process(os.getpid())

        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process multiple batches
        for _ in range(10):
            data = pd.DataFrame(np.random.randn(1000, 6))
            result = detector.detect_anomalies(data)

        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not leak memory excessively
        assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB"

    @patch('app.monitoring.metrics.PrometheusMetrics')
    def test_anomaly_detection_production_monitoring(self, mock_metrics):
        """Test production monitoring integration"""
        metrics_instance = Mock()
        mock_metrics.return_value = metrics_instance

        detector = EnsembleAnomalyDetector()
        detector.prometheus_metrics = metrics_instance

        # Simulate production data stream
        for batch_num in range(5):
            data = pd.DataFrame({
                'price': 100 + np.cumsum(np.random.randn(100) * 0.01),
                'volume': np.random.lognormal(7, 0.3, 100)
            })

            # Add occasional anomaly
            if batch_num % 3 == 0:
                data.loc[50, 'volume'] *= 15

            result = detector.detect_anomalies(data)

            # Validate monitoring data
            assert 'anomaly_scores' in result
            assert 'anomaly_flags' in result

            # Calculate production metrics
            anomaly_rate = np.mean(result['anomaly_flags'])
            avg_score = np.mean(result['anomaly_scores'])
            max_score = np.max(result['anomaly_scores'])

            # Metrics should be valid
            assert 0 <= anomaly_rate <= 1
            assert 0 <= avg_score <= 1
            assert 0 <= max_score <= 1

    def test_anomaly_detection_production_readiness(self):
        """Test production readiness criteria"""
        detector = EnsembleAnomalyDetector()

        # Test various market scenarios
        scenarios = [
            ('normal_trading', np.random.normal(0, 0.01, 500)),
            ('high_volatility', np.random.normal(0, 0.03, 500)),
            ('trending_market', np.cumsum(np.random.normal(0.001, 0.01, 500))),
            ('flash_crash', self._generate_flash_crash_scenario())
        ]

        for scenario_name, returns in scenarios:
            # Create market data
            prices = 100 * np.exp(np.cumsum(returns) if scenario_name != 'trending_market' else returns)
            data = pd.DataFrame({
                'price': prices,
                'returns': np.diff(prices) / prices[:-1] if len(prices) > 1 else [0],
                'volume': np.random.lognormal(7, 0.5, len(prices))
            })

            # Should handle all scenarios without errors
            try:
                result = detector.detect_anomalies(data)

                # Validate production requirements
                assert 'anomaly_scores' in result
                assert len(result['anomaly_scores']) == len(data)

                # Scores should be valid
                scores = result['anomaly_scores']
                assert np.all((scores >= 0) & (scores <= 1)), f"Invalid scores in {scenario_name}"

            except Exception as e:
                pytest.fail(f"Anomaly detection failed for {scenario_name}: {e}")

    def _generate_flash_crash_scenario(self) -> np.ndarray:
        """Generate flash crash scenario data"""
        returns = np.random.normal(0.0001, 0.01, 500)

        # Inject flash crash
        crash_start = 200
        crash_duration = 10

        for i in range(crash_duration):
            returns[crash_start + i] = np.random.normal(-0.05, 0.02)

        return returns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])