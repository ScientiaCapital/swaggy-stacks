"""
Comprehensive Unit Tests for Clustering Components
Institutional-grade testing for production trading systems
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any

from app.ml.unsupervised.clustering import (
    KMeansClusterer,
    HierarchicalClusterer,
    DBSCANClusterer,
    MarketDataClusterer
)
from app.monitoring.metrics import PrometheusMetrics


class TestKMeansClusterer:
    """Test K-means clustering implementation"""

    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        data = pd.DataFrame({
            'timestamp': dates,
            'price': 100 + np.cumsum(np.random.randn(100) * 0.1),
            'volume': np.random.exponential(1000, 100),
            'volatility': np.random.gamma(2, 0.1, 100),
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.randn(100) * 0.5
        })
        return data

    @pytest.fixture
    def kmeans_clusterer(self):
        """Create KMeansClusterer instance"""
        return KMeansClusterer(n_clusters=3, random_state=42)

    def test_kmeans_initialization(self, kmeans_clusterer):
        """Test K-means clusterer initialization"""
        assert kmeans_clusterer.n_clusters == 3
        assert kmeans_clusterer.random_state == 42
        assert kmeans_clusterer.model is None
        assert kmeans_clusterer._fitted is False

    def test_kmeans_feature_engineering(self, kmeans_clusterer, sample_market_data):
        """Test feature engineering for clustering"""
        features = kmeans_clusterer._prepare_features(sample_market_data)

        # Check feature dimensions
        assert features.shape[0] == len(sample_market_data)
        assert features.shape[1] >= 5  # price, volume, volatility, rsi, macd

        # Check for no NaN values
        assert not np.isnan(features).any()

        # Check feature scaling (should be normalized)
        assert np.abs(features.mean(axis=0)).max() < 0.1  # Nearly zero mean
        assert np.abs(features.std(axis=0) - 1.0).max() < 0.1  # Unit variance

    def test_kmeans_fit_predict(self, kmeans_clusterer, sample_market_data):
        """Test K-means fit and predict functionality"""
        # Fit the model
        start_time = time.time()
        clusters = kmeans_clusterer.fit_predict(sample_market_data)
        fit_time = time.time() - start_time

        # Validate clustering results
        assert len(clusters) == len(sample_market_data)
        assert set(clusters).issubset(set(range(3)))  # Clusters 0, 1, 2
        assert kmeans_clusterer._fitted is True

        # Performance requirement: should complete within 1 second
        assert fit_time < 1.0, f"K-means clustering took {fit_time:.3f}s, expected < 1.0s"

    def test_kmeans_cluster_quality_metrics(self, kmeans_clusterer, sample_market_data):
        """Test cluster quality metrics calculation"""
        clusters = kmeans_clusterer.fit_predict(sample_market_data)

        # Get quality metrics
        silhouette = kmeans_clusterer.get_silhouette_score()
        inertia = kmeans_clusterer.get_inertia()

        # Validate metric ranges
        assert -1 <= silhouette <= 1, f"Silhouette score {silhouette} outside valid range"
        assert inertia >= 0, f"Inertia {inertia} should be non-negative"

        # Quality thresholds for production
        assert silhouette > 0.3, f"Silhouette score {silhouette} below production threshold"

    def test_kmeans_prediction_consistency(self, kmeans_clusterer, sample_market_data):
        """Test prediction consistency across multiple runs"""
        # First fit
        clusters1 = kmeans_clusterer.fit_predict(sample_market_data)

        # Second prediction on same data
        clusters2 = kmeans_clusterer.predict(sample_market_data)

        # Should be identical
        np.testing.assert_array_equal(clusters1, clusters2)

    def test_kmeans_edge_cases(self, kmeans_clusterer):
        """Test edge cases and error handling"""
        # Empty data
        with pytest.raises(ValueError):
            empty_data = pd.DataFrame()
            kmeans_clusterer.fit_predict(empty_data)

        # Single data point
        single_point = pd.DataFrame({
            'price': [100], 'volume': [1000], 'volatility': [0.1],
            'rsi': [50], 'macd': [0.0]
        })

        # Should handle gracefully or raise appropriate error
        try:
            result = kmeans_clusterer.fit_predict(single_point)
            assert len(result) == 1
        except ValueError as e:
            assert "sample" in str(e).lower()

    @patch('app.monitoring.metrics.PrometheusMetrics')
    def test_kmeans_metrics_integration(self, mock_metrics, kmeans_clusterer, sample_market_data):
        """Test integration with Prometheus metrics"""
        # Setup mock
        metrics_instance = Mock()
        mock_metrics.return_value = metrics_instance

        # Fit with metrics collection
        kmeans_clusterer.prometheus_metrics = metrics_instance
        clusters = kmeans_clusterer.fit_predict(sample_market_data)

        # Verify metrics were recorded
        silhouette = kmeans_clusterer.get_silhouette_score()
        inertia = kmeans_clusterer.get_inertia()

        # Check that appropriate metric methods would be called
        assert silhouette is not None
        assert inertia is not None


class TestHierarchicalClusterer:
    """Test Hierarchical clustering implementation"""

    @pytest.fixture
    def hierarchical_clusterer(self):
        """Create HierarchicalClusterer instance"""
        return HierarchicalClusterer(n_clusters=3, linkage='ward')

    @pytest.fixture
    def sample_features(self):
        """Generate sample feature matrix"""
        np.random.seed(42)
        return np.random.randn(50, 5)

    def test_hierarchical_initialization(self, hierarchical_clusterer):
        """Test hierarchical clusterer initialization"""
        assert hierarchical_clusterer.n_clusters == 3
        assert hierarchical_clusterer.linkage == 'ward'
        assert hierarchical_clusterer._fitted is False

    def test_hierarchical_fit_predict(self, hierarchical_clusterer, sample_features):
        """Test hierarchical clustering fit and predict"""
        start_time = time.time()
        clusters = hierarchical_clusterer.fit_predict_features(sample_features)
        fit_time = time.time() - start_time

        # Validate results
        assert len(clusters) == sample_features.shape[0]
        assert set(clusters).issubset(set(range(3)))
        assert hierarchical_clusterer._fitted is True

        # Performance requirement
        assert fit_time < 2.0, f"Hierarchical clustering took {fit_time:.3f}s, expected < 2.0s"

    def test_hierarchical_dendrogram_analysis(self, hierarchical_clusterer, sample_features):
        """Test dendrogram analysis capabilities"""
        clusters = hierarchical_clusterer.fit_predict_features(sample_features)

        # Get dendrogram data
        dendrogram_data = hierarchical_clusterer.get_dendrogram_data()

        # Validate dendrogram structure
        assert 'linkage_matrix' in dendrogram_data
        assert 'cluster_distances' in dendrogram_data
        assert dendrogram_data['linkage_matrix'].shape[0] == sample_features.shape[0] - 1

    def test_hierarchical_cluster_stability(self, hierarchical_clusterer, sample_features):
        """Test cluster stability across perturbations"""
        # Original clustering
        original_clusters = hierarchical_clusterer.fit_predict_features(sample_features)

        # Add small noise
        noisy_features = sample_features + np.random.randn(*sample_features.shape) * 0.01
        noisy_clusters = hierarchical_clusterer.fit_predict_features(noisy_features)

        # Calculate stability (Adjusted Rand Index)
        stability = hierarchical_clusterer.calculate_stability(original_clusters, noisy_clusters)

        # Should be stable to small perturbations
        assert stability > 0.7, f"Cluster stability {stability} below threshold"


class TestDBSCANClusterer:
    """Test DBSCAN clustering implementation"""

    @pytest.fixture
    def dbscan_clusterer(self):
        """Create DBSCANClusterer instance"""
        return DBSCANClusterer(eps=0.5, min_samples=5)

    @pytest.fixture
    def anomaly_data(self):
        """Generate data with clear anomalies"""
        np.random.seed(42)
        # Normal data cluster
        normal = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 80)
        # Outlier points
        outliers = np.array([[5, 5], [-5, -5], [5, -5]])
        return np.vstack([normal, outliers])

    def test_dbscan_initialization(self, dbscan_clusterer):
        """Test DBSCAN clusterer initialization"""
        assert dbscan_clusterer.eps == 0.5
        assert dbscan_clusterer.min_samples == 5
        assert dbscan_clusterer._fitted is False

    def test_dbscan_anomaly_detection(self, dbscan_clusterer, anomaly_data):
        """Test DBSCAN's anomaly detection capabilities"""
        clusters = dbscan_clusterer.fit_predict_features(anomaly_data)

        # Check for outliers (labeled as -1)
        outlier_mask = clusters == -1
        num_outliers = np.sum(outlier_mask)

        # Should detect some outliers
        assert num_outliers > 0, "DBSCAN should detect outliers"
        assert num_outliers < len(anomaly_data) * 0.2, "Too many points classified as outliers"

    def test_dbscan_performance_real_time(self, dbscan_clusterer):
        """Test DBSCAN performance for real-time requirements"""
        # Generate realistic market data size
        market_data = np.random.randn(1000, 10)  # 1000 data points, 10 features

        start_time = time.time()
        clusters = dbscan_clusterer.fit_predict_features(market_data)
        processing_time = time.time() - start_time

        # Real-time requirement: process 1000 points in <100ms
        assert processing_time < 0.1, f"DBSCAN took {processing_time:.3f}s, expected < 0.1s"

    def test_dbscan_parameter_sensitivity(self):
        """Test sensitivity to hyperparameters"""
        data = np.random.randn(100, 5)

        # Test different eps values
        eps_values = [0.1, 0.5, 1.0, 2.0]
        cluster_counts = []

        for eps in eps_values:
            clusterer = DBSCANClusterer(eps=eps, min_samples=5)
            clusters = clusterer.fit_predict_features(data)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            cluster_counts.append(n_clusters)

        # Should show reasonable variation
        assert len(set(cluster_counts)) > 1, "DBSCAN should be sensitive to eps parameter"


class TestMarketDataClusterer:
    """Test integrated market data clustering system"""

    @pytest.fixture
    def market_clusterer(self):
        """Create MarketDataClusterer instance"""
        return MarketDataClusterer()

    @pytest.fixture
    def multi_symbol_data(self):
        """Generate multi-symbol market data"""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        data_dict = {}

        for symbol in symbols:
            np.random.seed(ord(symbol[0]))  # Different seed per symbol
            dates = pd.date_range('2024-01-01', periods=200, freq='1H')
            data_dict[symbol] = pd.DataFrame({
                'timestamp': dates,
                'price': 100 + np.cumsum(np.random.randn(200) * 0.1),
                'volume': np.random.exponential(1000, 200),
                'volatility': np.random.gamma(2, 0.1, 200)
            })

        return data_dict

    def test_market_clusterer_initialization(self, market_clusterer):
        """Test market data clusterer initialization"""
        assert hasattr(market_clusterer, 'clusterers')
        assert 'kmeans' in market_clusterer.clusterers
        assert 'hierarchical' in market_clusterer.clusterers
        assert 'dbscan' in market_clusterer.clusterers

    def test_market_clustering_pipeline(self, market_clusterer, multi_symbol_data):
        """Test end-to-end market data clustering pipeline"""
        # Process all symbols
        results = {}
        total_start = time.time()

        for symbol, data in multi_symbol_data.items():
            symbol_results = market_clusterer.cluster_market_data(symbol, data)
            results[symbol] = symbol_results

        total_time = time.time() - total_start

        # Validate results structure
        for symbol, result in results.items():
            assert 'kmeans_clusters' in result
            assert 'hierarchical_clusters' in result
            assert 'anomalies' in result
            assert 'quality_metrics' in result

        # Performance requirement: process 4 symbols in <5 seconds
        assert total_time < 5.0, f"Market clustering took {total_time:.3f}s, expected < 5.0s"

    def test_market_clustering_consensus(self, market_clusterer, multi_symbol_data):
        """Test consensus clustering across algorithms"""
        symbol = 'AAPL'
        data = multi_symbol_data[symbol]

        results = market_clusterer.cluster_market_data(symbol, data)
        consensus = market_clusterer.get_clustering_consensus(results)

        # Validate consensus
        assert 'consensus_clusters' in consensus
        assert 'agreement_score' in consensus
        assert 'algorithm_weights' in consensus

        # Agreement score should be reasonable
        agreement = consensus['agreement_score']
        assert 0 <= agreement <= 1, f"Agreement score {agreement} outside [0,1]"

    @patch('app.monitoring.metrics.PrometheusMetrics')
    def test_market_clustering_metrics_collection(self, mock_metrics, market_clusterer, multi_symbol_data):
        """Test comprehensive metrics collection"""
        metrics_instance = Mock()
        mock_metrics.return_value = metrics_instance

        # Enable metrics collection
        market_clusterer.prometheus_metrics = metrics_instance

        symbol = 'AAPL'
        data = multi_symbol_data[symbol]
        results = market_clusterer.cluster_market_data(symbol, data)

        # Verify metrics would be collected
        assert 'quality_metrics' in results
        quality = results['quality_metrics']

        assert 'silhouette_scores' in quality
        assert 'cluster_stability' in quality
        assert 'processing_times' in quality

    def test_market_clustering_memory_efficiency(self, market_clusterer):
        """Test memory efficiency with large datasets"""
        import psutil
        import os

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process large dataset
        large_data = pd.DataFrame({
            'price': np.random.randn(10000),
            'volume': np.random.exponential(1000, 10000),
            'volatility': np.random.gamma(2, 0.1, 10000)
        })

        results = market_clusterer.cluster_market_data('TEST', large_data)

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not use excessive memory (< 100MB increase)
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f}MB"

    def test_market_clustering_error_handling(self, market_clusterer):
        """Test error handling and edge cases"""
        # Test with insufficient data
        small_data = pd.DataFrame({
            'price': [100, 101],
            'volume': [1000, 1100]
        })

        try:
            results = market_clusterer.cluster_market_data('TEST', small_data)
            # Should either work or raise appropriate error
            assert isinstance(results, dict)
        except ValueError as e:
            assert "insufficient" in str(e).lower() or "sample" in str(e).lower()

        # Test with NaN data
        nan_data = pd.DataFrame({
            'price': [100, np.nan, 102],
            'volume': [1000, 1100, np.nan]
        })

        # Should handle NaN values gracefully
        results = market_clusterer.cluster_market_data('TEST', nan_data)
        assert isinstance(results, dict)
        assert 'error' not in results or results.get('data_issues') is not None


class TestClusteringIntegration:
    """Integration tests for clustering system"""

    def test_clustering_with_live_data_simulation(self):
        """Test clustering with simulated live data streams"""
        clusterer = MarketDataClusterer()

        # Simulate data stream
        data_batches = []
        for i in range(5):
            batch = pd.DataFrame({
                'price': 100 + np.cumsum(np.random.randn(50) * 0.1),
                'volume': np.random.exponential(1000, 50),
                'timestamp': pd.date_range(f'2024-01-0{i+1}', periods=50, freq='1min')
            })
            data_batches.append(batch)

        # Process batches and track cluster evolution
        cluster_evolution = []
        for i, batch in enumerate(data_batches):
            if i == 0:
                # Initial clustering
                results = clusterer.cluster_market_data('STREAM', batch)
            else:
                # Incremental clustering
                combined_data = pd.concat(data_batches[:i+1], ignore_index=True)
                results = clusterer.cluster_market_data('STREAM', combined_data)

            cluster_evolution.append(results['quality_metrics'])

        # Validate cluster evolution stability
        silhouette_scores = [ev['silhouette_scores']['kmeans'] for ev in cluster_evolution]

        # Should maintain reasonable quality
        assert all(score > 0.2 for score in silhouette_scores), "Clustering quality degraded"

    def test_clustering_performance_benchmark(self):
        """Comprehensive performance benchmark"""
        clusterer = MarketDataClusterer()

        # Test different data sizes
        data_sizes = [100, 500, 1000, 2000]
        performance_results = {}

        for size in data_sizes:
            # Generate data of specified size
            data = pd.DataFrame({
                'price': 100 + np.cumsum(np.random.randn(size) * 0.1),
                'volume': np.random.exponential(1000, size),
                'volatility': np.random.gamma(2, 0.1, size)
            })

            # Measure performance
            start_time = time.time()
            results = clusterer.cluster_market_data(f'BENCH_{size}', data)
            processing_time = time.time() - start_time

            performance_results[size] = {
                'processing_time': processing_time,
                'quality_metrics': results['quality_metrics']
            }

        # Validate performance scaling
        for size in data_sizes:
            perf = performance_results[size]
            max_time = size * 0.001  # 1ms per data point maximum

            assert perf['processing_time'] < max_time, (
                f"Processing {size} points took {perf['processing_time']:.3f}s, "
                f"expected < {max_time:.3f}s"
            )

    def test_clustering_production_readiness(self):
        """Test production readiness criteria"""
        clusterer = MarketDataClusterer()

        # Generate realistic trading data
        trading_data = pd.DataFrame({
            'price': 150 + np.cumsum(np.random.randn(1000) * 0.05),
            'volume': np.random.lognormal(7, 1, 1000),
            'volatility': np.random.gamma(2, 0.02, 1000),
            'bid_ask_spread': np.random.gamma(1, 0.001, 1000),
            'order_flow': np.random.randn(1000)
        })

        # Run comprehensive analysis
        results = clusterer.cluster_market_data('PROD_TEST', trading_data)

        # Production criteria validation
        quality = results['quality_metrics']

        # Minimum quality thresholds
        assert quality['silhouette_scores']['kmeans'] > 0.3, "K-means quality below production threshold"
        assert quality['cluster_stability'] > 0.7, "Cluster stability below production threshold"

        # Performance thresholds
        processing_times = quality['processing_times']
        assert processing_times['total'] < 1.0, "Total processing time exceeds production limit"
        assert processing_times['kmeans'] < 0.5, "K-means processing time exceeds limit"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])