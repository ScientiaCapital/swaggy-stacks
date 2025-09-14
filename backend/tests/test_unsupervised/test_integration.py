"""
Comprehensive Integration Tests for Unsupervised Learning System
End-to-end testing for production trading system validation
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Tuple

from app.ml.unsupervised.clustering import MarketDataClusterer
from app.ml.unsupervised.market_regime import MarketRegimeDetector
from app.ml.unsupervised.anomaly_detector import EnsembleAnomalyDetector
from app.ml.unsupervised.pattern_memory import PatternMemorySystem
from app.ml.unsupervised.agent_memory import ExperienceClusteringSystem
from app.ml.unsupervised.strategy_evolution import StrategyEvolutionSystem
from app.monitoring.metrics import PrometheusMetrics


class TestUnsupervisedSystemIntegration:
    """Test complete unsupervised learning system integration"""

    @pytest.fixture
    def integrated_system(self):
        """Create integrated unsupervised learning system"""
        return {
            'clusterer': MarketDataClusterer(),
            'regime_detector': MarketRegimeDetector(),
            'anomaly_detector': EnsembleAnomalyDetector(),
            'pattern_memory': PatternMemorySystem(),
            'experience_clustering': ExperienceClusteringSystem(),
            'strategy_evolution': StrategyEvolutionSystem()
        }

    @pytest.fixture
    def comprehensive_market_data(self):
        """Generate comprehensive market data for integration testing"""
        np.random.seed(42)

        # 30 days of 5-minute data
        n_points = 30 * 24 * 12  # 8640 data points
        timestamps = pd.date_range('2024-01-01', periods=n_points, freq='5min')

        # Multi-phase market simulation
        phases = [
            (2880, 0.0005, 0.01),    # Bull phase (10 days)
            (1440, -0.001, 0.02),   # Bear phase (5 days)
            (2160, 0, 0.008),       # Sideways phase (7.5 days)
            (720, 0.002, 0.03),     # Volatile phase (2.5 days)
            (1440, 0.0008, 0.012)   # Recovery phase (5 days)
        ]

        returns = []
        for duration, drift, volatility in phases:
            phase_returns = np.random.normal(drift, volatility, duration)
            returns.extend(phase_returns)

        # Calculate prices
        prices = 100 * np.exp(np.cumsum(returns))

        # Generate correlated features
        volumes = np.exp(7 + 0.5 * np.abs(returns) + np.random.normal(0, 0.3, len(returns)))
        volatility = np.abs(returns) + np.random.gamma(1, 0.005, len(returns))

        # Technical indicators
        rsi = 50 + 25 * np.sin(np.linspace(0, 20*np.pi, len(returns))) + np.random.normal(0, 10, len(returns))
        rsi = np.clip(rsi, 0, 100)

        macd = np.random.randn(len(returns)) * 0.5

        return pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes,
            'volatility': volatility,
            'returns': returns,
            'rsi': rsi,
            'macd': macd
        })

    def test_complete_system_pipeline(self, integrated_system, comprehensive_market_data):
        """Test complete unsupervised learning pipeline"""
        clusterer = integrated_system['clusterer']
        regime_detector = integrated_system['regime_detector']
        anomaly_detector = integrated_system['anomaly_detector']
        pattern_memory = integrated_system['pattern_memory']

        symbol = 'INTEGRATION_TEST'
        data = comprehensive_market_data

        # Stage 1: Clustering Analysis
        clustering_results = clusterer.cluster_market_data(symbol, data)

        assert 'kmeans_clusters' in clustering_results
        assert 'hierarchical_clusters' in clustering_results
        assert 'anomalies' in clustering_results
        assert 'quality_metrics' in clustering_results

        # Stage 2: Regime Detection
        regime_result = regime_detector.detect_regime(data)

        assert 'regime' in regime_result
        assert 'confidence' in regime_result
        assert 'stability' in regime_result

        # Stage 3: Anomaly Detection
        anomaly_result = anomaly_detector.detect_anomalies(data)

        assert 'anomaly_scores' in anomaly_result
        assert 'anomaly_flags' in anomaly_result
        assert 'severity_levels' in anomaly_result

        # Stage 4: Pattern Storage
        patterns = {
            'clusters': clustering_results['kmeans_clusters'],
            'regime': regime_result['regime'],
            'anomalies': anomaly_result['anomaly_flags']
        }

        pattern_storage_result = pattern_memory.store_patterns(symbol, patterns, data)

        assert 'pattern_id' in pattern_storage_result
        assert 'similarity_score' in pattern_storage_result

        # Integration validation
        assert len(clustering_results['kmeans_clusters']) == len(data)
        assert len(anomaly_result['anomaly_scores']) == len(data)

    def test_cross_component_data_flow(self, integrated_system, comprehensive_market_data):
        """Test data flow between unsupervised learning components"""
        clusterer = integrated_system['clusterer']
        regime_detector = integrated_system['regime_detector']
        anomaly_detector = integrated_system['anomaly_detector']

        data = comprehensive_market_data.head(1000)  # Smaller dataset for speed

        # Step 1: Clustering provides market state features
        clustering_results = clusterer.cluster_market_data('TEST', data)
        cluster_features = clustering_results['kmeans_clusters']

        # Step 2: Use clustering features in regime detection
        enhanced_data = data.copy()
        enhanced_data['cluster_id'] = cluster_features

        regime_result = regime_detector.detect_regime(enhanced_data)

        # Step 3: Use regime information in anomaly detection
        enhanced_data['regime'] = regime_result['regime']
        enhanced_data['regime_confidence'] = regime_result['confidence']

        anomaly_result = anomaly_detector.detect_anomalies(enhanced_data)

        # Validate cross-component enhancement
        baseline_data = data[['price', 'volume', 'volatility', 'returns']]
        baseline_anomaly_result = anomaly_detector.detect_anomalies(baseline_data)

        # Enhanced detection should be different (presumably better)
        enhanced_scores = anomaly_result['anomaly_scores']
        baseline_scores = baseline_anomaly_result['anomaly_scores']

        score_correlation = np.corrcoef(enhanced_scores, baseline_scores)[0, 1]
        assert score_correlation < 0.95, "Enhanced detection not significantly different from baseline"

    def test_real_time_processing_simulation(self, integrated_system):
        """Test real-time processing capabilities"""
        clusterer = integrated_system['clusterer']
        regime_detector = integrated_system['regime_detector']
        anomaly_detector = integrated_system['anomaly_detector']

        # Simulate real-time data stream
        batch_size = 50
        num_batches = 20
        processing_times = []

        for batch_num in range(num_batches):
            # Generate new batch
            returns = np.random.normal(0.0001, 0.01, batch_size)
            prices = 100 * np.exp(np.cumsum(returns))
            volumes = np.random.lognormal(7, 0.3, batch_size)

            batch_data = pd.DataFrame({
                'price': prices,
                'volume': volumes,
                'returns': returns,
                'volatility': np.abs(returns)
            })

            # Process batch through all components
            start_time = time.time()

            clustering_result = clusterer.cluster_market_data(f'STREAM_{batch_num}', batch_data)
            regime_result = regime_detector.detect_regime(batch_data)
            anomaly_result = anomaly_detector.detect_anomalies(batch_data)

            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Validate results
            assert len(clustering_result['kmeans_clusters']) == batch_size
            assert 'regime' in regime_result
            assert len(anomaly_result['anomaly_scores']) == batch_size

        # Performance requirements for real-time trading
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)

        assert avg_processing_time < 0.5, f"Average batch processing time {avg_processing_time:.3f}s too high"
        assert max_processing_time < 1.0, f"Max batch processing time {max_processing_time:.3f}s too high"

    def test_memory_and_learning_integration(self, integrated_system, comprehensive_market_data):
        """Test pattern memory and learning system integration"""
        pattern_memory = integrated_system['pattern_memory']
        experience_clustering = integrated_system['experience_clustering']
        strategy_evolution = integrated_system['strategy_evolution']

        data = comprehensive_market_data.head(500)

        # Generate trading experiences
        experiences = []
        for i in range(0, len(data) - 50, 50):
            chunk = data.iloc[i:i+50]

            # Simulate trading experience
            experience = {
                'market_data': chunk,
                'action': np.random.choice(['buy', 'sell', 'hold']),
                'outcome': np.random.uniform(-0.02, 0.02),  # P&L
                'features': {
                    'volatility': chunk['volatility'].mean(),
                    'volume': chunk['volume'].mean(),
                    'price_change': (chunk['price'].iloc[-1] - chunk['price'].iloc[0]) / chunk['price'].iloc[0]
                }
            }
            experiences.append(experience)

        # Store experiences in pattern memory
        for exp in experiences:
            pattern_storage = pattern_memory.store_patterns(
                'LEARNING_TEST',
                {'trading_pattern': exp['features']},
                exp['market_data']
            )
            assert 'pattern_id' in pattern_storage

        # Cluster experiences
        experience_features = [exp['features'] for exp in experiences]
        clustering_result = experience_clustering.cluster_experiences(experience_features)

        assert 'clusters' in clustering_result
        assert 'cluster_analysis' in clustering_result

        # Evolve strategies based on clustered experiences
        evolution_result = strategy_evolution.evolve_strategies(experiences, clustering_result)

        assert 'new_strategies' in evolution_result
        assert 'performance_metrics' in evolution_result

    def test_multi_symbol_coordination(self, integrated_system):
        """Test coordination across multiple trading symbols"""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        clusterer = integrated_system['clusterer']
        regime_detector = integrated_system['regime_detector']

        symbol_results = {}

        # Process each symbol
        for symbol in symbols:
            np.random.seed(ord(symbol[0]))  # Different but consistent data per symbol

            data = pd.DataFrame({
                'price': 100 + np.cumsum(np.random.randn(200) * 0.01),
                'volume': np.random.lognormal(7, 0.5, 200),
                'volatility': np.random.gamma(1, 0.01, 200)
            })

            clustering_result = clusterer.cluster_market_data(symbol, data)
            regime_result = regime_detector.detect_regime(data)

            symbol_results[symbol] = {
                'clustering': clustering_result,
                'regime': regime_result,
                'data': data
            }

        # Cross-symbol analysis
        regimes = [result['regime']['regime'] for result in symbol_results.values()]
        unique_regimes = set(regimes)

        # Should detect reasonable variety in regimes
        assert len(unique_regimes) >= 2, f"Only one regime detected across all symbols: {unique_regimes}"

        # Validate correlation analysis potential
        symbol_prices = {}
        for symbol, result in symbol_results.items():
            symbol_prices[symbol] = result['data']['price'].values

        # Should be able to perform cross-symbol correlation analysis
        price_matrix = np.array(list(symbol_prices.values()))
        correlation_matrix = np.corrcoef(price_matrix)

        assert correlation_matrix.shape == (len(symbols), len(symbols))
        assert not np.isnan(correlation_matrix).any()

    def test_error_handling_and_recovery(self, integrated_system):
        """Test system robustness and error recovery"""
        clusterer = integrated_system['clusterer']
        regime_detector = integrated_system['regime_detector']
        anomaly_detector = integrated_system['anomaly_detector']

        # Test various error conditions
        error_scenarios = [
            ('empty_data', pd.DataFrame()),
            ('single_row', pd.DataFrame({'price': [100], 'volume': [1000]})),
            ('nan_data', pd.DataFrame({
                'price': [100, np.nan, 102],
                'volume': [1000, 1100, np.nan],
                'volatility': [0.01, 0.02, 0.01]
            })),
            ('infinite_data', pd.DataFrame({
                'price': [100, np.inf, 102],
                'volume': [1000, 1100, 1200],
                'volatility': [0.01, 0.02, 0.01]
            }))
        ]

        for scenario_name, problematic_data in error_scenarios:
            # Each component should handle errors gracefully
            for component_name, component in [
                ('clusterer', clusterer),
                ('regime_detector', regime_detector),
                ('anomaly_detector', anomaly_detector)
            ]:
                try:
                    if component_name == 'clusterer':
                        result = component.cluster_market_data('ERROR_TEST', problematic_data)
                    elif component_name == 'regime_detector':
                        result = component.detect_regime(problematic_data)
                    else:  # anomaly_detector
                        result = component.detect_anomalies(problematic_data)

                    # If it succeeds, result should be valid
                    assert isinstance(result, dict), f"{component_name} returned non-dict for {scenario_name}"

                except (ValueError, RuntimeError) as e:
                    # Expected errors should be informative
                    error_msg = str(e).lower()
                    assert any(keyword in error_msg for keyword in [
                        'insufficient', 'empty', 'invalid', 'nan', 'infinite'
                    ]), f"Uninformative error from {component_name} for {scenario_name}: {e}"

    def test_performance_under_load(self, integrated_system):
        """Test system performance under high load"""
        clusterer = integrated_system['clusterer']
        regime_detector = integrated_system['regime_detector']
        anomaly_detector = integrated_system['anomaly_detector']

        # Simulate high-frequency trading load
        num_symbols = 10
        data_points_per_symbol = 500

        start_time = time.time()

        for symbol_idx in range(num_symbols):
            # Generate market data
            data = pd.DataFrame({
                'price': 100 + np.cumsum(np.random.randn(data_points_per_symbol) * 0.01),
                'volume': np.random.lognormal(7, 0.5, data_points_per_symbol),
                'volatility': np.random.gamma(1, 0.01, data_points_per_symbol)
            })

            symbol = f'LOAD_TEST_{symbol_idx}'

            # Process through all components
            clustering_result = clusterer.cluster_market_data(symbol, data)
            regime_result = regime_detector.detect_regime(data)
            anomaly_result = anomaly_detector.detect_anomalies(data)

            # Validate results
            assert len(clustering_result['kmeans_clusters']) == data_points_per_symbol
            assert 'regime' in regime_result
            assert len(anomaly_result['anomaly_scores']) == data_points_per_symbol

        total_time = time.time() - start_time
        avg_time_per_symbol = total_time / num_symbols

        # Performance target: process each symbol within 2 seconds
        assert avg_time_per_symbol < 2.0, f"Average processing time {avg_time_per_symbol:.3f}s too high"

    @patch('app.monitoring.metrics.PrometheusMetrics')
    def test_comprehensive_metrics_integration(self, mock_metrics, integrated_system, comprehensive_market_data):
        """Test comprehensive metrics collection across all components"""
        metrics_instance = Mock()
        mock_metrics.return_value = metrics_instance

        # Enable metrics for all components
        for component in integrated_system.values():
            component.prometheus_metrics = metrics_instance

        clusterer = integrated_system['clusterer']
        regime_detector = integrated_system['regime_detector']
        anomaly_detector = integrated_system['anomaly_detector']

        data = comprehensive_market_data.head(200)

        # Process data through all components
        clustering_result = clusterer.cluster_market_data('METRICS_TEST', data)
        regime_result = regime_detector.detect_regime(data)
        anomaly_result = anomaly_detector.detect_anomalies(data)

        # Validate metrics data availability
        assert 'quality_metrics' in clustering_result
        assert 'regime' in regime_result
        assert 'confidence' in regime_result
        assert 'anomaly_scores' in anomaly_result

        # Calculate key metrics that would be reported
        clustering_quality = clustering_result['quality_metrics']
        regime_confidence = regime_result['confidence']
        anomaly_rate = np.mean(anomaly_result['anomaly_flags'])
        avg_anomaly_score = np.mean(anomaly_result['anomaly_scores'])

        # Validate metric ranges
        assert 0 <= regime_confidence <= 1, f"Invalid regime confidence: {regime_confidence}"
        assert 0 <= anomaly_rate <= 1, f"Invalid anomaly rate: {anomaly_rate}"
        assert 0 <= avg_anomaly_score <= 1, f"Invalid average anomaly score: {avg_anomaly_score}"

    def test_system_state_consistency(self, integrated_system, comprehensive_market_data):
        """Test consistency of system state across components"""
        clusterer = integrated_system['clusterer']
        regime_detector = integrated_system['regime_detector']
        anomaly_detector = integrated_system['anomaly_detector']

        data = comprehensive_market_data.head(300)

        # Process data multiple times
        results_1 = {
            'clustering': clusterer.cluster_market_data('CONSISTENCY_TEST', data),
            'regime': regime_detector.detect_regime(data),
            'anomaly': anomaly_detector.detect_anomalies(data)
        }

        results_2 = {
            'clustering': clusterer.cluster_market_data('CONSISTENCY_TEST', data),
            'regime': regime_detector.detect_regime(data),
            'anomaly': anomaly_detector.detect_anomalies(data)
        }

        # Check consistency
        # Clustering should be deterministic
        clusters_1 = results_1['clustering']['kmeans_clusters']
        clusters_2 = results_2['clustering']['kmeans_clusters']
        cluster_consistency = np.mean(clusters_1 == clusters_2)
        assert cluster_consistency > 0.95, f"Low clustering consistency: {cluster_consistency}"

        # Regime detection should be stable
        regime_1 = results_1['regime']['regime']
        regime_2 = results_2['regime']['regime']
        assert regime_1 == regime_2, f"Regime detection inconsistent: {regime_1} vs {regime_2}"

        # Anomaly scores should be similar
        scores_1 = results_1['anomaly']['anomaly_scores']
        scores_2 = results_2['anomaly']['anomaly_scores']
        score_correlation = np.corrcoef(scores_1, scores_2)[0, 1]
        assert score_correlation > 0.95, f"Low anomaly score consistency: {score_correlation}"

    def test_scalability_analysis(self, integrated_system):
        """Test system scalability with increasing data sizes"""
        clusterer = integrated_system['clusterer']
        regime_detector = integrated_system['regime_detector']

        data_sizes = [100, 300, 500, 1000, 2000]
        processing_times = {'clustering': [], 'regime': []}

        for size in data_sizes:
            # Generate data of specified size
            data = pd.DataFrame({
                'price': 100 + np.cumsum(np.random.randn(size) * 0.01),
                'volume': np.random.lognormal(7, 0.5, size),
                'volatility': np.random.gamma(1, 0.01, size)
            })

            # Measure clustering performance
            start_time = time.time()
            clustering_result = clusterer.cluster_market_data(f'SCALE_{size}', data)
            clustering_time = time.time() - start_time
            processing_times['clustering'].append(clustering_time)

            # Measure regime detection performance
            start_time = time.time()
            regime_result = regime_detector.detect_regime(data)
            regime_time = time.time() - start_time
            processing_times['regime'].append(regime_time)

            # Validate results
            assert len(clustering_result['kmeans_clusters']) == size
            assert 'regime' in regime_result

        # Check scaling behavior
        for component, times in processing_times.items():
            for i, (size, time_taken) in enumerate(zip(data_sizes, times)):
                # Roughly linear scaling expectation
                max_time = size * 0.003  # 3ms per data point
                assert time_taken < max_time, f"{component} scaling issue: {size} points took {time_taken:.3f}s"

    def test_production_deployment_readiness(self, integrated_system):
        """Test production deployment readiness criteria"""
        components = integrated_system

        # Test 1: All components can be instantiated
        for component_name, component in components.items():
            assert component is not None, f"{component_name} failed to instantiate"
            assert hasattr(component, '_fitted') or hasattr(component, 'fit'), f"{component_name} missing fit interface"

        # Test 2: Components handle typical market data
        typical_data = pd.DataFrame({
            'price': 150 + np.cumsum(np.random.randn(100) * 0.01),
            'volume': np.random.lognormal(7, 0.3, 100),
            'volatility': np.random.gamma(1, 0.01, 100),
            'returns': np.random.normal(0.0001, 0.01, 100)
        })

        # Test each component
        try:
            clustering_result = components['clusterer'].cluster_market_data('PROD_TEST', typical_data)
            assert 'kmeans_clusters' in clustering_result

            regime_result = components['regime_detector'].detect_regime(typical_data)
            assert 'regime' in regime_result

            anomaly_result = components['anomaly_detector'].detect_anomalies(typical_data)
            assert 'anomaly_scores' in anomaly_result

        except Exception as e:
            pytest.fail(f"Production readiness test failed: {e}")

        # Test 3: Performance meets production requirements
        start_time = time.time()
        for _ in range(10):  # 10 iterations
            components['clusterer'].cluster_market_data('PROD_PERF', typical_data)
            components['regime_detector'].detect_regime(typical_data)
            components['anomaly_detector'].detect_anomalies(typical_data)

        total_time = time.time() - start_time
        avg_iteration_time = total_time / 10

        assert avg_iteration_time < 1.0, f"Production performance requirement not met: {avg_iteration_time:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])