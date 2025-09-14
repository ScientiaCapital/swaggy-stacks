"""
Comprehensive Unit Tests for Market Regime Detection
Critical for production trading system validation
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any

from app.ml.unsupervised.market_regime import (
    MarketRegimeDetector,
    RegimeTransitionPredictor,
    RegimeStabilityAnalyzer
)
from app.monitoring.metrics import PrometheusMetrics


class TestMarketRegimeDetector:
    """Test core market regime detection functionality"""

    @pytest.fixture
    def regime_detector(self):
        """Create MarketRegimeDetector instance"""
        return MarketRegimeDetector(
            lookback_window=50,
            min_regime_duration=10,
            confidence_threshold=0.8
        )

    @pytest.fixture
    def bull_market_data(self):
        """Generate bull market scenario data"""
        np.random.seed(42)
        n_points = 200
        dates = pd.date_range('2024-01-01', periods=n_points, freq='1H')

        # Bull market: consistent upward trend, moderate volatility
        returns = np.random.normal(0.001, 0.01, n_points)  # Positive drift
        prices = 100 * np.exp(np.cumsum(returns))
        volumes = np.random.lognormal(7, 0.5, n_points)
        volatility = np.random.gamma(1, 0.01, n_points)  # Low volatility

        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': volumes,
            'volatility': volatility,
            'returns': returns
        })

    @pytest.fixture
    def bear_market_data(self):
        """Generate bear market scenario data"""
        np.random.seed(123)
        n_points = 200
        dates = pd.date_range('2024-01-01', periods=n_points, freq='1H')

        # Bear market: consistent downward trend, high volatility
        returns = np.random.normal(-0.002, 0.02, n_points)  # Negative drift
        prices = 100 * np.exp(np.cumsum(returns))
        volumes = np.random.lognormal(7.5, 0.8, n_points)  # Higher volume
        volatility = np.random.gamma(2, 0.02, n_points)  # High volatility

        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': volumes,
            'volatility': volatility,
            'returns': returns
        })

    @pytest.fixture
    def sideways_market_data(self):
        """Generate sideways/range-bound market data"""
        np.random.seed(456)
        n_points = 200
        dates = pd.date_range('2024-01-01', periods=n_points, freq='1H')

        # Sideways market: mean-reverting with no trend
        prices = 100 + 5 * np.sin(np.linspace(0, 4*np.pi, n_points)) + np.random.normal(0, 1, n_points)
        returns = np.diff(prices) / prices[:-1]
        returns = np.concatenate([[0], returns])
        volumes = np.random.lognormal(6.8, 0.3, n_points)  # Lower volume
        volatility = np.random.gamma(1.5, 0.015, n_points)  # Moderate volatility

        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': volumes,
            'volatility': volatility,
            'returns': returns
        })

    @pytest.fixture
    def volatile_market_data(self):
        """Generate high volatility/crisis market data"""
        np.random.seed(789)
        n_points = 200
        dates = pd.date_range('2024-01-01', periods=n_points, freq='1H')

        # Volatile market: high volatility, erratic behavior
        returns = np.random.normal(0, 0.03, n_points)  # High volatility
        # Add some jumps/gaps
        jump_indices = np.random.choice(n_points, 10, replace=False)
        returns[jump_indices] += np.random.choice([-1, 1], 10) * np.random.exponential(0.05, 10)

        prices = 100 * np.exp(np.cumsum(returns))
        volumes = np.random.lognormal(8, 1.2, n_points)  # Very high volume
        volatility = np.random.gamma(3, 0.03, n_points)  # Very high volatility

        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': volumes,
            'volatility': volatility,
            'returns': returns
        })

    def test_regime_detector_initialization(self, regime_detector):
        """Test regime detector initialization"""
        assert regime_detector.lookback_window == 50
        assert regime_detector.min_regime_duration == 10
        assert regime_detector.confidence_threshold == 0.8
        assert regime_detector._fitted is False

    def test_feature_engineering_for_regime_detection(self, regime_detector, bull_market_data):
        """Test feature engineering for regime detection"""
        features = regime_detector._engineer_regime_features(bull_market_data)

        # Validate feature structure
        expected_features = [
            'return_mean', 'return_std', 'volume_mean', 'volatility_mean',
            'trend_strength', 'momentum', 'volume_profile'
        ]

        for feature in expected_features:
            assert feature in features.columns, f"Missing feature: {feature}"

        # Check feature quality
        assert not features.isnull().any().any(), "Features contain NaN values"
        assert features.shape[0] > 0, "No features generated"

    def test_bull_market_detection(self, regime_detector, bull_market_data):
        """Test detection of bull market regime"""
        start_time = time.time()
        regime_result = regime_detector.detect_regime(bull_market_data)
        detection_time = time.time() - start_time

        # Validate detection result
        assert 'regime' in regime_result
        assert 'confidence' in regime_result
        assert 'stability' in regime_result
        assert 'features' in regime_result

        # Should detect bull market
        detected_regime = regime_result['regime']
        assert detected_regime in ['bull', 'bull_trend', 'bullish'], f"Expected bull regime, got {detected_regime}"

        # High confidence expected for clear bull market
        confidence = regime_result['confidence']
        assert confidence > 0.7, f"Low confidence {confidence} for clear bull market"

        # Performance requirement: detect regime within 100ms
        assert detection_time < 0.1, f"Regime detection took {detection_time:.3f}s, expected < 0.1s"

    def test_bear_market_detection(self, regime_detector, bear_market_data):
        """Test detection of bear market regime"""
        regime_result = regime_detector.detect_regime(bear_market_data)

        # Should detect bear market
        detected_regime = regime_result['regime']
        assert detected_regime in ['bear', 'bear_trend', 'bearish'], f"Expected bear regime, got {detected_regime}"

        # High confidence expected for clear bear market
        confidence = regime_result['confidence']
        assert confidence > 0.7, f"Low confidence {confidence} for clear bear market"

    def test_sideways_market_detection(self, regime_detector, sideways_market_data):
        """Test detection of sideways market regime"""
        regime_result = regime_detector.detect_regime(sideways_market_data)

        # Should detect sideways market
        detected_regime = regime_result['regime']
        assert detected_regime in ['sideways', 'range_bound', 'neutral'], f"Expected sideways regime, got {detected_regime}"

        # Moderate confidence expected
        confidence = regime_result['confidence']
        assert confidence > 0.5, f"Low confidence {confidence} for sideways market"

    def test_volatile_market_detection(self, regime_detector, volatile_market_data):
        """Test detection of volatile/crisis market regime"""
        regime_result = regime_detector.detect_regime(volatile_market_data)

        # Should detect volatile/crisis regime
        detected_regime = regime_result['regime']
        assert detected_regime in ['volatile', 'crisis', 'high_volatility'], f"Expected volatile regime, got {detected_regime}"

        # Should have reasonable confidence
        confidence = regime_result['confidence']
        assert confidence > 0.6, f"Low confidence {confidence} for volatile market"

    def test_regime_detection_consistency(self, regime_detector, bull_market_data):
        """Test consistency of regime detection across multiple runs"""
        # Multiple detections on same data
        results = []
        for _ in range(5):
            result = regime_detector.detect_regime(bull_market_data)
            results.append((result['regime'], result['confidence']))

        # Should be consistent
        regimes = [r[0] for r in results]
        confidences = [r[1] for r in results]

        assert len(set(regimes)) <= 2, f"Too much variation in regime detection: {set(regimes)}"
        assert np.std(confidences) < 0.1, f"High variance in confidence scores: {confidences}"

    def test_regime_detection_with_insufficient_data(self, regime_detector):
        """Test regime detection with insufficient data"""
        # Very small dataset
        small_data = pd.DataFrame({
            'price': [100, 101, 102],
            'volume': [1000, 1100, 1200],
            'returns': [0, 0.01, 0.01]
        })

        try:
            result = regime_detector.detect_regime(small_data)
            # Should either work with low confidence or raise appropriate error
            if 'regime' in result:
                assert result['confidence'] < 0.5, "High confidence with insufficient data"
        except ValueError as e:
            assert "insufficient" in str(e).lower() or "data" in str(e).lower()

    @patch('app.monitoring.metrics.PrometheusMetrics')
    def test_regime_detection_metrics_integration(self, mock_metrics, regime_detector, bull_market_data):
        """Test integration with Prometheus metrics"""
        metrics_instance = Mock()
        mock_metrics.return_value = metrics_instance

        # Enable metrics collection
        regime_detector.prometheus_metrics = metrics_instance

        result = regime_detector.detect_regime(bull_market_data)

        # Verify metrics would be updated
        assert 'regime' in result
        assert 'confidence' in result
        assert 'stability' in result


class TestRegimeTransitionPredictor:
    """Test regime transition prediction functionality"""

    @pytest.fixture
    def transition_predictor(self):
        """Create RegimeTransitionPredictor instance"""
        return RegimeTransitionPredictor(
            transition_window=20,
            prediction_horizon=10
        )

    @pytest.fixture
    def regime_history(self):
        """Generate regime history for transition testing"""
        # Simulate regime transitions: bull -> sideways -> bear -> volatile -> bull
        regimes = (
            ['bull'] * 50 +
            ['sideways'] * 30 +
            ['bear'] * 40 +
            ['volatile'] * 20 +
            ['bull'] * 30
        )

        confidences = np.random.uniform(0.7, 0.95, len(regimes))
        timestamps = pd.date_range('2024-01-01', periods=len(regimes), freq='1H')

        return pd.DataFrame({
            'timestamp': timestamps,
            'regime': regimes,
            'confidence': confidences
        })

    def test_transition_predictor_initialization(self, transition_predictor):
        """Test transition predictor initialization"""
        assert transition_predictor.transition_window == 20
        assert transition_predictor.prediction_horizon == 10
        assert transition_predictor._fitted is False

    def test_transition_pattern_learning(self, transition_predictor, regime_history):
        """Test learning of transition patterns"""
        transition_predictor.fit(regime_history)

        # Should identify transition patterns
        patterns = transition_predictor.get_transition_patterns()

        assert 'transitions' in patterns
        assert 'probabilities' in patterns
        assert transition_predictor._fitted is True

        # Should learn some transitions
        transitions = patterns['transitions']
        assert len(transitions) > 0, "No transition patterns learned"

    def test_regime_transition_prediction(self, transition_predictor, regime_history):
        """Test prediction of regime transitions"""
        # Fit on history
        transition_predictor.fit(regime_history)

        # Predict next regime
        current_regime = 'bull'
        recent_history = regime_history.tail(transition_predictor.transition_window)

        prediction = transition_predictor.predict_transition(current_regime, recent_history)

        # Validate prediction structure
        assert 'next_regime' in prediction
        assert 'probability' in prediction
        assert 'transition_timing' in prediction
        assert 'confidence' in prediction

        # Probability should be valid
        prob = prediction['probability']
        assert 0 <= prob <= 1, f"Invalid probability {prob}"

    def test_transition_timing_accuracy(self, transition_predictor, regime_history):
        """Test accuracy of transition timing predictions"""
        # Use first 80% for training, last 20% for testing
        split_point = int(len(regime_history) * 0.8)
        train_data = regime_history[:split_point]
        test_data = regime_history[split_point:]

        # Fit on training data
        transition_predictor.fit(train_data)

        # Predict transitions in test period
        correct_predictions = 0
        total_predictions = 0

        for i in range(len(test_data) - transition_predictor.prediction_horizon):
            current_state = test_data.iloc[i]
            recent_history = train_data.tail(transition_predictor.transition_window)

            prediction = transition_predictor.predict_transition(
                current_state['regime'], recent_history
            )

            # Check if prediction matches actual future regime
            actual_future = test_data.iloc[i + transition_predictor.prediction_horizon]['regime']
            predicted_regime = prediction['next_regime']

            if predicted_regime == actual_future:
                correct_predictions += 1
            total_predictions += 1

        # Should achieve reasonable accuracy
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            assert accuracy > 0.3, f"Low transition prediction accuracy: {accuracy}"

    def test_transition_prediction_performance(self, transition_predictor, regime_history):
        """Test transition prediction performance"""
        transition_predictor.fit(regime_history)

        start_time = time.time()
        for _ in range(100):  # 100 predictions
            prediction = transition_predictor.predict_transition(
                'bull', regime_history.tail(20)
            )
        prediction_time = time.time() - start_time

        # Should make predictions quickly
        avg_time = prediction_time / 100
        assert avg_time < 0.01, f"Average prediction time {avg_time:.4f}s, expected < 0.01s"


class TestRegimeStabilityAnalyzer:
    """Test regime stability analysis functionality"""

    @pytest.fixture
    def stability_analyzer(self):
        """Create RegimeStabilityAnalyzer instance"""
        return RegimeStabilityAnalyzer(
            stability_window=30,
            min_stability_threshold=0.8
        )

    @pytest.fixture
    def stable_regime_sequence(self):
        """Generate stable regime sequence"""
        # Mostly bull with few brief interruptions
        regimes = ['bull'] * 80 + ['sideways'] * 5 + ['bull'] * 15
        confidences = np.random.uniform(0.8, 0.95, len(regimes))
        timestamps = pd.date_range('2024-01-01', periods=len(regimes), freq='1H')

        return pd.DataFrame({
            'timestamp': timestamps,
            'regime': regimes,
            'confidence': confidences
        })

    @pytest.fixture
    def unstable_regime_sequence(self):
        """Generate unstable regime sequence"""
        # Frequent regime changes
        regimes = []
        for _ in range(20):
            regimes.extend(['bull', 'sideways', 'bear', 'volatile'])

        confidences = np.random.uniform(0.5, 0.8, len(regimes))
        timestamps = pd.date_range('2024-01-01', periods=len(regimes), freq='1H')

        return pd.DataFrame({
            'timestamp': timestamps,
            'regime': regimes,
            'confidence': confidences
        })

    def test_stability_analyzer_initialization(self, stability_analyzer):
        """Test stability analyzer initialization"""
        assert stability_analyzer.stability_window == 30
        assert stability_analyzer.min_stability_threshold == 0.8

    def test_stable_regime_analysis(self, stability_analyzer, stable_regime_sequence):
        """Test analysis of stable regime periods"""
        analysis = stability_analyzer.analyze_stability(stable_regime_sequence)

        # Validate analysis structure
        assert 'stability_score' in analysis
        assert 'regime_duration' in analysis
        assert 'confidence_trend' in analysis
        assert 'transition_frequency' in analysis

        # Should detect high stability
        stability_score = analysis['stability_score']
        assert stability_score > 0.7, f"Low stability score {stability_score} for stable sequence"

        # Low transition frequency
        transition_freq = analysis['transition_frequency']
        assert transition_freq < 0.1, f"High transition frequency {transition_freq} for stable sequence"

    def test_unstable_regime_analysis(self, stability_analyzer, unstable_regime_sequence):
        """Test analysis of unstable regime periods"""
        analysis = stability_analyzer.analyze_stability(unstable_regime_sequence)

        # Should detect low stability
        stability_score = analysis['stability_score']
        assert stability_score < 0.5, f"High stability score {stability_score} for unstable sequence"

        # High transition frequency
        transition_freq = analysis['transition_frequency']
        assert transition_freq > 0.2, f"Low transition frequency {transition_freq} for unstable sequence"

    def test_stability_trend_detection(self, stability_analyzer):
        """Test detection of stability trends"""
        # Create sequence with increasing stability
        regimes = (['bull', 'bear'] * 10) + ['bull'] * 50  # Unstable then stable
        confidences = list(np.random.uniform(0.5, 0.7, 20)) + list(np.random.uniform(0.8, 0.95, 50))

        data = pd.DataFrame({
            'regime': regimes,
            'confidence': confidences,
            'timestamp': pd.date_range('2024-01-01', periods=len(regimes), freq='1H')
        })

        analysis = stability_analyzer.analyze_stability(data)

        # Should detect improving stability trend
        trend = analysis['confidence_trend']
        assert trend > 0, f"Expected positive trend, got {trend}"

    def test_regime_duration_analysis(self, stability_analyzer, stable_regime_sequence):
        """Test regime duration analysis"""
        analysis = stability_analyzer.analyze_stability(stable_regime_sequence)

        duration_stats = analysis['regime_duration']

        assert 'mean_duration' in duration_stats
        assert 'median_duration' in duration_stats
        assert 'max_duration' in duration_stats

        # For stable sequence, should have long durations
        mean_duration = duration_stats['mean_duration']
        assert mean_duration > 10, f"Low mean duration {mean_duration} for stable sequence"


class TestMarketRegimeIntegration:
    """Integration tests for complete market regime system"""

    @pytest.fixture
    def full_regime_system(self):
        """Create complete regime detection system"""
        detector = MarketRegimeDetector()
        predictor = RegimeTransitionPredictor()
        analyzer = RegimeStabilityAnalyzer()

        return {
            'detector': detector,
            'predictor': predictor,
            'analyzer': analyzer
        }

    @pytest.fixture
    def market_cycle_data(self):
        """Generate full market cycle data"""
        np.random.seed(42)

        # Bull market phase (6 months)
        bull_returns = np.random.normal(0.001, 0.01, 4320)  # 6 months * 30 days * 24 hours
        bull_prices = 100 * np.exp(np.cumsum(bull_returns))

        # Bear market phase (3 months)
        bear_returns = np.random.normal(-0.002, 0.02, 2160)
        bear_prices = bull_prices[-1] * np.exp(np.cumsum(bear_returns))

        # Recovery phase (3 months)
        recovery_returns = np.random.normal(0.0015, 0.015, 2160)
        recovery_prices = bear_prices[-1] * np.exp(np.cumsum(recovery_returns))

        # Combine all phases
        all_prices = np.concatenate([bull_prices, bear_prices, recovery_prices])
        all_returns = np.diff(all_prices) / all_prices[:-1]
        all_returns = np.concatenate([[0], all_returns])

        timestamps = pd.date_range('2024-01-01', periods=len(all_prices), freq='1H')
        volumes = np.random.lognormal(7, 0.5, len(all_prices))
        volatility = np.abs(all_returns) * 100  # Realized volatility

        return pd.DataFrame({
            'timestamp': timestamps,
            'price': all_prices,
            'volume': volumes,
            'volatility': volatility,
            'returns': all_returns
        })

    def test_complete_regime_detection_pipeline(self, full_regime_system, market_cycle_data):
        """Test complete regime detection pipeline"""
        detector = full_regime_system['detector']
        predictor = full_regime_system['predictor']
        analyzer = full_regime_system['analyzer']

        # Process data in chunks to simulate real-time operation
        chunk_size = 200
        regime_history = []

        total_start = time.time()

        for i in range(0, len(market_cycle_data) - chunk_size, chunk_size):
            chunk = market_cycle_data.iloc[i:i+chunk_size]

            # Detect current regime
            regime_result = detector.detect_regime(chunk)

            regime_history.append({
                'timestamp': chunk['timestamp'].iloc[-1],
                'regime': regime_result['regime'],
                'confidence': regime_result['confidence'],
                'stability': regime_result['stability']
            })

        total_time = time.time() - total_start

        # Convert to DataFrame for analysis
        regime_df = pd.DataFrame(regime_history)

        # Should detect regime changes throughout the cycle
        unique_regimes = regime_df['regime'].unique()
        assert len(unique_regimes) >= 2, f"Only detected {unique_regimes}, expected multiple regimes"

        # Performance requirement: process full cycle within reasonable time
        chunks_processed = len(regime_history)
        avg_time_per_chunk = total_time / chunks_processed
        assert avg_time_per_chunk < 0.1, f"Average chunk processing time {avg_time_per_chunk:.3f}s too high"

        # Test transition prediction on regime history
        if len(regime_df) > 30:  # Enough data for transition analysis
            predictor.fit(regime_df)

            # Test stability analysis
            stability_analysis = analyzer.analyze_stability(regime_df)

            assert 'stability_score' in stability_analysis
            assert 'transition_frequency' in stability_analysis

    def test_regime_detection_accuracy_validation(self, full_regime_system, market_cycle_data):
        """Test regime detection accuracy against known market phases"""
        detector = full_regime_system['detector']

        # Manually label expected regimes based on return characteristics
        data_with_labels = market_cycle_data.copy()

        # Define expected regimes based on data generation
        bull_end = 4320
        bear_end = bull_end + 2160

        expected_regimes = (
            ['bull'] * bull_end +
            ['bear'] * 2160 +
            ['bull'] * (len(data_with_labels) - bear_end)
        )

        data_with_labels['expected_regime'] = expected_regimes

        # Test detection accuracy on different phases
        phases = [
            ('bull_phase', 0, bull_end),
            ('bear_phase', bull_end, bear_end),
            ('recovery_phase', bear_end, len(data_with_labels))
        ]

        phase_accuracies = {}

        for phase_name, start_idx, end_idx in phases:
            phase_data = data_with_labels.iloc[start_idx:end_idx]

            if len(phase_data) < 50:  # Skip if too small
                continue

            # Sample points for testing (every 100 points to speed up)
            sample_indices = range(0, len(phase_data), 100)
            correct_detections = 0
            total_detections = 0

            for idx in sample_indices:
                if idx + 50 > len(phase_data):  # Need enough data for detection
                    continue

                chunk = phase_data.iloc[idx:idx+50]
                result = detector.detect_regime(chunk)
                detected = result['regime']
                expected = phase_data.iloc[idx]['expected_regime']

                # Map detected regimes to expected categories
                if self._regimes_match(detected, expected):
                    correct_detections += 1
                total_detections += 1

            if total_detections > 0:
                accuracy = correct_detections / total_detections
                phase_accuracies[phase_name] = accuracy

        # Should achieve reasonable accuracy for clear phases
        for phase, accuracy in phase_accuracies.items():
            assert accuracy > 0.5, f"Low accuracy {accuracy} for {phase}"

    def _regimes_match(self, detected: str, expected: str) -> bool:
        """Helper to match detected regimes with expected regimes"""
        regime_mappings = {
            'bull': ['bull', 'bullish', 'bull_trend'],
            'bear': ['bear', 'bearish', 'bear_trend'],
            'sideways': ['sideways', 'neutral', 'range_bound'],
            'volatile': ['volatile', 'crisis', 'high_volatility']
        }

        for expected_group, detected_variants in regime_mappings.items():
            if expected == expected_group and detected in detected_variants:
                return True
        return False

    @patch('app.monitoring.metrics.PrometheusMetrics')
    def test_regime_system_metrics_collection(self, mock_metrics, full_regime_system, market_cycle_data):
        """Test comprehensive metrics collection across regime system"""
        metrics_instance = Mock()
        mock_metrics.return_value = metrics_instance

        # Enable metrics for all components
        for component in full_regime_system.values():
            component.prometheus_metrics = metrics_instance

        detector = full_regime_system['detector']

        # Process sample data
        sample_data = market_cycle_data.head(100)
        result = detector.detect_regime(sample_data)

        # Verify structure for metrics collection
        assert 'regime' in result
        assert 'confidence' in result
        assert 'stability' in result

    def test_regime_system_production_readiness(self, full_regime_system):
        """Test production readiness of regime detection system"""
        detector = full_regime_system['detector']
        predictor = full_regime_system['predictor']
        analyzer = full_regime_system['analyzer']

        # Test with various market conditions
        market_scenarios = [
            ('trending_up', np.random.normal(0.001, 0.01, 200)),
            ('trending_down', np.random.normal(-0.001, 0.01, 200)),
            ('high_volatility', np.random.normal(0, 0.03, 200)),
            ('low_volatility', np.random.normal(0, 0.005, 200))
        ]

        for scenario_name, returns in market_scenarios:
            prices = 100 * np.exp(np.cumsum(returns))
            data = pd.DataFrame({
                'price': prices,
                'returns': returns,
                'volume': np.random.lognormal(7, 0.5, len(prices)),
                'volatility': np.abs(returns)
            })

            # Should handle all scenarios without errors
            try:
                result = detector.detect_regime(data)
                assert 'regime' in result
                assert 'confidence' in result
                assert 0 <= result['confidence'] <= 1
            except Exception as e:
                pytest.fail(f"Regime detection failed for {scenario_name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])