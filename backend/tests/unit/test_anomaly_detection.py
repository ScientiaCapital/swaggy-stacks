"""
Comprehensive tests for the anomaly detection system.

Tests both the AnomalyDetector and its integration with RiskManager.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from app.ml.unsupervised.anomaly_detector import AnomalyDetector, AnomalySeverity, AnomalyType
from app.ml.unsupervised.clustering import DensityBasedDetector
from app.trading.risk_manager import RiskManager


class TestAnomalyDetector:
    """Test suite for AnomalyDetector class."""

    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')

        # Generate realistic OHLCV data
        base_price = 100.0
        prices = []
        volumes = []

        for i in range(len(dates)):
            # Add some volatility and trend
            price_change = np.random.normal(0, 0.02)  # 2% volatility
            if i > 0:
                base_price = prices[-1] * (1 + price_change)

            # Generate OHLC from base price
            high = base_price * (1 + abs(np.random.normal(0, 0.01)))
            low = base_price * (1 - abs(np.random.normal(0, 0.01)))
            close = base_price
            open_price = prices[-1] if i > 0 else base_price

            prices.append(close)
            volumes.append(np.random.randint(10000, 100000))

        return pd.DataFrame({
            'timestamp': dates,
            'open': [prices[0]] + prices[:-1],
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': volumes
        })

    @pytest.fixture
    def anomalous_market_data(self):
        """Generate market data with known anomalies."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1H')

        prices = []
        volumes = []
        base_price = 100.0

        for i in range(len(dates)):
            # Normal data
            price_change = np.random.normal(0, 0.01)
            if i > 0:
                base_price = prices[-1] * (1 + price_change)

            # Inject anomalies
            if i == 20:  # Price gap anomaly
                base_price *= 1.05  # 5% gap
            elif i == 30:  # Volume spike
                volumes.append(500000)  # 5x normal volume
                prices.append(base_price)
                continue
            elif i == 40:  # High volatility anomaly
                base_price *= (1 + np.random.normal(0, 0.1))  # 10% volatility

            prices.append(base_price)
            volumes.append(np.random.randint(50000, 100000))

        return pd.DataFrame({
            'timestamp': dates,
            'open': [prices[0]] + prices[:-1],
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': volumes
        })

    @pytest.fixture
    def anomaly_detector(self):
        """Create AnomalyDetector instance for testing."""
        return AnomalyDetector(
            lookback_period=50,
            detection_timeframes=['1min', '5min', '1hour'],
            anomaly_threshold=0.7,
            critical_threshold=0.9,
            early_warning_minutes=5,
            volume_spike_threshold=3.0,
            price_gap_threshold=0.02
        )

    def test_anomaly_detector_initialization(self, anomaly_detector):
        """Test AnomalyDetector initialization."""
        assert anomaly_detector.anomaly_threshold == 0.7
        assert anomaly_detector.critical_threshold == 0.9
        assert anomaly_detector.detection_timeframes == ['1min', '5min', '1hour']
        assert anomaly_detector.volume_spike_threshold == 3.0
        assert anomaly_detector.price_gap_threshold == 0.02
        assert not anomaly_detector.is_fitted

    def test_feature_extraction(self, anomaly_detector, sample_market_data):
        """Test feature extraction from market data."""
        features = anomaly_detector._extract_anomaly_features(sample_market_data)

        assert 'price' in features
        assert 'volume' in features
        assert 'technical' in features
        assert 'multi_timeframe' in features

        # Check feature dimensions
        assert features['price'].shape[1] == 4  # returns, log_returns, gaps, true_ranges
        assert features['volume'].shape[1] == 2  # volume_returns, volume_relative
        assert len(features['multi_timeframe']) == 9  # 3 timeframes * 3 features

    def test_volume_spike_detection(self, anomaly_detector, anomalous_market_data):
        """Test volume spike detection."""
        features = anomaly_detector._extract_anomaly_features(anomalous_market_data)
        spike_mask, spike_scores = anomaly_detector._detect_volume_spikes(features)

        assert len(spike_mask) > 0
        assert len(spike_scores) == len(spike_mask)
        assert np.any(spike_scores > 0.5)  # Should detect the volume spike

    def test_price_gap_detection(self, anomaly_detector, anomalous_market_data):
        """Test price gap detection."""
        features = anomaly_detector._extract_anomaly_features(anomalous_market_data)
        gap_mask, gap_scores = anomaly_detector._detect_price_gaps(features)

        assert len(gap_mask) > 0
        assert len(gap_scores) == len(gap_mask)
        # Should detect the 5% price gap we injected
        assert np.any(gap_scores > 0.5)

    def test_correlation_break_detection(self, anomaly_detector, sample_market_data):
        """Test correlation break detection."""
        features = anomaly_detector._extract_anomaly_features(sample_market_data)
        corr_mask, corr_scores = anomaly_detector._detect_correlation_breaks(features)

        assert len(corr_mask) >= 0  # May be empty for normal data
        assert len(corr_scores) == len(corr_mask)

    @patch('app.ml.unsupervised.anomaly_detector.DensityBasedDetector')
    def test_anomaly_detector_fitting(self, mock_density_detector, anomaly_detector, sample_market_data):
        """Test fitting the anomaly detector."""
        # Mock the density detector
        mock_instance = Mock()
        mock_instance.fit.return_value = mock_instance
        mock_instance.is_fitted = True
        mock_density_detector.return_value = mock_instance

        # Fit the detector
        result = anomaly_detector.fit(sample_market_data)

        assert result == anomaly_detector
        assert anomaly_detector.is_fitted
        assert anomaly_detector.density_detector is not None
        mock_instance.fit.assert_called_once()

    @patch('app.ml.unsupervised.anomaly_detector.DensityBasedDetector')
    def test_anomaly_prediction(self, mock_density_detector, anomaly_detector, sample_market_data, anomalous_market_data):
        """Test anomaly prediction."""
        # Mock the density detector
        mock_instance = Mock()
        mock_instance.fit.return_value = mock_instance
        mock_instance.is_fitted = True
        mock_instance.get_anomaly_scores.return_value = np.random.random(len(anomalous_market_data))
        mock_density_detector.return_value = mock_instance

        # Fit and predict
        anomaly_detector.fit(sample_market_data)
        result = anomaly_detector.predict(anomalous_market_data)

        # Check result structure
        assert 'anomaly_detected' in result
        assert 'max_anomaly_score' in result
        assert 'anomaly_scores' in result
        assert 'anomaly_mask' in result
        assert 'severity_levels' in result
        assert 'anomaly_types' in result
        assert 'alerts' in result
        assert 'early_warnings' in result
        assert 'risk_assessment' in result
        assert 'detection_metadata' in result

    def test_severity_classification(self, anomaly_detector):
        """Test anomaly severity classification."""
        # Test different score levels
        detector_results = {
            'test_detector': (
                np.array([True, True, True, True]),
                np.array([0.2, 0.5, 0.8, 0.95])  # low, medium, high, critical
            )
        }

        combined = anomaly_detector._combine_anomaly_scores(detector_results)
        severity_levels = combined['severity_levels']

        assert len(severity_levels) == 4
        assert severity_levels[0] == AnomalySeverity.LOW
        assert severity_levels[1] == AnomalySeverity.MEDIUM
        assert severity_levels[2] == AnomalySeverity.HIGH
        assert severity_levels[3] == AnomalySeverity.CRITICAL

    def test_alert_generation(self, anomaly_detector):
        """Test alert generation."""
        # Mock detection results
        combined_results = {
            'combined_scores': np.array([0.3, 0.8, 0.95]),
            'combined_mask': np.array([False, True, True]),
            'severity_levels': [AnomalySeverity.LOW, AnomalySeverity.HIGH, AnomalySeverity.CRITICAL],
            'anomaly_types': [None, AnomalyType.VOLUME_SPIKE, AnomalyType.PRICE_GAP]
        }

        alerts = anomaly_detector._generate_alerts(combined_results)

        assert len(alerts) == 2  # Only scores > threshold should generate alerts
        assert alerts[0]['severity'] == 'high'
        assert alerts[1]['severity'] == 'critical'
        assert 'recommended_actions' in alerts[0]
        assert 'recommended_actions' in alerts[1]

    def test_early_warning_system(self, anomaly_detector):
        """Test early warning system."""
        # Test increasing trend
        combined_results = {
            'combined_scores': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        }

        early_warnings = anomaly_detector._calculate_early_warnings(combined_results)

        assert 'early_warning_active' in early_warnings
        assert 'trend_slope' in early_warnings
        assert 'recent_average_score' in early_warnings

    def test_risk_integration_data(self, anomaly_detector):
        """Test risk integration data format."""
        # Mock fitted state
        anomaly_detector.is_fitted = True
        anomaly_detector.anomaly_history = [
            {'anomaly_detected': True, 'timestamp': datetime.now()},
            {'anomaly_detected': False, 'timestamp': datetime.now()},
            {'anomaly_detected': True, 'timestamp': datetime.now()}
        ]

        risk_data = anomaly_detector.get_risk_integration_data()

        assert 'anomaly_system_active' in risk_data
        assert 'current_risk_multiplier' in risk_data
        assert 'alert_level' in risk_data
        assert 'recommended_position_adjustment' in risk_data
        assert 'recent_anomaly_rate' in risk_data


class TestRiskManagerAnomalyIntegration:
    """Test suite for RiskManager anomaly detection integration."""

    @pytest.fixture
    def mock_anomaly_detector(self):
        """Create mock anomaly detector."""
        detector = Mock()
        detector.is_fitted = True
        detector.get_risk_integration_data.return_value = {
            'anomaly_system_active': True,
            'current_risk_multiplier': 1.2,
            'alert_level': 'medium',
            'recommended_position_adjustment': -0.1,
            'recent_anomaly_rate': 0.3,
            'active_alerts': 2,
            'last_detection_time': datetime.now().isoformat()
        }
        return detector

    @pytest.fixture
    def risk_manager(self):
        """Create RiskManager instance."""
        return RiskManager(
            user_id=1,
            user_risk_params={
                'max_position_size': 10000,
                'max_daily_loss': 1000,
                'max_portfolio_exposure': 0.9,
                'max_single_stock_exposure': 0.15
            }
        )

    def test_anomaly_detector_integration(self, risk_manager, mock_anomaly_detector):
        """Test setting anomaly detector."""
        risk_manager.set_anomaly_detector(mock_anomaly_detector)

        assert risk_manager.anomaly_detector == mock_anomaly_detector
        assert risk_manager.anomaly_risk_adjustment_enabled

    def test_risk_adjustment_from_anomalies(self, risk_manager, mock_anomaly_detector):
        """Test risk parameter adjustment based on anomalies."""
        risk_manager.set_anomaly_detector(mock_anomaly_detector)

        # Store original parameters
        original_position_size = risk_manager.max_position_size
        original_portfolio_exposure = risk_manager.max_portfolio_exposure

        # Update risk from anomalies
        result = risk_manager.update_risk_from_anomalies()

        assert result['anomaly_system_active']
        assert result['current_anomaly_level'] == 'medium'
        assert result['risk_multiplier'] == 1.2

        # Check if parameters were adjusted (made more conservative)
        assert risk_manager.max_position_size < original_position_size
        assert risk_manager.max_portfolio_exposure < original_portfolio_exposure

    def test_order_validation_with_anomalies(self, risk_manager, mock_anomaly_detector):
        """Test order validation with anomaly detection."""
        risk_manager.set_anomaly_detector(mock_anomaly_detector)

        # Test normal order during medium anomaly
        positions = []
        is_valid, reason = risk_manager.validate_order(
            symbol="AAPL",
            quantity=50,
            price=150.0,
            side="BUY",
            current_positions=positions,
            account_value=50000,
            daily_pnl=0
        )

        assert is_valid
        assert "medium anomaly conditions" in reason

    def test_critical_anomaly_blocking(self, risk_manager, mock_anomaly_detector):
        """Test that critical anomalies block new buy orders."""
        # Set critical anomaly level
        mock_anomaly_detector.get_risk_integration_data.return_value = {
            'anomaly_system_active': True,
            'current_risk_multiplier': 2.0,
            'alert_level': 'critical',
            'recommended_position_adjustment': -0.3,
            'recent_anomaly_rate': 0.8,
            'active_alerts': 5,
            'last_detection_time': datetime.now().isoformat()
        }

        risk_manager.set_anomaly_detector(mock_anomaly_detector)

        # Test buy order during critical anomaly
        positions = []
        is_valid, reason = risk_manager.validate_order(
            symbol="AAPL",
            quantity=50,
            price=150.0,
            side="BUY",
            current_positions=positions,
            account_value=50000,
            daily_pnl=0
        )

        assert not is_valid
        assert "critical market anomalies" in reason

    def test_risk_summary_with_anomalies(self, risk_manager, mock_anomaly_detector):
        """Test risk summary includes anomaly information."""
        risk_manager.set_anomaly_detector(mock_anomaly_detector)
        risk_manager.update_risk_from_anomalies()

        positions = []
        summary = risk_manager.get_risk_summary(positions, 50000, 0)

        assert 'anomaly_detection' in summary
        anomaly_info = summary['anomaly_detection']

        assert anomaly_info['anomaly_system_active']
        assert anomaly_info['current_anomaly_level'] == 'medium'
        assert anomaly_info['risk_adjustment_active']
        assert 'risk_adjustments' in anomaly_info

    def test_risk_parameter_reset(self, risk_manager, mock_anomaly_detector):
        """Test risk parameter reset functionality."""
        risk_manager.set_anomaly_detector(mock_anomaly_detector)

        # Store original values
        original_position_size = risk_manager.max_position_size
        original_exposure = risk_manager.max_portfolio_exposure

        # Apply anomaly adjustments
        risk_manager.update_risk_from_anomalies()

        # Verify parameters changed
        assert risk_manager.max_position_size != original_position_size

        # Reset parameters
        risk_manager.reset_risk_parameters()

        # Verify parameters restored
        assert risk_manager.max_position_size == original_position_size
        assert risk_manager.max_portfolio_exposure == original_exposure
        assert risk_manager.current_anomaly_level == 'low'
        assert not risk_manager.risk_adjustment_active

    def test_disabled_anomaly_adjustment(self, risk_manager, mock_anomaly_detector):
        """Test behavior when anomaly adjustment is disabled."""
        risk_manager.anomaly_risk_adjustment_enabled = False
        risk_manager.set_anomaly_detector(mock_anomaly_detector)

        result = risk_manager.update_risk_from_anomalies()

        assert not result['anomaly_system_active']
        assert not result['risk_adjustments_made']
        assert 'disabled' in result['message']


class TestAnomalyDetectionPerformance:
    """Performance and accuracy tests for anomaly detection."""

    def test_detection_performance(self):
        """Test detection performance with larger datasets."""
        # Generate larger dataset
        np.random.seed(42)
        large_data = pd.DataFrame({
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(10000, 100000, 1000),
            'high': np.random.randn(1000).cumsum() + 101,
            'low': np.random.randn(1000).cumsum() + 99,
            'open': np.random.randn(1000).cumsum() + 100
        })

        detector = AnomalyDetector(lookback_period=100)

        # Measure fitting time
        import time
        start_time = time.time()
        detector.fit(large_data[:500])  # Fit on first half
        fit_time = time.time() - start_time

        # Measure prediction time
        start_time = time.time()
        result = detector.predict(large_data[500:])  # Predict on second half
        predict_time = time.time() - start_time

        # Performance assertions
        assert fit_time < 10.0  # Should fit within 10 seconds
        assert predict_time < 5.0  # Should predict within 5 seconds
        assert 'anomaly_detected' in result

    def test_false_positive_rate(self):
        """Test false positive rate on normal data."""
        np.random.seed(42)

        # Generate very normal data (low volatility)
        normal_data = pd.DataFrame({
            'close': np.random.normal(100, 0.5, 200),  # Very low volatility
            'volume': np.random.normal(50000, 5000, 200),
            'high': np.random.normal(100.5, 0.5, 200),
            'low': np.random.normal(99.5, 0.5, 200),
            'open': np.random.normal(100, 0.5, 200)
        })

        detector = AnomalyDetector(
            anomaly_threshold=0.8,  # High threshold
            critical_threshold=0.95
        )

        detector.fit(normal_data[:100])
        result = detector.predict(normal_data[100:])

        # Calculate false positive rate
        anomaly_rate = np.mean(result['anomaly_mask'])

        # Should have low false positive rate on normal data
        assert anomaly_rate < 0.1  # Less than 10% false positives

    def test_detection_sensitivity(self):
        """Test detection sensitivity on data with known anomalies."""
        np.random.seed(42)

        # Generate data with clear anomalies
        data = []
        for i in range(100):
            if i in [25, 50, 75]:  # Inject clear anomalies
                price = 100 + np.random.normal(10, 2)  # Large price jump
                volume = 200000  # Large volume spike
            else:
                price = 100 + np.random.normal(0, 1)  # Normal price
                volume = np.random.normal(50000, 5000)  # Normal volume

            data.append({
                'close': price,
                'volume': volume,
                'high': price * 1.01,
                'low': price * 0.99,
                'open': price
            })

        anomaly_data = pd.DataFrame(data)

        detector = AnomalyDetector(
            anomaly_threshold=0.5,  # Lower threshold for sensitivity
            volume_spike_threshold=2.0,
            price_gap_threshold=0.05
        )

        detector.fit(anomaly_data[:50])
        result = detector.predict(anomaly_data[50:])

        # Should detect some of the injected anomalies
        detected_anomalies = np.sum(result['anomaly_mask'])
        assert detected_anomalies > 0  # Should detect at least some anomalies

    def test_memory_usage(self):
        """Test memory usage with anomaly detection."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Generate large dataset
        large_data = pd.DataFrame({
            'close': np.random.randn(5000).cumsum() + 100,
            'volume': np.random.randint(10000, 100000, 5000),
            'high': np.random.randn(5000).cumsum() + 101,
            'low': np.random.randn(5000).cumsum() + 99,
            'open': np.random.randn(5000).cumsum() + 100
        })

        detector = AnomalyDetector()
        detector.fit(large_data)
        detector.predict(large_data[-100:])

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB)
        assert memory_increase < 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])