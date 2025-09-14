#!/usr/bin/env python3
"""
Simple validation script for anomaly detection system.

This script tests the core functionality without complex pytest dependencies.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.ml.unsupervised.anomaly_detector import AnomalyDetector, AnomalySeverity, AnomalyType
    from app.ml.unsupervised.clustering import DensityBasedDetector
    from app.trading.risk_manager import RiskManager
    print("✓ Successfully imported anomaly detection modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def generate_sample_data(n_points=100, with_anomalies=False):
    """Generate sample market data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='1H')

    base_price = 100.0
    prices = []
    volumes = []

    for i in range(n_points):
        # Normal price movement
        price_change = np.random.normal(0, 0.02)
        if i > 0:
            base_price = prices[-1] * (1 + price_change)

        # Inject anomalies if requested
        if with_anomalies and i in [20, 40, 60]:
            if i == 20:  # Price gap
                base_price *= 1.05
            elif i == 40:  # Volume spike
                volumes.append(500000)
                prices.append(base_price)
                continue
            elif i == 60:  # High volatility
                base_price *= (1 + np.random.normal(0, 0.1))

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


def test_anomaly_detector_basic():
    """Test basic AnomalyDetector functionality."""
    print("\n=== Testing AnomalyDetector Basic Functionality ===")

    try:
        # Initialize detector
        detector = AnomalyDetector(
            lookback_period=50,
            anomaly_threshold=0.7,
            critical_threshold=0.9
        )
        print("✓ AnomalyDetector initialized successfully")

        # Test feature extraction
        normal_data = generate_sample_data(50, with_anomalies=False)
        features = detector._extract_anomaly_features(normal_data)

        assert 'price' in features
        assert 'volume' in features
        assert 'technical' in features
        assert 'multi_timeframe' in features
        print("✓ Feature extraction working correctly")

        # Test specific detectors
        anomaly_data = generate_sample_data(50, with_anomalies=True)
        features_anomaly = detector._extract_anomaly_features(anomaly_data)

        # Volume spike detection
        spike_mask, spike_scores = detector._detect_volume_spikes(features_anomaly)
        print(f"✓ Volume spike detection: {len(spike_scores)} scores, max={np.max(spike_scores):.3f}")

        # Price gap detection
        gap_mask, gap_scores = detector._detect_price_gaps(features_anomaly)
        print(f"✓ Price gap detection: {len(gap_scores)} scores, max={np.max(gap_scores):.3f}")

        # Correlation break detection
        corr_mask, corr_scores = detector._detect_correlation_breaks(features_anomaly)
        print(f"✓ Correlation break detection: {len(corr_scores)} scores, max={np.max(corr_scores) if len(corr_scores) > 0 else 0:.3f}")

        print("✓ All specific detectors working")

        return True

    except Exception as e:
        print(f"✗ AnomalyDetector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_risk_manager_integration():
    """Test RiskManager integration with anomaly detection."""
    print("\n=== Testing RiskManager Integration ===")

    try:
        # Initialize risk manager
        risk_manager = RiskManager(
            user_id=1,
            user_risk_params={
                'max_position_size': 10000,
                'max_daily_loss': 1000,
                'anomaly_risk_adjustment_enabled': True
            }
        )
        print("✓ RiskManager initialized successfully")

        # Test anomaly detector integration
        detector = AnomalyDetector()
        risk_manager.set_anomaly_detector(detector)
        print("✓ Anomaly detector integrated with RiskManager")

        # Test risk adjustment without fitted detector
        result = risk_manager.update_risk_from_anomalies()
        assert not result.get('anomaly_system_active', True)
        print("✓ Handles unfitted detector correctly")

        # Test risk summary with anomaly info
        positions = []
        summary = risk_manager.get_risk_summary(positions, 50000, 0)
        assert 'anomaly_detection' in summary
        print("✓ Risk summary includes anomaly detection info")

        # Test order validation
        is_valid, reason = risk_manager.validate_order(
            symbol="AAPL",
            quantity=50,
            price=150.0,
            side="BUY",
            current_positions=[],
            account_value=50000,
            daily_pnl=0
        )
        print(f"✓ Order validation: {is_valid}, reason: {reason[:50]}...")

        return True

    except Exception as e:
        print(f"✗ RiskManager integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Test performance with larger datasets."""
    print("\n=== Testing Performance ===")

    try:
        # Generate larger dataset
        large_data = generate_sample_data(500, with_anomalies=True)
        print(f"✓ Generated dataset with {len(large_data)} points")

        detector = AnomalyDetector(lookback_period=100)

        # Test feature extraction performance
        import time
        start_time = time.time()
        features = detector._extract_anomaly_features(large_data)
        feature_time = time.time() - start_time
        print(f"✓ Feature extraction: {feature_time:.3f}s for {len(large_data)} points")

        # Test detection performance
        start_time = time.time()
        volume_mask, volume_scores = detector._detect_volume_spikes(features)
        gap_mask, gap_scores = detector._detect_price_gaps(features)
        corr_mask, corr_scores = detector._detect_correlation_breaks(features)
        detection_time = time.time() - start_time
        print(f"✓ Anomaly detection: {detection_time:.3f}s for {len(large_data)} points")

        # Performance assertions
        assert feature_time < 5.0, f"Feature extraction too slow: {feature_time:.3f}s"
        assert detection_time < 2.0, f"Detection too slow: {detection_time:.3f}s"
        print("✓ Performance tests passed")

        return True

    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


def test_anomaly_detection_sensitivity():
    """Test detection sensitivity on data with known anomalies."""
    print("\n=== Testing Detection Sensitivity ===")

    try:
        # Generate data with clear anomalies
        detector = AnomalyDetector(
            anomaly_threshold=0.5,
            volume_spike_threshold=2.0,
            price_gap_threshold=0.03
        )

        # Test on data with injected anomalies
        anomaly_data = generate_sample_data(80, with_anomalies=True)
        features = detector._extract_anomaly_features(anomaly_data)

        # Check volume spike detection
        volume_mask, volume_scores = detector._detect_volume_spikes(features)
        volume_detections = np.sum(volume_scores > 0.5)
        print(f"✓ Volume spike detections: {volume_detections}/{len(volume_scores)}")

        # Check price gap detection
        gap_mask, gap_scores = detector._detect_price_gaps(features)
        gap_detections = np.sum(gap_scores > 0.5)
        print(f"✓ Price gap detections: {gap_detections}/{len(gap_scores)}")

        # Should detect some anomalies
        total_detections = volume_detections + gap_detections
        assert total_detections > 0, "Should detect at least some injected anomalies"
        print(f"✓ Sensitivity test passed: {total_detections} total detections")

        return True

    except Exception as e:
        print(f"✗ Sensitivity test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("Starting Anomaly Detection System Validation")
    print("=" * 50)

    tests = [
        test_anomaly_detector_basic,
        test_risk_manager_integration,
        test_performance,
        test_anomaly_detection_sensitivity
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")

    print("\n" + "=" * 50)
    print(f"Validation Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Anomaly detection system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())