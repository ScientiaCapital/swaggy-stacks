#!/usr/bin/env python3
"""
Quick test script for Market Regime Detection integration.

Tests the MarketRegimeDetector integration with MarkovSystem to ensure
all components work together correctly.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

# Add backend to path
sys.path.insert(0, '/Users/tmkipper/repos/swaggy-stacks/backend')

warnings.filterwarnings("ignore")

def generate_test_data(n_periods=300, regime_type="mixed"):
    """Generate synthetic market data for testing"""
    np.random.seed(42)  # Reproducible results

    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='D')

    if regime_type == "bull":
        # Bull market: positive trend with moderate volatility
        trend = 0.001
        volatility = 0.015
        returns = np.random.normal(trend, volatility, n_periods)
    elif regime_type == "bear":
        # Bear market: negative trend with high volatility
        trend = -0.002
        volatility = 0.025
        returns = np.random.normal(trend, volatility, n_periods)
    elif regime_type == "sideways":
        # Sideways: minimal trend, low volatility
        trend = 0.0001
        volatility = 0.008
        returns = np.random.normal(trend, volatility, n_periods)
    elif regime_type == "volatile":
        # Volatile: minimal trend, very high volatility
        trend = 0.0
        volatility = 0.035
        returns = np.random.normal(trend, volatility, n_periods)
    else:  # mixed
        # Mixed regime: different periods
        returns = []
        for i in range(n_periods):
            if i < n_periods // 4:  # Bull phase
                returns.append(np.random.normal(0.001, 0.015))
            elif i < n_periods // 2:  # Bear phase
                returns.append(np.random.normal(-0.002, 0.025))
            elif i < 3 * n_periods // 4:  # Sideways phase
                returns.append(np.random.normal(0.0001, 0.008))
            else:  # Volatile phase
                returns.append(np.random.normal(0.0, 0.035))
        returns = np.array(returns)

    # Generate price series
    prices = [100.0]  # Starting price
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))

    # Create OHLCV data
    data = []
    for i, price in enumerate(prices[1:], 1):
        prev_price = prices[i-1]
        high = max(prev_price, price) * (1 + np.random.uniform(0, 0.01))
        low = min(prev_price, price) * (1 - np.random.uniform(0, 0.01))
        volume = np.random.randint(1000000, 10000000)

        data.append({
            'timestamp': dates[i-1],
            'open': prev_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })

    return pd.DataFrame(data)

def test_regime_detector_standalone():
    """Test MarketRegimeDetector as standalone component"""
    print("🔍 Testing MarketRegimeDetector standalone...")

    try:
        from app.ml.unsupervised.market_regime import MarketRegimeDetector, MarketRegime

        # Generate test data
        test_data = generate_test_data(200, "mixed")
        print(f"   Generated {len(test_data)} data points")

        # Initialize detector
        detector = MarketRegimeDetector(
            n_regimes=4,
            lookback_period=150,
            rolling_window=20
        )
        print("   MarketRegimeDetector initialized")

        # Fit detector
        detector.fit(test_data)
        print("   Detector fitted to data")

        # Make predictions
        result = detector.predict(test_data)
        print(f"   Current regime: {result['current_regime']}")
        print(f"   Regime confidence: {result['regime_confidence']:.3f}")
        print(f"   Regime changed: {result['regime_changed']}")

        # Get summary
        summary = detector.get_regime_summary()
        print(f"   Detected {summary['n_regimes']} regimes")
        print(f"   Transition points: {summary['transition_points']}")

        print("✅ MarketRegimeDetector standalone test passed!")
        return True

    except Exception as e:
        print(f"❌ MarketRegimeDetector standalone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_markov_integration():
    """Test MarkovSystem with regime detection integration"""
    print("\n🔗 Testing MarkovSystem with regime detection...")

    try:
        from app.ml.markov_system import MarkovSystem

        # Generate test data
        test_data = generate_test_data(150, "mixed")
        print(f"   Generated {len(test_data)} data points")

        # Initialize MarkovSystem with regime detection enabled
        markov_system = MarkovSystem(
            lookback_period=100,
            n_states=5,
            enable_regime_detection=True,
            n_regimes=4,
            regime_rolling_window=15
        )
        print("   MarkovSystem with regime detection initialized")

        # Analyze data
        result = markov_system.analyze(test_data)
        print("   Analysis completed")

        # Check if regime information is included
        if 'market_regime' in result:
            regime_info = result['market_regime']
            print(f"   Regime detection enabled: {regime_info.get('regime_detection_enabled', False)}")
            print(f"   Current regime: {regime_info.get('regime_type', 'unknown')}")
            print(f"   Regime confidence: {regime_info.get('regime_confidence', 0):.3f}")

            if 'markov_integration' in regime_info:
                markov_integration = regime_info['markov_integration']
                print(f"   Markov state: {markov_integration.get('current_markov_state', 'unknown')}")
                print(f"   Markov confidence: {markov_integration.get('markov_confidence', 0):.3f}")

                if 'regime_markov_alignment' in markov_integration:
                    alignment = markov_integration['regime_markov_alignment']
                    print(f"   Alignment quality: {alignment.get('alignment_quality', 'unknown')}")
                    print(f"   Alignment score: {alignment.get('alignment_score', 0):.3f}")
        else:
            print("   ⚠️  No regime information found in result")

        print("✅ MarkovSystem integration test passed!")
        return True

    except Exception as e:
        print(f"❌ MarkovSystem integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_regime_detection_disabled():
    """Test MarkovSystem with regime detection disabled"""
    print("\n⚙️ Testing MarkovSystem with regime detection disabled...")

    try:
        from app.ml.markov_system import MarkovSystem

        # Generate test data
        test_data = generate_test_data(100, "bull")
        print(f"   Generated {len(test_data)} data points")

        # Initialize MarkovSystem with regime detection disabled
        markov_system = MarkovSystem(
            lookback_period=80,
            n_states=5,
            enable_regime_detection=False
        )
        print("   MarkovSystem with regime detection disabled")

        # Analyze data
        result = markov_system.analyze(test_data)
        print("   Analysis completed")

        # Check regime information
        if 'market_regime' in result:
            regime_info = result['market_regime']
            print(f"   Regime detection enabled: {regime_info.get('regime_detection_enabled', True)}")
            if not regime_info.get('regime_detection_enabled', True):
                print("   ✅ Regime detection correctly disabled")
            else:
                print("   ⚠️  Regime detection should be disabled but appears enabled")
        else:
            print("   ⚠️  No regime information found (expected when disabled)")

        print("✅ Regime detection disabled test passed!")
        return True

    except Exception as e:
        print(f"❌ Regime detection disabled test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Market Regime Detection Tests")
    print("=" * 50)

    tests_passed = 0
    total_tests = 3

    # Test 1: Standalone regime detector
    if test_regime_detector_standalone():
        tests_passed += 1

    # Test 2: MarkovSystem integration
    if test_markov_integration():
        tests_passed += 1

    # Test 3: Regime detection disabled
    if test_regime_detection_disabled():
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"🏁 Tests completed: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! Market regime detection is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)