"""
Comprehensive unit tests for technical indicators based on actual API
"""

import pytest
import numpy as np
import pandas as pd
from app.indicators.technical_indicators import TechnicalIndicators


class TestTechnicalIndicators:
    """Test suite for technical indicators calculations"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data for testing (minimum 50 periods)"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # Generate realistic price data with trend
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 100)
        prices = [base_price]

        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))

        # Ensure proper OHLCV structure
        ohlcv_data = []
        for i, close_price in enumerate(prices):
            high = close_price * (1 + np.random.uniform(0, 0.02))
            low = close_price * (1 - np.random.uniform(0, 0.02))
            open_price = close_price + np.random.normal(0, close_price * 0.01)
            volume = np.random.uniform(100000, 1000000)

            ohlcv_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })

        return pd.DataFrame(ohlcv_data, index=dates)

    @pytest.fixture
    def indicators(self):
        """Create TechnicalIndicators instance"""
        return TechnicalIndicators()

    def test_calculate_all_indicators_structure(self, indicators, sample_data):
        """Test that all indicators returns proper structure"""
        result = indicators.calculate_all_indicators(sample_data)

        assert isinstance(result, dict)

        # Should have multiple indicators (flat structure)
        expected_indicators = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 'rsi',
            'bb_upper', 'bb_lower', 'atr', 'obv', 'composite_signals'
        ]

        # At least some indicators should be present
        present_indicators = [ind for ind in expected_indicators if ind in result]
        assert len(present_indicators) > 0, "No indicators found"

    def test_trend_indicators(self, indicators, sample_data):
        """Test trend indicator calculations"""
        result = indicators.calculate_all_indicators(sample_data)

        # Check common trend indicators
        common_trend_indicators = ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 'macd_signal', 'sar', 'adx']

        for indicator in common_trend_indicators:
            if indicator in result:
                value = result[indicator]
                assert value is not None
                # Should be either a number or list/array
                assert isinstance(value, (int, float, list, np.ndarray))

    def test_momentum_indicators(self, indicators, sample_data):
        """Test momentum indicator calculations"""
        result = indicators.calculate_all_indicators(sample_data)

        # Check for RSI bounds (should be between 0-100)
        if 'rsi' in result:
            rsi_value = result['rsi']
            if isinstance(rsi_value, (int, float)):
                assert 0 <= rsi_value <= 100
            elif isinstance(rsi_value, (list, np.ndarray)):
                rsi_array = np.array(rsi_value)
                valid_rsi = rsi_array[~np.isnan(rsi_array)]
                if len(valid_rsi) > 0:
                    assert np.all((valid_rsi >= 0) & (valid_rsi <= 100))

        # Check other momentum indicators
        momentum_indicators = ['stoch_k', 'stoch_d', 'williams_r', 'cci', 'roc', 'momentum']
        for indicator in momentum_indicators:
            if indicator in result:
                value = result[indicator]
                assert value is not None

    def test_volatility_indicators(self, indicators, sample_data):
        """Test volatility indicator calculations"""
        result = indicators.calculate_all_indicators(sample_data)

        # Check for common volatility indicators
        common_vol_indicators = ['atr', 'bb_upper', 'bb_lower', 'bb_middle', 'kc_upper', 'kc_lower']

        for indicator in common_vol_indicators:
            if indicator in result:
                value = result[indicator]
                assert value is not None

        # Bollinger Bands relationship test
        if all(key in result for key in ['bb_upper', 'bb_lower']):
            upper = result['bb_upper']
            lower = result['bb_lower']

            if isinstance(upper, (int, float)) and isinstance(lower, (int, float)):
                assert upper >= lower, "Bollinger upper band should be >= lower band"

    def test_volume_indicators(self, indicators, sample_data):
        """Test volume indicator calculations"""
        result = indicators.calculate_all_indicators(sample_data)

        # Volume indicators should exist
        common_volume_indicators = ['obv', 'ad', 'cmf', 'volume_roc']

        for indicator in common_volume_indicators:
            if indicator in result:
                value = result[indicator]
                assert value is not None

    def test_support_resistance_levels(self, indicators, sample_data):
        """Test support and resistance level calculations"""
        result = indicators.calculate_all_indicators(sample_data)

        # Check for support/resistance levels
        sr_indicators = ['pivot', 'resistance_1', 'support_1', 'resistance_2', 'support_2', 'recent_high', 'recent_low']

        for indicator in sr_indicators:
            if indicator in result:
                value = result[indicator]
                assert value is not None

    def test_fibonacci_levels(self, indicators, sample_data):
        """Test Fibonacci retracement calculations"""
        result = indicators.calculate_all_indicators(sample_data)

        # Common Fibonacci levels
        fib_levels = ['fib_0', 'fib_23.6', 'fib_38.2', 'fib_50', 'fib_61.8', 'fib_100']

        for level in fib_levels:
            if level in result:
                value = result[level]
                assert isinstance(value, (int, float))

    def test_composite_signals(self, indicators, sample_data):
        """Test composite signal generation"""
        result = indicators.calculate_all_indicators(sample_data)

        if 'composite_signals' in result:
            signals = result['composite_signals']
            assert isinstance(signals, dict)

            # Check for signal components
            signal_components = ['trend', 'momentum', 'volatility', 'volume', 'composite', 'signal_strength']

            for component in signal_components:
                if component in signals:
                    value = signals[component]
                    assert value is not None

    def test_insufficient_data_handling(self, indicators):
        """Test handling of insufficient data"""
        # Create dataset with less than 50 periods
        short_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })

        with pytest.raises(Exception):  # Should raise TradingError
            indicators.calculate_all_indicators(short_data)

    def test_invalid_data_structure(self, indicators):
        """Test handling of invalid data structure"""
        # Missing required columns
        invalid_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })

        with pytest.raises(Exception):
            indicators.calculate_all_indicators(invalid_data)

    def test_realistic_data_ranges(self, indicators, sample_data):
        """Test that calculated indicators are within realistic ranges"""
        result = indicators.calculate_all_indicators(sample_data)

        # Check that results are reasonable
        if 'momentum' in result and 'rsi' in result['momentum']:
            rsi = result['momentum']['rsi']
            if isinstance(rsi, (int, float)):
                assert 0 <= rsi <= 100

        # Check volatility indicators are positive
        if 'volatility' in result and 'atr' in result['volatility']:
            atr = result['volatility']['atr']
            if isinstance(atr, (int, float)):
                assert atr >= 0

    def test_data_with_gaps(self, indicators):
        """Test handling of data with missing values"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')

        # Create data with some NaN values
        prices = np.random.uniform(100, 110, 60)
        prices[10:15] = np.nan  # Insert gaps

        gap_data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(100000, 200000, 60)
        }, index=dates)

        # Should handle gaps gracefully or raise appropriate error
        try:
            result = indicators.calculate_all_indicators(gap_data.dropna())
            assert isinstance(result, dict)
        except Exception as e:
            # Expected if gaps cause issues
            assert "data" in str(e).lower()

    def test_extreme_volatility_data(self, indicators):
        """Test handling of extreme volatility scenarios"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')

        # Create extremely volatile data
        base_price = 100
        extreme_returns = np.random.normal(0, 0.1, 60)  # 10% daily volatility
        prices = [base_price]

        for ret in extreme_returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        extreme_data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.05))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.05))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(50000, 500000, 60)
        }, index=dates)

        result = indicators.calculate_all_indicators(extreme_data)
        assert isinstance(result, dict)

        # Should still produce valid indicators even with extreme data
        assert len(result) > 0

    @pytest.mark.parametrize("data_length", [50, 100, 200])
    def test_different_data_lengths(self, indicators, data_length):
        """Test indicators with different data lengths"""
        dates = pd.date_range('2023-01-01', periods=data_length, freq='D')

        # Generate price data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, data_length)
        prices = [100]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.uniform(100000, 200000, data_length)
        }, index=dates)

        result = indicators.calculate_all_indicators(test_data)
        assert isinstance(result, dict)
        assert len(result) > 0