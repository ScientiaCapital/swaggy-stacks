"""
Unit tests for VolatilityPredictor

Tests GARCH model volatility prediction, regime detection,
and volatility surface analysis for options trading.

Test Coverage:
- GARCH(1,1) model fitting and prediction
- Volatility regime identification
- Historical volatility calculations
- Implied volatility surface analysis
- Risk metrics and volatility forecasting
- Model validation and diagnostics
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.ml.volatility_predictor import VolatilityPredictor
from app.core.exceptions import ModelError, DataError


class TestVolatilityPredictor:
    """Test suite for VolatilityPredictor GARCH models and analysis"""

    @pytest.fixture
    def sample_price_data(self):
        """Sample price data for testing volatility calculations"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=252),
            end=datetime.now(),
            freq='D'
        )

        # Generate realistic price movement with volatility clustering
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, len(dates))

        # Add volatility clustering (GARCH-like behavior)
        volatility = np.ones(len(dates)) * 0.02
        for i in range(1, len(dates)):
            volatility[i] = 0.05 + 0.85 * volatility[i-1] + 0.10 * returns[i-1]**2
            returns[i] = np.random.normal(0.0005, volatility[i])

        # Convert to prices
        prices = [100.0]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        return pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'volume': np.random.randint(1000000, 5000000, len(dates))
        })

    @pytest.fixture
    def sample_options_data(self):
        """Sample options chain data for implied volatility analysis"""
        strikes = np.arange(95, 106, 1)
        expiry_dates = [
            datetime.now() + timedelta(days=7),
            datetime.now() + timedelta(days=14),
            datetime.now() + timedelta(days=30),
            datetime.now() + timedelta(days=60)
        ]

        options_data = []
        for expiry in expiry_dates:
            for strike in strikes:
                # Simulate realistic implied volatility smile
                moneyness = strike / 100.0
                time_to_expiry = (expiry - datetime.now()).days / 365.0

                # Volatility smile with higher vol for OTM options
                base_vol = 0.25
                smile_factor = 0.1 * (moneyness - 1.0)**2
                iv = base_vol + smile_factor + 0.05 * np.sqrt(time_to_expiry)

                options_data.append({
                    'symbol': f'AAPL{expiry.strftime("%y%m%d")}{"C" if strike >= 100 else "P"}{int(strike*1000):08d}',
                    'strike_price': strike,
                    'expiry_date': expiry,
                    'option_type': 'call' if strike >= 100 else 'put',
                    'implied_volatility': iv,
                    'bid': max(0.01, iv * 100 * 0.08),
                    'ask': iv * 100 * 0.12,
                    'volume': np.random.randint(0, 1000),
                    'open_interest': np.random.randint(100, 5000)
                })

        return pd.DataFrame(options_data)

    @pytest.fixture
    def volatility_predictor(self):
        """VolatilityPredictor instance for testing"""
        return VolatilityPredictor(
            lookback_days=252,
            garch_p=1,
            garch_q=1,
            forecast_horizon=5
        )

    @pytest.mark.asyncio
    async def test_calculate_historical_volatility_basic(self, volatility_predictor, sample_price_data):
        """Test basic historical volatility calculation"""
        vol_metrics = await volatility_predictor.calculate_historical_volatility(
            price_data=sample_price_data,
            symbol="AAPL",
            periods=[10, 20, 60]
        )

        assert vol_metrics.symbol == "AAPL"
        assert len(vol_metrics.period_volatilities) == 3
        assert 10 in vol_metrics.period_volatilities
        assert 20 in vol_metrics.period_volatilities
        assert 60 in vol_metrics.period_volatilities

        # Volatilities should be positive and reasonable
        for period, vol in vol_metrics.period_volatilities.items():
            assert vol > 0
            assert vol < 1.0  # Less than 100% annual vol

    @pytest.mark.asyncio
    async def test_fit_garch_model_success(self, volatility_predictor, sample_price_data):
        """Test successful GARCH(1,1) model fitting"""
        # Calculate returns for GARCH model
        returns = sample_price_data['close'].pct_change().dropna()

        garch_result = await volatility_predictor.fit_garch_model(
            returns=returns,
            symbol="AAPL"
        )

        assert garch_result.symbol == "AAPL"
        assert garch_result.model_type == "GARCH(1,1)"
        assert garch_result.omega > 0  # Constant term should be positive
        assert garch_result.alpha > 0  # ARCH coefficient should be positive
        assert garch_result.beta > 0   # GARCH coefficient should be positive
        assert garch_result.alpha + garch_result.beta < 1  # Stationarity condition
        assert garch_result.aic is not None
        assert garch_result.bic is not None

    @pytest.mark.asyncio
    async def test_predict_volatility_forecast(self, volatility_predictor, sample_price_data):
        """Test volatility forecasting with GARCH model"""
        returns = sample_price_data['close'].pct_change().dropna()

        forecast = await volatility_predictor.predict_volatility(
            returns=returns,
            symbol="AAPL",
            forecast_days=5
        )

        assert forecast.symbol == "AAPL"
        assert forecast.forecast_horizon == 5
        assert len(forecast.daily_forecasts) == 5
        assert len(forecast.confidence_intervals) == 5

        # Check forecast values are reasonable
        for i, vol_forecast in enumerate(forecast.daily_forecasts):
            assert vol_forecast > 0
            assert vol_forecast < 1.0

            # Confidence intervals should be properly ordered
            lower, upper = forecast.confidence_intervals[i]
            assert lower < vol_forecast < upper

    @pytest.mark.asyncio
    async def test_detect_volatility_regime(self, volatility_predictor, sample_price_data):
        """Test volatility regime detection"""
        returns = sample_price_data['close'].pct_change().dropna()

        regime = await volatility_predictor.detect_volatility_regime(
            returns=returns,
            symbol="AAPL"
        )

        assert regime.symbol == "AAPL"
        assert regime.current_regime in ["low", "normal", "high", "extreme"]
        assert 0 <= regime.regime_probability <= 1
        assert regime.volatility_percentile >= 0
        assert regime.volatility_percentile <= 100

        # Historical regimes should have valid classifications
        assert len(regime.historical_regimes) > 0
        for hist_regime in regime.historical_regimes:
            assert hist_regime in ["low", "normal", "high", "extreme"]

    @pytest.mark.asyncio
    async def test_analyze_implied_volatility_surface(self, volatility_predictor, sample_options_data):
        """Test implied volatility surface analysis"""
        iv_analysis = await volatility_predictor.analyze_implied_volatility_surface(
            options_data=sample_options_data,
            underlying_price=Decimal("100.00"),
            symbol="AAPL"
        )

        assert iv_analysis.symbol == "AAPL"
        assert iv_analysis.underlying_price == Decimal("100.00")
        assert len(iv_analysis.volatility_surface) > 0

        # Check term structure analysis
        assert len(iv_analysis.term_structure) > 0
        for expiry, atm_vol in iv_analysis.term_structure.items():
            assert atm_vol > 0
            assert atm_vol < 2.0  # Reasonable vol levels

        # Check volatility smile/skew analysis
        assert iv_analysis.volatility_skew is not None
        assert iv_analysis.smile_curvature is not None

    @pytest.mark.asyncio
    async def test_calculate_risk_metrics(self, volatility_predictor, sample_price_data):
        """Test risk metrics calculation from volatility analysis"""
        returns = sample_price_data['close'].pct_change().dropna()
        current_price = Decimal(str(sample_price_data['close'].iloc[-1]))

        risk_metrics = await volatility_predictor.calculate_risk_metrics(
            returns=returns,
            current_price=current_price,
            position_size=Decimal("1000"),
            confidence_level=0.95
        )

        assert risk_metrics.position_size == Decimal("1000")
        assert risk_metrics.confidence_level == 0.95
        assert risk_metrics.var_1day < 0  # VaR should be negative (loss)
        assert risk_metrics.expected_shortfall < risk_metrics.var_1day  # ES should be more negative than VaR
        assert risk_metrics.volatility_annual > 0
        assert 0 <= risk_metrics.sharpe_ratio <= 5  # Reasonable Sharpe ratio range

    @pytest.mark.asyncio
    async def test_validate_garch_assumptions(self, volatility_predictor, sample_price_data):
        """Test GARCH model assumption validation"""
        returns = sample_price_data['close'].pct_change().dropna()

        validation = await volatility_predictor.validate_garch_assumptions(
            returns=returns,
            symbol="AAPL"
        )

        assert validation.symbol == "AAPL"
        assert validation.ljung_box_pvalue is not None
        assert validation.arch_lm_pvalue is not None
        assert validation.jarque_bera_pvalue is not None
        assert validation.normality_test_pvalue is not None

        # Validation flags should be boolean
        assert isinstance(validation.serial_correlation_detected, bool)
        assert isinstance(validation.arch_effects_present, bool)
        assert isinstance(validation.normality_rejected, bool)
        assert isinstance(validation.model_adequate, bool)

    @pytest.mark.asyncio
    async def test_fit_garch_model_insufficient_data(self, volatility_predictor):
        """Test GARCH model fitting with insufficient data"""
        # Very short return series
        short_returns = pd.Series([0.01, -0.02, 0.015])

        with pytest.raises(DataError) as exc_info:
            await volatility_predictor.fit_garch_model(
                returns=short_returns,
                symbol="TEST"
            )

        assert "insufficient data" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_handle_extreme_volatility_values(self, volatility_predictor):
        """Test handling of extreme volatility values"""
        # Create returns with extreme values
        extreme_returns = pd.Series([0.001] * 100 + [0.5, -0.6] + [0.001] * 100)

        vol_metrics = await volatility_predictor.calculate_historical_volatility(
            price_data=pd.DataFrame({
                'close': np.cumprod(1 + extreme_returns) * 100,
                'timestamp': pd.date_range('2023-01-01', periods=len(extreme_returns), freq='D')
            }),
            symbol="EXTREME",
            periods=[10, 20]
        )

        # Should handle extreme values gracefully
        assert vol_metrics.symbol == "EXTREME"
        assert all(vol > 0 for vol in vol_metrics.period_volatilities.values())

    @pytest.mark.asyncio
    async def test_volatility_clustering_detection(self, volatility_predictor, sample_price_data):
        """Test detection of volatility clustering (GARCH effects)"""
        returns = sample_price_data['close'].pct_change().dropna()

        clustering = await volatility_predictor.detect_volatility_clustering(
            returns=returns,
            symbol="AAPL"
        )

        assert clustering.symbol == "AAPL"
        assert clustering.arch_lm_statistic > 0
        assert 0 <= clustering.arch_lm_pvalue <= 1
        assert isinstance(clustering.clustering_detected, bool)
        assert clustering.clustering_strength >= 0

    @pytest.mark.asyncio
    async def test_compare_realized_vs_implied_volatility(self, volatility_predictor, sample_price_data, sample_options_data):
        """Test comparison between realized and implied volatility"""
        returns = sample_price_data['close'].pct_change().dropna()

        comparison = await volatility_predictor.compare_realized_vs_implied(
            returns=returns,
            options_data=sample_options_data,
            symbol="AAPL",
            comparison_period=30
        )

        assert comparison.symbol == "AAPL"
        assert comparison.realized_vol_30d > 0
        assert comparison.avg_implied_vol > 0
        assert comparison.vol_spread is not None  # Can be positive or negative
        assert comparison.vol_ratio > 0
        assert comparison.regime_classification in ["backwardation", "contango", "fair_value"]

    @pytest.mark.asyncio
    async def test_calculate_volatility_cone(self, volatility_predictor, sample_price_data):
        """Test volatility cone calculation for historical context"""
        vol_cone = await volatility_predictor.calculate_volatility_cone(
            price_data=sample_price_data,
            symbol="AAPL",
            periods=[10, 20, 30, 60, 90]
        )

        assert vol_cone.symbol == "AAPL"
        assert len(vol_cone.periods) == 5

        for period in [10, 20, 30, 60, 90]:
            assert period in vol_cone.current_volatilities
            assert period in vol_cone.percentile_rankings
            assert period in vol_cone.historical_ranges

            # Current volatility should be positive
            assert vol_cone.current_volatilities[period] > 0

            # Percentile should be between 0 and 100
            assert 0 <= vol_cone.percentile_rankings[period] <= 100

            # Historical range should have min < max
            min_vol, max_vol = vol_cone.historical_ranges[period]
            assert min_vol < max_vol

    @pytest.mark.asyncio
    async def test_model_diagnostics_and_residuals(self, volatility_predictor, sample_price_data):
        """Test GARCH model diagnostics and residual analysis"""
        returns = sample_price_data['close'].pct_change().dropna()

        diagnostics = await volatility_predictor.analyze_model_diagnostics(
            returns=returns,
            symbol="AAPL"
        )

        assert diagnostics.symbol == "AAPL"
        assert diagnostics.log_likelihood is not None
        assert diagnostics.aic > 0
        assert diagnostics.bic > 0
        assert len(diagnostics.standardized_residuals) > 0
        assert len(diagnostics.squared_residuals) > 0

        # Residuals should have reasonable statistical properties
        assert abs(np.mean(diagnostics.standardized_residuals)) < 0.1  # Should be close to zero
        assert 0.8 < np.std(diagnostics.standardized_residuals) < 1.2   # Should be close to 1

    def test_volatility_predictor_initialization(self):
        """Test VolatilityPredictor initialization with various parameters"""
        # Test default initialization
        predictor = VolatilityPredictor()
        assert predictor.lookback_days == 252
        assert predictor.garch_p == 1
        assert predictor.garch_q == 1
        assert predictor.forecast_horizon == 10

        # Test custom initialization
        custom_predictor = VolatilityPredictor(
            lookback_days=500,
            garch_p=2,
            garch_q=2,
            forecast_horizon=20
        )
        assert custom_predictor.lookback_days == 500
        assert custom_predictor.garch_p == 2
        assert custom_predictor.garch_q == 2
        assert custom_predictor.forecast_horizon == 20

    @pytest.mark.asyncio
    async def test_error_handling_invalid_data(self, volatility_predictor):
        """Test error handling with invalid input data"""
        # Test with None data
        with pytest.raises(DataError):
            await volatility_predictor.calculate_historical_volatility(
                price_data=None,
                symbol="TEST"
            )

        # Test with empty DataFrame
        with pytest.raises(DataError):
            await volatility_predictor.calculate_historical_volatility(
                price_data=pd.DataFrame(),
                symbol="TEST"
            )

    @pytest.mark.asyncio
    async def test_performance_large_dataset(self, volatility_predictor):
        """Test performance with large dataset"""
        # Generate large dataset (5 years of daily data)
        large_dates = pd.date_range(start='2019-01-01', end='2024-01-01', freq='D')
        np.random.seed(123)

        large_returns = np.random.normal(0.0005, 0.02, len(large_dates))
        large_prices = np.cumprod(1 + large_returns) * 100

        large_data = pd.DataFrame({
            'timestamp': large_dates,
            'close': large_prices,
            'high': large_prices * 1.01,
            'low': large_prices * 0.99,
            'volume': np.random.randint(1000000, 5000000, len(large_dates))
        })

        # Should complete within reasonable time
        import time
        start_time = time.time()

        vol_metrics = await volatility_predictor.calculate_historical_volatility(
            price_data=large_data,
            symbol="LARGE_TEST",
            periods=[20, 60, 252]
        )

        execution_time = time.time() - start_time

        assert vol_metrics.symbol == "LARGE_TEST"
        assert execution_time < 10.0  # Should complete within 10 seconds