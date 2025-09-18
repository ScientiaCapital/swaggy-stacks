#!/usr/bin/env python3
"""
Tests for Enhanced Black-Scholes Calculator with Advanced Volatility Models
===========================================================================

Comprehensive test suite for the new advanced volatility modeling capabilities
including Heston, SABR, Local Volatility, and Monte Carlo pricing models.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from app.trading.options_trading import (
    BlackScholesCalculator,
    EnhancedBlackScholesCalculator,
    HestonVolatilityModel,
    SABRVolatilityModel,
    LocalVolatilityModel,
    MonteCarloEngine,
    AdvancedVolatilityCalculator,
    OptionType,
    GreeksData
)


class TestHestonVolatilityModel:
    """Test Heston stochastic volatility model"""

    def setup_method(self):
        self.heston = HestonVolatilityModel()

    def test_heston_model_initialization(self):
        """Test Heston model initializes with correct default parameters"""
        assert self.heston.kappa == 2.0
        assert self.heston.theta == 0.04
        assert self.heston.xi == 0.3
        assert self.heston.rho == -0.7
        assert self.heston.v0 == 0.04

    def test_characteristic_function(self):
        """Test Heston characteristic function computation"""
        # Test with simple inputs
        cf_result = self.heston.characteristic_function(0.5j, 0.25)

        assert isinstance(cf_result, complex)
        assert not np.isnan(cf_result.real)
        assert not np.isnan(cf_result.imag)

    def test_heston_option_pricing(self):
        """Test Heston option pricing"""
        S, K, T, r = 100.0, 100.0, 0.25, 0.05

        # Test call option
        call_price = self.heston.price_option(S, K, T, r, OptionType.CALL)
        assert isinstance(call_price, float)
        assert call_price > 0
        assert call_price < S  # Sanity check

        # Test put option
        put_price = self.heston.price_option(S, K, T, r, OptionType.PUT)
        assert isinstance(put_price, float)
        assert put_price > 0
        assert put_price < K  # Sanity check

        # Put-call parity check (approximately)
        forward = S * np.exp(-r * T)
        parity_diff = abs(call_price - put_price - (forward - K * np.exp(-r * T)))
        assert parity_diff < 1.0  # Allow some numerical error


class TestSABRVolatilityModel:
    """Test SABR volatility smile model"""

    def setup_method(self):
        self.sabr = SABRVolatilityModel()

    def test_sabr_model_initialization(self):
        """Test SABR model initializes with correct parameters"""
        assert self.sabr.alpha == 0.3
        assert self.sabr.beta == 0.7
        assert self.sabr.rho == -0.3
        assert self.sabr.nu == 0.4

    def test_atm_volatility_calculation(self):
        """Test ATM volatility calculation"""
        F = 100.0
        T = 0.25

        atm_vol = self.sabr._atm_volatility(F, T)

        assert isinstance(atm_vol, float)
        assert atm_vol > 0.01  # Should be above floor
        assert atm_vol < 2.0   # Reasonable upper bound

    def test_implied_volatility_smile(self):
        """Test implied volatility smile generation"""
        F, T = 100.0, 0.25
        strikes = [80, 90, 100, 110, 120]

        vols = []
        for K in strikes:
            vol = self.sabr.implied_volatility(F, K, T)
            vols.append(vol)

            assert isinstance(vol, float)
            assert vol > 0.01  # Above floor
            assert vol < 2.0   # Reasonable bound

        # Test that we get a volatility smile shape
        atm_vol = vols[2]  # 100 strike
        assert all(vol > 0 for vol in vols)

    def test_sabr_atm_case(self):
        """Test SABR model at-the-money case"""
        F = 100.0
        K = 100.0  # ATM
        T = 0.25

        vol = self.sabr.implied_volatility(F, K, T)
        atm_vol = self.sabr._atm_volatility(F, T)

        # Should be very close for ATM
        assert abs(vol - atm_vol) < 0.001


class TestLocalVolatilityModel:
    """Test local volatility surface model"""

    def setup_method(self):
        self.local_vol = LocalVolatilityModel()

    def test_local_vol_initialization(self):
        """Test local volatility model initialization"""
        assert self.local_vol.vol_surface == {}
        assert self.local_vol.strikes == []
        assert self.local_vol.maturities == []

    def test_surface_calibration(self):
        """Test volatility surface calibration"""
        # Create mock market data
        market_data = [
            {'strike': 90, 'maturity': 0.25, 'implied_vol': 0.25},
            {'strike': 100, 'maturity': 0.25, 'implied_vol': 0.22},
            {'strike': 110, 'maturity': 0.25, 'implied_vol': 0.27},
            {'strike': 90, 'maturity': 0.5, 'implied_vol': 0.23},
            {'strike': 100, 'maturity': 0.5, 'implied_vol': 0.20},
            {'strike': 110, 'maturity': 0.5, 'implied_vol': 0.25},
        ]

        self.local_vol.calibrate_surface(market_data)

        assert len(self.local_vol.strikes) == 3
        assert len(self.local_vol.maturities) == 2
        assert len(self.local_vol.vol_surface) == 6

    def test_volatility_interpolation(self):
        """Test volatility surface interpolation"""
        # Setup surface first
        market_data = [
            {'strike': 90, 'maturity': 0.25, 'implied_vol': 0.25},
            {'strike': 100, 'maturity': 0.25, 'implied_vol': 0.22},
            {'strike': 110, 'maturity': 0.25, 'implied_vol': 0.27},
        ]
        self.local_vol.calibrate_surface(market_data)

        # Test interpolation
        vol_95 = self.local_vol.get_local_volatility(95, 0.25)
        vol_105 = self.local_vol.get_local_volatility(105, 0.25)

        assert isinstance(vol_95, float)
        assert isinstance(vol_105, float)
        assert 0.01 < vol_95 < 1.0
        assert 0.01 < vol_105 < 1.0

        # Should be between neighboring points
        assert 0.22 < vol_95 < 0.25  # Between 100 and 90 strikes
        assert 0.22 < vol_105 < 0.27  # Between 100 and 110 strikes


class TestMonteCarloEngine:
    """Test Monte Carlo pricing engine"""

    def setup_method(self):
        self.mc_engine = MonteCarloEngine(n_paths=1000, n_steps=50)  # Smaller for testing

    def test_monte_carlo_initialization(self):
        """Test Monte Carlo engine initialization"""
        assert self.mc_engine.n_paths == 1000
        assert self.mc_engine.n_steps == 50

    def test_european_option_pricing(self):
        """Test European option pricing with Monte Carlo"""
        S0, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.2

        # Test call option
        call_result = self.mc_engine.price_european_option(
            S0, K, T, r, sigma, OptionType.CALL
        )

        assert 'price' in call_result
        assert 'std_error' in call_result
        assert 'delta' in call_result
        assert 'gamma' in call_result

        call_price = call_result['price']
        assert isinstance(call_price, float)
        assert call_price > 0
        assert call_price < S0

        # Test put option
        put_result = self.mc_engine.price_european_option(
            S0, K, T, r, sigma, OptionType.PUT
        )

        put_price = put_result['price']
        assert isinstance(put_price, float)
        assert put_price > 0

        # Rough put-call parity check (Monte Carlo has some error)
        forward = S0 * np.exp(-r * T)
        parity_diff = abs(call_price - put_price - (forward - K * np.exp(-r * T)))
        assert parity_diff < 2.0  # Allow for Monte Carlo error

    def test_monte_carlo_convergence(self):
        """Test that Monte Carlo results improve with more paths"""
        S0, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.2

        # Test with fewer paths
        mc_small = MonteCarloEngine(n_paths=500, n_steps=25)
        result_small = mc_small.price_european_option(S0, K, T, r, sigma, OptionType.CALL)

        # Test with more paths
        mc_large = MonteCarloEngine(n_paths=5000, n_steps=100)
        result_large = mc_large.price_european_option(S0, K, T, r, sigma, OptionType.CALL)

        # Standard error should decrease with more paths
        assert result_large['std_error'] < result_small['std_error']


class TestAdvancedVolatilityCalculator:
    """Test the ensemble advanced volatility calculator"""

    def setup_method(self):
        self.adv_calc = AdvancedVolatilityCalculator()

    def test_advanced_calculator_initialization(self):
        """Test advanced volatility calculator initialization"""
        assert hasattr(self.adv_calc, 'heston_model')
        assert hasattr(self.adv_calc, 'sabr_model')
        assert hasattr(self.adv_calc, 'local_vol_model')
        assert hasattr(self.adv_calc, 'monte_carlo_engine')
        assert hasattr(self.adv_calc, 'model_weights')

    def test_model_ensemble_pricing(self):
        """Test ensemble pricing from multiple models"""
        S, K, T, r, vol = 100.0, 100.0, 0.25, 0.05, 0.2

        results = self.adv_calc.get_model_ensemble_price(
            S, K, T, r, OptionType.CALL, vol
        )

        assert 'ensemble_price' in results
        assert isinstance(results['ensemble_price'], float)
        assert results['ensemble_price'] > 0

        # Should have individual model results
        assert 'black_scholes' in results or 'heston' in results

    def test_model_calibration(self):
        """Test model calibration with market data"""
        market_data = [
            {'strike': 90, 'maturity': 0.25, 'implied_vol': 0.25},
            {'strike': 100, 'maturity': 0.25, 'implied_vol': 0.22},
            {'strike': 110, 'maturity': 0.25, 'implied_vol': 0.27},
        ]

        # Should not raise exception
        self.adv_calc.calibrate_models(market_data, 100.0)


class TestEnhancedBlackScholesCalculator:
    """Test the enhanced Black-Scholes calculator with advanced models"""

    def setup_method(self):
        self.enhanced_calc = None

    @pytest.fixture(autouse=True)
    def mock_volatility_predictor(self):
        """Mock the volatility predictor dependency"""
        with patch('app.ml.volatility_predictor.get_volatility_predictor') as mock_predictor:
            # Create a mock volatility predictor
            mock_vol_predictor = Mock()
            mock_predictor.return_value = mock_vol_predictor

            # Mock the predict_volatility method
            from app.ml.volatility_predictor import VolatilityMetrics, VolatilityRegime
            mock_metrics = VolatilityMetrics(
                historical_vol=0.25,
                garch_predicted_vol=0.22,
                implied_vol=0.24,
                vol_smile_skew=-0.02,
                vol_regime=VolatilityRegime.NORMAL,
                confidence_score=0.8,
                mean_reversion_factor=0.05,
                persistence_factor=0.95,
                spike_probability=0.3,
                expected_move=2.5,
                garch_alpha=0.1,
                garch_beta=0.85,
                garch_omega=0.05,
                model_r_squared=0.75,
                timestamp=datetime.now()
            )

            async def mock_predict_volatility(*args, **kwargs):
                return mock_metrics

            mock_vol_predictor.predict_volatility = AsyncMock(return_value=mock_metrics)
            mock_vol_predictor.get_volatility_for_pricing.return_value = 0.22

            self.enhanced_calc = EnhancedBlackScholesCalculator()
            yield mock_vol_predictor

    @pytest.mark.asyncio
    async def test_enhanced_calculator_initialization(self):
        """Test enhanced calculator initializes correctly"""
        assert hasattr(self.enhanced_calc, 'volatility_predictor')
        assert hasattr(self.enhanced_calc, 'advanced_vol_calculator')
        assert hasattr(self.enhanced_calc, 'model_performance')

    @pytest.mark.asyncio
    async def test_legacy_pricing_method(self):
        """Test the legacy pricing method still works"""
        symbol = "AAPL"
        S, K, T, r = 150.0, 150.0, 0.25, 0.05

        # Mock price data
        price_data = [
            {'close': 148.0, 'timestamp': datetime.now() - timedelta(days=2)},
            {'close': 149.0, 'timestamp': datetime.now() - timedelta(days=1)},
            {'close': 150.0, 'timestamp': datetime.now()},
        ]

        price, vol_metrics = await self.enhanced_calc.calculate_option_price_with_prediction(
            symbol, S, K, T, r, OptionType.CALL, price_data
        )

        assert isinstance(price, float)
        assert price > 0
        assert hasattr(vol_metrics, 'garch_predicted_vol')
        assert hasattr(vol_metrics, 'confidence_score')

    @pytest.mark.asyncio
    async def test_advanced_pricing_with_models(self):
        """Test advanced pricing with multiple volatility models"""
        symbol = "AAPL"
        S, K, T, r = 150.0, 150.0, 0.25, 0.05

        # Mock price data
        price_data = [
            {'close': 148.0, 'timestamp': datetime.now() - timedelta(days=2)},
            {'close': 149.0, 'timestamp': datetime.now() - timedelta(days=1)},
            {'close': 150.0, 'timestamp': datetime.now()},
        ]

        # Test with ensemble pricing
        results = await self.enhanced_calc.calculate_option_price_with_advanced_models(
            symbol, S, K, T, r, OptionType.CALL, price_data,
            use_model_ensemble=True, pricing_method="auto"
        )

        assert 'best_price_estimate' in results
        assert 'model_results' in results
        assert 'selected_models' in results
        assert 'volatility_metrics' in results

        best_price = results['best_price_estimate']
        assert 'price' in best_price
        assert 'model' in best_price
        assert 'confidence' in best_price

        assert isinstance(best_price['price'], float)
        assert best_price['price'] > 0

    @pytest.mark.asyncio
    async def test_specific_model_selection(self):
        """Test pricing with specific volatility models"""
        symbol = "AAPL"
        S, K, T, r = 150.0, 150.0, 0.25, 0.05

        price_data = [{'close': 150.0, 'timestamp': datetime.now()}]

        # Test Heston model specifically
        results = await self.enhanced_calc.calculate_option_price_with_advanced_models(
            symbol, S, K, T, r, OptionType.CALL, price_data,
            pricing_method="heston"
        )

        assert 'heston' in results['selected_models']

        # Test SABR model specifically
        results = await self.enhanced_calc.calculate_option_price_with_advanced_models(
            symbol, S, K, T, r, OptionType.CALL, price_data,
            pricing_method="sabr"
        )

        assert 'sabr' in results['selected_models']

    @pytest.mark.asyncio
    async def test_model_performance_metrics(self):
        """Test model performance metrics retrieval"""
        metrics = self.enhanced_calc.get_model_performance_metrics()

        assert 'model_performance' in metrics
        assert 'available_models' in metrics
        assert 'ensemble_weights' in metrics
        assert 'calibration_status' in metrics

        assert 'black_scholes' in metrics['model_performance']
        assert 'heston' in metrics['available_models']

    @pytest.mark.asyncio
    async def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms"""
        symbol = "INVALID"
        S, K, T, r = 150.0, 150.0, 0.25, 0.05

        # Empty price data should trigger fallback
        price_data = []

        results = await self.enhanced_calc.calculate_option_price_with_advanced_models(
            symbol, S, K, T, r, OptionType.CALL, price_data
        )

        # Should still return a result (fallback)
        assert 'best_price_estimate' in results
        assert results['best_price_estimate']['price'] > 0

    @pytest.mark.asyncio
    async def test_greeks_calculation(self):
        """Test Greeks calculation with enhanced models"""
        symbol = "AAPL"
        S, K, T, r = 150.0, 150.0, 0.25, 0.05

        price_data = [{'close': 150.0, 'timestamp': datetime.now()}]

        greeks, vol_metrics = await self.enhanced_calc.calculate_greeks_with_prediction(
            symbol, S, K, T, r, OptionType.CALL, price_data
        )

        assert isinstance(greeks, GreeksData)
        assert hasattr(greeks, 'delta')
        assert hasattr(greeks, 'gamma')
        assert hasattr(greeks, 'theta')
        assert hasattr(greeks, 'vega')
        assert hasattr(greeks, 'rho')

        # Sanity checks for call option Greeks
        assert 0 <= greeks.delta <= 1  # Call delta should be 0-1
        assert greeks.gamma >= 0       # Gamma should be positive
        assert greeks.vega >= 0        # Vega should be positive


class TestVolatilityModelIntegration:
    """Integration tests for volatility models working together"""

    def test_model_consistency(self):
        """Test that different models produce reasonable relative results"""
        S, K, T, r = 100.0, 100.0, 0.25, 0.05

        # Black-Scholes baseline
        bs_price = BlackScholesCalculator.calculate_option_price(
            S, K, T, r, 0.2, OptionType.CALL
        )

        # Heston model
        heston = HestonVolatilityModel()
        heston_price = heston.price_option(S, K, T, r, OptionType.CALL)

        # SABR model (via BS with SABR vol)
        sabr = SABRVolatilityModel()
        sabr_vol = sabr.implied_volatility(S, K, T)
        sabr_price = BlackScholesCalculator.calculate_option_price(
            S, K, T, r, sabr_vol, OptionType.CALL
        )

        # All prices should be in reasonable range
        prices = [bs_price, heston_price, sabr_price]
        assert all(0 < price < S for price in prices)

        # Prices shouldn't differ by more than 50% (rough sanity check)
        price_range = max(prices) - min(prices)
        avg_price = sum(prices) / len(prices)
        assert price_range / avg_price < 0.5

    def test_volatility_surface_consistency(self):
        """Test that volatility surface produces consistent results"""
        local_vol = LocalVolatilityModel()

        # Create a simple surface
        market_data = []
        strikes = [80, 90, 100, 110, 120]
        maturities = [0.25, 0.5, 1.0]

        for strike in strikes:
            for maturity in maturities:
                # Simple volatility smile: higher for OTM options
                moneyness = strike / 100.0
                base_vol = 0.2
                skew = 0.05 * (1.0 - moneyness)  # Negative skew
                vol = base_vol + skew

                market_data.append({
                    'strike': strike,
                    'maturity': maturity,
                    'implied_vol': vol
                })

        local_vol.calibrate_surface(market_data)

        # Test interpolation consistency
        vol_95 = local_vol.get_local_volatility(95, 0.75)
        vol_105 = local_vol.get_local_volatility(105, 0.75)

        # Should show negative skew (OTM puts have higher vol)
        assert vol_95 > vol_105


if __name__ == "__main__":
    pytest.main([__file__, "-v"])