"""
GARCH-based volatility prediction model for options pricing

This module implements a comprehensive volatility prediction system using:
- GARCH(1,1) models for volatility forecasting
- Historical volatility calculation with multiple windows
- Implied vs realized volatility analysis
- Volatility smile modeling for different strikes
- Event-driven volatility spike detection
- Integration with existing options pricing system
"""

import asyncio
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import structlog

logger = structlog.get_logger(__name__)


class VolatilityRegime(Enum):
    """Market volatility regime classifications"""
    LOW = "low"           # VIX < 20, low realized volatility
    NORMAL = "normal"     # VIX 20-30, normal market conditions
    HIGH = "high"         # VIX 30-40, elevated stress
    EXTREME = "extreme"   # VIX > 40, crisis conditions


@dataclass
class VolatilityMetrics:
    """Comprehensive volatility analysis results"""
    historical_vol: float              # Annualized historical volatility
    garch_predicted_vol: float         # GARCH(1,1) forecast
    implied_vol: Optional[float]       # Market implied volatility
    vol_smile_skew: float              # Volatility smile skewness
    vol_regime: VolatilityRegime       # Current volatility regime
    confidence_score: float            # Prediction confidence [0-1]

    # Time series components
    mean_reversion_factor: float       # Speed of volatility mean reversion
    persistence_factor: float          # Volatility clustering persistence

    # Event analysis
    spike_probability: float           # Probability of volatility spike
    expected_move: float               # Expected price move based on volatility

    # Model diagnostics
    garch_alpha: float                 # GARCH alpha parameter
    garch_beta: float                  # GARCH beta parameter
    garch_omega: float                 # GARCH omega parameter
    model_r_squared: float             # Model fit quality

    timestamp: datetime


@dataclass
class VolatilitySmilePoint:
    """Single point on volatility smile curve"""
    strike: float
    moneyness: float                   # Strike/Spot ratio
    implied_vol: float
    delta: float
    volume: int
    open_interest: int


@dataclass
class VolatilitySmile:
    """Complete volatility smile analysis"""
    symbol: str
    expiration: str
    spot_price: float
    smile_points: List[VolatilitySmilePoint]
    atm_vol: float                     # At-the-money volatility
    skew: float                        # Put-call skew (90% put - 110% call)
    convexity: float                   # Smile convexity
    term_structure_slope: float        # Volatility term structure slope
    timestamp: datetime


class GARCHModel:
    """GARCH(1,1) volatility model implementation"""

    def __init__(self, returns: np.ndarray):
        self.returns = returns
        self.omega: float = 0.0
        self.alpha: float = 0.0
        self.beta: float = 0.0
        self.variance_forecast: float = 0.0
        self.log_likelihood: float = 0.0
        self.fitted: bool = False

    def fit(self) -> bool:
        """Fit GARCH(1,1) model to return series"""
        try:
            # Initial parameter estimates
            initial_variance = np.var(self.returns)

            def garch_likelihood(params):
                omega, alpha, beta = params

                # Parameter constraints
                if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                    return 1e10

                n = len(self.returns)
                variance = np.zeros(n)
                variance[0] = initial_variance

                # Calculate conditional variances
                for t in range(1, n):
                    variance[t] = omega + alpha * (self.returns[t-1]**2) + beta * variance[t-1]

                # Avoid numerical issues
                variance = np.maximum(variance, 1e-8)

                # Log-likelihood
                log_likelihood = -0.5 * np.sum(
                    np.log(2 * np.pi * variance) + (self.returns**2) / variance
                )

                return -log_likelihood  # Minimize negative log-likelihood

            # Optimize parameters
            from scipy.optimize import minimize

            # Initial guess
            x0 = [np.var(self.returns) * 0.01, 0.1, 0.85]

            # Bounds: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
            bounds = [(1e-8, 1.0), (0, 0.5), (0, 0.99)]

            # Constraint: alpha + beta < 1
            constraints = {'type': 'ineq', 'fun': lambda x: 0.999 - x[1] - x[2]}

            result = minimize(
                garch_likelihood,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                self.omega, self.alpha, self.beta = result.x
                self.log_likelihood = -result.fun

                # Calculate final variance forecast
                n = len(self.returns)
                variance = np.zeros(n)
                variance[0] = initial_variance

                for t in range(1, n):
                    variance[t] = self.omega + self.alpha * (self.returns[t-1]**2) + self.beta * variance[t-1]

                # Forecast next period variance
                self.variance_forecast = self.omega + self.alpha * (self.returns[-1]**2) + self.beta * variance[-1]
                self.fitted = True

                logger.debug(
                    "GARCH model fitted successfully",
                    omega=self.omega,
                    alpha=self.alpha,
                    beta=self.beta,
                    persistence=self.alpha + self.beta
                )

                return True

        except Exception as e:
            logger.error("GARCH model fitting failed", error=str(e))

        return False

    def forecast_volatility(self, horizon: int = 1) -> float:
        """Forecast volatility for given horizon (in days)"""
        if not self.fitted:
            return 0.0

        # Multi-step forecast using GARCH recursion
        unconditional_variance = self.omega / (1 - self.alpha - self.beta)

        if horizon == 1:
            return math.sqrt(self.variance_forecast * 252)  # Annualized

        # For multi-step forecast
        persistence = self.alpha + self.beta
        forecast_variance = (
            unconditional_variance +
            (persistence ** (horizon - 1)) *
            (self.variance_forecast - unconditional_variance)
        )

        return math.sqrt(forecast_variance * 252)  # Annualized


class VolatilityPredictor:
    """Main volatility prediction system"""

    def __init__(self):
        self.cache: Dict[str, Tuple[VolatilityMetrics, float]] = {}
        self.cache_ttl = 300  # 5 minute cache TTL
        self.smile_cache: Dict[str, Tuple[VolatilitySmile, float]] = {}

        # Model parameters
        self.min_data_points = 30
        self.vol_windows = [10, 20, 30, 60]  # Historical volatility windows

        # Volatility regime thresholds
        self.regime_thresholds = {
            VolatilityRegime.LOW: 0.20,
            VolatilityRegime.NORMAL: 0.30,
            VolatilityRegime.HIGH: 0.40
        }

        logger.info("VolatilityPredictor initialized")

    async def predict_volatility(
        self,
        symbol: str,
        price_data: List[Dict[str, Any]],
        option_chain: Optional[List[Dict[str, Any]]] = None,
        expiration: Optional[str] = None
    ) -> VolatilityMetrics:
        """Generate comprehensive volatility prediction"""

        # Check cache first
        cache_key = f"{symbol}_{expiration or 'default'}"
        if cache_key in self.cache:
            cached_metrics, cache_time = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                logger.debug("Returning cached volatility metrics", symbol=symbol)
                return cached_metrics

        try:
            # Calculate price returns
            returns = self._calculate_returns(price_data)

            if len(returns) < self.min_data_points:
                logger.warning(
                    "Insufficient data for volatility prediction",
                    symbol=symbol,
                    data_points=len(returns)
                )
                return self._create_fallback_metrics(symbol)

            # Historical volatility analysis
            historical_vol = self._calculate_historical_volatility(returns)

            # GARCH model prediction
            garch_model = GARCHModel(returns)
            garch_success = garch_model.fit()

            if garch_success:
                garch_predicted_vol = garch_model.forecast_volatility()
                garch_alpha = garch_model.alpha
                garch_beta = garch_model.beta
                garch_omega = garch_model.omega
                model_r_squared = self._calculate_garch_r_squared(garch_model, returns)
            else:
                # Fallback to historical volatility
                garch_predicted_vol = historical_vol
                garch_alpha = 0.1
                garch_beta = 0.85
                garch_omega = 0.05
                model_r_squared = 0.0

            # Volatility regime analysis
            vol_regime = self._classify_volatility_regime(garch_predicted_vol)

            # Calculate volatility smile analysis if option chain provided
            implied_vol = None
            vol_smile_skew = 0.0

            if option_chain and expiration:
                smile_analysis = await self._analyze_volatility_smile(
                    symbol, option_chain, expiration, price_data[-1]['close']
                )
                if smile_analysis:
                    implied_vol = smile_analysis.atm_vol
                    vol_smile_skew = smile_analysis.skew

            # Event-driven analysis
            spike_probability = self._calculate_spike_probability(returns, garch_predicted_vol)

            # Expected move calculation
            current_price = price_data[-1]['close']
            expected_move = self._calculate_expected_move(current_price, garch_predicted_vol)

            # Model diagnostics
            confidence_score = self._calculate_confidence_score(
                garch_success, model_r_squared, len(returns)
            )

            # Mean reversion and persistence factors
            mean_reversion_factor = 1 - (garch_alpha + garch_beta)
            persistence_factor = garch_alpha + garch_beta

            # Create comprehensive metrics
            metrics = VolatilityMetrics(
                historical_vol=historical_vol,
                garch_predicted_vol=garch_predicted_vol,
                implied_vol=implied_vol,
                vol_smile_skew=vol_smile_skew,
                vol_regime=vol_regime,
                confidence_score=confidence_score,
                mean_reversion_factor=mean_reversion_factor,
                persistence_factor=persistence_factor,
                spike_probability=spike_probability,
                expected_move=expected_move,
                garch_alpha=garch_alpha,
                garch_beta=garch_beta,
                garch_omega=garch_omega,
                model_r_squared=model_r_squared,
                timestamp=datetime.now()
            )

            # Cache the results
            self.cache[cache_key] = (metrics, time.time())

            logger.info(
                "Volatility prediction completed",
                symbol=symbol,
                historical_vol=historical_vol,
                garch_vol=garch_predicted_vol,
                regime=vol_regime.value,
                confidence=confidence_score
            )

            return metrics

        except Exception as e:
            logger.error("Volatility prediction failed", symbol=symbol, error=str(e))
            return self._create_fallback_metrics(symbol)

    def _calculate_returns(self, price_data: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate log returns from price data"""
        prices = np.array([float(candle['close']) for candle in price_data])
        returns = np.diff(np.log(prices))
        return returns

    def _calculate_historical_volatility(
        self,
        returns: np.ndarray,
        window: int = 20
    ) -> float:
        """Calculate annualized historical volatility"""
        if len(returns) < window:
            window = len(returns)

        # Use rolling window approach for more recent data emphasis
        recent_returns = returns[-window:]
        volatility = np.std(recent_returns) * math.sqrt(252)
        return volatility

    def _classify_volatility_regime(self, volatility: float) -> VolatilityRegime:
        """Classify current volatility regime"""
        if volatility < self.regime_thresholds[VolatilityRegime.LOW]:
            return VolatilityRegime.LOW
        elif volatility < self.regime_thresholds[VolatilityRegime.NORMAL]:
            return VolatilityRegime.NORMAL
        elif volatility < self.regime_thresholds[VolatilityRegime.HIGH]:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME

    def _calculate_spike_probability(self, returns: np.ndarray, current_vol: float) -> float:
        """Calculate probability of volatility spike based on recent patterns"""
        try:
            # Calculate rolling volatility
            window = 10
            rolling_vols = []

            for i in range(window, len(returns)):
                period_returns = returns[i-window:i]
                vol = np.std(period_returns) * math.sqrt(252)
                rolling_vols.append(vol)

            if len(rolling_vols) < 5:
                return 0.5  # Neutral probability

            rolling_vols = np.array(rolling_vols)

            # Calculate spike threshold (95th percentile)
            spike_threshold = np.percentile(rolling_vols, 95)

            # Recent volatility trend
            recent_vol_trend = np.mean(rolling_vols[-5:]) / np.mean(rolling_vols[-10:-5])

            # Base probability from historical spikes
            spike_count = np.sum(rolling_vols > spike_threshold)
            base_probability = spike_count / len(rolling_vols)

            # Adjust for current trend
            trend_adjustment = min(recent_vol_trend, 2.0) - 1.0  # [-1, 1]

            spike_probability = base_probability + 0.2 * trend_adjustment
            spike_probability = max(0.0, min(1.0, spike_probability))

            return spike_probability

        except Exception as e:
            logger.error("Spike probability calculation failed", error=str(e))
            return 0.5

    def _calculate_expected_move(self, price: float, volatility: float, days: int = 1) -> float:
        """Calculate expected price move based on volatility"""
        time_factor = math.sqrt(days / 252)
        expected_move = price * volatility * time_factor
        return expected_move

    def _calculate_confidence_score(
        self,
        garch_success: bool,
        r_squared: float,
        data_points: int
    ) -> float:
        """Calculate prediction confidence score"""
        base_score = 0.5

        if garch_success:
            base_score += 0.3

        # R-squared contribution
        base_score += 0.2 * r_squared

        # Data quantity contribution
        data_factor = min(data_points / 100, 1.0)
        base_score += 0.2 * data_factor

        return min(1.0, base_score)

    def _calculate_garch_r_squared(self, garch_model: GARCHModel, returns: np.ndarray) -> float:
        """Calculate R-squared for GARCH model fit"""
        try:
            # Calculate fitted variances
            n = len(returns)
            fitted_variance = np.zeros(n)
            fitted_variance[0] = np.var(returns)

            for t in range(1, n):
                fitted_variance[t] = (
                    garch_model.omega +
                    garch_model.alpha * (returns[t-1]**2) +
                    garch_model.beta * fitted_variance[t-1]
                )

            # Calculate R-squared
            realized_variance = returns**2
            ss_res = np.sum((realized_variance - fitted_variance)**2)
            ss_tot = np.sum((realized_variance - np.mean(realized_variance))**2)

            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            return max(0.0, min(1.0, r_squared))

        except Exception as e:
            logger.error("R-squared calculation failed", error=str(e))
            return 0.0

    async def _analyze_volatility_smile(
        self,
        symbol: str,
        option_chain: List[Dict[str, Any]],
        expiration: str,
        spot_price: float
    ) -> Optional[VolatilitySmile]:
        """Analyze volatility smile from option chain"""
        try:
            smile_points = []

            for option in option_chain:
                if option.get('expiration_date') != expiration:
                    continue

                strike = float(option['strike_price'])
                implied_vol = option.get('implied_volatility')

                if implied_vol is None or implied_vol <= 0:
                    continue

                moneyness = strike / spot_price
                delta = option.get('delta', 0.0)
                volume = option.get('volume', 0)
                open_interest = option.get('open_interest', 0)

                smile_points.append(VolatilitySmilePoint(
                    strike=strike,
                    moneyness=moneyness,
                    implied_vol=implied_vol,
                    delta=delta,
                    volume=volume,
                    open_interest=open_interest
                ))

            if len(smile_points) < 3:
                return None

            # Sort by moneyness
            smile_points.sort(key=lambda x: x.moneyness)

            # Find ATM volatility
            atm_vol = self._interpolate_atm_volatility(smile_points, spot_price)

            # Calculate skew (90% put vs 110% call implied vol difference)
            skew = self._calculate_volatility_skew(smile_points)

            # Calculate convexity
            convexity = self._calculate_smile_convexity(smile_points)

            smile = VolatilitySmile(
                symbol=symbol,
                expiration=expiration,
                spot_price=spot_price,
                smile_points=smile_points,
                atm_vol=atm_vol,
                skew=skew,
                convexity=convexity,
                term_structure_slope=0.0,  # Would need multiple expirations
                timestamp=datetime.now()
            )

            # Cache the smile
            cache_key = f"{symbol}_{expiration}_smile"
            self.smile_cache[cache_key] = (smile, time.time())

            return smile

        except Exception as e:
            logger.error("Volatility smile analysis failed", symbol=symbol, error=str(e))
            return None

    def _interpolate_atm_volatility(
        self,
        smile_points: List[VolatilitySmilePoint],
        spot_price: float
    ) -> float:
        """Interpolate at-the-money implied volatility"""
        # Find closest strikes to ATM
        closest_points = sorted(smile_points, key=lambda x: abs(x.strike - spot_price))[:3]

        if not closest_points:
            return 0.0

        # Simple linear interpolation for ATM
        if len(closest_points) == 1:
            return closest_points[0].implied_vol

        # Weighted average based on distance to ATM
        total_weight = 0
        weighted_vol = 0

        for point in closest_points:
            distance = abs(point.strike - spot_price)
            weight = 1.0 / (1.0 + distance)
            weighted_vol += point.implied_vol * weight
            total_weight += weight

        return weighted_vol / total_weight if total_weight > 0 else closest_points[0].implied_vol

    def _calculate_volatility_skew(self, smile_points: List[VolatilitySmilePoint]) -> float:
        """Calculate volatility skew (put-call vol difference)"""
        try:
            # Find 90% and 110% moneyness points
            put_vol = None
            call_vol = None

            for point in smile_points:
                if 0.88 <= point.moneyness <= 0.92:  # ~90% moneyness
                    put_vol = point.implied_vol
                elif 1.08 <= point.moneyness <= 1.12:  # ~110% moneyness
                    call_vol = point.implied_vol

            if put_vol and call_vol:
                return put_vol - call_vol

            # Fallback: general downward slope
            if len(smile_points) >= 2:
                return smile_points[0].implied_vol - smile_points[-1].implied_vol

            return 0.0

        except Exception as e:
            logger.error("Skew calculation failed", error=str(e))
            return 0.0

    def _calculate_smile_convexity(self, smile_points: List[VolatilitySmilePoint]) -> float:
        """Calculate volatility smile convexity"""
        try:
            if len(smile_points) < 3:
                return 0.0

            # Calculate second derivative approximation
            convexities = []

            for i in range(1, len(smile_points) - 1):
                left = smile_points[i-1]
                center = smile_points[i]
                right = smile_points[i+1]

                # Second derivative approximation
                convexity = (
                    (right.implied_vol - center.implied_vol) / (right.moneyness - center.moneyness) -
                    (center.implied_vol - left.implied_vol) / (center.moneyness - left.moneyness)
                ) / (right.moneyness - left.moneyness)

                convexities.append(convexity)

            return np.mean(convexities) if convexities else 0.0

        except Exception as e:
            logger.error("Convexity calculation failed", error=str(e))
            return 0.0

    def _create_fallback_metrics(self, symbol: str) -> VolatilityMetrics:
        """Create fallback metrics when prediction fails"""
        return VolatilityMetrics(
            historical_vol=0.25,  # Default 25% volatility
            garch_predicted_vol=0.25,
            implied_vol=None,
            vol_smile_skew=0.0,
            vol_regime=VolatilityRegime.NORMAL,
            confidence_score=0.1,  # Low confidence
            mean_reversion_factor=0.05,
            persistence_factor=0.95,
            spike_probability=0.5,
            expected_move=0.0,
            garch_alpha=0.1,
            garch_beta=0.85,
            garch_omega=0.05,
            model_r_squared=0.0,
            timestamp=datetime.now()
        )

    def get_volatility_for_pricing(
        self,
        symbol: str,
        expiration: Optional[str] = None,
        strike: Optional[float] = None
    ) -> float:
        """Get best volatility estimate for options pricing"""
        cache_key = f"{symbol}_{expiration or 'default'}"

        if cache_key in self.cache:
            metrics, cache_time = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                # Prefer implied volatility if available, otherwise use GARCH
                if metrics.implied_vol and metrics.implied_vol > 0:
                    return metrics.implied_vol
                return metrics.garch_predicted_vol

        # Fallback to default volatility
        return 0.25

    def clear_cache(self):
        """Clear all cached volatility data"""
        self.cache.clear()
        self.smile_cache.clear()
        logger.info("Volatility prediction cache cleared")


# Global singleton instance
_volatility_predictor: Optional[VolatilityPredictor] = None


def get_volatility_predictor() -> VolatilityPredictor:
    """Get or create global volatility predictor instance"""
    global _volatility_predictor
    if _volatility_predictor is None:
        _volatility_predictor = VolatilityPredictor()
    return _volatility_predictor


__all__ = [
    "VolatilityPredictor",
    "VolatilityMetrics",
    "VolatilitySmile",
    "VolatilitySmilePoint",
    "VolatilityRegime",
    "GARCHModel",
    "get_volatility_predictor"
]