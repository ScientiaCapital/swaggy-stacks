"""
Implied Volatility Calculator and Volatility Surface Management

This module provides advanced implied volatility calculations using numerical methods,
volatility smile/skew modeling, and volatility surface interpolation for accurate
options pricing across different strikes and expirations.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import structlog
from scipy.optimize import brentq, newton
from scipy.interpolate import griddata, interp2d
from scipy.stats import norm

from app.core.cache import get_market_cache
from app.trading.options_trading import BlackScholesCalculator, OptionType

logger = structlog.get_logger(__name__)


@dataclass
class VolatilityPoint:
    """Single point on volatility surface"""

    strike: float
    time_to_expiry: float  # In years
    implied_vol: float
    option_type: OptionType
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid_price: Optional[float] = None
    delta: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None


@dataclass
class VolatilitySmile:
    """Volatility smile for a single expiration"""

    expiration_date: datetime
    time_to_expiry: float
    underlying_price: float
    points: List[VolatilityPoint]

    # Smile characteristics
    atm_vol: Optional[float] = None
    skew: Optional[float] = None  # 25-delta call - 25-delta put vol
    convexity: Optional[float] = None  # Butterfly spread vol
    min_vol: Optional[float] = None
    max_vol: Optional[float] = None


@dataclass
class VolatilitySurface:
    """Complete volatility surface across strikes and expirations"""

    underlying_symbol: str
    underlying_price: float
    smiles: List[VolatilitySmile]

    # Surface metadata
    min_strike: float
    max_strike: float
    min_expiry: float
    max_expiry: float
    risk_free_rate: float
    dividend_yield: float = 0.0

    # Grid data for interpolation
    strike_grid: Optional[np.ndarray] = None
    expiry_grid: Optional[np.ndarray] = None
    vol_grid: Optional[np.ndarray] = None


class ImpliedVolatilityCalculator:
    """
    Advanced implied volatility calculator using multiple numerical methods

    Supports Brent's method, Newton-Raphson, and bisection methods for
    robust IV calculation across different market conditions.
    """

    def __init__(self):
        self.market_cache = get_market_cache()
        self.max_iterations = 100
        self.tolerance = 1e-8
        self.min_vol = 0.001  # 0.1% minimum volatility
        self.max_vol = 10.0   # 1000% maximum volatility

    def calculate_implied_volatility(
        self,
        option_price: float,
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: OptionType,
        dividend_yield: float = 0.0,
        initial_guess: float = 0.25,
        method: str = "brent"
    ) -> Optional[float]:
        """
        Calculate implied volatility using specified numerical method

        Args:
            option_price: Market price of the option
            underlying_price: Current price of underlying asset
            strike_price: Strike price of the option
            time_to_expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            option_type: Call or Put option
            dividend_yield: Dividend yield of underlying
            initial_guess: Initial volatility guess
            method: Numerical method ('brent', 'newton', 'bisection')

        Returns:
            Implied volatility if successful, None otherwise
        """
        try:
            # Validate inputs
            if option_price <= 0 or time_to_expiry <= 0:
                return None

            # Check intrinsic value bounds
            if option_type == OptionType.CALL:
                intrinsic = max(0, underlying_price - strike_price)
            else:
                intrinsic = max(0, strike_price - underlying_price)

            if option_price < intrinsic:
                logger.warning("Option price below intrinsic value",
                             option_price=option_price,
                             intrinsic=intrinsic)
                return None

            # Define objective function
            def objective_function(vol: float) -> float:
                if vol <= 0:
                    return float('inf')
                try:
                    theoretical_price = BlackScholesCalculator.calculate_option_price(
                        underlying_price=underlying_price,
                        strike_price=strike_price,
                        time_to_expiry=time_to_expiry,
                        risk_free_rate=risk_free_rate,
                        volatility=vol,
                        option_type=option_type,
                        dividend_yield=dividend_yield
                    )
                    return theoretical_price - option_price
                except Exception:
                    return float('inf')

            # Choose numerical method
            if method == "brent":
                result = self._solve_brent(objective_function)
            elif method == "newton":
                result = self._solve_newton(
                    objective_function, initial_guess,
                    underlying_price, strike_price, time_to_expiry,
                    risk_free_rate, option_type, dividend_yield
                )
            elif method == "bisection":
                result = self._solve_bisection(objective_function)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Validate result
            if result and self.min_vol <= result <= self.max_vol:
                return result
            else:
                return None

        except Exception as e:
            logger.error("Failed to calculate implied volatility",
                        error=str(e),
                        option_price=option_price,
                        strike=strike_price,
                        method=method)
            return None

    def _solve_brent(self, objective_function) -> Optional[float]:
        """Solve using Brent's method (most robust)"""
        try:
            # Find bracketing interval
            low_vol = self.min_vol
            high_vol = self.max_vol

            # Ensure we have opposite signs
            f_low = objective_function(low_vol)
            f_high = objective_function(high_vol)

            if f_low * f_high > 0:
                # Try to find better bounds
                if f_low > 0:  # Price too high, need lower vol
                    high_vol = low_vol
                    low_vol = self.min_vol / 10
                else:  # Price too low, need higher vol
                    low_vol = high_vol
                    high_vol = self.max_vol * 2

                f_low = objective_function(low_vol)
                f_high = objective_function(high_vol)

                if f_low * f_high > 0:
                    return None

            return brentq(objective_function, low_vol, high_vol,
                         xtol=self.tolerance, maxiter=self.max_iterations)

        except Exception:
            return None

    def _solve_newton(
        self, objective_function, initial_guess: float,
        underlying_price: float, strike_price: float,
        time_to_expiry: float, risk_free_rate: float,
        option_type: OptionType, dividend_yield: float
    ) -> Optional[float]:
        """Solve using Newton-Raphson method"""
        try:
            def vega_function(vol: float) -> float:
                """Calculate vega for Newton's method"""
                if vol <= 0:
                    return 0
                try:
                    greeks = BlackScholesCalculator.calculate_greeks(
                        underlying_price=underlying_price,
                        strike_price=strike_price,
                        time_to_expiry=time_to_expiry,
                        risk_free_rate=risk_free_rate,
                        volatility=vol,
                        option_type=option_type,
                        dividend_yield=dividend_yield
                    )
                    return greeks.vega * 100  # Convert to per 1% vol change
                except Exception:
                    return 0

            current_vol = initial_guess

            for _ in range(self.max_iterations):
                f_val = objective_function(current_vol)
                f_prime = vega_function(current_vol)

                if abs(f_val) < self.tolerance:
                    break

                if f_prime == 0:
                    return None

                current_vol = current_vol - f_val / f_prime

                if current_vol <= 0:
                    current_vol = initial_guess / 2

            return current_vol if self.min_vol <= current_vol <= self.max_vol else None

        except Exception:
            return None

    def _solve_bisection(self, objective_function) -> Optional[float]:
        """Solve using bisection method (slowest but most stable)"""
        try:
            low = self.min_vol
            high = self.max_vol

            for _ in range(self.max_iterations):
                mid = (low + high) / 2
                f_mid = objective_function(mid)

                if abs(f_mid) < self.tolerance:
                    return mid

                if objective_function(low) * f_mid < 0:
                    high = mid
                else:
                    low = mid

                if abs(high - low) < self.tolerance:
                    return (low + high) / 2

            return None

        except Exception:
            return None


class VolatilitySurfaceManager:
    """
    Volatility surface construction and interpolation manager

    Builds complete volatility surfaces from market data and provides
    interpolation for any strike/expiration combination.
    """

    def __init__(self):
        self.iv_calculator = ImpliedVolatilityCalculator()
        self.market_cache = get_market_cache()

    async def build_volatility_surface(
        self,
        symbol: str,
        option_chain_data: List[Dict[str, Any]],
        underlying_price: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0
    ) -> Optional[VolatilitySurface]:
        """
        Build complete volatility surface from option chain data

        Args:
            symbol: Underlying symbol
            option_chain_data: List of option contracts with pricing data
            underlying_price: Current underlying price
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield

        Returns:
            VolatilitySurface object or None if insufficient data
        """
        try:
            if not option_chain_data:
                return None

            # Group by expiration
            expirations = {}
            for option_data in option_chain_data:
                exp_date = option_data.get('expiration_date')
                if exp_date not in expirations:
                    expirations[exp_date] = []
                expirations[exp_date].append(option_data)

            smiles = []
            all_strikes = []
            all_expiries = []

            for exp_date_str, options in expirations.items():
                try:
                    exp_date = datetime.fromisoformat(exp_date_str)
                    time_to_expiry = max(0.001, (exp_date - datetime.now()).total_seconds() / (365.25 * 24 * 3600))

                    # Calculate IV for each option
                    vol_points = []
                    for option in options:
                        iv = await self._calculate_option_iv(
                            option, underlying_price, time_to_expiry,
                            risk_free_rate, dividend_yield
                        )
                        if iv:
                            vol_points.append(iv)
                            all_strikes.append(iv.strike)
                            all_expiries.append(time_to_expiry)

                    if len(vol_points) >= 3:  # Minimum points for smile
                        smile = VolatilitySmile(
                            expiration_date=exp_date,
                            time_to_expiry=time_to_expiry,
                            underlying_price=underlying_price,
                            points=vol_points
                        )

                        # Calculate smile characteristics
                        self._analyze_smile_characteristics(smile)
                        smiles.append(smile)

                except Exception as e:
                    logger.warning(f"Failed to process expiration {exp_date_str}",
                                 error=str(e))
                    continue

            if not smiles:
                return None

            # Create volatility surface
            surface = VolatilitySurface(
                underlying_symbol=symbol,
                underlying_price=underlying_price,
                smiles=smiles,
                min_strike=min(all_strikes),
                max_strike=max(all_strikes),
                min_expiry=min(all_expiries),
                max_expiry=max(all_expiries),
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield
            )

            # Build interpolation grid
            await self._build_interpolation_grid(surface)

            # Cache the surface
            cache_key = f"vol_surface_{symbol}"
            await self.market_cache.set(cache_key, surface, ttl_override=1800)  # 30 minutes

            logger.info("Volatility surface built successfully",
                       symbol=symbol,
                       expirations=len(smiles),
                       total_points=len(all_strikes))

            return surface

        except Exception as e:
            logger.error("Failed to build volatility surface",
                        symbol=symbol, error=str(e))
            return None

    async def _calculate_option_iv(
        self,
        option_data: Dict[str, Any],
        underlying_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        dividend_yield: float
    ) -> Optional[VolatilityPoint]:
        """Calculate IV for a single option"""
        try:
            strike = float(option_data.get('strike_price', 0))
            bid = option_data.get('bid')
            ask = option_data.get('ask')

            if not bid or not ask or bid <= 0 or ask <= 0:
                return None

            mid_price = (bid + ask) / 2
            option_type_str = option_data.get('type', 'call').lower()
            option_type = OptionType.CALL if option_type_str == 'call' else OptionType.PUT

            # Calculate implied volatility
            iv = self.iv_calculator.calculate_implied_volatility(
                option_price=mid_price,
                underlying_price=underlying_price,
                strike_price=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                option_type=option_type,
                dividend_yield=dividend_yield
            )

            if iv is None:
                return None

            # Calculate delta for moneyness
            greeks = BlackScholesCalculator.calculate_greeks(
                underlying_price=underlying_price,
                strike_price=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=iv,
                option_type=option_type,
                dividend_yield=dividend_yield
            )

            return VolatilityPoint(
                strike=strike,
                time_to_expiry=time_to_expiry,
                implied_vol=iv,
                option_type=option_type,
                bid=bid,
                ask=ask,
                mid_price=mid_price,
                delta=greeks.delta,
                volume=option_data.get('volume', 0),
                open_interest=option_data.get('open_interest', 0)
            )

        except Exception as e:
            logger.warning("Failed to calculate option IV", error=str(e))
            return None

    def _analyze_smile_characteristics(self, smile: VolatilitySmile):
        """Analyze volatility smile characteristics"""
        try:
            if len(smile.points) < 3:
                return

            # Sort by strike
            points = sorted(smile.points, key=lambda p: p.strike)
            vols = [p.implied_vol for p in points]
            strikes = [p.strike / smile.underlying_price for p in points]  # Normalized moneyness

            # Find ATM volatility (closest to moneyness = 1.0)
            atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - 1.0))
            smile.atm_vol = vols[atm_idx]

            # Calculate skew (slope at ATM)
            if len(points) >= 5:
                # Find 25-delta equivalent points (approximate)
                call_25d_idx = min(range(len(points)),
                                 key=lambda i: abs(points[i].delta - 0.25) if points[i].option_type == OptionType.CALL else float('inf'))
                put_25d_idx = min(range(len(points)),
                                key=lambda i: abs(points[i].delta - (-0.25)) if points[i].option_type == OptionType.PUT else float('inf'))

                if call_25d_idx < len(points) and put_25d_idx < len(points):
                    smile.skew = points[call_25d_idx].implied_vol - points[put_25d_idx].implied_vol

            # Basic statistics
            smile.min_vol = min(vols)
            smile.max_vol = max(vols)

            # Simple convexity measure (butterfly)
            if len(vols) >= 3:
                smile.convexity = vols[0] + vols[-1] - 2 * smile.atm_vol

        except Exception as e:
            logger.warning("Failed to analyze smile characteristics", error=str(e))

    async def _build_interpolation_grid(self, surface: VolatilitySurface):
        """Build interpolation grid for the volatility surface"""
        try:
            strikes = []
            expiries = []
            vols = []

            for smile in surface.smiles:
                for point in smile.points:
                    strikes.append(point.strike)
                    expiries.append(point.time_to_expiry)
                    vols.append(point.implied_vol)

            if len(strikes) < 4:  # Need minimum points for interpolation
                return

            # Create regular grids
            strike_range = np.linspace(surface.min_strike * 0.8,
                                     surface.max_strike * 1.2, 50)
            expiry_range = np.linspace(surface.min_expiry,
                                     surface.max_expiry, 20)

            strike_grid, expiry_grid = np.meshgrid(strike_range, expiry_range)

            # Interpolate volatility surface
            points = np.column_stack((strikes, expiries))
            vol_grid = griddata(points, vols, (strike_grid, expiry_grid),
                              method='cubic', fill_value=np.nan)

            # Store grids
            surface.strike_grid = strike_grid
            surface.expiry_grid = expiry_grid
            surface.vol_grid = vol_grid

        except Exception as e:
            logger.error("Failed to build interpolation grid", error=str(e))

    def interpolate_volatility(
        self,
        surface: VolatilitySurface,
        strike: float,
        time_to_expiry: float
    ) -> Optional[float]:
        """
        Interpolate volatility for given strike and time to expiry

        Args:
            surface: Volatility surface
            strike: Strike price
            time_to_expiry: Time to expiration in years

        Returns:
            Interpolated implied volatility or None
        """
        try:
            if not surface.vol_grid is not None:
                return None

            # Check bounds
            if (strike < surface.min_strike * 0.8 or
                strike > surface.max_strike * 1.2 or
                time_to_expiry < surface.min_expiry or
                time_to_expiry > surface.max_expiry):
                return None

            # Use scipy interpolation
            interp_func = interp2d(
                surface.strike_grid[0, :],
                surface.expiry_grid[:, 0],
                surface.vol_grid,
                kind='cubic',
                fill_value=np.nan
            )

            vol = interp_func(strike, time_to_expiry)[0]

            return vol if not np.isnan(vol) else None

        except Exception as e:
            logger.error("Failed to interpolate volatility",
                        strike=strike, time_to_expiry=time_to_expiry,
                        error=str(e))
            return None


# Global instances
_iv_calculator: Optional[ImpliedVolatilityCalculator] = None
_surface_manager: Optional[VolatilitySurfaceManager] = None


def get_iv_calculator() -> ImpliedVolatilityCalculator:
    """Get or create global IV calculator instance"""
    global _iv_calculator
    if _iv_calculator is None:
        _iv_calculator = ImpliedVolatilityCalculator()
    return _iv_calculator


def get_surface_manager() -> VolatilitySurfaceManager:
    """Get or create global surface manager instance"""
    global _surface_manager
    if _surface_manager is None:
        _surface_manager = VolatilitySurfaceManager()
    return _surface_manager


__all__ = [
    "VolatilityPoint",
    "VolatilitySmile",
    "VolatilitySurface",
    "ImpliedVolatilityCalculator",
    "VolatilitySurfaceManager",
    "get_iv_calculator",
    "get_surface_manager"
]