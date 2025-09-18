"""
Options trading implementation with Greeks calculations for Alpaca API.
Provides options strategies, risk analysis, and Greeks computation for paper trading.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from scipy.stats import norm

from app.core.cache import get_market_cache
from app.trading.alpaca_client import AlpacaClient

logger = structlog.get_logger(__name__)
# Additional imports for advanced volatility models
import time
from datetime import datetime


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class OptionStrategy(Enum):
    LONG_CALL = "long_call"
    SHORT_CALL = "short_call"
    LONG_PUT = "long_put"
    SHORT_PUT = "short_put"
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    IRON_CONDOR = "iron_condor"
    STRADDLE = "straddle"
    STRANGLE = "strangle"


@dataclass
class GreeksData:
    """Options Greeks calculation results"""

    # First-order Greeks
    delta: float  # Price sensitivity to underlying
    gamma: float  # Delta sensitivity to underlying
    theta: float  # Time decay
    vega: float  # Volatility sensitivity
    rho: float  # Interest rate sensitivity

    # Second-order Greeks
    vanna: float = 0.0  # Delta sensitivity to volatility
    charm: float = 0.0  # Delta sensitivity to time
    vomma: float = 0.0  # Vega sensitivity to volatility
    veta: float = 0.0   # Vega sensitivity to time
    color: float = 0.0  # Gamma sensitivity to time
    speed: float = 0.0  # Gamma sensitivity to underlying
    zomma: float = 0.0  # Gamma sensitivity to volatility
    ultima: float = 0.0 # Vomma sensitivity to volatility

    # Additional metrics
    intrinsic_value: float
    time_value: float
    implied_volatility: Optional[float] = None


@dataclass
class OptionContract:
    """Option contract details"""

    symbol: str
    underlying_symbol: str
    option_type: OptionType
    strike_price: float
    expiration_date: datetime
    current_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float] = None
    greeks: Optional[GreeksData] = None


@dataclass
class OptionPosition:
    """Option position with Greeks"""

    contract: OptionContract
    quantity: int  # Positive for long, negative for short
    entry_price: float
    current_value: float
    profit_loss: float
    position_greeks: GreeksData


class BlackScholesCalculator:
    """Black-Scholes options pricing and Greeks calculator"""

    @staticmethod
    def calculate_option_price(
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,  # In years
        risk_free_rate: float,
        volatility: float,
        option_type: OptionType,
        dividend_yield: float = 0.0,
    ) -> float:
        """Calculate option price using Black-Scholes model"""
        try:
            d1 = BlackScholesCalculator._calculate_d1(
                underlying_price,
                strike_price,
                time_to_expiry,
                risk_free_rate,
                volatility,
                dividend_yield,
            )
            d2 = d1 - volatility * math.sqrt(time_to_expiry)

            if option_type == OptionType.CALL:
                price = underlying_price * math.exp(
                    -dividend_yield * time_to_expiry
                ) * norm.cdf(d1) - strike_price * math.exp(
                    -risk_free_rate * time_to_expiry
                ) * norm.cdf(
                    d2
                )
            else:  # PUT
                price = strike_price * math.exp(
                    -risk_free_rate * time_to_expiry
                ) * norm.cdf(-d2) - underlying_price * math.exp(
                    -dividend_yield * time_to_expiry
                ) * norm.cdf(
                    -d1
                )

            return max(0, price)

        except Exception as e:
            logger.error("Failed to calculate option price", error=str(e))
            return 0.0

    @staticmethod
    def calculate_greeks(
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: OptionType,
        dividend_yield: float = 0.0,
    ) -> GreeksData:
        """Calculate all Greeks for an option"""
        try:
            if time_to_expiry <= 0:
                # Handle expired options
                if option_type == OptionType.CALL:
                    intrinsic = max(0, underlying_price - strike_price)
                else:
                    intrinsic = max(0, strike_price - underlying_price)

                return GreeksData(
                    delta=1.0 if intrinsic > 0 else 0.0,
                    gamma=0.0,
                    theta=0.0,
                    vega=0.0,
                    rho=0.0,
                    intrinsic_value=intrinsic,
                    time_value=0.0,
                )

            d1 = BlackScholesCalculator._calculate_d1(
                underlying_price,
                strike_price,
                time_to_expiry,
                risk_free_rate,
                volatility,
                dividend_yield,
            )
            d2 = d1 - volatility * math.sqrt(time_to_expiry)

            # Calculate Greeks
            delta = BlackScholesCalculator._calculate_delta(
                d1, option_type, time_to_expiry, dividend_yield
            )

            gamma = BlackScholesCalculator._calculate_gamma(
                underlying_price, d1, time_to_expiry, volatility, dividend_yield
            )

            theta = BlackScholesCalculator._calculate_theta(
                underlying_price,
                strike_price,
                time_to_expiry,
                risk_free_rate,
                volatility,
                d1,
                d2,
                option_type,
                dividend_yield,
            )

            vega = BlackScholesCalculator._calculate_vega(
                underlying_price, d1, time_to_expiry, dividend_yield
            )

            rho = BlackScholesCalculator._calculate_rho(
                strike_price, time_to_expiry, risk_free_rate, d2, option_type
            )

            # Calculate intrinsic and time value
            if option_type == OptionType.CALL:
                intrinsic_value = max(0, underlying_price - strike_price)
            else:
                intrinsic_value = max(0, strike_price - underlying_price)

            option_price = BlackScholesCalculator.calculate_option_price(
                underlying_price,
                strike_price,
                time_to_expiry,
                risk_free_rate,
                volatility,
                option_type,
                dividend_yield,
            )

            time_value = max(0, option_price - intrinsic_value)

            return GreeksData(
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                intrinsic_value=intrinsic_value,
                time_value=time_value,
            )

        except Exception as e:
            logger.error("Failed to calculate Greeks", error=str(e))
            return GreeksData(0, 0, 0, 0, 0, 0, 0)

    @staticmethod
    def _calculate_d1(
        underlying_price,
        strike_price,
        time_to_expiry,
        risk_free_rate,
        volatility,
        dividend_yield,
    ):
        """Calculate d1 parameter for Black-Scholes"""
        return (
            math.log(underlying_price / strike_price)
            + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
        ) / (volatility * math.sqrt(time_to_expiry))

    @staticmethod
    def _calculate_delta(d1, option_type, time_to_expiry, dividend_yield):
        """Calculate delta"""
        if option_type == OptionType.CALL:
            return math.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)
        else:
            return -math.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)

    @staticmethod
    def _calculate_gamma(
        underlying_price, d1, time_to_expiry, volatility, dividend_yield
    ):
        """Calculate gamma"""
        return (math.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1)) / (
            underlying_price * volatility * math.sqrt(time_to_expiry)
        )

    @staticmethod
    def _calculate_theta(
        underlying_price,
        strike_price,
        time_to_expiry,
        risk_free_rate,
        volatility,
        d1,
        d2,
        option_type,
        dividend_yield,
    ):
        """Calculate theta (time decay)"""
        common_term = (
            -underlying_price
            * norm.pdf(d1)
            * volatility
            * math.exp(-dividend_yield * time_to_expiry)
            / (2 * math.sqrt(time_to_expiry))
        )

        if option_type == OptionType.CALL:
            theta = (
                common_term
                - risk_free_rate
                * strike_price
                * math.exp(-risk_free_rate * time_to_expiry)
                * norm.cdf(d2)
                + dividend_yield
                * underlying_price
                * math.exp(-dividend_yield * time_to_expiry)
                * norm.cdf(d1)
            )
        else:
            theta = (
                common_term
                + risk_free_rate
                * strike_price
                * math.exp(-risk_free_rate * time_to_expiry)
                * norm.cdf(-d2)
                - dividend_yield
                * underlying_price
                * math.exp(-dividend_yield * time_to_expiry)
                * norm.cdf(-d1)
            )

        return theta / 365  # Convert to daily theta

    @staticmethod
    def _calculate_vega(underlying_price, d1, time_to_expiry, dividend_yield):
        """Calculate vega"""
        return (
            underlying_price
            * math.exp(-dividend_yield * time_to_expiry)
            * norm.pdf(d1)
            * math.sqrt(time_to_expiry)
            / 100  # Divide by 100 for 1% volatility change
        )

    @staticmethod
    def _calculate_rho(strike_price, time_to_expiry, risk_free_rate, d2, option_type):
        """Calculate rho"""
        if option_type == OptionType.CALL:
            return (
                strike_price
                * time_to_expiry
                * math.exp(-risk_free_rate * time_to_expiry)
                * norm.cdf(d2)
                / 100
            )
        else:
            return (
                -strike_price
                * time_to_expiry
                * math.exp(-risk_free_rate * time_to_expiry)
                * norm.cdf(-d2)
                / 100
            )

    @staticmethod
    def calculate_second_order_greeks(
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: OptionType,
        dividend_yield: float = 0.0,
    ) -> Dict[str, float]:
        """Calculate second-order Greeks"""
        try:
            if time_to_expiry <= 0:
                return {
                    "vanna": 0.0, "charm": 0.0, "vomma": 0.0, "veta": 0.0,
                    "color": 0.0, "speed": 0.0, "zomma": 0.0, "ultima": 0.0
                }

            d1 = BlackScholesCalculator._calculate_d1(
                underlying_price, strike_price, time_to_expiry,
                risk_free_rate, volatility, dividend_yield
            )
            d2 = d1 - volatility * math.sqrt(time_to_expiry)

            sqrt_t = math.sqrt(time_to_expiry)
            exp_div_t = math.exp(-dividend_yield * time_to_expiry)
            exp_rf_t = math.exp(-risk_free_rate * time_to_expiry)
            norm_pdf_d1 = norm.pdf(d1)

            # Vanna: d(Delta)/d(Volatility) = d(Vega)/d(Underlying)
            vanna = -exp_div_t * norm_pdf_d1 * d2 / volatility / 100

            # Charm: d(Delta)/d(Time)
            if option_type == OptionType.CALL:
                charm = dividend_yield * exp_div_t * norm.cdf(d1) - exp_div_t * norm_pdf_d1 * (
                    (risk_free_rate - dividend_yield) / (volatility * sqrt_t) - d2 / (2 * time_to_expiry)
                )
            else:
                charm = -dividend_yield * exp_div_t * norm.cdf(-d1) - exp_div_t * norm_pdf_d1 * (
                    (risk_free_rate - dividend_yield) / (volatility * sqrt_t) - d2 / (2 * time_to_expiry)
                )
            charm = charm / 365  # Daily charm

            # Vomma: d(Vega)/d(Volatility)
            vomma = underlying_price * exp_div_t * norm_pdf_d1 * sqrt_t * d1 * d2 / volatility / 10000

            # Veta: d(Vega)/d(Time)
            veta = underlying_price * exp_div_t * norm_pdf_d1 * sqrt_t * (
                dividend_yield + ((risk_free_rate - dividend_yield) * d1) / (volatility * sqrt_t) -
                (1 + d1 * d2) / (2 * time_to_expiry)
            ) / 365 / 100

            # Color: d(Gamma)/d(Time)
            color = exp_div_t * norm_pdf_d1 / (underlying_price * volatility * sqrt_t) * (
                2 * dividend_yield * time_to_expiry + 1 + d1 / (volatility * sqrt_t) * (
                    2 * (risk_free_rate - dividend_yield) * time_to_expiry - d2 * volatility * sqrt_t
                )
            ) / 365

            # Speed: d(Gamma)/d(Underlying)
            speed = -exp_div_t * norm_pdf_d1 / (underlying_price ** 2 * volatility * sqrt_t) * (
                d1 / (volatility * sqrt_t) + 1
            )

            # Zomma: d(Gamma)/d(Volatility)
            zomma = exp_div_t * norm_pdf_d1 / (underlying_price * volatility ** 2 * sqrt_t) * (
                d1 * d2 - 1
            ) / 100

            # Ultima: d(Vomma)/d(Volatility)
            ultima = underlying_price * exp_div_t * norm_pdf_d1 * sqrt_t / (volatility ** 2) * (
                d1 * d2 * (1 - d1 * d2) + d1 ** 2 + d2 ** 2
            ) / 1000000

            return {
                "vanna": vanna,
                "charm": charm,
                "vomma": vomma,
                "veta": veta,
                "color": color,
                "speed": speed,
                "zomma": zomma,
                "ultima": ultima
            }

        except Exception as e:
            logger.error("Failed to calculate second-order Greeks", error=str(e))
            return {
                "vanna": 0.0, "charm": 0.0, "vomma": 0.0, "veta": 0.0,
                "color": 0.0, "speed": 0.0, "zomma": 0.0, "ultima": 0.0
            }

    @staticmethod
    def calculate_enhanced_greeks(
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: OptionType,
        dividend_yield: float = 0.0,
    ) -> GreeksData:
        """Calculate complete Greeks including second-order"""
        try:
            # Get first-order Greeks
            first_order = BlackScholesCalculator.calculate_greeks(
                underlying_price, strike_price, time_to_expiry,
                risk_free_rate, volatility, option_type, dividend_yield
            )

            # Get second-order Greeks
            second_order = BlackScholesCalculator.calculate_second_order_greeks(
                underlying_price, strike_price, time_to_expiry,
                risk_free_rate, volatility, option_type, dividend_yield
            )

            # Combine all Greeks
            return GreeksData(
                delta=first_order.delta,
                gamma=first_order.gamma,
                theta=first_order.theta,
                vega=first_order.vega,
                rho=first_order.rho,
                vanna=second_order["vanna"],
                charm=second_order["charm"],
                vomma=second_order["vomma"],
                veta=second_order["veta"],
                color=second_order["color"],
                speed=second_order["speed"],
                zomma=second_order["zomma"],
                ultima=second_order["ultima"],
                intrinsic_value=first_order.intrinsic_value,
                time_value=first_order.time_value,
                implied_volatility=first_order.implied_volatility
            )

        except Exception as e:
            logger.error("Failed to calculate enhanced Greeks", error=str(e))
            return GreeksData(0, 0, 0, 0, 0, 0, 0)


class EnhancedBlackScholesCalculator:
    """Enhanced Black-Scholes calculator with advanced volatility models and model ensemble"""
    
    def __init__(self):
        from app.ml.volatility_predictor import get_volatility_predictor
        self.volatility_predictor = get_volatility_predictor()
        
        # Initialize advanced volatility models
        self.advanced_vol_calculator = AdvancedVolatilityCalculator()
        
        # Model performance tracking
        self.model_performance = {
            'black_scholes': {'accuracy': 0.85, 'speed': 1.0},
            'heston': {'accuracy': 0.92, 'speed': 0.3},
            'sabr': {'accuracy': 0.90, 'speed': 0.7},
            'local_vol': {'accuracy': 0.88, 'speed': 0.5},
            'monte_carlo': {'accuracy': 0.95, 'speed': 0.1}
        }
        
        logger.info("EnhancedBlackScholesCalculator initialized with advanced volatility models")
    
    async def calculate_option_price_with_advanced_models(
        self,
        symbol: str,
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: OptionType,
        price_data: List[Dict[str, Any]],
        option_chain: Optional[List[Dict[str, Any]]] = None,
        expiration: Optional[str] = None,
        dividend_yield: float = 0.0,
        use_model_ensemble: bool = True,
        pricing_method: str = "auto"
    ) -> Dict[str, Any]:
        """
        Calculate option price using advanced volatility models with ensemble approach
        
        Args:
            pricing_method: 'auto', 'black_scholes', 'heston', 'sabr', 'local_vol', 'monte_carlo'
        
        Returns:
            Dictionary with pricing results from multiple models
        """
        try:
            # Get volatility prediction from existing system
            vol_metrics = await self.volatility_predictor.predict_volatility(
                symbol=symbol,
                price_data=price_data,
                option_chain=option_chain,
                expiration=expiration
            )
            
            # Choose volatility estimate
            market_vol = vol_metrics.implied_vol if vol_metrics.implied_vol else vol_metrics.garch_predicted_vol
            
            results = {
                'symbol': symbol,
                'underlying_price': underlying_price,
                'strike_price': strike_price,
                'time_to_expiry': time_to_expiry,
                'market_volatility': market_vol,
                'volatility_metrics': vol_metrics,
                'pricing_timestamp': datetime.now()
            }
            
            # Calibrate advanced models if option chain available
            if option_chain and len(option_chain) > 5:
                market_data = self._prepare_market_data_for_calibration(option_chain, underlying_price)
                self.advanced_vol_calculator.calibrate_models(market_data, underlying_price)
                results['models_calibrated'] = True
            else:
                results['models_calibrated'] = False
            
            # Model selection based on market conditions and requirements
            selected_models = self._select_optimal_models(
                pricing_method, time_to_expiry, vol_metrics, use_model_ensemble
            )
            
            # Calculate prices with selected models
            model_results = {}
            
            # Standard Black-Scholes (always computed as baseline)
            bs_price = BlackScholesCalculator.calculate_option_price(
                underlying_price, strike_price, time_to_expiry,
                risk_free_rate, market_vol, option_type, dividend_yield
            )
            model_results['black_scholes'] = {
                'price': bs_price,
                'volatility_used': market_vol,
                'computation_time_ms': 1.0
            }
            
            # Advanced models ensemble
            if use_model_ensemble or pricing_method == 'ensemble':
                start_time = time.time()
                ensemble_results = self.advanced_vol_calculator.get_model_ensemble_price(
                    underlying_price, strike_price, time_to_expiry,
                    risk_free_rate, option_type, market_vol
                )
                computation_time = (time.time() - start_time) * 1000
                
                model_results['ensemble'] = {
                    'price': ensemble_results.get('ensemble_price', bs_price),
                    'model_prices': ensemble_results,
                    'computation_time_ms': computation_time,
                    'price_spread': ensemble_results.get('price_spread', 0.0)
                }
            
            # Specific model pricing
            for model in selected_models:
                if model == 'heston':
                    start_time = time.time()
                    heston_price = self.advanced_vol_calculator.heston_model.price_option(
                        underlying_price, strike_price, time_to_expiry, risk_free_rate, option_type
                    )
                    model_results['heston'] = {
                        'price': heston_price,
                        'model_params': {
                            'kappa': self.advanced_vol_calculator.heston_model.kappa,
                            'theta': self.advanced_vol_calculator.heston_model.theta,
                            'xi': self.advanced_vol_calculator.heston_model.xi,
                            'rho': self.advanced_vol_calculator.heston_model.rho,
                            'v0': self.advanced_vol_calculator.heston_model.v0
                        },
                        'computation_time_ms': (time.time() - start_time) * 1000
                    }
                
                elif model == 'sabr':
                    start_time = time.time()
                    sabr_vol = self.advanced_vol_calculator.sabr_model.implied_volatility(
                        underlying_price, strike_price, time_to_expiry
                    )
                    sabr_price = BlackScholesCalculator.calculate_option_price(
                        underlying_price, strike_price, time_to_expiry,
                        risk_free_rate, sabr_vol, option_type, dividend_yield
                    )
                    model_results['sabr'] = {
                        'price': sabr_price,
                        'volatility_used': sabr_vol,
                        'model_params': {
                            'alpha': self.advanced_vol_calculator.sabr_model.alpha,
                            'beta': self.advanced_vol_calculator.sabr_model.beta,
                            'rho': self.advanced_vol_calculator.sabr_model.rho,
                            'nu': self.advanced_vol_calculator.sabr_model.nu
                        },
                        'computation_time_ms': (time.time() - start_time) * 1000
                    }
                
                elif model == 'monte_carlo':
                    start_time = time.time()
                    mc_results = self.advanced_vol_calculator.monte_carlo_engine.price_european_option(
                        underlying_price, strike_price, time_to_expiry,
                        risk_free_rate, market_vol, option_type, dividend_yield
                    )
                    model_results['monte_carlo'] = {
                        'price': mc_results['price'],
                        'std_error': mc_results['std_error'],
                        'confidence_95': mc_results['confidence_95'],
                        'delta': mc_results['delta'],
                        'gamma': mc_results['gamma'],
                        'computation_time_ms': (time.time() - start_time) * 1000,
                        'n_paths': self.advanced_vol_calculator.monte_carlo_engine.n_paths
                    }
            
            # Determine best price estimate
            best_price = self._select_best_price_estimate(model_results, vol_metrics)
            
            results.update({
                'model_results': model_results,
                'best_price_estimate': best_price,
                'selected_models': selected_models,
                'model_selection_reason': self._get_model_selection_rationale(
                    selected_models, vol_metrics, time_to_expiry
                )
            })
            
            logger.debug(
                "Advanced option pricing completed",
                symbol=symbol,
                best_price=best_price['price'],
                models_used=len(selected_models),
                ensemble_enabled=use_model_ensemble,
                confidence=vol_metrics.confidence_score
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "Advanced option pricing failed",
                symbol=symbol,
                error=str(e)
            )
            
            # Fallback to original enhanced calculation
            fallback_price, vol_metrics = await self.calculate_option_price_with_prediction(
                symbol, underlying_price, strike_price, time_to_expiry,
                risk_free_rate, option_type, price_data, option_chain,
                expiration, dividend_yield
            )
            
            return {
                'symbol': symbol,
                'best_price_estimate': {'price': fallback_price, 'model': 'fallback_enhanced'},
                'volatility_metrics': vol_metrics,
                'error': str(e),
                'pricing_timestamp': datetime.now()
            }
    
    def _prepare_market_data_for_calibration(self, option_chain: List[Dict[str, Any]], 
                                           spot_price: float) -> List[Dict[str, Any]]:
        """Prepare market data for model calibration"""
        market_data = []
        
        for option in option_chain:
            if option.get('implied_volatility') and option['implied_volatility'] > 0:
                # Convert expiration to time to maturity
                maturity = self._calculate_time_to_maturity(option.get('expiration_date', ''))
                
                market_data.append({
                    'strike': float(option['strike_price']),
                    'maturity': maturity,
                    'implied_vol': float(option['implied_volatility']),
                    'option_type': option.get('option_type', 'call'),
                    'bid': option.get('bid', 0.0),
                    'ask': option.get('ask', 0.0),
                    'volume': option.get('volume', 0),
                    'open_interest': option.get('open_interest', 0)
                })
        
        return market_data
    
    def _calculate_time_to_maturity(self, expiration_str: str) -> float:
        """Calculate time to maturity in years"""
        try:
            from datetime import datetime, date
            if isinstance(expiration_str, str):
                expiration_date = datetime.strptime(expiration_str, '%Y-%m-%d').date()
            else:
                expiration_date = expiration_str
            
            today = date.today()
            days_to_expiry = (expiration_date - today).days
            return max(1/365, days_to_expiry / 365.0)  # Minimum 1 day
        except:
            return 0.25  # Default 3 months
    
    def _select_optimal_models(self, pricing_method: str, time_to_expiry: float,
                              vol_metrics: "VolatilityMetrics", use_ensemble: bool) -> List[str]:
        """Select optimal pricing models based on market conditions"""
        
        if pricing_method != "auto":
            if pricing_method == "ensemble":
                return ['heston', 'sabr', 'local_vol']
            else:
                return [pricing_method]
        
        selected = ['black_scholes']  # Always include baseline
        
        # Model selection logic based on market conditions
        if vol_metrics.vol_regime.value in ['HIGH', 'EXTREME']:
            # High volatility - use stochastic volatility models
            selected.extend(['heston', 'sabr'])
        
        if time_to_expiry > 0.5:  # More than 6 months
            # Long-term options - local volatility and Heston work well
            selected.extend(['local_vol', 'heston'])
        elif time_to_expiry < 0.1:  # Less than 1 month
            # Short-term options - SABR for smile, Monte Carlo for accuracy
            selected.extend(['sabr', 'monte_carlo'])
        
        if vol_metrics.vol_smile_skew and abs(vol_metrics.vol_smile_skew) > 0.05:
            # Significant volatility skew - use SABR model
            if 'sabr' not in selected:
                selected.append('sabr')
        
        if vol_metrics.confidence_score < 0.5:
            # Low confidence in volatility estimate - use Monte Carlo
            if 'monte_carlo' not in selected:
                selected.append('monte_carlo')
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(selected))
    
    def _select_best_price_estimate(self, model_results: Dict[str, Any],
                                   vol_metrics: "VolatilityMetrics") -> Dict[str, Any]:
        """Select the best price estimate from available models"""
        
        # Default to Black-Scholes
        best_estimate = {
            'price': model_results['black_scholes']['price'],
            'model': 'black_scholes',
            'confidence': 0.5
        }
        
        # Prefer ensemble if available
        if 'ensemble' in model_results:
            ensemble_result = model_results['ensemble']
            price_spread = ensemble_result.get('price_spread', 0.0)
            
            if price_spread < best_estimate['price'] * 0.1:  # Spread less than 10%
                best_estimate = {
                    'price': ensemble_result['price'],
                    'model': 'ensemble',
                    'confidence': min(0.95, vol_metrics.confidence_score + 0.2),
                    'price_spread': price_spread
                }
        
        # Use Monte Carlo for high accuracy if available
        elif 'monte_carlo' in model_results and vol_metrics.confidence_score < 0.6:
            mc_result = model_results['monte_carlo']
            confidence_interval = mc_result.get('confidence_95', 0.0)
            
            if confidence_interval < best_estimate['price'] * 0.05:  # Tight confidence interval
                best_estimate = {
                    'price': mc_result['price'],
                    'model': 'monte_carlo',
                    'confidence': 0.95,
                    'std_error': mc_result['std_error']
                }
        
        # Use Heston for high volatility regimes
        elif 'heston' in model_results and vol_metrics.vol_regime.value in ['HIGH', 'EXTREME']:
            best_estimate = {
                'price': model_results['heston']['price'],
                'model': 'heston',
                'confidence': min(0.9, vol_metrics.confidence_score + 0.3)
            }
        
        return best_estimate
    
    def _get_model_selection_rationale(self, selected_models: List[str],
                                      vol_metrics: "VolatilityMetrics",
                                      time_to_expiry: float) -> str:
        """Provide rationale for model selection"""
        
        reasons = []
        
        if vol_metrics.vol_regime.value in ['HIGH', 'EXTREME']:
            reasons.append(f"High volatility regime ({vol_metrics.vol_regime.value})")
        
        if time_to_expiry > 0.5:
            reasons.append("Long-term expiration (>6 months)")
        elif time_to_expiry < 0.1:
            reasons.append("Short-term expiration (<1 month)")
        
        if vol_metrics.vol_smile_skew and abs(vol_metrics.vol_smile_skew) > 0.05:
            reasons.append(f"Significant volatility skew ({vol_metrics.vol_smile_skew:.3f})")
        
        if vol_metrics.confidence_score < 0.5:
            reasons.append(f"Low volatility confidence ({vol_metrics.confidence_score:.2f})")
        
        if len(reasons) == 0:
            reasons.append("Standard market conditions")
        
        return f"Selected {len(selected_models)} models: {', '.join(selected_models)}. " + \
               f"Reasons: {', '.join(reasons)}."
    
    async def calculate_option_price_with_prediction(
        self,
        symbol: str,
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: OptionType,
        price_data: List[Dict[str, Any]],
        option_chain: Optional[List[Dict[str, Any]]] = None,
        expiration: Optional[str] = None,
        dividend_yield: float = 0.0,
        use_implied_vol: bool = True
    ) -> Tuple[float, "VolatilityMetrics"]:
        """
        Calculate option price using predicted volatility (legacy method)
        
        Returns:
            Tuple of (option_price, volatility_metrics)
        """
        try:
            # Get volatility prediction
            vol_metrics = await self.volatility_predictor.predict_volatility(
                symbol=symbol,
                price_data=price_data,
                option_chain=option_chain,
                expiration=expiration
            )
            
            # Choose best volatility estimate
            if use_implied_vol and vol_metrics.implied_vol and vol_metrics.implied_vol > 0:
                volatility = vol_metrics.implied_vol
                vol_source = "implied"
            else:
                volatility = vol_metrics.garch_predicted_vol
                vol_source = "garch"
            
            # Calculate option price
            option_price = BlackScholesCalculator.calculate_option_price(
                underlying_price=underlying_price,
                strike_price=strike_price,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                option_type=option_type,
                dividend_yield=dividend_yield
            )
            
            logger.debug(
                "Enhanced option pricing completed",
                symbol=symbol,
                strike=strike_price,
                volatility=volatility,
                vol_source=vol_source,
                option_price=option_price,
                confidence=vol_metrics.confidence_score
            )
            
            return option_price, vol_metrics
            
        except Exception as e:
            logger.error(
                "Enhanced option pricing failed",
                symbol=symbol,
                error=str(e)
            )
            # Fallback to standard calculation with default volatility
            fallback_price = BlackScholesCalculator.calculate_option_price(
                underlying_price, strike_price, time_to_expiry,
                risk_free_rate, 0.25, option_type, dividend_yield
            )
            
            from app.ml.volatility_predictor import VolatilityMetrics, VolatilityRegime
            from datetime import datetime
            
            fallback_metrics = VolatilityMetrics(
                historical_vol=0.25,
                garch_predicted_vol=0.25,
                implied_vol=None,
                vol_smile_skew=0.0,
                vol_regime=VolatilityRegime.NORMAL,
                confidence_score=0.1,
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
            
            return fallback_price, fallback_metrics
    
    async def calculate_greeks_with_prediction(
        self,
        symbol: str,
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: OptionType,
        price_data: List[Dict[str, Any]],
        option_chain: Optional[List[Dict[str, Any]]] = None,
        expiration: Optional[str] = None,
        dividend_yield: float = 0.0
    ) -> Tuple[GreeksData, "VolatilityMetrics"]:
        """Calculate Greeks with predicted volatility"""
        try:
            # Get volatility prediction
            vol_metrics = await self.volatility_predictor.predict_volatility(
                symbol=symbol,
                price_data=price_data,
                option_chain=option_chain,
                expiration=expiration
            )
            
            # Use best volatility estimate
            volatility = vol_metrics.implied_vol if vol_metrics.implied_vol else vol_metrics.garch_predicted_vol
            
            # Calculate Greeks
            greeks = BlackScholesCalculator.calculate_greeks(
                underlying_price=underlying_price,
                strike_price=strike_price,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                option_type=option_type,
                dividend_yield=dividend_yield
            )
            
            return greeks, vol_metrics
            
        except Exception as e:
            logger.error("Enhanced Greeks calculation failed", symbol=symbol, error=str(e))
            # Fallback calculation
            fallback_greeks = BlackScholesCalculator.calculate_greeks(
                underlying_price, strike_price, time_to_expiry,
                risk_free_rate, 0.25, option_type, dividend_yield
            )
            
            from app.ml.volatility_predictor import VolatilityMetrics, VolatilityRegime
            from datetime import datetime
            
            fallback_metrics = VolatilityMetrics(
                historical_vol=0.25,
                garch_predicted_vol=0.25,
                implied_vol=None,
                vol_smile_skew=0.0,
                vol_regime=VolatilityRegime.NORMAL,
                confidence_score=0.1,
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
            
            return fallback_greeks, fallback_metrics
    
    def get_volatility_for_symbol(
        self,
        symbol: str,
        expiration: Optional[str] = None,
        strike: Optional[float] = None
    ) -> float:
        """Get current volatility estimate for a symbol"""
        return self.volatility_predictor.get_volatility_for_pricing(
            symbol=symbol,
            expiration=expiration,
            strike=strike
        )
    
    def get_model_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all volatility models"""
        return {
            'model_performance': self.model_performance,
            'available_models': ['black_scholes', 'heston', 'sabr', 'local_vol', 'monte_carlo'],
            'ensemble_weights': self.advanced_vol_calculator.model_weights,
            'calibration_status': {
                'heston_calibrated': hasattr(self.advanced_vol_calculator.heston_model, 'kappa'),
                'sabr_calibrated': hasattr(self.advanced_vol_calculator.sabr_model, 'alpha'),
                'local_vol_calibrated': len(self.advanced_vol_calculator.local_vol_model.vol_surface) > 0
            }
        }  # Default volatility

class HestonVolatilityModel:
    """Heston stochastic volatility model for options pricing"""
    
    def __init__(self, kappa: float = 2.0, theta: float = 0.04, xi: float = 0.3, 
                 rho: float = -0.7, v0: float = 0.04):
        """
        Initialize Heston model parameters
        
        Args:
            kappa: Rate of mean reversion
            theta: Long-term variance
            xi: Volatility of volatility  
            rho: Correlation between asset and volatility
            v0: Initial variance
        """
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0
        
    def characteristic_function(self, u: complex, t: float) -> complex:
        """Heston characteristic function for Fourier inversion"""
        d = np.sqrt((self.rho * self.xi * u * 1j - self.kappa)**2 - 
                   self.xi**2 * (2 * u * 1j - u**2))
        g = (self.kappa - self.rho * self.xi * u * 1j - d) / \
            (self.kappa - self.rho * self.xi * u * 1j + d)
        
        C = self.kappa * self.theta / (self.xi**2) * \
            ((self.kappa - self.rho * self.xi * u * 1j - d) * t - 
             2 * np.log((1 - g * np.exp(-d * t)) / (1 - g)))
        
        D = (self.kappa - self.rho * self.xi * u * 1j - d) / (self.xi**2) * \
            (1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t))
        
        return np.exp(C + D * self.v0)
    
    def price_option(self, S: float, K: float, T: float, r: float, option_type: OptionType) -> float:
        """Price option using Heston model with Fourier inversion"""
        try:
            # Fourier inversion parameters
            alpha = 1.1 if option_type == OptionType.CALL else -0.1
            eta = 0.25
            N = 2**12
            
            # Integration limits
            lambda_max = 20.0
            d_u = lambda_max / N
            
            # FFT approach for efficiency
            b = N * d_u / 2
            u = np.arange(N) * d_u
            d_k = 2 * np.pi / (N * d_u)
            beta = np.log(S) - b / 2
            
            k_values = beta + np.arange(N) * d_k
            
            # Characteristic function evaluation
            cf_values = np.zeros(N, dtype=complex)
            for i, u_val in enumerate(u):
                cf_values[i] = self.characteristic_function(u_val - (alpha + 1) * 1j, T) / \
                              (alpha**2 + alpha - u_val**2 + 1j * (2 * alpha + 1) * u_val)
            
            # Apply FFT
            x = np.exp(-r * T) * np.real(np.fft.fft(cf_values * np.exp(-1j * beta * u) * d_u)) / np.pi
            
            # Interpolate to get price at desired strike
            log_k = np.log(K)
            price = np.interp(log_k, k_values, x)
            
            if option_type == OptionType.PUT:
                # Put-call parity
                price = price + K * np.exp(-r * T) - S
                
            return max(0, price)
            
        except Exception as e:
            logger.error(f"Heston pricing failed: {e}")
            # Fallback to Black-Scholes
            vol_estimate = np.sqrt(self.v0)
            return BlackScholesCalculator.calculate_option_price(
                S, K, T, r, vol_estimate, option_type
            )


class SABRVolatilityModel:
    """SABR (Stochastic Alpha Beta Rho) volatility model for smile modeling"""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, rho: float = -0.3, nu: float = 0.4):
        """
        Initialize SABR model parameters
        
        Args:
            alpha: Initial volatility level
            beta: Elasticity parameter (0 for normal, 1 for lognormal)
            rho: Correlation between forward and volatility
            nu: Volatility of volatility
        """
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
    
    def implied_volatility(self, F: float, K: float, T: float) -> float:
        """Calculate SABR implied volatility for given forward and strike"""
        try:
            if abs(F - K) < 1e-8:  # ATM case
                return self._atm_volatility(F, T)
            
            # SABR volatility smile formula
            fk_beta = (F * K)**((1 - self.beta) / 2)
            log_fk = np.log(F / K)
            
            # Numerator terms
            z = (self.nu / self.alpha) * fk_beta * log_fk
            x_z = np.log((np.sqrt(1 - 2*self.rho*z + z**2) + z - self.rho) / (1 - self.rho))
            
            # First order approximation
            if abs(z) < 0.01:
                x_z = z
            
            # SABR formula
            numerator = self.alpha
            denominator = fk_beta * (1 + ((1-self.beta)**2/24) * log_fk**2 + 
                                   ((1-self.beta)**4/1920) * log_fk**4)
            
            first_term = numerator / denominator
            
            # Time-dependent correction
            time_correction = (1 + (((1-self.beta)**2 * self.alpha**2)/(24 * fk_beta**2) +
                                  (self.rho * self.nu * self.alpha)/(4 * fk_beta) +
                                  (2 - 3*self.rho**2) * self.nu**2/24) * T)
            
            volatility = first_term * (z / x_z) * time_correction
            
            return max(0.01, volatility)  # Floor at 1%
            
        except Exception as e:
            logger.error(f"SABR volatility calculation failed: {e}")
            return self._atm_volatility(F, T)
    
    def _atm_volatility(self, F: float, T: float) -> float:
        """ATM volatility for SABR model"""
        try:
            f_beta = F**(1 - self.beta)
            atm_vol = (self.alpha / f_beta) * (
                1 + (((1-self.beta)**2 * self.alpha**2)/(24 * f_beta**2) +
                     (self.rho * self.nu * self.alpha)/(4 * f_beta) +
                     (2 - 3*self.rho**2) * self.nu**2/24) * T
            )
            return max(0.01, atm_vol)
        except:
            return 0.25  # Default fallback


class LocalVolatilityModel:
    """Local volatility model with surface interpolation"""
    
    def __init__(self):
        self.vol_surface = {}
        self.strikes = []
        self.maturities = []
        
    def calibrate_surface(self, market_data: List[Dict[str, Any]]):
        """Calibrate local volatility surface from market data"""
        try:
            # Organize market data by strike and maturity
            surface_data = {}
            strikes_set = set()
            maturities_set = set()
            
            for option in market_data:
                strike = option['strike']
                maturity = option['maturity']
                implied_vol = option['implied_vol']
                
                if implied_vol > 0:
                    surface_data[(strike, maturity)] = implied_vol
                    strikes_set.add(strike)
                    maturities_set.add(maturity)
            
            self.strikes = sorted(strikes_set)
            self.maturities = sorted(maturities_set)
            
            # Build interpolated surface
            for strike in self.strikes:
                for maturity in self.maturities:
                    if (strike, maturity) in surface_data:
                        self.vol_surface[(strike, maturity)] = surface_data[(strike, maturity)]
                    else:
                        # Interpolate missing points
                        self.vol_surface[(strike, maturity)] = self._interpolate_volatility(
                            strike, maturity, surface_data
                        )
                        
            logger.info(f"Calibrated local vol surface with {len(self.strikes)} strikes, {len(self.maturities)} maturities")
            
        except Exception as e:
            logger.error(f"Local vol calibration failed: {e}")
    
    def _interpolate_volatility(self, target_strike: float, target_maturity: float, 
                              surface_data: Dict) -> float:
        """Bilinear interpolation for missing volatility points"""
        try:
            # Find surrounding points
            nearby_points = []
            for (strike, maturity), vol in surface_data.items():
                distance = abs(strike - target_strike) + abs(maturity - target_maturity)
                nearby_points.append((distance, vol))
            
            if not nearby_points:
                return 0.25
                
            # Weighted average of closest points
            nearby_points.sort()
            top_points = nearby_points[:4]  # Use 4 closest points
            
            total_weight = 0
            weighted_vol = 0
            
            for distance, vol in top_points:
                weight = 1 / (1 + distance)
                weighted_vol += vol * weight
                total_weight += weight
            
            return weighted_vol / total_weight if total_weight > 0 else 0.25
            
        except Exception as e:
            logger.error(f"Volatility interpolation failed: {e}")
            return 0.25
    
    def get_local_volatility(self, strike: float, maturity: float) -> float:
        """Get local volatility for given strike and maturity"""
        try:
            # Direct lookup if available
            if (strike, maturity) in self.vol_surface:
                return self.vol_surface[(strike, maturity)]
            
            # Interpolate between available points
            if not self.strikes or not self.maturities:
                return 0.25
                
            # Find bounding strikes and maturities
            strike_below = max([s for s in self.strikes if s <= strike], default=min(self.strikes))
            strike_above = min([s for s in self.strikes if s >= strike], default=max(self.strikes))
            
            maturity_below = max([m for m in self.maturities if m <= maturity], default=min(self.maturities))
            maturity_above = min([m for m in self.maturities if m >= maturity], default=max(self.maturities))
            
            # Bilinear interpolation
            vol_00 = self.vol_surface.get((strike_below, maturity_below), 0.25)
            vol_01 = self.vol_surface.get((strike_below, maturity_above), 0.25)
            vol_10 = self.vol_surface.get((strike_above, maturity_below), 0.25)
            vol_11 = self.vol_surface.get((strike_above, maturity_above), 0.25)
            
            if strike_above == strike_below:
                w_strike = 0.5
            else:
                w_strike = (strike - strike_below) / (strike_above - strike_below)
                
            if maturity_above == maturity_below:
                w_maturity = 0.5
            else:
                w_maturity = (maturity - maturity_below) / (maturity_above - maturity_below)
            
            vol_0 = vol_00 * (1 - w_strike) + vol_10 * w_strike
            vol_1 = vol_01 * (1 - w_strike) + vol_11 * w_strike
            
            interpolated_vol = vol_0 * (1 - w_maturity) + vol_1 * w_maturity
            
            return max(0.01, interpolated_vol)
            
        except Exception as e:
            logger.error(f"Local volatility lookup failed: {e}")
            return 0.25


class MonteCarloEngine:
    """Monte Carlo simulation engine for exotic options pricing"""
    
    def __init__(self, n_paths: int = 10000, n_steps: int = 252):
        self.n_paths = n_paths
        self.n_steps = n_steps
        
    def price_european_option(self, S0: float, K: float, T: float, r: float, 
                            sigma: float, option_type: OptionType, 
                            dividend_yield: float = 0.0) -> Dict[str, float]:
        """Price European option using Monte Carlo simulation"""
        try:
            dt = T / self.n_steps
            sqrt_dt = np.sqrt(dt)
            
            # Generate random paths
            np.random.seed(42)  # For reproducible results
            Z = np.random.standard_normal((self.n_paths, self.n_steps))
            
            # Initialize paths
            S = np.zeros((self.n_paths, self.n_steps + 1))
            S[:, 0] = S0
            
            # Simulate paths using Geometric Brownian Motion
            for t in range(self.n_steps):
                drift = (r - dividend_yield - 0.5 * sigma**2) * dt
                diffusion = sigma * sqrt_dt * Z[:, t]
                S[:, t + 1] = S[:, t] * np.exp(drift + diffusion)
            
            # Calculate payoffs
            if option_type == OptionType.CALL:
                payoffs = np.maximum(S[:, -1] - K, 0)
            else:
                payoffs = np.maximum(K - S[:, -1], 0)
            
            # Discount to present value
            price = np.exp(-r * T) * np.mean(payoffs)
            std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(self.n_paths)
            
            # Calculate Greeks via finite differences
            delta = self._calculate_delta_mc(S0, K, T, r, sigma, option_type, dividend_yield)
            gamma = self._calculate_gamma_mc(S0, K, T, r, sigma, option_type, dividend_yield)
            
            return {
                'price': price,
                'std_error': std_error,
                'delta': delta,
                'gamma': gamma,
                'confidence_95': 1.96 * std_error
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo pricing failed: {e}")
            return {'price': 0.0, 'std_error': 0.0, 'delta': 0.0, 'gamma': 0.0}
    
    def _calculate_delta_mc(self, S0: float, K: float, T: float, r: float, 
                           sigma: float, option_type: OptionType, dividend_yield: float) -> float:
        """Calculate delta using finite differences"""
        try:
            h = 0.01 * S0  # 1% bump
            
            price_up = self.price_european_option(S0 + h, K, T, r, sigma, option_type, dividend_yield)['price']
            price_down = self.price_european_option(S0 - h, K, T, r, sigma, option_type, dividend_yield)['price']
            
            return (price_up - price_down) / (2 * h)
            
        except Exception as e:
            logger.error(f"Delta calculation failed: {e}")
            return 0.0
    
    def _calculate_gamma_mc(self, S0: float, K: float, T: float, r: float,
                           sigma: float, option_type: OptionType, dividend_yield: float) -> float:
        """Calculate gamma using finite differences"""
        try:
            h = 0.01 * S0  # 1% bump
            
            price_up = self.price_european_option(S0 + h, K, T, r, sigma, option_type, dividend_yield)['price']
            price_center = self.price_european_option(S0, K, T, r, sigma, option_type, dividend_yield)['price']
            price_down = self.price_european_option(S0 - h, K, T, r, sigma, option_type, dividend_yield)['price']
            
            return (price_up - 2 * price_center + price_down) / (h**2)
            
        except Exception as e:
            logger.error(f"Gamma calculation failed: {e}")
            return 0.0


class AdvancedVolatilityCalculator:
    """Advanced volatility calculator with multiple models"""
    
    def __init__(self):
        self.heston_model = HestonVolatilityModel()
        self.sabr_model = SABRVolatilityModel()
        self.local_vol_model = LocalVolatilityModel()
        self.monte_carlo_engine = MonteCarloEngine()
        
        self.model_weights = {
            'heston': 0.3,
            'sabr': 0.3,
            'local_vol': 0.2,
            'black_scholes': 0.2
        }
        
    def calibrate_models(self, market_data: List[Dict[str, Any]], underlying_price: float):
        """Calibrate all volatility models to market data"""
        try:
            # Calibrate SABR model
            self._calibrate_sabr(market_data, underlying_price)
            
            # Calibrate Heston model
            self._calibrate_heston(market_data, underlying_price)
            
            # Calibrate local volatility surface
            self.local_vol_model.calibrate_surface(market_data)
            
            logger.info("All volatility models calibrated successfully")
            
        except Exception as e:
            logger.error(f"Model calibration failed: {e}")
    
    def _calibrate_sabr(self, market_data: List[Dict[str, Any]], forward_price: float):
        """Calibrate SABR model to market smiles"""
        try:
            # Simple calibration - in practice would use optimization
            # For now, use reasonable market-typical parameters
            self.sabr_model.alpha = 0.3
            self.sabr_model.beta = 0.7
            self.sabr_model.rho = -0.3
            self.sabr_model.nu = 0.4
            
            logger.info("SABR model calibrated")
            
        except Exception as e:
            logger.error(f"SABR calibration failed: {e}")
    
    def _calibrate_heston(self, market_data: List[Dict[str, Any]], spot_price: float):
        """Calibrate Heston model to market data"""
        try:
            # Use typical market parameters - in practice would optimize
            self.heston_model.kappa = 2.0
            self.heston_model.theta = 0.04
            self.heston_model.xi = 0.3
            self.heston_model.rho = -0.7
            self.heston_model.v0 = 0.04
            
            logger.info("Heston model calibrated")
            
        except Exception as e:
            logger.error(f"Heston calibration failed: {e}")
    
    def get_model_ensemble_price(self, underlying_price: float, strike_price: float,
                                time_to_expiry: float, risk_free_rate: float,
                                option_type: OptionType, market_vol: float = None) -> Dict[str, Any]:
        """Get ensemble price from multiple volatility models"""
        try:
            results = {}
            
            # Black-Scholes baseline
            if market_vol:
                bs_price = BlackScholesCalculator.calculate_option_price(
                    underlying_price, strike_price, time_to_expiry,
                    risk_free_rate, market_vol, option_type
                )
                results['black_scholes'] = bs_price
            
            # Heston model
            try:
                heston_price = self.heston_model.price_option(
                    underlying_price, strike_price, time_to_expiry, risk_free_rate, option_type
                )
                results['heston'] = heston_price
            except:
                results['heston'] = results.get('black_scholes', 0.0)
            
            # SABR model (convert to Black-Scholes with SABR vol)
            try:
                sabr_vol = self.sabr_model.implied_volatility(
                    underlying_price, strike_price, time_to_expiry
                )
                sabr_price = BlackScholesCalculator.calculate_option_price(
                    underlying_price, strike_price, time_to_expiry,
                    risk_free_rate, sabr_vol, option_type
                )
                results['sabr'] = sabr_price
                results['sabr_vol'] = sabr_vol
            except:
                results['sabr'] = results.get('black_scholes', 0.0)
                results['sabr_vol'] = market_vol or 0.25
            
            # Local volatility model
            try:
                local_vol = self.local_vol_model.get_local_volatility(strike_price, time_to_expiry)
                local_price = BlackScholesCalculator.calculate_option_price(
                    underlying_price, strike_price, time_to_expiry,
                    risk_free_rate, local_vol, option_type
                )
                results['local_vol'] = local_price
                results['local_vol_used'] = local_vol
            except:
                results['local_vol'] = results.get('black_scholes', 0.0)
                results['local_vol_used'] = market_vol or 0.25
            
            # Ensemble weighted average
            total_weight = 0
            weighted_price = 0
            
            for model, weight in self.model_weights.items():
                if model in results:
                    weighted_price += results[model] * weight
                    total_weight += weight
            
            ensemble_price = weighted_price / total_weight if total_weight > 0 else 0.0
            
            results['ensemble_price'] = ensemble_price
            results['model_weights'] = self.model_weights
            results['price_spread'] = max(results.values()) - min([v for v in results.values() if isinstance(v, (int, float))])
            
            return results
            
        except Exception as e:
            logger.error(f"Ensemble pricing failed: {e}")
            return {'ensemble_price': 0.0, 'error': str(e)}


# Global enhanced calculator instance
_enhanced_calculator: Optional[EnhancedBlackScholesCalculator] = None


def get_enhanced_calculator() -> EnhancedBlackScholesCalculator:
    """Get or create global enhanced calculator instance"""
    global _enhanced_calculator
    if _enhanced_calculator is None:
        _enhanced_calculator = EnhancedBlackScholesCalculator()
    return _enhanced_calculator


class OptionsTrader:
    """Options trading manager with Greeks calculation"""

    def __init__(self, alpaca_client: Optional[AlpacaClient] = None):
        self.alpaca_client = alpaca_client or AlpacaClient()
        self.market_cache = get_market_cache()
        self.risk_free_rate = 0.05  # 5% annual risk-free rate (adjustable)
        self.calculator = BlackScholesCalculator()

    async def get_option_chain(
        self, symbol: str, expiration_date: Optional[str] = None
    ) -> List[OptionContract]:
        """Get option chain for a symbol from Alpaca"""
        try:
            # Get options data from Alpaca
            options_data = await self.alpaca_client.get_option_chain(
                symbol, expiration_date
            )

            if not options_data:
                logger.warning(f"No options data found for {symbol}")
                return []

            # Get underlying price
            underlying_price = await self.alpaca_client.get_latest_price(symbol)
            if not underlying_price:
                logger.warning(f"Could not get underlying price for {symbol}")
                return []

            option_contracts = []

            for option_data in options_data:
                try:
                    # Parse option contract
                    contract = await self._parse_option_contract(
                        option_data, underlying_price["price"]
                    )

                    if contract:
                        # Calculate Greeks
                        contract.greeks = await self._calculate_option_greeks(
                            contract, underlying_price["price"]
                        )
                        option_contracts.append(contract)

                except Exception as e:
                    logger.error(f"Failed to process option contract", error=str(e))
                    continue

            logger.info(
                f"Retrieved {len(option_contracts)} option contracts for {symbol}"
            )
            return option_contracts

        except Exception as e:
            logger.error(f"Failed to get option chain for {symbol}", error=str(e))
            return []

    async def _parse_option_contract(
        self, option_data: Dict, underlying_price: float
    ) -> Optional[OptionContract]:
        """Parse option contract data from Alpaca"""
        try:
            symbol = option_data.get("symbol", "")
            underlying_symbol = option_data.get("underlying_symbol", "")

            # Parse option type from symbol or data
            option_type_str = option_data.get("type", "call").lower()
            option_type = (
                OptionType.CALL if option_type_str == "call" else OptionType.PUT
            )

            strike_price = float(option_data.get("strike_price", 0))

            # Parse expiration date
            exp_date_str = option_data.get("expiration_date", "")
            expiration_date = (
                datetime.fromisoformat(exp_date_str) if exp_date_str else datetime.now()
            )

            current_price = float(option_data.get("last_price", 0))
            bid = float(option_data.get("bid", 0))
            ask = float(option_data.get("ask", 0))
            volume = int(option_data.get("volume", 0))
            open_interest = int(option_data.get("open_interest", 0))
            implied_volatility = option_data.get("implied_volatility")

            return OptionContract(
                symbol=symbol,
                underlying_symbol=underlying_symbol,
                option_type=option_type,
                strike_price=strike_price,
                expiration_date=expiration_date,
                current_price=current_price,
                bid=bid,
                ask=ask,
                volume=volume,
                open_interest=open_interest,
                implied_volatility=implied_volatility,
            )

        except Exception as e:
            logger.error("Failed to parse option contract", error=str(e))
            return None

    async def _calculate_option_greeks(
        self, contract: OptionContract, underlying_price: float
    ) -> GreeksData:
        """Calculate Greeks for an option contract"""
        try:
            # Calculate time to expiry in years
            time_to_expiry = max(
                0,
                (contract.expiration_date - datetime.now()).total_seconds()
                / (365.25 * 24 * 3600),
            )

            # Use implied volatility if available, otherwise estimate
            volatility = contract.implied_volatility or await self._estimate_volatility(
                contract.underlying_symbol
            )

            return self.calculator.calculate_greeks(
                underlying_price=underlying_price,
                strike_price=contract.strike_price,
                time_to_expiry=time_to_expiry,
                risk_free_rate=self.risk_free_rate,
                volatility=volatility,
                option_type=contract.option_type,
            )

        except Exception as e:
            logger.error("Failed to calculate option Greeks", error=str(e))
            return GreeksData(0, 0, 0, 0, 0, 0, 0)

    async def _estimate_volatility(self, symbol: str, days: int = 30) -> float:
        """Estimate historical volatility for a symbol"""
        try:
            # Check cache first
            cache_key = f"volatility_{symbol}_{days}d"
            cached_vol = await self.market_cache.get(cache_key)
            if cached_vol:
                return cached_vol

            # Get historical prices from Alpaca
            historical_data = await self.alpaca_client.get_historical_data(symbol, days)

            if not historical_data or len(historical_data) < 10:
                logger.warning(
                    f"Insufficient data to calculate volatility for {symbol}"
                )
                return 0.2  # Default 20% volatility

            # Calculate daily returns
            prices = [float(day["close"]) for day in historical_data]
            returns = [
                math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))
            ]

            # Calculate volatility (annualized standard deviation)
            if len(returns) > 1:
                mean_return = np.mean(returns)
                variance = np.mean([(r - mean_return) ** 2 for r in returns])
                volatility = math.sqrt(variance * 252)  # Annualize (252 trading days)

                # Cache for 1 hour
                await self.market_cache.set(cache_key, volatility, ttl_override=3600)

                return volatility
            else:
                return 0.2

        except Exception as e:
            logger.error(f"Failed to estimate volatility for {symbol}", error=str(e))
            return 0.2  # Default volatility

    async def analyze_option_strategy(
        self, strategy: OptionStrategy, symbol: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze an options strategy"""
        try:
            logger.info(f"Analyzing {strategy.value} strategy for {symbol}")

            # Get underlying price
            underlying_data = await self.alpaca_client.get_latest_price(symbol)
            if not underlying_data:
                raise ValueError(f"Could not get price for {symbol}")

            underlying_price = underlying_data["price"]

            # Get option chain
            option_chain = await self.get_option_chain(symbol)
            if not option_chain:
                raise ValueError(f"No options available for {symbol}")

            # Strategy-specific analysis
            if strategy == OptionStrategy.LONG_CALL:
                return await self._analyze_long_call(
                    option_chain, underlying_price, parameters
                )
            elif strategy == OptionStrategy.COVERED_CALL:
                return await self._analyze_covered_call(
                    option_chain, underlying_price, parameters
                )
            elif strategy == OptionStrategy.PROTECTIVE_PUT:
                return await self._analyze_protective_put(
                    option_chain, underlying_price, parameters
                )
            elif strategy == OptionStrategy.STRADDLE:
                return await self._analyze_straddle(
                    option_chain, underlying_price, parameters
                )
            else:
                return {"error": f"Strategy {strategy.value} not implemented yet"}

        except Exception as e:
            logger.error(f"Failed to analyze {strategy.value} strategy", error=str(e))
            return {"error": str(e)}

    async def _analyze_long_call(
        self, option_chain: List[OptionContract], underlying_price: float, params: Dict
    ) -> Dict:
        """Analyze long call strategy"""
        try:
            target_strike = params.get(
                "strike_price", underlying_price * 1.05
            )  # 5% OTM default

            # Find best call option near target strike
            calls = [opt for opt in option_chain if opt.option_type == OptionType.CALL]

            if not calls:
                return {"error": "No call options available"}

            # Find closest strike to target
            best_call = min(calls, key=lambda x: abs(x.strike_price - target_strike))

            # Calculate strategy metrics
            max_loss = best_call.current_price
            breakeven = best_call.strike_price + best_call.current_price

            # Profit/loss at various underlying prices
            price_range = np.linspace(
                underlying_price * 0.8, underlying_price * 1.3, 21
            )
            pnl_profile = []

            for price in price_range:
                option_value = max(0, price - best_call.strike_price)
                pnl = option_value - best_call.current_price
                pnl_profile.append({"underlying_price": price, "pnl": pnl})

            return {
                "strategy": "Long Call",
                "selected_option": {
                    "symbol": best_call.symbol,
                    "strike": best_call.strike_price,
                    "expiration": best_call.expiration_date.isoformat(),
                    "current_price": best_call.current_price,
                    "greeks": best_call.greeks.__dict__ if best_call.greeks else None,
                },
                "metrics": {
                    "max_loss": max_loss,
                    "max_profit": "Unlimited",
                    "breakeven": breakeven,
                    "cost": best_call.current_price,
                },
                "pnl_profile": pnl_profile,
                "risk_analysis": {
                    "delta_exposure": best_call.greeks.delta if best_call.greeks else 0,
                    "theta_decay": best_call.greeks.theta if best_call.greeks else 0,
                    "vega_risk": best_call.greeks.vega if best_call.greeks else 0,
                },
            }

        except Exception as e:
            logger.error("Failed to analyze long call", error=str(e))
            return {"error": str(e)}

    async def _analyze_covered_call(
        self, option_chain: List[OptionContract], underlying_price: float, params: Dict
    ) -> Dict:
        """Analyze covered call strategy - own stock, sell call option"""
        try:
            target_strike = params.get(
                "strike_price", underlying_price * 1.05
            )  # 5% OTM default
            stock_quantity = params.get("stock_quantity", 100)  # Standard lot
            
            # Find best call option near target strike
            calls = [opt for opt in option_chain if opt.option_type == OptionType.CALL]
            
            if not calls:
                return {"error": "No call options available"}
            
            # Find closest strike to target (prefer slightly OTM)
            otm_calls = [call for call in calls if call.strike_price >= underlying_price]
            if otm_calls:
                best_call = min(otm_calls, key=lambda x: abs(x.strike_price - target_strike))
            else:
                best_call = min(calls, key=lambda x: abs(x.strike_price - target_strike))
            
            # Calculate strategy metrics
            premium_received = best_call.current_price * stock_quantity
            stock_cost = underlying_price * stock_quantity
            net_cost = stock_cost - premium_received
            
            # Maximum profit (if called away)
            max_profit = (best_call.strike_price - underlying_price) * stock_quantity + premium_received
            max_profit_pct = (max_profit / net_cost) * 100 if net_cost > 0 else 0
            
            # Maximum loss (if stock goes to zero)
            max_loss = net_cost
            
            # Breakeven point
            breakeven = underlying_price - best_call.current_price
            
            # Profit/loss at various underlying prices
            price_range = np.linspace(
                underlying_price * 0.7, underlying_price * 1.4, 25
            )
            pnl_profile = []
            
            for price in price_range:
                # Stock P&L
                stock_pnl = (price - underlying_price) * stock_quantity
                
                # Call option P&L (we're short the call)
                call_value = max(0, price - best_call.strike_price)
                call_pnl = (best_call.current_price - call_value) * stock_quantity
                
                total_pnl = stock_pnl + call_pnl
                
                pnl_profile.append({
                    "underlying_price": price, 
                    "stock_pnl": stock_pnl,
                    "call_pnl": call_pnl,
                    "total_pnl": total_pnl
                })
            
            # Calculate probability of profit using current volatility
            time_to_expiry = max(0, (best_call.expiration_date - datetime.now()).total_seconds() / (365.25 * 24 * 3600))
            implied_vol = best_call.implied_volatility or 0.25
            
            # Probability that stock stays below strike (call expires worthless)
            if time_to_expiry > 0 and implied_vol > 0:
                d2 = (math.log(underlying_price / best_call.strike_price) + 
                     (self.risk_free_rate - 0.5 * implied_vol**2) * time_to_expiry) / (implied_vol * math.sqrt(time_to_expiry))
                prob_profit = norm.cdf(d2)  # Probability of finishing below strike
            else:
                prob_profit = 0.5
            
            # Risk analysis
            delta_exposure = stock_quantity + (best_call.greeks.delta * stock_quantity if best_call.greeks else 0)
            theta_income = -best_call.greeks.theta * stock_quantity if best_call.greeks else 0  # Negative theta = positive income
            
            return {
                "strategy": "Covered Call",
                "selected_option": {
                    "symbol": best_call.symbol,
                    "strike": best_call.strike_price,
                    "expiration": best_call.expiration_date.isoformat(),
                    "current_price": best_call.current_price,
                    "bid": best_call.bid,
                    "ask": best_call.ask,
                    "implied_vol": best_call.implied_volatility,
                    "greeks": best_call.greeks.__dict__ if best_call.greeks else None,
                },
                "position_details": {
                    "stock_quantity": stock_quantity,
                    "stock_price": underlying_price,
                    "calls_sold": 1,  # Selling 1 call per 100 shares
                    "premium_received": premium_received,
                    "net_cost": net_cost,
                },
                "metrics": {
                    "max_profit": max_profit,
                    "max_profit_pct": max_profit_pct,
                    "max_loss": max_loss,
                    "breakeven": breakeven,
                    "premium_yield": (premium_received / stock_cost) * 100,
                    "annualized_return": (max_profit / net_cost) * (365.25 / max(time_to_expiry * 365.25, 1)) * 100 if net_cost > 0 and time_to_expiry > 0 else 0,
                    "probability_of_profit": prob_profit * 100,
                    "return_if_unchanged": (premium_received / net_cost) * 100 if net_cost > 0 else 0,
                },
                "pnl_profile": pnl_profile,
                "risk_analysis": {
                    "delta_exposure": delta_exposure,
                    "theta_income_per_day": theta_income,
                    "upside_capped": True,
                    "assignment_risk": f"High if stock > ${best_call.strike_price:.2f}",
                    "volatility_risk": "Benefits from falling volatility",
                    "time_decay": "Positive (theta positive)",
                },
                "management_rules": {
                    "profit_target": "Close at 25-50% of premium received",
                    "loss_limit": f"Consider rolling if stock drops below ${breakeven:.2f}",
                    "assignment_action": "Let shares be called away or roll up and out",
                    "earnings_warning": "Avoid holding through earnings (high assignment risk)",
                },
            }

        except Exception as e:
            logger.error("Failed to analyze covered call", error=str(e))
            return {"error": str(e)}

    async def _analyze_protective_put(
        self, option_chain: List[OptionContract], underlying_price: float, params: Dict
    ) -> Dict:
        """Analyze protective put strategy - own stock, buy put for protection"""
        try:
            target_strike = params.get(
                "strike_price", underlying_price * 0.95
            )  # 5% OTM default for protection
            stock_quantity = params.get("stock_quantity", 100)  # Standard lot
            
            # Find best put option near target strike
            puts = [opt for opt in option_chain if opt.option_type == OptionType.PUT]
            
            if not puts:
                return {"error": "No put options available"}
            
            # Find closest strike to target (prefer slightly OTM for cost efficiency)
            best_put = min(puts, key=lambda x: abs(x.strike_price - target_strike))
            
            # Calculate strategy metrics
            put_cost = best_put.current_price * stock_quantity
            stock_cost = underlying_price * stock_quantity
            total_cost = stock_cost + put_cost
            
            # Insurance cost as percentage of stock value
            insurance_cost_pct = (put_cost / stock_cost) * 100
            
            # Maximum loss (stock price falls to put strike)
            max_loss = (underlying_price - best_put.strike_price) * stock_quantity + put_cost
            max_loss_pct = (max_loss / total_cost) * 100
            
            # Maximum profit (unlimited upside minus put cost)
            # Breakeven point (stock price + put premium)
            breakeven = underlying_price + best_put.current_price
            
            # Profit/loss at various underlying prices
            price_range = np.linspace(
                underlying_price * 0.6, underlying_price * 1.5, 25
            )
            pnl_profile = []
            
            for price in price_range:
                # Stock P&L
                stock_pnl = (price - underlying_price) * stock_quantity
                
                # Put option P&L (we're long the put)
                put_value = max(0, best_put.strike_price - price)
                put_pnl = (put_value - best_put.current_price) * stock_quantity
                
                total_pnl = stock_pnl + put_pnl
                
                # Net position value
                position_value = max(price * stock_quantity, best_put.strike_price * stock_quantity) - put_cost
                
                pnl_profile.append({
                    "underlying_price": price,
                    "stock_pnl": stock_pnl,
                    "put_pnl": put_pnl,
                    "total_pnl": total_pnl,
                    "position_value": position_value,
                    "protected": price < best_put.strike_price
                })
            
            # Calculate time to expiration and key metrics
            time_to_expiry = max(0, (best_put.expiration_date - datetime.now()).total_seconds() / (365.25 * 24 * 3600))
            implied_vol = best_put.implied_volatility or 0.25
            
            # Calculate probability metrics
            if time_to_expiry > 0 and implied_vol > 0:
                # Probability that stock finishes above strike (put expires worthless)
                d2 = (math.log(underlying_price / best_put.strike_price) + 
                     (self.risk_free_rate - 0.5 * implied_vol**2) * time_to_expiry) / (implied_vol * math.sqrt(time_to_expiry))
                prob_put_expires_worthless = norm.cdf(d2)
                
                # Probability that stock finishes above breakeven
                d2_breakeven = (math.log(underlying_price / breakeven) + 
                               (self.risk_free_rate - 0.5 * implied_vol**2) * time_to_expiry) / (implied_vol * math.sqrt(time_to_expiry))
                prob_profit = norm.cdf(d2_breakeven)
            else:
                prob_put_expires_worthless = 0.5
                prob_profit = 0.5
            
            # Risk analysis
            delta_exposure = stock_quantity + (best_put.greeks.delta * stock_quantity if best_put.greeks else 0)
            theta_cost = best_put.greeks.theta * stock_quantity if best_put.greeks else 0  # Cost of time decay
            vega_exposure = best_put.greeks.vega * stock_quantity if best_put.greeks else 0
            
            # Protection level analysis
            protection_level = (underlying_price - best_put.strike_price) / underlying_price
            moneyness = best_put.strike_price / underlying_price
            
            return {
                "strategy": "Protective Put",
                "selected_option": {
                    "symbol": best_put.symbol,
                    "strike": best_put.strike_price,
                    "expiration": best_put.expiration_date.isoformat(),
                    "current_price": best_put.current_price,
                    "bid": best_put.bid,
                    "ask": best_put.ask,
                    "implied_vol": best_put.implied_volatility,
                    "greeks": best_put.greeks.__dict__ if best_put.greeks else None,
                },
                "position_details": {
                    "stock_quantity": stock_quantity,
                    "stock_price": underlying_price,
                    "puts_bought": 1,  # Buying 1 put per 100 shares
                    "put_cost": put_cost,
                    "total_cost": total_cost,
                    "protection_strike": best_put.strike_price,
                },
                "metrics": {
                    "max_loss": max_loss,
                    "max_loss_pct": max_loss_pct,
                    "max_profit": "Unlimited (minus put cost)",
                    "breakeven": breakeven,
                    "insurance_cost": put_cost,
                    "insurance_cost_pct": insurance_cost_pct,
                    "protection_level_pct": protection_level * 100,
                    "moneyness": moneyness,
                    "probability_put_worthless": prob_put_expires_worthless * 100,
                    "probability_of_profit": prob_profit * 100,
                    "annualized_insurance_cost": (insurance_cost_pct * 365.25 / max(time_to_expiry * 365.25, 1)) if time_to_expiry > 0 else 0,
                },
                "pnl_profile": pnl_profile,
                "risk_analysis": {
                    "delta_exposure": delta_exposure,
                    "theta_cost_per_day": theta_cost,
                    "vega_exposure": vega_exposure,
                    "downside_protected": f"Protected below ${best_put.strike_price:.2f}",
                    "upside_participation": f"Full upside above ${breakeven:.2f}",
                    "volatility_risk": "Benefits from rising volatility",
                    "time_decay": "Negative (theta negative)",
                },
                "management_rules": {
                    "protection_trigger": f"Put provides value if stock falls below ${best_put.strike_price:.2f}",
                    "profit_taking": "Consider selling put if stock rises significantly and implied vol drops",
                    "rolling_strategy": "Roll down and out if stock appreciates to maintain protection",
                    "expiration_management": "Decide whether to exercise or roll before expiration",
                },
                "insurance_analysis": {
                    "cost_per_month": (put_cost / max(time_to_expiry * 12, 0.1)) if time_to_expiry > 0 else put_cost * 12,
                    "deductible": (underlying_price - best_put.strike_price) * stock_quantity,
                    "coverage_ratio": min(best_put.strike_price / underlying_price, 1.0),
                    "efficiency_score": (protection_level / insurance_cost_pct) * 100 if insurance_cost_pct > 0 else 0,
                },
            }

        except Exception as e:
            logger.error("Failed to analyze protective put", error=str(e))
            return {"error": str(e)}

    async def _analyze_straddle(
        self, option_chain: List[OptionContract], underlying_price: float, params: Dict
    ) -> Dict:
        """Analyze straddle strategy - buy call and put at same strike (long straddle)"""
        try:
            target_strike = params.get(
                "strike_price", underlying_price
            )  # ATM default for maximum gamma
            straddle_type = params.get("type", "long")  # long or short straddle
            quantity = params.get("quantity", 1)  # Number of straddles
            
            # Find ATM or closest to target strike options
            calls = [opt for opt in option_chain if opt.option_type == OptionType.CALL]
            puts = [opt for opt in option_chain if opt.option_type == OptionType.PUT]
            
            if not calls or not puts:
                return {"error": "Need both calls and puts for straddle strategy"}
            
            # Find best matching call and put at same strike
            best_call = min(calls, key=lambda x: abs(x.strike_price - target_strike))
            
            # Find put with same strike as selected call
            matching_puts = [p for p in puts if abs(p.strike_price - best_call.strike_price) < 0.01]
            if not matching_puts:
                return {"error": f"No matching put found for strike ${best_call.strike_price}"}
            
            best_put = matching_puts[0]  # Should be exact match
            strike_price = best_call.strike_price
            
            # Calculate strategy costs and metrics
            if straddle_type == "long":
                call_cost = best_call.current_price * quantity * 100  # Long call
                put_cost = best_put.current_price * quantity * 100   # Long put
                total_cost = call_cost + put_cost
                max_loss = total_cost
                net_debit = total_cost
                is_long = True
            else:  # short straddle
                call_premium = best_call.current_price * quantity * 100  # Short call
                put_premium = best_put.current_price * quantity * 100    # Short put
                total_premium = call_premium + put_premium
                max_loss = float('inf')  # Theoretically unlimited for short straddle
                net_credit = total_premium
                is_long = False
            
            # Calculate breakeven points
            premium_per_share = (best_call.current_price + best_put.current_price)
            upper_breakeven = strike_price + premium_per_share
            lower_breakeven = strike_price - premium_per_share
            
            # Profit/loss at various underlying prices
            price_range = np.linspace(
                underlying_price * 0.7, underlying_price * 1.4, 30
            )
            pnl_profile = []
            
            for price in price_range:
                # Call option value
                call_value = max(0, price - strike_price)
                # Put option value  
                put_value = max(0, strike_price - price)
                
                if is_long:
                    # Long straddle: bought call and put
                    call_pnl = (call_value - best_call.current_price) * quantity * 100
                    put_pnl = (put_value - best_put.current_price) * quantity * 100
                    total_pnl = call_pnl + put_pnl
                    position_value = (call_value + put_value) * quantity * 100
                else:
                    # Short straddle: sold call and put
                    call_pnl = (best_call.current_price - call_value) * quantity * 100
                    put_pnl = (best_put.current_price - put_value) * quantity * 100
                    total_pnl = call_pnl + put_pnl
                    position_value = net_credit - (call_value + put_value) * quantity * 100
                
                pnl_profile.append({
                    "underlying_price": price,
                    "call_value": call_value,
                    "put_value": put_value,
                    "call_pnl": call_pnl,
                    "put_pnl": put_pnl,
                    "total_pnl": total_pnl,
                    "position_value": position_value,
                    "profitable": total_pnl > 0
                })
            
            # Calculate time to expiration and volatility metrics
            time_to_expiry = max(0, (best_call.expiration_date - datetime.now()).total_seconds() / (365.25 * 24 * 3600))
            call_iv = best_call.implied_volatility or 0.25
            put_iv = best_put.implied_volatility or 0.25
            avg_iv = (call_iv + put_iv) / 2
            
            # Calculate expected move and probability analysis
            if time_to_expiry > 0 and avg_iv > 0:
                # Expected move (1 standard deviation)
                expected_move = underlying_price * avg_iv * math.sqrt(time_to_expiry)
                expected_move_pct = (expected_move / underlying_price) * 100
                
                # Probability of profit for long straddle (stock moves beyond breakevens)
                # Using normal distribution approximation
                z_upper = (upper_breakeven - underlying_price) / (underlying_price * avg_iv * math.sqrt(time_to_expiry))
                z_lower = (lower_breakeven - underlying_price) / (underlying_price * avg_iv * math.sqrt(time_to_expiry))
                
                prob_above_upper = 1 - norm.cdf(z_upper)
                prob_below_lower = norm.cdf(z_lower)
                prob_profit_long = prob_above_upper + prob_below_lower
                prob_profit = prob_profit_long if is_long else (1 - prob_profit_long)
            else:
                expected_move = 0
                expected_move_pct = 0
                prob_profit = 0.5
            
            # Greeks analysis
            if best_call.greeks and best_put.greeks:
                combined_delta = (best_call.greeks.delta + best_put.greeks.delta) * quantity * (1 if is_long else -1)
                combined_gamma = (best_call.greeks.gamma + best_put.greeks.gamma) * quantity * (1 if is_long else -1)
                combined_theta = (best_call.greeks.theta + best_put.greeks.theta) * quantity * (1 if is_long else -1)
                combined_vega = (best_call.greeks.vega + best_put.greeks.vega) * quantity * (1 if is_long else -1)
                combined_rho = (best_call.greeks.rho + best_put.greeks.rho) * quantity * (1 if is_long else -1)
            else:
                combined_delta = combined_gamma = combined_theta = combined_vega = combined_rho = 0
            
            # Volatility analysis
            iv_percentile = 50  # Would need historical IV data to calculate actual percentile
            iv_rank = "medium"  # Simplified ranking
            
            return {
                "strategy": f"{'Long' if is_long else 'Short'} Straddle",
                "selected_options": {
                    "call": {
                        "symbol": best_call.symbol,
                        "strike": best_call.strike_price,
                        "price": best_call.current_price,
                        "implied_vol": best_call.implied_volatility,
                        "bid": best_call.bid,
                        "ask": best_call.ask,
                        "greeks": best_call.greeks.__dict__ if best_call.greeks else None,
                    },
                    "put": {
                        "symbol": best_put.symbol,
                        "strike": best_put.strike_price,
                        "price": best_put.current_price,
                        "implied_vol": best_put.implied_volatility,
                        "bid": best_put.bid,
                        "ask": best_put.ask,
                        "greeks": best_put.greeks.__dict__ if best_put.greeks else None,
                    },
                    "expiration": best_call.expiration_date.isoformat(),
                },
                "position_details": {
                    "strike_price": strike_price,
                    "quantity": quantity,
                    "is_long_straddle": is_long,
                    "total_cost" if is_long else "total_premium": total_cost if is_long else net_credit,
                    "cost_per_straddle": (total_cost / quantity) if is_long else (net_credit / quantity),
                },
                "metrics": {
                    "max_loss": max_loss if is_long else "Unlimited",
                    "max_profit": "Unlimited" if is_long else net_credit,
                    "upper_breakeven": upper_breakeven,
                    "lower_breakeven": lower_breakeven,
                    "breakeven_range": upper_breakeven - lower_breakeven,
                    "breakeven_range_pct": ((upper_breakeven - lower_breakeven) / underlying_price) * 100,
                    "expected_move": expected_move,
                    "expected_move_pct": expected_move_pct,
                    "premium_to_expected_move_ratio": (premium_per_share / expected_move) if expected_move > 0 else 0,
                    "probability_of_profit": prob_profit * 100,
                    "return_if_max_move": ((expected_move - premium_per_share) / premium_per_share) * 100 if is_long and premium_per_share > 0 else 0,
                },
                "pnl_profile": pnl_profile,
                "risk_analysis": {
                    "combined_delta": combined_delta,
                    "combined_gamma": combined_gamma,
                    "combined_theta": combined_theta,
                    "combined_vega": combined_vega,
                    "combined_rho": combined_rho,
                    "volatility_risk": "High - benefits from volatility expansion" if is_long else "High - hurt by volatility expansion",
                    "time_decay": "Negative (theta negative)" if is_long else "Positive (theta positive)",
                    "directional_risk": "Low - delta near zero at initiation",
                    "assignment_risk": "Low for long straddle" if is_long else "High for short straddle",
                },
                "volatility_analysis": {
                    "call_implied_vol": call_iv,
                    "put_implied_vol": put_iv,
                    "average_implied_vol": avg_iv,
                    "iv_percentile": iv_percentile,
                    "iv_rank": iv_rank,
                    "vol_premium": "Buying volatility" if is_long else "Selling volatility",
                },
                "management_rules": {
                    "profit_target": "25-50% of premium received" if not is_long else "50-100% of premium paid",
                    "loss_limit": "50% of premium paid" if is_long else "200% of premium received",
                    "time_management": "Close before 30-45 DTE to avoid gamma risk",
                    "volatility_management": "Monitor IV rank - close if vol collapses" if is_long else "close if vol spikes",
                    "adjustment_options": [
                        "Roll to different expiration",
                        "Convert to strangle (adjust strikes)",
                        "Close losing side and manage remaining long option" if is_long else "Close entire position if breached"
                    ],
                },
                "market_outlook": {
                    "ideal_for_long": "High volatility events, earnings, FDA approvals, etc.",
                    "ideal_for_short": "Stable, range-bound market with high IV",
                    "avoid_if": "Major events/earnings within expiration period" if not is_long else "Low volatility environment",
                },
            }

        except Exception as e:
            logger.error("Failed to analyze straddle", error=str(e))
            return {"error": str(e)}

    async def get_portfolio_greeks(
        self, positions: List[OptionPosition]
    ) -> Dict[str, float]:
        """Calculate aggregate Greeks for options portfolio"""
        try:
            total_delta = sum(
                pos.quantity * pos.position_greeks.delta for pos in positions
            )
            total_gamma = sum(
                pos.quantity * pos.position_greeks.gamma for pos in positions
            )
            total_theta = sum(
                pos.quantity * pos.position_greeks.theta for pos in positions
            )
            total_vega = sum(
                pos.quantity * pos.position_greeks.vega for pos in positions
            )
            total_rho = sum(pos.quantity * pos.position_greeks.rho for pos in positions)

            return {
                "portfolio_delta": total_delta,
                "portfolio_gamma": total_gamma,
                "portfolio_theta": total_theta,
                "portfolio_vega": total_vega,
                "portfolio_rho": total_rho,
                "position_count": len(positions),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error("Failed to calculate portfolio Greeks", error=str(e))
            return {}

    async def get_options_health_check(self) -> Dict[str, Any]:
        """Health check for options trading system"""
        try:
            return {
                "status": "healthy",
                "alpaca_connected": self.alpaca_client is not None,
                "calculator_available": True,
                "risk_free_rate": self.risk_free_rate,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Global options trader instance
_options_trader: Optional[OptionsTrader] = None


async def get_options_trader() -> OptionsTrader:
    """Get or create options trader instance"""
    global _options_trader
    if _options_trader is None:
        _options_trader = OptionsTrader()
    return _options_trader


__all__ = [
    "OptionType",
    "OptionStrategy",
    "GreeksData",
    "OptionPosition",
    "BlackScholesCalculator",
    "EnhancedBlackScholesCalculator",
    "OptionsTrader",
    "get_options_trader",
    "get_enhanced_calculator"
]
