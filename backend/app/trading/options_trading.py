"""
Options trading implementation with Greeks calculations for Alpaca API.
Provides options strategies, risk analysis, and Greeks computation for paper trading.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.stats import norm
import structlog

from app.trading.alpaca_client import AlpacaClient
from app.core.cache import get_market_cache

logger = structlog.get_logger(__name__)


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
    delta: float  # Price sensitivity to underlying
    gamma: float  # Delta sensitivity to underlying
    theta: float  # Time decay
    vega: float   # Volatility sensitivity
    rho: float    # Interest rate sensitivity
    
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
        dividend_yield: float = 0.0
    ) -> float:
        """Calculate option price using Black-Scholes model"""
        try:
            d1 = BlackScholesCalculator._calculate_d1(
                underlying_price, strike_price, time_to_expiry, 
                risk_free_rate, volatility, dividend_yield
            )
            d2 = d1 - volatility * math.sqrt(time_to_expiry)
            
            if option_type == OptionType.CALL:
                price = (
                    underlying_price * math.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1) -
                    strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
                )
            else:  # PUT
                price = (
                    strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) -
                    underlying_price * math.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
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
        dividend_yield: float = 0.0
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
                    time_value=0.0
                )
            
            d1 = BlackScholesCalculator._calculate_d1(
                underlying_price, strike_price, time_to_expiry,
                risk_free_rate, volatility, dividend_yield
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
                underlying_price, strike_price, time_to_expiry, risk_free_rate,
                volatility, d1, d2, option_type, dividend_yield
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
                underlying_price, strike_price, time_to_expiry,
                risk_free_rate, volatility, option_type, dividend_yield
            )
            
            time_value = max(0, option_price - intrinsic_value)
            
            return GreeksData(
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                intrinsic_value=intrinsic_value,
                time_value=time_value
            )
            
        except Exception as e:
            logger.error("Failed to calculate Greeks", error=str(e))
            return GreeksData(0, 0, 0, 0, 0, 0, 0)
    
    @staticmethod
    def _calculate_d1(underlying_price, strike_price, time_to_expiry, 
                     risk_free_rate, volatility, dividend_yield):
        """Calculate d1 parameter for Black-Scholes"""
        return (
            (math.log(underlying_price / strike_price) + 
             (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) /
            (volatility * math.sqrt(time_to_expiry))
        )
    
    @staticmethod
    def _calculate_delta(d1, option_type, time_to_expiry, dividend_yield):
        """Calculate delta"""
        if option_type == OptionType.CALL:
            return math.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)
        else:
            return -math.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
    
    @staticmethod
    def _calculate_gamma(underlying_price, d1, time_to_expiry, volatility, dividend_yield):
        """Calculate gamma"""
        return (
            (math.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1)) /
            (underlying_price * volatility * math.sqrt(time_to_expiry))
        )
    
    @staticmethod
    def _calculate_theta(underlying_price, strike_price, time_to_expiry, risk_free_rate,
                        volatility, d1, d2, option_type, dividend_yield):
        """Calculate theta (time decay)"""
        common_term = (
            -underlying_price * norm.pdf(d1) * volatility * math.exp(-dividend_yield * time_to_expiry) /
            (2 * math.sqrt(time_to_expiry))
        )
        
        if option_type == OptionType.CALL:
            theta = (
                common_term -
                risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) +
                dividend_yield * underlying_price * math.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)
            )
        else:
            theta = (
                common_term +
                risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) -
                dividend_yield * underlying_price * math.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
            )
        
        return theta / 365  # Convert to daily theta
    
    @staticmethod
    def _calculate_vega(underlying_price, d1, time_to_expiry, dividend_yield):
        """Calculate vega"""
        return (
            underlying_price * math.exp(-dividend_yield * time_to_expiry) *
            norm.pdf(d1) * math.sqrt(time_to_expiry) / 100  # Divide by 100 for 1% volatility change
        )
    
    @staticmethod
    def _calculate_rho(strike_price, time_to_expiry, risk_free_rate, d2, option_type):
        """Calculate rho"""
        if option_type == OptionType.CALL:
            return (
                strike_price * time_to_expiry *
                math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100
            )
        else:
            return (
                -strike_price * time_to_expiry *
                math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100
            )


class OptionsTrader:
    """Options trading manager with Greeks calculation"""
    
    def __init__(self, alpaca_client: Optional[AlpacaClient] = None):
        self.alpaca_client = alpaca_client or AlpacaClient()
        self.market_cache = get_market_cache()
        self.risk_free_rate = 0.05  # 5% annual risk-free rate (adjustable)
        self.calculator = BlackScholesCalculator()
        
    async def get_option_chain(self, symbol: str, expiration_date: Optional[str] = None) -> List[OptionContract]:
        """Get option chain for a symbol from Alpaca"""
        try:
            # Get options data from Alpaca
            options_data = await self.alpaca_client.get_option_chain(symbol, expiration_date)
            
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
                    contract = await self._parse_option_contract(option_data, underlying_price['price'])
                    
                    if contract:
                        # Calculate Greeks
                        contract.greeks = await self._calculate_option_greeks(contract, underlying_price['price'])
                        option_contracts.append(contract)
                        
                except Exception as e:
                    logger.error(f"Failed to process option contract", error=str(e))
                    continue
            
            logger.info(f"Retrieved {len(option_contracts)} option contracts for {symbol}")
            return option_contracts
            
        except Exception as e:
            logger.error(f"Failed to get option chain for {symbol}", error=str(e))
            return []
    
    async def _parse_option_contract(self, option_data: Dict, underlying_price: float) -> Optional[OptionContract]:
        """Parse option contract data from Alpaca"""
        try:
            symbol = option_data.get('symbol', '')
            underlying_symbol = option_data.get('underlying_symbol', '')
            
            # Parse option type from symbol or data
            option_type_str = option_data.get('type', 'call').lower()
            option_type = OptionType.CALL if option_type_str == 'call' else OptionType.PUT
            
            strike_price = float(option_data.get('strike_price', 0))
            
            # Parse expiration date
            exp_date_str = option_data.get('expiration_date', '')
            expiration_date = datetime.fromisoformat(exp_date_str) if exp_date_str else datetime.now()
            
            current_price = float(option_data.get('last_price', 0))
            bid = float(option_data.get('bid', 0))
            ask = float(option_data.get('ask', 0))
            volume = int(option_data.get('volume', 0))
            open_interest = int(option_data.get('open_interest', 0))
            implied_volatility = option_data.get('implied_volatility')
            
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
                implied_volatility=implied_volatility
            )
            
        except Exception as e:
            logger.error("Failed to parse option contract", error=str(e))
            return None
    
    async def _calculate_option_greeks(self, contract: OptionContract, underlying_price: float) -> GreeksData:
        """Calculate Greeks for an option contract"""
        try:
            # Calculate time to expiry in years
            time_to_expiry = max(0, (contract.expiration_date - datetime.now()).total_seconds() / (365.25 * 24 * 3600))
            
            # Use implied volatility if available, otherwise estimate
            volatility = contract.implied_volatility or await self._estimate_volatility(contract.underlying_symbol)
            
            return self.calculator.calculate_greeks(
                underlying_price=underlying_price,
                strike_price=contract.strike_price,
                time_to_expiry=time_to_expiry,
                risk_free_rate=self.risk_free_rate,
                volatility=volatility,
                option_type=contract.option_type
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
                logger.warning(f"Insufficient data to calculate volatility for {symbol}")
                return 0.2  # Default 20% volatility
            
            # Calculate daily returns
            prices = [float(day['close']) for day in historical_data]
            returns = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
            
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
        self, 
        strategy: OptionStrategy, 
        symbol: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze an options strategy"""
        try:
            logger.info(f"Analyzing {strategy.value} strategy for {symbol}")
            
            # Get underlying price
            underlying_data = await self.alpaca_client.get_latest_price(symbol)
            if not underlying_data:
                raise ValueError(f"Could not get price for {symbol}")
            
            underlying_price = underlying_data['price']
            
            # Get option chain
            option_chain = await self.get_option_chain(symbol)
            if not option_chain:
                raise ValueError(f"No options available for {symbol}")
            
            # Strategy-specific analysis
            if strategy == OptionStrategy.LONG_CALL:
                return await self._analyze_long_call(option_chain, underlying_price, parameters)
            elif strategy == OptionStrategy.COVERED_CALL:
                return await self._analyze_covered_call(option_chain, underlying_price, parameters)
            elif strategy == OptionStrategy.PROTECTIVE_PUT:
                return await self._analyze_protective_put(option_chain, underlying_price, parameters)
            elif strategy == OptionStrategy.STRADDLE:
                return await self._analyze_straddle(option_chain, underlying_price, parameters)
            else:
                return {"error": f"Strategy {strategy.value} not implemented yet"}
                
        except Exception as e:
            logger.error(f"Failed to analyze {strategy.value} strategy", error=str(e))
            return {"error": str(e)}
    
    async def _analyze_long_call(self, option_chain: List[OptionContract], underlying_price: float, params: Dict) -> Dict:
        """Analyze long call strategy"""
        try:
            target_strike = params.get('strike_price', underlying_price * 1.05)  # 5% OTM default
            
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
            price_range = np.linspace(underlying_price * 0.8, underlying_price * 1.3, 21)
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
                    "greeks": best_call.greeks.__dict__ if best_call.greeks else None
                },
                "metrics": {
                    "max_loss": max_loss,
                    "max_profit": "Unlimited",
                    "breakeven": breakeven,
                    "cost": best_call.current_price
                },
                "pnl_profile": pnl_profile,
                "risk_analysis": {
                    "delta_exposure": best_call.greeks.delta if best_call.greeks else 0,
                    "theta_decay": best_call.greeks.theta if best_call.greeks else 0,
                    "vega_risk": best_call.greeks.vega if best_call.greeks else 0
                }
            }
            
        except Exception as e:
            logger.error("Failed to analyze long call", error=str(e))
            return {"error": str(e)}
    
    async def _analyze_covered_call(self, option_chain: List[OptionContract], underlying_price: float, params: Dict) -> Dict:
        """Analyze covered call strategy"""
        # Implementation for covered call analysis
        return {"strategy": "Covered Call", "status": "Analysis pending"}
    
    async def _analyze_protective_put(self, option_chain: List[OptionContract], underlying_price: float, params: Dict) -> Dict:
        """Analyze protective put strategy"""
        # Implementation for protective put analysis
        return {"strategy": "Protective Put", "status": "Analysis pending"}
    
    async def _analyze_straddle(self, option_chain: List[OptionContract], underlying_price: float, params: Dict) -> Dict:
        """Analyze straddle strategy"""
        # Implementation for straddle analysis
        return {"strategy": "Straddle", "status": "Analysis pending"}
    
    async def get_portfolio_greeks(self, positions: List[OptionPosition]) -> Dict[str, float]:
        """Calculate aggregate Greeks for options portfolio"""
        try:
            total_delta = sum(pos.quantity * pos.position_greeks.delta for pos in positions)
            total_gamma = sum(pos.quantity * pos.position_greeks.gamma for pos in positions)
            total_theta = sum(pos.quantity * pos.position_greeks.theta for pos in positions)
            total_vega = sum(pos.quantity * pos.position_greeks.vega for pos in positions)
            total_rho = sum(pos.quantity * pos.position_greeks.rho for pos in positions)
            
            return {
                "portfolio_delta": total_delta,
                "portfolio_gamma": total_gamma,
                "portfolio_theta": total_theta,
                "portfolio_vega": total_vega,
                "portfolio_rho": total_rho,
                "position_count": len(positions),
                "timestamp": datetime.now().isoformat()
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
                "timestamp": datetime.now().isoformat()
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