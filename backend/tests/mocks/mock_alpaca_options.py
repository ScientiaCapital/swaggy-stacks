"""
Mock Alpaca Options Data Generator

Generates realistic options chain data with proper Greeks calculations,
volatility smiles, and market scenarios for comprehensive testing.

Features:
- Realistic option pricing using Black-Scholes model
- Proper Greeks calculations (delta, gamma, theta, vega, rho)
- Volatility smile patterns
- Multiple expiration dates
- Various strike ranges and moneyness levels
- Edge cases and extreme scenarios
"""

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from decimal import Decimal
import numpy as np
from scipy.stats import norm


class MockAlpacaOptionsGenerator:
    """Generates realistic mock options data for testing"""

    def __init__(self, seed: int = 42):
        """Initialize with reproducible seed for deterministic testing"""
        random.seed(seed)
        np.random.seed(seed)

    def generate_option_chain(
        self,
        symbol: str = "AAPL",
        underlying_price: float = 150.0,
        expiry_days: List[int] = None,
        strike_range: float = 0.2,  # 20% around current price
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.01,
        base_volatility: float = 0.25,
        volatility_smile: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate comprehensive options chain with realistic pricing

        Args:
            symbol: Underlying symbol
            underlying_price: Current stock price
            expiry_days: List of days to expiration [7, 14, 30, 60]
            strike_range: Strike range as percentage of underlying price
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
            base_volatility: Base implied volatility
            volatility_smile: Whether to apply volatility smile

        Returns:
            List of option contracts with realistic pricing and Greeks
        """
        if expiry_days is None:
            expiry_days = [7, 14, 30, 60, 90]

        option_chain = []
        current_time = datetime.now(timezone.utc)

        # Generate strikes around current price
        strike_min = underlying_price * (1 - strike_range)
        strike_max = underlying_price * (1 + strike_range)
        strikes = self._generate_strike_ladder(strike_min, strike_max, underlying_price)

        for days_to_expiry in expiry_days:
            expiry_date = current_time + timedelta(days=days_to_expiry)
            time_to_expiry = days_to_expiry / 365.0

            for strike in strikes:
                moneyness = strike / underlying_price

                # Calculate implied volatility with smile
                if volatility_smile:
                    iv = self._calculate_implied_volatility_smile(
                        moneyness, time_to_expiry, base_volatility
                    )
                else:
                    iv = base_volatility

                # Generate both call and put options
                for option_type in ['call', 'put']:
                    option_data = self._generate_option_contract(
                        symbol=symbol,
                        strike=strike,
                        expiry_date=expiry_date,
                        option_type=option_type,
                        underlying_price=underlying_price,
                        volatility=iv,
                        risk_free_rate=risk_free_rate,
                        dividend_yield=dividend_yield,
                        time_to_expiry=time_to_expiry
                    )
                    option_chain.append(option_data)

        return option_chain

    def _generate_strike_ladder(
        self,
        strike_min: float,
        strike_max: float,
        underlying_price: float
    ) -> List[float]:
        """Generate realistic strike ladder with appropriate spacing"""
        strikes = []

        # Strike spacing based on price level
        if underlying_price < 25:
            spacing = 1.0
        elif underlying_price < 100:
            spacing = 2.5
        elif underlying_price < 200:
            spacing = 5.0
        else:
            spacing = 10.0

        # Generate strikes with proper spacing
        strike = math.floor(strike_min / spacing) * spacing
        while strike <= strike_max:
            if strike > 0:  # Ensure positive strikes
                strikes.append(strike)
            strike += spacing

        return strikes

    def _calculate_implied_volatility_smile(
        self,
        moneyness: float,
        time_to_expiry: float,
        base_vol: float
    ) -> float:
        """Calculate implied volatility with realistic smile pattern"""
        # Volatility smile: higher vol for OTM options
        smile_factor = 0.1 * (moneyness - 1.0) ** 2

        # Term structure: slightly higher vol for longer expirations
        term_factor = 0.05 * math.sqrt(time_to_expiry)

        # Add some random noise
        noise = random.gauss(0, 0.02)

        implied_vol = base_vol + smile_factor + term_factor + noise

        # Keep within reasonable bounds
        return max(0.05, min(2.0, implied_vol))

    def _generate_option_contract(
        self,
        symbol: str,
        strike: float,
        expiry_date: datetime,
        option_type: str,
        underlying_price: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float,
        time_to_expiry: float
    ) -> Dict[str, Any]:
        """Generate single option contract with Black-Scholes pricing"""

        # Calculate option price and Greeks using Black-Scholes
        pricing = self._black_scholes_pricing(
            underlying_price, strike, time_to_expiry, risk_free_rate,
            dividend_yield, volatility, option_type
        )

        # Generate option symbol
        expiry_str = expiry_date.strftime("%y%m%d")
        option_symbol = f"{symbol}{expiry_str}{'C' if option_type == 'call' else 'P'}{int(strike * 1000):08d}"

        # Calculate bid-ask spread (wider for illiquid options)
        spread_factor = self._calculate_spread_factor(strike, underlying_price, time_to_expiry)
        bid_price = max(0.01, pricing['price'] * (1 - spread_factor))
        ask_price = pricing['price'] * (1 + spread_factor)

        # Generate volume and open interest
        volume = self._generate_realistic_volume(strike, underlying_price, time_to_expiry)
        open_interest = self._generate_realistic_open_interest(volume)

        return {
            'symbol': option_symbol,
            'underlying_symbol': symbol,
            'strike_price': strike,
            'expiry_date': expiry_date,
            'option_type': option_type,
            'last_price': round(pricing['price'], 2),
            'bid': round(bid_price, 2),
            'ask': round(ask_price, 2),
            'bid_size': random.randint(1, 50) * 10,
            'ask_size': random.randint(1, 50) * 10,
            'volume': volume,
            'open_interest': open_interest,
            'implied_volatility': round(volatility, 4),
            'delta': round(pricing['delta'], 4),
            'gamma': round(pricing['gamma'], 4),
            'theta': round(pricing['theta'], 4),
            'vega': round(pricing['vega'], 4),
            'rho': round(pricing['rho'], 4),
            'intrinsic_value': max(0, underlying_price - strike if option_type == 'call' else strike - underlying_price),
            'time_value': max(0, pricing['price'] - max(0, underlying_price - strike if option_type == 'call' else strike - underlying_price)),
            'moneyness': strike / underlying_price,
            'time_to_expiry': time_to_expiry,
            'updated_at': datetime.now(timezone.utc)
        }

    def _black_scholes_pricing(
        self,
        S: float,  # Current stock price
        K: float,  # Strike price
        T: float,  # Time to expiration
        r: float,  # Risk-free rate
        q: float,  # Dividend yield
        sigma: float,  # Volatility
        option_type: str
    ) -> Dict[str, float]:
        """Calculate Black-Scholes option price and Greeks"""

        if T <= 0:
            # Handle expiration case
            intrinsic = max(0, S - K if option_type == 'call' else K - S)
            return {
                'price': intrinsic,
                'delta': 1.0 if (option_type == 'call' and S > K) or (option_type == 'put' and S < K) else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }

        # Standard Black-Scholes calculations
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == 'call':
            price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            delta = math.exp(-q * T) * norm.cdf(d1)
            rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
        else:  # put
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
            delta = -math.exp(-q * T) * norm.cdf(-d1)
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

        # Greeks (common for both calls and puts)
        gamma = math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma * math.exp(-q * T) / (2 * math.sqrt(T))
                - r * K * math.exp(-r * T) * (norm.cdf(d2) if option_type == 'call' else norm.cdf(-d2))
                + q * S * math.exp(-q * T) * (norm.cdf(d1) if option_type == 'call' else norm.cdf(-d1))) / 365
        vega = S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T) / 100

        return {
            'price': max(0.01, price),  # Minimum price of $0.01
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    def _calculate_spread_factor(self, strike: float, underlying_price: float, time_to_expiry: float) -> float:
        """Calculate bid-ask spread factor based on liquidity"""
        # Base spread
        base_spread = 0.02  # 2%

        # Wider spreads for OTM options
        moneyness = abs(strike / underlying_price - 1.0)
        moneyness_penalty = moneyness * 0.1

        # Wider spreads for short-dated options
        time_penalty = max(0, (0.1 - time_to_expiry) * 0.5)

        total_spread = base_spread + moneyness_penalty + time_penalty
        return min(0.2, total_spread)  # Cap at 20%

    def _generate_realistic_volume(self, strike: float, underlying_price: float, time_to_expiry: float) -> int:
        """Generate realistic trading volume"""
        # Higher volume for ATM options
        moneyness = abs(strike / underlying_price - 1.0)
        atm_factor = max(0.1, 1.0 - moneyness * 5)

        # Higher volume for near-term options
        time_factor = max(0.2, math.exp(-time_to_expiry * 2))

        # Base volume with randomness
        base_volume = random.randint(50, 500)
        volume = int(base_volume * atm_factor * time_factor)

        return max(0, volume)

    def _generate_realistic_open_interest(self, volume: int) -> int:
        """Generate realistic open interest based on volume"""
        # Open interest is typically 2-10x daily volume
        multiplier = random.uniform(2, 10)
        return int(volume * multiplier)

    def generate_market_scenario_chain(
        self,
        scenario: str,
        symbol: str = "SPY",
        base_price: float = 450.0
    ) -> List[Dict[str, Any]]:
        """
        Generate options chain for specific market scenarios

        Args:
            scenario: 'bull_market', 'bear_market', 'high_vol', 'low_vol', 'crisis'
            symbol: Underlying symbol
            base_price: Base underlying price

        Returns:
            Options chain tailored for the market scenario
        """
        scenario_params = self._get_scenario_parameters(scenario)

        return self.generate_option_chain(
            symbol=symbol,
            underlying_price=base_price,
            expiry_days=scenario_params['expiry_days'],
            base_volatility=scenario_params['volatility'],
            volatility_smile=scenario_params['volatility_smile']
        )

    def _get_scenario_parameters(self, scenario: str) -> Dict[str, Any]:
        """Get parameters for different market scenarios"""
        scenarios = {
            'bull_market': {
                'volatility': 0.18,
                'expiry_days': [7, 14, 30, 60],
                'volatility_smile': True
            },
            'bear_market': {
                'volatility': 0.35,
                'expiry_days': [7, 14, 30],
                'volatility_smile': True
            },
            'high_vol': {
                'volatility': 0.45,
                'expiry_days': [7, 14, 21, 30],
                'volatility_smile': True
            },
            'low_vol': {
                'volatility': 0.12,
                'expiry_days': [14, 30, 60, 90],
                'volatility_smile': False
            },
            'crisis': {
                'volatility': 0.80,
                'expiry_days': [1, 3, 7, 14],
                'volatility_smile': True
            }
        }

        return scenarios.get(scenario, scenarios['bull_market'])

    def generate_earnings_event_chain(
        self,
        symbol: str = "AAPL",
        underlying_price: float = 150.0,
        earnings_date: datetime = None
    ) -> List[Dict[str, Any]]:
        """Generate options chain around earnings event"""
        if earnings_date is None:
            earnings_date = datetime.now(timezone.utc) + timedelta(days=3)

        # Higher implied volatility before earnings
        pre_earnings_vol = 0.45
        post_earnings_vol = 0.25

        current_time = datetime.now(timezone.utc)
        days_to_earnings = (earnings_date - current_time).days

        option_chain = []

        # Generate options with earnings volatility crush
        for days_offset in [-7, -3, -1, 1, 7, 14]:
            expiry_date = earnings_date + timedelta(days=days_offset)

            if expiry_date < current_time:
                continue

            # Volatility depends on position relative to earnings
            if expiry_date < earnings_date:
                vol = pre_earnings_vol
            else:
                vol = post_earnings_vol

            time_to_expiry = (expiry_date - current_time).days / 365.0

            strikes = self._generate_strike_ladder(
                underlying_price * 0.85,
                underlying_price * 1.15,
                underlying_price
            )

            for strike in strikes:
                for option_type in ['call', 'put']:
                    option_data = self._generate_option_contract(
                        symbol=symbol,
                        strike=strike,
                        expiry_date=expiry_date,
                        option_type=option_type,
                        underlying_price=underlying_price,
                        volatility=vol,
                        risk_free_rate=0.05,
                        dividend_yield=0.01,
                        time_to_expiry=time_to_expiry
                    )
                    option_chain.append(option_data)

        return option_chain

    def generate_edge_case_options(self) -> List[Dict[str, Any]]:
        """Generate edge case options for robust testing"""
        edge_cases = []

        # Deep ITM call
        edge_cases.append(self._generate_option_contract(
            symbol="TEST",
            strike=50.0,
            expiry_date=datetime.now(timezone.utc) + timedelta(days=30),
            option_type="call",
            underlying_price=100.0,
            volatility=0.25,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            time_to_expiry=30/365
        ))

        # Deep OTM put
        edge_cases.append(self._generate_option_contract(
            symbol="TEST",
            strike=50.0,
            expiry_date=datetime.now(timezone.utc) + timedelta(days=30),
            option_type="put",
            underlying_price=100.0,
            volatility=0.25,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            time_to_expiry=30/365
        ))

        # Near expiration option
        edge_cases.append(self._generate_option_contract(
            symbol="TEST",
            strike=100.0,
            expiry_date=datetime.now(timezone.utc) + timedelta(hours=4),
            option_type="call",
            underlying_price=100.0,
            volatility=0.25,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            time_to_expiry=4/24/365
        ))

        # Very high volatility option
        edge_cases.append(self._generate_option_contract(
            symbol="TEST",
            strike=100.0,
            expiry_date=datetime.now(timezone.utc) + timedelta(days=30),
            option_type="call",
            underlying_price=100.0,
            volatility=1.5,  # 150% volatility
            risk_free_rate=0.05,
            dividend_yield=0.0,
            time_to_expiry=30/365
        ))

        return edge_cases


# Global generator instance for consistent testing
mock_options_generator = MockAlpacaOptionsGenerator()


def get_mock_option_chain(scenario: str = "normal", **kwargs) -> List[Dict[str, Any]]:
    """Convenience function to get mock option chain"""
    if scenario == "normal":
        return mock_options_generator.generate_option_chain(**kwargs)
    elif scenario in ["bull_market", "bear_market", "high_vol", "low_vol", "crisis"]:
        return mock_options_generator.generate_market_scenario_chain(scenario, **kwargs)
    elif scenario == "earnings":
        return mock_options_generator.generate_earnings_event_chain(**kwargs)
    elif scenario == "edge_cases":
        return mock_options_generator.generate_edge_case_options()
    else:
        return mock_options_generator.generate_option_chain(**kwargs)


def get_mock_single_option(
    symbol: str = "AAPL",
    strike: float = 150.0,
    option_type: str = "call",
    days_to_expiry: int = 30,
    **kwargs
) -> Dict[str, Any]:
    """Get a single mock option for simple testing"""
    chain = mock_options_generator.generate_option_chain(
        symbol=symbol,
        underlying_price=kwargs.get('underlying_price', strike),
        expiry_days=[days_to_expiry],
        **kwargs
    )

    # Find the matching option
    for option in chain:
        if (option['strike_price'] == strike and
            option['option_type'] == option_type):
            return option

    # Return first option if exact match not found
    return chain[0] if chain else {}


__all__ = [
    'MockAlpacaOptionsGenerator',
    'mock_options_generator',
    'get_mock_option_chain',
    'get_mock_single_option'
]