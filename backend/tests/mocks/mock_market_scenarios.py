"""
Mock Market Scenarios Generator

Creates realistic market data scenarios for testing different trading conditions:
- Bull markets, bear markets, sideways markets
- High volatility and low volatility periods
- Crisis scenarios and earnings events
- Trend reversals and regime changes

Used for comprehensive testing of options strategies across all market conditions.
"""

import random
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Tuple
import numpy as np


class MockMarketScenariosGenerator:
    """Generates realistic market scenario data for testing"""

    def __init__(self, seed: int = 42):
        """Initialize with reproducible seed"""
        random.seed(seed)
        np.random.seed(seed)

    def generate_bull_market_scenario(
        self,
        symbol: str = "AAPL",
        start_price: float = 140.0,
        duration_days: int = 60,
        daily_drift: float = 0.0008,  # ~20% annual return
        volatility: float = 0.18
    ) -> List[Dict[str, Any]]:
        """Generate bull market price data with consistent upward trend"""

        price_data = []
        current_price = start_price
        start_date = datetime.now(timezone.utc) - timedelta(days=duration_days)

        for i in range(duration_days):
            date = start_date + timedelta(days=i)

            # Bull market: positive drift with normal volatility
            daily_return = daily_drift + random.gauss(0, volatility / math.sqrt(252))
            current_price *= (1 + daily_return)

            # Generate OHLC data
            open_price = current_price * random.uniform(0.995, 1.005)
            high_price = max(open_price, current_price) * random.uniform(1.0, 1.02)
            low_price = min(open_price, current_price) * random.uniform(0.98, 1.0)

            volume = random.randint(2000000, 8000000)  # Higher volume in bull markets

            price_data.append({
                'symbol': symbol,
                'timestamp': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(current_price, 2),
                'volume': volume,
                'market_regime': 'bull_market',
                'volatility_regime': 'low' if volatility < 0.20 else 'normal'
            })

        return price_data

    def generate_bear_market_scenario(
        self,
        symbol: str = "SPY",
        start_price: float = 450.0,
        duration_days: int = 45,
        daily_drift: float = -0.002,  # ~-40% annual decline
        volatility: float = 0.35
    ) -> List[Dict[str, Any]]:
        """Generate bear market with declining prices and higher volatility"""

        price_data = []
        current_price = start_price
        start_date = datetime.now(timezone.utc) - timedelta(days=duration_days)

        for i in range(duration_days):
            date = start_date + timedelta(days=i)

            # Bear market: negative drift with higher volatility
            daily_return = daily_drift + random.gauss(0, volatility / math.sqrt(252))
            current_price *= (1 + daily_return)

            # More dramatic intraday moves in bear markets
            open_price = current_price * random.uniform(0.99, 1.01)
            high_price = max(open_price, current_price) * random.uniform(1.0, 1.03)
            low_price = min(open_price, current_price) * random.uniform(0.95, 1.0)

            volume = random.randint(3000000, 12000000)  # Higher volume in bear markets

            price_data.append({
                'symbol': symbol,
                'timestamp': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(current_price, 2),
                'volume': volume,
                'market_regime': 'bear_market',
                'volatility_regime': 'high'
            })

        return price_data

    def generate_sideways_market_scenario(
        self,
        symbol: str = "QQQ",
        center_price: float = 380.0,
        duration_days: int = 90,
        range_percentage: float = 0.05,  # 5% range
        volatility: float = 0.15
    ) -> List[Dict[str, Any]]:
        """Generate sideways/neutral market with range-bound trading"""

        price_data = []
        start_date = datetime.now(timezone.utc) - timedelta(days=duration_days)

        # Range bounds
        upper_bound = center_price * (1 + range_percentage)
        lower_bound = center_price * (1 - range_percentage)

        current_price = center_price

        for i in range(duration_days):
            date = start_date + timedelta(days=i)

            # Mean-reverting process for sideways movement
            distance_from_center = (current_price - center_price) / center_price
            mean_reversion = -distance_from_center * 0.1  # Pull back to center

            daily_return = mean_reversion + random.gauss(0, volatility / math.sqrt(252))
            new_price = current_price * (1 + daily_return)

            # Keep within range bounds
            current_price = max(lower_bound, min(upper_bound, new_price))

            open_price = current_price * random.uniform(0.998, 1.002)
            high_price = max(open_price, current_price) * random.uniform(1.0, 1.015)
            low_price = min(open_price, current_price) * random.uniform(0.985, 1.0)

            volume = random.randint(1500000, 5000000)  # Lower volume in sideways markets

            price_data.append({
                'symbol': symbol,
                'timestamp': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(current_price, 2),
                'volume': volume,
                'market_regime': 'sideways',
                'volatility_regime': 'low'
            })

        return price_data

    def generate_high_volatility_scenario(
        self,
        symbol: str = "TSLA",
        start_price: float = 200.0,
        duration_days: int = 30,
        volatility: float = 0.55
    ) -> List[Dict[str, Any]]:
        """Generate high volatility market scenario"""

        price_data = []
        current_price = start_price
        start_date = datetime.now(timezone.utc) - timedelta(days=duration_days)

        for i in range(duration_days):
            date = start_date + timedelta(days=i)

            # High volatility with no clear trend
            daily_return = random.gauss(0, volatility / math.sqrt(252))
            current_price *= (1 + daily_return)

            # Extreme intraday moves
            open_price = current_price * random.uniform(0.95, 1.05)
            high_price = max(open_price, current_price) * random.uniform(1.0, 1.08)
            low_price = min(open_price, current_price) * random.uniform(0.92, 1.0)

            volume = random.randint(5000000, 15000000)  # Very high volume

            price_data.append({
                'symbol': symbol,
                'timestamp': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(current_price, 2),
                'volume': volume,
                'market_regime': 'high_volatility',
                'volatility_regime': 'extreme'
            })

        return price_data

    def generate_crisis_scenario(
        self,
        symbol: str = "SPY",
        start_price: float = 420.0,
        duration_days: int = 20,
        crash_magnitude: float = -0.25  # 25% decline
    ) -> List[Dict[str, Any]]:
        """Generate market crisis scenario with sharp decline"""

        price_data = []
        current_price = start_price
        start_date = datetime.now(timezone.utc) - timedelta(days=duration_days)

        # Calculate daily decline to reach target crash magnitude
        daily_decline = (1 + crash_magnitude) ** (1/duration_days) - 1

        for i in range(duration_days):
            date = start_date + timedelta(days=i)

            # Crisis: consistent decline with extreme volatility
            base_return = daily_decline
            volatility_component = random.gauss(0, 0.8 / math.sqrt(252))  # 80% annual vol
            daily_return = base_return + volatility_component

            current_price *= (1 + daily_return)

            # Extreme intraday ranges during crisis
            open_price = current_price * random.uniform(0.9, 1.1)
            high_price = max(open_price, current_price) * random.uniform(1.0, 1.15)
            low_price = min(open_price, current_price) * random.uniform(0.85, 1.0)

            volume = random.randint(8000000, 20000000)  # Panic selling volume

            price_data.append({
                'symbol': symbol,
                'timestamp': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(current_price, 2),
                'volume': volume,
                'market_regime': 'crisis',
                'volatility_regime': 'extreme'
            })

        return price_data

    def generate_earnings_event_scenario(
        self,
        symbol: str = "AAPL",
        start_price: float = 150.0,
        earnings_day: int = 15,  # Day of earnings in sequence
        duration_days: int = 30,
        earnings_surprise: str = "positive"  # "positive", "negative", "neutral"
    ) -> List[Dict[str, Any]]:
        """Generate price action around earnings event"""

        price_data = []
        current_price = start_price
        start_date = datetime.now(timezone.utc) - timedelta(days=duration_days)

        for i in range(duration_days):
            date = start_date + timedelta(days=i)

            if i < earnings_day:
                # Pre-earnings: building anticipation, moderate volatility
                daily_return = random.gauss(0.0005, 0.02)  # Slight bullish bias
                volume_multiplier = 1.0 + (earnings_day - i) * 0.1  # Increasing volume

            elif i == earnings_day:
                # Earnings day: big move based on surprise
                if earnings_surprise == "positive":
                    daily_return = random.gauss(0.08, 0.03)  # 8% average jump
                elif earnings_surprise == "negative":
                    daily_return = random.gauss(-0.06, 0.03)  # 6% average drop
                else:  # neutral
                    daily_return = random.gauss(0.01, 0.04)  # Small move, high vol

                volume_multiplier = 3.0  # 3x normal volume

            else:
                # Post-earnings: volatility crush, trending move
                base_trend = 0.002 if earnings_surprise == "positive" else -0.001
                daily_return = random.gauss(base_trend, 0.015)  # Lower vol after earnings
                volume_multiplier = max(0.8, 2.0 - (i - earnings_day) * 0.1)

            current_price *= (1 + daily_return)

            # Generate OHLC with appropriate volatility
            vol_factor = 0.02 if i != earnings_day else 0.05
            open_price = current_price * random.uniform(1-vol_factor, 1+vol_factor)
            high_price = max(open_price, current_price) * random.uniform(1.0, 1+vol_factor*2)
            low_price = min(open_price, current_price) * random.uniform(1-vol_factor*2, 1.0)

            base_volume = random.randint(2000000, 5000000)
            volume = int(base_volume * volume_multiplier)

            # Determine regime
            if i < earnings_day:
                regime = "pre_earnings"
                vol_regime = "elevated"
            elif i == earnings_day:
                regime = "earnings_day"
                vol_regime = "extreme"
            else:
                regime = "post_earnings"
                vol_regime = "normalizing"

            price_data.append({
                'symbol': symbol,
                'timestamp': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(current_price, 2),
                'volume': volume,
                'market_regime': regime,
                'volatility_regime': vol_regime,
                'earnings_day': i == earnings_day
            })

        return price_data

    def generate_trend_reversal_scenario(
        self,
        symbol: str = "SPY",
        start_price: float = 400.0,
        duration_days: int = 60,
        reversal_day: int = 30
    ) -> List[Dict[str, Any]]:
        """Generate trend reversal from bearish to bullish"""

        price_data = []
        current_price = start_price
        start_date = datetime.now(timezone.utc) - timedelta(days=duration_days)

        for i in range(duration_days):
            date = start_date + timedelta(days=i)

            if i < reversal_day:
                # Bearish phase
                daily_drift = -0.001
                volatility = 0.03
                regime = "bear_trend"
            elif i == reversal_day:
                # Reversal day - high volatility, unclear direction
                daily_drift = 0.0
                volatility = 0.06
                regime = "reversal"
            else:
                # Bullish phase
                daily_drift = 0.0008
                volatility = 0.025
                regime = "bull_trend"

            daily_return = daily_drift + random.gauss(0, volatility)
            current_price *= (1 + daily_return)

            open_price = current_price * random.uniform(0.995, 1.005)
            high_price = max(open_price, current_price) * random.uniform(1.0, 1.02)
            low_price = min(open_price, current_price) * random.uniform(0.98, 1.0)

            volume = random.randint(2000000, 8000000)
            if i == reversal_day:
                volume *= 2  # High volume on reversal day

            price_data.append({
                'symbol': symbol,
                'timestamp': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(current_price, 2),
                'volume': volume,
                'market_regime': regime,
                'volatility_regime': 'elevated' if i == reversal_day else 'normal'
            })

        return price_data

    def generate_multi_asset_scenario(
        self,
        scenario_type: str = "bull_market",
        symbols: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate coordinated multi-asset scenario"""

        if symbols is None:
            symbols = ["SPY", "AAPL", "TSLA", "QQQ", "IWM"]

        scenarios = {}

        for symbol in symbols:
            if scenario_type == "bull_market":
                scenarios[symbol] = self.generate_bull_market_scenario(symbol=symbol)
            elif scenario_type == "bear_market":
                scenarios[symbol] = self.generate_bear_market_scenario(symbol=symbol)
            elif scenario_type == "sideways":
                scenarios[symbol] = self.generate_sideways_market_scenario(symbol=symbol)
            elif scenario_type == "high_volatility":
                scenarios[symbol] = self.generate_high_volatility_scenario(symbol=symbol)
            elif scenario_type == "crisis":
                scenarios[symbol] = self.generate_crisis_scenario(symbol=symbol)
            else:
                scenarios[symbol] = self.generate_bull_market_scenario(symbol=symbol)

        return scenarios


# Global generator instance
mock_market_generator = MockMarketScenariosGenerator()


def get_market_scenario(scenario_type: str, **kwargs) -> List[Dict[str, Any]]:
    """Convenience function to get market scenario data"""
    scenario_map = {
        'bull_market': mock_market_generator.generate_bull_market_scenario,
        'bear_market': mock_market_generator.generate_bear_market_scenario,
        'sideways': mock_market_generator.generate_sideways_market_scenario,
        'high_volatility': mock_market_generator.generate_high_volatility_scenario,
        'crisis': mock_market_generator.generate_crisis_scenario,
        'earnings': mock_market_generator.generate_earnings_event_scenario,
        'trend_reversal': mock_market_generator.generate_trend_reversal_scenario
    }

    generator_func = scenario_map.get(scenario_type, mock_market_generator.generate_bull_market_scenario)
    return generator_func(**kwargs)


def get_recent_market_data(
    symbol: str = "AAPL",
    days: int = 30,
    scenario: str = "normal"
) -> List[Dict[str, Any]]:
    """Get recent market data for testing"""
    if scenario == "normal":
        return mock_market_generator.generate_bull_market_scenario(
            symbol=symbol,
            duration_days=days,
            daily_drift=0.0002,  # Mild bullish bias
            volatility=0.22
        )
    else:
        return get_market_scenario(scenario, symbol=symbol, duration_days=days)


__all__ = [
    'MockMarketScenariosGenerator',
    'mock_market_generator',
    'get_market_scenario',
    'get_recent_market_data'
]