"""
Options Strategies Module

This module provides a comprehensive suite of options trading strategies
for the SwaggyStacks trading system. All strategies follow a consistent
architecture with configuration classes, position tracking, and signal generation.

Available Strategies:
- Zero DTE: Day trading options expiring same day
- Wheel: Cash-secured puts + covered calls cycle
- Iron Condor: Range-bound market neutral strategy
- Gamma Scalping: Delta hedging with gamma exposure
- Long Straddle: Volatility play for large price movements
- Iron Butterfly: Credit strategy for low volatility environments
- Calendar Spread: Time decay arbitrage between expirations
- Bull Call Spread: Limited-risk bullish directional play
- Bear Put Spread: Limited-risk bearish directional play
- Covered Call: Income generation from stock holdings
- Protective Put: Portfolio insurance against downside risk
"""

# Existing strategies
from .zero_dte_strategy import ZeroDTEStrategy, ZeroDTEConfig, ZeroDTEPosition
from .wheel_strategy import WheelStrategy, WheelConfig, WheelPosition, WheelPhase, BollingerBands
from .iron_condor_strategy import IronCondorStrategy, IronCondorConfig, IronCondorPosition, IronCondorLeg
from .gamma_scalping_strategy import GammaScalpingStrategy, GammaScalpingConfig, GammaPosition, RebalanceOrder, RebalanceAction

# New advanced strategies
from .long_straddle_strategy import LongStraddleStrategy, LongStraddleConfig, StraddlePosition
from .iron_butterfly_strategy import IronButterflyStrategy, IronButterflyConfig, IronButterflyPosition
from .calendar_spread_strategy import CalendarSpreadStrategy, CalendarSpreadConfig, CalendarSpreadPosition
from .bull_call_spread_strategy import BullCallSpreadStrategy, BullCallSpreadConfig, BullCallSpreadPosition
from .bear_put_spread_strategy import BearPutSpreadStrategy, BearPutSpreadConfig, BearPutSpreadPosition
from .covered_call_strategy import CoveredCallStrategy, CoveredCallConfig, CoveredCallPosition
from .protective_put_strategy import ProtectivePutStrategy, ProtectivePutConfig, ProtectivePutPosition

# Utility classes
from .black_scholes import BlackScholesCalculator, GreeksData

# Factory
from .options_strategy_factory import OptionsStrategyFactory, StrategyType, MarketRegime

__all__ = [
    # Existing strategies
    "ZeroDTEStrategy", "ZeroDTEConfig", "ZeroDTEPosition",
    "WheelStrategy", "WheelConfig", "WheelPosition", "WheelPhase", "BollingerBands",
    "IronCondorStrategy", "IronCondorConfig", "IronCondorPosition", "IronCondorLeg",
    "GammaScalpingStrategy", "GammaScalpingConfig", "GammaPosition", "RebalanceOrder", "RebalanceAction",

    # Advanced strategies
    "LongStraddleStrategy", "LongStraddleConfig", "StraddlePosition",
    "IronButterflyStrategy", "IronButterflyConfig", "IronButterflyPosition",
    "CalendarSpreadStrategy", "CalendarSpreadConfig", "CalendarSpreadPosition",
    "BullCallSpreadStrategy", "BullCallSpreadConfig", "BullCallSpreadPosition",
    "BearPutSpreadStrategy", "BearPutSpreadConfig", "BearPutSpreadPosition",
    "CoveredCallStrategy", "CoveredCallConfig", "CoveredCallPosition",
    "ProtectivePutStrategy", "ProtectivePutConfig", "ProtectivePutPosition",

    # Utility classes
    "BlackScholesCalculator", "GreeksData",

    # Factory
    "OptionsStrategyFactory", "StrategyType", "MarketRegime",
]