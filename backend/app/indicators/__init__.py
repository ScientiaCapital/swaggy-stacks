"""
Technical Indicators Module

This module contains all technical indicator calculations and related functionality.
Focuses purely on mathematical computations for trading indicators.

Key Components:
- TechnicalIndicators: Core indicator calculations (RSI, MACD, etc.)
- ModernIndicators: Advanced and custom indicators
- IndicatorFactory: Factory for creating and managing indicators
"""

from .technical_indicators import TechnicalIndicators
from .modern_indicators import *
from .indicator_factory import *

__all__ = [
    "TechnicalIndicators",
    # ModernIndicators exports will be added during file moves
    # IndicatorFactory exports will be added during file moves
]