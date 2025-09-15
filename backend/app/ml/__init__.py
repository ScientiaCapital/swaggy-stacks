"""
Machine Learning Module

This module contains all ML-based prediction logic and model management.
Focuses on predictive models, LLM integration, and learning algorithms.

Key Components:
- LLMPredictor: LLM-based market prediction
- MarkovSystem: Markov chain analysis for trend detection
- VolatilityPredictor: GARCH-based volatility forecasting for options pricing
- PredictionResult: Standard prediction data structures
"""

from .llm_predictors import LLMPredictor, PredictionResult, MarketContext, get_llm_predictor, predict_symbol_direction
from .markov_system import MarkovSystem, MarkovCore, DataHandler, PositionSizer
from .volatility_predictor import (
    VolatilityPredictor,
    VolatilityMetrics,
    VolatilitySmile,
    VolatilityRegime,
    get_volatility_predictor
)

__all__ = [
    "LLMPredictor",
    "PredictionResult",
    "MarketContext",
    "get_llm_predictor",
    "predict_symbol_direction",
    "MarkovSystem",
    "MarkovCore",
    "DataHandler",
    "PositionSizer",
    "VolatilityPredictor",
    "VolatilityMetrics",
    "VolatilitySmile",
    "VolatilityRegime",
    "get_volatility_predictor",
]