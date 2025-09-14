"""
Analysis Module

This module contains pattern analysis, validation frameworks, and alpha detection.
Focuses on analyzing patterns and validating trading strategies.

Key Components:
- PatternValidationFramework: Framework for validating trading patterns
- IntegratedAlphaDetector: Alpha signal detection and analysis
- ToolFeedbackTracker: Analysis feedback and learning system

Note: Technical indicators moved to app.indicators module
Note: ML predictions moved to app.ml module
"""

from .pattern_validation_framework import (
    ValidationStatus,
    PatternValidationResult,
    PatternValidationFramework,
    validate_pattern_library_expansion
)
from .integrated_alpha_detector import (
    AlphaConfidenceLevel,
    IntegratedAlphaSignal,
    IntegratedAlphaDetector,
    create_alpha_detector
)
from .tool_feedback_tracker import *

__all__ = [
    "ValidationStatus",
    "PatternValidationResult",
    "PatternValidationFramework",
    "validate_pattern_library_expansion",
    "AlphaConfidenceLevel",
    "IntegratedAlphaSignal",
    "IntegratedAlphaDetector",
    "create_alpha_detector",
    # ToolFeedbackTracker exports will be available
]
