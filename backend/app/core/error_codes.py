"""
Error Code Registry System
Backward-compatible facade over refactored error handling components
"""

from typing import Any, Dict, List, Optional

# Import all components from the refactored modules
from app.core.error_definitions import (
    ErrorSeverity,
    ErrorCategory,
    ErrorCodeInfo,
    TradingErrorCodes,
)
from app.core.error_registry import ErrorCodeRegistry
from app.core.error_responses import (
    get_error_info,
    create_error_response,
    validate_error_code,
)

# Maintain backward compatibility by exposing the global registry
error_registry = ErrorCodeRegistry()
