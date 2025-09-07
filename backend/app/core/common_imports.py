"""
Common Imports
Provides commonly used imports across the application to reduce redundancy
"""

import asyncio
import json

# Standard library imports (most commonly used)
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports (frequently used)
import numpy as np
import pandas as pd

# FastAPI imports (for API modules)
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String

# Database imports (for data access modules)
from sqlalchemy.orm import Session

# Analysis imports
from app.analysis.markov_system import MarkovSystem, MarkovCore

# Common model imports
from app.api.v1.models import (
    MarketAnalysisRequest,
    MarketAnalysisResponse,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderType,
    RiskLevel,
    TradingAction,
)

# Core application imports
from app.core.config import settings
from app.core.exceptions import MarketDataError, RiskManagementError, TradingError

# Logging (centralized)
from app.core.logging import get_logger, get_trading_logger

# Trading system imports
from app.trading.trading_utils import (
    calculate_pnl,
    calculate_risk_metrics,
    validate_price,
    validate_quantity,
    validate_symbol,
)

# ============================================================================
# COMMON PATTERNS AND UTILITIES
# ============================================================================

# Common logger pattern - use this instead of manual logger setup
logger = get_logger()


# Common datetime patterns
def now() -> datetime:
    """Get current datetime"""
    return datetime.now()


def utc_now() -> datetime:
    """Get current UTC datetime"""
    from datetime import timezone

    return datetime.now(timezone.utc)


def format_timestamp(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime to string"""
    return dt.strftime(format_str)


# Common validation patterns
def ensure_positive(value: Union[int, float], name: str = "value") -> Union[int, float]:
    """Ensure value is positive"""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def ensure_not_empty(value: str, name: str = "value") -> str:
    """Ensure string is not empty"""
    if not value or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    return value.strip()


# Common dictionary patterns
def safe_get(d: Dict, key: str, default: Any = None) -> Any:
    """Safely get value from dictionary"""
    return d.get(key, default) if isinstance(d, dict) else default


def merge_dicts(*dicts) -> Dict:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result


# Common async patterns
async def safe_async_call(coro, default=None, log_errors: bool = True):
    """Safely call async function with error handling"""
    try:
        return await coro
    except Exception as e:
        if log_errors:
            logger.error(f"Async call failed: {str(e)}")
        return default


# ============================================================================
# COMMON RESPONSE PATTERNS
# ============================================================================


def success_response(data: Any = None, message: str = "Success") -> Dict[str, Any]:
    """Standard success response"""
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": now().isoformat(),
    }


def error_response(
    error: str, message: str = None, details: Dict = None
) -> Dict[str, Any]:
    """Standard error response"""
    return {
        "success": False,
        "error": error,
        "message": message or error,
        "details": details or {},
        "timestamp": now().isoformat(),
    }


# ============================================================================
# COMMON DECORATORS (IMPORTS)
# ============================================================================

from app.core.logging import log_execution_time, log_trading_operation
from app.trading.trading_utils import handle_trading_error

# ============================================================================
# COMMON EXCEPTIONS
# ============================================================================


class ValidationError(ValueError):
    """Custom validation error"""

    pass


class ConfigurationError(Exception):
    """Configuration error"""

    pass


class ServiceUnavailableError(Exception):
    """Service unavailable error"""

    pass


# ============================================================================
# EXPORT GROUPS FOR DIFFERENT MODULE TYPES
# ============================================================================

# For API modules
API_IMPORTS = [
    "APIRouter",
    "Depends",
    "HTTPException",
    "status",
    "Query",
    "Path",
    "Body",
    "JSONResponse",
    "OrderRequest",
    "OrderResponse",
    "MarketAnalysisRequest",
    "MarketAnalysisResponse",
    "TradingAction",
    "OrderSide",
    "OrderType",
    "RiskLevel",
    "success_response",
    "error_response",
    "logger",
]

# For trading modules
TRADING_IMPORTS = [
    "TradingError",
    "RiskManagementError",
    "validate_symbol",
    "validate_quantity",
    "validate_price",
    "calculate_pnl",
    "calculate_risk_metrics",
    "get_trading_logger",
    "log_trading_operation",
    "now",
    "ensure_positive",
]

# For analysis modules
ANALYSIS_IMPORTS = [
    "np",
    "pd",
    "MarkovSystem",
    "MarkovCore",
    "datetime",
    "timedelta",
    "logger",
    "log_execution_time",
    "safe_get",
    "merge_dicts",
]

# For data models
MODEL_IMPORTS = [
    "BaseModel",
    "Field",
    "validator",
    "dataclass",
    "Enum",
    "Optional",
    "Dict",
    "List",
    "Any",
    "Union",
    "ValidationError",
    "ensure_not_empty",
]

# For database modules
DATABASE_IMPORTS = [
    "Session",
    "Column",
    "Integer",
    "String",
    "Float",
    "DateTime",
    "Boolean",
    "logger",
    "safe_async_call",
    "now",
    "utc_now",
]

# ============================================================================
# CONVENIENCE IMPORT FUNCTIONS
# ============================================================================


def get_api_imports():
    """Get common imports for API modules"""
    return {name: globals()[name] for name in API_IMPORTS if name in globals()}


def get_trading_imports():
    """Get common imports for trading modules"""
    return {name: globals()[name] for name in TRADING_IMPORTS if name in globals()}


def get_analysis_imports():
    """Get common imports for analysis modules"""
    return {name: globals()[name] for name in ANALYSIS_IMPORTS if name in globals()}


def get_model_imports():
    """Get common imports for model modules"""
    return {name: globals()[name] for name in MODEL_IMPORTS if name in globals()}


def get_database_imports():
    """Get common imports for database modules"""
    return {name: globals()[name] for name in DATABASE_IMPORTS if name in globals()}


# ============================================================================
# USAGE EXAMPLES (IN COMMENTS)
# ============================================================================

"""
Usage Examples:

# In API modules:
from app.core.common_imports import *
# or more specifically:
from app.core.common_imports import APIRouter, Depends, success_response, logger

# In trading modules:
from app.core.common_imports import validate_symbol, get_trading_logger, log_trading_operation

# In analysis modules:
from app.core.common_imports import np, pd, logger, MarkovSystem

# In model modules:
from app.core.common_imports import BaseModel, Field, validator, ValidationError
"""

# Export everything for wildcard imports (use judiciously)
__all__ = [
    # Time utilities
    "now",
    "utc_now",
    "format_timestamp",
    # Validation utilities
    "ensure_positive",
    "ensure_not_empty",
    # Dictionary utilities
    "safe_get",
    "merge_dicts",
    # Async utilities
    "safe_async_call",
    # Response patterns
    "success_response",
    "error_response",
    # Custom exceptions
    "ValidationError",
    "ConfigurationError",
    "ServiceUnavailableError",
    # Import groups
    "get_api_imports",
    "get_trading_imports",
    "get_analysis_imports",
    "get_model_imports",
    "get_database_imports",
]
