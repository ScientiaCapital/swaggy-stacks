"""
Logging convenience functions and utilities
"""

from typing import Optional

import structlog

from app.core.logger_setup import LoggerSetup
from app.core.specialized_loggers import TradingLogger, PerformanceLogger


def get_logger(name: str = None) -> structlog.stdlib.BoundLogger:
    """
    Get a standard structured logger

    Args:
        name: Logger name (auto-detected if not provided)

    Returns:
        Configured structlog logger
    """
    return LoggerSetup.get_logger(name)


def get_trading_logger(name: str = None) -> TradingLogger:
    """
    Get a trading-specific logger with specialized methods

    Args:
        name: Logger name (auto-detected if not provided)

    Returns:
        TradingLogger instance
    """
    base_logger = LoggerSetup.get_logger(name)
    return TradingLogger(base_logger)


def get_performance_logger(name: str = None) -> PerformanceLogger:
    """
    Get a performance monitoring logger

    Args:
        name: Logger name (auto-detected if not provided)

    Returns:
        PerformanceLogger instance
    """
    base_logger = LoggerSetup.get_logger(name)
    return PerformanceLogger(base_logger)


def configure_logging(**kwargs) -> None:
    """Configure application-wide logging"""
    LoggerSetup.configure_logging(**kwargs)


def reset_logging() -> None:
    """Reset logging configuration (primarily for testing)"""
    LoggerSetup.reset()