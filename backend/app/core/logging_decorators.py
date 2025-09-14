"""
Logging decorators for cross-cutting concerns
"""

import functools
import time
from typing import Any, Callable, Optional

import structlog

from app.core.logger_setup import LoggerSetup
from app.core.specialized_loggers import TradingLogger


def log_execution_time(logger: Optional[structlog.stdlib.BoundLogger] = None):
    """Decorator to log function execution time"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            log = logger or LoggerSetup.get_logger()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                log.info(
                    "Function executed",
                    function=func.__name__,
                    duration_seconds=duration,
                    success=True,
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                log.error(
                    "Function failed",
                    function=func.__name__,
                    duration_seconds=duration,
                    error=str(e),
                    success=False,
                )
                raise

        return wrapper

    return decorator


def log_trading_operation(operation_type: str):
    """Decorator for trading operations"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            base_logger = LoggerSetup.get_logger()
            logger = TradingLogger(base_logger)

            try:
                logger.trade_event(
                    "operation_started",
                    operation=operation_type,
                    function=func.__name__,
                )
                result = func(*args, **kwargs)
                logger.trade_event(
                    "operation_completed",
                    operation=operation_type,
                    function=func.__name__,
                )
                return result
            except Exception as e:
                logger.error_event(
                    "operation_failed",
                    error=e,
                    operation=operation_type,
                    function=func.__name__,
                )
                raise

        return wrapper

    return decorator