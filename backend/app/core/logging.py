"""
Centralized Logging Configuration
Eliminates repeated logger setup across all modules
Provides consistent structured logging throughout the application
"""

import sys
import logging
from typing import Optional, Dict, Any
import structlog
from pathlib import Path

from app.core.config import settings


class LoggerSetup:
    """Centralized logger setup and configuration"""
    
    _configured = False
    _loggers = {}
    
    @classmethod
    def configure_logging(cls, 
                         log_level: str = None,
                         log_format: str = "json",
                         enable_colors: bool = True) -> None:
        """
        Configure structured logging for the entire application
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Format type (json, console)
            enable_colors: Enable colored output for console logging
        """
        if cls._configured:
            return
        
        # Get log level from settings or parameter
        level = log_level or getattr(settings, 'LOG_LEVEL', 'INFO')
        log_level_num = getattr(logging, level.upper(), logging.INFO)
        
        # Configure Python's standard logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=log_level_num,
        )
        
        # Configure structlog
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.add_logger_name,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        if log_format == "json":
            # JSON formatter for production
            processors.append(structlog.processors.JSONRenderer())
        else:
            # Console formatter for development
            if enable_colors:
                processors.append(structlog.dev.ConsoleRenderer(colors=True))
            else:
                processors.append(structlog.dev.ConsoleRenderer(colors=False))
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str = None) -> structlog.stdlib.BoundLogger:
        """
        Get a configured logger instance
        
        Args:
            name: Logger name (defaults to calling module name)
            
        Returns:
            Configured structlog logger
        """
        # Ensure logging is configured
        if not cls._configured:
            cls.configure_logging()
        
        # Use calling module name if not provided
        if name is None:
            import inspect
            frame = inspect.currentframe().f_back
            name = frame.f_globals.get('__name__', 'unknown')
        
        # Cache loggers to avoid recreation
        if name not in cls._loggers:
            cls._loggers[name] = structlog.get_logger(name)
        
        return cls._loggers[name]


class TradingLogger:
    """Specialized logger for trading operations with context management"""
    
    def __init__(self, base_logger: structlog.stdlib.BoundLogger):
        self.base_logger = base_logger
        self._context = {}
    
    def bind(self, **kwargs) -> 'TradingLogger':
        """Bind additional context to the logger"""
        new_context = {**self._context, **kwargs}
        new_logger = self.base_logger.bind(**new_context)
        return TradingLogger(new_logger)
    
    def trade_event(self, event: str, **kwargs):
        """Log trading-specific events"""
        self.base_logger.info(
            "TRADE_EVENT",
            event_type=event,
            **kwargs
        )
    
    def risk_event(self, event: str, **kwargs):
        """Log risk management events"""
        self.base_logger.warning(
            "RISK_EVENT",
            event_type=event,
            **kwargs
        )
    
    def market_event(self, event: str, **kwargs):
        """Log market data events"""
        self.base_logger.info(
            "MARKET_EVENT",
            event_type=event,
            **kwargs
        )
    
    def strategy_event(self, event: str, **kwargs):
        """Log strategy-related events"""
        self.base_logger.info(
            "STRATEGY_EVENT",
            event_type=event,
            **kwargs
        )
    
    def error_event(self, event: str, error: Exception, **kwargs):
        """Log error events with exception details"""
        self.base_logger.error(
            "ERROR_EVENT",
            event_type=event,
            error=str(error),
            error_type=type(error).__name__,
            **kwargs
        )
    
    # Delegate other methods to base logger
    def debug(self, msg, **kwargs):
        return self.base_logger.debug(msg, **kwargs)
    
    def info(self, msg, **kwargs):
        return self.base_logger.info(msg, **kwargs)
    
    def warning(self, msg, **kwargs):
        return self.base_logger.warning(msg, **kwargs)
    
    def error(self, msg, **kwargs):
        return self.base_logger.error(msg, **kwargs)
    
    def critical(self, msg, **kwargs):
        return self.base_logger.critical(msg, **kwargs)


class PerformanceLogger:
    """Logger for performance monitoring"""
    
    def __init__(self, logger: structlog.stdlib.BoundLogger):
        self.logger = logger
    
    def log_execution_time(self, operation: str, duration: float, **kwargs):
        """Log operation execution time"""
        self.logger.info(
            "PERFORMANCE",
            operation=operation,
            duration_seconds=duration,
            **kwargs
        )
    
    def log_memory_usage(self, operation: str, memory_mb: float, **kwargs):
        """Log memory usage"""
        self.logger.info(
            "MEMORY_USAGE",
            operation=operation,
            memory_mb=memory_mb,
            **kwargs
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

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


# ============================================================================
# DECORATORS
# ============================================================================

import functools
import time
from typing import Callable, Any

def log_execution_time(logger: Optional[structlog.stdlib.BoundLogger] = None):
    """Decorator to log function execution time"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            log = logger or get_logger()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                log.info(
                    "Function executed",
                    function=func.__name__,
                    duration_seconds=duration,
                    success=True
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                log.error(
                    "Function failed",
                    function=func.__name__,
                    duration_seconds=duration,
                    error=str(e),
                    success=False
                )
                raise
        return wrapper
    return decorator


def log_trading_operation(operation_type: str):
    """Decorator for trading operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = get_trading_logger()
            
            try:
                logger.trade_event(
                    "operation_started",
                    operation=operation_type,
                    function=func.__name__
                )
                result = func(*args, **kwargs)
                logger.trade_event(
                    "operation_completed",
                    operation=operation_type,
                    function=func.__name__
                )
                return result
            except Exception as e:
                logger.error_event(
                    "operation_failed",
                    error=e,
                    operation=operation_type,
                    function=func.__name__
                )
                raise
        return wrapper
    return decorator


# ============================================================================
# INITIALIZATION
# ============================================================================

# Configure logging on import if not already configured
if not LoggerSetup._configured:
    try:
        LoggerSetup.configure_logging()
    except Exception as e:
        # Fallback to basic logging if configuration fails
        import logging
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Failed to configure structured logging: {e}")


# Export main functions and classes
__all__ = [
    'get_logger', 
    'get_trading_logger', 
    'get_performance_logger',
    'configure_logging',
    'TradingLogger',
    'PerformanceLogger',
    'log_execution_time',
    'log_trading_operation'
]