"""
Centralized Logging Configuration
Backward-compatible facade over refactored logging components
"""

# Import all components from refactored modules
from app.core.logger_setup import LoggerSetup
from app.core.specialized_loggers import TradingLogger, PerformanceLogger
from app.core.logging_decorators import log_execution_time, log_trading_operation
from app.core.logging_utils import (
    get_logger,
    get_trading_logger,
    get_performance_logger,
    configure_logging,
)

# Initialize logging on import if not already configured
if not LoggerSetup.is_configured():
    try:
        LoggerSetup.configure_logging()
    except Exception as e:
        # Fallback to basic logging if configuration fails
        import logging
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Failed to configure structured logging: {e}")

# Export main functions and classes
__all__ = [
    "get_logger",
    "get_trading_logger",
    "get_performance_logger",
    "configure_logging",
    "TradingLogger",
    "PerformanceLogger",
    "LoggerSetup",
    "log_execution_time",
    "log_trading_operation",
]
