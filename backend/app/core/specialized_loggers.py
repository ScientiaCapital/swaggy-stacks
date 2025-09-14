"""
Specialized logger classes for domain-specific logging
"""

import structlog


class TradingLogger:
    """Specialized logger for trading operations with context management"""

    def __init__(self, base_logger: structlog.stdlib.BoundLogger):
        self.base_logger = base_logger
        self._context = {}

    def bind(self, **kwargs) -> "TradingLogger":
        """Bind additional context to the logger"""
        new_context = {**self._context, **kwargs}
        new_logger = self.base_logger.bind(**new_context)
        return TradingLogger(new_logger)

    def trade_event(self, event: str, **kwargs):
        """Log trading-specific events"""
        self.base_logger.info("TRADE_EVENT", event_type=event, **kwargs)

    def risk_event(self, event: str, **kwargs):
        """Log risk management events"""
        self.base_logger.warning("RISK_EVENT", event_type=event, **kwargs)

    def market_event(self, event: str, **kwargs):
        """Log market data events"""
        self.base_logger.info("MARKET_EVENT", event_type=event, **kwargs)

    def strategy_event(self, event: str, **kwargs):
        """Log strategy-related events"""
        self.base_logger.info("STRATEGY_EVENT", event_type=event, **kwargs)

    def error_event(self, event: str, error: Exception, **kwargs):
        """Log error events with exception details"""
        self.base_logger.error(
            "ERROR_EVENT",
            event_type=event,
            error=str(error),
            error_type=type(error).__name__,
            **kwargs,
        )

    # Delegate standard methods to base logger
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
            "PERFORMANCE", operation=operation, duration_seconds=duration, **kwargs
        )

    def log_memory_usage(self, operation: str, memory_mb: float, **kwargs):
        """Log memory usage"""
        self.logger.info(
            "MEMORY_USAGE", operation=operation, memory_mb=memory_mb, **kwargs
        )