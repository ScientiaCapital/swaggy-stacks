"""
Logger configuration and setup
Pure configuration logic for structlog
"""

import logging
import sys
from typing import Optional

import structlog

from app.core.config import settings


class LoggerSetup:
    """Centralized logger setup and configuration"""

    _configured = False
    _loggers = {}

    @classmethod
    def configure_logging(
        cls, log_level: str = None, log_format: str = "json", enable_colors: bool = True
    ) -> None:
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
        level = log_level or getattr(settings, "LOG_LEVEL", "INFO")
        log_level_num = getattr(logging, level.upper(), logging.INFO)

        # Configure Python's standard logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=log_level_num,
        )

        # Configure structlog processors
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]

        # Add appropriate renderer based on format
        if log_format == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=enable_colors))

        # Configure structlog
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
            name = frame.f_globals.get("__name__", "unknown")

        # Cache loggers to avoid recreation
        if name not in cls._loggers:
            cls._loggers[name] = structlog.get_logger(name)

        return cls._loggers[name]

    @classmethod
    def is_configured(cls) -> bool:
        """Check if logging has been configured"""
        return cls._configured

    @classmethod
    def reset(cls) -> None:
        """Reset configuration (primarily for testing)"""
        cls._configured = False
        cls._loggers.clear()