"""
Test backward compatibility of refactored logging module
"""

import pytest
from unittest.mock import Mock, patch

# Test importing from the original logging module
from app.core.logging import (
    get_logger,
    get_trading_logger,
    get_performance_logger,
    configure_logging,
    TradingLogger,
    PerformanceLogger,
    LoggerSetup,
    log_execution_time,
    log_trading_operation,
)


class TestLoggingBackwardCompatibility:
    """Test that refactored logging maintains backward compatibility"""

    def setup_method(self):
        """Reset logging setup before each test"""
        LoggerSetup.reset()

    def test_imports_work(self):
        """Test all imports are available"""
        # Test classes are importable
        assert TradingLogger is not None
        assert PerformanceLogger is not None
        assert LoggerSetup is not None

        # Test functions are importable
        assert callable(get_logger)
        assert callable(get_trading_logger)
        assert callable(get_performance_logger)
        assert callable(configure_logging)
        assert callable(log_execution_time)
        assert callable(log_trading_operation)

    def test_get_logger_function(self):
        """Test get_logger convenience function"""
        logger = get_logger("test")

        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'debug')

    def test_get_trading_logger_function(self):
        """Test get_trading_logger convenience function"""
        trading_logger = get_trading_logger("trading_test")

        assert isinstance(trading_logger, TradingLogger)
        assert hasattr(trading_logger, 'trade_event')
        assert hasattr(trading_logger, 'risk_event')
        assert hasattr(trading_logger, 'market_event')

    def test_get_performance_logger_function(self):
        """Test get_performance_logger convenience function"""
        perf_logger = get_performance_logger("perf_test")

        assert isinstance(perf_logger, PerformanceLogger)
        assert hasattr(perf_logger, 'log_execution_time')
        assert hasattr(perf_logger, 'log_memory_usage')

    def test_configure_logging_function(self):
        """Test configure_logging convenience function"""
        # Should not raise an exception
        configure_logging(log_level="DEBUG", log_format="json")
        assert LoggerSetup.is_configured()

    def test_trading_logger_class_methods(self):
        """Test TradingLogger class methods work"""
        base_logger = Mock()
        trading_logger = TradingLogger(base_logger)

        # Test specialized methods
        trading_logger.trade_event("test_event", symbol="AAPL")
        base_logger.info.assert_called_with(
            "TRADE_EVENT", event_type="test_event", symbol="AAPL"
        )

        # Test standard methods
        trading_logger.info("test message")
        base_logger.info.assert_called_with("test message")

    def test_performance_logger_class_methods(self):
        """Test PerformanceLogger class methods work"""
        base_logger = Mock()
        perf_logger = PerformanceLogger(base_logger)

        # Test performance logging methods
        perf_logger.log_execution_time("test_op", 1.5, extra="data")
        base_logger.info.assert_called_with(
            "PERFORMANCE",
            operation="test_op",
            duration_seconds=1.5,
            extra="data"
        )

        perf_logger.log_memory_usage("memory_op", 256.0)
        base_logger.info.assert_called_with(
            "MEMORY_USAGE",
            operation="memory_op",
            memory_mb=256.0
        )

    def test_log_execution_time_decorator(self):
        """Test log_execution_time decorator works"""
        mock_logger = Mock()

        @log_execution_time(mock_logger)
        def test_function():
            return "success"

        result = test_function()

        assert result == "success"
        mock_logger.info.assert_called_once()

    @patch('app.core.logging_decorators.LoggerSetup.get_logger')
    @patch('app.core.logging_decorators.TradingLogger')
    def test_log_trading_operation_decorator(self, mock_trading_logger_class, mock_get_logger):
        """Test log_trading_operation decorator works"""
        mock_base_logger = Mock()
        mock_trading_logger = Mock()
        mock_get_logger.return_value = mock_base_logger
        mock_trading_logger_class.return_value = mock_trading_logger

        @log_trading_operation("test_operation")
        def test_trading_function():
            return "completed"

        result = test_trading_function()

        assert result == "completed"
        # Should log start and completion events
        assert mock_trading_logger.trade_event.call_count == 2

    def test_logger_setup_class_methods(self):
        """Test LoggerSetup class methods work"""
        # Test configuration
        LoggerSetup.configure_logging(log_level="INFO")
        assert LoggerSetup.is_configured()

        # Test getting logger
        logger = LoggerSetup.get_logger("setup_test")
        assert logger is not None

        # Test reset
        LoggerSetup.reset()
        assert not LoggerSetup.is_configured()

    def test_existing_code_patterns(self):
        """Test common existing code patterns still work"""
        # Pattern 1: Basic logger usage
        logger = get_logger()
        assert hasattr(logger, 'info')

        # Pattern 2: Trading logger usage
        trading_logger = get_trading_logger()
        assert hasattr(trading_logger, 'trade_event')

        # Pattern 3: Performance logger usage
        perf_logger = get_performance_logger()
        assert hasattr(perf_logger, 'log_execution_time')

        # Pattern 4: Configuration
        configure_logging(log_format="console")
        assert LoggerSetup.is_configured()

    def test_module_initialization(self):
        """Test module initializes correctly on import"""
        # Logging should be configured automatically on import
        # Reset first to test auto-configuration
        LoggerSetup.reset()

        # Re-import should trigger auto-configuration
        import importlib
        import app.core.logging
        importlib.reload(app.core.logging)

        # Should be configured now
        assert LoggerSetup.is_configured()

    def test_all_exports_available(self):
        """Test all expected exports are in __all__"""
        from app.core.logging import __all__

        expected_exports = [
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

        for export in expected_exports:
            assert export in __all__, f"Missing export: {export}"