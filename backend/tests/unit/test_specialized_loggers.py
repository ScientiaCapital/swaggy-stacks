"""
Test specialized logger classes
"""

import pytest
from unittest.mock import Mock, MagicMock
from app.core.specialized_loggers import TradingLogger, PerformanceLogger


class TestTradingLogger:
    """Test TradingLogger functionality"""

    @pytest.fixture
    def mock_base_logger(self):
        """Create a mock base logger"""
        logger = Mock()
        logger.info = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        logger.debug = Mock()
        logger.critical = Mock()
        logger.bind = Mock(return_value=logger)
        return logger

    @pytest.fixture
    def trading_logger(self, mock_base_logger):
        """Create a TradingLogger instance"""
        return TradingLogger(mock_base_logger)

    def test_initialization(self, mock_base_logger):
        """Test TradingLogger initialization"""
        logger = TradingLogger(mock_base_logger)

        assert logger.base_logger is mock_base_logger
        assert logger._context == {}

    def test_bind_context(self, trading_logger, mock_base_logger):
        """Test binding context to logger"""
        context = {"user_id": "123", "session": "abc"}
        bound_logger = trading_logger.bind(**context)

        assert isinstance(bound_logger, TradingLogger)
        mock_base_logger.bind.assert_called_once_with(**context)

    def test_trade_event(self, trading_logger, mock_base_logger):
        """Test logging trade events"""
        trading_logger.trade_event("order_placed", symbol="AAPL", quantity=100)

        mock_base_logger.info.assert_called_once_with(
            "TRADE_EVENT",
            event_type="order_placed",
            symbol="AAPL",
            quantity=100
        )

    def test_risk_event(self, trading_logger, mock_base_logger):
        """Test logging risk events"""
        trading_logger.risk_event("position_limit_exceeded", symbol="MSFT")

        mock_base_logger.warning.assert_called_once_with(
            "RISK_EVENT",
            event_type="position_limit_exceeded",
            symbol="MSFT"
        )

    def test_market_event(self, trading_logger, mock_base_logger):
        """Test logging market events"""
        trading_logger.market_event("price_update", symbol="GOOGL", price=150.50)

        mock_base_logger.info.assert_called_once_with(
            "MARKET_EVENT",
            event_type="price_update",
            symbol="GOOGL",
            price=150.50
        )

    def test_strategy_event(self, trading_logger, mock_base_logger):
        """Test logging strategy events"""
        trading_logger.strategy_event("signal_generated", strategy="momentum")

        mock_base_logger.info.assert_called_once_with(
            "STRATEGY_EVENT",
            event_type="signal_generated",
            strategy="momentum"
        )

    def test_error_event(self, trading_logger, mock_base_logger):
        """Test logging error events"""
        error = ValueError("Invalid price")
        trading_logger.error_event("order_failed", error, order_id="12345")

        mock_base_logger.error.assert_called_once_with(
            "ERROR_EVENT",
            event_type="order_failed",
            error="Invalid price",
            error_type="ValueError",
            order_id="12345"
        )

    def test_standard_log_methods(self, trading_logger, mock_base_logger):
        """Test standard logging methods are delegated"""
        # Test debug
        trading_logger.debug("Debug message", extra="data")
        mock_base_logger.debug.assert_called_with("Debug message", extra="data")

        # Test info
        trading_logger.info("Info message", count=5)
        mock_base_logger.info.assert_called_with("Info message", count=5)

        # Test warning
        trading_logger.warning("Warning message")
        mock_base_logger.warning.assert_called_with("Warning message")

        # Test error
        trading_logger.error("Error message", code="E001")
        mock_base_logger.error.assert_called_with("Error message", code="E001")

        # Test critical
        trading_logger.critical("Critical message")
        mock_base_logger.critical.assert_called_with("Critical message")

    def test_chained_binding(self, mock_base_logger):
        """Test chaining bind operations"""
        logger = TradingLogger(mock_base_logger)

        # Mock bind to return new logger each time
        mock_logger_1 = Mock()
        mock_logger_1.bind = Mock(return_value=Mock())
        mock_base_logger.bind.return_value = mock_logger_1

        # Chain bindings
        bound1 = logger.bind(user="alice")
        bound2 = bound1.bind(session="xyz")

        # Verify initial bind was called
        mock_base_logger.bind.assert_called_once_with(user="alice")
        # Verify second bind was called on the returned logger
        mock_logger_1.bind.assert_called_once_with(session="xyz")


class TestPerformanceLogger:
    """Test PerformanceLogger functionality"""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger"""
        logger = Mock()
        logger.info = Mock()
        return logger

    @pytest.fixture
    def perf_logger(self, mock_logger):
        """Create a PerformanceLogger instance"""
        return PerformanceLogger(mock_logger)

    def test_initialization(self, mock_logger):
        """Test PerformanceLogger initialization"""
        logger = PerformanceLogger(mock_logger)
        assert logger.logger is mock_logger

    def test_log_execution_time(self, perf_logger, mock_logger):
        """Test logging execution time"""
        perf_logger.log_execution_time("database_query", 0.125, table="users")

        mock_logger.info.assert_called_once_with(
            "PERFORMANCE",
            operation="database_query",
            duration_seconds=0.125,
            table="users"
        )

    def test_log_memory_usage(self, perf_logger, mock_logger):
        """Test logging memory usage"""
        perf_logger.log_memory_usage("data_processing", 256.5, rows=10000)

        mock_logger.info.assert_called_once_with(
            "MEMORY_USAGE",
            operation="data_processing",
            memory_mb=256.5,
            rows=10000
        )

    def test_multiple_operations(self, perf_logger, mock_logger):
        """Test logging multiple performance metrics"""
        perf_logger.log_execution_time("operation_1", 1.5)
        perf_logger.log_memory_usage("operation_2", 128.0)
        perf_logger.log_execution_time("operation_3", 0.05)

        assert mock_logger.info.call_count == 3

        # Verify calls
        calls = mock_logger.info.call_args_list
        assert calls[0][0] == ("PERFORMANCE",)
        assert calls[1][0] == ("MEMORY_USAGE",)
        assert calls[2][0] == ("PERFORMANCE",)