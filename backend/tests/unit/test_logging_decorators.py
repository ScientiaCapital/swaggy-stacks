"""
Test logging decorators
"""

import pytest
import time
from unittest.mock import Mock, patch
from app.core.logging_decorators import log_execution_time, log_trading_operation


class TestLogExecutionTimeDecorator:
    """Test log_execution_time decorator"""

    def test_successful_execution(self):
        """Test decorator logs successful function execution"""
        mock_logger = Mock()

        @log_execution_time(mock_logger)
        def test_function(x, y):
            return x + y

        result = test_function(3, 4)

        assert result == 7
        mock_logger.info.assert_called_once()

        # Check the log call
        call_args = mock_logger.info.call_args
        assert "Function executed" in call_args[0]
        assert call_args[1]["function"] == "test_function"
        assert call_args[1]["success"] is True
        assert "duration_seconds" in call_args[1]

    def test_function_with_exception(self):
        """Test decorator logs function failures"""
        mock_logger = Mock()

        @log_execution_time(mock_logger)
        def failing_function():
            raise ValueError("Something went wrong")

        with pytest.raises(ValueError, match="Something went wrong"):
            failing_function()

        mock_logger.error.assert_called_once()

        # Check the error log call
        call_args = mock_logger.error.call_args
        assert "Function failed" in call_args[0]
        assert call_args[1]["function"] == "failing_function"
        assert call_args[1]["success"] is False
        assert call_args[1]["error"] == "Something went wrong"
        assert "duration_seconds" in call_args[1]

    @patch('app.core.logging_decorators.LoggerSetup.get_logger')
    def test_default_logger(self, mock_get_logger):
        """Test decorator uses default logger when none provided"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        @log_execution_time()
        def test_function():
            return "success"

        result = test_function()

        assert result == "success"
        mock_get_logger.assert_called_once()
        mock_logger.info.assert_called_once()

    def test_preserves_function_metadata(self):
        """Test decorator preserves function metadata"""
        mock_logger = Mock()

        @log_execution_time(mock_logger)
        def documented_function(param1, param2="default"):
            """This is a documented function"""
            return param1

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented function"

    def test_measures_execution_time(self):
        """Test decorator measures execution time"""
        mock_logger = Mock()

        @log_execution_time(mock_logger)
        def slow_function():
            time.sleep(0.01)  # Sleep for 10ms
            return "done"

        result = slow_function()

        assert result == "done"
        call_args = mock_logger.info.call_args
        duration = call_args[1]["duration_seconds"]
        assert duration >= 0.01  # Should be at least 10ms
        assert duration < 1.0     # But not too long


class TestLogTradingOperationDecorator:
    """Test log_trading_operation decorator"""

    @patch('app.core.logging_decorators.LoggerSetup.get_logger')
    @patch('app.core.logging_decorators.TradingLogger')
    def test_successful_trading_operation(self, mock_trading_logger_class, mock_get_logger):
        """Test decorator logs successful trading operations"""
        mock_base_logger = Mock()
        mock_trading_logger = Mock()
        mock_get_logger.return_value = mock_base_logger
        mock_trading_logger_class.return_value = mock_trading_logger

        @log_trading_operation("order_placement")
        def place_order(symbol, quantity):
            return f"Order placed: {quantity} shares of {symbol}"

        result = place_order("AAPL", 100)

        assert result == "Order placed: 100 shares of AAPL"

        # Verify TradingLogger was created with base logger
        mock_trading_logger_class.assert_called_once_with(mock_base_logger)

        # Verify start and completion events were logged
        assert mock_trading_logger.trade_event.call_count == 2

        # Check start event
        start_call = mock_trading_logger.trade_event.call_args_list[0]
        assert start_call[0][0] == "operation_started"
        assert start_call[1]["operation"] == "order_placement"
        assert start_call[1]["function"] == "place_order"

        # Check completion event
        completion_call = mock_trading_logger.trade_event.call_args_list[1]
        assert completion_call[0][0] == "operation_completed"
        assert completion_call[1]["operation"] == "order_placement"
        assert completion_call[1]["function"] == "place_order"

    @patch('app.core.logging_decorators.LoggerSetup.get_logger')
    @patch('app.core.logging_decorators.TradingLogger')
    def test_failed_trading_operation(self, mock_trading_logger_class, mock_get_logger):
        """Test decorator logs failed trading operations"""
        mock_base_logger = Mock()
        mock_trading_logger = Mock()
        mock_get_logger.return_value = mock_base_logger
        mock_trading_logger_class.return_value = mock_trading_logger

        @log_trading_operation("order_placement")
        def failing_order():
            raise RuntimeError("Market is closed")

        with pytest.raises(RuntimeError, match="Market is closed"):
            failing_order()

        # Verify start event was logged
        mock_trading_logger.trade_event.assert_called_once_with(
            "operation_started",
            operation="order_placement",
            function="failing_order"
        )

        # Verify error event was logged
        mock_trading_logger.error_event.assert_called_once()
        error_call = mock_trading_logger.error_event.call_args
        assert error_call[0][0] == "operation_failed"
        assert isinstance(error_call[1]["error"], RuntimeError)
        assert error_call[1]["operation"] == "order_placement"
        assert error_call[1]["function"] == "failing_order"

    @patch('app.core.logging_decorators.LoggerSetup.get_logger')
    @patch('app.core.logging_decorators.TradingLogger')
    def test_preserves_function_metadata(self, mock_trading_logger_class, mock_get_logger):
        """Test decorator preserves function metadata"""
        mock_base_logger = Mock()
        mock_trading_logger = Mock()
        mock_get_logger.return_value = mock_base_logger
        mock_trading_logger_class.return_value = mock_trading_logger

        @log_trading_operation("test_operation")
        def trading_function(param):
            """Trading function with documentation"""
            return param * 2

        assert trading_function.__name__ == "trading_function"
        assert trading_function.__doc__ == "Trading function with documentation"

    @patch('app.core.logging_decorators.LoggerSetup.get_logger')
    @patch('app.core.logging_decorators.TradingLogger')
    def test_different_operation_types(self, mock_trading_logger_class, mock_get_logger):
        """Test decorator with different operation types"""
        mock_base_logger = Mock()
        mock_trading_logger = Mock()
        mock_get_logger.return_value = mock_base_logger
        mock_trading_logger_class.return_value = mock_trading_logger

        @log_trading_operation("risk_check")
        def check_risk():
            return "risk_ok"

        @log_trading_operation("portfolio_update")
        def update_portfolio():
            return "updated"

        # Call both functions
        check_risk()
        update_portfolio()

        # Verify correct operation types were logged
        calls = mock_trading_logger.trade_event.call_args_list
        assert len(calls) == 4  # 2 start + 2 completion events

        # Check risk_check operations
        assert calls[0][1]["operation"] == "risk_check"
        assert calls[1][1]["operation"] == "risk_check"

        # Check portfolio_update operations
        assert calls[2][1]["operation"] == "portfolio_update"
        assert calls[3][1]["operation"] == "portfolio_update"