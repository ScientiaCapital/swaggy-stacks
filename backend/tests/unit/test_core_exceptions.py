"""
Test core exception classes
"""

import pytest
from app.core.exceptions import (
    TradingSystemException,
    TradingError,
    MarketDataError,
    RiskManagementError,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    ConfigurationError,
)


class TestTradingSystemExceptions:
    """Test trading system exception classes"""

    def test_trading_system_exception_basic(self):
        """Test basic TradingSystemException creation"""
        exc = TradingSystemException("Test error")
        assert exc.detail == "Test error"
        assert str(exc) == "Test error"

    def test_trading_system_exception_with_context(self):
        """Test TradingSystemException with additional context"""
        context = {"symbol": "AAPL", "price": 150.0}
        exc = TradingSystemException(
            "Trading failed",
            additional_context=context
        )
        assert exc.detail == "Trading failed"
        assert exc.additional_context == context

    def test_trading_error(self):
        """Test TradingError exception"""
        exc = TradingError("Order execution failed")
        assert exc.detail == "Order execution failed"
        assert isinstance(exc, TradingSystemException)

    def test_market_data_error(self):
        """Test MarketDataError exception"""
        exc = MarketDataError("Failed to fetch market data")
        assert exc.detail == "Failed to fetch market data"
        assert isinstance(exc, TradingSystemException)

    def test_risk_management_error(self):
        """Test RiskManagementError exception"""
        exc = RiskManagementError("Risk limit exceeded")
        assert exc.detail == "Risk limit exceeded"
        assert isinstance(exc, TradingSystemException)

    def test_authentication_error(self):
        """Test AuthenticationError exception"""
        exc = AuthenticationError("Invalid credentials")
        assert exc.detail == "Invalid credentials"
        assert isinstance(exc, TradingSystemException)

    def test_authorization_error(self):
        """Test AuthorizationError exception"""
        exc = AuthorizationError("Access denied")
        assert exc.detail == "Access denied"
        assert isinstance(exc, TradingSystemException)

    def test_validation_error(self):
        """Test ValidationError exception"""
        exc = ValidationError("Invalid input data")
        assert exc.detail == "Invalid input data"
        assert isinstance(exc, TradingSystemException)

    def test_configuration_error(self):
        """Test ConfigurationError exception"""
        exc = ConfigurationError("Missing configuration")
        assert exc.detail == "Missing configuration"
        assert isinstance(exc, TradingSystemException)

    def test_exception_inheritance(self):
        """Test that all custom exceptions inherit from TradingSystemException"""
        exceptions = [
            TradingError,
            MarketDataError,
            RiskManagementError,
            AuthenticationError,
            AuthorizationError,
            ValidationError,
            ConfigurationError,
        ]

        for exc_class in exceptions:
            exc = exc_class("test message")
            assert isinstance(exc, TradingSystemException)
            assert isinstance(exc, Exception)

    def test_exception_with_none_values(self):
        """Test exception handling with None values"""
        exc = TradingSystemException(
            "test",
            error_code=None,
            status_code=None,
            additional_context=None
        )
        assert exc.detail == "test"
        assert exc.error_code is None
        assert exc.additional_context == {}

    def test_exception_string_representation(self):
        """Test string representation of exceptions"""
        exc = TradingError("Test error message")
        assert str(exc) == "Test error message"

    def test_exception_with_complex_detail(self):
        """Test exception with complex detail object"""
        detail = {"error": "Complex error", "code": 500}
        exc = TradingSystemException(str(detail))
        assert str(detail) in str(exc)