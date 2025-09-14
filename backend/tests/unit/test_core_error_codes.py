"""
Test error codes module
"""

import pytest
from app.core.error_codes import (
    ErrorSeverity,
    ErrorCategory,
    ErrorCodeInfo
)


class TestErrorSeverity:
    """Test ErrorSeverity enum"""

    def test_error_severity_values(self):
        """Test ErrorSeverity enum values"""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_error_severity_members(self):
        """Test ErrorSeverity enum members"""
        severities = [e.value for e in ErrorSeverity]
        assert "low" in severities
        assert "medium" in severities
        assert "high" in severities
        assert "critical" in severities


class TestErrorCategory:
    """Test ErrorCategory enum"""

    def test_error_category_values(self):
        """Test ErrorCategory enum values"""
        assert ErrorCategory.AUTHENTICATION.value == "authentication"
        assert ErrorCategory.AUTHORIZATION.value == "authorization"
        assert ErrorCategory.VALIDATION.value == "validation"
        assert ErrorCategory.TRADING.value == "trading"
        assert ErrorCategory.MARKET_DATA.value == "market_data"

    def test_error_category_members(self):
        """Test ErrorCategory enum has expected members"""
        categories = [e.value for e in ErrorCategory]
        expected_categories = [
            "authentication", "authorization", "validation", "trading",
            "market_data", "database", "network", "system", "configuration",
            "mcp", "api", "rate_limit", "timeout", "unknown"
        ]

        for category in expected_categories:
            assert category in categories


class TestErrorCodeInfo:
    """Test ErrorCodeInfo dataclass"""

    def test_error_code_info_creation(self):
        """Test ErrorCodeInfo creation"""
        error_info = ErrorCodeInfo(
            code="TEST_001",
            name="Test Error",
            description="A test error for validation",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            http_status=400,
            user_message="Input validation failed"
        )

        assert error_info.code == "TEST_001"
        assert error_info.name == "Test Error"
        assert error_info.description == "A test error for validation"
        assert error_info.category == ErrorCategory.VALIDATION
        assert error_info.severity == ErrorSeverity.MEDIUM
        assert error_info.http_status == 400
        assert error_info.user_message == "Input validation failed"
        assert error_info.is_retryable is False  # Default value

    def test_error_code_info_with_retryable(self):
        """Test ErrorCodeInfo with retryable flag"""
        error_info = ErrorCodeInfo(
            code="NET_001",
            name="Network Error",
            description="Network timeout error",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            http_status=503,
            user_message="Service temporarily unavailable",
            is_retryable=True
        )

        assert error_info.is_retryable is True


class TestErrorCodeIntegration:
    """Test error code integration with exception system"""

    def test_error_code_info_integration(self):
        """Test ErrorCodeInfo integrates properly with enums"""
        # Test that we can create error codes with different categories and severities
        validation_error = ErrorCodeInfo(
            code="VAL_001",
            name="Validation Error",
            description="Input validation failed",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            http_status=400,
            user_message="Please check your input"
        )

        trading_error = ErrorCodeInfo(
            code="TRADE_001",
            name="Trading Error",
            description="Trade execution failed",
            category=ErrorCategory.TRADING,
            severity=ErrorSeverity.HIGH,
            http_status=500,
            user_message="Trade could not be executed"
        )

        # Verify the objects maintain their properties
        assert validation_error.category == ErrorCategory.VALIDATION
        assert trading_error.category == ErrorCategory.TRADING
        assert validation_error.severity == ErrorSeverity.MEDIUM
        assert trading_error.severity == ErrorSeverity.HIGH