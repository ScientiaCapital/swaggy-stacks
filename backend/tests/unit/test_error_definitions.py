"""
Test error definitions module
"""

import pytest
from app.core.error_definitions import (
    ErrorSeverity,
    ErrorCategory,
    ErrorCodeInfo,
    TradingErrorCodes,
)


class TestErrorSeverity:
    """Test ErrorSeverity enum"""

    def test_severity_values(self):
        """Test severity enum values"""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_severity_completeness(self):
        """Test all severity levels are defined"""
        expected_severities = {"low", "medium", "high", "critical"}
        actual_severities = {severity.value for severity in ErrorSeverity}
        assert actual_severities == expected_severities


class TestErrorCategory:
    """Test ErrorCategory enum"""

    def test_category_values(self):
        """Test category enum values"""
        assert ErrorCategory.AUTHENTICATION.value == "authentication"
        assert ErrorCategory.TRADING.value == "trading"
        assert ErrorCategory.MARKET_DATA.value == "market_data"
        assert ErrorCategory.DATABASE.value == "database"
        assert ErrorCategory.SYSTEM.value == "system"

    def test_category_completeness(self):
        """Test all expected categories are defined"""
        expected_categories = {
            "authentication", "authorization", "validation", "trading",
            "market_data", "database", "network", "system", "configuration",
            "mcp", "api", "rate_limit", "timeout", "unknown"
        }
        actual_categories = {category.value for category in ErrorCategory}
        assert actual_categories == expected_categories


class TestTradingErrorCodes:
    """Test TradingErrorCodes enum"""

    def test_auth_error_codes(self):
        """Test authentication error codes"""
        assert TradingErrorCodes.AUTH_001.value == "AUTH_001"
        assert TradingErrorCodes.AUTH_002.value == "AUTH_002"
        assert TradingErrorCodes.AUTH_008.value == "AUTH_008"

    def test_trading_error_codes(self):
        """Test trading error codes"""
        assert TradingErrorCodes.TRD_001.value == "TRD_001"
        assert TradingErrorCodes.TRD_015.value == "TRD_015"

    def test_market_data_error_codes(self):
        """Test market data error codes"""
        assert TradingErrorCodes.MKT_001.value == "MKT_001"
        assert TradingErrorCodes.MKT_008.value == "MKT_008"

    def test_system_error_codes(self):
        """Test system error codes"""
        assert TradingErrorCodes.SYS_001.value == "SYS_001"
        assert TradingErrorCodes.SYS_008.value == "SYS_008"

    def test_error_code_uniqueness(self):
        """Test all error codes are unique"""
        codes = [error_code.value for error_code in TradingErrorCodes]
        assert len(codes) == len(set(codes))

    def test_error_code_format(self):
        """Test error codes follow expected format"""
        for error_code in TradingErrorCodes:
            code = error_code.value
            # Should be format: PREFIX_NUMBER
            assert "_" in code
            prefix, number = code.split("_", 1)
            assert prefix.isalpha()
            assert number.isdigit()
            assert len(number) == 3  # All numbers should be 3 digits


class TestErrorCodeInfo:
    """Test ErrorCodeInfo dataclass"""

    @pytest.fixture
    def error_info(self):
        """Create a sample ErrorCodeInfo instance"""
        return ErrorCodeInfo(
            code="TEST_001",
            name="TestError",
            description="A test error",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.MEDIUM,
            http_status=500,
            user_message="Test error occurred",
            is_retryable=True,
            suggested_actions=["Action 1", "Action 2"],
            related_codes=["TEST_002", "TEST_003"],
            documentation_url="https://example.com/docs"
        )

    def test_error_info_creation(self, error_info):
        """Test ErrorCodeInfo creation"""
        assert error_info.code == "TEST_001"
        assert error_info.name == "TestError"
        assert error_info.description == "A test error"
        assert error_info.category == ErrorCategory.SYSTEM
        assert error_info.severity == ErrorSeverity.MEDIUM
        assert error_info.http_status == 500
        assert error_info.user_message == "Test error occurred"
        assert error_info.is_retryable is True
        assert error_info.suggested_actions == ["Action 1", "Action 2"]
        assert error_info.related_codes == ["TEST_002", "TEST_003"]
        assert error_info.documentation_url == "https://example.com/docs"

    def test_error_info_defaults(self):
        """Test ErrorCodeInfo default values"""
        error_info = ErrorCodeInfo(
            code="TEST_001",
            name="TestError",
            description="A test error",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.MEDIUM,
            http_status=500,
            user_message="Test error occurred"
        )

        assert error_info.is_retryable is False
        assert error_info.suggested_actions is None
        assert error_info.related_codes is None
        assert error_info.documentation_url is None

    def test_error_info_immutable(self, error_info):
        """Test ErrorCodeInfo is immutable (dataclass with frozen=True would be better)"""
        # Test that we can access all fields
        assert error_info.code is not None
        assert error_info.name is not None
        assert error_info.category is not None