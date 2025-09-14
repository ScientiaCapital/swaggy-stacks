"""
Test error registry module
"""

import pytest
from app.core.error_registry import ErrorCodeRegistry
from app.core.error_definitions import (
    ErrorCategory,
    ErrorSeverity,
    TradingErrorCodes,
)


class TestErrorCodeRegistry:
    """Test ErrorCodeRegistry functionality"""

    @pytest.fixture
    def registry(self):
        """Create a fresh error registry"""
        return ErrorCodeRegistry()

    def test_registry_initialization(self, registry):
        """Test registry is properly initialized"""
        # Should have predefined error codes
        assert registry.get_error_info("AUTH_001") is not None
        assert registry.get_error_info("TRD_001") is not None
        assert registry.get_error_info("SYS_001") is not None

    def test_get_error_info_existing(self, registry):
        """Test getting existing error info"""
        error_info = registry.get_error_info("AUTH_001")

        assert error_info is not None
        assert error_info.code == "AUTH_001"
        assert error_info.name == "InvalidCredentials"
        assert error_info.category == ErrorCategory.AUTHENTICATION
        assert error_info.severity == ErrorSeverity.LOW
        assert error_info.http_status == 401
        assert "credentials" in error_info.user_message.lower()

    def test_get_error_info_nonexistent(self, registry):
        """Test getting non-existent error info"""
        error_info = registry.get_error_info("NONEXISTENT_001")
        assert error_info is None

    def test_validate_error_code_valid(self, registry):
        """Test validating valid error code"""
        assert registry.validate_error_code("AUTH_001") is True
        assert registry.validate_error_code("TRD_001") is True

    def test_validate_error_code_invalid(self, registry):
        """Test validating invalid error code"""
        assert registry.validate_error_code("INVALID_001") is False
        assert registry.validate_error_code("") is False
        assert registry.validate_error_code("nonsense") is False

    def test_get_errors_by_category(self, registry):
        """Test getting errors by category"""
        auth_errors = registry.get_errors_by_category(ErrorCategory.AUTHENTICATION)

        assert len(auth_errors) > 0
        for error in auth_errors:
            assert error.category == ErrorCategory.AUTHENTICATION
            assert error.code.startswith("AUTH_")

    def test_get_errors_by_severity(self, registry):
        """Test getting errors by severity"""
        critical_errors = registry.get_errors_by_severity(ErrorSeverity.CRITICAL)

        assert len(critical_errors) > 0
        for error in critical_errors:
            assert error.severity == ErrorSeverity.CRITICAL

    def test_search_errors_by_code(self, registry):
        """Test searching errors by code pattern"""
        results = registry.search_errors("auth")

        assert len(results) > 0
        for result in results:
            assert "auth" in result.code.lower() or "auth" in result.name.lower()

    def test_search_errors_by_name(self, registry):
        """Test searching errors by name pattern"""
        results = registry.search_errors("credentials")

        assert len(results) > 0
        for result in results:
            found = (
                "credentials" in result.code.lower() or
                "credentials" in result.name.lower() or
                "credentials" in result.description.lower()
            )
            assert found

    def test_search_errors_case_insensitive(self, registry):
        """Test search is case insensitive"""
        results_lower = registry.search_errors("invalid")
        results_upper = registry.search_errors("INVALID")
        results_mixed = registry.search_errors("Invalid")

        assert len(results_lower) > 0
        assert results_lower == results_upper
        assert results_lower == results_mixed

    def test_get_registry_stats(self, registry):
        """Test getting registry statistics"""
        stats = registry.get_registry_stats()

        assert "total_errors" in stats
        assert "categories" in stats
        assert "severities" in stats
        assert "retryable_errors" in stats

        assert stats["total_errors"] > 0
        assert isinstance(stats["categories"], dict)
        assert isinstance(stats["severities"], dict)
        assert isinstance(stats["retryable_errors"], int)

        # Should have authentication category
        assert "authentication" in stats["categories"]
        assert stats["categories"]["authentication"] > 0

    def test_predefined_auth_errors(self, registry):
        """Test predefined authentication errors"""
        # AUTH_001 - Invalid credentials
        auth_001 = registry.get_error_info("AUTH_001")
        assert auth_001.name == "InvalidCredentials"
        assert auth_001.http_status == 401
        assert not auth_001.is_retryable

        # AUTH_002 - Token expired
        auth_002 = registry.get_error_info("AUTH_002")
        assert auth_002.name == "TokenExpired"
        assert auth_002.http_status == 401

    def test_predefined_trading_errors(self, registry):
        """Test predefined trading errors"""
        # TRD_001 - Order execution failed
        trd_001 = registry.get_error_info("TRD_001")
        assert trd_001.name == "OrderExecutionFailed"
        assert trd_001.severity == ErrorSeverity.HIGH
        assert trd_001.is_retryable is True  # Trading errors are often retryable

        # TRD_002 - Insufficient funds
        trd_002 = registry.get_error_info("TRD_002")
        assert trd_002.name == "InsufficientFunds"
        assert trd_002.category == ErrorCategory.TRADING

    def test_predefined_system_errors(self, registry):
        """Test predefined system errors"""
        # SYS_001 - Internal server error
        sys_001 = registry.get_error_info("SYS_001")
        assert sys_001.name == "InternalServerError"
        assert sys_001.severity == ErrorSeverity.CRITICAL
        assert sys_001.http_status == 500

    def test_error_code_coverage(self, registry):
        """Test that registry covers key error codes"""
        key_codes = [
            "AUTH_001", "AUTH_002", "AUTH_003",
            "TRD_001", "TRD_002", "TRD_003",
            "MKT_001", "MKT_002",
            "DB_001", "SYS_001", "API_001"
        ]

        for code in key_codes:
            error_info = registry.get_error_info(code)
            assert error_info is not None, f"Missing error code: {code}"
            assert error_info.code == code