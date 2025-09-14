"""
Test error response utilities
"""

import pytest
from app.core.error_responses import (
    ErrorResponseBuilder,
    get_error_info,
    create_error_response,
    validate_error_code,
)
from app.core.error_registry import ErrorCodeRegistry
from app.core.error_definitions import ErrorCategory, ErrorSeverity


class TestErrorResponseBuilder:
    """Test ErrorResponseBuilder functionality"""

    @pytest.fixture
    def registry(self):
        """Create a fresh error registry"""
        return ErrorCodeRegistry()

    @pytest.fixture
    def builder(self, registry):
        """Create an error response builder"""
        return ErrorResponseBuilder(registry)

    def test_get_error_response_existing(self, builder):
        """Test getting response for existing error code"""
        response = builder.get_error_response("AUTH_001")

        assert "error" in response
        error = response["error"]

        assert error["code"] == "AUTH_001"
        assert error["name"] == "InvalidCredentials"
        assert "message" in error
        assert error["category"] == "authentication"
        assert error["severity"] == "low"
        assert error["retryable"] is False
        assert error["http_status"] == 401

    def test_get_error_response_with_context(self, builder):
        """Test getting response with additional context"""
        context = {"user_id": "12345", "timestamp": "2024-01-01T12:00:00Z"}
        response = builder.get_error_response("AUTH_001", context)

        assert "error" in response
        assert response["error"]["context"] == context

    def test_get_error_response_with_actions(self, builder):
        """Test response includes suggested actions when available"""
        response = builder.get_error_response("AUTH_001")

        assert "error" in response
        error = response["error"]

        if "suggested_actions" in error:
            assert isinstance(error["suggested_actions"], list)
            assert len(error["suggested_actions"]) > 0

    def test_get_error_response_with_related_codes(self, builder):
        """Test response includes related codes when available"""
        response = builder.get_error_response("AUTH_001")

        assert "error" in response
        error = response["error"]

        if "related_codes" in error:
            assert isinstance(error["related_codes"], list)

    def test_get_error_response_unknown_code(self, builder):
        """Test getting response for unknown error code"""
        response = builder.get_error_response("UNKNOWN_999")

        assert "error" in response
        error = response["error"]

        assert error["code"] == "UNKNOWN_999"
        assert error["name"] == "UnknownError"
        assert error["category"] == "unknown"
        assert error["severity"] == "medium"
        assert error["retryable"] is False
        assert error["http_status"] == 500
        assert "Contact support" in error["suggested_actions"]

    def test_trading_error_response(self, builder):
        """Test trading error response structure"""
        response = builder.get_error_response("TRD_001")

        assert "error" in response
        error = response["error"]

        assert error["code"] == "TRD_001"
        assert error["category"] == "trading"
        assert error["severity"] == "high"
        assert error["retryable"] is True  # Trading errors often retryable

    def test_system_error_response(self, builder):
        """Test system error response structure"""
        response = builder.get_error_response("SYS_001")

        assert "error" in response
        error = response["error"]

        assert error["code"] == "SYS_001"
        assert error["category"] == "system"
        assert error["severity"] == "critical"
        assert error["http_status"] == 500


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_get_error_info_function(self):
        """Test global get_error_info function"""
        error_info = get_error_info("AUTH_001")

        assert error_info is not None
        assert error_info.code == "AUTH_001"
        assert error_info.name == "InvalidCredentials"

    def test_get_error_info_nonexistent(self):
        """Test getting non-existent error info"""
        error_info = get_error_info("NONEXISTENT_001")
        assert error_info is None

    def test_create_error_response_function(self):
        """Test global create_error_response function"""
        response = create_error_response("AUTH_001")

        assert "error" in response
        assert response["error"]["code"] == "AUTH_001"

    def test_create_error_response_with_context(self):
        """Test create_error_response with context"""
        context = {"user_id": "test123"}
        response = create_error_response("AUTH_001", context)

        assert "error" in response
        assert response["error"]["context"] == context

    def test_validate_error_code_function(self):
        """Test global validate_error_code function"""
        assert validate_error_code("AUTH_001") is True
        assert validate_error_code("TRD_001") is True
        assert validate_error_code("INVALID_001") is False

    def test_response_consistency(self):
        """Test that responses are consistent across different methods"""
        # Test same error code returns same structure
        response1 = create_error_response("AUTH_001")
        response2 = create_error_response("AUTH_001")

        assert response1["error"]["code"] == response2["error"]["code"]
        assert response1["error"]["name"] == response2["error"]["name"]
        assert response1["error"]["category"] == response2["error"]["category"]

    def test_error_response_structure_completeness(self):
        """Test error responses have all required fields"""
        required_fields = [
            "code", "name", "message", "category",
            "severity", "retryable", "http_status"
        ]

        response = create_error_response("AUTH_001")
        error = response["error"]

        for field in required_fields:
            assert field in error, f"Missing required field: {field}"

    def test_multiple_error_categories(self):
        """Test responses for different error categories"""
        test_codes = [
            ("AUTH_001", "authentication"),
            ("TRD_001", "trading"),
            ("MKT_001", "market_data"),
            ("SYS_001", "system"),
        ]

        for code, expected_category in test_codes:
            response = create_error_response(code)
            assert response["error"]["category"] == expected_category

    def test_error_severity_mapping(self):
        """Test error severity is properly mapped"""
        # Test different severity levels
        critical_response = create_error_response("SYS_001")
        assert critical_response["error"]["severity"] == "critical"

        high_response = create_error_response("TRD_001")
        assert high_response["error"]["severity"] == "high"

        low_response = create_error_response("AUTH_001")
        assert low_response["error"]["severity"] == "low"