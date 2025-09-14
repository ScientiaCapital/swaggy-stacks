"""
Test backward compatibility of refactored error_codes module
"""

import pytest

# Test importing from the original error_codes module
from app.core.error_codes import (
    ErrorSeverity,
    ErrorCategory,
    ErrorCodeInfo,
    TradingErrorCodes,
    error_registry,
    get_error_info,
    create_error_response,
    validate_error_code,
)


class TestBackwardCompatibility:
    """Test that refactored error_codes maintains backward compatibility"""

    def test_enum_imports(self):
        """Test that enums are still importable"""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorCategory.AUTHENTICATION.value == "authentication"
        assert TradingErrorCodes.AUTH_001.value == "AUTH_001"

    def test_error_registry_available(self):
        """Test that error_registry is available"""
        assert error_registry is not None
        assert hasattr(error_registry, 'get_error_info')
        assert hasattr(error_registry, 'validate_error_code')

    def test_convenience_functions_work(self):
        """Test that convenience functions still work"""
        # Test get_error_info
        error_info = get_error_info("AUTH_001")
        assert error_info is not None
        assert error_info.code == "AUTH_001"

        # Test create_error_response
        response = create_error_response("AUTH_001")
        assert "error" in response
        assert response["error"]["code"] == "AUTH_001"

        # Test validate_error_code
        assert validate_error_code("AUTH_001") is True
        assert validate_error_code("INVALID_001") is False

    def test_error_registry_methods(self):
        """Test that error_registry methods work"""
        # Test get_error_info
        error_info = error_registry.get_error_info("TRD_001")
        assert error_info is not None
        assert error_info.name == "OrderExecutionFailed"

        # Test get_errors_by_category
        auth_errors = error_registry.get_errors_by_category(ErrorCategory.AUTHENTICATION)
        assert len(auth_errors) > 0

        # Test get_errors_by_severity
        critical_errors = error_registry.get_errors_by_severity(ErrorSeverity.CRITICAL)
        assert len(critical_errors) > 0

        # Test search_errors
        results = error_registry.search_errors("invalid")
        assert len(results) > 0

        # Test get_registry_stats
        stats = error_registry.get_registry_stats()
        assert "total_errors" in stats

    def test_dataclass_structure(self):
        """Test that ErrorCodeInfo structure is preserved"""
        error_info = get_error_info("AUTH_001")
        assert error_info is not None

        # Test all expected fields exist
        assert hasattr(error_info, 'code')
        assert hasattr(error_info, 'name')
        assert hasattr(error_info, 'description')
        assert hasattr(error_info, 'category')
        assert hasattr(error_info, 'severity')
        assert hasattr(error_info, 'http_status')
        assert hasattr(error_info, 'user_message')
        assert hasattr(error_info, 'is_retryable')
        assert hasattr(error_info, 'suggested_actions')
        assert hasattr(error_info, 'related_codes')
        assert hasattr(error_info, 'documentation_url')

    def test_existing_code_patterns_work(self):
        """Test that existing code patterns still work"""
        # Pattern 1: Direct error registry usage
        error_info = error_registry.get_error_info("SYS_001")
        assert error_info.severity == ErrorSeverity.CRITICAL

        # Pattern 2: Creating error responses
        response = create_error_response("TRD_002", {"order_id": "12345"})
        assert response["error"]["context"]["order_id"] == "12345"

        # Pattern 3: Error code validation
        valid_codes = ["AUTH_001", "TRD_001", "MKT_001", "SYS_001"]
        for code in valid_codes:
            assert validate_error_code(code)

    def test_error_code_definitions_complete(self):
        """Test that all expected error codes are defined"""
        # Test auth codes
        for i in range(1, 9):  # AUTH_001 to AUTH_008
            code = f"AUTH_{i:03d}"
            assert hasattr(TradingErrorCodes, code.replace("_", "_"))

        # Test some key trading codes
        key_trading_codes = ["TRD_001", "TRD_002", "TRD_003"]
        for code in key_trading_codes:
            error_info = get_error_info(code)
            assert error_info is not None