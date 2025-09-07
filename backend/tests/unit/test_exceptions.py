"""
Unit tests for custom exceptions
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
    MCPError,
    MCPConnectionError,
    MCPTimeoutError
)


class TestTradingSystemExceptions:
    """Test suite for trading system exceptions"""
    
    def test_base_trading_system_exception(self):
        """Test base TradingSystemException"""
        error = TradingSystemException("Test error", "TEST_001", 500)
        
        assert str(error) == "Test error"
        assert error.detail == "Test error"
        assert error.error_code == "TEST_001"
        assert error.status_code == 500
    
    def test_base_exception_defaults(self):
        """Test base exception with default values"""
        error = TradingSystemException("Test error")
        
        assert error.detail == "Test error"
        assert error.error_code is None
        assert error.status_code == 500
    
    def test_trading_error(self):
        """Test TradingError specific behavior"""
        error = TradingError("Order failed", "TRADE_001")
        
        assert str(error) == "Order failed"
        assert error.detail == "Order failed"
        assert error.error_code == "TRADE_001"
        assert error.status_code == 400  # Trading errors are 400
    
    def test_trading_error_defaults(self):
        """Test TradingError with defaults"""
        error = TradingError("Order failed")
        
        assert error.detail == "Order failed"
        assert error.error_code is None
        assert error.status_code == 400
    
    def test_market_data_error(self):
        """Test MarketDataError specific behavior"""
        error = MarketDataError("Data unavailable", "MARKET_001")
        
        assert str(error) == "Data unavailable"
        assert error.detail == "Data unavailable"
        assert error.error_code == "MARKET_001"
        assert error.status_code == 503  # Service unavailable
    
    def test_risk_management_error(self):
        """Test RiskManagementError specific behavior"""
        error = RiskManagementError("Risk limit exceeded", "RISK_001")
        
        assert str(error) == "Risk limit exceeded"
        assert error.detail == "Risk limit exceeded"
        assert error.error_code == "RISK_001"
        assert error.status_code == 400
    
    def test_authentication_error(self):
        """Test AuthenticationError specific behavior"""
        error = AuthenticationError("Invalid credentials", "AUTH_001")
        
        assert str(error) == "Invalid credentials"
        assert error.detail == "Invalid credentials"
        assert error.error_code == "AUTH_001"
        assert error.status_code == 401
    
    def test_authorization_error(self):
        """Test AuthorizationError specific behavior"""
        error = AuthorizationError("Access denied", "AUTHZ_001")
        
        assert str(error) == "Access denied"
        assert error.detail == "Access denied"
        assert error.error_code == "AUTHZ_001"
        assert error.status_code == 403
    
    def test_validation_error(self):
        """Test ValidationError specific behavior"""
        error = ValidationError("Invalid input", "VALID_001")
        
        assert str(error) == "Invalid input"
        assert error.detail == "Invalid input"
        assert error.error_code == "VALID_001"
        assert error.status_code == 422
    
    def test_configuration_error(self):
        """Test ConfigurationError specific behavior"""
        error = ConfigurationError("Missing config", "CONFIG_001")
        
        assert str(error) == "Missing config"
        assert error.detail == "Missing config"
        assert error.error_code == "CONFIG_001"
        assert error.status_code == 500
    
    def test_mcp_error(self):
        """Test MCPError specific behavior"""
        error = MCPError("MCP server error", "MCP_001")
        
        assert str(error) == "MCP server error"
        assert error.detail == "MCP server error"
        assert error.error_code == "MCP_001"
        assert error.status_code == 500
    
    def test_mcp_connection_error(self):
        """Test MCPConnectionError specific behavior"""
        error = MCPConnectionError("Connection failed", "MCP_CONN_001")
        
        assert str(error) == "Connection failed"
        assert error.detail == "Connection failed"
        assert error.error_code == "MCP_CONN_001"
        assert error.status_code == 503  # Service unavailable
        assert isinstance(error, MCPError)  # Inheritance check
    
    def test_mcp_timeout_error(self):
        """Test MCPTimeoutError specific behavior"""
        error = MCPTimeoutError("Request timeout", "MCP_TIMEOUT_001")
        
        assert str(error) == "Request timeout"
        assert error.detail == "Request timeout"
        assert error.error_code == "MCP_TIMEOUT_001"
        assert error.status_code == 504  # Gateway timeout
        assert isinstance(error, MCPError)  # Inheritance check
    
    def test_exception_inheritance(self):
        """Test that all exceptions inherit from TradingSystemException"""
        exceptions = [
            TradingError("test"),
            MarketDataError("test"),
            RiskManagementError("test"),
            AuthenticationError("test"),
            AuthorizationError("test"),
            ValidationError("test"),
            ConfigurationError("test"),
            MCPError("test"),
            MCPConnectionError("test"),
            MCPTimeoutError("test")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, TradingSystemException)
            assert isinstance(exc, Exception)
    
    def test_exception_with_none_values(self):
        """Test exceptions handle None values gracefully"""
        error = TradingSystemException(None, None, None)
        
        assert error.detail is None
        assert error.error_code is None
        assert error.status_code is None
    
    def test_exception_string_representation(self):
        """Test string representation of exceptions"""
        error = TradingError("Test trading error", "TRADE_TEST")
        
        # Should use the detail as string representation
        assert str(error) == "Test trading error"
        
        # Should be able to format in f-strings
        formatted = f"Error occurred: {error}"
        assert formatted == "Error occurred: Test trading error"
    
    def test_exception_with_complex_detail(self):
        """Test exceptions with complex detail messages"""
        complex_detail = "Order submission failed: Symbol 'AAPL', Quantity: 100, Side: 'buy', Reason: 'Insufficient buying power'"
        error = TradingError(complex_detail, "TRADE_COMPLEX")
        
        assert error.detail == complex_detail
        assert str(error) == complex_detail
        assert error.error_code == "TRADE_COMPLEX"
    
    def test_exception_status_code_mapping(self):
        """Test that status codes are properly mapped"""
        status_code_mapping = {
            TradingError: 400,
            MarketDataError: 503,
            RiskManagementError: 400,
            AuthenticationError: 401,
            AuthorizationError: 403,
            ValidationError: 422,
            ConfigurationError: 500,
            MCPError: 500,
            MCPConnectionError: 503,
            MCPTimeoutError: 504
        }
        
        for exception_class, expected_status in status_code_mapping.items():
            error = exception_class("Test error")
            assert error.status_code == expected_status
    
    def test_nested_exception_handling(self):
        """Test exceptions can be properly nested/chained"""
        original_error = ValueError("Original error")
        
        try:
            raise original_error
        except ValueError as e:
            trading_error = TradingError(f"Trading failed: {str(e)}", "NESTED_001")
            
            assert "Original error" in str(trading_error)
            assert trading_error.error_code == "NESTED_001"
    
    def test_exception_equality(self):
        """Test exception equality comparison"""
        error1 = TradingError("Same error", "SAME_001")
        error2 = TradingError("Same error", "SAME_001")
        error3 = TradingError("Different error", "DIFF_001")
        
        # Exceptions should be compared by content, not identity
        assert str(error1) == str(error2)
        assert str(error1) != str(error3)
        assert error1.error_code == error2.error_code
        assert error1.error_code != error3.error_code


class TestExceptionUsagePatterns:
    """Test common exception usage patterns"""
    
    def test_exception_in_try_catch(self):
        """Test exceptions work properly in try/catch blocks"""
        def risky_operation():
            raise TradingError("Operation failed", "RISK_001")
        
        with pytest.raises(TradingError) as exc_info:
            risky_operation()
        
        assert exc_info.value.detail == "Operation failed"
        assert exc_info.value.error_code == "RISK_001"
        assert exc_info.value.status_code == 400
    
    def test_exception_reraise_pattern(self):
        """Test exception re-raising with additional context"""
        def low_level_operation():
            raise ValueError("Low level error")
        
        def high_level_operation():
            try:
                low_level_operation()
            except ValueError as e:
                raise TradingError(f"High level error: {str(e)}", "HIGH_LEVEL") from e
        
        with pytest.raises(TradingError) as exc_info:
            high_level_operation()
        
        assert "High level error: Low level error" in str(exc_info.value)
        assert exc_info.value.error_code == "HIGH_LEVEL"
        assert isinstance(exc_info.value.__cause__, ValueError)
    
    def test_multiple_exception_types(self):
        """Test handling multiple exception types"""
        def operation_with_multiple_failures(failure_type):
            if failure_type == "auth":
                raise AuthenticationError("Auth failed")
            elif failure_type == "validation":
                raise ValidationError("Validation failed")
            elif failure_type == "mcp":
                raise MCPConnectionError("MCP failed")
            else:
                raise TradingError("Generic trading error")
        
        # Test different exception types
        with pytest.raises(AuthenticationError):
            operation_with_multiple_failures("auth")
        
        with pytest.raises(ValidationError):
            operation_with_multiple_failures("validation")
        
        with pytest.raises(MCPConnectionError):
            operation_with_multiple_failures("mcp")
        
        with pytest.raises(TradingError):
            operation_with_multiple_failures("other")
    
    def test_exception_logging_context(self):
        """Test exceptions provide good context for logging"""
        error = MarketDataError(
            "Failed to fetch market data for AAPL: API rate limit exceeded",
            "MARKET_RATE_LIMIT"
        )
        
        # Should contain enough context for debugging
        assert "AAPL" in str(error)
        assert "rate limit" in str(error)
        assert error.error_code == "MARKET_RATE_LIMIT"
        assert error.status_code == 503  # Service unavailable due to rate limiting