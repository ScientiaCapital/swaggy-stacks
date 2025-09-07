"""
Custom exceptions for the trading system
"""

from typing import Optional


class TradingSystemException(Exception):
    """Base exception for trading system errors with error code registry integration"""

    def __init__(
        self, 
        detail: str, 
        error_code: Optional[str] = None, 
        status_code: Optional[int] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        from app.core.error_codes import get_error_info
        
        self.detail = detail
        self.error_code = error_code
        self.additional_context = additional_context or {}
        
        # Get error info from registry if error code is provided
        if error_code:
            error_info = get_error_info(error_code)
            if error_info:
                self.status_code = status_code or error_info.http_status
                self.error_name = error_info.name
                self.error_category = error_info.category.value
                self.error_severity = error_info.severity.value
                self.is_retryable = error_info.is_retryable
                self.suggested_actions = error_info.suggested_actions
                self.user_message = error_info.user_message
            else:
                self.status_code = status_code or 500
                self.error_name = "UnknownError"
                self.error_category = "unknown"
                self.error_severity = "medium"
                self.is_retryable = False
                self.suggested_actions = []
                self.user_message = detail
        else:
            self.status_code = status_code or 500
            self.error_name = "GenericError"
            self.error_category = "unknown"
            self.error_severity = "medium"
            self.is_retryable = False
            self.suggested_actions = []
            self.user_message = detail
        
        super().__init__(detail)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        error_dict = {
            "code": self.error_code,
            "name": self.error_name,
            "message": self.user_message,
            "detail": self.detail,
            "category": self.error_category,
            "severity": self.error_severity,
            "retryable": self.is_retryable,
            "http_status": self.status_code,
        }
        
        if self.suggested_actions:
            error_dict["suggested_actions"] = self.suggested_actions
            
        if self.additional_context:
            error_dict["context"] = self.additional_context
            
        return error_dict


class TradingError(TradingSystemException):
    """Trading-specific errors"""

    def __init__(self, detail: str, error_code: Optional[str] = None):
        super().__init__(detail, error_code, status_code=400)


class MarketDataError(TradingSystemException):
    """Market data related errors"""

    def __init__(self, detail: str, error_code: Optional[str] = None):
        super().__init__(detail, error_code, status_code=503)


class RiskManagementError(TradingSystemException):
    """Risk management related errors"""

    def __init__(self, detail: str, error_code: Optional[str] = None):
        super().__init__(detail, error_code, status_code=400)


class AuthenticationError(TradingSystemException):
    """Authentication related errors"""

    def __init__(self, detail: str, error_code: Optional[str] = None):
        super().__init__(detail, error_code, status_code=401)


class AuthorizationError(TradingSystemException):
    """Authorization related errors"""

    def __init__(self, detail: str, error_code: Optional[str] = None):
        super().__init__(detail, error_code, status_code=403)


class ValidationError(TradingSystemException):
    """Data validation errors"""

    def __init__(self, detail: str, error_code: Optional[str] = None):
        super().__init__(detail, error_code, status_code=422)


class ConfigurationError(TradingSystemException):
    """Configuration related errors"""

    def __init__(self, detail: str, error_code: Optional[str] = None):
        super().__init__(detail, error_code, status_code=500)


class MCPError(TradingSystemException):
    """Base MCP server errors"""

    def __init__(self, detail: str, error_code: Optional[str] = None):
        super().__init__(detail, error_code, status_code=500)


class MCPConnectionError(MCPError):
    """MCP connection errors"""

    def __init__(self, detail: str, error_code: Optional[str] = None):
        super().__init__(detail, error_code, status_code=503)


class MCPTimeoutError(MCPError):
    """MCP timeout errors"""

    def __init__(self, detail: str, error_code: Optional[str] = None):
        super().__init__(detail, error_code, status_code=504)
