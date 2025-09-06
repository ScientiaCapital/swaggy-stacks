"""
Custom exceptions for the trading system
"""

from typing import Optional


class TradingSystemException(Exception):
    """Base exception for trading system errors"""
    
    def __init__(
        self,
        detail: str,
        error_code: Optional[str] = None,
        status_code: int = 500
    ):
        self.detail = detail
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(detail)


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
