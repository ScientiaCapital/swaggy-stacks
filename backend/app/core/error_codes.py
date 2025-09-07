"""
Error Code Registry System
Centralized error code management for consistent error handling across the trading system
"""
from enum import Enum, unique
from typing import Dict, Any, Optional, List, NamedTuple
from dataclasses import dataclass
import re


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    TRADING = "trading"
    MARKET_DATA = "market_data"
    DATABASE = "database"
    NETWORK = "network"
    SYSTEM = "system"
    CONFIGURATION = "configuration"
    MCP = "mcp"
    API = "api"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ErrorCodeInfo:
    """Information about a specific error code"""
    code: str
    name: str
    description: str
    category: ErrorCategory
    severity: ErrorSeverity
    http_status: int
    user_message: str
    is_retryable: bool = False
    suggested_actions: Optional[List[str]] = None
    related_codes: Optional[List[str]] = None
    documentation_url: Optional[str] = None


@unique
class TradingErrorCodes(Enum):
    """Trading system error codes"""
    
    # Authentication Errors (AUTH_001-099)
    AUTH_001 = "AUTH_001"  # Invalid credentials
    AUTH_002 = "AUTH_002"  # Token expired
    AUTH_003 = "AUTH_003"  # Token invalid
    AUTH_004 = "AUTH_004"  # User not found
    AUTH_005 = "AUTH_005"  # Account disabled
    AUTH_006 = "AUTH_006"  # Password too weak
    AUTH_007 = "AUTH_007"  # Rate limit exceeded
    AUTH_008 = "AUTH_008"  # Multi-factor authentication required
    
    # Authorization Errors (AUTHZ_001-099)
    AUTHZ_001 = "AUTHZ_001"  # Insufficient permissions
    AUTHZ_002 = "AUTHZ_002"  # Resource access denied
    AUTHZ_003 = "AUTHZ_003"  # Admin privileges required
    AUTHZ_004 = "AUTHZ_004"  # Trading permissions required
    
    # Validation Errors (VAL_001-099)
    VAL_001 = "VAL_001"    # Invalid input format
    VAL_002 = "VAL_002"    # Missing required field
    VAL_003 = "VAL_003"    # Value out of range
    VAL_004 = "VAL_004"    # Invalid symbol format
    VAL_005 = "VAL_005"    # Invalid quantity
    VAL_006 = "VAL_006"    # Invalid price
    VAL_007 = "VAL_007"    # Invalid date format
    VAL_008 = "VAL_008"    # Invalid order type
    VAL_009 = "VAL_009"    # Invalid time frame
    
    # Trading Errors (TRD_001-199)
    TRD_001 = "TRD_001"    # Order execution failed
    TRD_002 = "TRD_002"    # Insufficient funds
    TRD_003 = "TRD_003"    # Market closed
    TRD_004 = "TRD_004"    # Symbol not tradeable
    TRD_005 = "TRD_005"    # Position limit exceeded
    TRD_006 = "TRD_006"    # Risk limit exceeded
    TRD_007 = "TRD_007"    # Order not found
    TRD_008 = "TRD_008"    # Order already filled
    TRD_009 = "TRD_009"    # Order already cancelled
    TRD_010 = "TRD_010"    # Invalid order state
    TRD_011 = "TRD_011"    # Broker API error
    TRD_012 = "TRD_012"    # Paper trading only
    TRD_013 = "TRD_013"    # Strategy execution failed
    TRD_014 = "TRD_014"    # Portfolio analysis failed
    TRD_015 = "TRD_015"    # Risk assessment failed
    
    # Market Data Errors (MKT_001-099)
    MKT_001 = "MKT_001"    # Market data unavailable
    MKT_002 = "MKT_002"    # Symbol not found
    MKT_003 = "MKT_003"    # Historical data unavailable
    MKT_004 = "MKT_004"    # Real-time data unavailable
    MKT_005 = "MKT_005"    # Data provider error
    MKT_006 = "MKT_006"    # Data quality issues
    MKT_007 = "MKT_007"    # Market data delayed
    MKT_008 = "MKT_008"    # Subscription required
    
    # Database Errors (DB_001-099)
    DB_001 = "DB_001"      # Connection failed
    DB_002 = "DB_002"      # Query timeout
    DB_003 = "DB_003"      # Transaction failed
    DB_004 = "DB_004"      # Record not found
    DB_005 = "DB_005"      # Duplicate record
    DB_006 = "DB_006"      # Constraint violation
    DB_007 = "DB_007"      # Migration failed
    DB_008 = "DB_008"      # Backup failed
    
    # Network Errors (NET_001-099)
    NET_001 = "NET_001"    # Connection timeout
    NET_002 = "NET_002"    # Connection refused
    NET_003 = "NET_003"    # Network unreachable
    NET_004 = "NET_004"    # DNS resolution failed
    NET_005 = "NET_005"    # SSL/TLS error
    NET_006 = "NET_006"    # Proxy error
    
    # System Errors (SYS_001-099)
    SYS_001 = "SYS_001"    # Internal server error
    SYS_002 = "SYS_002"    # Service unavailable
    SYS_003 = "SYS_003"    # Memory error
    SYS_004 = "SYS_004"    # File system error
    SYS_005 = "SYS_005"    # Configuration error
    SYS_006 = "SYS_006"    # Service dependency error
    SYS_007 = "SYS_007"    # Rate limit exceeded
    SYS_008 = "SYS_008"    # Maintenance mode
    
    # MCP (Model Context Protocol) Errors (MCP_001-099)
    MCP_001 = "MCP_001"    # MCP connection failed
    MCP_002 = "MCP_002"    # MCP timeout
    MCP_003 = "MCP_003"    # MCP invalid response
    MCP_004 = "MCP_004"    # MCP service unavailable
    MCP_005 = "MCP_005"    # MCP authentication failed
    MCP_006 = "MCP_006"    # MCP rate limit exceeded
    
    # API Errors (API_001-099)
    API_001 = "API_001"    # Invalid API version
    API_002 = "API_002"    # API key missing
    API_003 = "API_003"    # API key invalid
    API_004 = "API_004"    # API quota exceeded
    API_005 = "API_005"    # Request too large
    API_006 = "API_006"    # Invalid content type
    API_007 = "API_007"    # Malformed request
    API_008 = "API_008"    # Resource not found


class ErrorCodeRegistry:
    """Central registry for error codes and their metadata"""
    
    def __init__(self):
        self._registry: Dict[str, ErrorCodeInfo] = {}
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize the error code registry with all error codes"""
        
        # Authentication Errors
        self._register_error(
            TradingErrorCodes.AUTH_001.value,
            "InvalidCredentials",
            "The provided username or password is incorrect",
            ErrorCategory.AUTHENTICATION,
            ErrorSeverity.LOW,
            401,
            "Invalid username or password. Please check your credentials and try again.",
            False,
            ["Check username and password", "Reset password if forgotten"],
            ["AUTH_004", "AUTH_005"]
        )
        
        self._register_error(
            TradingErrorCodes.AUTH_002.value,
            "TokenExpired",
            "The authentication token has expired",
            ErrorCategory.AUTHENTICATION,
            ErrorSeverity.LOW,
            401,
            "Your session has expired. Please log in again.",
            False,
            ["Refresh the authentication token", "Log in again"],
            ["AUTH_003"]
        )
        
        self._register_error(
            TradingErrorCodes.AUTH_003.value,
            "TokenInvalid",
            "The authentication token is invalid or malformed",
            ErrorCategory.AUTHENTICATION,
            ErrorSeverity.MEDIUM,
            401,
            "Authentication token is invalid. Please log in again.",
            False,
            ["Obtain a new authentication token", "Check token format"]
        )
        
        # Trading Errors
        self._register_error(
            TradingErrorCodes.TRD_001.value,
            "OrderExecutionFailed",
            "Failed to execute the trading order",
            ErrorCategory.TRADING,
            ErrorSeverity.HIGH,
            422,
            "Unable to execute your order. Please try again or contact support.",
            True,
            ["Retry the order", "Check market conditions", "Verify order parameters"],
            ["TRD_003", "TRD_004", "TRD_011"]
        )
        
        self._register_error(
            TradingErrorCodes.TRD_002.value,
            "InsufficientFunds",
            "Account has insufficient funds for the requested transaction",
            ErrorCategory.TRADING,
            ErrorSeverity.MEDIUM,
            422,
            "Insufficient funds to complete this transaction.",
            False,
            ["Add funds to your account", "Reduce order quantity", "Check account balance"]
        )
        
        # Market Data Errors
        self._register_error(
            TradingErrorCodes.MKT_001.value,
            "MarketDataUnavailable",
            "Market data is currently unavailable",
            ErrorCategory.MARKET_DATA,
            ErrorSeverity.MEDIUM,
            503,
            "Market data is temporarily unavailable. Please try again later.",
            True,
            ["Retry after a short delay", "Check market hours", "Verify symbol"]
        )
        
        # Database Errors
        self._register_error(
            TradingErrorCodes.DB_001.value,
            "DatabaseConnectionFailed",
            "Failed to connect to the database",
            ErrorCategory.DATABASE,
            ErrorSeverity.CRITICAL,
            503,
            "Service temporarily unavailable. Please try again later.",
            True,
            ["Check database connectivity", "Verify connection parameters", "Check database status"]
        )
        
        # System Errors
        self._register_error(
            TradingErrorCodes.SYS_001.value,
            "InternalServerError",
            "An unexpected internal error occurred",
            ErrorCategory.SYSTEM,
            ErrorSeverity.CRITICAL,
            500,
            "An unexpected error occurred. Our team has been notified.",
            False,
            ["Contact support", "Try again later", "Report the issue"]
        )
        
        # Add more error codes as needed...
        self._register_remaining_errors()
    
    def _register_error(
        self,
        code: str,
        name: str,
        description: str,
        category: ErrorCategory,
        severity: ErrorSeverity,
        http_status: int,
        user_message: str,
        is_retryable: bool = False,
        suggested_actions: Optional[List[str]] = None,
        related_codes: Optional[List[str]] = None,
        documentation_url: Optional[str] = None
    ):
        """Register an error code with its metadata"""
        self._registry[code] = ErrorCodeInfo(
            code=code,
            name=name,
            description=description,
            category=category,
            severity=severity,
            http_status=http_status,
            user_message=user_message,
            is_retryable=is_retryable,
            suggested_actions=suggested_actions or [],
            related_codes=related_codes or [],
            documentation_url=documentation_url
        )
    
    def _register_remaining_errors(self):
        """Register remaining error codes with basic information"""
        # This would typically be expanded with full error definitions
        # For now, adding basic definitions for demonstration
        
        remaining_codes = [
            (TradingErrorCodes.VAL_001.value, "InvalidInputFormat", ErrorCategory.VALIDATION, 400),
            (TradingErrorCodes.VAL_002.value, "MissingRequiredField", ErrorCategory.VALIDATION, 400),
            (TradingErrorCodes.TRD_003.value, "MarketClosed", ErrorCategory.TRADING, 422),
            (TradingErrorCodes.MKT_002.value, "SymbolNotFound", ErrorCategory.MARKET_DATA, 404),
            (TradingErrorCodes.API_001.value, "InvalidAPIVersion", ErrorCategory.API, 400),
        ]
        
        for code, name, category, status in remaining_codes:
            if code not in self._registry:
                self._register_error(
                    code,
                    name,
                    f"Error code {code}: {name}",
                    category,
                    ErrorSeverity.MEDIUM,
                    status,
                    f"An error occurred: {name}. Please try again.",
                    category in [ErrorCategory.NETWORK, ErrorCategory.DATABASE, ErrorCategory.SYSTEM]
                )
    
    def get_error_info(self, code: str) -> Optional[ErrorCodeInfo]:
        """Get error information by code"""
        return self._registry.get(code)
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorCodeInfo]:
        """Get all errors in a specific category"""
        return [info for info in self._registry.values() if info.category == category]
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ErrorCodeInfo]:
        """Get all errors with a specific severity"""
        return [info for info in self._registry.values() if info.severity == severity]
    
    def search_errors(self, pattern: str) -> List[ErrorCodeInfo]:
        """Search errors by code, name, or description"""
        pattern = pattern.lower()
        results = []
        
        for info in self._registry.values():
            if (pattern in info.code.lower() or
                pattern in info.name.lower() or
                pattern in info.description.lower()):
                results.append(info)
        
        return results
    
    def validate_error_code(self, code: str) -> bool:
        """Validate if an error code exists in the registry"""
        return code in self._registry
    
    def get_error_response(self, code: str, additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get a standardized error response for an error code"""
        error_info = self.get_error_info(code)
        if not error_info:
            return self._get_unknown_error_response(code)
        
        response = {
            "error": {
                "code": error_info.code,
                "name": error_info.name,
                "message": error_info.user_message,
                "category": error_info.category.value,
                "severity": error_info.severity.value,
                "retryable": error_info.is_retryable,
                "http_status": error_info.http_status,
            }
        }
        
        if error_info.suggested_actions:
            response["error"]["suggested_actions"] = error_info.suggested_actions
        
        if error_info.related_codes:
            response["error"]["related_codes"] = error_info.related_codes
        
        if additional_context:
            response["error"]["context"] = additional_context
        
        return response
    
    def _get_unknown_error_response(self, code: str) -> Dict[str, Any]:
        """Get response for unknown error codes"""
        return {
            "error": {
                "code": code,
                "name": "UnknownError",
                "message": "An unknown error occurred.",
                "category": ErrorCategory.UNKNOWN.value,
                "severity": ErrorSeverity.MEDIUM.value,
                "retryable": False,
                "http_status": 500,
                "suggested_actions": ["Contact support"],
            }
        }
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the error registry"""
        total_errors = len(self._registry)
        category_counts = {}
        severity_counts = {}
        
        for info in self._registry.values():
            category_counts[info.category.value] = category_counts.get(info.category.value, 0) + 1
            severity_counts[info.severity.value] = severity_counts.get(info.severity.value, 0) + 1
        
        return {
            "total_errors": total_errors,
            "categories": category_counts,
            "severities": severity_counts,
            "retryable_errors": len([info for info in self._registry.values() if info.is_retryable]),
        }


# Global error code registry instance
error_registry = ErrorCodeRegistry()


def get_error_info(code: str) -> Optional[ErrorCodeInfo]:
    """Convenience function to get error information"""
    return error_registry.get_error_info(code)


def create_error_response(code: str, additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to create error response"""
    return error_registry.get_error_response(code, additional_context)


def validate_error_code(code: str) -> bool:
    """Convenience function to validate error code"""
    return error_registry.validate_error_code(code)