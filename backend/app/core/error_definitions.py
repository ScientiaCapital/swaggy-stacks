"""
Error definitions and enums
Pure data structures for error classification
"""

from dataclasses import dataclass
from enum import Enum, unique
from typing import List, Optional


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
    VAL_001 = "VAL_001"  # Invalid input format
    VAL_002 = "VAL_002"  # Missing required field
    VAL_003 = "VAL_003"  # Value out of range
    VAL_004 = "VAL_004"  # Invalid symbol format
    VAL_005 = "VAL_005"  # Invalid quantity
    VAL_006 = "VAL_006"  # Invalid price
    VAL_007 = "VAL_007"  # Invalid date format
    VAL_008 = "VAL_008"  # Invalid order type
    VAL_009 = "VAL_009"  # Invalid time frame

    # Trading Errors (TRD_001-199)
    TRD_001 = "TRD_001"  # Order execution failed
    TRD_002 = "TRD_002"  # Insufficient funds
    TRD_003 = "TRD_003"  # Market closed
    TRD_004 = "TRD_004"  # Symbol not tradeable
    TRD_005 = "TRD_005"  # Position limit exceeded
    TRD_006 = "TRD_006"  # Risk limit exceeded
    TRD_007 = "TRD_007"  # Order not found
    TRD_008 = "TRD_008"  # Order already filled
    TRD_009 = "TRD_009"  # Order already cancelled
    TRD_010 = "TRD_010"  # Invalid order state
    TRD_011 = "TRD_011"  # Broker API error
    TRD_012 = "TRD_012"  # Paper trading only
    TRD_013 = "TRD_013"  # Strategy execution failed
    TRD_014 = "TRD_014"  # Portfolio analysis failed
    TRD_015 = "TRD_015"  # Risk assessment failed

    # Market Data Errors (MKT_001-099)
    MKT_001 = "MKT_001"  # Market data unavailable
    MKT_002 = "MKT_002"  # Symbol not found
    MKT_003 = "MKT_003"  # Historical data unavailable
    MKT_004 = "MKT_004"  # Real-time data unavailable
    MKT_005 = "MKT_005"  # Data provider error
    MKT_006 = "MKT_006"  # Data quality issues
    MKT_007 = "MKT_007"  # Market data delayed
    MKT_008 = "MKT_008"  # Subscription required

    # Database Errors (DB_001-099)
    DB_001 = "DB_001"  # Connection failed
    DB_002 = "DB_002"  # Query timeout
    DB_003 = "DB_003"  # Transaction failed
    DB_004 = "DB_004"  # Record not found
    DB_005 = "DB_005"  # Duplicate record
    DB_006 = "DB_006"  # Constraint violation
    DB_007 = "DB_007"  # Migration failed
    DB_008 = "DB_008"  # Backup failed

    # Network Errors (NET_001-099)
    NET_001 = "NET_001"  # Connection timeout
    NET_002 = "NET_002"  # Connection refused
    NET_003 = "NET_003"  # Network unreachable
    NET_004 = "NET_004"  # DNS resolution failed
    NET_005 = "NET_005"  # SSL/TLS error
    NET_006 = "NET_006"  # Proxy error

    # System Errors (SYS_001-099)
    SYS_001 = "SYS_001"  # Internal server error
    SYS_002 = "SYS_002"  # Service unavailable
    SYS_003 = "SYS_003"  # Memory error
    SYS_004 = "SYS_004"  # File system error
    SYS_005 = "SYS_005"  # Configuration error
    SYS_006 = "SYS_006"  # Service dependency error
    SYS_007 = "SYS_007"  # Rate limit exceeded
    SYS_008 = "SYS_008"  # Maintenance mode

    # MCP (Model Context Protocol) Errors (MCP_001-099)
    MCP_001 = "MCP_001"  # MCP connection failed
    MCP_002 = "MCP_002"  # MCP timeout
    MCP_003 = "MCP_003"  # MCP invalid response
    MCP_004 = "MCP_004"  # MCP service unavailable
    MCP_005 = "MCP_005"  # MCP authentication failed
    MCP_006 = "MCP_006"  # MCP rate limit exceeded

    # API Errors (API_001-099)
    API_001 = "API_001"  # Invalid API version
    API_002 = "API_002"  # API key missing
    API_003 = "API_003"  # API key invalid
    API_004 = "API_004"  # API quota exceeded
    API_005 = "API_005"  # Request too large
    API_006 = "API_006"  # Invalid content type
    API_007 = "API_007"  # Malformed request
    API_008 = "API_008"  # Resource not found