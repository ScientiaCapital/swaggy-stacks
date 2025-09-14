"""
Error code registry system
Registration and lookup logic for error codes
"""

from typing import Dict, List, Optional, Any

from app.core.error_definitions import (
    ErrorCodeInfo,
    ErrorCategory,
    ErrorSeverity,
    TradingErrorCodes,
)


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
            ["AUTH_004", "AUTH_005"],
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
            ["AUTH_003"],
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
            ["Obtain a new authentication token", "Check token format"],
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
            ["TRD_003", "TRD_004", "TRD_011"],
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
            [
                "Add funds to your account",
                "Reduce order quantity",
                "Check account balance",
            ],
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
            ["Retry after a short delay", "Check market hours", "Verify symbol"],
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
            [
                "Check database connectivity",
                "Verify connection parameters",
                "Check database status",
            ],
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
            ["Contact support", "Try again later", "Report the issue"],
        )

        # Add remaining error codes with basic information
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
        documentation_url: Optional[str] = None,
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
            documentation_url=documentation_url,
        )

    def _register_remaining_errors(self):
        """Register remaining error codes with basic information"""
        remaining_codes = [
            (
                TradingErrorCodes.VAL_001.value,
                "InvalidInputFormat",
                ErrorCategory.VALIDATION,
                400,
            ),
            (
                TradingErrorCodes.VAL_002.value,
                "MissingRequiredField",
                ErrorCategory.VALIDATION,
                400,
            ),
            (
                TradingErrorCodes.TRD_003.value,
                "MarketClosed",
                ErrorCategory.TRADING,
                422,
            ),
            (
                TradingErrorCodes.MKT_002.value,
                "SymbolNotFound",
                ErrorCategory.MARKET_DATA,
                404,
            ),
            (
                TradingErrorCodes.API_001.value,
                "InvalidAPIVersion",
                ErrorCategory.API,
                400,
            ),
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
                    category
                    in [
                        ErrorCategory.NETWORK,
                        ErrorCategory.DATABASE,
                        ErrorCategory.SYSTEM,
                    ],
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
            if (
                pattern in info.code.lower()
                or pattern in info.name.lower()
                or pattern in info.description.lower()
            ):
                results.append(info)

        return results

    def validate_error_code(self, code: str) -> bool:
        """Validate if an error code exists in the registry"""
        return code in self._registry

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the error registry"""
        total_errors = len(self._registry)
        category_counts = {}
        severity_counts = {}

        for info in self._registry.values():
            category_counts[info.category.value] = (
                category_counts.get(info.category.value, 0) + 1
            )
            severity_counts[info.severity.value] = (
                severity_counts.get(info.severity.value, 0) + 1
            )

        return {
            "total_errors": total_errors,
            "categories": category_counts,
            "severities": severity_counts,
            "retryable_errors": len(
                [info for info in self._registry.values() if info.is_retryable]
            ),
        }