"""
Error response utilities
Formatting and convenience functions for error handling
"""

from typing import Dict, Any, Optional

from app.core.error_definitions import ErrorCategory, ErrorSeverity
from app.core.error_registry import ErrorCodeRegistry


class ErrorResponseBuilder:
    """Builds standardized error responses"""

    def __init__(self, registry: ErrorCodeRegistry):
        self.registry = registry

    def get_error_response(
        self, code: str, additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get a standardized error response for an error code"""
        error_info = self.registry.get_error_info(code)
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


# Global instances
_error_registry = ErrorCodeRegistry()
_response_builder = ErrorResponseBuilder(_error_registry)


def get_error_info(code: str):
    """Convenience function to get error information"""
    return _error_registry.get_error_info(code)


def create_error_response(
    code: str, additional_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function to create error response"""
    return _response_builder.get_error_response(code, additional_context)


def validate_error_code(code: str) -> bool:
    """Convenience function to validate error code"""
    return _error_registry.validate_error_code(code)