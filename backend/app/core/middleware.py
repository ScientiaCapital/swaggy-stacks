"""
Middleware for detailed error tracking and diagnostics
"""

import time
import traceback
import uuid
from typing import Any, Dict

import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger()


class ErrorDiagnosticMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive error tracking and diagnostics"""

    def __init__(self, app, enable_detailed_errors: bool = True):
        super().__init__(app)
        self.enable_detailed_errors = enable_detailed_errors

    async def dispatch(self, request: Request, call_next):
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()

        # Log request start
        await self._log_request_start(request, request_id)

        try:
            response = await call_next(request)

            # Log successful request
            duration = time.time() - start_time
            await self._log_request_success(request, response, request_id, duration)

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as exc:
            # Log and handle error
            duration = time.time() - start_time
            error_details = await self._log_and_analyze_error(
                request, exc, request_id, duration
            )

            # Return structured error response
            return await self._create_error_response(exc, error_details, request_id)

    async def _log_request_start(self, request: Request, request_id: str):
        """Log request start with context"""
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

    async def _log_request_success(
        self, request: Request, response: Response, request_id: str, duration: float
    ):
        """Log successful request completion"""
        logger.info(
            "Request completed successfully",
            request_id=request_id,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
            method=request.method,
            path=request.url.path,
        )

    async def _log_and_analyze_error(
        self, request: Request, exc: Exception, request_id: str, duration: float
    ) -> Dict[str, Any]:
        """Log error with detailed diagnostics"""

        # Extract error details
        error_type = type(exc).__name__
        error_message = str(exc)

        # Get traceback
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        full_traceback = "".join(tb_lines)

        # Extract request context
        request_context = await self._extract_request_context(request)

        # Analyze error severity and category
        error_analysis = self._analyze_error(exc, request_context)

        # Create comprehensive error details
        error_details = {
            "request_id": request_id,
            "timestamp": time.time(),
            "duration_ms": round(duration * 1000, 2),
            "error_type": error_type,
            "error_message": error_message,
            "error_category": error_analysis["category"],
            "severity": error_analysis["severity"],
            "is_retryable": error_analysis["retryable"],
            "request_context": request_context,
            "system_context": await self._get_system_context(),
        }

        # Add traceback for detailed errors
        if self.enable_detailed_errors:
            error_details["traceback"] = full_traceback
            error_details["traceback_lines"] = tb_lines

        # Log the error with all context
        logger.error(
            "Request failed with error",
            **error_details,
            exc_info=exc if self.enable_detailed_errors else None,
        )

        return error_details

    async def _extract_request_context(self, request: Request) -> Dict[str, Any]:
        """Extract comprehensive request context"""
        context = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
            "path_params": getattr(request, "path_params", {}),
        }

        # Try to extract body for non-GET requests (be careful with large bodies)
        if request.method != "GET":
            try:
                body = await request.body()
                if len(body) < 10000:  # Only log small bodies
                    context["body_size"] = len(body)
                    if body:
                        context["body_preview"] = body[:500].decode(
                            "utf-8", errors="ignore"
                        )
                else:
                    context["body_size"] = len(body)
                    context["body_too_large"] = True
            except Exception:
                context["body_extraction_failed"] = True

        return context

    async def _get_system_context(self) -> Dict[str, Any]:
        """Get system context for error analysis"""
        import os

        import psutil

        try:
            return {
                "process_id": os.getpid(),
                "memory_usage_mb": round(
                    psutil.Process().memory_info().rss / 1024 / 1024, 2
                ),
                "cpu_percent": psutil.cpu_percent(),
                "open_files": len(psutil.Process().open_files()),
            }
        except Exception:
            return {"system_context_failed": True}

    def _analyze_error(
        self, exc: Exception, request_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze error to categorize severity and type"""

        error_type = type(exc).__name__
        error_message = str(exc).lower()
        path = request_context.get("path", "")

        # Determine category
        category = "unknown"
        if "auth" in error_message or "token" in error_message or "/auth" in path:
            category = "authentication"
        elif "permission" in error_message or "forbidden" in error_message:
            category = "authorization"
        elif "database" in error_message or "connection" in error_message:
            category = "database"
        elif "validation" in error_message or "invalid" in error_message:
            category = "validation"
        elif "timeout" in error_message:
            category = "timeout"
        elif "network" in error_message or "connection" in error_message:
            category = "network"
        elif "trading" in error_message or "/trading" in path:
            category = "trading"
        elif "market" in error_message or "/market" in path:
            category = "market_data"

        # Determine severity
        severity = "medium"
        if error_type in ["ValidationError", "ValueError", "KeyError"]:
            severity = "low"
        elif error_type in ["DatabaseError", "ConnectionError", "TimeoutError"]:
            severity = "high"
        elif error_type in ["SystemError", "MemoryError", "OSError"]:
            severity = "critical"
        elif "500" in error_message:
            severity = "high"
        elif "400" in error_message or "404" in error_message:
            severity = "low"

        # Determine if retryable
        retryable = False
        if error_type in ["ConnectionError", "TimeoutError", "TemporaryFailure"]:
            retryable = True
        elif "rate limit" in error_message or "too many requests" in error_message:
            retryable = True
        elif category in ["database", "network"]:
            retryable = True

        return {
            "category": category,
            "severity": severity,
            "retryable": retryable,
        }

    async def _create_error_response(
        self, exc: Exception, error_details: Dict[str, Any], request_id: str
    ) -> JSONResponse:
        """Create structured error response"""

        # Default status code
        status_code = 500

        # Extract status code from exception if available
        if hasattr(exc, "status_code"):
            status_code = exc.status_code
        elif "404" in str(exc):
            status_code = 404
        elif "400" in str(exc) or "validation" in str(exc).lower():
            status_code = 400
        elif "401" in str(exc) or "auth" in str(exc).lower():
            status_code = 401
        elif "403" in str(exc) or "forbidden" in str(exc).lower():
            status_code = 403

        # Create response content
        response_content = {
            "error": {
                "type": error_details["error_type"],
                "message": error_details["error_message"],
                "category": error_details["error_category"],
                "severity": error_details["severity"],
                "retryable": error_details["is_retryable"],
                "request_id": request_id,
                "timestamp": error_details["timestamp"],
            }
        }

        # Add error code if present in exception
        if hasattr(exc, "error_code") and exc.error_code:
            response_content["error"]["code"] = exc.error_code

        # Add detailed information for development
        if self.enable_detailed_errors:
            response_content["debug"] = {
                "duration_ms": error_details["duration_ms"],
                "request_context": {
                    "method": error_details["request_context"]["method"],
                    "path": error_details["request_context"]["path"],
                    "query_params": error_details["request_context"]["query_params"],
                },
                "system_context": error_details["system_context"],
            }

        return JSONResponse(
            status_code=status_code,
            content=response_content,
            headers={"X-Request-ID": request_id},
        )
