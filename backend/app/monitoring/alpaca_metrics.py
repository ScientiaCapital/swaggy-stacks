"""
Custom Prometheus exporter for Alpaca API monitoring.

Tracks API performance, rate limits, and error patterns for the Alpaca trading API.
Designed for minimal overhead in high-frequency trading systems.
"""

import time
from functools import wraps
from typing import Optional, Dict, Any
from prometheus_client import Counter, Gauge, Histogram

from app.core.logging import get_logger

logger = get_logger(__name__)


class AlpacaAPIMetrics:
    """Low-overhead Prometheus metrics for Alpaca API monitoring"""

    def __init__(self, registry=None):
        """Initialize Alpaca API metrics with optimized histogram buckets"""

        # API Request Metrics
        self.api_requests_total = Counter(
            "alpaca_api_requests_total",
            "Total Alpaca API requests by endpoint",
            ["endpoint", "method"],
            registry=registry
        )

        # API Response Time with optimized buckets for trading latency
        # Buckets: 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s
        self.api_response_time = Histogram(
            "alpaca_api_response_time_seconds",
            "Alpaca API response time in seconds",
            ["endpoint", "method"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=registry
        )

        # Error Tracking
        self.api_errors_total = Counter(
            "alpaca_api_errors_total",
            "Total Alpaca API errors by type",
            ["endpoint", "error_type", "status_code"],
            registry=registry
        )

        # Rate Limit Metrics (extracted from X-RateLimit-* headers)
        self.rate_limit_remaining = Gauge(
            "alpaca_rate_limit_remaining",
            "Remaining API requests before rate limit",
            ["endpoint"],
            registry=registry
        )

        self.rate_limit_reset_seconds = Gauge(
            "alpaca_rate_limit_reset_seconds",
            "Seconds until rate limit reset",
            ["endpoint"],
            registry=registry
        )

        # Retry Metrics
        self.api_retries_total = Counter(
            "alpaca_api_retries_total",
            "Total API retry attempts",
            ["endpoint", "retry_reason"],
            registry=registry
        )

    def track_request(self, endpoint: str, method: str = "GET"):
        """
        Decorator to track Alpaca API requests with minimal overhead.

        Captures:
        - Request count
        - Response time
        - Rate limit headers
        - Errors and retries

        Args:
            endpoint: API endpoint path (e.g., '/v2/orders', '/v2/positions')
            method: HTTP method (default: GET)

        Example:
            @alpaca_metrics.track_request('/v2/orders', method='POST')
            async def place_order(self, symbol: str, qty: int):
                response = await self.api_client.post('/v2/orders', ...)
                return response
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()

                # Increment request counter
                self.api_requests_total.labels(
                    endpoint=endpoint,
                    method=method
                ).inc()

                try:
                    # Execute API call
                    result = await func(*args, **kwargs)

                    # Record successful response time
                    duration = time.time() - start_time
                    self.api_response_time.labels(
                        endpoint=endpoint,
                        method=method
                    ).observe(duration)

                    # Extract rate limit headers if available
                    if isinstance(result, dict) and '_headers' in result:
                        self._extract_rate_limits(result['_headers'], endpoint)

                    return result

                except Exception as e:
                    # Record error metrics
                    duration = time.time() - start_time
                    self.api_response_time.labels(
                        endpoint=endpoint,
                        method=method
                    ).observe(duration)

                    # Classify error type
                    error_type = self._classify_error(e)
                    status_code = getattr(e, 'status_code', 'unknown')

                    self.api_errors_total.labels(
                        endpoint=endpoint,
                        error_type=error_type,
                        status_code=str(status_code)
                    ).inc()

                    # Re-raise for caller to handle
                    raise

            return wrapper
        return decorator

    def record_rate_limits(self, endpoint: str, remaining: int, reset_timestamp: int):
        """
        Record rate limit information from API response headers.

        Extract from Alpaca response headers:
        - X-RateLimit-Remaining: Requests remaining
        - X-RateLimit-Reset: Unix timestamp of reset

        Args:
            endpoint: API endpoint path
            remaining: Requests remaining before limit
            reset_timestamp: Unix timestamp when limit resets

        Example:
            # From Alpaca API response headers
            headers = response.headers
            alpaca_metrics.record_rate_limits(
                endpoint='/v2/orders',
                remaining=int(headers.get('X-RateLimit-Remaining', 0)),
                reset_timestamp=int(headers.get('X-RateLimit-Reset', 0))
            )
        """
        # Update remaining requests gauge
        self.rate_limit_remaining.labels(endpoint=endpoint).set(remaining)

        # Calculate seconds until reset
        current_time = int(time.time())
        seconds_until_reset = max(0, reset_timestamp - current_time)

        self.rate_limit_reset_seconds.labels(endpoint=endpoint).set(seconds_until_reset)

        # Log warning if approaching rate limit
        if remaining < 20:  # Configurable threshold
            logger.warning(
                "Alpaca API rate limit approaching",
                endpoint=endpoint,
                remaining=remaining,
                reset_in_seconds=seconds_until_reset
            )

    def record_retry(self, endpoint: str, reason: str):
        """
        Record API retry attempt with reason.

        Args:
            endpoint: API endpoint being retried
            reason: Retry reason (e.g., 'rate_limit', 'timeout', 'server_error')

        Example:
            try:
                response = await alpaca_client.get('/v2/positions')
            except RateLimitError:
                alpaca_metrics.record_retry('/v2/positions', 'rate_limit')
                await asyncio.sleep(1)
                response = await alpaca_client.get('/v2/positions')
        """
        self.api_retries_total.labels(
            endpoint=endpoint,
            retry_reason=reason
        ).inc()

    def _extract_rate_limits(self, headers: Dict[str, str], endpoint: str):
        """Extract and record rate limit headers (internal method)"""
        try:
            remaining = headers.get('X-RateLimit-Remaining') or headers.get('x-ratelimit-remaining')
            reset = headers.get('X-RateLimit-Reset') or headers.get('x-ratelimit-reset')

            if remaining and reset:
                self.record_rate_limits(
                    endpoint=endpoint,
                    remaining=int(remaining),
                    reset_timestamp=int(reset)
                )
        except Exception as e:
            logger.debug(f"Failed to extract rate limits: {e}")

    def _classify_error(self, error: Exception) -> str:
        """Classify error type for metrics (internal method)"""
        error_str = str(error).lower()

        if 'rate limit' in error_str or '429' in error_str:
            return 'rate_limit'
        elif 'timeout' in error_str:
            return 'timeout'
        elif 'connection' in error_str:
            return 'connection'
        elif '401' in error_str or 'unauthorized' in error_str:
            return 'auth'
        elif '500' in error_str or '502' in error_str or '503' in error_str:
            return 'server_error'
        elif '400' in error_str or 'bad request' in error_str:
            return 'bad_request'
        else:
            return 'unknown'
