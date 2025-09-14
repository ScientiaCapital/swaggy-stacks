"""
Streamlined Metrics - Clean replacement for metrics.py
Institutional-grade monitoring without code bloat
"""

from typing import Dict, Any, Optional
from .core_metrics import CoreTradingMetrics, CoreMetricsCollector, core_metrics_collector
from .health_checks import HealthChecker
import structlog

logger = structlog.get_logger(__name__)


class PrometheusMetrics:
    """Streamlined Prometheus metrics interface"""

    def __init__(self):
        self.core_metrics = CoreTradingMetrics()

    def update_health_metrics(self, health_status):
        """Update system health metrics"""
        try:
            # Extract basic health info and update core metrics
            if hasattr(health_status, 'system_health'):
                health_data = health_status.system_health

                # Update system resources
                cpu_usage = health_data.get('cpu_percent', 0)
                memory_mb = health_data.get('memory_usage_mb', 0)

                self.core_metrics.update_system_resources(
                    component="system",
                    cpu_percent=cpu_usage,
                    memory_mb=memory_mb
                )

        except Exception as e:
            logger.warning("Failed to update health metrics", error=str(e))

    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        # Use core metrics for essential API monitoring
        pass

    def update_mcp_server_status(self, server_name: str, server_type: str, available: bool):
        """Update MCP server status"""
        # Core system health monitoring only
        pass

    def update_trading_metrics(
        self,
        strategy: str,
        symbol: str,
        pnl: float,
        win_rate: float,
        sharpe: float
    ):
        """Update trading performance metrics"""
        self.core_metrics.update_trading_performance(
            strategy=strategy,
            symbol=symbol,
            pnl_usd=pnl,
            win_rate=win_rate,
            sharpe_ratio=sharpe
        )

    def record_order_metrics(
        self,
        order_type: str,
        symbol: str,
        latency_ms: float,
        slippage_bps: float,
        strategy: str
    ):
        """Record order execution metrics"""
        self.core_metrics.record_order_execution(
            order_type=order_type,
            symbol=symbol,
            latency_ms=latency_ms,
            slippage_bps=slippage_bps,
            strategy=strategy
        )

    def update_ai_metrics(
        self,
        symbol: str,
        pattern_confidence: float,
        regime_accuracy: float,
        anomaly_score: float,
        prediction_accuracy: float
    ):
        """Update AI intelligence metrics"""
        self.core_metrics.update_ai_intelligence(
            symbol=symbol,
            pattern_confidence=pattern_confidence,
            regime_accuracy=regime_accuracy,
            anomaly_score=anomaly_score,
            prediction_accuracy=prediction_accuracy
        )

    def get_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        return core_metrics_collector.get_prometheus_metrics()


class MetricsCollector:
    """Streamlined metrics collector"""

    def __init__(self):
        self.prometheus_metrics = PrometheusMetrics()
        self.health_checker = HealthChecker()
        self._last_update = 0
        self._update_interval = 10  # 10 seconds for trading systems

    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect essential system metrics"""
        current_time = time.time()

        if current_time - self._last_update < self._update_interval:
            return await self.get_cached_metrics()

        try:
            # Get system health
            health_status = await self.health_checker.check_all_components()

            # Update Prometheus metrics
            self.prometheus_metrics.update_health_metrics(health_status)

            # Collect core metrics
            core_metrics = await core_metrics_collector.collect_essential_metrics()

            metrics = {
                "health_status": health_status,
                "core_metrics": core_metrics,
                "timestamp": current_time,
            }

            self._last_update = current_time
            self._cached_metrics = metrics

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {"error": str(e), "timestamp": current_time}

    async def get_cached_metrics(self) -> Dict[str, Any]:
        """Get cached metrics"""
        if hasattr(self, "_cached_metrics"):
            return self._cached_metrics
        return await self.collect_system_metrics()

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        return self.prometheus_metrics.get_metrics()

    async def update_mcp_metrics(self, server_statuses: Dict[str, bool]):
        """Update MCP server metrics"""
        for server_name, available in server_statuses.items():
            self.prometheus_metrics.update_mcp_server_status(
                server_name=server_name, server_type="mcp", available=available
            )

    def record_request_metrics(
        self, method: str, endpoint: str, status_code: int, duration: float
    ):
        """Record HTTP request metrics"""
        self.prometheus_metrics.record_http_request(method, endpoint, status_code, duration)


# Export the same interfaces as the original metrics.py for compatibility
prometheus_metrics = PrometheusMetrics()
metrics_collector = MetricsCollector()

# Maintain backward compatibility
def get_metrics() -> str:
    """Get Prometheus metrics (backward compatibility)"""
    return prometheus_metrics.get_metrics()