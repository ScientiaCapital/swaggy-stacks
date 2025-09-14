"""
Core Metrics - Institutional-Grade Trading System Monitoring
Focused, essential metrics for algorithmic trading systems
"""

import time
from typing import Dict, Any, Optional
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry
import structlog

logger = structlog.get_logger(__name__)

# Check if unsupervised components are available
try:
    from ..analysis.tool_feedback_tracker import UNSUPERVISED_AVAILABLE
except ImportError:
    UNSUPERVISED_AVAILABLE = False


class CoreTradingMetrics:
    """Essential metrics for institutional-grade trading system"""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        # === CORE TRADING METRICS ===

        # P&L and Performance
        self.total_pnl = Gauge(
            "trading_total_pnl_usd",
            "Total P&L in USD",
            ["strategy", "symbol"],
            registry=self.registry,
        )

        self.win_rate = Gauge(
            "trading_win_rate",
            "Trading win rate percentage (0-1)",
            ["strategy", "timeframe"],
            registry=self.registry,
        )

        self.sharpe_ratio = Gauge(
            "trading_sharpe_ratio",
            "Strategy Sharpe ratio",
            ["strategy"],
            registry=self.registry,
        )

        # Execution Quality
        self.order_latency = Histogram(
            "trading_order_latency_ms",
            "Order execution latency in milliseconds",
            ["order_type", "symbol"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
            registry=self.registry,
        )

        self.slippage = Histogram(
            "trading_slippage_bps",
            "Execution slippage in basis points",
            ["symbol", "strategy"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0],
            registry=self.registry,
        )

        # Risk Management
        self.portfolio_exposure = Gauge(
            "trading_portfolio_exposure",
            "Current portfolio exposure as percentage of equity",
            ["symbol"],
            registry=self.registry,
        )

        self.daily_var = Gauge(
            "trading_daily_var_usd",
            "Daily Value at Risk in USD",
            ["confidence_level"],
            registry=self.registry,
        )

        # === AI INTELLIGENCE METRICS ===

        # Pattern Recognition (Core AI)
        self.pattern_match_confidence = Gauge(
            "ai_pattern_match_confidence",
            "AI pattern matching confidence (0-1)",
            ["symbol", "pattern_type"],
            registry=self.registry,
        )

        self.regime_detection_accuracy = Gauge(
            "ai_regime_detection_accuracy",
            "Market regime detection accuracy (0-1)",
            ["regime_type"],
            registry=self.registry,
        )

        # Anomaly Detection (Critical for Risk)
        self.anomaly_score = Gauge(
            "ai_anomaly_score",
            "Current market anomaly score (0-1)",
            ["symbol", "anomaly_type"],
            registry=self.registry,
        )

        self.anomaly_alerts = Counter(
            "ai_anomaly_alerts_total",
            "Total anomaly alerts triggered",
            ["symbol", "severity"],
            registry=self.registry,
        )

        # AI Performance
        self.ai_prediction_accuracy = Gauge(
            "ai_prediction_accuracy",
            "AI prediction accuracy over time (0-1)",
            ["model_type", "timeframe"],
            registry=self.registry,
        )

        # === SYSTEM HEALTH METRICS ===

        # Trading System Health
        self.system_uptime = Gauge(
            "system_uptime_seconds",
            "System uptime in seconds",
            registry=self.registry,
        )

        self.data_feed_latency = Histogram(
            "system_data_feed_latency_ms",
            "Market data feed latency in milliseconds",
            ["feed_type", "symbol"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500],
            registry=self.registry,
        )

        # Resource Usage (Critical for Performance)
        self.cpu_usage = Gauge(
            "system_cpu_usage_percent",
            "CPU usage percentage",
            ["component"],
            registry=self.registry,
        )

        self.memory_usage = Gauge(
            "system_memory_usage_mb",
            "Memory usage in MB",
            ["component"],
            registry=self.registry,
        )

        logger.info("Core trading metrics initialized", total_metrics=len(self._get_all_metrics()))

    def _get_all_metrics(self):
        """Get all registered metrics for debugging"""
        return [attr for attr in dir(self) if hasattr(getattr(self, attr), 'labels')]

    # === HIGH-LEVEL UPDATE METHODS ===

    def update_trading_performance(
        self,
        strategy: str,
        symbol: str,
        pnl_usd: float,
        win_rate: float,
        sharpe_ratio: float,
        timeframe: str = "1d"
    ):
        """Update core trading performance metrics"""
        self.total_pnl.labels(strategy=strategy, symbol=symbol).set(pnl_usd)
        self.win_rate.labels(strategy=strategy, timeframe=timeframe).set(win_rate)
        self.sharpe_ratio.labels(strategy=strategy).set(sharpe_ratio)

    def record_order_execution(
        self,
        order_type: str,
        symbol: str,
        latency_ms: float,
        slippage_bps: float,
        strategy: str
    ):
        """Record order execution metrics"""
        self.order_latency.labels(order_type=order_type, symbol=symbol).observe(latency_ms)
        self.slippage.labels(symbol=symbol, strategy=strategy).observe(slippage_bps)

    def update_risk_metrics(
        self,
        symbol: str,
        exposure_pct: float,
        daily_var_usd: float,
        confidence_level: str = "95"
    ):
        """Update risk management metrics"""
        self.portfolio_exposure.labels(symbol=symbol).set(exposure_pct)
        self.daily_var.labels(confidence_level=confidence_level).set(daily_var_usd)

    def update_ai_intelligence(
        self,
        symbol: str,
        pattern_confidence: float,
        regime_accuracy: float,
        anomaly_score: float,
        prediction_accuracy: float,
        pattern_type: str = "price_volume",
        regime_type: str = "trend",
        model_type: str = "ensemble"
    ):
        """Update AI intelligence metrics"""
        self.pattern_match_confidence.labels(
            symbol=symbol, pattern_type=pattern_type
        ).set(pattern_confidence)

        self.regime_detection_accuracy.labels(regime_type=regime_type).set(regime_accuracy)

        self.anomaly_score.labels(symbol=symbol, anomaly_type="market").set(anomaly_score)

        self.ai_prediction_accuracy.labels(
            model_type=model_type, timeframe="1h"
        ).set(prediction_accuracy)

    def trigger_anomaly_alert(self, symbol: str, severity: str = "medium"):
        """Trigger anomaly alert"""
        self.anomaly_alerts.labels(symbol=symbol, severity=severity).inc()

    def record_data_feed_latency(self, feed_type: str, symbol: str, latency_ms: float):
        """Record market data feed latency"""
        self.data_feed_latency.labels(feed_type=feed_type, symbol=symbol).observe(latency_ms)

    def update_system_resources(self, component: str, cpu_percent: float, memory_mb: float):
        """Update system resource usage"""
        self.cpu_usage.labels(component=component).set(cpu_percent)
        self.memory_usage.labels(component=component).set(memory_mb)

    def update_system_uptime(self, uptime_seconds: float):
        """Update system uptime"""
        self.system_uptime.set(uptime_seconds)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics state"""
        return {
            "metrics_count": len(self._get_all_metrics()),
            "unsupervised_available": UNSUPERVISED_AVAILABLE,
            "registry_collectors": len(list(self.registry._collector_to_names.keys())),
            "last_updated": time.time()
        }


class CoreMetricsCollector:
    """Lightweight metrics collector for core trading metrics"""

    def __init__(self):
        self.core_metrics = CoreTradingMetrics()
        self._last_update = 0
        self._update_interval = 10  # Update every 10 seconds for trading

    async def collect_essential_metrics(self) -> Dict[str, Any]:
        """Collect only essential metrics for trading"""
        current_time = time.time()

        if current_time - self._last_update < self._update_interval:
            return {"status": "cached", "timestamp": current_time}

        try:
            # Update system uptime
            self.core_metrics.update_system_uptime(current_time)

            # Basic system resource monitoring
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()

            self.core_metrics.update_system_resources(
                component="trading_system",
                cpu_percent=cpu_percent,
                memory_mb=memory_info.used / 1024 / 1024
            )

            self._last_update = current_time

            return {
                "status": "updated",
                "timestamp": current_time,
                "cpu_percent": cpu_percent,
                "memory_mb": memory_info.used / 1024 / 1024
            }

        except Exception as e:
            logger.warning("Failed to collect core metrics", error=str(e))
            return {"status": "error", "error": str(e), "timestamp": current_time}

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        from prometheus_client import generate_latest
        return generate_latest(self.core_metrics.registry)


# Global instance for easy access
core_metrics_collector = CoreMetricsCollector()