"""
Integration layer for custom Prometheus exporters.

This module provides the integration point for Alpaca API and Database metrics
with the existing PrometheusMetrics system.
"""

from typing import TYPE_CHECKING

from app.core.logging import get_logger
from app.monitoring.alpaca_metrics import AlpacaAPIMetrics
from app.monitoring.db_metrics import DatabaseMetrics, initialize_db_metrics

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from prometheus_client import CollectorRegistry

logger = get_logger(__name__)


class MetricsExporter:
    """
    Unified exporter for all custom Prometheus metrics.

    Integrates Alpaca API and Database metrics with the main PrometheusMetrics system.
    """

    def __init__(self, registry: "CollectorRegistry" = None):
        """
        Initialize metrics exporters.

        Args:
            registry: Prometheus registry (shared with PrometheusMetrics)
        """
        # Initialize Alpaca API metrics
        self.alpaca_metrics = AlpacaAPIMetrics(registry=registry)
        logger.info("Alpaca API metrics initialized")

        # Initialize Database metrics
        self.db_metrics = initialize_db_metrics(registry=registry)
        logger.info("Database metrics initialized")

    def instrument_database(self, engine: "Engine"):
        """
        Instrument SQLAlchemy engine for automatic metrics collection.

        Args:
            engine: SQLAlchemy Engine instance

        Example:
            from app.core.database import engine
            from app.monitoring.exporter_integration import metrics_exporter

            # During app startup
            metrics_exporter.instrument_database(engine)
        """
        self.db_metrics.instrument_engine(engine)
        logger.info("Database engine instrumented with metrics")

    async def collect_pool_metrics(self, engine: "Engine"):
        """
        Collect and update connection pool metrics.

        Call this periodically (e.g., every 30 seconds in a background task).

        Args:
            engine: SQLAlchemy Engine instance

        Example:
            from app.core.database import engine
            from app.monitoring.exporter_integration import metrics_exporter

            # In a background task
            async def periodic_metrics_collection():
                while True:
                    await metrics_exporter.collect_pool_metrics(engine)
                    await asyncio.sleep(30)
        """
        try:
            pool = engine.pool
            self.db_metrics.update_pool_metrics(pool)
            logger.debug("Connection pool metrics updated")
        except Exception as e:
            logger.error(f"Failed to collect pool metrics: {e}")


# Global instance (initialized with registry from PrometheusMetrics)
metrics_exporter: MetricsExporter = None


def initialize_exporters(registry: "CollectorRegistry" = None) -> MetricsExporter:
    """
    Initialize global metrics exporter instance.

    Call this once during application startup, after PrometheusMetrics initialization.

    Args:
        registry: Prometheus registry (shared with PrometheusMetrics)

    Returns:
        MetricsExporter instance

    Example:
        from app.monitoring.metrics import PrometheusMetrics
        from app.monitoring.exporter_integration import initialize_exporters

        # During app startup
        prometheus_metrics = PrometheusMetrics()
        metrics_exporter = initialize_exporters(prometheus_metrics.registry)
    """
    global metrics_exporter
    metrics_exporter = MetricsExporter(registry=registry)
    return metrics_exporter
