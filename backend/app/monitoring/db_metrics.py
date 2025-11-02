"""
Custom Prometheus exporter for SQLAlchemy database performance monitoring.

Tracks connection pool metrics, query performance, and database health with minimal overhead.
Optimized for high-frequency trading systems with QueuePool (20+30 connections).
"""

import time
from functools import wraps
from typing import Optional
from prometheus_client import Gauge, Histogram, Counter

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import Pool

from app.core.logging import get_logger

logger = get_logger(__name__)


class DatabaseMetrics:
    """Low-overhead Prometheus metrics for database performance monitoring"""

    def __init__(self, registry=None):
        """Initialize database metrics with optimized histogram buckets"""

        # Connection Pool Metrics
        self.connection_pool_size = Gauge(
            "db_connection_pool_size",
            "Total database connection pool size",
            registry=registry
        )

        self.connection_pool_available = Gauge(
            "db_connection_pool_available",
            "Available database connections in pool",
            registry=registry
        )

        self.connection_pool_in_use = Gauge(
            "db_connection_pool_in_use",
            "Database connections currently in use",
            registry=registry
        )

        # Query Performance Metrics
        # Buckets optimized for trading queries: 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 5s
        self.query_duration = Histogram(
            "db_query_duration_seconds",
            "Database query execution time in seconds",
            ["query_type", "table"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0],
            registry=registry
        )

        # Connection Checkout Metrics
        # Measures time to acquire a connection from the pool
        self.connection_checkout_duration = Histogram(
            "db_connection_checkout_duration_seconds",
            "Time to acquire database connection from pool",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            registry=registry
        )

        # Connection Pool Events
        self.connection_events_total = Counter(
            "db_connection_events_total",
            "Total database connection events",
            ["event_type"],
            registry=registry
        )

    def instrument_engine(self, engine: Engine):
        """
        Attach event listeners to SQLAlchemy engine for automatic metrics collection.

        This method sets up low-overhead event listeners that track:
        - Connection pool checkout/checkin times
        - Query execution times
        - Connection lifecycle events

        Args:
            engine: SQLAlchemy Engine instance to instrument

        Example:
            from app.core.database import engine
            from app.monitoring.db_metrics import db_metrics

            # Instrument the engine (call once during app startup)
            db_metrics.instrument_engine(engine)
        """
        # Track connection checkout duration
        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            # Record checkout event
            self.connection_events_total.labels(event_type="checkout").inc()

            # Store checkout timestamp for duration calculation
            connection_record.info['checkout_time'] = time.time()

        # Track connection checkin and calculate checkout duration
        @event.listens_for(engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            # Record checkin event
            self.connection_events_total.labels(event_type="checkin").inc()

            # Calculate and record checkout duration
            if 'checkout_time' in connection_record.info:
                duration = time.time() - connection_record.info['checkout_time']
                self.connection_checkout_duration.observe(duration)
                del connection_record.info['checkout_time']

        # Track new connections
        @event.listens_for(engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            self.connection_events_total.labels(event_type="connect").inc()

        # Track query execution via before/after cursor execute
        @event.listens_for(Engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            # Store query start time
            conn.info.setdefault('query_start_time', []).append(time.time())

        @event.listens_for(Engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            # Calculate query duration
            if 'query_start_time' in conn.info and conn.info['query_start_time']:
                duration = time.time() - conn.info['query_start_time'].pop(-1)

                # Classify query type from SQL statement
                query_type = self._classify_query_type(statement)
                table = self._extract_table_name(statement)

                # Record query duration
                self.query_duration.labels(
                    query_type=query_type,
                    table=table
                ).observe(duration)

        logger.info("Database metrics instrumentation completed")

    def update_pool_metrics(self, pool: Pool):
        """
        Update connection pool metrics from Pool instance.

        Call this periodically (e.g., every 30 seconds) to track pool state.

        Args:
            pool: SQLAlchemy Pool instance (e.g., QueuePool)

        Example:
            from app.core.database import engine
            from app.monitoring.db_metrics import db_metrics

            # In a background task or periodic health check
            async def collect_pool_metrics():
                pool = engine.pool
                db_metrics.update_pool_metrics(pool)
        """
        # Get pool statistics
        pool_size = pool.size()
        checked_out = pool.checkedout()
        overflow = pool.overflow()

        # Total connections = base pool + overflow
        total_connections = pool_size + overflow

        # Available connections = total - checked_out
        available = total_connections - checked_out

        # Update gauges
        self.connection_pool_size.set(total_connections)
        self.connection_pool_available.set(available)
        self.connection_pool_in_use.set(checked_out)

        # Log warning if pool is near exhaustion
        if available < 3:  # Configurable threshold
            logger.warning(
                "Database connection pool near exhaustion",
                total=total_connections,
                available=available,
                in_use=checked_out,
                overflow=overflow
            )

    def track_query(self, query_type: str, table: str = "unknown"):
        """
        Decorator to manually track query execution time.

        Use this when automatic instrumentation isn't sufficient or for custom metrics.

        Args:
            query_type: Query type (e.g., 'SELECT', 'INSERT', 'UPDATE', 'DELETE')
            table: Table name being queried

        Example:
            @db_metrics.track_query('SELECT', 'orders')
            async def get_recent_orders(session):
                result = await session.execute(
                    select(Order).where(Order.created_at > cutoff_time)
                )
                return result.scalars().all()
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.query_duration.labels(
                        query_type=query_type,
                        table=table
                    ).observe(duration)

            return wrapper
        return decorator

    def _classify_query_type(self, statement: str) -> str:
        """Classify SQL query type from statement (internal method)"""
        statement_upper = statement.strip().upper()

        if statement_upper.startswith('SELECT'):
            return 'SELECT'
        elif statement_upper.startswith('INSERT'):
            return 'INSERT'
        elif statement_upper.startswith('UPDATE'):
            return 'UPDATE'
        elif statement_upper.startswith('DELETE'):
            return 'DELETE'
        elif any(statement_upper.startswith(cmd) for cmd in ['CREATE', 'ALTER', 'DROP']):
            return 'DDL'
        else:
            return 'OTHER'

    def _extract_table_name(self, statement: str) -> str:
        """Extract primary table name from SQL statement (internal method)"""
        try:
            statement_upper = statement.strip().upper()

            # Simple table extraction (covers most cases)
            if 'FROM' in statement_upper:
                # Extract table after FROM
                parts = statement_upper.split('FROM', 1)[1].strip().split()
                if parts:
                    # Remove schema prefix if present (e.g., 'public.orders' -> 'orders')
                    table = parts[0].replace('"', '').replace('`', '')
                    if '.' in table:
                        table = table.split('.')[-1]
                    return table.lower()

            elif 'INTO' in statement_upper:
                # Extract table after INSERT INTO
                parts = statement_upper.split('INTO', 1)[1].strip().split()
                if parts:
                    table = parts[0].replace('"', '').replace('`', '')
                    if '.' in table:
                        table = table.split('.')[-1]
                    return table.lower()

            elif 'UPDATE' in statement_upper:
                # Extract table after UPDATE
                parts = statement_upper.split('UPDATE', 1)[1].strip().split()
                if parts:
                    table = parts[0].replace('"', '').replace('`', '')
                    if '.' in table:
                        table = table.split('.')[-1]
                    return table.lower()

        except Exception as e:
            logger.debug(f"Failed to extract table name: {e}")

        return 'unknown'


# Global instance (initialized with registry from metrics.py)
db_metrics: Optional[DatabaseMetrics] = None


def initialize_db_metrics(registry=None) -> DatabaseMetrics:
    """
    Initialize global database metrics instance.

    Call this once during application startup.

    Args:
        registry: Optional Prometheus registry (use default if None)

    Returns:
        DatabaseMetrics instance

    Example:
        from app.monitoring.db_metrics import initialize_db_metrics
        from app.core.database import engine

        # During app startup
        db_metrics = initialize_db_metrics()
        db_metrics.instrument_engine(engine)
    """
    global db_metrics
    db_metrics = DatabaseMetrics(registry=registry)
    return db_metrics
