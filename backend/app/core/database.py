"""
Database configuration and session management for high-performance trading system
"""

import time
from contextlib import contextmanager

import redis
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine

from app.core.config import settings
from app.monitoring.metrics import PrometheusMetrics

# PostgreSQL Database with optimized connection pooling for trading system
engine = create_engine(
    settings.DATABASE_URL,
    # Use QueuePool for better connection management under high load
    poolclass=QueuePool,
    # Connection pool settings optimized for trading workloads
    pool_size=20,                    # Base number of connections to maintain
    max_overflow=30,                 # Additional connections during peak load
    pool_pre_ping=True,              # Validate connections before use
    pool_recycle=3600,               # Recycle connections every hour
    pool_timeout=30,                 # Timeout when getting connection from pool
    # PostgreSQL-specific optimizations
    connect_args={
        "application_name": "swaggy_stacks_trading",
        "connect_timeout": 10,
    } if "postgresql" in settings.DATABASE_URL else (
        {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
    ),
    echo=settings.DEBUG,
    # Additional engine options for performance
    future=True,                     # Use SQLAlchemy 2.0 style
    echo_pool=False,                 # Don't log pool events unless debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize metrics for database performance tracking
metrics = PrometheusMetrics()

# Track query execution times
@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Track query start time"""
    conn.info.setdefault("query_start_time", []).append(time.time())

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Track query duration and type"""
    total_time = time.time() - conn.info["query_start_time"].pop()
    
    # Determine query type from SQL statement
    query_type = statement.strip().split()[0].upper() if statement else "UNKNOWN"
    
    # Extract table name (simplified - gets first table mentioned)
    table = "unknown"
    if "FROM" in statement.upper():
        parts = statement.upper().split("FROM")[1].split()
        if parts:
            table = parts[0].strip().lower().replace('"', '').replace("'", '')
    elif "INTO" in statement.upper():
        parts = statement.upper().split("INTO")[1].split()
        if parts:
            table = parts[0].strip().lower().replace('"', '').replace("'", '')
    
    # Track the query metrics
    metrics.track_db_query(
        query_type=query_type,
        table=table,
        duration=total_time
    )

# Track connection errors
@event.listens_for(Engine, "connect_error")
def connect_error(dbapi_connection, connection_record, exception):
    """Track database connection errors"""
    error_type = type(exception).__name__
    metrics.track_db_connection_error(error_type=error_type)

Base = declarative_base()

# Redis connection with optimized settings for trading system
redis_client = redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    # Connection pool settings for high throughput
    max_connections=100,
    # Health check settings
    health_check_interval=30,
    # Socket settings for performance
    socket_keepalive=True,
    socket_keepalive_options={},
    # Retry logic
    retry_on_timeout=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)


# PostgreSQL session optimization event listeners
@event.listens_for(Engine, "connect")
def set_postgresql_pragmas(dbapi_connection, connection_record):
    """Set PostgreSQL connection-level optimizations for trading system"""
    if hasattr(dbapi_connection, 'cursor'):
        with dbapi_connection.cursor() as cursor:
            # Session-level optimizations (only parameters that can be changed at session level)
            cursor.execute("SET synchronous_commit = off")  # Faster writes (acceptable for trading analysis)
            cursor.execute("SET work_mem = '256MB'")        # Larger work memory for complex queries
            cursor.execute("SET maintenance_work_mem = '512MB'")
            cursor.execute("SET default_statistics_target = 100")
            # Connection-specific optimizations
            cursor.execute("SET lock_timeout = '10s'")
            cursor.execute("SET deadlock_timeout = '1s'")
            cursor.execute("SET statement_timeout = '30s'")  # Prevent long-running queries
            cursor.execute("SET idle_in_transaction_session_timeout = '60s'")


@event.listens_for(Engine, "first_connect")
def set_postgresql_search_path(dbapi_connection, connection_record):
    """Set search path for new connections"""
    if hasattr(dbapi_connection, 'cursor'):
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET search_path = public")



def update_connection_pool_metrics():
    """Update connection pool metrics from the engine pool"""
    try:
        pool = engine.pool
        
        # Get pool statistics
        pool_size = pool.size()  # Total pool size
        checked_out = pool.checkedout()  # Connections in use
        available = pool_size - checked_out  # Available connections
        
        # Update metrics
        metrics.update_connection_pool_stats(
            pool_size=pool_size,
            available=available,
            in_use=checked_out
        )
    except Exception as e:
        # Log error but don't fail - metrics collection should be non-blocking
        import structlog
        logger = structlog.get_logger()
        logger.error("Failed to update connection pool metrics", error=str(e))

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_redis():
    """Dependency to get Redis client"""
    return redis_client


def get_db_session():
    """Get a database session for non-FastAPI contexts"""
    return SessionLocal()


async def init_db():
    """Initialize database tables"""
    # Import all models to ensure they are registered with Base

    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized successfully")
