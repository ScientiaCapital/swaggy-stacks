"""
Database configuration and session management for high-performance trading system
"""

import redis
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine

from app.core.config import settings

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
        # Optimize for trading system performance
        "statement_timeout": "30s",   # Prevent long-running queries
        "idle_in_transaction_session_timeout": "60s",
        # Connection-level settings for better performance
        "tcp_keepalives_idle": "300",
        "tcp_keepalives_interval": "30",
        "tcp_keepalives_count": "3",
    } if "postgresql" in settings.DATABASE_URL else (
        {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
    ),
    echo=settings.DEBUG,
    # Additional engine options for performance
    future=True,                     # Use SQLAlchemy 2.0 style
    echo_pool=False,                 # Don't log pool events unless debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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
            # Optimize for trading system workloads
            cursor.execute("SET synchronous_commit = off")  # Faster writes (acceptable for trading analysis)
            cursor.execute("SET wal_buffers = '16MB'")      # Larger WAL buffers
            cursor.execute("SET checkpoint_completion_target = 0.9")
            cursor.execute("SET random_page_cost = 1.1")    # SSD-optimized
            cursor.execute("SET effective_cache_size = '1GB'")
            cursor.execute("SET work_mem = '256MB'")        # Larger work memory for complex queries
            cursor.execute("SET maintenance_work_mem = '512MB'")
            cursor.execute("SET max_connections = 200")
            cursor.execute("SET shared_buffers = '512MB'")
            # Optimize for analytical queries
            cursor.execute("SET default_statistics_target = 100")
            # Connection-specific optimizations
            cursor.execute("SET lock_timeout = '10s'")
            cursor.execute("SET deadlock_timeout = '1s'")


@event.listens_for(Engine, "first_connect")
def set_postgresql_search_path(dbapi_connection, connection_record):
    """Set search path for new connections"""
    if hasattr(dbapi_connection, 'cursor'):
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET search_path = public")


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
