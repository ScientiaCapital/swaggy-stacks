"""
Database configuration and session management
"""

import redis
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings

# PostgreSQL Database
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=StaticPool,
    connect_args=(
        {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
    ),
    echo=settings.DEBUG,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Redis connection
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


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
