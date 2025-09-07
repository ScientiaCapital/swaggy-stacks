"""
Shared API Dependencies
Common dependencies used across all API endpoints to eliminate redundancy
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from fastapi import Depends, Header, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.analysis.markov_system import MarkovSystem
from app.core.auth import create_jwt_exception, verify_token
from app.core.config import settings
from app.core.database import get_db
from app.core.exceptions import MarketDataError, RiskManagementError, TradingError
from app.models.user import User
from app.trading.trading_manager import TradingManager, get_trading_manager

logger = structlog.get_logger()

# Conditional import for ML features
ConsolidatedStrategyAgent = None
if settings.ML_FEATURES_ENABLED:
    try:
        from app.rag.agents.consolidated_strategy_agent import ConsolidatedStrategyAgent
        logger.info("ML features enabled - ConsolidatedStrategyAgent available")
    except ImportError as e:
        logger.warning(f"ML features enabled but ConsolidatedStrategyAgent unavailable: {e}")
else:
    logger.info("ML features disabled - ConsolidatedStrategyAgent not available")

# Security
security = HTTPBearer(auto_error=False)


# ============================================================================
# AUTHENTICATION & AUTHORIZATION
# ============================================================================


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """
    Get current authenticated user with proper JWT validation
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify JWT token
    token_data = verify_token(credentials.credentials)
    if token_data is None:
        raise create_jwt_exception()

    username = token_data.get("username")
    if username is None:
        raise create_jwt_exception()

    # In production, query the user from database
    # For demo purposes, return a demo user if token is valid
    demo_user = User(
        id=1,
        username="demo_user",
        email="demo@example.com",
        alpaca_api_key=settings.ALPACA_API_KEY,
        alpaca_secret_key=settings.ALPACA_SECRET_KEY,
        is_active=True,
    )

    return demo_user


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Get user if authenticated, None otherwise"""
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None


# ============================================================================
# TRADING SYSTEM DEPENDENCIES
# ============================================================================


async def get_initialized_trading_manager(
    current_user: User = Depends(get_current_user),
) -> TradingManager:
    """Get initialized trading manager for the current user"""
    try:
        trading_manager = get_trading_manager()

        # Initialize with user-specific configuration
        user_config = {
            "user_id": current_user.id,
            "api_key": current_user.alpaca_api_key,
            "secret_key": current_user.alpaca_secret_key,
        }

        async with trading_manager.trading_session(user_config):
            return trading_manager

    except Exception as e:
        logger.error("Failed to initialize trading manager", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Trading system unavailable: {str(e)}",
        )


async def get_strategy_agent(
    strategies: Optional[List[str]] = Query(None, description="Strategies to use"),
    consensus_method: str = Query("weighted_average", description="Consensus method"),
) -> Optional[ConsolidatedStrategyAgent]:
    """Get configured strategy agent (returns None if ML features disabled)"""
    # Check if ML features are enabled and agent is available
    if ConsolidatedStrategyAgent is None:
        logger.info("ConsolidatedStrategyAgent not available - ML features disabled or import failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI strategy features are not available. Set ML_FEATURES_ENABLED=true and ensure ML dependencies are installed.",
        )
    
    try:
        # Default strategies if none specified
        default_strategies = ["markov", "wyckoff", "fibonacci"]
        strategies_to_use = strategies or default_strategies

        agent = ConsolidatedStrategyAgent(
            strategies=strategies_to_use, consensus_method=consensus_method
        )

        await agent.initialize()
        return agent

    except Exception as e:
        logger.error("Failed to initialize strategy agent", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Strategy system unavailable: {str(e)}",
        )


async def get_markov_system() -> MarkovSystem:
    """Get Markov analysis system"""
    try:
        return MarkovSystem(
            lookback_period=getattr(settings, "MARKOV_LOOKBACK_PERIOD", 100),
            n_states=getattr(settings, "MARKOV_N_STATES", 5),
        )
    except Exception as e:
        logger.error("Failed to initialize Markov system", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Analysis system unavailable: {str(e)}",
        )


# ============================================================================
# VALIDATION & SANITIZATION
# ============================================================================


def validate_symbol(symbol: str) -> str:
    """Validate and sanitize trading symbol"""
    if not symbol or not isinstance(symbol, str):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Symbol is required and must be a string",
        )

    clean_symbol = symbol.strip().upper()

    if not clean_symbol.replace("-", "").replace(".", "").isalnum():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid symbol format: {symbol}",
        )

    if len(clean_symbol) < 1 or len(clean_symbol) > 10:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Symbol length must be 1-10 characters: {symbol}",
        )

    return clean_symbol


def validate_quantity(quantity: float) -> float:
    """Validate trading quantity"""
    if quantity <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Quantity must be positive: {quantity}",
        )

    if quantity > 1000000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Quantity too large: {quantity}",
        )

    return float(quantity)


def validate_price(price: float) -> float:
    """Validate price value"""
    if price < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Price cannot be negative: {price}",
        )

    if price > 100000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Price too high: {price}",
        )

    return float(price)


# ============================================================================
# PAGINATION & FILTERING
# ============================================================================


class PaginationParams:
    """Common pagination parameters"""

    def __init__(
        self,
        page: int = Query(1, ge=1, description="Page number"),
        per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    ):
        self.page = page
        self.per_page = per_page
        self.offset = (page - 1) * per_page
        self.limit = per_page


class TimeRangeParams:
    """Common time range parameters"""

    def __init__(
        self,
        start_date: Optional[datetime] = Query(None, description="Start date"),
        end_date: Optional[datetime] = Query(None, description="End date"),
    ):
        if end_date and start_date and end_date <= start_date:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="end_date must be after start_date",
            )

        self.start_date = start_date
        self.end_date = end_date


class FilterParams:
    """Common filter parameters"""

    def __init__(
        self,
        symbol: Optional[str] = Query(None, description="Filter by symbol"),
        strategy: Optional[str] = Query(None, description="Filter by strategy"),
        status: Optional[str] = Query(None, description="Filter by status"),
    ):
        self.symbol = symbol.upper() if symbol else None
        self.strategy = strategy
        self.status = status


# ============================================================================
# ERROR HANDLING
# ============================================================================


def handle_trading_error(error: Exception) -> HTTPException:
    """Convert trading errors to appropriate HTTP responses"""
    if isinstance(error, TradingError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "trading_error",
                "message": str(error),
                "type": "TradingError",
            },
        )
    elif isinstance(error, RiskManagementError):
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "risk_management_error",
                "message": str(error),
                "type": "RiskManagementError",
            },
        )
    elif isinstance(error, MarketDataError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "market_data_error",
                "message": str(error),
                "type": "MarketDataError",
            },
        )
    else:
        # Generic server error
        logger.error("Unexpected error", error=str(error))
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_error",
                "message": "An unexpected error occurred",
                "type": type(error).__name__,
            },
        )


# ============================================================================
# RATE LIMITING (PLACEHOLDER)
# ============================================================================


class RateLimiter:
    """Simple rate limiter - implement proper rate limiting in production"""

    def __init__(self, calls: int = 100, period: int = 60):
        self.calls = calls
        self.period = period
        self.requests = {}

    async def check_rate_limit(self, request: Request) -> bool:
        # Placeholder implementation
        client_ip = request.client.host
        current_time = datetime.now()

        # Clean old entries
        self.requests = {
            ip: times
            for ip, times in self.requests.items()
            if any(
                current_time.timestamp() - t.timestamp() <= self.period for t in times
            )
        }

        # Check current client
        if client_ip not in self.requests:
            self.requests[client_ip] = []

        recent_requests = [
            t
            for t in self.requests[client_ip]
            if current_time.timestamp() - t.timestamp() <= self.period
        ]

        if len(recent_requests) >= self.calls:
            return False

        self.requests[client_ip] = recent_requests + [current_time]
        return True


rate_limiter = RateLimiter()


async def check_rate_limit(request: Request) -> None:
    """Rate limiting dependency"""
    if not await rate_limiter.check_rate_limit(request):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded"
        )


# ============================================================================
# HEALTH CHECK HELPERS
# ============================================================================


async def check_system_health() -> Dict[str, Any]:
    """Check overall system health"""
    health_status = {
        "status": "healthy",
        "components": {},
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # Check trading manager
        trading_manager = get_trading_manager()
        tm_health = await trading_manager.health_check()
        health_status["components"]["trading_manager"] = tm_health.get(
            "trading_manager", "unknown"
        )
        health_status["components"].update(tm_health.get("components", {}))

        # Check database
        try:
            db = next(get_db())
            db.execute("SELECT 1")
            health_status["components"]["database"] = "healthy"
            db.close()
        except Exception as e:
            health_status["components"]["database"] = f"error: {str(e)}"
            health_status["status"] = "degraded"

        # Check market data (placeholder)
        health_status["components"]["market_data"] = "healthy"

    except Exception as e:
        health_status["status"] = "error"
        health_status["error"] = str(e)

    return health_status


# ============================================================================
# COMMON RESPONSE HELPERS
# ============================================================================


def create_success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    """Create standardized success response"""
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }


def create_error_response(
    error: str, message: str, details: Dict = None
) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "success": False,
        "error": error,
        "message": message,
        "details": details or {},
        "timestamp": datetime.now().isoformat(),
    }


# Export dependencies for easy importing
__all__ = [
    # Authentication
    "get_current_user",
    "get_optional_user",
    # Trading system
    "get_initialized_trading_manager",
    "get_strategy_agent",
    "get_markov_system",
    # Validation
    "validate_symbol",
    "validate_quantity",
    "validate_price",
    # Parameters
    "PaginationParams",
    "TimeRangeParams",
    "FilterParams",
    # Error handling
    "handle_trading_error",
    # Rate limiting
    "check_rate_limit",
    # Health checks
    "check_system_health",
    # Response helpers
    "create_success_response",
    "create_error_response",
]
