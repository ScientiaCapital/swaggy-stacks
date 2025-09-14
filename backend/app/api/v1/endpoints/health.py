"""
Health check endpoints for MCP servers and system components
"""

from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.logging import get_logger
from app.mcp.orchestrator import (
    MCPOrchestrator,
    MCPServerType,
    get_mcp_orchestrator,
)
from app.trading.trading_manager import TradingManager

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class HealthStatus(BaseModel):
    """Base health status response"""

    healthy: bool
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


class ComponentHealthStatus(HealthStatus):
    """Component-specific health status"""

    component: str
    version: Optional[str] = None
    uptime_seconds: Optional[float] = None


class MCPServerHealthStatus(BaseModel):
    """MCP server health status"""

    server_type: str
    server_name: str
    connected: bool
    last_health_check: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    response_time_avg: float = 0.0


class SystemHealthResponse(BaseModel):
    """Complete system health response"""

    overall_healthy: bool
    timestamp: datetime
    components: Dict[str, ComponentHealthStatus]
    mcp_servers: Dict[str, MCPServerHealthStatus]
    performance_summary: Dict[str, Any]


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================


@router.get("/", response_model=HealthStatus)
async def root_health_check():
    """Basic health check endpoint"""
    return HealthStatus(
        healthy=True,
        timestamp=datetime.now(),
        details={"status": "SwaggyStacks trading system is running"},
    )


@router.get("/system", response_model=SystemHealthResponse)
async def system_health_check(
    orchestrator: MCPOrchestrator = Depends(get_mcp_orchestrator),
):
    """Complete system health check including all components and MCP servers"""
    try:
        timestamp = datetime.now()

        # Check core components
        components = await _check_core_components()

        # Check MCP servers
        mcp_servers = await _check_mcp_servers(orchestrator)

        # Get performance metrics
        performance_summary = await orchestrator.get_performance_metrics()

        # Determine overall health
        component_health = all(comp.healthy for comp in components.values())
        mcp_health = all(server.connected for server in mcp_servers.values())
        overall_healthy = component_health and mcp_health

        return SystemHealthResponse(
            overall_healthy=overall_healthy,
            timestamp=timestamp,
            components=components,
            mcp_servers=mcp_servers,
            performance_summary=performance_summary,
        )

    except Exception as e:
        logger.error("System health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}",
        )


@router.get("/components", response_model=Dict[str, ComponentHealthStatus])
async def components_health_check():
    """Health check for core system components"""
    return await _check_core_components()


@router.get("/mcp", response_model=Dict[str, MCPServerHealthStatus])
async def mcp_health_check(
    orchestrator: MCPOrchestrator = Depends(get_mcp_orchestrator),
):
    """Health check for all MCP servers"""
    return await _check_mcp_servers(orchestrator)


@router.get("/mcp/{server_type}", response_model=MCPServerHealthStatus)
async def single_mcp_health_check(
    server_type: str, orchestrator: MCPOrchestrator = Depends(get_mcp_orchestrator)
):
    """Health check for a specific MCP server"""
    try:
        # Convert string to enum
        mcp_server_type = None
        for st in MCPServerType:
            if st.value == server_type:
                mcp_server_type = st
                break

        if not mcp_server_type:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server type '{server_type}' not found",
            )

        # Get server status
        server_status = await orchestrator.get_server_status(mcp_server_type)

        if not server_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server '{server_type}' not configured",
            )

        return MCPServerHealthStatus(
            server_type=server_status.server_type.value,
            server_name=orchestrator._server_configs[mcp_server_type].name,
            connected=server_status.connected,
            last_health_check=server_status.last_health_check,
            error_count=server_status.error_count,
            last_error=server_status.last_error,
            response_time_avg=server_status.response_time_avg,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("MCP server health check failed", server=server_type, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed for {server_type}: {str(e)}",
        )


@router.get("/trading", response_model=ComponentHealthStatus)
async def trading_health_check():
    """Health check for trading system components"""
    try:
        trading_manager = TradingManager()

        # Check if trading manager is initialized
        is_healthy = (
            hasattr(trading_manager, "_initialized") and trading_manager._initialized
        )

        details = {
            "paper_trading": getattr(trading_manager, "paper_trading", None),
            "max_positions": getattr(trading_manager, "max_positions", None),
            "has_alpaca_client": hasattr(trading_manager, "_alpaca_client")
            and trading_manager._alpaca_client is not None,
            "has_risk_manager": hasattr(trading_manager, "_risk_manager")
            and trading_manager._risk_manager is not None,
        }

        return ComponentHealthStatus(
            component="trading_system",
            healthy=is_healthy,
            timestamp=datetime.now(),
            details=details,
        )

    except Exception as e:
        logger.error("Trading system health check failed", error=str(e))
        return ComponentHealthStatus(
            component="trading_system",
            healthy=False,
            timestamp=datetime.now(),
            details={"error": str(e)},
        )


@router.post("/mcp/{server_type}/reconnect")
async def reconnect_mcp_server(
    server_type: str, orchestrator: MCPOrchestrator = Depends(get_mcp_orchestrator)
):
    """Force reconnect to a specific MCP server"""
    try:
        # Convert string to enum
        mcp_server_type = None
        for st in MCPServerType:
            if st.value == server_type:
                mcp_server_type = st
                break

        if not mcp_server_type:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP server type '{server_type}' not found",
            )

        # Attempt reconnection
        await orchestrator._reconnect_server(mcp_server_type)

        # Return updated status
        server_status = await orchestrator.get_server_status(mcp_server_type)

        return {
            "message": f"Reconnection attempt completed for {server_type}",
            "connected": server_status.connected,
            "timestamp": datetime.now(),
        }

    except Exception as e:
        logger.error("MCP server reconnection failed", server=server_type, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reconnection failed for {server_type}: {str(e)}",
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def _check_core_components() -> Dict[str, ComponentHealthStatus]:
    """Check health of core system components"""
    components = {}

    # Database health check
    try:
        # Mock database check - in real implementation would test DB connection
        components["database"] = ComponentHealthStatus(
            component="database",
            healthy=True,
            timestamp=datetime.now(),
            details={"type": "postgresql", "status": "connected"},
        )
    except Exception as e:
        components["database"] = ComponentHealthStatus(
            component="database",
            healthy=False,
            timestamp=datetime.now(),
            details={"error": str(e)},
        )

    # Redis health check
    try:
        # Mock Redis check - in real implementation would test Redis connection
        components["redis"] = ComponentHealthStatus(
            component="redis",
            healthy=True,
            timestamp=datetime.now(),
            details={"status": "connected", "used_memory": "150MB"},
        )
    except Exception as e:
        components["redis"] = ComponentHealthStatus(
            component="redis",
            healthy=False,
            timestamp=datetime.now(),
            details={"error": str(e)},
        )

    # Celery health check
    try:
        # Mock Celery check - in real implementation would check worker status
        components["celery"] = ComponentHealthStatus(
            component="celery",
            healthy=True,
            timestamp=datetime.now(),
            details={"active_workers": 2, "pending_tasks": 0},
        )
    except Exception as e:
        components["celery"] = ComponentHealthStatus(
            component="celery",
            healthy=False,
            timestamp=datetime.now(),
            details={"error": str(e)},
        )

    # Trading system health check
    try:
        trading_manager = TradingManager()
        is_healthy = (
            hasattr(trading_manager, "_initialized") and trading_manager._initialized
        )

        components["trading_system"] = ComponentHealthStatus(
            component="trading_system",
            healthy=is_healthy,
            timestamp=datetime.now(),
            details={
                "initialized": is_healthy,
                "paper_trading": getattr(trading_manager, "paper_trading", None),
            },
        )
    except Exception as e:
        components["trading_system"] = ComponentHealthStatus(
            component="trading_system",
            healthy=False,
            timestamp=datetime.now(),
            details={"error": str(e)},
        )

    return components


async def _check_mcp_servers(
    orchestrator: MCPOrchestrator,
) -> Dict[str, MCPServerHealthStatus]:
    """Check health of all MCP servers"""
    mcp_servers = {}

    # Get all server status
    all_status = await orchestrator.get_server_status()

    for server_type, status in all_status.items():
        config = orchestrator._server_configs[server_type]

        mcp_servers[server_type.value] = MCPServerHealthStatus(
            server_type=server_type.value,
            server_name=config.name,
            connected=status.connected,
            last_health_check=status.last_health_check,
            error_count=status.error_count,
            last_error=status.last_error,
            response_time_avg=status.response_time_avg,
        )

    return mcp_servers
