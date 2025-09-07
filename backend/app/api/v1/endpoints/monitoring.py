"""
Health monitoring and system observability endpoints.
"""

import asyncio
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import PlainTextResponse

from app.core.logging import get_logger
from app.monitoring import HealthChecker, MetricsCollector, AlertManager, SystemHealthStatus
from app.monitoring.health_checks import HealthStatus

logger = get_logger(__name__)

router = APIRouter()

# Initialize monitoring components
health_checker = HealthChecker()
metrics_collector = MetricsCollector()
alert_manager = AlertManager()


@router.get("/health", summary="Overall system health check")
async def get_system_health() -> Dict[str, Any]:
    """Get overall system health status"""
    try:
        health_status = await health_checker.check_all_components()
        
        return {
            "status": health_status.overall_status.value,
            "timestamp": health_status.timestamp.isoformat(),
            "uptime_seconds": health_status.uptime_seconds,
            "components_summary": health_status.summary,
            "issues": health_status.issues,
            "total_components": len(health_status.components)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/health/detailed", summary="Detailed system health check")
async def get_detailed_health() -> Dict[str, Any]:
    """Get detailed system health status including all components"""
    try:
        health_status = await health_checker.check_all_components()
        
        # Process alerts
        await alert_manager.process_health_status(health_status)
        
        return {
            "overall_status": health_status.overall_status.value,
            "timestamp": health_status.timestamp.isoformat(),
            "uptime_seconds": health_status.uptime_seconds,
            "summary": health_status.summary,
            "issues": health_status.issues,
            "components": [
                {
                    "component": comp.component,
                    "type": comp.component_type.value,
                    "status": comp.status.value,
                    "message": comp.message,
                    "response_time_ms": comp.response_time_ms,
                    "timestamp": comp.timestamp.isoformat(),
                    "details": comp.details,
                    "error": comp.error
                }
                for comp in health_status.components
            ],
            "active_alerts": len(alert_manager.get_active_alerts()),
            "alert_stats": alert_manager.get_alert_stats()
        }
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/health/component/{component_name}", summary="Check specific component health")
async def get_component_health(component_name: str) -> Dict[str, Any]:
    """Get health status for a specific component"""
    try:
        result = await health_checker.check_component(component_name)
        
        return {
            "component": result.component,
            "type": result.component_type.value,
            "status": result.status.value,
            "message": result.message,
            "response_time_ms": result.response_time_ms,
            "timestamp": result.timestamp.isoformat(),
            "details": result.details,
            "error": result.error
        }
    except Exception as e:
        logger.error(f"Component health check failed for {component_name}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Component health check failed: {str(e)}"
        )


@router.get("/health/mcp", summary="MCP-specific health check")
async def get_mcp_health() -> Dict[str, Any]:
    """Get comprehensive MCP system health"""
    try:
        # Get all components and filter MCP-related ones
        health_status = await health_checker.check_all_components()
        
        mcp_components = [
            comp for comp in health_status.components 
            if 'mcp' in comp.component.lower() or comp.component_type.value in ['mcp_orchestrator', 'mcp_server']
        ]
        
        # Determine MCP overall status
        mcp_statuses = [comp.status for comp in mcp_components]
        if HealthStatus.CRITICAL in mcp_statuses:
            mcp_overall_status = HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in mcp_statuses:
            mcp_overall_status = HealthStatus.DEGRADED
        else:
            mcp_overall_status = HealthStatus.HEALTHY
        
        return {
            "overall_status": mcp_overall_status.value,
            "timestamp": health_status.timestamp.isoformat(),
            "mcp_servers": [
                {
                    "server": comp.component,
                    "status": comp.status.value,
                    "response_time_ms": comp.response_time_ms,
                    "message": comp.message,
                    "details": comp.details,
                    "error": comp.error
                }
                for comp in mcp_components
            ],
            "servers_healthy": sum(1 for comp in mcp_components if comp.status == HealthStatus.HEALTHY),
            "servers_total": len(mcp_components),
            "issues": [issue for issue in health_status.issues if 'mcp' in issue.lower()]
        }
    except Exception as e:
        logger.error(f"MCP health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"MCP health check failed: {str(e)}")


@router.get("/metrics", response_class=PlainTextResponse, summary="Prometheus metrics endpoint")
async def get_metrics() -> str:
    """Get Prometheus metrics for system monitoring"""
    try:
        # Collect latest system metrics
        await metrics_collector.collect_system_metrics()
        
        # Return Prometheus formatted metrics
        return metrics_collector.get_prometheus_metrics()
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")


@router.get("/metrics/json", summary="Get metrics in JSON format")
async def get_metrics_json() -> Dict[str, Any]:
    """Get system metrics in JSON format"""
    try:
        return await metrics_collector.collect_system_metrics()
    except Exception as e:
        logger.error(f"JSON metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")


@router.get("/alerts", summary="Get active alerts")
async def get_active_alerts() -> Dict[str, Any]:
    """Get all active system alerts"""
    try:
        active_alerts = alert_manager.get_active_alerts()
        
        return {
            "active_alerts": [
                {
                    "rule_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "component": alert.component,
                    "component_type": alert.component_type.value if alert.component_type else None,
                    "details": alert.details
                }
                for alert in active_alerts
            ],
            "total_active": len(active_alerts),
            "stats": alert_manager.get_alert_stats()
        }
    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.get("/alerts/history", summary="Get alert history")
async def get_alert_history(
    hours: int = Query(default=24, ge=1, le=168, description="Hours of history to retrieve")
) -> Dict[str, Any]:
    """Get alert history for specified time period"""
    try:
        alert_history = alert_manager.get_alert_history(hours=hours)
        
        return {
            "alert_history": [
                {
                    "rule_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "component": alert.component,
                    "component_type": alert.component_type.value if alert.component_type else None,
                    "resolved": alert.resolved,
                    "resolved_timestamp": alert.resolved_timestamp.isoformat() if alert.resolved_timestamp else None,
                    "details": alert.details
                }
                for alert in alert_history
            ],
            "total_alerts": len(alert_history),
            "time_period_hours": hours
        }
    except Exception as e:
        logger.error(f"Failed to get alert history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alert history: {str(e)}")


@router.post("/health/refresh", summary="Force refresh of health checks")
async def refresh_health_checks(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Force refresh of all health checks"""
    try:
        # Add background task to refresh metrics
        background_tasks.add_task(metrics_collector.collect_system_metrics)
        
        # Perform immediate health check
        health_status = await health_checker.check_all_components()
        
        return {
            "status": "refresh_initiated",
            "timestamp": health_status.timestamp.isoformat(),
            "message": "Health checks refreshed successfully"
        }
    except Exception as e:
        logger.error(f"Health refresh failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health refresh failed: {str(e)}")


@router.get("/status", summary="System status summary")
async def get_system_status() -> Dict[str, Any]:
    """Get concise system status summary"""
    try:
        health_status = await health_checker.check_all_components()
        
        # Simple status check
        is_healthy = health_status.overall_status == HealthStatus.HEALTHY
        
        return {
            "status": "ok" if is_healthy else "error",
            "healthy": is_healthy,
            "uptime_seconds": health_status.uptime_seconds,
            "timestamp": health_status.timestamp.isoformat(),
            "version": "1.0.0"  # Could be loaded from config
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "status": "error",
            "healthy": False,
            "error": str(e),
            "timestamp": health_status.timestamp.isoformat() if 'health_status' in locals() else None
        }


@router.get("/readiness", summary="Kubernetes readiness probe")
async def readiness_probe() -> Dict[str, Any]:
    """Kubernetes readiness probe endpoint"""
    try:
        # Check critical components only for readiness
        db_result = await health_checker.check_component('database')
        redis_result = await health_checker.check_component('redis')
        
        is_ready = (
            db_result.status != HealthStatus.CRITICAL and 
            redis_result.status != HealthStatus.CRITICAL
        )
        
        if not is_ready:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        return {
            "status": "ready",
            "checks": {
                "database": db_result.status.value,
                "redis": redis_result.status.value
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness probe failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/liveness", summary="Kubernetes liveness probe")
async def liveness_probe() -> Dict[str, str]:
    """Kubernetes liveness probe endpoint"""
    try:
        # Basic liveness check - just ensure the app is running
        return {"status": "alive"}
    except Exception as e:
        logger.error(f"Liveness probe failed: {e}")
        raise HTTPException(status_code=503, detail="Service not alive")