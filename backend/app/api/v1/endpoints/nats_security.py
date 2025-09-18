"""
NATS Security Management API Endpoints
====================================

Provides REST API endpoints for managing NATS agent security:
- Agent credential management
- JWT token generation and revocation
- Security audit logs and metrics
- Access control and permissions

Authentication required for all endpoints.
Admin role required for credential management.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from app.core.deps import get_current_user
from app.messaging.nats_coordinator import get_nats_coordinator, SecurityEvent, SecurityCredentials
import structlog

logger = structlog.get_logger()

router = APIRouter()
security = HTTPBearer()


class CreateAgentCredentialsRequest(BaseModel):
    """Request to create new agent credentials"""
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(..., description="Agent type (market_analyst, risk_advisor, etc.)")
    validity_days: int = Field(30, description="Credential validity in days", ge=1, le=365)


class AgentCredentialsResponse(BaseModel):
    """Response containing agent credentials"""
    agent_id: str
    api_key: str
    secret_key: str
    permissions: List[str]
    expires_at: datetime
    created_at: datetime
    last_used: Optional[datetime]


class JWTTokenRequest(BaseModel):
    """Request to generate JWT token"""
    agent_id: str = Field(..., description="Agent ID for token generation")


class JWTTokenResponse(BaseModel):
    """Response containing JWT token"""
    token: str
    expires_in_hours: int
    agent_id: str


class RevokeTokenRequest(BaseModel):
    """Request to revoke JWT token"""
    token: str = Field(..., description="JWT token to revoke")
    agent_id: str = Field(..., description="Agent ID that owns the token")


class SecurityAuditResponse(BaseModel):
    """Response containing security audit events"""
    events: List[Dict[str, Any]]
    total_events: int
    timestamp: datetime


class SecurityMetricsResponse(BaseModel):
    """Response containing security metrics"""
    total_audit_events: int
    auth_failures: int
    permission_denials: int
    rate_limit_violations: int
    active_agents: int
    revoked_tokens: int
    encryption_enabled: bool
    message_signing_enabled: bool
    rate_limiting_enabled: bool
    security_score: int
    timestamp: datetime


def verify_admin_role(current_user: dict = Depends(get_current_user)):
    """Verify user has admin role for security management"""
    if not current_user or not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for security management"
        )
    return current_user


@router.post("/credentials", response_model=AgentCredentialsResponse)
async def create_agent_credentials(
    request: CreateAgentCredentialsRequest,
    current_user: dict = Depends(verify_admin_role)
) -> AgentCredentialsResponse:
    """
    Create new agent credentials with proper permissions

    Requires admin role. Creates secure credentials for trading agents
    with appropriate permissions based on agent type.
    """
    try:
        nats_coordinator = await get_nats_coordinator()

        if not nats_coordinator.security_enabled:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="NATS security framework is disabled"
            )

        # Check if agent already exists
        existing_creds = nats_coordinator.get_agent_credentials(request.agent_id)
        if existing_creds:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Agent credentials already exist for {request.agent_id}"
            )

        credentials = nats_coordinator.create_agent_credentials(
            agent_id=request.agent_id,
            agent_type=request.agent_type,
            validity_days=request.validity_days
        )

        logger.info("Admin created agent credentials",
                   admin_user=current_user.get("email", "unknown"),
                   agent_id=request.agent_id,
                   agent_type=request.agent_type)

        return AgentCredentialsResponse(
            agent_id=credentials.agent_id,
            api_key=credentials.api_key,
            secret_key=credentials.secret_key,
            permissions=credentials.permissions,
            expires_at=credentials.expires_at,
            created_at=credentials.created_at,
            last_used=credentials.last_used
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create agent credentials", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create agent credentials"
        )


@router.get("/credentials/{agent_id}", response_model=AgentCredentialsResponse)
async def get_agent_credentials(
    agent_id: str,
    current_user: dict = Depends(verify_admin_role)
) -> AgentCredentialsResponse:
    """
    Get agent credentials by ID

    Requires admin role. Returns existing credentials for specified agent.
    """
    try:
        nats_coordinator = await get_nats_coordinator()

        credentials = nats_coordinator.get_agent_credentials(agent_id)
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No credentials found for agent {agent_id}"
            )

        return AgentCredentialsResponse(
            agent_id=credentials.agent_id,
            api_key=credentials.api_key,
            secret_key=credentials.secret_key,
            permissions=credentials.permissions,
            expires_at=credentials.expires_at,
            created_at=credentials.created_at,
            last_used=credentials.last_used
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent credentials", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve agent credentials"
        )


@router.post("/tokens", response_model=JWTTokenResponse)
async def generate_jwt_token(
    request: JWTTokenRequest,
    current_user: dict = Depends(get_current_user)
) -> JWTTokenResponse:
    """
    Generate JWT token for agent authentication

    Creates a time-limited JWT token for agent to authenticate NATS messages.
    """
    try:
        nats_coordinator = await get_nats_coordinator()

        if not nats_coordinator.security_enabled:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="NATS security framework is disabled"
            )

        token = nats_coordinator.create_agent_jwt_token(request.agent_id)
        if not token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot generate token for agent {request.agent_id}"
            )

        logger.info("Generated JWT token for agent",
                   user=current_user.get("email", "unknown"),
                   agent_id=request.agent_id)

        return JWTTokenResponse(
            token=token,
            expires_in_hours=nats_coordinator.security_manager.jwt_expiry_hours,
            agent_id=request.agent_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate JWT token", agent_id=request.agent_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate JWT token"
        )


@router.post("/tokens/revoke")
async def revoke_jwt_token(
    request: RevokeTokenRequest,
    current_user: dict = Depends(verify_admin_role)
) -> dict:
    """
    Revoke JWT token

    Requires admin role. Adds token to revocation list to prevent further use.
    """
    try:
        nats_coordinator = await get_nats_coordinator()

        nats_coordinator.revoke_agent_token(request.token, request.agent_id)

        logger.info("Admin revoked JWT token",
                   admin_user=current_user.get("email", "unknown"),
                   agent_id=request.agent_id)

        return {"message": "Token revoked successfully"}

    except Exception as e:
        logger.error("Failed to revoke JWT token", agent_id=request.agent_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke JWT token"
        )


@router.get("/audit", response_model=SecurityAuditResponse)
async def get_security_audit_log(
    limit: int = 100,
    severity: Optional[str] = None,
    current_user: dict = Depends(verify_admin_role)
) -> SecurityAuditResponse:
    """
    Get security audit log

    Requires admin role. Returns recent security events with optional filtering.
    """
    try:
        nats_coordinator = await get_nats_coordinator()

        events = nats_coordinator.get_security_audit_log(limit=limit, severity=severity)

        # Convert events to dict format
        events_dict = []
        for event in events:
            events_dict.append({
                "event_type": event.event_type,
                "agent_id": event.agent_id,
                "subject": event.subject,
                "timestamp": event.timestamp.isoformat(),
                "details": event.details,
                "ip_address": event.ip_address,
                "severity": event.severity
            })

        return SecurityAuditResponse(
            events=events_dict,
            total_events=len(events_dict),
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error("Failed to get security audit log", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security audit log"
        )


@router.get("/metrics", response_model=SecurityMetricsResponse)
async def get_security_metrics(
    current_user: dict = Depends(get_current_user)
) -> SecurityMetricsResponse:
    """
    Get comprehensive security metrics

    Returns security statistics and health information.
    """
    try:
        nats_coordinator = await get_nats_coordinator()

        if not nats_coordinator.security_enabled:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="NATS security framework is disabled"
            )

        metrics = nats_coordinator.get_security_metrics()

        return SecurityMetricsResponse(
            total_audit_events=metrics["total_audit_events"],
            auth_failures=metrics["auth_failures"],
            permission_denials=metrics["permission_denials"],
            rate_limit_violations=metrics["rate_limit_violations"],
            active_agents=metrics["active_agents"],
            revoked_tokens=metrics["revoked_tokens"],
            encryption_enabled=metrics["encryption_enabled"],
            message_signing_enabled=metrics["message_signing_enabled"],
            rate_limiting_enabled=metrics["rate_limiting_enabled"],
            security_score=metrics["security_score"],
            timestamp=datetime.fromisoformat(metrics["timestamp"])
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get security metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security metrics"
        )


@router.get("/health")
async def security_health_check(
    current_user: dict = Depends(get_current_user)
) -> dict:
    """
    Security framework health check

    Returns current status of security features and any issues.
    """
    try:
        nats_coordinator = await get_nats_coordinator()

        health_status = {
            "security_enabled": nats_coordinator.security_enabled,
            "timestamp": datetime.now().isoformat()
        }

        if nats_coordinator.security_enabled:
            metrics = nats_coordinator.get_security_metrics()

            # Determine overall health based on metrics
            health_score = metrics["security_score"]
            if health_score >= 90:
                status = "healthy"
            elif health_score >= 70:
                status = "warning"
            else:
                status = "unhealthy"

            health_status.update({
                "status": status,
                "security_score": health_score,
                "active_agents": metrics["active_agents"],
                "recent_auth_failures": metrics["auth_failures"],
                "encryption_active": metrics["encryption_enabled"],
                "rate_limiting_active": metrics["rate_limiting_enabled"]
            })
        else:
            health_status["status"] = "disabled"

        return health_status

    except Exception as e:
        logger.error("Security health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Security health check failed"
        )