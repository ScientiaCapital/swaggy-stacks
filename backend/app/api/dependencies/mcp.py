"""
FastAPI dependencies for MCP services
Provides dependency injection for MCP orchestrator and server access
"""

from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from contextlib import asynccontextmanager

from app.core.logging import get_logger
from app.mcp.orchestrator import (
    get_mcp_orchestrator,
    MCPOrchestrator, 
    MCPServerType
)
from app.core.exceptions import MCPConnectionError, MCPError

logger = get_logger(__name__)


# ============================================================================
# CORE MCP DEPENDENCIES
# ============================================================================

async def get_orchestrator() -> MCPOrchestrator:
    """
    FastAPI dependency to get the MCP orchestrator singleton
    """
    try:
        return await get_mcp_orchestrator()
    except Exception as e:
        logger.error("Failed to get MCP orchestrator", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MCP orchestrator not available"
        )


async def require_mcp_available(
    orchestrator: MCPOrchestrator = Depends(get_orchestrator)
) -> MCPOrchestrator:
    """
    FastAPI dependency that ensures MCP orchestrator is initialized and available
    """
    if not orchestrator._initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MCP orchestrator not initialized"
        )
    
    return orchestrator


# ============================================================================
# SERVER-SPECIFIC DEPENDENCIES
# ============================================================================

class MCPServerDependency:
    """Base class for MCP server dependencies"""
    
    def __init__(self, server_type: MCPServerType, required: bool = True):
        self.server_type = server_type
        self.required = required
    
    async def __call__(
        self, 
        orchestrator: MCPOrchestrator = Depends(require_mcp_available)
    ) -> Optional[MCPOrchestrator]:
        """Check if server is available and return orchestrator"""
        is_available = await orchestrator.is_server_available(self.server_type)
        
        if self.required and not is_available:
            server_name = orchestrator._server_configs[self.server_type].name
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"MCP server '{server_name}' is not available"
            )
        
        return orchestrator if is_available else None


# Individual server dependencies
require_taskmaster_ai = MCPServerDependency(MCPServerType.TASKMASTER_AI, required=True)
require_serena = MCPServerDependency(MCPServerType.SERENA, required=True) 
require_memory = MCPServerDependency(MCPServerType.MEMORY, required=True)
require_tavily = MCPServerDependency(MCPServerType.TAVILY, required=True)
require_sequential_thinking = MCPServerDependency(MCPServerType.SEQUENTIAL_THINKING, required=True)
require_github = MCPServerDependency(MCPServerType.GITHUB, required=True)
require_shrimp_task_manager = MCPServerDependency(MCPServerType.SHRIMP_TASK_MANAGER, required=True)

# Optional server dependencies (won't raise error if unavailable)
optional_taskmaster_ai = MCPServerDependency(MCPServerType.TASKMASTER_AI, required=False)
optional_serena = MCPServerDependency(MCPServerType.SERENA, required=False)
optional_memory = MCPServerDependency(MCPServerType.MEMORY, required=False)
optional_tavily = MCPServerDependency(MCPServerType.TAVILY, required=False)
optional_sequential_thinking = MCPServerDependency(MCPServerType.SEQUENTIAL_THINKING, required=False)
optional_github = MCPServerDependency(MCPServerType.GITHUB, required=False)
optional_shrimp_task_manager = MCPServerDependency(MCPServerType.SHRIMP_TASK_MANAGER, required=False)


# ============================================================================
# FEATURE-BASED DEPENDENCIES
# ============================================================================

async def require_market_research(
    orchestrator: MCPOrchestrator = Depends(optional_tavily),
    sequential_thinking: MCPOrchestrator = Depends(optional_sequential_thinking)
) -> Dict[str, Optional[MCPOrchestrator]]:
    """
    Dependency for market research features (requires Tavily and/or Sequential Thinking)
    """
    available_services = {}
    
    if orchestrator:
        available_services['tavily'] = orchestrator
    if sequential_thinking:
        available_services['sequential_thinking'] = sequential_thinking
    
    if not available_services:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market research services not available (requires Tavily or Sequential Thinking)"
        )
    
    return available_services


async def require_task_management(
    taskmaster: MCPOrchestrator = Depends(optional_taskmaster_ai),
    shrimp: MCPOrchestrator = Depends(optional_shrimp_task_manager)
) -> Dict[str, Optional[MCPOrchestrator]]:
    """
    Dependency for task management features
    """
    available_services = {}
    
    if taskmaster:
        available_services['taskmaster_ai'] = taskmaster
    if shrimp:
        available_services['shrimp_task_manager'] = shrimp
    
    if not available_services:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task management services not available"
        )
    
    return available_services


async def require_code_intelligence(
    serena: MCPOrchestrator = Depends(optional_serena),
    memory: MCPOrchestrator = Depends(optional_memory)
) -> Dict[str, Optional[MCPOrchestrator]]:
    """
    Dependency for code intelligence features
    """
    available_services = {}
    
    if serena:
        available_services['serena'] = serena
    if memory:
        available_services['memory'] = memory
    
    if not available_services:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Code intelligence services not available"
        )
    
    return available_services


async def require_ci_cd_automation(
    github: MCPOrchestrator = Depends(require_github)
) -> MCPOrchestrator:
    """
    Dependency for CI/CD automation features (requires GitHub)
    """
    return github


# ============================================================================
# CONNECTION POOL DEPENDENCIES
# ============================================================================

@asynccontextmanager
async def mcp_server_connection(
    server_type: MCPServerType,
    orchestrator: MCPOrchestrator
):
    """
    Context manager dependency for MCP server connections
    Provides automatic connection management and cleanup
    """
    try:
        async with orchestrator.server_connection(server_type) as connection:
            yield connection
    except MCPConnectionError as e:
        logger.error("MCP connection error", 
                    server=server_type.value, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Connection to {server_type.value} failed"
        )
    except Exception as e:
        logger.error("Unexpected MCP error",
                    server=server_type.value,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MCP service error"
        )


# ============================================================================
# HELPER FUNCTIONS FOR DEPENDENCIES
# ============================================================================

async def validate_server_availability(
    server_types: list[MCPServerType],
    orchestrator: MCPOrchestrator
) -> Dict[MCPServerType, bool]:
    """
    Validate availability of multiple servers
    """
    availability = {}
    
    for server_type in server_types:
        availability[server_type] = await orchestrator.is_server_available(server_type)
    
    return availability


async def get_server_status_summary(
    orchestrator: MCPOrchestrator
) -> Dict[str, Any]:
    """
    Get summary of all server statuses for dependency resolution
    """
    try:
        all_status = await orchestrator.get_server_status()
        
        summary = {
            "total_servers": len(all_status),
            "connected_servers": sum(1 for s in all_status.values() if s.connected),
            "servers": {}
        }
        
        for server_type, status in all_status.items():
            summary["servers"][server_type.value] = {
                "connected": status.connected,
                "error_count": status.error_count,
                "last_health_check": status.last_health_check.isoformat() if status.last_health_check else None
            }
        
        return summary
        
    except Exception as e:
        logger.error("Failed to get server status summary", error=str(e))
        return {"error": str(e)}


# ============================================================================
# MIDDLEWARE-STYLE DEPENDENCY FOR MCP CONTEXT
# ============================================================================

class MCPContext:
    """Context object for MCP operations in request handlers"""
    
    def __init__(self, orchestrator: MCPOrchestrator):
        self.orchestrator = orchestrator
        self._server_cache = {}
    
    async def get_server_client(self, server_type: MCPServerType):
        """Get a client for a specific MCP server with caching"""
        if server_type not in self._server_cache:
            if await self.orchestrator.is_server_available(server_type):
                self._server_cache[server_type] = server_type
            else:
                raise MCPConnectionError(f"Server {server_type.value} not available")
        
        return self._server_cache[server_type]
    
    async def call_server(self, server_type: MCPServerType, method: str, **kwargs):
        """Convenient method to call any server through orchestrator"""
        server_methods = {
            MCPServerType.TASKMASTER_AI: self.orchestrator.call_taskmaster_ai,
            MCPServerType.SERENA: self.orchestrator.call_serena,
            MCPServerType.MEMORY: self.orchestrator.call_memory,
            MCPServerType.TAVILY: self.orchestrator.call_tavily,
            MCPServerType.SEQUENTIAL_THINKING: self.orchestrator.call_sequential_thinking,
            MCPServerType.GITHUB: self.orchestrator.call_github,
            MCPServerType.SHRIMP_TASK_MANAGER: self.orchestrator.call_shrimp_task_manager
        }
        
        if server_type not in server_methods:
            raise ValueError(f"Unknown server type: {server_type}")
        
        return await server_methods[server_type](method, **kwargs)


async def get_mcp_context(
    orchestrator: MCPOrchestrator = Depends(require_mcp_available)
) -> MCPContext:
    """
    FastAPI dependency that provides MCP context for request handlers
    """
    return MCPContext(orchestrator)

async def get_github_service() -> 'GitHubAutomationService':
    """
    FastAPI dependency that provides configured GitHub automation service
    """
    from app.services.github_automation import GitHubAutomationService
    
    service = GitHubAutomationService()
    
    # Initialize with default repository (could be from config)
    # For now, using placeholder values - would need actual repo config
    await service.initialize(owner="tmkipper", repo="swaggy-stacks")
    
    return service
