"""
MCP Orchestra Coordinator
Central coordination system for all MCP server connections
Follows singleton pattern from TradingManager for consistency
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from app.core.config import settings
from app.core.exceptions import MCPConnectionError, MCPError, MCPTimeoutError
from app.core.logging import get_logger, log_execution_time

logger = get_logger(__name__)


class MCPServerType(Enum):
    """Supported MCP server types"""

    TASKMASTER_AI = "taskmaster_ai"
    SERENA = "serena"
    MEMORY = "memory"
    TAVILY = "tavily"
    SEQUENTIAL_THINKING = "sequential_thinking"
    GITHUB = "github"
    SHRIMP_TASK_MANAGER = "shrimp_task_manager"
    OPENAI_GPT = "openai_gpt"
    ANTHROPIC_CLAUDE = "anthropic_claude"


@dataclass
class MCPServerConfig:
    """Configuration for MCP server connections"""

    server_type: MCPServerType
    name: str
    command: str
    args: List[str]
    env: Dict[str, str]
    enabled: bool = True
    timeout: int = 30
    max_retries: int = 3
    health_check_interval: int = 60


@dataclass
class MCPServerStatus:
    """Status information for MCP server"""

    server_type: MCPServerType
    connected: bool
    last_health_check: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    connection_time: Optional[datetime] = None
    response_time_avg: float = 0.0


class MCPOrchestrator:
    """
    Singleton MCP Orchestra Coordinator
    Manages all MCP server connections with pooling and health monitoring
    """

    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MCPOrchestrator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Server configurations
        self._server_configs: Dict[MCPServerType, MCPServerConfig] = {}
        self._server_status: Dict[MCPServerType, MCPServerStatus] = {}
        self._server_connections: Dict[MCPServerType, Any] = {}

        # Connection pooling
        self._connection_pool: Dict[MCPServerType, List[Any]] = {}
        self._pool_locks: Dict[MCPServerType, asyncio.Lock] = {}
        self._max_connections_per_server = 5

        # Health monitoring
        self._health_check_tasks: Dict[MCPServerType, asyncio.Task] = {}
        self._monitoring_enabled = True

        # Performance metrics
        self._request_counts: Dict[MCPServerType, int] = {}
        self._response_times: Dict[MCPServerType, List[float]] = {}

        self._initialized = True
        logger.info("MCPOrchestrator singleton initialized")

    async def initialize(self, config_overrides: Optional[Dict] = None):
        """Initialize MCP server connections"""
        async with self._lock:
            try:
                # Setup default configurations
                await self._setup_default_configs()

                # Apply any configuration overrides
                if config_overrides:
                    await self._apply_config_overrides(config_overrides)

                # Initialize server connections
                await self._initialize_server_connections()

                # Start health monitoring
                if self._monitoring_enabled:
                    await self._start_health_monitoring()

                logger.info(
                    "MCPOrchestrator fully initialized",
                    enabled_servers=[
                        s.name for s in self._server_configs.values() if s.enabled
                    ],
                    total_servers=len(self._server_configs),
                )

            except Exception as e:
                logger.error("Failed to initialize MCPOrchestrator", error=str(e))
                raise MCPError(f"MCPOrchestrator initialization failed: {str(e)}")

    async def _setup_default_configs(self):
        """Setup default MCP server configurations"""
        # TaskMaster-AI
        self._server_configs[MCPServerType.TASKMASTER_AI] = MCPServerConfig(
            server_type=MCPServerType.TASKMASTER_AI,
            name="TaskMaster-AI",
            command="npx",
            args=["-y", "--package=task-master-ai", "task-master-ai"],
            env={
                "ANTHROPIC_API_KEY": getattr(settings, "ANTHROPIC_API_KEY", ""),
                "PERPLEXITY_API_KEY": getattr(settings, "PERPLEXITY_API_KEY", ""),
            },
            timeout=60,
            max_retries=3,
        )

        # Serena
        self._server_configs[MCPServerType.SERENA] = MCPServerConfig(
            server_type=MCPServerType.SERENA,
            name="Serena",
            command="mcp-serena",
            args=[],
            env={},
            timeout=30,
            max_retries=2,
        )

        # MCP Memory
        self._server_configs[MCPServerType.MEMORY] = MCPServerConfig(
            server_type=MCPServerType.MEMORY,
            name="Memory",
            command="mcp-memory",
            args=[],
            env={},
            timeout=45,
            max_retries=3,
        )

        # Tavily
        self._server_configs[MCPServerType.TAVILY] = MCPServerConfig(
            server_type=MCPServerType.TAVILY,
            name="Tavily",
            command="mcp-tavily",
            args=[],
            env={
                "TAVILY_API_KEY": getattr(settings, "TAVILY_API_KEY", ""),
            },
            timeout=60,
            max_retries=2,
        )

        # Sequential Thinking
        self._server_configs[MCPServerType.SEQUENTIAL_THINKING] = MCPServerConfig(
            server_type=MCPServerType.SEQUENTIAL_THINKING,
            name="Sequential Thinking",
            command="mcp-sequential-thinking",
            args=[],
            env={},
            timeout=90,
            max_retries=2,
        )

        # GitHub
        self._server_configs[MCPServerType.GITHUB] = MCPServerConfig(
            server_type=MCPServerType.GITHUB,
            name="GitHub",
            command="mcp-github",
            args=[],
            env={
                "GITHUB_TOKEN": getattr(settings, "GITHUB_TOKEN", ""),
            },
            timeout=30,
            max_retries=3,
        )

        # Shrimp Task Manager
        self._server_configs[MCPServerType.SHRIMP_TASK_MANAGER] = MCPServerConfig(
            server_type=MCPServerType.SHRIMP_TASK_MANAGER,
            name="Shrimp Task Manager",
            command="mcp-shrimp-task-manager",
            args=[],
            env={},
            timeout=45,
            max_retries=2,
        )

        # OpenAI GPT
        self._server_configs[MCPServerType.OPENAI_GPT] = MCPServerConfig(
            server_type=MCPServerType.OPENAI_GPT,
            name="OpenAI GPT",
            command="mcp-openai",
            args=["gpt-4"],
            env={
                "OPENAI_API_KEY": getattr(settings, "OPENAI_API_KEY", ""),
            },
            timeout=60,
            max_retries=3,
            enabled=bool(getattr(settings, "OPENAI_API_KEY", "")),
        )

        # Anthropic Claude
        self._server_configs[MCPServerType.ANTHROPIC_CLAUDE] = MCPServerConfig(
            server_type=MCPServerType.ANTHROPIC_CLAUDE,
            name="Anthropic Claude",
            command="mcp-anthropic",
            args=["claude-3-5-sonnet-20241022"],
            env={
                "ANTHROPIC_API_KEY": getattr(settings, "ANTHROPIC_API_KEY", ""),
            },
            timeout=60,
            max_retries=3,
            enabled=bool(getattr(settings, "ANTHROPIC_API_KEY", "")),
        )

        # Initialize status for all servers
        for server_type in self._server_configs:
            self._server_status[server_type] = MCPServerStatus(
                server_type=server_type, connected=False
            )
            self._pool_locks[server_type] = asyncio.Lock()
            self._connection_pool[server_type] = []
            self._request_counts[server_type] = 0
            self._response_times[server_type] = []

    async def _apply_config_overrides(self, overrides: Dict):
        """Apply configuration overrides"""
        for server_name, config in overrides.items():
            server_type = None
            for st in MCPServerType:
                if (
                    st.value == server_name
                    or self._server_configs.get(st, {}).name == server_name
                ):
                    server_type = st
                    break

            if server_type and server_type in self._server_configs:
                current_config = self._server_configs[server_type]

                # Update configuration fields
                for key, value in config.items():
                    if hasattr(current_config, key):
                        setattr(current_config, key, value)

                logger.info(
                    "Applied config override",
                    server=server_name,
                    overrides=list(config.keys()),
                )

    async def _initialize_server_connections(self):
        """Initialize connections to all enabled MCP servers"""
        connection_tasks = []

        for server_type, config in self._server_configs.items():
            if config.enabled:
                connection_tasks.append(self._connect_to_server(server_type))

        # Connect to all servers concurrently
        if connection_tasks:
            results = await asyncio.gather(*connection_tasks, return_exceptions=True)

            # Log connection results
            for i, result in enumerate(results):
                server_type = list(self._server_configs.keys())[i]
                if isinstance(result, Exception):
                    logger.error(
                        "Failed to connect to MCP server",
                        server=server_type.value,
                        error=str(result),
                    )
                else:
                    logger.info(
                        "Successfully connected to MCP server", server=server_type.value
                    )

    async def _connect_to_server(self, server_type: MCPServerType) -> bool:
        """Connect to a specific MCP server"""
        config = self._server_configs[server_type]
        status = self._server_status[server_type]

        try:
            # Mock connection implementation
            # In real implementation, this would use actual MCP client libraries
            await asyncio.sleep(0.1)  # Simulate connection time

            # Update status
            status.connected = True
            status.connection_time = datetime.now()
            status.error_count = 0
            status.last_error = None

            # Store mock connection object
            self._server_connections[server_type] = {
                "type": server_type.value,
                "config": config,
                "mock": True,
            }

            logger.info(
                "Connected to MCP server", server=config.name, timeout=config.timeout
            )

            return True

        except Exception as e:
            status.connected = False
            status.error_count += 1
            status.last_error = str(e)

            logger.error(
                "Failed to connect to MCP server",
                server=config.name,
                error=str(e),
                attempt=status.error_count,
            )

            raise MCPConnectionError(f"Failed to connect to {config.name}: {str(e)}")

    async def _start_health_monitoring(self):
        """Start health monitoring for all connected servers"""
        for server_type in self._server_configs:
            if self._server_status[server_type].connected:
                self._health_check_tasks[server_type] = asyncio.create_task(
                    self._health_monitor_loop(server_type)
                )

        logger.info(
            "Started health monitoring",
            monitoring_servers=len(self._health_check_tasks),
        )

    async def _health_monitor_loop(self, server_type: MCPServerType):
        """Health monitoring loop for a specific server"""
        config = self._server_configs[server_type]

        while self._monitoring_enabled:
            try:
                await asyncio.sleep(config.health_check_interval)

                # Perform health check
                is_healthy = await self._perform_health_check(server_type)

                # Update status
                status = self._server_status[server_type]
                status.last_health_check = datetime.now()

                if not is_healthy:
                    logger.warning(
                        "Health check failed", server=config.name, will_reconnect=True
                    )

                    # Attempt reconnection
                    await self._reconnect_server(server_type)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Health monitoring error", server=config.name, error=str(e)
                )

    async def _perform_health_check(self, server_type: MCPServerType) -> bool:
        """Perform health check for a specific server"""
        try:
            # Mock health check - in real implementation would ping the server
            start_time = datetime.now()
            await asyncio.sleep(0.01)  # Simulate health check
            response_time = (datetime.now() - start_time).total_seconds()

            # Update response time metrics
            response_times = self._response_times[server_type]
            response_times.append(response_time)

            # Keep only last 100 measurements
            if len(response_times) > 100:
                response_times.pop(0)

            # Update average response time
            status = self._server_status[server_type]
            status.response_time_avg = sum(response_times) / len(response_times)

            return True

        except Exception as e:
            logger.error("Health check failed", server=server_type.value, error=str(e))
            return False

    async def _reconnect_server(self, server_type: MCPServerType):
        """Reconnect to a server after health check failure"""
        config = self._server_configs[server_type]

        for attempt in range(config.max_retries):
            try:
                logger.info(
                    "Attempting to reconnect", server=config.name, attempt=attempt + 1
                )

                await self._connect_to_server(server_type)
                logger.info("Successfully reconnected", server=config.name)
                return

            except Exception as e:
                if attempt == config.max_retries - 1:
                    logger.error(
                        "Failed to reconnect after all attempts",
                        server=config.name,
                        error=str(e),
                    )
                else:
                    await asyncio.sleep(2**attempt)  # Exponential backoff

    # ============================================================================
    # PUBLIC API METHODS
    # ============================================================================

    async def get_server_status(
        self, server_type: Optional[MCPServerType] = None
    ) -> Union[MCPServerStatus, Dict[MCPServerType, MCPServerStatus]]:
        """Get status for specific server or all servers"""
        if server_type:
            return self._server_status.get(server_type)
        return dict(self._server_status)

    async def is_server_available(self, server_type: MCPServerType) -> bool:
        """Check if a server is available"""
        status = self._server_status.get(server_type)
        return status.connected if status else False

    @log_execution_time()
    async def call_taskmaster_ai(self, method: str, **kwargs) -> Any:
        """Call TaskMaster-AI MCP server"""
        return await self._call_server(MCPServerType.TASKMASTER_AI, method, **kwargs)

    @log_execution_time()
    async def call_serena(self, method: str, **kwargs) -> Any:
        """Call Serena MCP server"""
        return await self._call_server(MCPServerType.SERENA, method, **kwargs)

    @log_execution_time()
    async def call_memory(self, method: str, **kwargs) -> Any:
        """Call Memory MCP server"""
        return await self._call_server(MCPServerType.MEMORY, method, **kwargs)

    @log_execution_time()
    async def call_tavily(self, method: str, **kwargs) -> Any:
        """Call Tavily MCP server"""
        return await self._call_server(MCPServerType.TAVILY, method, **kwargs)

    @log_execution_time()
    async def call_sequential_thinking(self, method: str, **kwargs) -> Any:
        """Call Sequential Thinking MCP server"""
        return await self._call_server(
            MCPServerType.SEQUENTIAL_THINKING, method, **kwargs
        )

    @log_execution_time()
    async def call_github(self, method: str, **kwargs) -> Any:
        """Call GitHub MCP server"""
        return await self._call_server(MCPServerType.GITHUB, method, **kwargs)

    @log_execution_time()
    async def call_shrimp_task_manager(self, method: str, **kwargs) -> Any:
        """Call Shrimp Task Manager MCP server"""
        return await self._call_server(
            MCPServerType.SHRIMP_TASK_MANAGER, method, **kwargs
        )

    @log_execution_time()
    async def call_openai_gpt(self, method: str, **kwargs) -> Any:
        """Call OpenAI GPT MCP server"""
        return await self._call_server(MCPServerType.OPENAI_GPT, method, **kwargs)

    @log_execution_time()
    async def call_anthropic_claude(self, method: str, **kwargs) -> Any:
        """Call Anthropic Claude MCP server"""
        return await self._call_server(MCPServerType.ANTHROPIC_CLAUDE, method, **kwargs)

    async def _call_server(
        self, server_type: MCPServerType, method: str, **kwargs
    ) -> Any:
        """Internal method to call any MCP server"""
        if not await self.is_server_available(server_type):
            raise MCPConnectionError(f"Server {server_type.value} is not available")

        config = self._server_configs[server_type]

        try:
            start_time = datetime.now()

            # Mock server call - in real implementation would make actual MCP call
            await asyncio.sleep(0.05)  # Simulate API call

            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            self._response_times[server_type].append(response_time)
            self._request_counts[server_type] += 1

            # Mock response
            result = {
                "server": server_type.value,
                "method": method,
                "kwargs": kwargs,
                "timestamp": datetime.now().isoformat(),
                "response_time": response_time,
                "success": True,
            }

            logger.info(
                "MCP server call completed",
                server=server_type.value,
                method=method,
                response_time=response_time,
            )

            return result

        except asyncio.TimeoutError:
            raise MCPTimeoutError(
                f"Call to {server_type.value} timed out after {config.timeout}s"
            )
        except Exception as e:
            logger.error(
                "MCP server call failed",
                server=server_type.value,
                method=method,
                error=str(e),
            )
            raise MCPError(f"Call to {server_type.value} failed: {str(e)}")

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all servers"""
        metrics = {}

        for server_type in self._server_configs:
            status = self._server_status[server_type]
            metrics[server_type.value] = {
                "connected": status.connected,
                "request_count": self._request_counts[server_type],
                "avg_response_time": status.response_time_avg,
                "error_count": status.error_count,
                "last_health_check": (
                    status.last_health_check.isoformat()
                    if status.last_health_check
                    else None
                ),
            }

        return metrics

    @asynccontextmanager
    async def server_connection(self, server_type: MCPServerType):
        """Context manager for MCP server connections with automatic cleanup"""
        if not await self.is_server_available(server_type):
            raise MCPConnectionError(f"Server {server_type.value} is not available")

        connection = self._server_connections[server_type]

        try:
            yield connection
        finally:
            # Connection cleanup would go here in real implementation
            pass

    async def shutdown(self):
        """Shutdown MCP orchestrator and clean up resources"""
        logger.info("Shutting down MCPOrchestrator")

        # Stop health monitoring
        self._monitoring_enabled = False

        # Cancel health check tasks
        for task in self._health_check_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self._health_check_tasks:
            await asyncio.gather(
                *self._health_check_tasks.values(), return_exceptions=True
            )

        # Close server connections
        for server_type in self._server_connections:
            # Connection cleanup would go here in real implementation
            self._server_status[server_type].connected = False

        logger.info("MCPOrchestrator shutdown complete")


# ============================================================================
# SINGLETON INSTANCE ACCESS
# ============================================================================


async def get_mcp_orchestrator() -> MCPOrchestrator:
    """Get the singleton MCPOrchestrator instance"""
    orchestrator = MCPOrchestrator()
    if not orchestrator._initialized:
        await orchestrator.initialize()
    return orchestrator


# Export main classes and functions
__all__ = [
    "MCPOrchestrator",
    "MCPServerType",
    "MCPServerConfig",
    "MCPServerStatus",
    "get_mcp_orchestrator",
]
