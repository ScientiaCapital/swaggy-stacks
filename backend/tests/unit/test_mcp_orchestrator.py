"""
Unit tests for MCP Orchestrator
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from app.mcp.orchestrator import MCPOrchestrator, MCPServerType, MCPServerStatus
from app.core.exceptions import MCPConnectionError, MCPTimeoutError, ConfigurationError


class TestMCPOrchestrator:
    """Test suite for MCP Orchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing"""
        return MCPOrchestrator()
    
    async def test_singleton_pattern(self):
        """Test that MCPOrchestrator follows singleton pattern"""
        instance1 = MCPOrchestrator()
        instance2 = MCPOrchestrator()
        
        assert instance1 is instance2
        assert MCPOrchestrator.get_instance() is instance1
    
    async def test_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert not orchestrator._initialized
        assert orchestrator._server_configs == {}
        assert orchestrator._server_connections == {}
        assert orchestrator._connection_pool == {}
        assert orchestrator._health_monitor_task is None
    
    async def test_setup_default_configs(self, orchestrator):
        """Test default configuration setup"""
        await orchestrator._setup_default_configs()
        
        # Check that all expected server types are configured
        expected_servers = [
            MCPServerType.TASKMASTER_AI,
            MCPServerType.SERENA,
            MCPServerType.MEMORY,
            MCPServerType.TAVILY,
            MCPServerType.SEQUENTIAL_THINKING,
            MCPServerType.GITHUB,
            MCPServerType.SHRIMP_TASK_MANAGER
        ]
        
        for server_type in expected_servers:
            assert server_type in orchestrator._server_configs
            config = orchestrator._server_configs[server_type]
            assert config.name is not None
            assert config.enabled is True
    
    async def test_server_availability_check(self, orchestrator):
        """Test server availability checking"""
        await orchestrator._setup_default_configs()
        
        # Test with mock server status
        mock_status = MCPServerStatus(
            server_type=MCPServerType.GITHUB,
            connected=True,
            last_health_check=datetime.utcnow(),
            error_count=0
        )
        
        orchestrator._server_status[MCPServerType.GITHUB] = mock_status
        
        is_available = await orchestrator.is_server_available(MCPServerType.GITHUB)
        assert is_available is True
        
        # Test unavailable server
        mock_status.connected = False
        is_available = await orchestrator.is_server_available(MCPServerType.GITHUB)
        assert is_available is False
    
    async def test_get_server_status(self, orchestrator):
        """Test retrieving server status"""
        await orchestrator._setup_default_configs()
        
        # Add mock status
        mock_status = MCPServerStatus(
            server_type=MCPServerType.MEMORY,
            connected=True,
            last_health_check=datetime.utcnow(),
            error_count=2
        )
        orchestrator._server_status[MCPServerType.MEMORY] = mock_status
        
        all_status = await orchestrator.get_server_status()
        assert MCPServerType.MEMORY in all_status
        assert all_status[MCPServerType.MEMORY].connected is True
        assert all_status[MCPServerType.MEMORY].error_count == 2
    
    @patch('app.mcp.orchestrator.MCPOrchestrator.call_mcp_method')
    async def test_call_github_method(self, mock_call, orchestrator):
        """Test GitHub-specific method calls"""
        mock_call.return_value = {'commits': [{'sha': 'abc123'}]}
        
        result = await orchestrator.call_github('list_commits', owner='test', repo='test')
        
        mock_call.assert_called_once_with('github', 'list_commits', owner='test', repo='test')
        assert result == {'commits': [{'sha': 'abc123'}]}
    
    @patch('app.mcp.orchestrator.MCPOrchestrator.call_mcp_method')
    async def test_call_memory_method(self, mock_call, orchestrator):
        """Test Memory-specific method calls"""
        mock_call.return_value = {'entities': ['entity1', 'entity2']}
        
        result = await orchestrator.call_memory('create_entities', entities=[])
        
        mock_call.assert_called_once_with('memory', 'create_entities', entities=[])
        assert result == {'entities': ['entity1', 'entity2']}
    
    @patch('app.mcp.orchestrator.MCPOrchestrator.call_mcp_method')
    async def test_call_tavily_method(self, mock_call, orchestrator):
        """Test Tavily-specific method calls"""
        mock_call.return_value = {'results': [{'title': 'Test News'}]}
        
        result = await orchestrator.call_tavily('search', query='test query')
        
        mock_call.assert_called_once_with('tavily', 'search', query='test query')
        assert result == {'results': [{'title': 'Test News'}]}
    
    async def test_error_handling_server_not_available(self, orchestrator):
        """Test error handling when server is not available"""
        await orchestrator._setup_default_configs()
        
        # Mock server as unavailable
        mock_status = MCPServerStatus(
            server_type=MCPServerType.GITHUB,
            connected=False,
            last_health_check=datetime.utcnow(),
            error_count=5
        )
        orchestrator._server_status[MCPServerType.GITHUB] = mock_status
        
        with pytest.raises(MCPConnectionError):
            await orchestrator.call_mcp_method('github', 'test_method')
    
    async def test_connection_error_increment(self, orchestrator):
        """Test that connection errors increment error count"""
        await orchestrator._setup_default_configs()
        
        # Initialize status
        orchestrator._server_status[MCPServerType.MEMORY] = MCPServerStatus(
            server_type=MCPServerType.MEMORY,
            connected=True,
            last_health_check=datetime.utcnow(),
            error_count=0
        )
        
        initial_error_count = orchestrator._server_status[MCPServerType.MEMORY].error_count
        
        # Simulate connection error
        with patch.object(orchestrator, '_make_actual_mcp_call', side_effect=MCPConnectionError("Test error")):
            with pytest.raises(MCPConnectionError):
                await orchestrator.call_mcp_method('memory', 'test_method')
        
        # Check error count incremented
        final_error_count = orchestrator._server_status[MCPServerType.MEMORY].error_count
        assert final_error_count == initial_error_count + 1
    
    async def test_health_monitoring_lifecycle(self, orchestrator):
        """Test health monitoring task lifecycle"""
        await orchestrator._setup_default_configs()
        
        # Enable monitoring
        orchestrator._monitoring_enabled = True
        
        # Start monitoring
        await orchestrator._start_health_monitoring()
        assert orchestrator._health_monitor_task is not None
        assert not orchestrator._health_monitor_task.done()
        
        # Stop monitoring
        await orchestrator._stop_health_monitoring()
        assert orchestrator._health_monitor_task is None or orchestrator._health_monitor_task.cancelled()
    
    async def test_config_overrides(self, orchestrator):
        """Test configuration overrides"""
        config_overrides = {
            'github': {'enabled': False},
            'memory': {'timeout': 60}
        }
        
        await orchestrator.initialize(config_overrides)
        
        # Check GitHub is disabled
        github_config = orchestrator._server_configs.get(MCPServerType.GITHUB)
        assert github_config is not None
        assert github_config.enabled is False
        
        # Check memory timeout is set
        memory_config = orchestrator._server_configs.get(MCPServerType.MEMORY)
        assert memory_config is not None
        assert memory_config.timeout == 60
    
    async def test_server_type_validation(self, orchestrator):
        """Test server type validation"""
        await orchestrator._setup_default_configs()
        
        # Valid server type
        is_valid = orchestrator._validate_server_type('github')
        assert is_valid is True
        
        # Invalid server type
        is_valid = orchestrator._validate_server_type('invalid_server')
        assert is_valid is False
    
    async def test_batch_server_calls(self, orchestrator):
        """Test batch operations across multiple servers"""
        await orchestrator._setup_default_configs()
        
        # Mock multiple server calls
        with patch.object(orchestrator, 'call_mcp_method') as mock_call:
            mock_call.return_value = {'status': 'success'}
            
            servers_to_call = ['github', 'memory', 'tavily']
            results = []
            
            for server in servers_to_call:
                result = await orchestrator.call_mcp_method(server, 'health_check')
                results.append(result)
            
            assert len(results) == 3
            assert all(result['status'] == 'success' for result in results)
            assert mock_call.call_count == 3
    
    async def test_connection_pooling_behavior(self, orchestrator):
        """Test connection pooling behavior"""
        await orchestrator._setup_default_configs()
        
        # Mock connection pool
        mock_connection = AsyncMock()
        orchestrator._connection_pool[MCPServerType.GITHUB] = [mock_connection]
        
        # Test connection retrieval
        async with orchestrator.server_connection(MCPServerType.GITHUB) as conn:
            assert conn is mock_connection
        
        # Connection should be returned to pool
        assert mock_connection in orchestrator._connection_pool[MCPServerType.GITHUB]
    
    async def test_concurrent_server_calls(self, orchestrator):
        """Test concurrent calls to different servers"""
        await orchestrator._setup_default_configs()
        
        with patch.object(orchestrator, 'call_mcp_method') as mock_call:
            # Simulate different response times
            async def mock_response(server, method, **kwargs):
                await asyncio.sleep(0.1 if server == 'github' else 0.05)
                return {'server': server, 'method': method}
            
            mock_call.side_effect = mock_response
            
            # Make concurrent calls
            tasks = [
                orchestrator.call_github('test_method'),
                orchestrator.call_memory('test_method'),
                orchestrator.call_tavily('test_method')
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            assert {result['server'] for result in results} == {'github', 'memory', 'tavily'}
    
    async def test_cleanup_on_shutdown(self, orchestrator):
        """Test proper cleanup during shutdown"""
        await orchestrator._setup_default_configs()
        orchestrator._monitoring_enabled = True
        await orchestrator._start_health_monitoring()
        
        # Add some mock connections
        mock_conn1 = AsyncMock()
        mock_conn2 = AsyncMock()
        orchestrator._server_connections[MCPServerType.GITHUB] = mock_conn1
        orchestrator._connection_pool[MCPServerType.MEMORY] = [mock_conn2]
        
        # Perform cleanup
        await orchestrator.cleanup()
        
        # Check cleanup occurred
        assert orchestrator._health_monitor_task is None or orchestrator._health_monitor_task.cancelled()
        mock_conn1.close.assert_called_once()
        mock_conn2.close.assert_called_once()
    
    async def test_resilience_to_partial_failures(self, orchestrator):
        """Test system resilience when some servers fail"""
        await orchestrator._setup_default_configs()
        
        # Mock one server failing, others working
        async def mock_selective_failure(server, method, **kwargs):
            if server == 'github':
                raise MCPConnectionError("GitHub unavailable")
            return {'status': 'success', 'server': server}
        
        with patch.object(orchestrator, '_make_actual_mcp_call', side_effect=mock_selective_failure):
            # GitHub should fail
            with pytest.raises(MCPConnectionError):
                await orchestrator.call_github('test_method')
            
            # Others should work
            memory_result = await orchestrator.call_memory('test_method')
            assert memory_result['status'] == 'success'
            
            tavily_result = await orchestrator.call_tavily('test_method')  
            assert tavily_result['status'] == 'success'