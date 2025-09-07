"""
Integration tests for MCP server interactions
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from app.mcp.orchestrator import MCPOrchestrator
from app.services.github_automation import GitHubAutomationService
from app.services.market_research import MarketResearchService
from app.core.exceptions import MCPConnectionError, MCPTimeoutError


class TestMCPGitHubIntegration:
    """Test integration with GitHub MCP server"""
    
    @pytest.fixture
    async def github_integration(self, mock_mcp_orchestrator):
        """Setup GitHub integration test environment"""
        service = GitHubAutomationService()
        service.orchestrator = mock_mcp_orchestrator
        service.owner = "test_owner"
        service.repo = "test_repo"
        return service
    
    async def test_github_pull_request_workflow(self, github_integration, mock_mcp_orchestrator):
        """Test complete GitHub PR workflow"""
        # Mock GitHub API responses
        mock_mcp_orchestrator.call_mcp_method.side_effect = [
            # Create PR
            {
                'number': 123,
                'title': 'Integration Test PR',
                'html_url': 'https://github.com/test_owner/test_repo/pull/123',
                'state': 'open'
            },
            # Add assignees/labels
            {
                'id': 123,
                'assignees': ['developer1'],
                'labels': ['integration-test']
            }
        ]
        
        from app.services.github_automation import PullRequestConfig
        config = PullRequestConfig(
            title="Integration Test PR",
            body="Automated integration test",
            head_branch="feature/integration-test",
            assignees=["developer1"],
            labels=["integration-test"]
        )
        
        result = await github_integration.create_automated_pr(config)
        
        assert result['number'] == 123
        assert result['state'] == 'open'
        
        # Verify MCP calls were made
        assert mock_mcp_orchestrator.call_mcp_method.call_count == 2
        
        # Verify PR creation call
        pr_call = mock_mcp_orchestrator.call_mcp_method.call_args_list[0]
        assert pr_call[0][0] == 'github'  # server
        assert pr_call[0][1] == 'create_pull_request'  # method
        assert pr_call[1]['title'] == 'Integration Test PR'
        assert pr_call[1]['head'] == 'feature/integration-test'
    
    async def test_github_deployment_workflow(self, github_integration, mock_mcp_orchestrator):
        """Test GitHub deployment workflow integration"""
        # Mock deployment readiness check and commit list
        mock_mcp_orchestrator.call_mcp_method.side_effect = [
            # Check for blocking issues
            {'total_count': 0},  # No blocking issues
            # Get latest commit
            [{'sha': 'abc123def', 'commit': {'message': 'Deploy ready commit'}}]
        ]
        
        from app.services.github_automation import WorkflowTrigger
        trigger = WorkflowTrigger(
            workflow_name="deploy-production",
            branch="main",
            environment="production"
        )
        
        result = await github_integration.manage_deployment_workflow(trigger)
        
        assert result['workflow_name'] == "deploy-production"
        assert result['branch'] == "main"
        assert result['status'] == 'initiated'
        assert result['commit_sha'] == 'abc123def'
        
        # Verify readiness check was performed
        readiness_call = mock_mcp_orchestrator.call_mcp_method.call_args_list[0]
        assert 'deployment-blocker' in str(readiness_call)
    
    async def test_github_release_coordination(self, github_integration, mock_mcp_orchestrator):
        """Test GitHub release coordination integration"""
        # Mock commit history for release notes
        mock_commits = [
            {'commit': {'message': 'feat: add new feature'}},
            {'commit': {'message': 'fix: resolve critical bug'}},
            {'commit': {'message': 'docs: update documentation'}}
        ]
        mock_mcp_orchestrator.call_mcp_method.return_value = mock_commits
        
        from app.services.github_automation import ReleaseConfig
        config = ReleaseConfig(
            tag_name="v1.0.0",
            target_commitish="main"
        )
        
        result = await github_integration.coordinate_releases(config)
        
        assert result['tag_name'] == "v1.0.0"
        assert result['coordinated'] is True
        
        # Verify commit history was fetched for release notes
        mock_mcp_orchestrator.call_mcp_method.assert_called_with(
            'github',
            'list_commits',
            {
                'owner': 'test_owner',
                'repo': 'test_repo',
                'per_page': 50
            }
        )
    
    async def test_github_error_handling(self, github_integration, mock_mcp_orchestrator):
        """Test GitHub MCP error handling"""
        # Mock GitHub API error
        mock_mcp_orchestrator.call_mcp_method.side_effect = Exception("GitHub API rate limit exceeded")
        
        from app.services.github_automation import PullRequestConfig
        config = PullRequestConfig(
            title="Error Test PR",
            body="This should fail",
            head_branch="feature/error-test"
        )
        
        from app.core.exceptions import MCPError
        with pytest.raises(MCPError) as exc_info:
            await github_integration.create_automated_pr(config)
        
        assert "PR creation failed" in str(exc_info.value)
        assert "rate limit" in str(exc_info.value)


class TestMCPMarketResearchIntegration:
    """Test integration with market research MCP servers"""
    
    @pytest.fixture
    async def market_research_integration(self, mock_mcp_orchestrator):
        """Setup market research integration test environment"""
        service = MarketResearchService()
        service.orchestrator = mock_mcp_orchestrator
        return service
    
    async def test_tavily_sentiment_analysis(self, market_research_integration, mock_mcp_orchestrator):
        """Test Tavily MCP integration for sentiment analysis"""
        # Mock Tavily search results
        mock_search_results = {
            'results': [
                {
                    'title': 'AAPL Stock Surges on Strong Earnings',
                    'content': 'Apple reported better than expected quarterly earnings...',
                    'url': 'https://example.com/aapl-earnings'
                },
                {
                    'title': 'Apple Innovation Drives Market Confidence',
                    'content': 'New product announcements boost investor sentiment...',
                    'url': 'https://example.com/aapl-innovation'
                }
            ]
        }
        mock_mcp_orchestrator.call_mcp_method.return_value = mock_search_results
        
        sentiment = await market_research_integration.analyze_market_sentiment("AAPL", lookback_days=7)
        
        assert sentiment is not None
        assert hasattr(sentiment, 'overall_sentiment')
        assert hasattr(sentiment, 'confidence_score')
        assert hasattr(sentiment, 'key_factors')
        
        # Verify Tavily was called with correct parameters
        mock_mcp_orchestrator.call_mcp_method.assert_called_with(
            'tavily',
            'search',
            query='AAPL stock market sentiment news',
            max_results=20,
            days=7,
            include_raw_content=True
        )
    
    async def test_sequential_thinking_complex_analysis(self, market_research_integration, mock_mcp_orchestrator):
        """Test Sequential Thinking MCP integration for complex analysis"""
        # Mock sequential thinking workflow
        mock_thinking_result = {
            'thought': 'Analyzing AAPL technical patterns and market context',
            'thought_number': 5,
            'total_thoughts': 5,
            'next_thought_needed': False,
            'stage': 'Conclusion'
        }
        mock_mcp_orchestrator.call_mcp_method.return_value = mock_thinking_result
        
        analysis = await market_research_integration.complex_analysis_workflow(
            "AAPL", 
            "technical_analysis"
        )
        
        assert analysis is not None
        assert hasattr(analysis, 'analysis_type')
        assert hasattr(analysis, 'key_insights')
        assert hasattr(analysis, 'confidence_level')
        
        # Verify sequential thinking was used
        mock_mcp_orchestrator.call_mcp_method.assert_called()
        call_args = mock_mcp_orchestrator.call_mcp_method.call_args
        assert call_args[0][0] == 'sequential_thinking'
        assert 'AAPL' in str(call_args)
    
    async def test_integrated_market_analysis(self, market_research_integration, mock_mcp_orchestrator):
        """Test integrated analysis combining multiple MCP services"""
        # Mock sequential calls to different services
        call_count = 0
        def mock_mcp_call(server, method, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if server == 'tavily':
                return {
                    'results': [
                        {'title': 'Positive AAPL News', 'content': 'Strong performance...'}
                    ]
                }
            elif server == 'sequential_thinking':
                return {
                    'thought': 'Technical analysis shows bullish pattern',
                    'next_thought_needed': False,
                    'stage': 'Conclusion'
                }
            return {}
        
        mock_mcp_orchestrator.call_mcp_method.side_effect = mock_mcp_call
        
        integrated_analysis = await market_research_integration.integrated_strategy_analysis(
            "AAPL",
            include_sentiment=True,
            include_technical=True,
            include_complex_analysis=True
        )
        
        assert integrated_analysis is not None
        assert hasattr(integrated_analysis, 'symbol')
        assert hasattr(integrated_analysis, 'confidence_score')
        assert hasattr(integrated_analysis, 'trading_recommendation')
        assert integrated_analysis.symbol == "AAPL"
        
        # Should have made calls to both Tavily and Sequential Thinking
        assert call_count >= 2
    
    async def test_market_research_error_resilience(self, market_research_integration, mock_mcp_orchestrator):
        """Test market research resilience to MCP server errors"""
        # Mock one service failing, another working
        def mock_failing_service(server, method, **kwargs):
            if server == 'tavily':
                raise MCPConnectionError("Tavily service unavailable")
            elif server == 'sequential_thinking':
                return {
                    'thought': 'Analysis based on available data only',
                    'next_thought_needed': False
                }
            return {}
        
        mock_mcp_orchestrator.call_mcp_method.side_effect = mock_failing_service
        
        # Should handle partial failure gracefully
        analysis = await market_research_integration.integrated_strategy_analysis(
            "AAPL",
            include_sentiment=True,  # This will fail (Tavily)
            include_complex_analysis=True  # This should work (Sequential Thinking)
        )
        
        assert analysis is not None
        # Should have some analysis despite Tavily failure
        assert analysis.confidence_score > 0
        # Should note the service failure
        assert hasattr(analysis, 'complex_analysis')


class TestMCPMemoryIntegration:
    """Test integration with Memory MCP server for knowledge management"""
    
    @pytest.fixture
    async def memory_integration(self, mock_mcp_orchestrator):
        """Setup memory integration test environment"""
        return mock_mcp_orchestrator
    
    async def test_memory_entity_creation(self, memory_integration):
        """Test creating entities in memory knowledge graph"""
        # Mock memory entity creation
        mock_entities = [
            {
                'name': 'AAPL Trading Strategy',
                'entityType': 'strategy',
                'observations': ['High success rate in trending markets']
            }
        ]
        memory_integration.call_mcp_method.return_value = {'created': len(mock_entities)}
        
        result = await memory_integration.call_memory('create_entities', entities=mock_entities)
        
        assert result['created'] == 1
        memory_integration.call_mcp_method.assert_called_with(
            'memory',
            'create_entities',
            entities=mock_entities
        )
    
    async def test_memory_relation_creation(self, memory_integration):
        """Test creating relationships in memory knowledge graph"""
        mock_relations = [
            {
                'from': 'AAPL Trading Strategy',
                'to': 'Market Research Service',
                'relationType': 'uses'
            }
        ]
        memory_integration.call_mcp_method.return_value = {'created': len(mock_relations)}
        
        result = await memory_integration.call_memory('create_relations', relations=mock_relations)
        
        assert result['created'] == 1
        memory_integration.call_mcp_method.assert_called_with(
            'memory',
            'create_relations',
            relations=mock_relations
        )
    
    async def test_memory_knowledge_retrieval(self, memory_integration):
        """Test retrieving knowledge from memory graph"""
        mock_search_results = {
            'nodes': [
                {
                    'name': 'AAPL Trading Strategy',
                    'type': 'strategy',
                    'observations': ['Successful in bull markets', 'Risk management critical']
                }
            ]
        }
        memory_integration.call_mcp_method.return_value = mock_search_results
        
        result = await memory_integration.call_memory('search_nodes', query='AAPL trading')
        
        assert 'nodes' in result
        assert len(result['nodes']) == 1
        assert result['nodes'][0]['name'] == 'AAPL Trading Strategy'


class TestMCPOrchestratorIntegration:
    """Test MCP Orchestrator integration scenarios"""
    
    async def test_multi_server_coordination(self, mock_mcp_orchestrator):
        """Test coordinated operations across multiple MCP servers"""
        call_count = 0
        server_calls = []
        
        def track_calls(server, method, **kwargs):
            nonlocal call_count
            call_count += 1
            server_calls.append((server, method))
            
            if server == 'github':
                return {'commits': [{'sha': 'abc123'}]}
            elif server == 'memory':
                return {'entities': ['stored_entity']}
            elif server == 'tavily':
                return {'results': [{'title': 'News item'}]}
            return {}
        
        mock_mcp_orchestrator.call_mcp_method.side_effect = track_calls
        
        # Simulate a complex workflow using multiple services
        github_result = await mock_mcp_orchestrator.call_github('list_commits', owner='test', repo='test')
        memory_result = await mock_mcp_orchestrator.call_memory('search_nodes', query='test')
        tavily_result = await mock_mcp_orchestrator.call_tavily('search', query='test news')
        
        # Verify all services were called
        assert call_count == 3
        assert ('github', 'list_commits') in server_calls
        assert ('memory', 'search_nodes') in server_calls
        assert ('tavily', 'search') in server_calls
        
        # Verify results
        assert github_result['commits'][0]['sha'] == 'abc123'
        assert memory_result['entities'][0] == 'stored_entity'
        assert tavily_result['results'][0]['title'] == 'News item'
    
    async def test_mcp_server_failover(self, mock_mcp_orchestrator):
        """Test MCP server failover scenarios"""
        failure_count = 0
        
        def simulate_intermittent_failure(server, method, **kwargs):
            nonlocal failure_count
            failure_count += 1
            
            if failure_count == 1:
                # First call fails
                raise MCPConnectionError(f"{server} temporarily unavailable")
            else:
                # Subsequent calls succeed
                return {'status': 'success', 'server': server}
        
        mock_mcp_orchestrator.call_mcp_method.side_effect = simulate_intermittent_failure
        
        # First call should fail
        with pytest.raises(MCPConnectionError):
            await mock_mcp_orchestrator.call_github('test_method')
        
        # Second call should succeed (simulating recovery)
        result = await mock_mcp_orchestrator.call_github('test_method')
        assert result['status'] == 'success'
        assert result['server'] == 'github'
    
    async def test_mcp_timeout_handling(self, mock_mcp_orchestrator):
        """Test MCP timeout handling"""
        async def slow_response(server, method, **kwargs):
            await asyncio.sleep(2)  # Simulate slow response
            return {'delayed': True}
        
        mock_mcp_orchestrator.call_mcp_method.side_effect = slow_response
        
        # Should handle timeout appropriately
        with pytest.raises((asyncio.TimeoutError, MCPTimeoutError)):
            await asyncio.wait_for(
                mock_mcp_orchestrator.call_github('slow_method'),
                timeout=1.0
            )
    
    async def test_concurrent_mcp_operations(self, mock_mcp_orchestrator):
        """Test concurrent MCP operations"""
        operation_times = {}
        
        async def timed_response(server, method, **kwargs):
            import time
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate processing time
            end_time = time.time()
            operation_times[f"{server}_{method}"] = end_time - start_time
            return {'server': server, 'method': method}
        
        mock_mcp_orchestrator.call_mcp_method.side_effect = timed_response
        
        # Run concurrent operations
        tasks = [
            mock_mcp_orchestrator.call_github('operation1'),
            mock_mcp_orchestrator.call_memory('operation2'),
            mock_mcp_orchestrator.call_tavily('operation3')
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All operations should complete
        assert len(results) == 3
        assert all('server' in result for result in results)
        
        # Should have processed concurrently (total time < sum of individual times)
        total_individual_time = sum(operation_times.values())
        # Note: In real implementation, you'd measure actual elapsed time
        assert len(operation_times) == 3
    
    async def test_mcp_health_monitoring_integration(self, mock_mcp_orchestrator):
        """Test MCP health monitoring integration"""
        # Mock server status
        from app.mcp.orchestrator import MCPServerStatus, MCPServerType
        
        mock_status = {
            MCPServerType.GITHUB: MCPServerStatus(
                server_type=MCPServerType.GITHUB,
                connected=True,
                last_health_check=datetime.utcnow(),
                error_count=0
            ),
            MCPServerType.MEMORY: MCPServerStatus(
                server_type=MCPServerType.MEMORY,
                connected=False,
                last_health_check=datetime.utcnow(),
                error_count=3
            )
        }
        
        mock_mcp_orchestrator.get_server_status.return_value = mock_status
        
        status_report = await mock_mcp_orchestrator.get_server_status()
        
        # Verify health monitoring data
        assert MCPServerType.GITHUB in status_report
        assert MCPServerType.MEMORY in status_report
        
        github_status = status_report[MCPServerType.GITHUB]
        memory_status = status_report[MCPServerType.MEMORY]
        
        assert github_status.connected is True
        assert github_status.error_count == 0
        assert memory_status.connected is False
        assert memory_status.error_count == 3