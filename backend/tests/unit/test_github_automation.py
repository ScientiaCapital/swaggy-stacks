"""
Unit tests for GitHub Automation Service
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from app.services.github_automation import (
    GitHubAutomationService,
    PullRequestConfig,
    WorkflowTrigger,
    ReleaseConfig
)
from app.core.exceptions import MCPError, ConfigurationError


class TestGitHubAutomationService:
    """Test suite for GitHub Automation Service"""
    
    @pytest.fixture
    async def github_service(self, mock_mcp_orchestrator):
        """Create GitHub service with mock orchestrator"""
        service = GitHubAutomationService()
        service.orchestrator = mock_mcp_orchestrator
        service.owner = "test_owner"
        service.repo = "test_repo"
        return service
    
    async def test_initialization(self, mock_mcp_orchestrator):
        """Test service initialization"""
        service = GitHubAutomationService()
        
        await service.initialize("test_owner", "test_repo")
        
        assert service.owner == "test_owner"
        assert service.repo == "test_repo"
        assert service.orchestrator is not None
    
    async def test_initialization_without_orchestrator(self):
        """Test initialization failure without orchestrator"""
        with patch('app.services.github_automation.MCPOrchestrator.get_instance', return_value=None):
            service = GitHubAutomationService()
            
            with pytest.raises(ConfigurationError):
                await service.initialize("test_owner", "test_repo")
    
    async def test_initialization_github_unavailable(self, mock_mcp_orchestrator):
        """Test initialization when GitHub MCP server unavailable"""
        mock_mcp_orchestrator.is_server_available.return_value = False
        
        with patch('app.services.github_automation.MCPOrchestrator.get_instance', return_value=mock_mcp_orchestrator):
            service = GitHubAutomationService()
            
            with pytest.raises(MCPError):
                await service.initialize("test_owner", "test_repo")
    
    async def test_create_automated_pr_success(self, github_service, mock_mcp_orchestrator):
        """Test successful automated PR creation"""
        # Mock successful PR creation
        mock_mcp_orchestrator.call_mcp_method.return_value = {
            'number': 123,
            'title': 'Test PR',
            'state': 'open',
            'html_url': 'https://github.com/test_owner/test_repo/pull/123'
        }
        
        config = PullRequestConfig(
            title="Test PR",
            body="Test PR body",
            head_branch="feature/test",
            base_branch="main",
            assignees=["user1"],
            labels=["enhancement"]
        )
        
        result = await github_service.create_automated_pr(config)
        
        assert result['number'] == 123
        assert result['title'] == "Test PR"
        
        # Verify correct parameters were passed
        mock_mcp_orchestrator.call_mcp_method.assert_called_with(
            'github',
            'create_pull_request',
            {
                'owner': 'test_owner',
                'repo': 'test_repo',
                'title': 'Test PR',
                'body': 'Test PR body',
                'head': 'feature/test',
                'base': 'main',
                'draft': False
            }
        )
    
    async def test_create_automated_pr_with_metadata(self, github_service, mock_mcp_orchestrator):
        """Test PR creation with assignees and labels"""
        # Mock successful PR creation
        mock_mcp_orchestrator.call_mcp_method.side_effect = [
            # First call - create PR
            {'number': 123, 'title': 'Test PR'},
            # Second call - update metadata
            {'id': 123, 'assignees': ['user1'], 'labels': ['bug']}
        ]
        
        config = PullRequestConfig(
            title="Bug Fix PR",
            body="Fix critical bug",
            head_branch="fix/bug",
            assignees=["user1"],
            labels=["bug"]
        )
        
        result = await github_service.create_automated_pr(config)
        
        assert result['number'] == 123
        assert mock_mcp_orchestrator.call_mcp_method.call_count == 2
        
        # Check metadata update call
        metadata_call = mock_mcp_orchestrator.call_mcp_method.call_args_list[1]
        assert metadata_call[0][1] == 'update_issue'  # method
        assert metadata_call[1]['issue_number'] == 123
        assert metadata_call[1]['assignees'] == ['user1']
        assert metadata_call[1]['labels'] == ['bug']
    
    async def test_create_automated_pr_failure(self, github_service, mock_mcp_orchestrator):
        """Test PR creation failure"""
        mock_mcp_orchestrator.call_mcp_method.side_effect = Exception("GitHub API error")
        
        config = PullRequestConfig(
            title="Test PR",
            body="Test PR body",
            head_branch="feature/test"
        )
        
        with pytest.raises(MCPError) as exc_info:
            await github_service.create_automated_pr(config)
        
        assert "PR creation failed" in str(exc_info.value)
    
    async def test_manage_deployment_workflow(self, github_service, mock_mcp_orchestrator):
        """Test deployment workflow management"""
        # Mock readiness check and commit listing
        mock_mcp_orchestrator.call_mcp_method.side_effect = [
            # Deployment readiness check (search for blocking issues)
            {'total_count': 0},
            # List commits for branch
            [{'sha': 'abc123', 'commit': {'message': 'Test commit'}}]
        ]
        
        trigger = WorkflowTrigger(
            workflow_name="deploy-production",
            branch="main",
            environment="production"
        )
        
        result = await github_service.manage_deployment_workflow(trigger)
        
        assert result['workflow_name'] == "deploy-production"
        assert result['branch'] == "main"
        assert result['environment'] == "production"
        assert result['commit_sha'] == 'abc123'
        assert result['status'] == 'initiated'
    
    async def test_deployment_readiness_check_blocking_issues(self, github_service, mock_mcp_orchestrator):
        """Test deployment blocked by issues"""
        # Mock blocking issues found
        mock_mcp_orchestrator.call_mcp_method.return_value = {'total_count': 2}
        
        trigger = WorkflowTrigger(
            workflow_name="deploy-production",
            branch="main"
        )
        
        with pytest.raises(MCPError) as exc_info:
            await github_service.manage_deployment_workflow(trigger)
        
        assert "Deployment readiness check failed" in str(exc_info.value)
    
    async def test_coordinate_releases_success(self, github_service, mock_mcp_orchestrator):
        """Test successful release coordination"""
        config = ReleaseConfig(
            tag_name="v1.0.0",
            target_commitish="main",
            name="Release v1.0.0",
            body="Release notes here"
        )
        
        result = await github_service.coordinate_releases(config)
        
        assert result['tag_name'] == "v1.0.0"
        assert result['target_commitish'] == "main"
        assert result['name'] == "Release v1.0.0"
        assert result['coordinated'] is True
    
    async def test_coordinate_releases_with_generated_notes(self, github_service, mock_mcp_orchestrator):
        """Test release coordination with auto-generated notes"""
        # Mock commits for release notes
        mock_mcp_orchestrator.call_mcp_method.return_value = [
            {'commit': {'message': 'feat: add new feature'}},
            {'commit': {'message': 'fix: resolve bug'}},
            {'commit': {'message': 'docs: update readme'}}
        ]
        
        config = ReleaseConfig(
            tag_name="v1.1.0",
            target_commitish="main"
            # No body provided, should auto-generate
        )
        
        result = await github_service.coordinate_releases(config)
        
        assert result['tag_name'] == "v1.1.0"
        assert result['coordinated'] is True
        
        # Should have called list_commits for release notes
        mock_mcp_orchestrator.call_mcp_method.assert_called_with(
            'github',
            'list_commits',
            {
                'owner': 'test_owner',
                'repo': 'test_repo',
                'per_page': 50
            }
        )
    
    async def test_generate_release_notes(self, github_service, mock_mcp_orchestrator):
        """Test release notes generation"""
        # Mock commit data
        mock_commits = [
            {'commit': {'message': 'feat: add authentication system'}},
            {'commit': {'message': 'feat: implement trading dashboard'}},
            {'commit': {'message': 'fix: resolve login bug'}},
            {'commit': {'message': 'fix: handle API timeout'}},
            {'commit': {'message': 'docs: update API documentation'}},
            {'commit': {'message': 'chore: update dependencies'}}
        ]
        
        mock_mcp_orchestrator.call_mcp_method.return_value = mock_commits
        
        notes = await github_service._generate_release_notes("v2.0.0")
        
        # Check structure
        assert "# Release v2.0.0" in notes
        assert "## ✨ New Features" in notes
        assert "## 🐛 Bug Fixes" in notes
        assert "## 📝 Other Changes" in notes
        
        # Check content
        assert "feat: add authentication system" in notes
        assert "fix: resolve login bug" in notes
        assert "Full Changelog" in notes
    
    async def test_create_issue_comment(self, github_service, mock_mcp_orchestrator):
        """Test issue comment creation"""
        mock_mcp_orchestrator.call_mcp_method.return_value = {
            'id': 456,
            'body': 'Test comment',
            'created_at': datetime.utcnow().isoformat()
        }
        
        result = await github_service.create_issue_comment(123, "Test comment")
        
        assert result['id'] == 456
        assert result['body'] == 'Test comment'
        
        mock_mcp_orchestrator.call_mcp_method.assert_called_with(
            'github',
            'add_issue_comment',
            {
                'owner': 'test_owner',
                'repo': 'test_repo',
                'issue_number': 123,
                'body': 'Test comment'
            }
        )
    
    async def test_get_workflow_status(self, github_service, mock_mcp_orchestrator):
        """Test workflow status retrieval"""
        mock_mcp_orchestrator.call_mcp_method.return_value = [
            {'sha': 'latest_commit', 'commit': {'message': 'Latest commit'}}
        ]
        
        result = await github_service.get_workflow_status("ci-pipeline")
        
        assert result['workflow_name'] == "ci-pipeline"
        assert result['recent_commits'] == 1
        assert result['last_commit_sha'] == 'latest_commit'
        assert 'checked_at' in result
    
    async def test_health_check_healthy(self, github_service, mock_mcp_orchestrator):
        """Test health check when service is healthy"""
        mock_mcp_orchestrator.is_server_available.return_value = True
        mock_mcp_orchestrator.call_mcp_method.return_value = [{'sha': 'test'}]
        
        result = await github_service.health_check()
        
        assert result['service'] == 'github_automation'
        assert result['status'] == 'healthy'
        assert result['checks']['orchestrator'] == 'ok'
        assert result['checks']['github_mcp'] == 'ok'
        assert result['checks']['repository_access'] == 'ok'
    
    async def test_health_check_degraded(self, github_service, mock_mcp_orchestrator):
        """Test health check when service is degraded"""
        # GitHub MCP unavailable
        mock_mcp_orchestrator.is_server_available.return_value = False
        
        result = await github_service.health_check()
        
        assert result['service'] == 'github_automation'
        assert result['status'] == 'degraded'
        assert result['checks']['github_mcp'] == 'unavailable'
    
    async def test_health_check_unhealthy(self):
        """Test health check when service is unhealthy"""
        service = GitHubAutomationService()
        # No orchestrator initialized
        
        result = await service.health_check()
        
        assert result['service'] == 'github_automation'
        assert result['status'] == 'unhealthy'
        assert result['checks']['orchestrator'] == 'not_initialized'
    
    async def test_health_check_repository_access_failed(self, github_service, mock_mcp_orchestrator):
        """Test health check when repository access fails"""
        mock_mcp_orchestrator.is_server_available.return_value = True
        mock_mcp_orchestrator.call_mcp_method.side_effect = Exception("Access denied")
        
        result = await github_service.health_check()
        
        assert result['service'] == 'github_automation'
        assert result['status'] == 'degraded'
        assert result['checks']['repository_access'] == 'failed'
    
    async def test_health_check_exception_handling(self, github_service):
        """Test health check handles exceptions gracefully"""
        # Force an exception during health check
        github_service.orchestrator = None
        
        with patch.object(github_service, 'orchestrator', side_effect=Exception("Unexpected error")):
            result = await github_service.health_check()
            
            assert result['service'] == 'github_automation'
            assert result['status'] == 'unhealthy'
            assert 'error' in result


class TestGitHubServiceConfigModels:
    """Test configuration models for GitHub service"""
    
    def test_pull_request_config_creation(self):
        """Test PullRequestConfig creation and validation"""
        config = PullRequestConfig(
            title="Test PR",
            body="Test body",
            head_branch="feature/test",
            base_branch="develop",
            draft=True,
            assignees=["user1", "user2"],
            labels=["bug", "high-priority"]
        )
        
        assert config.title == "Test PR"
        assert config.body == "Test body"
        assert config.head_branch == "feature/test"
        assert config.base_branch == "develop"
        assert config.draft is True
        assert config.assignees == ["user1", "user2"]
        assert config.labels == ["bug", "high-priority"]
    
    def test_pull_request_config_defaults(self):
        """Test PullRequestConfig default values"""
        config = PullRequestConfig(
            title="Test PR",
            body="Test body",
            head_branch="feature/test"
        )
        
        assert config.base_branch == "main"  # Default
        assert config.draft is False  # Default
        assert config.assignees == []  # Default
        assert config.labels == []  # Default
    
    def test_workflow_trigger_config(self):
        """Test WorkflowTrigger configuration"""
        trigger = WorkflowTrigger(
            workflow_name="deploy-production",
            branch="main",
            inputs={"environment": "prod", "version": "1.0.0"},
            environment="production"
        )
        
        assert trigger.workflow_name == "deploy-production"
        assert trigger.branch == "main"
        assert trigger.inputs == {"environment": "prod", "version": "1.0.0"}
        assert trigger.environment == "production"
    
    def test_workflow_trigger_defaults(self):
        """Test WorkflowTrigger default values"""
        trigger = WorkflowTrigger(
            workflow_name="test-workflow",
            branch="main"
        )
        
        assert trigger.inputs == {}  # Default
        assert trigger.environment is None  # Default
    
    def test_release_config_creation(self):
        """Test ReleaseConfig creation"""
        config = ReleaseConfig(
            tag_name="v1.0.0",
            target_commitish="release-branch",
            name="Major Release",
            body="Release notes",
            draft=True,
            prerelease=False
        )
        
        assert config.tag_name == "v1.0.0"
        assert config.target_commitish == "release-branch"
        assert config.name == "Major Release"
        assert config.body == "Release notes"
        assert config.draft is True
        assert config.prerelease is False
    
    def test_release_config_defaults(self):
        """Test ReleaseConfig default values"""
        config = ReleaseConfig(tag_name="v1.0.0")
        
        assert config.target_commitish == "main"  # Default
        assert config.name is None  # Default
        assert config.body is None  # Default
        assert config.draft is False  # Default
        assert config.prerelease is False  # Default