"""
Integration tests for GitHub automation endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, Mock
import json
from datetime import datetime

from app.main import app


class TestGitHubWebhookEndpoints:
    """Test GitHub webhook endpoints"""
    
    def test_github_webhook_pull_request_opened(self, client: TestClient):
        """Test GitHub webhook for pull request opened event"""
        webhook_payload = {
            "action": "opened",
            "pull_request": {
                "number": 123,
                "title": "Test PR",
                "base": {"ref": "main"},
                "head": {"ref": "feature/test"}
            },
            "repository": {
                "name": "test-repo",
                "full_name": "owner/test-repo"
            }
        }
        
        with patch('app.api.v1.endpoints.github.process_github_webhook') as mock_process:
            response = client.post(
                "/api/v1/github/webhooks/github",
                json=webhook_payload,
                headers={"X-GitHub-Event": "pull_request"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Webhook processed successfully"
            
            # Verify background task was scheduled
            mock_process.assert_called_once()
    
    def test_github_webhook_push_event(self, client: TestClient):
        """Test GitHub webhook for push event"""
        webhook_payload = {
            "ref": "refs/heads/main",
            "commits": [
                {"id": "abc123", "message": "Test commit"},
                {"id": "def456", "message": "Another commit"}
            ],
            "repository": {
                "name": "test-repo"
            }
        }
        
        response = client.post(
            "/api/v1/github/webhooks/github",
            json=webhook_payload,
            headers={"X-GitHub-Event": "push"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Webhook processed successfully"
    
    def test_github_webhook_invalid_payload(self, client: TestClient):
        """Test GitHub webhook with invalid payload"""
        invalid_payload = "not json"
        
        response = client.post(
            "/api/v1/github/webhooks/github",
            data=invalid_payload,
            headers={"X-GitHub-Event": "push", "Content-Type": "text/plain"}
        )
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_github_webhook_missing_event_header(self, client: TestClient):
        """Test GitHub webhook without X-GitHub-Event header"""
        webhook_payload = {
            "action": "opened",
            "pull_request": {"number": 123}
        }
        
        response = client.post(
            "/api/v1/github/webhooks/github",
            json=webhook_payload
            # Missing X-GitHub-Event header
        )
        
        assert response.status_code == 200  # Should still process
    
    def test_github_webhook_signature_verification_disabled(self, client: TestClient):
        """Test webhook processes without signature verification (dev mode)"""
        webhook_payload = {"test": "payload"}
        
        # Without signature, should still work in development
        response = client.post(
            "/api/v1/github/webhooks/github",
            json=webhook_payload,
            headers={"X-GitHub-Event": "ping"}
        )
        
        assert response.status_code == 200


class TestGitHubAutomationEndpoints:
    """Test GitHub automation API endpoints"""
    
    @patch('app.api.v1.endpoints.github.get_github_service')
    def test_create_pull_request_success(self, mock_get_service, client: TestClient):
        """Test successful PR creation via API"""
        # Mock GitHub service
        mock_service = AsyncMock()
        mock_service.create_automated_pr.return_value = {
            "number": 123,
            "title": "Automated PR",
            "html_url": "https://github.com/owner/repo/pull/123",
            "state": "open"
        }
        mock_get_service.return_value = mock_service
        
        pr_request = {
            "title": "Automated PR",
            "body": "This PR was created automatically",
            "head_branch": "feature/automation",
            "base_branch": "main",
            "draft": False,
            "assignees": ["developer1"],
            "labels": ["automation"]
        }
        
        response = client.post("/api/v1/github/automation/create-pr", json=pr_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["number"] == 123
        assert data["title"] == "Automated PR"
        assert data["state"] == "open"
        
        # Verify service was called correctly
        mock_service.create_automated_pr.assert_called_once()
    
    @patch('app.api.v1.endpoints.github.get_github_service')
    def test_create_pull_request_service_error(self, mock_get_service, client: TestClient):
        """Test PR creation with service error"""
        from app.core.exceptions import MCPError
        
        # Mock service that raises error
        mock_service = AsyncMock()
        mock_service.create_automated_pr.side_effect = MCPError("GitHub API failed")
        mock_get_service.return_value = mock_service
        
        pr_request = {
            "title": "Test PR",
            "body": "Test body",
            "head_branch": "test-branch"
        }
        
        response = client.post("/api/v1/github/automation/create-pr", json=pr_request)
        
        assert response.status_code == 503
        data = response.json()
        assert "GitHub API failed" in data["detail"]
    
    def test_create_pull_request_invalid_data(self, client: TestClient):
        """Test PR creation with invalid request data"""
        invalid_request = {
            "title": "",  # Empty title should be invalid
            "body": "Test body"
            # Missing required head_branch
        }
        
        response = client.post("/api/v1/github/automation/create-pr", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    @patch('app.api.v1.endpoints.github.get_github_service')
    def test_trigger_workflow_success(self, mock_get_service, client: TestClient):
        """Test successful workflow trigger"""
        mock_service = AsyncMock()
        mock_service.manage_deployment_workflow.return_value = {
            "workflow_name": "deploy-production",
            "branch": "main",
            "status": "initiated",
            "triggered_at": datetime.utcnow().isoformat()
        }
        mock_get_service.return_value = mock_service
        
        workflow_request = {
            "workflow_name": "deploy-production",
            "branch": "main",
            "environment": "production",
            "inputs": {
                "version": "1.0.0",
                "notify_slack": "true"
            }
        }
        
        response = client.post("/api/v1/github/automation/trigger-workflow", json=workflow_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["workflow_name"] == "deploy-production"
        assert data["branch"] == "main"
        assert data["status"] == "initiated"
    
    @patch('app.api.v1.endpoints.github.get_github_service')
    def test_create_release_success(self, mock_get_service, client: TestClient):
        """Test successful release creation"""
        mock_service = AsyncMock()
        mock_service.coordinate_releases.return_value = {
            "tag_name": "v1.0.0",
            "name": "Release v1.0.0",
            "html_url": "https://github.com/owner/repo/releases/tag/v1.0.0",
            "created_at": datetime.utcnow().isoformat(),
            "coordinated": True
        }
        mock_get_service.return_value = mock_service
        
        release_request = {
            "tag_name": "v1.0.0",
            "target_commitish": "main",
            "name": "Release v1.0.0",
            "body": "## Features\n- New authentication system\n- Performance improvements",
            "draft": False,
            "prerelease": False
        }
        
        response = client.post("/api/v1/github/automation/create-release", json=release_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["tag_name"] == "v1.0.0"
        assert data["coordinated"] is True
    
    @patch('app.api.v1.endpoints.github.get_github_service')
    def test_get_workflow_status(self, mock_get_service, client: TestClient):
        """Test workflow status retrieval"""
        mock_service = AsyncMock()
        mock_service.get_workflow_status.return_value = {
            "workflow_name": "ci-pipeline",
            "recent_commits": 5,
            "last_commit_sha": "abc123",
            "checked_at": datetime.utcnow().isoformat()
        }
        mock_get_service.return_value = mock_service
        
        response = client.get("/api/v1/github/automation/workflow-status/ci-pipeline")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["workflow_name"] == "ci-pipeline"
        assert data["recent_commits"] == 5
        assert data["last_commit_sha"] == "abc123"
    
    @patch('app.api.v1.endpoints.github.get_github_service')
    def test_github_automation_health_check(self, mock_get_service, client: TestClient):
        """Test GitHub automation health check endpoint"""
        mock_service = AsyncMock()
        mock_service.health_check.return_value = {
            "service": "github_automation",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "orchestrator": "ok",
                "github_mcp": "ok",
                "repository_access": "ok"
            }
        }
        mock_get_service.return_value = mock_service
        
        response = client.get("/api/v1/github/automation/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["service"] == "github_automation"
        assert data["status"] == "healthy"
        assert "checks" in data
    
    @patch('app.api.v1.endpoints.github.get_github_service')
    def test_github_automation_health_check_unhealthy(self, mock_get_service, client: TestClient):
        """Test GitHub automation health check when service is unhealthy"""
        mock_service = AsyncMock()
        mock_service.health_check.return_value = {
            "service": "github_automation",
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": "MCP orchestrator not available"
        }
        mock_get_service.return_value = mock_service
        
        response = client.get("/api/v1/github/automation/health")
        
        assert response.status_code == 503
        data = response.json()
        
        assert data["service"] == "github_automation"
        assert data["status"] == "unhealthy"
        assert "error" in data
    
    @patch('app.api.v1.endpoints.github.get_github_service')
    def test_github_service_initialization_error(self, mock_get_service, client: TestClient):
        """Test endpoint behavior when GitHub service fails to initialize"""
        from app.core.exceptions import ConfigurationError
        
        mock_get_service.side_effect = ConfigurationError("Service initialization failed")
        
        pr_request = {
            "title": "Test PR",
            "body": "Test body",
            "head_branch": "test-branch"
        }
        
        response = client.post("/api/v1/github/automation/create-pr", json=pr_request)
        
        assert response.status_code == 500
    
    def test_github_endpoints_require_valid_json(self, client: TestClient):
        """Test GitHub endpoints validate JSON input"""
        invalid_json = "not json"
        
        response = client.post(
            "/api/v1/github/automation/create-pr",
            data=invalid_json,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # JSON parsing error
    
    def test_github_endpoints_handle_large_payloads(self, client: TestClient):
        """Test GitHub endpoints handle reasonably large payloads"""
        large_pr_request = {
            "title": "Large PR",
            "body": "A" * 10000,  # 10KB body
            "head_branch": "feature/large-change",
            "assignees": [f"user{i}" for i in range(50)],  # Many assignees
            "labels": [f"label{i}" for i in range(20)]  # Many labels
        }
        
        # Should handle large but reasonable payloads
        response = client.post("/api/v1/github/automation/create-pr", json=large_pr_request)
        
        # Might fail for other reasons, but not due to payload size
        assert response.status_code != 413  # Not "Payload Too Large"


class TestGitHubEndpointsSecurity:
    """Test security aspects of GitHub endpoints"""
    
    def test_webhook_endpoints_csrf_protection(self, client: TestClient):
        """Test webhook endpoints don't require CSRF tokens"""
        # Webhooks should work without CSRF protection
        webhook_payload = {"test": "payload"}
        
        response = client.post(
            "/api/v1/github/webhooks/github",
            json=webhook_payload,
            headers={"X-GitHub-Event": "ping"}
        )
        
        # Should not fail due to CSRF
        assert response.status_code != 403
    
    def test_automation_endpoints_input_validation(self, client: TestClient):
        """Test automation endpoints validate input thoroughly"""
        # Test with potentially malicious input
        malicious_request = {
            "title": "<script>alert('xss')</script>",
            "body": "javascript:void(0)",
            "head_branch": "../../../etc/passwd",
            "assignees": ["'; DROP TABLE users; --"],
            "labels": ["<img src=x onerror=alert(1)>"]
        }
        
        response = client.post("/api/v1/github/automation/create-pr", json=malicious_request)
        
        # Should either validate/sanitize input or reject it
        if response.status_code == 200:
            # If accepted, should be sanitized
            data = response.json()
            # Response should not contain raw malicious content
            response_str = json.dumps(data)
            assert "<script>" not in response_str
            assert "javascript:" not in response_str
        else:
            # Or should be rejected with validation error
            assert response.status_code in [400, 422]
    
    def test_github_endpoints_rate_limiting_headers(self, client: TestClient):
        """Test GitHub endpoints include appropriate rate limiting info"""
        response = client.get("/api/v1/github/automation/health")
        
        # Should include rate limiting headers if implemented
        # This is optional but good practice
        if "X-RateLimit-Limit" in response.headers:
            assert int(response.headers["X-RateLimit-Limit"]) > 0
        
        if "X-RateLimit-Remaining" in response.headers:
            assert int(response.headers["X-RateLimit-Remaining"]) >= 0
    
    def test_github_endpoints_no_sensitive_data_in_errors(self, client: TestClient):
        """Test GitHub endpoints don't leak sensitive data in errors"""
        # Force an error condition
        with patch('app.api.v1.endpoints.github.get_github_service') as mock_get_service:
            mock_get_service.side_effect = Exception("Internal error with API key: sk-secret-key-123")
            
            response = client.get("/api/v1/github/automation/health")
            
            assert response.status_code == 500
            error_text = response.text.lower()
            
            # Should not expose sensitive information
            sensitive_patterns = ["api key", "secret", "token", "password", "sk-"]
            for pattern in sensitive_patterns:
                assert pattern not in error_text


class TestGitHubEndpointsPerformance:
    """Test performance characteristics of GitHub endpoints"""
    
    def test_webhook_endpoints_respond_quickly(self, client: TestClient):
        """Test webhook endpoints respond within reasonable time"""
        import time
        
        webhook_payload = {
            "action": "opened",
            "pull_request": {"number": 123}
        }
        
        start_time = time.time()
        response = client.post(
            "/api/v1/github/webhooks/github",
            json=webhook_payload,
            headers={"X-GitHub-Event": "pull_request"}
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0  # Should respond within 5 seconds
    
    @patch('app.api.v1.endpoints.github.get_github_service')
    def test_automation_endpoints_timeout_handling(self, mock_get_service, client: TestClient):
        """Test automation endpoints handle timeouts gracefully"""
        import asyncio
        
        # Mock service that times out
        mock_service = AsyncMock()
        mock_service.create_automated_pr.side_effect = asyncio.TimeoutError("Request timeout")
        mock_get_service.return_value = mock_service
        
        pr_request = {
            "title": "Test PR",
            "body": "Test body",
            "head_branch": "test-branch"
        }
        
        response = client.post("/api/v1/github/automation/create-pr", json=pr_request)
        
        # Should handle timeout gracefully
        assert response.status_code in [500, 503, 504]
        
        data = response.json()
        assert "timeout" in data.get("detail", "").lower()
    
    def test_concurrent_webhook_processing(self, client: TestClient):
        """Test webhook endpoints handle concurrent requests"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def send_webhook():
            try:
                response = client.post(
                    "/api/v1/github/webhooks/github",
                    json={"test": "concurrent"},
                    headers={"X-GitHub-Event": "ping"}
                )
                results.put(response.status_code)
            except Exception as e:
                results.put(f"Error: {e}")
        
        # Send 3 concurrent webhooks
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=send_webhook)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All should succeed
        status_codes = []
        while not results.empty():
            status_codes.append(results.get())
        
        assert len(status_codes) == 3
        assert all(code == 200 for code in status_codes)