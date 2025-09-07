"""
Integration tests for health check endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, Mock
import json

from app.main import app


class TestHealthEndpoints:
    """Test suite for health check endpoints"""
    
    def test_root_health_endpoint(self, client: TestClient):
        """Test root health check endpoint"""
        response = client.get("/api/v1/health/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
    
    def test_system_health_endpoint(self, client: TestClient):
        """Test system health endpoint"""
        response = client.get("/api/v1/health/system")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "status" in data
        assert "timestamp" in data
        assert "checks" in data
        assert "version" in data
        
        # Check system components
        checks = data["checks"]
        expected_checks = ["database", "redis", "trading_system"]
        
        for check in expected_checks:
            assert check in checks
            assert "status" in checks[check]
    
    @patch('app.api.v1.endpoints.health.get_mcp_orchestrator')
    def test_mcp_health_endpoint_healthy(self, mock_get_orchestrator, client: TestClient):
        """Test MCP health endpoint when all servers are healthy"""
        # Mock healthy MCP orchestrator
        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_server_status.return_value = {
            'github': Mock(connected=True, error_count=0, last_health_check='2024-01-01T00:00:00'),
            'memory': Mock(connected=True, error_count=0, last_health_check='2024-01-01T00:00:00'),
            'tavily': Mock(connected=True, error_count=1, last_health_check='2024-01-01T00:00:00')
        }
        mock_get_orchestrator.return_value = mock_orchestrator
        
        response = client.get("/api/v1/health/mcp")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "servers" in data
        assert len(data["servers"]) == 3
        
        # Check server details
        for server_name, server_status in data["servers"].items():
            assert "connected" in server_status
            assert "error_count" in server_status
    
    @patch('app.api.v1.endpoints.health.get_mcp_orchestrator')
    def test_mcp_health_endpoint_degraded(self, mock_get_orchestrator, client: TestClient):
        """Test MCP health endpoint when some servers are down"""
        # Mock partially degraded MCP orchestrator
        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_server_status.return_value = {
            'github': Mock(connected=True, error_count=0, last_health_check='2024-01-01T00:00:00'),
            'memory': Mock(connected=False, error_count=5, last_health_check='2024-01-01T00:00:00'),
            'tavily': Mock(connected=True, error_count=2, last_health_check='2024-01-01T00:00:00')
        }
        mock_get_orchestrator.return_value = mock_orchestrator
        
        response = client.get("/api/v1/health/mcp")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "degraded"
        assert data["servers"]["memory"]["connected"] is False
        assert data["servers"]["memory"]["error_count"] == 5
    
    @patch('app.api.v1.endpoints.health.get_mcp_orchestrator')
    def test_mcp_health_endpoint_unhealthy(self, mock_get_orchestrator, client: TestClient):
        """Test MCP health endpoint when orchestrator is unavailable"""
        # Mock unavailable orchestrator
        mock_get_orchestrator.side_effect = Exception("Orchestrator unavailable")
        
        response = client.get("/api/v1/health/mcp")
        
        assert response.status_code == 503
        data = response.json()
        
        assert data["status"] == "unhealthy"
        assert "error" in data
    
    def test_specific_mcp_server_health(self, client: TestClient):
        """Test health check for specific MCP server"""
        with patch('app.api.v1.endpoints.health.get_mcp_orchestrator') as mock_get_orchestrator:
            # Mock orchestrator with specific server status
            mock_orchestrator = AsyncMock()
            mock_orchestrator.is_server_available.return_value = True
            mock_orchestrator.get_server_status.return_value = {
                'github': Mock(
                    connected=True, 
                    error_count=0, 
                    last_health_check='2024-01-01T00:00:00'
                )
            }
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.get("/api/v1/health/mcp/github")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["server_type"] == "github"
            assert data["status"] == "healthy"
            assert data["connected"] is True
            assert data["error_count"] == 0
    
    def test_specific_mcp_server_not_found(self, client: TestClient):
        """Test health check for non-existent MCP server"""
        response = client.get("/api/v1/health/mcp/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    def test_health_endpoints_with_database_error(self, client: TestClient):
        """Test health endpoints handle database errors gracefully"""
        with patch('app.core.database.get_db') as mock_get_db:
            mock_get_db.side_effect = Exception("Database connection failed")
            
            response = client.get("/api/v1/health/system")
            
            assert response.status_code == 200  # Should still respond
            data = response.json()
            
            assert data["status"] in ["degraded", "unhealthy"]
            assert data["checks"]["database"]["status"] == "error"
    
    def test_health_endpoints_response_time(self, client: TestClient):
        """Test health endpoints respond quickly"""
        import time
        
        start_time = time.time()
        response = client.get("/api/v1/health/")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Should respond within 1 second
    
    def test_health_endpoint_caching(self, client: TestClient):
        """Test health endpoint response includes appropriate cache headers"""
        response = client.get("/api/v1/health/")
        
        assert response.status_code == 200
        
        # Health checks should not be cached
        cache_control = response.headers.get('cache-control', '').lower()
        assert 'no-cache' in cache_control or 'no-store' in cache_control
    
    def test_health_endpoint_concurrent_requests(self, client: TestClient):
        """Test health endpoints handle concurrent requests"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            try:
                response = client.get("/api/v1/health/")
                results.put(response.status_code)
            except Exception as e:
                results.put(f"Error: {e}")
        
        # Make 5 concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check all requests succeeded
        status_codes = []
        while not results.empty():
            status_codes.append(results.get())
        
        assert len(status_codes) == 5
        assert all(code == 200 for code in status_codes)
    
    def test_health_endpoint_content_type(self, client: TestClient):
        """Test health endpoints return correct content type"""
        response = client.get("/api/v1/health/")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Ensure response is valid JSON
        data = response.json()
        assert isinstance(data, dict)
    
    @patch('app.api.v1.endpoints.health.get_mcp_orchestrator')
    def test_mcp_health_with_timeout(self, mock_get_orchestrator, client: TestClient):
        """Test MCP health endpoint handles timeout gracefully"""
        import asyncio
        
        # Mock orchestrator that times out
        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_server_status.side_effect = asyncio.TimeoutError("Request timeout")
        mock_get_orchestrator.return_value = mock_orchestrator
        
        response = client.get("/api/v1/health/mcp")
        
        assert response.status_code == 503
        data = response.json()
        
        assert data["status"] == "unhealthy"
        assert "timeout" in data.get("error", "").lower()


class TestHealthEndpointsSecurity:
    """Test security aspects of health endpoints"""
    
    def test_health_endpoints_no_sensitive_info(self, client: TestClient):
        """Test health endpoints don't expose sensitive information"""
        response = client.get("/api/v1/health/system")
        
        assert response.status_code == 200
        data = response.json()
        
        # Convert response to string for searching
        response_text = json.dumps(data).lower()
        
        # Check for sensitive information that shouldn't be exposed
        sensitive_keywords = [
            "password", "secret", "key", "token", "credential",
            "connection_string", "database_url", "redis_url"
        ]
        
        for keyword in sensitive_keywords:
            assert keyword not in response_text, f"Sensitive keyword '{keyword}' found in health response"
    
    def test_health_endpoints_no_authentication_required(self, client: TestClient):
        """Test health endpoints are accessible without authentication"""
        # Health endpoints should be public for monitoring
        endpoints = [
            "/api/v1/health/",
            "/api/v1/health/system",
            "/api/v1/health/mcp"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            # Should not return 401 (Unauthorized) or 403 (Forbidden)
            assert response.status_code not in [401, 403]
    
    def test_health_endpoints_rate_limiting_friendly(self, client: TestClient):
        """Test health endpoints can be called frequently (for monitoring)"""
        # Make multiple rapid requests
        for i in range(10):
            response = client.get("/api/v1/health/")
            assert response.status_code == 200
    
    def test_health_endpoints_minimal_logging(self, client: TestClient):
        """Test health endpoints don't generate excessive logs"""
        with patch('structlog.get_logger') as mock_logger:
            mock_log = Mock()
            mock_logger.return_value = mock_log
            
            # Make health check request
            response = client.get("/api/v1/health/")
            assert response.status_code == 200
            
            # Health checks should log minimally
            # Allow some logging but not excessive
            assert mock_log.info.call_count <= 2


class TestHealthEndpointsMonitoring:
    """Test health endpoints for monitoring integration"""
    
    def test_health_prometheus_metrics_format(self, client: TestClient):
        """Test health endpoints return monitoring-friendly data"""
        response = client.get("/api/v1/health/system")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check for monitoring-friendly fields
        assert "status" in data
        assert "timestamp" in data
        
        # Status should be one of standard values
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        
        # Timestamp should be ISO format
        from datetime import datetime
        try:
            datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
        except ValueError:
            pytest.fail("Timestamp is not in valid ISO format")
    
    def test_health_endpoints_uptime_info(self, client: TestClient):
        """Test health endpoints include uptime information"""
        response = client.get("/api/v1/health/system")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should include timing information useful for monitoring
        assert "timestamp" in data
        
        # If available, uptime should be numeric
        if "uptime" in data:
            assert isinstance(data["uptime"], (int, float))
            assert data["uptime"] >= 0
    
    def test_health_endpoints_service_discovery(self, client: TestClient):
        """Test health endpoints provide service discovery information"""
        response = client.get("/api/v1/health/system")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should include version for service discovery
        if "version" in data:
            assert isinstance(data["version"], str)
            assert len(data["version"]) > 0
        
        # Should include environment if available
        if "environment" in data:
            assert data["environment"] in ["development", "testing", "staging", "production"]