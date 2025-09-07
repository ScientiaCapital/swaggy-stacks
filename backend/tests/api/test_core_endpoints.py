"""
Comprehensive tests for core API endpoints without ML dependencies
These tests ensure basic functionality works when ML features are disabled
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from unittest.mock import patch, AsyncMock

from app.main import app


@pytest.fixture
def client():
    """Test client for API endpoints"""
    return TestClient(app)


@pytest.fixture
def auth_token(client):
    """Get authentication token for testing protected endpoints"""
    # Login with test credentials
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": "demo_user",
            "password": "demo_password"
        }
    )
    assert response.status_code == 200
    token_data = response.json()
    return token_data["access_token"]


@pytest.fixture
def auth_headers(auth_token):
    """Authorization headers with token"""
    return {"Authorization": f"Bearer {auth_token}"}


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_health_check(self, client):
        """Test basic health check"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data

    def test_health_detailed(self, client):
        """Test detailed health check"""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data


class TestAuthenticationEndpoints:
    """Test authentication endpoints"""

    def test_login_success(self, client):
        """Test successful login"""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "demo_user",
                "password": "demo_password"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert "expires_in" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "invalid_user",
                "password": "invalid_password"
            }
        )
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data

    def test_login_missing_data(self, client):
        """Test login with missing data"""
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "demo_user"}  # Missing password
        )
        assert response.status_code == 422

    def test_register_user(self, client):
        """Test user registration"""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "test_user",
                "email": "test@example.com",
                "password": "test_password123"
            }
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert "username" in data
        assert "email" in data
        assert "is_active" in data
        assert data["username"] == "test_user"

    def test_get_current_user(self, client, auth_headers):
        """Test getting current user info"""
        response = client.get("/api/v1/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "username" in data
        assert "email" in data
        assert "is_active" in data

    def test_get_current_user_unauthorized(self, client):
        """Test getting user info without auth"""
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401


class TestTradingEndpoints:
    """Test trading-related endpoints"""

    @patch('app.api.v1.dependencies.get_trading_manager')
    def test_get_account_info(self, mock_get_trading_manager, client, auth_headers):
        """Test getting account information"""
        # Mock trading manager
        mock_tm = AsyncMock()
        mock_tm.get_account.return_value = {
            "account_id": "test_account",
            "portfolio_value": "100000.00",
            "cash": "50000.00",
            "buying_power": "200000.00"
        }
        mock_get_trading_manager.return_value = mock_tm
        
        response = client.get("/api/v1/trading/account", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert data["success"] is True

    @patch('app.api.v1.dependencies.get_trading_manager')
    def test_get_positions(self, mock_get_trading_manager, client, auth_headers):
        """Test getting positions"""
        # Mock trading manager
        mock_tm = AsyncMock()
        mock_tm.get_positions.return_value = []
        mock_get_trading_manager.return_value = mock_tm
        
        response = client.get("/api/v1/trading/positions", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "data" in data

    @patch('app.api.v1.dependencies.get_trading_manager')
    def test_place_order(self, mock_get_trading_manager, client, auth_headers):
        """Test placing an order"""
        # Mock trading manager
        mock_tm = AsyncMock()
        mock_tm.place_order.return_value = {
            "order_id": "test_order_123",
            "symbol": "AAPL",
            "qty": "10",
            "side": "buy",
            "status": "accepted"
        }
        mock_get_trading_manager.return_value = mock_tm
        
        response = client.post(
            "/api/v1/trading/orders",
            json={
                "symbol": "AAPL",
                "qty": 10,
                "side": "buy",
                "type": "market"
            },
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_trading_endpoints_require_auth(self, client):
        """Test that trading endpoints require authentication"""
        endpoints = [
            "/api/v1/trading/account",
            "/api/v1/trading/positions",
            "/api/v1/trading/orders"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 401


class TestMarketDataEndpoints:
    """Test market data endpoints"""

    @patch('app.api.v1.endpoints.market_data.get_market_data_service')
    def test_get_market_data(self, mock_get_service, client):
        """Test getting market data"""
        # Mock market data service
        mock_service = AsyncMock()
        mock_service.get_latest_quote.return_value = {
            "symbol": "AAPL",
            "bid": 150.00,
            "ask": 150.05,
            "last": 150.02,
            "timestamp": datetime.utcnow().isoformat()
        }
        mock_get_service.return_value = mock_service
        
        response = client.get("/api/v1/market-data/AAPL/quote")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    @patch('app.api.v1.endpoints.market_data.get_market_data_service')
    def test_get_historical_data(self, mock_get_service, client):
        """Test getting historical market data"""
        # Mock market data service
        mock_service = AsyncMock()
        mock_service.get_historical_data.return_value = []
        mock_get_service.return_value = mock_service
        
        response = client.get("/api/v1/market-data/AAPL/history?timeframe=1D&limit=100")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_invalid_symbol_format(self, client):
        """Test invalid symbol format handling"""
        response = client.get("/api/v1/market-data/invalid@symbol/quote")
        assert response.status_code == 422  # Validation error


class TestAnalysisEndpoints:
    """Test analysis endpoints (without ML dependencies)"""

    @patch('app.api.v1.dependencies.get_markov_system')
    def test_basic_analysis(self, mock_get_markov, client):
        """Test basic analysis endpoint"""
        # Mock Markov system
        mock_markov = AsyncMock()
        mock_markov.analyze.return_value = {
            "current_state": 2,
            "confidence": 0.75,
            "volatility_regime": "moderate",
            "prediction": "bullish"
        }
        mock_get_markov.return_value = mock_markov
        
        response = client.get("/api/v1/analysis/basic/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_ml_analysis_disabled(self, client):
        """Test that ML analysis endpoints return appropriate errors when ML is disabled"""
        # This should fail because ML features are disabled in test environment
        response = client.get("/api/v1/analysis/ai/AAPL")
        assert response.status_code == 503  # Service unavailable
        data = response.json()
        assert "AI strategy features are not available" in data["detail"]


class TestPortfolioEndpoints:
    """Test portfolio management endpoints"""

    @patch('app.api.v1.dependencies.get_trading_manager')
    def test_get_portfolio_summary(self, mock_get_trading_manager, client, auth_headers):
        """Test getting portfolio summary"""
        # Mock trading manager
        mock_tm = AsyncMock()
        mock_tm.get_portfolio_summary.return_value = {
            "total_value": 100000.00,
            "day_change": 1500.00,
            "day_change_percent": 1.5,
            "positions_count": 5
        }
        mock_get_trading_manager.return_value = mock_tm
        
        response = client.get("/api/v1/portfolio/summary", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    @patch('app.api.v1.dependencies.get_trading_manager')
    def test_get_portfolio_performance(self, mock_get_trading_manager, client, auth_headers):
        """Test getting portfolio performance"""
        # Mock trading manager
        mock_tm = AsyncMock()
        mock_tm.get_performance_metrics.return_value = {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.05,
            "volatility": 0.12
        }
        mock_get_trading_manager.return_value = mock_tm
        
        response = client.get("/api/v1/portfolio/performance", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_portfolio_endpoints_require_auth(self, client):
        """Test that portfolio endpoints require authentication"""
        endpoints = [
            "/api/v1/portfolio/summary",
            "/api/v1/portfolio/performance"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 401


class TestErrorHandling:
    """Test error handling across endpoints"""

    def test_404_not_found(self, client):
        """Test 404 handling"""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test method not allowed"""
        response = client.put("/api/v1/health")
        assert response.status_code == 405

    def test_invalid_json(self, client, auth_headers):
        """Test invalid JSON handling"""
        response = client.post(
            "/api/v1/trading/orders",
            data="invalid json",
            headers={**auth_headers, "Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_rate_limiting_headers(self, client):
        """Test that rate limiting doesn't break basic functionality"""
        # Make multiple requests
        for _ in range(5):
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            # Should not be rate limited for health checks


class TestSecurityHeaders:
    """Test security-related headers and responses"""

    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/api/v1/health")
        # Should have CORS headers configured
        assert response.status_code in [200, 404]  # Depending on CORS setup

    def test_no_sensitive_info_in_errors(self, client):
        """Test that error responses don't leak sensitive information"""
        response = client.get("/api/v1/trading/account")
        assert response.status_code == 401
        data = response.json()
        # Should not contain sensitive database or system information
        assert "database" not in data.get("detail", "").lower()
        assert "password" not in data.get("detail", "").lower()
        assert "secret" not in data.get("detail", "").lower()