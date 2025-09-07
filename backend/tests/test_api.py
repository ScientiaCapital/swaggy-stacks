"""
API endpoint tests for Swaggy Stacks
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


def test_api_version():
    """Test the API version endpoint"""
    response = client.get("/api/v1/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "api_version" in data
    assert data["api_version"] == "v1"


def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Swaggy Stacks" in data["message"]


def test_invalid_endpoint():
    """Test that invalid endpoints return 404"""
    response = client.get("/invalid/endpoint")
    assert response.status_code == 404


def test_trading_status_endpoint():
    """Test the trading status endpoint"""
    response = client.get("/api/v1/trading/status")
    # May require auth, so accept 401 or 200
    assert response.status_code in [200, 401, 403]
    
    
def test_market_data_endpoint():
    """Test the market data endpoint"""
    response = client.get("/api/v1/market/symbols")
    # May require auth or configuration
    assert response.status_code in [200, 401, 403, 503]