"""
Authentication test fixtures for comprehensive testing
"""
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Generator
from unittest.mock import Mock, AsyncMock
import jwt

from app.core.auth import (
    create_access_token,
    get_password_hash,
    verify_password,
    authenticate_user,
)
from app.core.config import settings


# User Data Fixtures
@pytest.fixture
def demo_user_data() -> Dict[str, Any]:
    """Standard demo user data"""
    return {
        "id": 1,
        "username": "demo_user",
        "email": "demo@example.com",
        "hashed_password": get_password_hash("demo_password"),
        "is_active": True,
        "is_superuser": False,
        "created_at": datetime.utcnow(),
        "last_login": None,
    }


@pytest.fixture
def admin_user_data() -> Dict[str, Any]:
    """Admin user data for testing privileged operations"""
    return {
        "id": 2,
        "username": "admin_user",
        "email": "admin@example.com",
        "hashed_password": get_password_hash("admin_password"),
        "is_active": True,
        "is_superuser": True,
        "created_at": datetime.utcnow(),
        "last_login": datetime.utcnow() - timedelta(hours=1),
    }


@pytest.fixture
def inactive_user_data() -> Dict[str, Any]:
    """Inactive user data for testing disabled accounts"""
    return {
        "id": 3,
        "username": "inactive_user",
        "email": "inactive@example.com",
        "hashed_password": get_password_hash("inactive_password"),
        "is_active": False,
        "is_superuser": False,
        "created_at": datetime.utcnow() - timedelta(days=30),
        "last_login": None,
    }


@pytest.fixture
def multiple_test_users(demo_user_data, admin_user_data, inactive_user_data) -> Dict[str, Dict[str, Any]]:
    """Collection of test users for comprehensive testing"""
    return {
        "demo_user": demo_user_data,
        "admin_user": admin_user_data,
        "inactive_user": inactive_user_data,
    }


# Token Fixtures
@pytest.fixture
def valid_access_token(demo_user_data) -> str:
    """Valid JWT access token for demo user"""
    token_data = {
        "sub": demo_user_data["username"],
        "user_id": demo_user_data["id"],
        "email": demo_user_data["email"],
    }
    return create_access_token(token_data)


@pytest.fixture
def admin_access_token(admin_user_data) -> str:
    """Valid JWT access token for admin user"""
    token_data = {
        "sub": admin_user_data["username"],
        "user_id": admin_user_data["id"],
        "email": admin_user_data["email"],
        "is_superuser": admin_user_data["is_superuser"],
    }
    return create_access_token(token_data)


@pytest.fixture
def expired_token(demo_user_data) -> str:
    """Expired JWT token for testing token validation"""
    token_data = {
        "sub": demo_user_data["username"],
        "user_id": demo_user_data["id"],
        "email": demo_user_data["email"],
    }
    # Create token that expired 1 hour ago
    expires_delta = timedelta(hours=-1)
    return create_access_token(token_data, expires_delta)


@pytest.fixture
def invalid_signature_token(demo_user_data) -> str:
    """Token with invalid signature for testing token verification"""
    token_data = {
        "sub": demo_user_data["username"],
        "user_id": demo_user_data["id"],
        "email": demo_user_data["email"],
        "exp": datetime.utcnow() + timedelta(hours=1),
    }
    # Sign with wrong secret
    return jwt.encode(token_data, "wrong_secret", algorithm="HS256")


@pytest.fixture
def malformed_token() -> str:
    """Malformed JWT token for testing error handling"""
    return "invalid.jwt.token"


# Authentication Headers
@pytest.fixture
def auth_headers(valid_access_token) -> Dict[str, str]:
    """Standard authentication headers for API requests"""
    return {"Authorization": f"Bearer {valid_access_token}"}


@pytest.fixture
def admin_auth_headers(admin_access_token) -> Dict[str, str]:
    """Admin authentication headers for privileged API requests"""
    return {"Authorization": f"Bearer {admin_access_token}"}


@pytest.fixture
def expired_auth_headers(expired_token) -> Dict[str, str]:
    """Expired token headers for testing token expiration"""
    return {"Authorization": f"Bearer {expired_token}"}


@pytest.fixture
def invalid_auth_headers(invalid_signature_token) -> Dict[str, str]:
    """Invalid token headers for testing token validation"""
    return {"Authorization": f"Bearer {invalid_signature_token}"}


# Mock Database Fixtures
@pytest.fixture
def mock_user_db(multiple_test_users):
    """Mock user database for testing authentication functions"""
    mock_db = Mock()
    mock_db.users = multiple_test_users
    
    def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
        return multiple_test_users.get(username)
    
    def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
        for user in multiple_test_users.values():
            if user["id"] == user_id:
                return user
        return None
    
    def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
        for user in multiple_test_users.values():
            if user["email"] == email:
                return user
        return None
    
    mock_db.get_user_by_username = Mock(side_effect=get_user_by_username)
    mock_db.get_user_by_id = Mock(side_effect=get_user_by_id)
    mock_db.get_user_by_email = Mock(side_effect=get_user_by_email)
    
    return mock_db


# Authentication Function Fixtures
@pytest.fixture
def mock_authenticate_user(multiple_test_users):
    """Mock authenticate_user function for testing"""
    def _authenticate(username: str, password: str, user_db=None) -> Any:
        user = multiple_test_users.get(username)
        if not user:
            return False
            
        if not user["is_active"]:
            return False
            
        # Check password (simulate password verification)
        expected_passwords = {
            "demo_user": "demo_password",
            "admin_user": "admin_password",
            "inactive_user": "inactive_password",
        }
        
        if expected_passwords.get(username) != password:
            return False
            
        return user
    
    return Mock(side_effect=_authenticate)


# Test Scenarios
@pytest.fixture
def auth_test_scenarios():
    """Common authentication test scenarios"""
    return {
        "valid_login": {
            "username": "demo_user",
            "password": "demo_password",
            "should_succeed": True,
        },
        "invalid_password": {
            "username": "demo_user",
            "password": "wrong_password",
            "should_succeed": False,
        },
        "nonexistent_user": {
            "username": "nonexistent",
            "password": "password",
            "should_succeed": False,
        },
        "inactive_user": {
            "username": "inactive_user",
            "password": "inactive_password",
            "should_succeed": False,
        },
        "admin_login": {
            "username": "admin_user",
            "password": "admin_password",
            "should_succeed": True,
            "is_admin": True,
        },
    }


# Password Testing Fixtures
@pytest.fixture
def password_test_cases():
    """Test cases for password validation"""
    return [
        {"password": "demo_password", "valid": True},
        {"password": "short", "valid": False, "reason": "too_short"},
        {"password": "", "valid": False, "reason": "empty"},
        {"password": "a" * 100, "valid": True},  # Very long password
        {"password": "special!@#$%^&*()_+", "valid": True},
        {"password": "unicode_ñóñ_test", "valid": True},
    ]


# API Request Fixtures
@pytest.fixture
def login_request_data():
    """Standard login request data"""
    return {
        "username": "demo_user",
        "password": "demo_password",
    }


@pytest.fixture
def invalid_login_request_data():
    """Invalid login request data for testing failures"""
    return [
        {"username": "", "password": "demo_password"},
        {"username": "demo_user", "password": ""},
        {"username": "", "password": ""},
        {"username": "demo_user"},  # Missing password
        {"password": "demo_password"},  # Missing username
        {},  # Empty request
    ]


# Session and Context Fixtures
@pytest.fixture
def authenticated_user_context(demo_user_data):
    """Context representing an authenticated user session"""
    return {
        "user": demo_user_data,
        "token_issued_at": datetime.utcnow(),
        "token_expires_at": datetime.utcnow() + timedelta(hours=24),
        "session_id": "test_session_123",
        "ip_address": "127.0.0.1",
        "user_agent": "pytest-client",
    }


@pytest.fixture
def admin_user_context(admin_user_data):
    """Context representing an authenticated admin user session"""
    return {
        "user": admin_user_data,
        "token_issued_at": datetime.utcnow(),
        "token_expires_at": datetime.utcnow() + timedelta(hours=24),
        "session_id": "test_admin_session_456",
        "ip_address": "127.0.0.1",
        "user_agent": "pytest-client",
        "permissions": ["read", "write", "admin"],
    }


# Async Authentication Fixtures
@pytest.fixture
def async_mock_authenticate():
    """Async mock for authentication functions"""
    async_mock = AsyncMock()
    
    async def _authenticate(username: str, password: str) -> Dict[str, Any]:
        if username == "demo_user" and password == "demo_password":
            return {
                "id": 1,
                "username": "demo_user",
                "email": "demo@example.com",
                "is_active": True,
            }
        return {}
    
    async_mock.side_effect = _authenticate
    return async_mock


# Error Simulation Fixtures
@pytest.fixture
def auth_error_simulator():
    """Fixture for simulating authentication errors"""
    class AuthErrorSimulator:
        def __init__(self):
            self.should_fail = False
            self.error_type = None
            
        def enable_failure(self, error_type: str = "generic"):
            self.should_fail = True
            self.error_type = error_type
            
        def disable_failure(self):
            self.should_fail = False
            self.error_type = None
            
        def check_and_raise(self):
            if not self.should_fail:
                return
                
            if self.error_type == "database":
                raise ConnectionError("Database connection failed")
            elif self.error_type == "timeout":
                raise TimeoutError("Authentication timeout")
            elif self.error_type == "rate_limit":
                raise Exception("Rate limit exceeded")
            else:
                raise Exception("Authentication failed")
    
    return AuthErrorSimulator()


# Cleanup Fixtures
@pytest.fixture
def auth_cleanup():
    """Fixture for cleaning up authentication state after tests"""
    cleanup_tasks = []
    
    def add_cleanup(task):
        cleanup_tasks.append(task)
    
    yield add_cleanup
    
    # Execute cleanup tasks
    for task in cleanup_tasks:
        try:
            if callable(task):
                task()
        except Exception:
            pass  # Ignore cleanup errors in tests