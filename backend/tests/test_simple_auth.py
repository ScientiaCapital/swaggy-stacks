"""
Simple authentication tests without heavy ML dependencies
"""

import pytest
from app.core.auth import (
    verify_password, 
    get_password_hash, 
    create_access_token, 
    verify_token,
    authenticate_user
)


def test_password_hashing():
    """Test password hashing and verification"""
    password = "test_password_123"
    hashed = get_password_hash(password)
    
    assert verify_password(password, hashed) is True
    assert verify_password("wrong_password", hashed) is False


def test_jwt_token_creation_and_verification():
    """Test JWT token creation and verification"""
    test_data = {"sub": "test_user", "user_id": 1}
    
    token = create_access_token(test_data)
    assert token is not None
    assert isinstance(token, str)
    
    # Verify token
    decoded_data = verify_token(token)
    assert decoded_data is not None
    assert decoded_data["username"] == "test_user"
    assert decoded_data["user_id"] == 1


def test_jwt_token_invalid():
    """Test JWT token verification with invalid token"""
    invalid_token = "invalid.token.here"
    decoded_data = verify_token(invalid_token)
    assert decoded_data is None


def test_authenticate_demo_user():
    """Test demo user authentication"""
    # Test valid credentials
    user = authenticate_user("demo_user", "demo_password", None)
    assert user is not False
    assert user["username"] == "demo_user"
    assert user["id"] == 1
    
    # Test invalid credentials
    user = authenticate_user("demo_user", "wrong_password", None)
    assert user is False
    
    user = authenticate_user("wrong_user", "demo_password", None)
    assert user is False