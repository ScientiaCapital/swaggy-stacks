"""
Comprehensive NATS Security Framework Tests
==========================================

Tests for the complete NATS security implementation including:
- JWT authentication and token management
- Message encryption/decryption (Fernet)
- HMAC-SHA256 message signing
- Rate limiting per agent type
- Subject-based permissions
- Security audit logging
- TLS connection security
"""

import asyncio
import json
import pytest
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from app.messaging.nats_coordinator import (
    NATSSecurityManager,
    NATSAgentCoordinator,
    SecurityCredentials,
    SecurityEvent,
    get_nats_coordinator
)
from app.core.config import settings


class TestNATSSecurityManager:
    """Test suite for NATSSecurityManager core functionality"""

    @pytest.fixture
    def security_manager(self):
        """Create a test security manager"""
        test_key = "test_key_" + secrets.token_urlsafe(32)
        return NATSSecurityManager(master_key=test_key)

    def test_security_manager_initialization(self, security_manager):
        """Test security manager initializes properly"""
        assert security_manager.encryption_enabled
        assert security_manager.message_signing_enabled
        assert security_manager.rate_limiting_enabled
        assert security_manager.jwt_algorithm == "HS256"
        assert security_manager.jwt_expiry_hours == 24
        assert len(security_manager.credentials_store) == 0
        assert len(security_manager.audit_log) == 0

    def test_agent_credentials_creation(self, security_manager):
        """Test creating agent credentials with proper security"""
        agent_id = "test_market_analyst_001"
        agent_type = "market_analyst"

        credentials = security_manager.create_agent_credentials(
            agent_id=agent_id,
            agent_type=agent_type,
            validity_days=30
        )

        assert credentials.agent_id == agent_id
        assert len(credentials.api_key) > 30  # Secure key length
        assert len(credentials.secret_key) > 30  # Secure key length
        assert credentials.expires_at > datetime.now()
        assert credentials.created_at <= datetime.now()

        # Check permissions are assigned correctly
        expected_permissions = [
            "agents.decisions.market_analyst.*",
            "agents.status.market_analyst.*",
            "agents.market.*",
            "agents.consensus.request.*",
            "agents.consensus.response.*"
        ]
        assert credentials.permissions == expected_permissions

        # Verify stored in credentials store
        assert agent_id in security_manager.credentials_store

        # Check audit log
        assert len(security_manager.audit_log) == 1
        assert security_manager.audit_log[0].event_type == "CREDENTIALS_CREATED"

    def test_jwt_token_generation_and_verification(self, security_manager):
        """Test JWT token lifecycle - generation, verification, expiry"""
        agent_id = "test_risk_advisor_001"
        agent_type = "risk_advisor"

        # Create credentials first
        credentials = security_manager.create_agent_credentials(agent_id, agent_type)

        # Generate JWT token
        token = security_manager.generate_jwt_token(agent_id)
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are longer

        # Verify token
        payload = security_manager.verify_jwt_token(token)
        assert payload is not None
        assert payload["agent_id"] == agent_id
        assert payload["permissions"] == credentials.permissions
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload  # Unique token ID

        # Check last_used is updated
        updated_credentials = security_manager.credentials_store[agent_id]
        assert updated_credentials.last_used is not None

    def test_jwt_token_expiry_handling(self, security_manager):
        """Test JWT token expiry is properly handled"""
        agent_id = "test_expired_agent"
        agent_type = "strategy_optimizer"

        # Create credentials with past expiry
        credentials = security_manager.create_agent_credentials(agent_id, agent_type)
        credentials.expires_at = datetime.now() - timedelta(days=1)  # Expired

        # Try to generate token for expired credentials
        token = security_manager.generate_jwt_token(agent_id)
        assert token is None

        # Check audit log for expiry event
        expiry_events = [e for e in security_manager.audit_log if e.event_type == "TOKEN_EXPIRED"]
        assert len(expiry_events) == 1

    def test_token_revocation(self, security_manager):
        """Test JWT token revocation functionality"""
        agent_id = "test_revoke_agent"
        agent_type = "performance_coach"

        # Create credentials and token
        security_manager.create_agent_credentials(agent_id, agent_type)
        token = security_manager.generate_jwt_token(agent_id)

        # Verify token works initially
        payload = security_manager.verify_jwt_token(token)
        assert payload is not None

        # Revoke token
        security_manager.revoke_token(token, agent_id)

        # Verify token is now invalid
        payload = security_manager.verify_jwt_token(token)
        assert payload is None

        # Check revocation in audit log
        revoke_events = [e for e in security_manager.audit_log if e.event_type == "TOKEN_REVOKED"]
        assert len(revoke_events) == 1

    def test_subject_permissions_validation(self, security_manager):
        """Test subject-based permission validation"""
        agent_id = "test_permissions_agent"
        agent_type = "market_analyst"

        # Create credentials
        security_manager.create_agent_credentials(agent_id, agent_type)

        # Test allowed subjects
        allowed_subjects = [
            "agents.decisions.market_analyst.buy_signal",
            "agents.status.market_analyst.health",
            "agents.market.price_update",
            "agents.consensus.request.vote",
            "agents.consensus.response.result"
        ]

        for subject in allowed_subjects:
            assert security_manager.check_permissions(agent_id, subject)

        # Test forbidden subjects
        forbidden_subjects = [
            "agents.decisions.risk_advisor.sell_signal",  # Wrong agent type
            "agents.execution.place_order",  # No execution permissions
            "admin.system.shutdown",  # Admin only
            "agents.coordination.all_agents"  # Not in permissions
        ]

        for subject in forbidden_subjects:
            assert not security_manager.check_permissions(agent_id, subject)

    def test_rate_limiting_functionality(self, security_manager):
        """Test rate limiting per agent type"""
        agent_id = "test_rate_limit_agent"
        agent_type = "performance_coach"  # 200 msgs/min limit

        # Create credentials
        security_manager.create_agent_credentials(agent_id, agent_type)

        # Test within limits
        for i in range(50):  # Well within 200/min limit
            assert security_manager.check_rate_limit(agent_id, agent_type)

        # Mock many requests to exceed limit
        import time
        security_manager.rate_tracking[agent_id] = [
            datetime.now() for _ in range(201)  # Exceed 200/min limit
        ]

        # Should now be rate limited
        assert not security_manager.check_rate_limit(agent_id, agent_type)

        # Check audit log for rate limit violation
        rate_limit_events = [e for e in security_manager.audit_log if e.event_type == "RATE_LIMIT_EXCEEDED"]
        assert len(rate_limit_events) == 1

    @pytest.mark.asyncio
    async def test_message_encryption_decryption(self, security_manager):
        """Test message encryption and decryption"""
        agent_id = "test_crypto_agent"
        test_message = b"This is a secret trading signal: BUY AAPL 100 shares"

        # Encrypt message
        encrypted = await security_manager.encrypt_message(test_message, agent_id)
        assert encrypted != test_message
        assert len(encrypted) > len(test_message)  # Encryption adds overhead

        # Decrypt message
        decrypted = await security_manager.decrypt_message(encrypted, agent_id)
        assert decrypted == test_message

        # Check audit log for crypto events
        crypto_events = [e for e in security_manager.audit_log if "ENCRYPTED" in e.event_type or "DECRYPTED" in e.event_type]
        assert len(crypto_events) == 2

    def test_message_signing_and_verification(self, security_manager):
        """Test HMAC message signing and verification"""
        agent_id = "test_signing_agent"
        agent_type = "risk_advisor"
        test_message = b"Risk assessment: Portfolio exposure too high"

        # Create credentials
        security_manager.create_agent_credentials(agent_id, agent_type)

        # Sign message
        signature = security_manager.sign_message(test_message, agent_id)
        assert signature
        assert len(signature) == 64  # SHA256 hex digest length

        # Verify signature
        is_valid = security_manager.verify_signature(test_message, signature, agent_id)
        assert is_valid

        # Test invalid signature
        invalid_signature = "invalid_signature_12345"
        is_valid = security_manager.verify_signature(test_message, invalid_signature, agent_id)
        assert not is_valid

        # Test tampered message
        tampered_message = b"Risk assessment: Portfolio exposure is fine"  # Different message
        is_valid = security_manager.verify_signature(tampered_message, signature, agent_id)
        assert not is_valid

    def test_security_metrics_reporting(self, security_manager):
        """Test security metrics collection and reporting"""
        # Create some test activity
        agent_id = "test_metrics_agent"
        agent_type = "market_analyst"

        security_manager.create_agent_credentials(agent_id, agent_type)
        token = security_manager.generate_jwt_token(agent_id)

        # Generate some failures for metrics
        security_manager.verify_jwt_token("invalid_token")
        security_manager.check_permissions("nonexistent_agent", "some.subject")

        # Get security metrics
        metrics = security_manager.get_security_metrics()

        assert "total_audit_events" in metrics
        assert "auth_failures" in metrics
        assert "permission_denials" in metrics
        assert "active_agents" in metrics
        assert "security_score" in metrics
        assert "encryption_enabled" in metrics
        assert "timestamp" in metrics

        assert metrics["active_agents"] == 1
        assert metrics["auth_failures"] >= 1
        assert metrics["permission_denials"] >= 1
        assert 0 <= metrics["security_score"] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])