"""
NATS Agent Coordinator - Ultra-Low Latency Messaging System
===========================================================

Replaces WebSocket-based coordination with NATS messaging for sub-millisecond latency.
Provides the same interface as AgentCoordinationManager but with NATS pub/sub.

Key Performance Features:
- 0.5-2ms latency (vs 5-15ms WebSocket)
- JetStream persistence for critical decisions
- Automatic failover and reconnection
- Subject-based routing for efficient message delivery

Subject Hierarchy:
- agents.decisions.{agent_id}.{symbol}     - Individual agent decisions
- agents.consensus.request.{symbol}        - Consensus voting requests
- agents.consensus.response.{consensus_id} - Consensus responses
- agents.status.{agent_type}.{agent_id}    - Agent status updates
- agents.coordination.{channel}            - Coordination channels
- agents.execution.{order_id}              - Trade execution status
- agents.market.{symbol}                   - Market data distribution
"""

import asyncio
import gzip
import hashlib
import hmac
import json
import os
import secrets
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import asdict, dataclass
import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import base64
import jwt

from nats.aio.client import Client as NATS
from nats.aio.errors import ErrConnectionClosed, ErrTimeout, ErrNoServers
from nats.js.api import StreamConfig, ConsumerConfig
from nats.js.errors import NotFoundError

from app.websockets.agent_coordination_socket import (
    AgentDecisionUpdate,
    AgentStatusUpdate,
    AgentCoordinationMessage,
    ToolExecutionResult,
)

logger = structlog.get_logger()


@dataclass
class SecurityCredentials:
    """Security credentials for agent authentication"""
    agent_id: str
    api_key: str
    secret_key: str
    permissions: List[str]
    expires_at: datetime
    created_at: datetime
    last_used: Optional[datetime] = None


@dataclass
class SecurityEvent:
    """Security audit event"""
    event_type: str  # AUTH_SUCCESS, AUTH_FAILURE, PERMISSION_DENIED, MESSAGE_ENCRYPTED, etc.
    agent_id: str
    subject: str
    timestamp: datetime
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    severity: str = "INFO"  # INFO, WARNING, CRITICAL


class NATSSecurityManager:
    """Comprehensive security manager for NATS agent communication"""

    def __init__(self, master_key: Optional[str] = None):
        # Encryption setup
        self.master_key = master_key or os.getenv("NATS_MASTER_KEY")
        if not self.master_key:
            # Generate a new master key if none provided
            self.master_key = Fernet.generate_key().decode()
            logger.warning("Generated new NATS master key - store securely!",
                          key_preview=self.master_key[:16] + "...")

        # Handle Fernet key initialization properly
        if isinstance(self.master_key, str):
            # If it's a string, check if it's already a valid Fernet key
            try:
                # Try to use it directly (assuming it's base64-encoded)
                self.cipher_suite = Fernet(self.master_key.encode() if isinstance(self.master_key, str) else self.master_key)
            except ValueError:
                # If not a valid Fernet key, generate one from the string
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
                import base64
                
                # Use PBKDF2 to derive a Fernet key from the provided string
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'nats_security_salt',  # Fixed salt for deterministic keys
                    iterations=100000,
                )
                derived_key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
                self.cipher_suite = Fernet(derived_key)
        else:
            # If it's already bytes, use directly
            self.cipher_suite = Fernet(self.master_key)

        # JWT configuration
        self.jwt_secret = os.getenv("NATS_JWT_SECRET", secrets.token_urlsafe(32))
        self.jwt_algorithm = "HS256"
        self.jwt_expiry_hours = int(os.getenv("NATS_JWT_EXPIRY_HOURS", "24"))

        # Authentication store
        self.credentials_store: Dict[str, SecurityCredentials] = {}
        self.revoked_tokens: Set[str] = set()

        # Security audit log
        self.audit_log: List[SecurityEvent] = []
        self.max_audit_entries = int(os.getenv("NATS_MAX_AUDIT_ENTRIES", "10000"))

        # Security policies
        self.encryption_enabled = os.getenv("NATS_ENCRYPTION_ENABLED", "true").lower() == "true"
        self.message_signing_enabled = os.getenv("NATS_MESSAGE_SIGNING_ENABLED", "true").lower() == "true"
        self.rate_limiting_enabled = os.getenv("NATS_RATE_LIMITING_ENABLED", "true").lower() == "true"

        # Rate limiting (messages per minute per agent)
        self.rate_limits = {
            "market_analyst": int(os.getenv("NATS_RATE_LIMIT_MARKET_ANALYST", "1000")),
            "risk_advisor": int(os.getenv("NATS_RATE_LIMIT_RISK_ADVISOR", "500")),
            "strategy_optimizer": int(os.getenv("NATS_RATE_LIMIT_STRATEGY_OPTIMIZER", "300")),
            "performance_coach": int(os.getenv("NATS_RATE_LIMIT_PERFORMANCE_COACH", "200")),
            "default": int(os.getenv("NATS_RATE_LIMIT_DEFAULT", "100"))
        }
        self.rate_tracking: Dict[str, deque] = {}

        # Subject-based permissions
        self.permission_matrix = {
            "market_analyst": [
                "agents.decisions.market_analyst.*",
                "agents.status.market_analyst.*",
                "agents.market.*",
                "agents.consensus.request.*",
                "agents.consensus.response.*"
            ],
            "risk_advisor": [
                "agents.decisions.risk_advisor.*",
                "agents.status.risk_advisor.*",
                "agents.consensus.request.*",
                "agents.consensus.response.*",
                "agents.execution.*"
            ],
            "strategy_optimizer": [
                "agents.decisions.strategy_optimizer.*",
                "agents.status.strategy_optimizer.*",
                "agents.coordination.*",
                "agents.consensus.request.*",
                "agents.consensus.response.*"
            ],
            "performance_coach": [
                "agents.decisions.performance_coach.*",
                "agents.status.performance_coach.*",
                "agents.coordination.performance_coaches"
            ],
            "admin": ["agents.>"],  # Full access for admin users
            "default": [  # Default permissions for unknown agent types
                "agents.status.*.{agent_id}",
                "agents.coordination.all_agents"
            ]
        }

        logger.info("NATS Security Manager initialized",
                   encryption_enabled=self.encryption_enabled,
                   message_signing=self.message_signing_enabled,
                   rate_limiting=self.rate_limiting_enabled)

    def create_agent_credentials(self, agent_id: str, agent_type: str,
                                validity_days: int = 30) -> SecurityCredentials:
        """Create secure credentials for an agent"""
        api_key = secrets.token_urlsafe(32)
        secret_key = secrets.token_urlsafe(48)

        permissions = self.permission_matrix.get(agent_type, self.permission_matrix["default"])
        expires_at = datetime.now() + timedelta(days=validity_days)

        credentials = SecurityCredentials(
            agent_id=agent_id,
            api_key=api_key,
            secret_key=secret_key,
            permissions=permissions,
            expires_at=expires_at,
            created_at=datetime.now()
        )

        self.credentials_store[agent_id] = credentials

        self._log_security_event("CREDENTIALS_CREATED", agent_id, "admin.credentials",
                                {"agent_type": agent_type, "validity_days": validity_days})

        return credentials

    def generate_jwt_token(self, agent_id: str) -> Optional[str]:
        """Generate JWT token for agent authentication"""
        if agent_id not in self.credentials_store:
            return None

        credentials = self.credentials_store[agent_id]
        if datetime.now() > credentials.expires_at:
            self._log_security_event("TOKEN_EXPIRED", agent_id, "auth.token",
                                    {"reason": "credentials_expired"}, severity="WARNING")
            return None

        payload = {
            "agent_id": agent_id,
            "permissions": credentials.permissions,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=self.jwt_expiry_hours),
            "jti": secrets.token_urlsafe(16)  # Unique token ID
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        credentials.last_used = datetime.now()

        self._log_security_event("TOKEN_GENERATED", agent_id, "auth.token",
                                {"expires_in_hours": self.jwt_expiry_hours})

        return token

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            # Check if token is revoked
            if token in self.revoked_tokens:
                self._log_security_event("AUTH_FAILURE", "unknown", "auth.verify",
                                        {"reason": "token_revoked"}, severity="WARNING")
                return None

            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])

            # Validate agent still exists and is active
            agent_id = payload.get("agent_id")
            if agent_id not in self.credentials_store:
                self._log_security_event("AUTH_FAILURE", agent_id, "auth.verify",
                                        {"reason": "agent_not_found"}, severity="WARNING")
                return None

            credentials = self.credentials_store[agent_id]
            if datetime.now() > credentials.expires_at:
                self._log_security_event("AUTH_FAILURE", agent_id, "auth.verify",
                                        {"reason": "credentials_expired"}, severity="WARNING")
                return None

            credentials.last_used = datetime.now()
            self._log_security_event("AUTH_SUCCESS", agent_id, "auth.verify", {})

            return payload

        except jwt.ExpiredSignatureError:
            self._log_security_event("AUTH_FAILURE", "unknown", "auth.verify",
                                    {"reason": "token_expired"}, severity="WARNING")
            return None
        except jwt.InvalidTokenError as e:
            self._log_security_event("AUTH_FAILURE", "unknown", "auth.verify",
                                    {"reason": "invalid_token", "error": str(e)}, severity="WARNING")
            return None

    def revoke_token(self, token: str, agent_id: str):
        """Revoke a JWT token"""
        self.revoked_tokens.add(token)
        self._log_security_event("TOKEN_REVOKED", agent_id, "auth.revoke", {}, severity="WARNING")

    def check_permissions(self, agent_id: str, subject: str) -> bool:
        """Check if agent has permission for subject"""
        if agent_id not in self.credentials_store:
            self._log_security_event("PERMISSION_DENIED", agent_id, subject,
                                    {"reason": "agent_not_found"}, severity="WARNING")
            return False

        credentials = self.credentials_store[agent_id]

        # Check each permission pattern
        for permission in credentials.permissions:
            if self._match_subject_pattern(subject, permission):
                self._log_security_event("PERMISSION_GRANTED", agent_id, subject,
                                        {"matched_pattern": permission})
                return True

        self._log_security_event("PERMISSION_DENIED", agent_id, subject,
                                {"available_permissions": credentials.permissions}, severity="WARNING")
        return False

    def check_rate_limit(self, agent_id: str, agent_type: str = "default") -> bool:
        """Check if agent is within rate limits"""
        if not self.rate_limiting_enabled:
            return True

        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Initialize tracking for new agent
        if agent_id not in self.rate_tracking:
            self.rate_tracking[agent_id] = deque()

        # Clean old entries
        agent_requests = self.rate_tracking[agent_id]
        while agent_requests and agent_requests[0] < minute_ago:
            agent_requests.popleft()

        # Check limit
        limit = self.rate_limits.get(agent_type, self.rate_limits["default"])
        if len(agent_requests) >= limit:
            self._log_security_event("RATE_LIMIT_EXCEEDED", agent_id, "rate_limit",
                                    {"requests_per_minute": len(agent_requests), "limit": limit},
                                    severity="WARNING")
            return False

        # Record this request
        agent_requests.append(now)
        return True

    async def encrypt_message(self, message: bytes, agent_id: str) -> bytes:
        """Encrypt message payload"""
        if not self.encryption_enabled:
            return message

        try:
            encrypted = self.cipher_suite.encrypt(message)
            self._log_security_event("MESSAGE_ENCRYPTED", agent_id, "crypto.encrypt",
                                    {"payload_size": len(message)})
            return encrypted
        except Exception as e:
            logger.error("Failed to encrypt message", agent_id=agent_id, error=str(e))
            self._log_security_event("ENCRYPTION_FAILED", agent_id, "crypto.encrypt",
                                    {"error": str(e)}, severity="CRITICAL")
            raise

    async def decrypt_message(self, encrypted_message: bytes, agent_id: str) -> bytes:
        """Decrypt message payload"""
        if not self.encryption_enabled:
            return encrypted_message

        try:
            decrypted = self.cipher_suite.decrypt(encrypted_message)
            self._log_security_event("MESSAGE_DECRYPTED", agent_id, "crypto.decrypt",
                                    {"payload_size": len(decrypted)})
            return decrypted
        except Exception as e:
            logger.error("Failed to decrypt message", agent_id=agent_id, error=str(e))
            self._log_security_event("DECRYPTION_FAILED", agent_id, "crypto.decrypt",
                                    {"error": str(e)}, severity="CRITICAL")
            raise

    def sign_message(self, message: bytes, agent_id: str) -> str:
        """Create HMAC signature for message"""
        if not self.message_signing_enabled:
            return ""

        if agent_id not in self.credentials_store:
            return ""

        secret_key = self.credentials_store[agent_id].secret_key
        signature = hmac.new(
            secret_key.encode(),
            message,
            hashlib.sha256
        ).hexdigest()

        self._log_security_event("MESSAGE_SIGNED", agent_id, "crypto.sign", {})
        return signature

    def verify_signature(self, message: bytes, signature: str, agent_id: str) -> bool:
        """Verify HMAC signature"""
        if not self.message_signing_enabled:
            return True

        if agent_id not in self.credentials_store:
            self._log_security_event("SIGNATURE_VERIFICATION_FAILED", agent_id, "crypto.verify",
                                    {"reason": "agent_not_found"}, severity="WARNING")
            return False

        secret_key = self.credentials_store[agent_id].secret_key
        expected_signature = hmac.new(
            secret_key.encode(),
            message,
            hashlib.sha256
        ).hexdigest()

        is_valid = hmac.compare_digest(signature, expected_signature)

        event_type = "SIGNATURE_VERIFIED" if is_valid else "SIGNATURE_VERIFICATION_FAILED"
        severity = "INFO" if is_valid else "WARNING"
        self._log_security_event(event_type, agent_id, "crypto.verify",
                                {"signature_valid": is_valid}, severity=severity)

        return is_valid

    def get_security_audit_log(self, limit: int = 100,
                              severity: Optional[str] = None) -> List[SecurityEvent]:
        """Get security audit log entries"""
        filtered_log = self.audit_log

        if severity:
            filtered_log = [event for event in filtered_log if event.severity == severity]

        return filtered_log[-limit:]

    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics"""
        total_events = len(self.audit_log)
        auth_failures = len([e for e in self.audit_log if e.event_type == "AUTH_FAILURE"])
        permission_denials = len([e for e in self.audit_log if e.event_type == "PERMISSION_DENIED"])
        rate_limit_violations = len([e for e in self.audit_log if e.event_type == "RATE_LIMIT_EXCEEDED"])

        return {
            "total_audit_events": total_events,
            "auth_failures": auth_failures,
            "permission_denials": permission_denials,
            "rate_limit_violations": rate_limit_violations,
            "active_agents": len(self.credentials_store),
            "revoked_tokens": len(self.revoked_tokens),
            "encryption_enabled": self.encryption_enabled,
            "message_signing_enabled": self.message_signing_enabled,
            "rate_limiting_enabled": self.rate_limiting_enabled,
            "security_score": max(0, 100 - (auth_failures * 2) - (permission_denials * 1) - (rate_limit_violations * 3)),
            "timestamp": datetime.now().isoformat()
        }

    def _match_subject_pattern(self, subject: str, pattern: str) -> bool:
        """Match NATS subject against permission pattern"""
        # Convert NATS wildcard patterns to regex
        # '>' matches one or more tokens
        # '*' matches exactly one token
        regex_pattern = pattern.replace('.', r'\.').replace('*', r'[^\.]+').replace('>', r'.+')
        regex_pattern = f"^{regex_pattern}$"

        import re
        return bool(re.match(regex_pattern, subject))

    def _log_security_event(self, event_type: str, agent_id: str, subject: str,
                          details: Dict[str, Any], severity: str = "INFO",
                          ip_address: Optional[str] = None):
        """Log security event to audit trail"""
        event = SecurityEvent(
            event_type=event_type,
            agent_id=agent_id,
            subject=subject,
            timestamp=datetime.now(),
            details=details,
            ip_address=ip_address,
            severity=severity
        )

        self.audit_log.append(event)

        # Maintain audit log size limit
        if len(self.audit_log) > self.max_audit_entries:
            self.audit_log = self.audit_log[-self.max_audit_entries:]

        # Log critical events immediately
        if severity == "CRITICAL":
            logger.critical("NATS security event",
                           event_type=event_type, agent_id=agent_id,
                           subject=subject, details=details)
        elif severity == "WARNING":
            logger.warning("NATS security event",
                          event_type=event_type, agent_id=agent_id,
                          subject=subject, details=details)


class NATSAgentCoordinator:
    """Ultra-low latency NATS-based agent coordination system"""

    def __init__(self, nats_url: str = None, master_key: Optional[str] = None):
    """Initialize NATS coordinator with enhanced performance optimizations"""
    from app.core.config import settings
    
    self.nats_url = nats_url or os.getenv("NATS_URL", "nats://localhost:4222")
    self.nc: Optional[NATS] = None
    self.js = None  # JetStream context
    self.settings = settings  # Store settings reference

    # === SECURITY FRAMEWORK ===
    self.security_manager = NATSSecurityManager(master_key)
    self.security_enabled = os.getenv("NATS_SECURITY_ENABLED", "true").lower() == "true"

    # Initialize default agent credentials for the trading system
    self._initialize_default_agents()

    # Connection management
    self.connected = False
    self.reconnect_attempts = 0
    self.max_reconnect_attempts = settings.NATS_MAX_RECONNECT_ATTEMPTS

    # Subscription tracking
    self.subscriptions: Dict[str, Any] = {}
    self.active_subscribers: Set[str] = set()

    # Data caches (same as WebSocket version)
    self.agent_status_cache: Dict[str, AgentStatusUpdate] = {}
    self.decision_history: Dict[str, List[AgentDecisionUpdate]] = {}

    # Coordination channels
    self.coordination_channels = {
        "all_agents",
        "market_analysts",
        "risk_advisors",
        "strategy_optimizers",
        "performance_coaches"
    }

    # Performance metrics
    self.message_count = 0
    self.last_message_time = None

    # === ENHANCED PERFORMANCE OPTIMIZATIONS ===
    
    # High-performance connection pooling 
    self.connection_pool_size = settings.NATS_CONNECTION_POOL_SIZE
    self.connection_pool: List[NATS] = []
    self.pool_lock = asyncio.Lock()
    
    # Intelligent message batching with configurable settings
    self.batch_size = settings.NATS_MESSAGE_BATCH_SIZE
    self.batch_timeout_ms = int(settings.NATS_BATCH_TIMEOUT * 1000)  # Convert to ms
    self.high_throughput_mode = settings.NATS_HIGH_THROUGHPUT_MODE
    self.message_batch = []
    self.batch_lock = asyncio.Lock()
    self.batch_task: Optional[asyncio.Task] = None
    
    # Advanced payload compression with performance tracking
    self.compression_enabled = os.getenv("NATS_COMPRESSION_ENABLED", "true").lower() == "true"
    self.compression_threshold = int(os.getenv("NATS_COMPRESSION_THRESHOLD", "1024"))  # bytes
    
    # Enhanced circuit breaker with configurable parameters
    self.circuit_breaker_enabled = True
    self.circuit_failure_threshold = 5
    self.circuit_recovery_timeout = 30  # seconds
    self.circuit_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    self.circuit_failure_count = 0
    self.circuit_last_failure_time = None
    
    # Production-grade Quality of Service parameters
    self.qos_config = {
        "max_pending": int(os.getenv("NATS_MAX_PENDING", "1000")),
        "max_waiting": int(os.getenv("NATS_MAX_WAITING", "100")),
        "ack_wait": int(os.getenv("NATS_ACK_WAIT", "30")),  # seconds
        "max_deliver": int(os.getenv("NATS_MAX_DELIVER", "3")),
        "idle_heartbeat": int(os.getenv("NATS_IDLE_HEARTBEAT", "5")),  # seconds
    }
    
    # Real-time performance metrics with statistical tracking
    self.latency_samples: List[float] = []
    self.throughput_counter = 0
    self.compression_stats = {"compressed": 0, "uncompressed": 0, "bytes_saved": 0}
    
    # Enhanced connection statistics
    self.connection_stats = {
        "created_connections": 0,
        "failed_connections": 0,
        "reconnections": 0,
        "total_bytes_sent": 0,
        "total_bytes_received": 0,
        "avg_connection_latency_ms": 0.0,
        "connection_errors": 0
    }
    
    # Performance monitoring configuration
    self.performance_monitoring_enabled = True
    self.metrics_collection_interval = 30  # seconds
    self.performance_history: List[Dict] = []
    
    logger.info("NATS Agent Coordinator initialized with enhanced performance optimizations", 
               security_enabled=self.security_enabled,
               high_throughput_mode=self.high_throughput_mode,
               connection_pool_size=self.connection_pool_size,
               batch_size=self.batch_size,
               batch_timeout_ms=self.batch_timeout_ms,
               performance_monitoring=self.performance_monitoring_enabled)

    def _initialize_default_agents(self):
        """Initialize default agent credentials for the trading system"""
        if not self.security_enabled:
            return

        default_agents = [
            ("market_analyst_001", "market_analyst"),
            ("market_analyst_002", "market_analyst"),
            ("risk_advisor_001", "risk_advisor"),
            ("risk_advisor_002", "risk_advisor"),
            ("strategy_optimizer_001", "strategy_optimizer"),
            ("performance_coach_001", "performance_coach"),
            ("system_admin", "admin")
        ]

        for agent_id, agent_type in default_agents:
            credentials = self.security_manager.create_agent_credentials(
                agent_id=agent_id,
                agent_type=agent_type,
                validity_days=90  # Long-term credentials for system agents
            )
            logger.info("Initialized agent credentials",
                       agent_id=agent_id, agent_type=agent_type,
                       expires_at=credentials.expires_at.isoformat())

    async def connect(self) -> bool:
    """Establish NATS connection with enhanced performance configuration"""
    try:
        logger.info("Connecting to NATS with enhanced performance optimizations", 
                   nats_url=self.nats_url,
                   connection_pool_size=self.connection_pool_size,
                   high_throughput_mode=self.high_throughput_mode,
                   batch_size=self.batch_size)
        
        # Create NATS connection with performance-optimized settings
        self.nc = NATS()
        
        # Build high-performance connection options
        connect_options = {
            "servers": [self.nats_url],
            "name": "swaggy_stacks_trading_coordinator",
            "max_reconnect_attempts": self.settings.NATS_MAX_RECONNECT_ATTEMPTS,
            "reconnect_time_wait": self.settings.NATS_RECONNECT_TIME_WAIT,
            "max_outstanding_pings": self.settings.NATS_MAX_OUTSTANDING_PINGS,
            "ping_interval": self.settings.NATS_PING_INTERVAL,
            "max_payload": self.settings.NATS_MAX_PAYLOAD,
            "flush_timeout": self.settings.NATS_FLUSH_TIMEOUT,
            "verbose": False,  # Disable verbose mode for performance
            "pedantic": False,  # Disable pedantic mode for performance
            "allow_reconnect": True,
            "dont_randomize": True,  # Consistent server order for predictable latency
            "error_cb": self._error_callback,
            "disconnected_cb": self._disconnected_callback,
            "reconnected_cb": self._reconnected_callback,
        }
        
        # Add TLS configuration if enabled
        if self.settings.NATS_TLS_ENABLED:
            import ssl
            
            # Create SSL context with performance optimizations
            ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            
            # Configure certificates if provided
            if self.settings.NATS_TLS_CA_FILE:
                ssl_context.load_verify_locations(cafile=self.settings.NATS_TLS_CA_FILE)
            
            if self.settings.NATS_TLS_CERT_FILE and self.settings.NATS_TLS_KEY_FILE:
                ssl_context.load_cert_chain(
                    certfile=self.settings.NATS_TLS_CERT_FILE,
                    keyfile=self.settings.NATS_TLS_KEY_FILE
                )
            
            # Hostname verification based on settings
            if not self.settings.NATS_TLS_VERIFY_HOSTNAME:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                logger.warning("TLS hostname verification disabled - use only in development")
            
            connect_options["tls"] = ssl_context
            logger.info("NATS TLS enabled", 
                       cert_file=self.settings.NATS_TLS_CERT_FILE,
                       verify_hostname=self.settings.NATS_TLS_VERIFY_HOSTNAME)
        
        # Establish connection with performance monitoring
        connection_start = time.time()
        await self.nc.connect(**connect_options)
        connection_latency = (time.time() - connection_start) * 1000
        
        # Update connection stats
        self.connection_stats["created_connections"] += 1
        self.connection_stats["avg_connection_latency_ms"] = connection_latency

        # Initialize JetStream with performance settings
        self.js = self.nc.jetstream()

        # Create JetStream streams for persistence
        await self._create_jetstream_streams()

        # Pre-populate connection pool for immediate availability
        await self._initialize_connection_pool()

        self.connected = True
        self.reconnect_attempts = 0

        # Start performance monitoring task
        if self.performance_monitoring_enabled:
            asyncio.create_task(self._performance_monitoring_task())

        # Start batch processing if in high throughput mode
        if self.high_throughput_mode and not self.batch_task:
            self.batch_task = asyncio.create_task(self._process_batch())

        # Get server info safely
        try:
            server_info = getattr(self.nc, 'server_info', None) or {"server_name": "unknown"}
        except (AttributeError, TypeError):
            server_info = {"server_name": "unknown"}

        logger.info(
            "NATS coordinator connected with enhanced performance",
            server=self.nats_url,
            connection_latency_ms=round(connection_latency, 2),
            tls_enabled=self.settings.NATS_TLS_ENABLED,
            security_enabled=self.security_enabled,
            connection_pool_initialized=len(self.connection_pool),
            high_throughput_mode=self.high_throughput_mode,
            server_info=server_info
        )

        return True

    except Exception as e:
        self.connection_stats["failed_connections"] += 1
        logger.error("Failed to connect to NATS with enhanced configuration", 
                    error=str(e), 
                    tls_enabled=self.settings.NATS_TLS_ENABLED,
                    connection_pool_size=self.connection_pool_size)
        self.connected = False
        return False

async def _initialize_connection_pool(self):
    """Pre-populate connection pool with available connections"""
    try:
        for i in range(min(3, self.connection_pool_size)):  # Pre-create 3 connections
            pool_connection = await self._create_pooled_connection(f"pool_{i}")
            if pool_connection:
                self.connection_pool.append(pool_connection)
        
        logger.info("Connection pool initialized", 
                   pool_size=len(self.connection_pool),
                   max_pool_size=self.connection_pool_size)
                   
    except Exception as e:
        logger.warning("Failed to initialize full connection pool", error=str(e))

async def _create_pooled_connection(self, connection_name: str) -> Optional[NATS]:
    """Create a new optimized connection for the pool"""
    try:
        pool_nc = NATS()
        
        # Optimized connection options for pool connections
        connect_options = {
            "servers": [self.nats_url],
            "name": f"swaggy_stacks_pool_{connection_name}",
            "max_reconnect_attempts": 10,
            "reconnect_time_wait": 1,
            "ping_interval": 30,
            "max_outstanding_pings": 2,
            "flush_timeout": 2.0,
            "verbose": False,
            "pedantic": False,
        }
        
        # Add TLS if enabled
        if self.settings.NATS_TLS_ENABLED:
            import ssl
            ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            if not self.settings.NATS_TLS_VERIFY_HOSTNAME:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            connect_options["tls"] = ssl_context
        
        await pool_nc.connect(**connect_options)
        return pool_nc
        
    except Exception as e:
        logger.error("Failed to create pooled connection", connection_name=connection_name, error=str(e))
        return None

async def _performance_monitoring_task(self):
    """Background task for continuous performance monitoring"""
    while self.connected:
        try:
            await asyncio.sleep(self.metrics_collection_interval)
            
            # Collect current performance metrics
            metrics = await self.get_performance_metrics()
            
            # Log performance summary
            latency = metrics["latency_stats"]["avg_ms"]
            throughput = metrics["throughput"]["messages_per_second"]
            pool_utilization = (metrics["connection_pool"]["size"] / max(1, metrics["connection_pool"]["max_size"])) * 100
            
            logger.info("Performance monitoring update",
                       avg_latency_ms=round(latency, 2),
                       messages_per_second=round(throughput, 2),
                       pool_utilization_percent=round(pool_utilization, 1),
                       circuit_breaker_state=metrics["circuit_breaker"]["state"],
                       total_messages=self.message_count)
            
            # Check for performance degradation
            if latency > 10:  # More than 10ms is concerning
                logger.warning("High latency detected", 
                             avg_latency_ms=round(latency, 2),
                             recommendation="Consider performance optimization")
            
            if metrics["circuit_breaker"]["state"] != "CLOSED":
                logger.warning("Circuit breaker not closed", 
                             state=metrics["circuit_breaker"]["state"],
                             failure_count=metrics["circuit_breaker"]["failure_count"])
                             
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Performance monitoring error", error=str(e))

    async def disconnect(self):
        """Gracefully disconnect from NATS"""
        if self.nc and self.connected:
            try:
                # Unsubscribe from all subjects
                for subscription in self.subscriptions.values():
                    await subscription.unsubscribe()

                await self.nc.close()
                self.connected = False
                logger.info("NATS coordinator disconnected gracefully")

            except Exception as e:
                logger.warning("Error during NATS disconnect", error=str(e))

    async def _create_jetstream_streams(self):
        """Create JetStream streams for persistent storage"""
        streams = [
            {
                "name": "AGENT_DECISIONS",
                "subjects": ["agents.decisions.>"],
                "description": "Agent trading decisions with persistence"
            },
            {
                "name": "CONSENSUS_REQUESTS",
                "subjects": ["agents.consensus.>"],
                "description": "Consensus voting requests and responses"
            },
            {
                "name": "TRADE_EXECUTION",
                "subjects": ["agents.execution.>"],
                "description": "Trade execution status and results"
            }
        ]

        for stream_config in streams:
            try:
                await self.js.add_stream(StreamConfig(
                    name=stream_config["name"],
                    subjects=stream_config["subjects"],
                    description=stream_config["description"],
                    max_age=24 * 60 * 60,  # 24 hour retention
                    max_msgs=1000000,      # 1M message limit
                    storage="file"         # Persistent storage
                ))
                logger.info("Created JetStream stream", name=stream_config["name"])

            except Exception as e:
                if "stream name already in use" not in str(e):
                    logger.warning("Failed to create stream", name=stream_config["name"], error=str(e))

    async def subscribe_to_agent_decisions(self, client_id: str, agent_ids: List[str]):
        """Subscribe to specific agent decision streams"""
        for agent_id in agent_ids:
            subject = f"agents.decisions.{agent_id}.>"
            subscription_key = f"{client_id}_decisions_{agent_id}"

            if subscription_key not in self.subscriptions:
                try:
                    sub = await self.nc.subscribe(
                        subject,
                        cb=self._create_decision_handler(client_id)
                    )
                    self.subscriptions[subscription_key] = sub
                    self.active_subscribers.add(client_id)

                    logger.info("Subscribed to agent decisions",
                              client_id=client_id, agent_id=agent_id, subject=subject)

                except Exception as e:
                    logger.error("Failed to subscribe to agent decisions",
                               client_id=client_id, agent_id=agent_id, error=str(e))

    async def subscribe_to_coordination_channel(self, client_id: str, channel: str):
        """Subscribe to inter-agent coordination channel"""
        if channel in self.coordination_channels:
            subject = f"agents.coordination.{channel}"
            subscription_key = f"{client_id}_coordination_{channel}"

            if subscription_key not in self.subscriptions:
                try:
                    sub = await self.nc.subscribe(
                        subject,
                        cb=self._create_coordination_handler(client_id)
                    )
                    self.subscriptions[subscription_key] = sub
                    self.active_subscribers.add(client_id)

                    logger.info("Subscribed to coordination channel",
                              client_id=client_id, channel=channel, subject=subject)

                except Exception as e:
                    logger.error("Failed to subscribe to coordination channel",
                               client_id=client_id, channel=channel, error=str(e))

    async def broadcast_agent_decision(self, decision: AgentDecisionUpdate, jwt_token: Optional[str] = None):
        """Broadcast agent decision with JetStream persistence, performance optimizations, and security"""
        if not self.connected:
            logger.warning("Cannot broadcast decision - NATS not connected")
            return

        # Security validation
        if self.security_enabled and not await self._validate_agent_request(decision.agent_id, jwt_token):
            return

        subject = f"agents.decisions.{decision.agent_id}.{decision.symbol}"

        # Check permissions
        if self.security_enabled and not self.security_manager.check_permissions(decision.agent_id, subject):
            return

        # Check rate limits
        agent_type = self._get_agent_type_from_id(decision.agent_id)
        if self.security_enabled and not self.security_manager.check_rate_limit(decision.agent_id, agent_type):
            return

        message = {
            "type": "agent_decision",
            "data": asdict(decision),
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Secure message processing
            data = json.dumps(message).encode()

            if self.security_enabled:
                # Encrypt message if encryption is enabled
                data = await self.security_manager.encrypt_message(data, decision.agent_id)

                # Sign message for integrity
                signature = self.security_manager.sign_message(data, decision.agent_id)

                # Add security metadata
                message["security"] = {
                    "encrypted": self.security_manager.encryption_enabled,
                    "signed": bool(signature),
                    "signature": signature,
                    "agent_id": decision.agent_id
                }

                # Re-encode with security metadata
                data = json.dumps(message).encode()
                if self.security_manager.encryption_enabled:
                    data = await self.security_manager.encrypt_message(data, decision.agent_id)

            await self._batch_publish(subject, data)

            # Cache decision (same as WebSocket version)
            if decision.symbol not in self.decision_history:
                self.decision_history[decision.symbol] = []
            self.decision_history[decision.symbol].append(decision)

            # Keep only last 100 decisions per symbol
            if len(self.decision_history[decision.symbol]) > 100:
                self.decision_history[decision.symbol] = self.decision_history[decision.symbol][-100:]

            self._update_metrics()
            logger.debug("Queued secure agent decision for batch processing",
                        agent_id=decision.agent_id, symbol=decision.symbol, subject=subject,
                        encrypted=self.security_manager.encryption_enabled if self.security_enabled else False)

        except Exception as e:
            logger.error("Failed to queue agent decision",
                        agent_id=decision.agent_id, symbol=decision.symbol, error=str(e))

    async def broadcast_agent_status(self, status: AgentStatusUpdate):
        """Broadcast agent status updates"""
        if not self.connected:
            logger.warning("Cannot broadcast status - NATS not connected")
            return

        subject = f"agents.status.{status.agent_type}.{status.agent_id}"
        message = {
            "type": "agent_status",
            "data": asdict(status),
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Regular publish for status (non-persistent)
            await self.nc.publish(subject, json.dumps(message).encode())

            # Cache status
            self.agent_status_cache[status.agent_id] = status

            self._update_metrics()
            logger.debug("Broadcasted agent status",
                        agent_id=status.agent_id, agent_type=status.agent_type)

        except Exception as e:
            logger.error("Failed to broadcast agent status",
                        agent_id=status.agent_id, error=str(e))

    async def broadcast_coordination_message(self, coordination: AgentCoordinationMessage):
        """Broadcast inter-agent coordination messages"""
        if not self.connected:
            logger.warning("Cannot broadcast coordination - NATS not connected")
            return

        # Determine subject based on recipient
        if coordination.recipient_agent_id:
            subject = f"agents.coordination.direct.{coordination.recipient_agent_id}"
        else:
            subject = "agents.coordination.all_agents"

        message = {
            "type": "agent_coordination",
            "data": asdict(coordination),
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Use JetStream for coordination messages requiring responses
            if coordination.requires_response:
                await self.js.publish(subject, json.dumps(message).encode())
            else:
                await self.nc.publish(subject, json.dumps(message).encode())

            self._update_metrics()
            logger.debug("Broadcasted coordination message",
                        sender=coordination.sender_agent_id,
                        recipient=coordination.recipient_agent_id,
                        message_type=coordination.message_type)

        except Exception as e:
            logger.error("Failed to broadcast coordination message",
                        sender=coordination.sender_agent_id, error=str(e))

    async def request_agent_consensus(self, symbol: str, decision_context: Dict[str, Any]) -> str:
        """Request consensus from all active agents"""
        if not self.connected:
            logger.warning("Cannot request consensus - NATS not connected")
            return None

        consensus_id = f"consensus_{symbol}_{datetime.now().timestamp()}"
        subject = f"agents.consensus.request.{symbol}"

        coordination_message = AgentCoordinationMessage(
            sender_agent_id="coordination_manager",
            recipient_agent_id=None,  # broadcast
            message_type="consensus_request",
            payload={
                "consensus_id": consensus_id,
                "symbol": symbol,
                "context": decision_context
            },
            timestamp=datetime.now().isoformat(),
            requires_response=True
        )

        message = {
            "type": "consensus_request",
            "data": asdict(coordination_message),
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Publish consensus request with JetStream
            await self.js.publish(subject, json.dumps(message).encode())

            logger.info("Consensus request sent",
                       consensus_id=consensus_id, symbol=symbol, subject=subject)
            return consensus_id

        except Exception as e:
            logger.error("Failed to send consensus request",
                        consensus_id=consensus_id, symbol=symbol, error=str(e))
            return None

    def get_agent_decision_history(self, symbol: str, limit: int = 50) -> List[AgentDecisionUpdate]:
        """Get recent decision history for a symbol"""
        if symbol not in self.decision_history:
            return []
        return self.decision_history[symbol][-limit:]

    def get_active_agents(self) -> List[AgentStatusUpdate]:
        """Get list of currently active agents"""
        return [
            status for status in self.agent_status_cache.values()
            if status.status == "active"
        ]

    def _create_decision_handler(self, client_id: str):
        """Create message handler for agent decisions"""
        async def handler(msg):
            try:
                data = json.loads(msg.data.decode())
                # In production, this would forward to WebSocket clients or other subscribers
                logger.debug("Received agent decision",
                           client_id=client_id, subject=msg.subject, data_type=data.get("type"))

            except Exception as e:
                logger.error("Failed to handle agent decision",
                           client_id=client_id, error=str(e))
        return handler

    def _create_coordination_handler(self, client_id: str):
        """Create message handler for coordination messages"""
        async def handler(msg):
            try:
                data = json.loads(msg.data.decode())
                # In production, this would forward to WebSocket clients or other subscribers
                logger.debug("Received coordination message",
                           client_id=client_id, subject=msg.subject, data_type=data.get("type"))

            except Exception as e:
                logger.error("Failed to handle coordination message",
                           client_id=client_id, error=str(e))
        return handler

    def _update_metrics(self):
        """Update performance metrics"""
        self.message_count += 1
        self.last_message_time = datetime.now()

    async def _error_callback(self, error):
        """Handle NATS connection errors"""
        logger.error("NATS connection error", error=str(error))

    async def _disconnected_callback(self):
        """Handle NATS disconnection"""
        self.connected = False
        logger.warning("NATS coordinator disconnected")

    async def _reconnected_callback(self):
        """Handle NATS reconnection"""
        self.connected = True
        self.reconnect_attempts = 0
        logger.info("NATS coordinator reconnected successfully")

    # === PERFORMANCE OPTIMIZATION METHODS ===
    
    async def _compress_payload(self, data: bytes) -> bytes:
        """Compress payload if it exceeds threshold"""
        if not self.compression_enabled or len(data) < self.compression_threshold:
            return data
            
        try:
            compressed = gzip.compress(data, compresslevel=1)  # Fast compression
            bytes_saved = len(data) - len(compressed)
            
            # Update stats
            self.compression_stats["compressed"] += 1
            self.compression_stats["bytes_saved"] += bytes_saved
            
            logger.debug("Compressed payload", 
                        original_size=len(data), 
                        compressed_size=len(compressed),
                        compression_ratio=len(compressed)/len(data))
            return compressed
            
        except Exception as e:
            logger.warning("Failed to compress payload", error=str(e))
            self.compression_stats["uncompressed"] += 1
            return data
    
    async def _decompress_payload(self, data: bytes) -> bytes:
        """Decompress payload if compressed"""
        if not self.compression_enabled:
            return data
            
        try:
            # Check if data is gzip compressed (magic bytes)
            if data.startswith(b'\x1f\x8b'):
                return gzip.decompress(data)
            return data
        except Exception as e:
            logger.warning("Failed to decompress payload", error=str(e))
            return data
    
    async def _get_pooled_connection(self) -> Optional[NATS]:
        """Get connection from pool or create new one"""
        async with self.pool_lock:
            if self.connection_pool:
                return self.connection_pool.pop()
            
            # Create new connection if pool is empty
            if len(self.connection_pool) < self.connection_pool_size:
                try:
                    new_nc = NATS()
                    await new_nc.connect(
                        servers=[self.nats_url],
                        name=f"trading_agent_pool_{len(self.connection_pool)}",
                        max_reconnect_attempts=3,
                        reconnect_time_wait=0.1,
                        ping_interval=30,
                    )
                    return new_nc
                except Exception as e:
                    logger.error("Failed to create pooled connection", error=str(e))
                    
            return self.nc  # Fallback to main connection
    
    async def _return_to_pool(self, connection: NATS):
        """Return connection to pool"""
        async with self.pool_lock:
            if len(self.connection_pool) < self.connection_pool_size:
                self.connection_pool.append(connection)
            else:
                try:
                    await connection.close()
                except Exception:
                    pass  # Ignore close errors
    
    async def _batch_publish(self, subject: str, data: bytes):
        """Add message to batch for efficient publishing"""
        async with self.batch_lock:
            self.message_batch.append((subject, data, time.time()))
            
            # Start batch processing task if not running
            if self.batch_task is None or self.batch_task.done():
                self.batch_task = asyncio.create_task(self._process_batch())
    
    async def _process_batch(self):
        """Process batched messages efficiently"""
        while True:
            try:
                await asyncio.sleep(self.batch_timeout_ms / 1000.0)
                
                async with self.batch_lock:
                    if not self.message_batch:
                        continue
                        
                    # Get batch to process
                    batch = self.message_batch.copy()
                    self.message_batch.clear()
                
                # Process batch efficiently
                if batch:
                    await self._send_batch(batch)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error processing message batch", error=str(e))
    
    async def _send_batch(self, batch: List[tuple]):
        """Send batch of messages efficiently"""
        try:
            # Group by subject for efficiency
            subject_groups = {}
            for subject, data, timestamp in batch:
                if subject not in subject_groups:
                    subject_groups[subject] = []
                subject_groups[subject].append((data, timestamp))
            
            # Send grouped messages
            for subject, messages in subject_groups.items():
                for data, timestamp in messages:
                    # Measure latency
                    latency = (time.time() - timestamp) * 1000  # ms
                    self.latency_samples.append(latency)
                    
                    # Keep only recent samples (last 1000)
                    if len(self.latency_samples) > 1000:
                        self.latency_samples = self.latency_samples[-1000:]
                    
                    # Compress if needed
                    compressed_data = await self._compress_payload(data)
                    
                    # Send with circuit breaker protection
                    await self._publish_with_circuit_breaker(subject, compressed_data)
                    
        except Exception as e:
            logger.error("Failed to send message batch", error=str(e))
    
    async def _publish_with_circuit_breaker(self, subject: str, data: bytes):
        """Publish message with circuit breaker protection"""
        if not self._is_circuit_closed():
            logger.warning("Circuit breaker OPEN - dropping message", subject=subject)
            return
            
        try:
            start_time = time.time()
            
            if self.js and subject.startswith("agents.decisions."):
                # Use JetStream for persistent messages
                await self.js.publish(subject, data)
            else:
                # Use regular publish for non-persistent messages
                await self.nc.publish(subject, data)
            
            # Record success
            self._circuit_success()
            
            # Update performance metrics
            latency_ms = (time.time() - start_time) * 1000
            self.latency_samples.append(latency_ms)
            self.throughput_counter += 1
            
        except Exception as e:
            self._circuit_failure()
            logger.error("Failed to publish message", subject=subject, error=str(e))
            raise
    
    def _is_circuit_closed(self) -> bool:
        """Check if circuit breaker is closed (allowing requests)"""
        if not self.circuit_breaker_enabled:
            return True
            
        if self.circuit_state == "CLOSED":
            return True
        elif self.circuit_state == "OPEN":
            # Check if recovery timeout has passed
            if (self.circuit_last_failure_time and 
                time.time() - self.circuit_last_failure_time > self.circuit_recovery_timeout):
                self.circuit_state = "HALF_OPEN"
                self.circuit_failure_count = 0
                logger.info("Circuit breaker moving to HALF_OPEN state")
                return True
            return False
        elif self.circuit_state == "HALF_OPEN":
            return True
            
        return False
    
    def _circuit_success(self):
        """Record successful operation"""
        if self.circuit_state == "HALF_OPEN":
            self.circuit_state = "CLOSED"
            self.circuit_failure_count = 0
            logger.info("Circuit breaker CLOSED - service recovered")
    
    def _circuit_failure(self):
        """Record failed operation"""
        self.circuit_failure_count += 1
        self.circuit_last_failure_time = time.time()
        
        if self.circuit_failure_count >= self.circuit_failure_threshold:
            self.circuit_state = "OPEN"
            logger.warning("Circuit breaker OPEN - too many failures", 
                          failure_count=self.circuit_failure_count)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            "latency_stats": {
                "samples": len(self.latency_samples),
                "avg_ms": sum(self.latency_samples) / len(self.latency_samples) if self.latency_samples else 0,
                "min_ms": min(self.latency_samples) if self.latency_samples else 0,
                "max_ms": max(self.latency_samples) if self.latency_samples else 0,
                "p95_ms": sorted(self.latency_samples)[int(0.95 * len(self.latency_samples))] if self.latency_samples else 0,
            },
            "throughput": {
                "messages_per_second": self.throughput_counter / max(1, (time.time() - (self.last_message_time.timestamp() if self.last_message_time else time.time()))),
                "total_messages": self.throughput_counter,
            },
            "compression": self.compression_stats.copy(),
            "circuit_breaker": {
                "state": self.circuit_state,
                "failure_count": self.circuit_failure_count,
                "enabled": self.circuit_breaker_enabled,
            },
            "connection_pool": {
                "size": len(self.connection_pool),
                "max_size": self.connection_pool_size,
            },
            "message_batching": {
                "batch_size": self.batch_size,
                "timeout_ms": self.batch_timeout_ms,
                "queued_messages": len(self.message_batch),
            },
            "timestamp": datetime.now().isoformat()
        }

    async def publish_high_performance(
        self, 
        subject: str, 
        data: Union[str, bytes, dict], 
        agent_id: Optional[str] = None,
        use_batching: bool = True
    ) -> bool:
        """
        High-performance message publishing with enhanced batching and connection pooling
        
        Args:
            subject: NATS subject to publish to
            data: Message data (string, bytes, or dict)
            agent_id: Agent ID for security validation
            use_batching: Whether to use message batching (default: True)
        
        Returns:
            bool: True if message was successfully queued/sent
        """
        if not self.connected:
            raise ConnectionError("NATS not connected")
        
        start_time = time.time()
        
        try:
            # Security checks if enabled
            if self.security_enabled and agent_id:
                if not self.security_manager.check_permissions(agent_id, subject):
                    logger.warning("Permission denied for agent", 
                                 agent_id=agent_id, subject=subject)
                    return False
                    
                agent_type = self._get_agent_type_from_id(agent_id)
                if not self.security_manager.check_rate_limit(agent_id, agent_type):
                    logger.warning("Rate limit exceeded for agent", agent_id=agent_id)
                    return False
            
            # Prepare message data
            if isinstance(data, dict):
                message_bytes = json.dumps(data).encode()
            elif isinstance(data, str):
                message_bytes = data.encode()
            else:
                message_bytes = data
            
            # Apply encryption if security is enabled
            if self.security_enabled and agent_id:
                message_bytes = await self.security_manager.encrypt_message(message_bytes, agent_id)
                
                # Add message signature
                signature = self.security_manager.sign_message(message_bytes, agent_id)
                if signature:
                    message_bytes = json.dumps({
                        'data': message_bytes.decode('utf-8'),
                        'signature': signature,
                        'agent_id': agent_id
                    }).encode()
            
            # Use enhanced message batching for high throughput
            if use_batching and self.high_throughput_mode and len(message_bytes) < 32768:  # 32KB limit for batching
                await self._batch_publish(subject, message_bytes)
            else:
                # Send immediately using optimized connection pool
                conn = await self._get_pooled_connection()
                try:
                    await conn.publish(subject, message_bytes)
                    await conn.flush(timeout=self.settings.NATS_FLUSH_TIMEOUT)
                    
                    # Update connection stats
                    self.connection_stats["total_bytes_sent"] += len(message_bytes)
                finally:
                    await self._return_to_pool(conn)
            
            # Update performance metrics
            latency = time.time() - start_time
            self.latency_samples.append(latency * 1000)  # Convert to ms
            
            # Keep only recent samples for memory efficiency
            if len(self.latency_samples) > 1000:
                self.latency_samples = self.latency_samples[-1000:]
            
            self.throughput_counter += 1
            
            logger.debug("High-performance message published", 
                        subject=subject, 
                        agent_id=agent_id,
                        latency_ms=round(latency * 1000, 2),
                        use_batching=use_batching)
            return True
            
        except Exception as e:
            self.connection_stats["connection_errors"] += 1
            logger.error("Failed to publish high-performance message", 
                        subject=subject, 
                        agent_id=agent_id,
                        error=str(e))
            return False

    async def bulk_publish(
        self,
        messages: List[Dict[str, Any]],
        agent_id: Optional[str] = None
    ) -> Dict[str, int]:
        """
        High-performance bulk publish for multiple messages with connection pooling
        
        Args:
            messages: List of message dictionaries with 'subject' and 'data' keys
            agent_id: Agent ID for security validation
        
        Returns:
            Dict with success/failure counts and performance metrics
        """
        if not self.connected:
            raise ConnectionError("NATS not connected")
        
        results = {'success': 0, 'failed': 0, 'total_latency_ms': 0.0}
        start_time = time.time()
        
        # Group messages by subject for efficient batching
        subject_groups = defaultdict(list)
        for msg in messages:
            subject_groups[msg['subject']].append(msg['data'])
        
        # Process each subject group using connection pool
        for subject, data_list in subject_groups.items():
            try:
                # Security check once per subject if enabled
                if self.security_enabled and agent_id:
                    if not self.security_manager.check_permissions(agent_id, subject):
                        results['failed'] += len(data_list)
                        continue
                
                # Get dedicated connection for this bulk operation
                conn = await self._get_pooled_connection()
                
                try:
                    batch_start = time.time()
                    
                    # Send all messages for this subject
                    for data in data_list:
                        try:
                            if isinstance(data, dict):
                                message_bytes = json.dumps(data).encode()
                            elif isinstance(data, str):
                                message_bytes = data.encode()
                            else:
                                message_bytes = data
                            
                            await conn.publish(subject, message_bytes)
                            results['success'] += 1
                            
                            # Update connection stats
                            self.connection_stats["total_bytes_sent"] += len(message_bytes)
                            
                        except Exception as e:
                            logger.error("Failed to publish bulk message", 
                                       subject=subject, error=str(e))
                            results['failed'] += 1
                    
                    # Flush all messages for this subject
                    await conn.flush(timeout=self.settings.NATS_FLUSH_TIMEOUT)
                    
                    # Track batch performance
                    batch_latency = time.time() - batch_start
                    results['total_latency_ms'] += batch_latency * 1000
                    
                finally:
                    await self._return_to_pool(conn)
                    
            except Exception as e:
                logger.error("Failed to process bulk messages for subject", 
                           subject=subject, error=str(e))
                results['failed'] += len(data_list)
                self.connection_stats["connection_errors"] += 1
        
        # Update performance metrics
        total_latency = time.time() - start_time
        self.throughput_counter += results['success']
        results['total_latency_ms'] = round(total_latency * 1000, 2)
        results['messages_per_second'] = round(len(messages) / total_latency if total_latency > 0 else 0, 2)
        
        logger.info("Bulk publish completed", 
                   total_messages=len(messages),
                   success_count=results['success'],
                   failed_count=results['failed'],
                   messages_per_second=results['messages_per_second'],
                   total_latency_ms=results['total_latency_ms'])
        
        return results

    async def optimize_performance(self) -> Dict[str, Any]:
        """
        Automatically analyze and optimize performance based on current metrics
        
        Returns:
            Dict with current metrics, optimization recommendations, and performance score
        """
        current_metrics = await self.get_performance_metrics()
        optimizations = []
        warnings = []
        
        # Analyze latency performance
        avg_latency = current_metrics["latency_stats"]["avg_ms"]
        if avg_latency > 10:  # > 10ms is concerning for trading
            optimizations.append("High latency detected - consider Chronicle FIX for ultra-low latency")
            warnings.append(f"Average latency {avg_latency:.2f}ms exceeds 10ms threshold")
        elif avg_latency > 5:
            optimizations.append("Consider reducing batch timeout or increasing connection pool size")
        
        # Analyze connection pool utilization
        pool_size = current_metrics["connection_pool"]["size"]
        max_pool = current_metrics["connection_pool"]["max_size"]
        pool_utilization = (pool_size / max_pool) * 100 if max_pool > 0 else 0
        
        if pool_utilization > 90:
            optimizations.append("Consider increasing NATS_CONNECTION_POOL_SIZE")
            warnings.append(f"Connection pool at {pool_utilization:.1f}% utilization")
        elif pool_utilization < 10:
            optimizations.append("Consider decreasing NATS_CONNECTION_POOL_SIZE to save resources")
        
        # Analyze message batching efficiency
        if self.high_throughput_mode:
            queued_messages = current_metrics["message_batching"]["queued_messages"]
            batch_size = current_metrics["message_batching"]["batch_size"]
            
            if queued_messages > batch_size * 2:
                optimizations.append("Consider increasing NATS_MESSAGE_BATCH_SIZE or decreasing NATS_BATCH_TIMEOUT")
                warnings.append(f"Message queue backing up: {queued_messages} messages queued")
        
        # Analyze compression effectiveness
        compression_stats = current_metrics["compression"]
        if compression_stats["compressed"] > 0:
            compression_ratio = compression_stats["bytes_saved"] / (compression_stats["bytes_saved"] + compression_stats["compressed"])
            if compression_ratio < 0.1:  # Less than 10% compression
                optimizations.append("Consider disabling compression for small messages")
        
        # Analyze circuit breaker state
        circuit_state = current_metrics["circuit_breaker"]["state"]
        if circuit_state == "OPEN":
            warnings.append("Circuit breaker is OPEN - service may be experiencing issues")
        elif circuit_state == "HALF_OPEN":
            warnings.append("Circuit breaker is HALF_OPEN - monitoring service recovery")
        
        # Calculate performance score (0-100)
        performance_score = self._calculate_performance_score(current_metrics)
        
        # Store performance snapshot for trending
        performance_snapshot = {
            'timestamp': time.time(),
            'score': performance_score,
            'latency_ms': avg_latency,
            'throughput_msgs_sec': current_metrics["throughput"]["messages_per_second"],
            'pool_utilization': pool_utilization
        }
        self.performance_history.append(performance_snapshot)
        
        # Keep only last 24 hours of history (assuming 30s intervals = 2880 snapshots)
        if len(self.performance_history) > 2880:
            self.performance_history = self.performance_history[-2880:]
        
        result = {
            'current_metrics': current_metrics,
            'optimizations': optimizations,
            'warnings': warnings,
            'performance_score': performance_score,
            'performance_trend': self._calculate_performance_trend(),
            'recommended_actions': self._generate_performance_actions(optimizations, warnings),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("Performance optimization analysis completed", 
                   performance_score=performance_score,
                   optimization_count=len(optimizations),
                   warning_count=len(warnings))
        
        return result

    def _calculate_performance_score(self, metrics: Dict) -> int:
        """Calculate overall performance score (0-100) based on multiple factors"""
        score = 100
        
        # Latency scoring (40% weight)
        latency_ms = metrics["latency_stats"]["avg_ms"]
        if latency_ms > 50:
            score -= 40  # Terrible latency
        elif latency_ms > 10:
            score -= int((latency_ms - 10) * 2)  # -2 points per ms over 10ms
        elif latency_ms > 5:
            score -= int((latency_ms - 5) * 1)  # -1 point per ms over 5ms
        elif latency_ms > 1:
            score -= int((latency_ms - 1) * 0.5)  # -0.5 points per ms over 1ms
        
        # Throughput scoring (25% weight)
        throughput = metrics["throughput"]["messages_per_second"]
        if throughput < 100:
            score -= 25  # Very low throughput
        elif throughput < 500:
            score -= int((500 - throughput) / 20)  # Scale down from 500
        elif throughput > 10000:
            score += 5  # Bonus for high throughput
        
        # Connection pool efficiency (20% weight)
        pool_stats = metrics["connection_pool"]
        if pool_stats["max_size"] > 0:
            utilization = (pool_stats["size"] / pool_stats["max_size"]) * 100
            if utilization > 95 or utilization < 5:
                score -= 20  # Poor utilization
            elif utilization > 85 or utilization < 15:
                score -= 10  # Suboptimal utilization
        
        # Circuit breaker state (15% weight)
        circuit_state = metrics["circuit_breaker"]["state"]
        if circuit_state == "OPEN":
            score -= 30  # Major penalty for circuit breaker open
        elif circuit_state == "HALF_OPEN":
            score -= 10  # Minor penalty for recovery state
        
        return max(0, min(100, score))

    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend over recent history"""
        if len(self.performance_history) < 10:
            return "insufficient_data"
        
        recent_scores = [snapshot['score'] for snapshot in self.performance_history[-10:]]
        older_scores = [snapshot['score'] for snapshot in self.performance_history[-20:-10]] if len(self.performance_history) >= 20 else []
        
        if not older_scores:
            return "stable"
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        older_avg = sum(older_scores) / len(older_scores)
        
        if recent_avg > older_avg + 5:
            return "improving"
        elif recent_avg < older_avg - 5:
            return "degrading"
        else:
            return "stable"

    def _generate_performance_actions(self, optimizations: List[str], warnings: List[str]) -> List[Dict[str, str]]:
        """Generate specific actionable performance recommendations"""
        actions = []
        
        # Convert optimizations to actions
        for optimization in optimizations:
            if "Chronicle FIX" in optimization:
                actions.append({
                    "priority": "high",
                    "action": "consider_protocol_upgrade",
                    "description": "Evaluate Chronicle FIX for sub-microsecond latency",
                    "impact": "dramatic_performance_improvement"
                })
            elif "NATS_CONNECTION_POOL_SIZE" in optimization:
                actions.append({
                    "priority": "medium",
                    "action": "adjust_pool_size",
                    "description": optimization,
                    "impact": "moderate_performance_improvement"
                })
            elif "batch" in optimization.lower():
                actions.append({
                    "priority": "medium",
                    "action": "tune_batching",
                    "description": optimization,
                    "impact": "throughput_optimization"
                })
        
        # Convert warnings to actions
        for warning in warnings:
            if "Circuit breaker is OPEN" in warning:
                actions.append({
                    "priority": "critical",
                    "action": "investigate_failures",
                    "description": "Service is experiencing failures - investigate immediately",
                    "impact": "service_availability"
                })
            elif "latency" in warning.lower():
                actions.append({
                    "priority": "high",
                    "action": "investigate_latency",
                    "description": warning,
                    "impact": "trading_performance"
                })
        
        return actions

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for monitoring"""
        if not self.nc:
            return {"status": "disconnected", "connected": False}

        try:
            # Test roundtrip
            start_time = datetime.now()
            await self.nc.publish("health.ping", b"ping")
            roundtrip_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Get server info safely
            server_info = {}
            try:
                server_info = self.nc.server_info or {}
            except (AttributeError, TypeError):
                server_info = {"server_name": "unknown"}

            return {
                "status": "healthy" if self.connected else "unhealthy",
                "connected": self.connected,
                "server_info": server_info,
                "message_count": self.message_count,
                "last_message": self.last_message_time.isoformat() if self.last_message_time else None,
                "roundtrip_ms": roundtrip_ms,
                "active_subscribers": len(self.active_subscribers),
                "subscriptions": len(self.subscriptions)
            }

        except Exception as e:
            return {
                "status": "error",
                "connected": False,
                "error": str(e)
            }

    # === SECURITY METHODS ===

    async def _validate_agent_request(self, agent_id: str, jwt_token: Optional[str] = None) -> bool:
        """Validate agent request with JWT token"""
        if not self.security_enabled:
            return True

        if not jwt_token:
            logger.warning("Missing JWT token for secured request", agent_id=agent_id)
            return False

        payload = self.security_manager.verify_jwt_token(jwt_token)
        if not payload:
            return False

        # Validate agent_id matches token
        if payload.get("agent_id") != agent_id:
            self.security_manager._log_security_event(
                "AUTH_FAILURE", agent_id, "auth.validate",
                {"reason": "agent_id_mismatch", "token_agent": payload.get("agent_id")},
                severity="WARNING"
            )
            return False

        return True

    def _get_agent_type_from_id(self, agent_id: str) -> str:
        """Extract agent type from agent ID"""
        if "market_analyst" in agent_id:
            return "market_analyst"
        elif "risk_advisor" in agent_id:
            return "risk_advisor"
        elif "strategy_optimizer" in agent_id:
            return "strategy_optimizer"
        elif "performance_coach" in agent_id:
            return "performance_coach"
        elif "admin" in agent_id:
            return "admin"
        return "default"

    def create_agent_jwt_token(self, agent_id: str) -> Optional[str]:
        """Create JWT token for agent authentication"""
        return self.security_manager.generate_jwt_token(agent_id)

    def revoke_agent_token(self, token: str, agent_id: str):
        """Revoke agent JWT token"""
        self.security_manager.revoke_token(token, agent_id)

    def get_agent_credentials(self, agent_id: str) -> Optional[SecurityCredentials]:
        """Get agent credentials"""
        return self.security_manager.credentials_store.get(agent_id)

    def create_agent_credentials(self, agent_id: str, agent_type: str, validity_days: int = 30) -> SecurityCredentials:
        """Create new agent credentials"""
        return self.security_manager.create_agent_credentials(agent_id, agent_type, validity_days)

    def get_security_audit_log(self, limit: int = 100, severity: Optional[str] = None) -> List[SecurityEvent]:
        """Get security audit log"""
        return self.security_manager.get_security_audit_log(limit, severity)

    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics"""
        return self.security_manager.get_security_metrics()

    async def process_encrypted_message(self, encrypted_data: bytes, agent_id: str) -> Optional[Dict[str, Any]]:
        """Process and decrypt received message"""
        if not self.security_enabled:
            try:
                return json.loads(encrypted_data.decode())
            except Exception:
                return None

        try:
            # Decrypt the message
            decrypted_data = await self.security_manager.decrypt_message(encrypted_data, agent_id)
            message = json.loads(decrypted_data.decode())

            # Verify signature if present
            if message.get("security", {}).get("signed") and message.get("security", {}).get("signature"):
                signature = message["security"]["signature"]
                # Re-encode original message data for verification
                original_data = json.dumps({k: v for k, v in message.items() if k != "security"}).encode()

                if not self.security_manager.verify_signature(original_data, signature, agent_id):
                    logger.warning("Message signature verification failed", agent_id=agent_id)
                    return None

            return message

        except Exception as e:
            logger.error("Failed to process encrypted message", agent_id=agent_id, error=str(e))
            return None


# Global coordinator instance
nats_coordinator = NATSAgentCoordinator()


async def get_nats_coordinator() -> NATSAgentCoordinator:
    """Get the global NATS coordinator instance"""
    if not nats_coordinator.connected:
        await nats_coordinator.connect()
    return nats_coordinator


async def initialize_nats_coordinator() -> bool:
    """Initialize the NATS coordinator at startup"""
    return await nats_coordinator.connect()


async def shutdown_nats_coordinator():
    """Shutdown the NATS coordinator gracefully"""
    await nats_coordinator.disconnect()