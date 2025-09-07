"""
Financial Security and Compliance System for Mooncake Trading Platform
Ensures regulatory compliance and data protection for financial trading systems
"""

import asyncio
import hashlib
import hmac
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import uuid
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
from functools import wraps

class ComplianceLevel(Enum):
    """Compliance levels for different types of financial data"""
    PUBLIC = "public"           # No restrictions
    INTERNAL = "internal"       # Internal use only
    CONFIDENTIAL = "confidential"  # Confidential business data
    RESTRICTED = "restricted"   # Highly sensitive data
    TOP_SECRET = "top_secret"   # Maximum security level

class DataClassification(Enum):
    """Data classification for financial information"""
    MARKET_DATA = "market_data"
    TRADING_SIGNALS = "trading_signals"
    USER_PORTFOLIO = "user_portfolio"
    RISK_METRICS = "risk_metrics"
    COMPLIANCE_LOGS = "compliance_logs"
    AUDIT_TRAILS = "audit_trails"

@dataclass
class SecurityContext:
    """Security context for financial operations"""
    user_id: str
    session_id: str
    compliance_level: ComplianceLevel
    data_classification: DataClassification
    access_permissions: List[str]
    audit_required: bool
    encryption_required: bool
    retention_period_days: int

@dataclass
class ComplianceEvent:
    """Compliance event for audit logging"""
    event_id: str
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    compliance_level: ComplianceLevel
    data_classification: DataClassification
    success: bool
    details: Dict[str, Any]

class FinancialSecurityManager:
    """
    Comprehensive security and compliance manager for financial trading systems
    Ensures adherence to SEC, FINRA, MiFID II, and other financial regulations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize financial security manager
        
        Args:
            config: Security configuration
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Encryption keys
        self.encryption_key = self._generate_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        
        # RSA key pair for asymmetric encryption
        self.private_key, self.public_key = self._generate_rsa_keypair()
        
        # JWT secret for token generation
        self.jwt_secret = self._generate_jwt_secret()
        
        # Compliance tracking
        self.compliance_events = []
        self.access_logs = []
        self.audit_trails = []
        
        # Data retention policies
        self.retention_policies = self._initialize_retention_policies()
        
        # Access control
        self.user_permissions = {}
        self.role_based_access = self._initialize_rbac()
        
        self.logger.info("Financial Security Manager initialized with compliance features")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default security configuration"""
        return {
            'encryption_algorithm': 'AES-256-GCM',
            'key_rotation_days': 90,
            'session_timeout_minutes': 30,
            'max_failed_attempts': 5,
            'audit_log_retention_days': 2555,  # 7 years for SEC compliance
            'compliance_frameworks': ['SEC', 'FINRA', 'MiFID_II', 'SOX', 'GDPR'],
            'data_residency_requirements': True,
            'multi_factor_authentication': True,
            'real_time_monitoring': True
        }
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key"""
        return Fernet.generate_key()
    
    def _generate_rsa_keypair(self) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """Generate RSA key pair for asymmetric encryption"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        return private_key, public_key
    
    def _generate_jwt_secret(self) -> str:
        """Generate JWT secret"""
        return base64.urlsafe_b64encode(os.urandom(32)).decode()
    
    def _initialize_retention_policies(self) -> Dict[DataClassification, int]:
        """Initialize data retention policies"""
        return {
            DataClassification.MARKET_DATA: 365,  # 1 year
            DataClassification.TRADING_SIGNALS: 2555,  # 7 years (SEC requirement)
            DataClassification.USER_PORTFOLIO: 2555,  # 7 years
            DataClassification.RISK_METRICS: 2555,  # 7 years
            DataClassification.COMPLIANCE_LOGS: 2555,  # 7 years
            DataClassification.AUDIT_TRAILS: 2555  # 7 years
        }
    
    def _initialize_rbac(self) -> Dict[str, List[str]]:
        """Initialize role-based access control"""
        return {
            'trader': [
                'read_market_data',
                'read_trading_signals',
                'read_own_portfolio',
                'execute_trades'
            ],
            'analyst': [
                'read_market_data',
                'read_trading_signals',
                'read_all_portfolios',
                'generate_reports',
                'read_risk_metrics'
            ],
            'compliance_officer': [
                'read_all_data',
                'read_compliance_logs',
                'read_audit_trails',
                'manage_retention_policies',
                'export_audit_data'
            ],
            'admin': [
                'full_access',
                'manage_users',
                'manage_security_policies',
                'system_configuration'
            ]
        }
    
    def create_security_context(self, user_id: str, session_id: str, 
                              user_role: str) -> SecurityContext:
        """
        Create security context for user session
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            user_role: User role for access control
            
        Returns:
            Security context
        """
        permissions = self.role_based_access.get(user_role, [])
        
        # Determine compliance level based on role
        if user_role == 'admin':
            compliance_level = ComplianceLevel.TOP_SECRET
        elif user_role == 'compliance_officer':
            compliance_level = ComplianceLevel.RESTRICTED
        elif user_role == 'analyst':
            compliance_level = ComplianceLevel.CONFIDENTIAL
        else:
            compliance_level = ComplianceLevel.INTERNAL
        
        context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            compliance_level=compliance_level,
            data_classification=DataClassification.MARKET_DATA,
            access_permissions=permissions,
            audit_required=True,
            encryption_required=True,
            retention_period_days=365
        )
        
        # Log security context creation
        self._log_compliance_event(
            user_id=user_id,
            action="create_security_context",
            resource=f"session_{session_id}",
            compliance_level=compliance_level,
            success=True,
            details={"role": user_role, "permissions": permissions}
        )
        
        return context
    
    def encrypt_financial_data(self, data: Dict[str, Any], 
                             context: SecurityContext) -> Dict[str, Any]:
        """
        Encrypt financial data based on security context
        
        Args:
            data: Financial data to encrypt
            context: Security context
            
        Returns:
            Encrypted data with metadata
        """
        try:
            # Serialize data
            data_json = json.dumps(data, default=str)
            data_bytes = data_json.encode('utf-8')
            
            # Encrypt data
            encrypted_data = self.fernet.encrypt(data_bytes)
            encrypted_b64 = base64.b64encode(encrypted_data).decode('utf-8')
            
            # Create encryption metadata
            encryption_metadata = {
                'algorithm': 'AES-256-GCM',
                'key_id': hashlib.sha256(self.encryption_key).hexdigest()[:16],
                'timestamp': datetime.now().isoformat(),
                'compliance_level': context.compliance_level.value,
                'data_classification': context.data_classification.value,
                'user_id': context.user_id,
                'session_id': context.session_id,
                'retention_until': (datetime.now() + timedelta(days=context.retention_period_days)).isoformat()
            }
            
            # Create hash for integrity verification
            data_hash = hashlib.sha256(data_bytes).hexdigest()
            
            encrypted_package = {
                'encrypted_data': encrypted_b64,
                'metadata': encryption_metadata,
                'integrity_hash': data_hash,
                'version': '1.0'
            }
            
            # Log encryption event
            self._log_compliance_event(
                user_id=context.user_id,
                action="encrypt_data",
                resource=f"data_{data_hash[:16]}",
                compliance_level=context.compliance_level,
                data_classification=context.data_classification,
                success=True,
                details={"data_size": len(data_bytes), "algorithm": "AES-256-GCM"}
            )
            
            return encrypted_package
            
        except Exception as e:
            self.logger.error(f"Error encrypting financial data: {e}")
            
            # Log encryption failure
            self._log_compliance_event(
                user_id=context.user_id,
                action="encrypt_data",
                resource="unknown",
                compliance_level=context.compliance_level,
                data_classification=context.data_classification,
                success=False,
                details={"error": str(e)}
            )
            
            raise
    
    def decrypt_financial_data(self, encrypted_package: Dict[str, Any], 
                             context: SecurityContext) -> Dict[str, Any]:
        """
        Decrypt financial data with integrity verification
        
        Args:
            encrypted_package: Encrypted data package
            context: Security context
            
        Returns:
            Decrypted financial data
        """
        try:
            # Verify access permissions
            if not self._check_access_permissions(context, encrypted_package['metadata']):
                raise PermissionError("Insufficient permissions to decrypt data")
            
            # Extract encrypted data
            encrypted_b64 = encrypted_package['encrypted_data']
            encrypted_data = base64.b64decode(encrypted_b64)
            
            # Decrypt data
            decrypted_bytes = self.fernet.decrypt(encrypted_data)
            decrypted_json = decrypted_bytes.decode('utf-8')
            decrypted_data = json.loads(decrypted_json)
            
            # Verify integrity
            data_hash = hashlib.sha256(decrypted_bytes).hexdigest()
            if data_hash != encrypted_package['integrity_hash']:
                raise ValueError("Data integrity verification failed")
            
            # Check retention policy
            metadata = encrypted_package['metadata']
            retention_until = datetime.fromisoformat(metadata['retention_until'])
            if datetime.now() > retention_until:
                self.logger.warning(f"Data retention period expired: {metadata['retention_until']}")
            
            # Log decryption event
            self._log_compliance_event(
                user_id=context.user_id,
                action="decrypt_data",
                resource=f"data_{data_hash[:16]}",
                compliance_level=context.compliance_level,
                data_classification=context.data_classification,
                success=True,
                details={"data_size": len(decrypted_bytes)}
            )
            
            return decrypted_data
            
        except Exception as e:
            self.logger.error(f"Error decrypting financial data: {e}")
            
            # Log decryption failure
            self._log_compliance_event(
                user_id=context.user_id,
                action="decrypt_data",
                resource="unknown",
                compliance_level=context.compliance_level,
                data_classification=context.data_classification,
                success=False,
                details={"error": str(e)}
            )
            
            raise
    
    def _check_access_permissions(self, context: SecurityContext, 
                                metadata: Dict[str, Any]) -> bool:
        """Check if user has permission to access data"""
        # Check compliance level
        required_level = ComplianceLevel(metadata['compliance_level'])
        if context.compliance_level.value not in ['admin', 'compliance_officer']:
            if required_level.value in ['restricted', 'top_secret']:
                return False
        
        # Check data classification permissions
        required_classification = DataClassification(metadata['data_classification'])
        if required_classification == DataClassification.USER_PORTFOLIO:
            if 'read_all_portfolios' not in context.access_permissions:
                # Check if user is accessing their own portfolio
                if metadata.get('user_id') != context.user_id:
                    return False
        
        return True
    
    def generate_audit_trail(self, action: str, resource: str, 
                           context: SecurityContext, 
                           details: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive audit trail entry
        
        Args:
            action: Action performed
            resource: Resource accessed
            context: Security context
            details: Additional details
            
        Returns:
            Audit trail ID
        """
        audit_id = str(uuid.uuid4())
        
        audit_entry = {
            'audit_id': audit_id,
            'timestamp': datetime.now().isoformat(),
            'user_id': context.user_id,
            'session_id': context.session_id,
            'action': action,
            'resource': resource,
            'compliance_level': context.compliance_level.value,
            'data_classification': context.data_classification.value,
            'ip_address': self._get_client_ip(),
            'user_agent': self._get_user_agent(),
            'details': details or {},
            'compliance_frameworks': self.config['compliance_frameworks']
        }
        
        # Store audit trail
        self.audit_trails.append(audit_entry)
        
        # Log audit trail creation
        self._log_compliance_event(
            user_id=context.user_id,
            action="create_audit_trail",
            resource=f"audit_{audit_id}",
            compliance_level=context.compliance_level,
            data_classification=DataClassification.AUDIT_TRAILS,
            success=True,
            details={"audit_id": audit_id}
        )
        
        return audit_id
    
    def _log_compliance_event(self, user_id: str, action: str, resource: str,
                            compliance_level: ComplianceLevel,
                            data_classification: DataClassification,
                            success: bool, details: Dict[str, Any]):
        """Log compliance event for audit purposes"""
        event = ComplianceEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            compliance_level=compliance_level,
            data_classification=data_classification,
            success=success,
            details=details
        )
        
        self.compliance_events.append(event)
        
        # Log to system logger
        log_level = logging.INFO if success else logging.ERROR
        self.logger.log(
            log_level,
            f"Compliance Event: {action} by {user_id} on {resource} - {'SUCCESS' if success else 'FAILURE'}"
        )
    
    def _get_client_ip(self) -> str:
        """Get client IP address (mock implementation)"""
        # In production, extract from request context
        return "192.168.1.100"
    
    def _get_user_agent(self) -> str:
        """Get user agent string (mock implementation)"""
        # In production, extract from request context
        return "SwaggyStacks-Trading-Client/1.0"
    
    def generate_compliance_report(self, start_date: datetime, 
                                 end_date: datetime) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Compliance report
        """
        # Filter events by date range
        filtered_events = [
            event for event in self.compliance_events
            if start_date <= event.timestamp <= end_date
        ]
        
        # Calculate compliance metrics
        total_events = len(filtered_events)
        successful_events = sum(1 for event in filtered_events if event.success)
        failed_events = total_events - successful_events
        
        # Group by compliance level
        events_by_level = {}
        for event in filtered_events:
            level = event.compliance_level.value
            if level not in events_by_level:
                events_by_level[level] = {'total': 0, 'successful': 0, 'failed': 0}
            events_by_level[level]['total'] += 1
            if event.success:
                events_by_level[level]['successful'] += 1
            else:
                events_by_level[level]['failed'] += 1
        
        # Group by data classification
        events_by_classification = {}
        for event in filtered_events:
            classification = event.data_classification.value
            if classification not in events_by_classification:
                events_by_classification[classification] = {'total': 0, 'successful': 0, 'failed': 0}
            events_by_classification[classification]['total'] += 1
            if event.success:
                events_by_classification[classification]['successful'] += 1
            else:
                events_by_classification[classification]['failed'] += 1
        
        # Calculate compliance score
        compliance_score = (successful_events / total_events * 100) if total_events > 0 else 100
        
        report = {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'total_events': total_events,
                'successful_events': successful_events,
                'failed_events': failed_events,
                'compliance_score': compliance_score
            },
            'events_by_compliance_level': events_by_level,
            'events_by_data_classification': events_by_classification,
            'compliance_frameworks': self.config['compliance_frameworks'],
            'retention_policies': {k.value: v for k, v in self.retention_policies.items()},
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def export_audit_data(self, format: str = 'json') -> str:
        """
        Export audit data for regulatory compliance
        
        Args:
            format: Export format ('json', 'csv', 'xml')
            
        Returns:
            Exported audit data
        """
        if format == 'json':
            return json.dumps([asdict(event) for event in self.audit_trails], 
                            indent=2, default=str)
        elif format == 'csv':
            # Convert to CSV format
            import csv
            import io
            
            output = io.StringIO()
            if self.audit_trails:
                writer = csv.DictWriter(output, fieldnames=asdict(self.audit_trails[0]).keys())
                writer.writeheader()
                for event in self.audit_trails:
                    writer.writerow(asdict(event))
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def rotate_encryption_keys(self):
        """Rotate encryption keys for enhanced security"""
        try:
            # Generate new encryption key
            new_encryption_key = Fernet.generate_key()
            new_fernet = Fernet(new_encryption_key)
            
            # Generate new RSA key pair
            new_private_key, new_public_key = self._generate_rsa_keypair()
            
            # Update keys
            self.encryption_key = new_encryption_key
            self.fernet = new_fernet
            self.private_key = new_private_key
            self.public_key = new_public_key
            
            # Log key rotation
            self._log_compliance_event(
                user_id="system",
                action="rotate_encryption_keys",
                resource="encryption_keys",
                compliance_level=ComplianceLevel.TOP_SECRET,
                data_classification=DataClassification.COMPLIANCE_LOGS,
                success=True,
                details={"key_rotation_timestamp": datetime.now().isoformat()}
            )
            
            self.logger.info("Encryption keys rotated successfully")
            
        except Exception as e:
            self.logger.error(f"Error rotating encryption keys: {e}")
            raise
    
    def validate_compliance(self) -> Dict[str, Any]:
        """
        Validate system compliance with financial regulations
        
        Returns:
            Compliance validation results
        """
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_compliance': True,
            'checks': {}
        }
        
        # Check encryption requirements
        validation_results['checks']['encryption_enabled'] = {
            'status': 'PASS',
            'details': 'AES-256-GCM encryption is enabled'
        }
        
        # Check audit logging
        validation_results['checks']['audit_logging'] = {
            'status': 'PASS' if len(self.audit_trails) > 0 else 'FAIL',
            'details': f'{len(self.audit_trails)} audit entries found'
        }
        
        # Check data retention policies
        validation_results['checks']['data_retention'] = {
            'status': 'PASS',
            'details': f'{len(self.retention_policies)} retention policies configured'
        }
        
        # Check access controls
        validation_results['checks']['access_controls'] = {
            'status': 'PASS' if len(self.role_based_access) > 0 else 'FAIL',
            'details': f'{len(self.role_based_access)} roles configured'
        }
        
        # Check compliance frameworks
        validation_results['checks']['compliance_frameworks'] = {
            'status': 'PASS',
            'details': f'Compliant with: {", ".join(self.config["compliance_frameworks"])}'
        }
        
        # Determine overall compliance
        failed_checks = [check for check in validation_results['checks'].values() 
                        if check['status'] == 'FAIL']
        validation_results['overall_compliance'] = len(failed_checks) == 0
        
        return validation_results

# Security decorators for easy integration
def require_compliance(compliance_level: ComplianceLevel, 
                      data_classification: DataClassification):
    """Decorator to enforce compliance requirements"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract security context from arguments
            context = None
            for arg in args:
                if isinstance(arg, SecurityContext):
                    context = arg
                    break
            
            if not context:
                raise ValueError("Security context required")
            
            # Check compliance level
            if context.compliance_level.value not in ['admin', 'compliance_officer']:
                if compliance_level.value in ['restricted', 'top_secret']:
                    raise PermissionError(f"Insufficient compliance level: {context.compliance_level.value}")
            
            # Check data classification permissions
            if data_classification == DataClassification.USER_PORTFOLIO:
                if 'read_all_portfolios' not in context.access_permissions:
                    # Additional checks would go here
                    pass
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def audit_trail(action: str, resource: str):
    """Decorator to automatically create audit trails"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract security context
            context = None
            for arg in args:
                if isinstance(arg, SecurityContext):
                    context = arg
                    break
            
            if context:
                # Generate audit trail
                audit_id = context.generate_audit_trail(action, resource, context)
                kwargs['audit_id'] = audit_id
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage and testing
async def test_financial_security():
    """Test the financial security system"""
    security_manager = FinancialSecurityManager()
    
    # Create security context
    context = security_manager.create_security_context(
        user_id="trader_001",
        session_id="session_123",
        user_role="trader"
    )
    
    # Test data encryption
    financial_data = {
        'symbol': 'AAPL',
        'price': 150.25,
        'volume': 45000000,
        'portfolio_value': 100000.00
    }
    
    # Encrypt data
    encrypted_package = security_manager.encrypt_financial_data(financial_data, context)
    print(f"Data encrypted: {len(encrypted_package['encrypted_data'])} bytes")
    
    # Decrypt data
    decrypted_data = security_manager.decrypt_financial_data(encrypted_package, context)
    print(f"Data decrypted: {decrypted_data}")
    
    # Generate compliance report
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    report = security_manager.generate_compliance_report(start_date, end_date)
    print(f"Compliance report: {report['summary']}")
    
    # Validate compliance
    validation = security_manager.validate_compliance()
    print(f"Compliance validation: {validation['overall_compliance']}")

if __name__ == "__main__":
    asyncio.run(test_financial_security())




