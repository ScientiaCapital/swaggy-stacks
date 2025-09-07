"""
Automated alerting system for system health monitoring.
"""

import asyncio
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from app.core.logging import get_logger
from .health_checks import SystemHealthStatus, HealthStatus, ComponentType

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Alert delivery channels"""
    LOG = "log"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"


@dataclass
class AlertRule:
    """Definition of an alert rule"""
    name: str
    condition: str
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_minutes: int = 15
    description: str = ""
    threshold: Optional[float] = None
    component_type: Optional[ComponentType] = None


@dataclass
class Alert:
    """Alert instance"""
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    component: Optional[str] = None
    component_type: Optional[ComponentType] = None
    details: Dict = None
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None


class AlertManager:
    """Manages alerting for system health monitoring"""
    
    def __init__(self):
        self.alert_rules = self._setup_default_alert_rules()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_handlers: Dict[AlertChannel, Callable] = {
            AlertChannel.LOG: self._log_alert,
            AlertChannel.WEBHOOK: self._webhook_alert,
        }
        self._last_alert_times: Dict[str, datetime] = {}
    
    def _setup_default_alert_rules(self) -> List[AlertRule]:
        """Setup default alert rules"""
        return [
            # System Health Rules
            AlertRule(
                name="system_critical",
                condition="overall_status == 'critical'",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                cooldown_minutes=5,
                description="System is in critical state"
            ),
            
            AlertRule(
                name="system_degraded",
                condition="overall_status == 'degraded'",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                cooldown_minutes=10,
                description="System performance is degraded"
            ),
            
            # Database Rules
            AlertRule(
                name="database_critical",
                condition="component_status == 'critical'",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                component_type=ComponentType.DATABASE,
                cooldown_minutes=5,
                description="Database is not accessible"
            ),
            
            AlertRule(
                name="database_slow",
                condition="response_time > 1000",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                component_type=ComponentType.DATABASE,
                threshold=1000.0,
                cooldown_minutes=10,
                description="Database response time is high"
            ),
            
            # Redis Rules
            AlertRule(
                name="redis_critical",
                condition="component_status == 'critical'",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                component_type=ComponentType.REDIS,
                cooldown_minutes=5,
                description="Redis is not accessible"
            ),
            
            # MCP Rules
            AlertRule(
                name="mcp_orchestrator_critical",
                condition="component_status == 'critical'",
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                component_type=ComponentType.MCP_ORCHESTRATOR,
                cooldown_minutes=10,
                description="MCP orchestrator is not functional"
            ),
            
            AlertRule(
                name="mcp_server_down",
                condition="component_status == 'critical'",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                component_type=ComponentType.MCP_SERVER,
                cooldown_minutes=15,
                description="MCP server is down or unreachable"
            ),
            
            # Trading System Rules
            AlertRule(
                name="trading_system_critical",
                condition="component_status == 'critical'",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                component_type=ComponentType.TRADING_SYSTEM,
                cooldown_minutes=5,
                description="Trading system is not operational"
            ),
            
            # External API Rules
            AlertRule(
                name="external_api_degraded",
                condition="component_status == 'degraded'",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                component_type=ComponentType.EXTERNAL_API,
                cooldown_minutes=20,
                description="External API connectivity issues"
            ),
        ]
    
    async def process_health_status(self, health_status: SystemHealthStatus):
        """Process health status and trigger alerts"""
        
        # Check overall system status
        await self._check_system_alerts(health_status)
        
        # Check individual components
        for component in health_status.components:
            await self._check_component_alerts(component)
        
        # Resolve alerts that are no longer active
        await self._resolve_inactive_alerts(health_status)
    
    async def _check_system_alerts(self, health_status: SystemHealthStatus):
        """Check system-level alerts"""
        
        system_rules = [rule for rule in self.alert_rules if 'overall_status' in rule.condition]
        
        for rule in system_rules:
            should_alert = False
            
            if rule.condition == "overall_status == 'critical'" and health_status.overall_status == HealthStatus.CRITICAL:
                should_alert = True
            elif rule.condition == "overall_status == 'degraded'" and health_status.overall_status == HealthStatus.DEGRADED:
                should_alert = True
            
            if should_alert:
                alert_key = f"system_{rule.name}"
                if self._should_send_alert(rule.name, rule.cooldown_minutes):
                    alert = Alert(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"System alert: {rule.description}",
                        timestamp=datetime.utcnow(),
                        details={
                            'overall_status': health_status.overall_status.value,
                            'issues': health_status.issues,
                            'healthy_components': health_status.summary.get(HealthStatus.HEALTHY, 0),
                            'degraded_components': health_status.summary.get(HealthStatus.DEGRADED, 0),
                            'critical_components': health_status.summary.get(HealthStatus.CRITICAL, 0)
                        }
                    )
                    
                    await self._send_alert(alert, rule.channels)
                    self.active_alerts[alert_key] = alert
    
    async def _check_component_alerts(self, component):
        """Check component-specific alerts"""
        
        component_rules = [
            rule for rule in self.alert_rules 
            if rule.component_type == component.component_type
        ]
        
        for rule in component_rules:
            should_alert = False
            
            # Check status conditions
            if 'component_status' in rule.condition:
                if rule.condition == "component_status == 'critical'" and component.status == HealthStatus.CRITICAL:
                    should_alert = True
                elif rule.condition == "component_status == 'degraded'" and component.status == HealthStatus.DEGRADED:
                    should_alert = True
            
            # Check threshold conditions
            if 'response_time' in rule.condition and rule.threshold:
                if component.response_time_ms > rule.threshold:
                    should_alert = True
            
            if should_alert:
                alert_key = f"{component.component}_{rule.name}"
                if self._should_send_alert(alert_key, rule.cooldown_minutes):
                    alert = Alert(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"Component alert: {component.component} - {rule.description}",
                        timestamp=datetime.utcnow(),
                        component=component.component,
                        component_type=component.component_type,
                        details={
                            'component_status': component.status.value,
                            'response_time_ms': component.response_time_ms,
                            'component_message': component.message,
                            'error': component.error
                        }
                    )
                    
                    await self._send_alert(alert, rule.channels)
                    self.active_alerts[alert_key] = alert
    
    async def _resolve_inactive_alerts(self, health_status: SystemHealthStatus):
        """Resolve alerts that are no longer active"""
        
        alerts_to_resolve = []
        
        for alert_key, alert in self.active_alerts.items():
            should_resolve = False
            
            # Check if system alerts should be resolved
            if alert_key.startswith('system_'):
                if health_status.overall_status == HealthStatus.HEALTHY:
                    should_resolve = True
            
            # Check if component alerts should be resolved
            else:
                component_name = alert.component
                matching_components = [
                    c for c in health_status.components 
                    if c.component == component_name
                ]
                
                if matching_components:
                    component = matching_components[0]
                    if component.status == HealthStatus.HEALTHY:
                        should_resolve = True
                else:
                    # Component not found, resolve alert
                    should_resolve = True
            
            if should_resolve:
                alerts_to_resolve.append(alert_key)
        
        # Resolve alerts
        for alert_key in alerts_to_resolve:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolved_timestamp = datetime.utcnow()
            
            # Send resolution notification
            await self._send_resolution(alert)
            
            # Move to history and remove from active
            self.alert_history.append(alert)
            del self.active_alerts[alert_key]
    
    def _should_send_alert(self, alert_name: str, cooldown_minutes: int) -> bool:
        """Check if alert should be sent based on cooldown period"""
        
        last_alert_time = self._last_alert_times.get(alert_name)
        if last_alert_time:
            time_since_last = datetime.utcnow() - last_alert_time
            if time_since_last.total_seconds() < (cooldown_minutes * 60):
                return False
        
        self._last_alert_times[alert_name] = datetime.utcnow()
        return True
    
    async def _send_alert(self, alert: Alert, channels: List[AlertChannel]):
        """Send alert through specified channels"""
        
        for channel in channels:
            if channel in self.alert_handlers:
                try:
                    await self.alert_handlers[channel](alert)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel}: {e}")
    
    async def _send_resolution(self, alert: Alert):
        """Send alert resolution notification"""
        
        resolution_message = f"RESOLVED: {alert.message}"
        logger.info(resolution_message)
    
    async def _log_alert(self, alert: Alert):
        """Log alert to application logs"""
        
        log_message = f"ALERT [{alert.severity.upper()}]: {alert.message}"
        if alert.details:
            log_message += f" | Details: {alert.details}"
        
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(log_message)
        elif alert.severity == AlertSeverity.ERROR:
            logger.error(log_message)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    async def _webhook_alert(self, alert: Alert):
        """Send alert via webhook (placeholder implementation)"""
        
        webhook_payload = {
            'alert_name': alert.rule_name,
            'severity': alert.severity.value,
            'message': alert.message,
            'timestamp': alert.timestamp.isoformat(),
            'component': alert.component,
            'details': alert.details
        }
        
        # In production, this would send to actual webhook endpoints
        logger.info(f"Webhook alert: {webhook_payload}")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule"""
        self.alert_rules.append(rule)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history 
            if alert.timestamp > cutoff_time
        ]
    
    def get_alert_stats(self) -> Dict[str, int]:
        """Get alert statistics"""
        return {
            'active_alerts': len(self.active_alerts),
            'total_rules': len(self.alert_rules),
            'alerts_last_24h': len(self.get_alert_history(24)),
            'critical_active': len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]),
            'warning_active': len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.WARNING])
        }