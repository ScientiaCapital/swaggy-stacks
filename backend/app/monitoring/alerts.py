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

    def configure_prometheus_alerts(self) -> List[AlertRule]:
        """Configure Prometheus metric-based alert rules with intelligent thresholds"""
        prometheus_rules = [
            # System Health Metrics Alerts
            AlertRule(
                name="system_health_degraded",
                condition="trading_system_health_status < 2",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                cooldown_minutes=5,
                description="System health status indicates degraded performance",
                threshold=2.0
            ),
            
            AlertRule(
                name="system_uptime_low",
                condition="trading_system_uptime_seconds < 300",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                cooldown_minutes=2,
                description="System uptime is critically low (less than 5 minutes)",
                threshold=300.0
            ),
            
            # MCP Agent Coordination Alerts
            AlertRule(
                name="mcp_agent_coordination_failure",
                condition="mcp_agent_coordination_success_rate < 0.8",
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                cooldown_minutes=10,
                description="MCP agent coordination success rate below 80%",
                threshold=0.8
            ),
            
            AlertRule(
                name="mcp_agent_high_response_time",
                condition="mcp_agent_response_time_seconds > 5.0",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                cooldown_minutes=15,
                description="MCP agent response time exceeds 5 seconds",
                threshold=5.0
            ),
            
            AlertRule(
                name="mcp_agent_queue_depth_high",
                condition="mcp_agent_queue_depth > 50",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                cooldown_minutes=10,
                description="MCP agent queue depth is critically high",
                threshold=50.0
            ),
            
            AlertRule(
                name="mcp_server_unavailable",
                condition="mcp_server_status == 0",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                cooldown_minutes=5,
                description="MCP server is completely unavailable",
                threshold=0.0
            ),
            
            # Trading System Alerts
            AlertRule(
                name="trading_portfolio_value_drop",
                condition="trading_portfolio_value_usd_change_rate < -0.05",
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                cooldown_minutes=5,
                description="Portfolio value dropped more than 5% in short period",
                threshold=-0.05
            ),
            
            AlertRule(
                name="trading_orders_failure_rate_high",
                condition="trading_orders_failure_rate > 0.1",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                cooldown_minutes=10,
                description="Trading order failure rate exceeds 10%",
                threshold=0.1
            ),
            
            # Database Performance Alerts
            AlertRule(
                name="db_connection_pool_exhausted",
                condition="db_connection_pool_size < 2",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                cooldown_minutes=5,
                description="Database connection pool nearly exhausted",
                threshold=2.0
            ),
            
            AlertRule(
                name="db_query_duration_high",
                condition="db_query_duration_seconds > 2.0",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                cooldown_minutes=10,
                description="Database query duration exceeds 2 seconds",
                threshold=2.0
            ),
            
            # Redis Performance Alerts
            AlertRule(
                name="redis_response_time_high",
                condition="redis_response_time_seconds > 0.1",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                cooldown_minutes=10,
                description="Redis response time exceeds 100ms",
                threshold=0.1
            ),
            
            AlertRule(
                name="redis_operations_failure_rate_high",
                condition="redis_operations_failure_rate > 0.05",
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                cooldown_minutes=5,
                description="Redis operation failure rate exceeds 5%",
                threshold=0.05
            ),
            
            # HTTP API Performance Alerts
            AlertRule(
                name="http_request_duration_high",
                condition="http_request_duration_seconds > 5.0",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                cooldown_minutes=10,
                description="HTTP request duration exceeds 5 seconds",
                threshold=5.0
            ),
            
            AlertRule(
                name="http_error_rate_high",
                condition="http_error_rate > 0.1",
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                cooldown_minutes=5,
                description="HTTP error rate exceeds 10%",
                threshold=0.1
            ),
            
            # AI & Market Research Alerts
            AlertRule(
                name="ai_processing_duration_high",
                condition="ai_processing_duration_seconds > 30.0",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                cooldown_minutes=15,
                description="AI processing duration exceeds 30 seconds",
                threshold=30.0
            ),
            
            AlertRule(
                name="market_sentiment_extreme",
                condition="abs(market_sentiment_score) > 0.9",
                severity=AlertSeverity.INFO,
                channels=[AlertChannel.LOG],
                cooldown_minutes=60,
                description="Market sentiment score indicates extreme conditions",
                threshold=0.9
            ),
        ]
        
        # Add all Prometheus-based rules to the alert system
        for rule in prometheus_rules:
            self.add_alert_rule(rule)
            
        logger.info(f"Configured {len(prometheus_rules)} Prometheus-based alert rules")
        return prometheus_rules

    async def evaluate_prometheus_alerts(self, metrics_data: Dict) -> List[Alert]:
        """Evaluate Prometheus metrics against alert rules and trigger alerts"""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            # Skip non-Prometheus rules (legacy health-based rules)
            if not self._is_prometheus_rule(rule):
                continue
                
            try:
                # Evaluate rule condition against current metrics
                if self._evaluate_metric_condition(rule, metrics_data):
                    alert = Alert(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"{rule.description}",
                        timestamp=datetime.utcnow(),
                        component=self._extract_component_from_rule(rule),
                        component_type=rule.component_type,
                        details={
                            'threshold': rule.threshold,
                            'condition': rule.condition,
                            'metric_value': self._get_metric_value(rule.condition, metrics_data)
                        }
                    )
                    
                    # Check if we should send this alert (respecting cooldown)
                    if self._should_send_alert(rule.name, rule.cooldown_minutes):
                        triggered_alerts.append(alert)
                        self.active_alerts[rule.name] = alert
                        await self._send_alert(alert, rule.channels)
                        
            except Exception as e:
                logger.error(f"Error evaluating Prometheus alert rule {rule.name}: {e}")
                
        return triggered_alerts
    
    def _is_prometheus_rule(self, rule: AlertRule) -> bool:
        """Check if rule is a Prometheus-based metric rule"""
        prometheus_prefixes = [
            'trading_system_', 'mcp_', 'db_', 'redis_', 
            'http_', 'ai_', 'market_'
        ]
        return any(rule.condition.startswith(prefix) for prefix in prometheus_prefixes)
    
    def _evaluate_metric_condition(self, rule: AlertRule, metrics_data: Dict) -> bool:
        """Evaluate a metric-based condition against current data"""
        try:
            condition = rule.condition
            threshold = rule.threshold
            
            # Extract metric name from condition
            if '<' in condition:
                metric_name = condition.split('<')[0].strip()
                current_value = metrics_data.get(metric_name, 0)
                return current_value < threshold
            elif '>' in condition:
                metric_name = condition.split('>')[0].strip()
                current_value = metrics_data.get(metric_name, 0)
                return current_value > threshold
            elif '==' in condition:
                metric_name = condition.split('==')[0].strip()
                current_value = metrics_data.get(metric_name, 0)
                return current_value == threshold
            elif 'abs(' in condition:
                # Handle absolute value conditions like abs(market_sentiment_score) > 0.9
                metric_name = condition.split('abs(')[1].split(')')[0].strip()
                current_value = abs(metrics_data.get(metric_name, 0))
                return current_value > threshold
                
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
            
        return False
    
    def _extract_component_from_rule(self, rule: AlertRule) -> str:
        """Extract component name from alert rule"""
        if 'mcp' in rule.name:
            return 'MCP Agent System'
        elif 'trading' in rule.name:
            return 'Trading System'
        elif 'db' in rule.name:
            return 'Database'
        elif 'redis' in rule.name:
            return 'Redis Cache'
        elif 'http' in rule.name:
            return 'HTTP API'
        elif 'ai' in rule.name:
            return 'AI Processing'
        elif 'market' in rule.name:
            return 'Market Research'
        else:
            return 'System'
    
    def _get_metric_value(self, condition: str, metrics_data: Dict) -> float:
        """Get the actual metric value for alert details"""
        try:
            if '<' in condition or '>' in condition or '==' in condition:
                metric_name = condition.split('<')[0] if '<' in condition else \
                             condition.split('>')[0] if '>' in condition else \
                             condition.split('==')[0]
                metric_name = metric_name.strip()
                return metrics_data.get(metric_name, 0.0)
        except Exception:
            pass
        return 0.0
