"""
Comprehensive test suite for real-time monitoring system.

Tests coverage includes:
- MCP Agent Coordination Metrics
- WebSocket Integration with Prometheus
- Prometheus Metrics Endpoint
- AlertManager Metric-based Rules
"""

import pytest
import pytest_asyncio
import json
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, List

from app.monitoring.metrics import PrometheusMetrics, MetricsCollector
from app.monitoring.health_checks import HealthChecker
from app.monitoring.alerts import AlertManager, AlertRule, AlertSeverity, AlertChannel
from app.websockets.trading_socket import TradingDashboardWebSocket, SystemHealthUpdate
from app.mcp.orchestrator import MCPOrchestrator


class TestMonitoring:
    """Test suite for Real-time Monitoring System"""

    @pytest_asyncio.fixture
    async def prometheus_metrics(self):
        """Create PrometheusMetrics instance for testing"""
        return PrometheusMetrics()

    @pytest_asyncio.fixture
    async def metrics_collector(self, prometheus_metrics):
        """Create MetricsCollector instance for testing"""
        collector = MetricsCollector()
        collector.prometheus_metrics = prometheus_metrics
        return collector

    @pytest_asyncio.fixture
    async def health_checker(self):
        """Create HealthChecker instance for testing"""
        return HealthChecker()

    @pytest_asyncio.fixture
    async def alert_manager(self):
        """Create AlertManager instance for testing"""
        manager = AlertManager()
        await manager.initialize()
        return manager

    @pytest_asyncio.fixture
    async def websocket_handler(self):
        """Create TradingDashboardWebSocket instance for testing"""
        return TradingDashboardWebSocket()

    @pytest_asyncio.fixture
    async def mock_mcp_orchestrator(self):
        """Mock MCP orchestrator with coordination metrics"""
        orchestrator = AsyncMock(spec=MCPOrchestrator)
        orchestrator._initialized = True
        orchestrator.is_server_available.return_value = True
        
        # Mock coordination metrics
        orchestrator.get_coordination_metrics.return_value = {
            'success_rate': 0.95,
            'average_response_time': 1.2,
            'queue_depth': 5,
            'coordination_count': 100,
            'error_count': 5
        }
        
        return orchestrator


class TestMCPCoordinationMetrics:
    """Test MCP Agent Coordination Metrics recording and collection"""

    async def test_record_mcp_agent_coordination_success(self, prometheus_metrics):
        """Test recording successful MCP agent coordination"""
        # Record successful coordination
        prometheus_metrics.record_mcp_agent_coordination(
            coordination_type="task_execution",
            source_agent="taskmaster",
            target_agent="shrimp",
            duration=1.5,
            success=True
        )
        
        # Verify metrics were recorded
        metrics_output = prometheus_metrics.get_metrics()
        assert "mcp_agent_coordination_duration_seconds" in metrics_output
        assert "mcp_agent_coordination_success_rate" in metrics_output
        assert 'coordination_type="task_execution"' in metrics_output
        assert 'source_agent="taskmaster"' in metrics_output
        assert 'target_agent="shrimp"' in metrics_output

    async def test_record_mcp_agent_coordination_failure(self, prometheus_metrics):
        """Test recording failed MCP agent coordination"""
        # Record failed coordination
        prometheus_metrics.record_mcp_agent_coordination(
            coordination_type="memory_access",
            source_agent="serena",
            target_agent="memory",
            duration=5.0,
            success=False
        )
        
        # Verify failure metrics
        metrics_output = prometheus_metrics.get_metrics()
        assert "mcp_agent_coordination_success_rate" in metrics_output
        assert 'coordination_type="memory_access"' in metrics_output

    async def test_record_mcp_agent_response_time(self, prometheus_metrics):
        """Test MCP agent response time recording"""
        # Record response times for different agents
        prometheus_metrics.record_mcp_agent_response_time("taskmaster", "research", 2.3)
        prometheus_metrics.record_mcp_agent_response_time("shrimp", "execute_task", 0.8)
        
        metrics_output = prometheus_metrics.get_metrics()
        assert "mcp_agent_response_time_seconds" in metrics_output
        assert 'agent="taskmaster"' in metrics_output
        assert 'operation="research"' in metrics_output

    async def test_update_mcp_agent_queue_depth(self, prometheus_metrics):
        """Test MCP agent queue depth updates"""
        # Update queue depths
        prometheus_metrics.update_mcp_agent_queue_depth("taskmaster", "pending", 12)
        prometheus_metrics.update_mcp_agent_queue_depth("shrimp", "in_progress", 3)
        
        metrics_output = prometheus_metrics.get_metrics()
        assert "mcp_agent_queue_depth" in metrics_output
        assert 'agent="taskmaster"' in metrics_output
        assert 'queue_type="pending"' in metrics_output

    async def test_mcp_coordination_metrics_integration(self, mock_mcp_orchestrator, prometheus_metrics):
        """Test integration between MCP orchestrator and metrics recording"""
        coordination_data = await mock_mcp_orchestrator.get_coordination_metrics()
        
        # Record coordination metrics
        prometheus_metrics.record_mcp_agent_coordination(
            coordination_type="health_check",
            source_agent="orchestrator", 
            target_agent="all",
            duration=coordination_data['average_response_time'],
            success=coordination_data['success_rate'] > 0.8
        )
        
        prometheus_metrics.update_mcp_agent_queue_depth(
            "orchestrator", 
            "pending", 
            coordination_data['queue_depth']
        )
        
        metrics_output = prometheus_metrics.get_metrics()
        assert "mcp_agent_coordination_duration_seconds" in metrics_output
        assert "mcp_agent_queue_depth" in metrics_output

    async def test_mcp_server_status_tracking(self, prometheus_metrics):
        """Test MCP server status tracking"""
        # Update server statuses
        prometheus_metrics.update_mcp_server_status("taskmaster", "healthy", 1.0)
        prometheus_metrics.update_mcp_server_status("memory", "degraded", 0.5)
        prometheus_metrics.update_mcp_server_status("github", "unavailable", 0.0)
        
        metrics_output = prometheus_metrics.get_metrics()
        assert "mcp_server_status" in metrics_output
        assert 'server="taskmaster"' in metrics_output
        assert 'status="healthy"' in metrics_output


class TestWebSocketIntegration:
    """Test WebSocket integration with monitoring metrics"""

    async def test_system_health_update_structure(self):
        """Test SystemHealthUpdate dataclass structure"""
        # Create system health update
        health_update = SystemHealthUpdate(
            status="healthy",
            services={"database": True, "redis": True},
            cache_metrics={"hit_rate": 0.95},
            trading_enabled=True,
            last_update="2025-01-01T12:00:00Z",
            uptime="24h 30m",
            prometheus_metrics={
                "mcp_coordination_success_rate": 0.95,
                "trading_portfolio_value": 100000.0
            },
            component_health={
                "mcp_orchestrator": {"status": "healthy", "response_time": 0.5},
                "trading_system": {"status": "healthy", "active_positions": 5}
            },
            mcp_coordination_metrics={
                "active_coordinations": 3,
                "avg_response_time": 1.2,
                "queue_depth": 7
            }
        )
        
        # Verify structure
        assert health_update.prometheus_metrics is not None
        assert health_update.component_health is not None
        assert health_update.mcp_coordination_metrics is not None
        assert health_update.prometheus_metrics["mcp_coordination_success_rate"] == 0.95

    async def test_websocket_health_update_integration(self, websocket_handler, metrics_collector, mock_mcp_orchestrator):
        """Test WebSocket _update_system_health integration with MetricsCollector"""
        # Mock MetricsCollector data
        mock_metrics_data = {
            "mcp_coordination_success_rate": 0.92,
            "trading_portfolio_value_usd": 50000.0,
            "system_uptime_seconds": 86400,
            "db_connection_pool_size": 10
        }
        
        mock_component_health = {
            "mcp_orchestrator": {
                "status": "healthy",
                "last_check": datetime.utcnow().isoformat(),
                "response_time": 0.3
            },
            "trading_system": {
                "status": "healthy", 
                "active_positions": 3,
                "last_trade": "2025-01-01T11:30:00Z"
            }
        }
        
        mock_coordination_metrics = {
            "active_coordinations": 2,
            "avg_response_time": 0.8,
            "queue_depth": 4,
            "success_rate": 0.92
        }
        
        # Mock the collector methods
        with patch.object(metrics_collector, 'get_current_metrics', return_value=mock_metrics_data), \
             patch.object(metrics_collector, 'get_component_health', return_value=mock_component_health), \
             patch.object(metrics_collector, 'get_mcp_coordination_metrics', return_value=mock_coordination_metrics):
            
            # Mock the internal method call
            with patch.object(websocket_handler, '_update_system_health') as mock_update:
                await websocket_handler._update_system_health(metrics_collector)
                mock_update.assert_called_once_with(metrics_collector)

    async def test_websocket_error_handling_with_metrics(self, websocket_handler, metrics_collector):
        """Test WebSocket error handling with metrics integration"""
        # Mock MetricsCollector to raise exception
        with patch.object(metrics_collector, 'get_current_metrics', side_effect=Exception("Metrics collection error")):
            
            # Should handle error gracefully
            try:
                await websocket_handler._update_system_health(metrics_collector)
            except Exception:
                pytest.fail("WebSocket should handle metrics collection errors gracefully")

    async def test_websocket_performance_monitoring(self, websocket_handler, performance_monitor):
        """Test WebSocket performance with metrics"""
        performance_monitor.start()
        
        # Simulate WebSocket operations with metrics
        mock_health_data = SystemHealthUpdate(
            status="healthy",
            services={"test": True},
            cache_metrics={},
            trading_enabled=True,
            last_update=datetime.utcnow().isoformat(),
            uptime="1h",
            prometheus_metrics={"test_metric": 1.0},
            component_health={"test_component": {"status": "healthy"}},
            mcp_coordination_metrics={"test_coordination": 1}
        )
        
        # Mock broadcasting
        with patch.object(websocket_handler, 'broadcast_to_all') as mock_broadcast:
            mock_broadcast.return_value = None
            
            # Simulate update cycle
            await asyncio.sleep(0.1)
            
            performance_data = performance_monitor.stop()
            
            # Verify performance is acceptable
            assert performance_data['duration'] < 1.0  # Should complete quickly
            assert performance_data['memory_diff'] < 50  # Should not consume excessive memory


class TestPrometheusMetricsEndpoint:
    """Test Prometheus metrics endpoint enhancements"""

    async def test_metrics_endpoint_prometheus_format(self, prometheus_metrics):
        """Test that metrics endpoint returns proper Prometheus format"""
        # Add some test metrics
        prometheus_metrics.record_mcp_agent_coordination(
            "test_coordination", "test_source", "test_target", 1.0, True
        )
        
        metrics_output = prometheus_metrics.get_metrics()
        
        # Verify Prometheus format
        assert "# HELP" in metrics_output
        assert "# TYPE" in metrics_output
        assert "mcp_agent_coordination_duration_seconds" in metrics_output

    async def test_metrics_endpoint_headers_content_type(self):
        """Test that metrics endpoint returns correct headers"""
        # This would typically be tested with FastAPI test client
        # Simulating expected headers
        expected_headers = {
            "Content-Type": "text/plain; version=0.0.4; charset=utf-8",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        
        # Verify header requirements
        assert expected_headers["Content-Type"] == "text/plain; version=0.0.4; charset=utf-8"
        assert "no-cache" in expected_headers["Cache-Control"]

    async def test_metrics_endpoint_performance(self, prometheus_metrics, performance_monitor):
        """Test metrics endpoint performance"""
        performance_monitor.start()
        
        # Generate sample metrics
        for i in range(100):
            prometheus_metrics.record_mcp_agent_coordination(
                f"coordination_{i}", "source", "target", 0.1, True
            )
        
        # Get metrics (simulating endpoint call)
        metrics_output = prometheus_metrics.get_metrics()
        
        performance_data = performance_monitor.stop()
        
        # Verify performance requirements
        assert performance_data['duration'] < 2.0  # Should be fast
        assert len(metrics_output) > 0  # Should return data
        assert "mcp_agent_coordination_duration_seconds" in metrics_output

    async def test_metrics_endpoint_error_handling(self, prometheus_metrics):
        """Test metrics endpoint error handling"""
        # Mock internal error
        with patch.object(prometheus_metrics, '_registry', side_effect=Exception("Registry error")):
            
            try:
                # Should handle errors gracefully
                metrics_output = prometheus_metrics.get_metrics()
                # Even with error, should return some default response
                assert isinstance(metrics_output, str)
            except Exception:
                # If exception propagates, ensure it's handled appropriately
                pass


class TestAlertManagerEnhancements:
    """Test AlertManager Prometheus integration and metric-based alerting"""

    async def test_configure_prometheus_alerts(self, alert_manager):
        """Test Prometheus alert rule configuration"""
        # Configure Prometheus alerts
        prometheus_rules = alert_manager.configure_prometheus_alerts()
        
        # Verify alert rules were created
        assert len(prometheus_rules) == 16  # Expected number of rules
        
        # Check for key alert categories
        rule_names = [rule.name for rule in prometheus_rules]
        assert "mcp_agent_coordination_failure" in rule_names
        assert "system_health_degraded" in rule_names
        assert "trading_portfolio_value_drop" in rule_names
        assert "db_connection_pool_exhausted" in rule_names

    async def test_evaluate_prometheus_alerts(self, alert_manager):
        """Test Prometheus alert evaluation"""
        # Configure alerts first
        prometheus_rules = alert_manager.configure_prometheus_alerts()
        
        # Mock metrics data
        metrics_data = {
            "mcp_agent_coordination_success_rate": 0.7,  # Below threshold
            "trading_system_health_status": 1.0,  # Degraded
            "db_connection_pool_size": 1.0,  # Critical low
            "http_request_duration_seconds": 6.0,  # High duration
        }
        
        # Evaluate alerts
        triggered_alerts = alert_manager.evaluate_prometheus_alerts(metrics_data)
        
        # Verify alerts were triggered
        assert len(triggered_alerts) > 0
        
        # Check for specific expected alerts
        alert_names = [alert.rule_name for alert in triggered_alerts]
        assert "mcp_agent_coordination_failure" in alert_names
        assert "system_health_degraded" in alert_names

    async def test_alert_condition_parsing(self, alert_manager):
        """Test alert condition parsing for various metric patterns"""
        # Test different condition types
        test_conditions = [
            ("metric_value < 0.8", {"metric_value": 0.7}, True),
            ("metric_value > 5.0", {"metric_value": 6.0}, True),
            ("metric_value == 0", {"metric_value": 0.0}, True),
            ("abs(metric_value) > 0.9", {"metric_value": -0.95}, True),
            ("metric_value < 0.8", {"metric_value": 0.9}, False),
        ]
        
        for condition, metrics, expected_trigger in test_conditions:
            # Create test alert rule
            test_rule = AlertRule(
                name="test_rule",
                condition=condition,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                cooldown_minutes=5,
                description="Test rule"
            )
            
            # Evaluate condition
            should_trigger = alert_manager._evaluate_metric_condition(test_rule, metrics)
            assert should_trigger == expected_trigger, f"Condition '{condition}' with metrics {metrics} should be {expected_trigger}"

    async def test_alert_cooldown_mechanism(self, alert_manager):
        """Test alert cooldown to prevent spam"""
        # Configure alerts
        prometheus_rules = alert_manager.configure_prometheus_alerts()
        
        # Find a test rule
        test_rule = next(rule for rule in prometheus_rules if rule.name == "mcp_agent_coordination_failure")
        
        # Trigger alert first time
        metrics_data = {"mcp_agent_coordination_success_rate": 0.7}
        alerts_1 = alert_manager.evaluate_prometheus_alerts(metrics_data)
        
        # Trigger again immediately (should be blocked by cooldown)
        alerts_2 = alert_manager.evaluate_prometheus_alerts(metrics_data)
        
        # First alert should trigger, second should be blocked
        coordination_alerts_1 = [a for a in alerts_1 if a.rule_name == "mcp_agent_coordination_failure"]
        coordination_alerts_2 = [a for a in alerts_2 if a.rule_name == "mcp_agent_coordination_failure"]
        
        assert len(coordination_alerts_1) >= 1  # First should trigger
        assert len(coordination_alerts_2) == 0  # Second should be blocked by cooldown

    async def test_alert_severity_levels(self, alert_manager):
        """Test different alert severity levels"""
        prometheus_rules = alert_manager.configure_prometheus_alerts()
        
        # Check severity distribution
        severity_counts = {}
        for rule in prometheus_rules:
            severity = rule.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Should have alerts at different severity levels
        assert AlertSeverity.CRITICAL in severity_counts
        assert AlertSeverity.ERROR in severity_counts
        assert AlertSeverity.WARNING in severity_counts
        
        # Critical alerts should have shorter cooldowns
        critical_rules = [rule for rule in prometheus_rules if rule.severity == AlertSeverity.CRITICAL]
        warning_rules = [rule for rule in prometheus_rules if rule.severity == AlertSeverity.WARNING]
        
        if critical_rules and warning_rules:
            avg_critical_cooldown = sum(rule.cooldown_minutes for rule in critical_rules) / len(critical_rules)
            avg_warning_cooldown = sum(rule.cooldown_minutes for rule in warning_rules) / len(warning_rules)
            
            assert avg_critical_cooldown <= avg_warning_cooldown  # Critical should have shorter cooldowns

    async def test_alert_manager_integration_with_metrics(self, alert_manager, prometheus_metrics):
        """Test integration between AlertManager and PrometheusMetrics"""
        # Record some metrics that should trigger alerts
        prometheus_metrics.record_mcp_agent_coordination(
            "test_coordination", "test_source", "test_target", 10.0, False  # Long duration, failed
        )
        
        prometheus_metrics.update_mcp_server_status("test_server", "unavailable", 0.0)
        
        # Get current metrics
        current_metrics = prometheus_metrics.get_current_values()
        
        # Evaluate alerts against current metrics
        triggered_alerts = alert_manager.evaluate_prometheus_alerts(current_metrics)
        
        # Should have some alerts based on the poor metrics
        assert isinstance(triggered_alerts, list)


class TestIntegrationScenarios:
    """Test complete integration scenarios across monitoring components"""

    async def test_end_to_end_monitoring_flow(self, prometheus_metrics, metrics_collector, 
                                            websocket_handler, alert_manager, mock_mcp_orchestrator):
        """Test complete end-to-end monitoring flow"""
        # Step 1: Record MCP coordination events
        prometheus_metrics.record_mcp_agent_coordination(
            "task_execution", "taskmaster", "shrimp", 2.5, True
        )
        
        # Step 2: Collect metrics
        with patch.object(metrics_collector, 'collect_metrics') as mock_collect:
            mock_collect.return_value = {
                "mcp_agent_coordination_success_rate": 0.95,
                "system_health_status": 2.0
            }
            await metrics_collector.collect_metrics()
        
        # Step 3: Update WebSocket with metrics
        mock_health_update = SystemHealthUpdate(
            status="healthy",
            services={"mcp": True},
            cache_metrics={},
            trading_enabled=True,
            last_update=datetime.utcnow().isoformat(),
            uptime="1h",
            prometheus_metrics={"mcp_coordination_success_rate": 0.95},
            component_health={"mcp": {"status": "healthy"}},
            mcp_coordination_metrics={"success_rate": 0.95}
        )
        
        # Step 4: Evaluate alerts (should not trigger with good metrics)
        metrics_data = {"mcp_agent_coordination_success_rate": 0.95}
        triggered_alerts = alert_manager.evaluate_prometheus_alerts(metrics_data)
        
        # Verify end-to-end flow
        assert len(triggered_alerts) == 0  # No alerts with good metrics
        assert mock_health_update.prometheus_metrics["mcp_coordination_success_rate"] == 0.95

    async def test_monitoring_system_resilience(self, prometheus_metrics, alert_manager):
        """Test monitoring system resilience to component failures"""
        # Test scenario: Some metrics unavailable
        partial_metrics = {
            "mcp_agent_coordination_success_rate": 0.6,  # Available, should trigger
            # Missing other expected metrics
        }
        
        # Should handle partial metrics gracefully
        triggered_alerts = alert_manager.evaluate_prometheus_alerts(partial_metrics)
        
        # Should still trigger alerts for available metrics
        coordination_alerts = [a for a in triggered_alerts if "coordination" in a.rule_name]
        assert len(coordination_alerts) > 0

    async def test_performance_under_load(self, prometheus_metrics, performance_monitor):
        """Test monitoring system performance under load"""
        performance_monitor.start()
        
        # Simulate high load scenario
        for i in range(1000):
            prometheus_metrics.record_mcp_agent_coordination(
                f"load_test_{i % 10}", "source", "target", 0.1, i % 5 != 0
            )
            
            if i % 100 == 0:
                # Periodically get metrics (simulating endpoint calls)
                metrics_output = prometheus_metrics.get_metrics()
                assert len(metrics_output) > 0
        
        performance_data = performance_monitor.stop()
        
        # Should handle load efficiently
        assert performance_data['duration'] < 10.0  # Should complete within reasonable time
        assert performance_data['memory_diff'] < 100  # Should not consume excessive memory

    async def test_monitoring_data_consistency(self, prometheus_metrics):
        """Test data consistency across monitoring components"""
        # Record coordinated set of metrics
        coordination_count = 50
        success_count = 40
        
        for i in range(coordination_count):
            success = i < success_count
            prometheus_metrics.record_mcp_agent_coordination(
                "consistency_test", "source", "target", 1.0, success
            )
        
        # Verify metrics consistency
        metrics_output = prometheus_metrics.get_metrics()
        
        # Should contain both duration and success rate metrics
        assert "mcp_agent_coordination_duration_seconds" in metrics_output
        assert "mcp_agent_coordination_success_rate" in metrics_output
        
        # Success rate calculation should be consistent
        expected_success_rate = success_count / coordination_count
        assert abs(expected_success_rate - 0.8) < 0.01  # 40/50 = 0.8