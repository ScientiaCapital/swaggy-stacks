#!/usr/bin/env python3
"""
Simple test script to validate monitoring system functionality
"""

import asyncio
from datetime import datetime
from app.monitoring.health_checks import HealthChecker, HealthStatus, SystemHealthStatus, HealthCheckResult, ComponentType
from app.monitoring.alerts import AlertManager, AlertSeverity, AlertRule, AlertChannel, Alert
from app.monitoring.metrics import PrometheusMetrics

async def test_health_checker():
    """Test basic health checker functionality"""
    print("🔍 Testing Health Checker...")
    
    # Test creating a health check result
    result = HealthCheckResult(
        component="test_component",
        component_type=ComponentType.EXTERNAL_API,
        status=HealthStatus.HEALTHY,
        message="Test component is healthy",
        response_time_ms=50.0
    )
    
    print(f"✅ Created health check result: {result.component} - {result.status}")
    
    # Test system health status compilation
    health_status = SystemHealthStatus(
        overall_status=HealthStatus.HEALTHY,
        timestamp=datetime.utcnow(),
        components=[result],
        summary={HealthStatus.HEALTHY: 1},
        issues=[],
        uptime_seconds=3600.0
    )
    
    print(f"✅ Created system health status: {health_status.overall_status}")
    return True

def test_alert_manager():
    """Test alert manager functionality"""
    print("🚨 Testing Alert Manager...")
    
    # Create alert manager
    alert_manager = AlertManager()
    
    # Test alert rule creation
    test_rule = AlertRule(
        name="test_alert",
        condition="test_condition",
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.LOG],
        cooldown_minutes=5,
        description="Test alert rule"
    )
    
    alert_manager.add_alert_rule(test_rule)
    print(f"✅ Added alert rule: {test_rule.name}")
    
    # Test alert creation
    test_alert = Alert(
        rule_name="test_alert",
        severity=AlertSeverity.WARNING,
        message="Test alert message",
        timestamp=datetime.utcnow(),
        component="test_component"
    )
    
    print(f"✅ Created alert: {test_alert.rule_name} - {test_alert.severity}")
    
    # Test alert stats
    stats = alert_manager.get_alert_stats()
    print(f"✅ Alert stats: {stats}")
    
    return True

def test_prometheus_metrics():
    """Test Prometheus metrics functionality"""
    print("📊 Testing Prometheus Metrics...")
    
    # Create metrics collector
    metrics = PrometheusMetrics()
    print("✅ Created Prometheus metrics collector")
    
    # Test metric recording
    metrics.record_mcp_request("test_server", "test_method", 0.1, "success")
    print("✅ Recorded MCP request metric")
    
    metrics.update_mcp_server_status("test_server", "mcp", True)
    print("✅ Updated MCP server status metric")
    
    # Test getting metrics
    prometheus_output = metrics.get_metrics()
    print(f"✅ Generated Prometheus metrics ({len(prometheus_output)} characters)")
    
    return True

async def main():
    """Run all monitoring tests"""
    print("🚀 Starting Monitoring System Tests\n")
    
    try:
        # Test health checker
        health_test = await test_health_checker()
        print()
        
        # Test alert manager
        alert_test = test_alert_manager()
        print()
        
        # Test prometheus metrics
        metrics_test = test_prometheus_metrics()
        print()
        
        if health_test and alert_test and metrics_test:
            print("✅ All monitoring system tests PASSED!")
            return True
        else:
            print("❌ Some tests FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)