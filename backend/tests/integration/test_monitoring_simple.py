#!/usr/bin/env python3
"""
Simple test to verify monitoring modules can be imported
"""

import sys
import os

# Add the backend directory to Python path
backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_path)

def test_imports():
    """Test that all monitoring modules can be imported"""
    print("🔍 Testing monitoring module imports...")
    
    try:
        # Test individual component imports without app dependencies
        print("  • Testing health check enums...")
        from app.monitoring.health_checks import HealthStatus, ComponentType
        print(f"    ✅ HealthStatus: {list(HealthStatus)}")
        print(f"    ✅ ComponentType: {list(ComponentType)}")
        
        print("  • Testing alert system...")
        from app.monitoring.alerts import AlertSeverity, AlertChannel, AlertRule, Alert
        print(f"    ✅ AlertSeverity: {list(AlertSeverity)}")
        print(f"    ✅ AlertChannel: {list(AlertChannel)}")
        
        print("  • Testing metrics collection...")
        # Test basic Prometheus client import
        from prometheus_client import Counter, Histogram, Gauge
        print("    ✅ Prometheus client components imported")
        
        print("✅ All core monitoring components can be imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_data_structures():
    """Test creating monitoring data structures"""
    print("🏗️  Testing data structure creation...")
    
    try:
        from datetime import datetime
        from app.monitoring.health_checks import HealthStatus, ComponentType, HealthCheckResult
        from app.monitoring.alerts import AlertSeverity, AlertChannel, AlertRule, Alert
        
        # Test health check result
        result = HealthCheckResult(
            component="test_component",
            component_type=ComponentType.EXTERNAL_API,
            status=HealthStatus.HEALTHY,
            message="Test component is healthy",
            response_time_ms=50.0
        )
        print(f"  ✅ Created HealthCheckResult: {result.component}")
        
        # Test alert rule
        rule = AlertRule(
            name="test_rule",
            condition="test_condition",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
            description="Test rule"
        )
        print(f"  ✅ Created AlertRule: {rule.name}")
        
        # Test alert
        alert = Alert(
            rule_name="test_alert",
            severity=AlertSeverity.ERROR,
            message="Test alert",
            timestamp=datetime.utcnow()
        )
        print(f"  ✅ Created Alert: {alert.rule_name}")
        
        print("✅ All data structures created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False

def test_metrics_basic():
    """Test basic metrics functionality"""
    print("📊 Testing basic metrics...")
    
    try:
        from prometheus_client import Counter, Gauge, CollectorRegistry, generate_latest
        
        # Create a test registry
        registry = CollectorRegistry()
        
        # Create test metrics
        test_counter = Counter('test_requests_total', 'Test requests', registry=registry)
        test_gauge = Gauge('test_system_status', 'Test system status', registry=registry)
        
        # Update metrics
        test_counter.inc()
        test_gauge.set(1.0)
        
        # Generate metrics output
        metrics_output = generate_latest(registry)
        print(f"  ✅ Generated {len(metrics_output)} bytes of metrics data")
        
        # Verify metrics contain expected data
        metrics_str = metrics_output.decode('utf-8')
        if 'test_requests_total' in metrics_str and 'test_system_status' in metrics_str:
            print("  ✅ Metrics contain expected test data")
        else:
            print("  ⚠️  Metrics missing expected test data")
        
        print("✅ Basic metrics functionality works!")
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False

def main():
    """Run all basic tests"""
    print("🚀 Starting Basic Monitoring System Tests\n")
    
    results = []
    
    # Run tests
    results.append(test_imports())
    print()
    results.append(test_data_structures())
    print()
    results.append(test_metrics_basic())
    print()
    
    # Summary
    if all(results):
        print("🎉 ALL TESTS PASSED!")
        print("✅ Monitoring system components are properly structured and importable")
        return True
    else:
        failed_count = len(results) - sum(results)
        print(f"❌ {failed_count} test(s) FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)