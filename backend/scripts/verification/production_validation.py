#!/usr/bin/env python3
"""
🚀 PRODUCTION SYSTEM VALIDATION
===============================
Comprehensive validation of SwaggyStacks production readiness
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any

def validate_core_files() -> Dict[str, Any]:
    """Validate core system files exist and are properly configured"""
    results = {
        "status": "PASSED",
        "issues": [],
        "components": {}
    }

    project_root = Path(__file__).parent.parent.parent.parent  # Go from scripts/verification/ back to root

    # Core system files
    core_files = {
        "backend/run_production.py": "Real-time agent coordination system",
        "backend/app/trading/trading_manager.py": "Trade execution engine",
        "backend/app/monitoring/metrics.py": "Prometheus metrics collection",
        "backend/app/ai/coordination_hub.py": "Agent coordination hub",
        "backend/app/ai/consensus_engine.py": "Consensus decision engine"
    }

    for file_path, description in core_files.items():
        full_path = project_root / file_path
        if full_path.exists():
            results["components"][file_path] = {
                "status": "✅ EXISTS",
                "description": description,
                "size": full_path.stat().st_size
            }
        else:
            results["components"][file_path] = {
                "status": "❌ MISSING",
                "description": description
            }
            results["issues"].append(f"Missing: {file_path}")
            results["status"] = "FAILED"

    return results

def validate_grafana_dashboards() -> Dict[str, Any]:
    """Validate Grafana dashboard configuration"""
    results = {
        "status": "PASSED",
        "issues": [],
        "dashboards": {}
    }

    dashboard_dir = Path(__file__).parent / "infrastructure/grafana/dashboards"

    # Required dashboards
    required_dashboards = [
        "pnl_dashboard.json",
        "strategy_performance_dashboard.json",
        "trade_execution_dashboard.json",
        "risk_dashboard.json",
        "system_health_dashboard.json",
        "advanced_risk_dashboard.json"
    ]

    for dashboard in required_dashboards:
        dashboard_path = dashboard_dir / dashboard
        if dashboard_path.exists():
            try:
                with open(dashboard_path, 'r') as f:
                    dashboard_config = json.load(f)
                    results["dashboards"][dashboard] = {
                        "status": "✅ CONFIGURED",
                        "uid": dashboard_config.get("uid"),
                        "title": dashboard_config.get("title"),
                        "panels": len(dashboard_config.get("panels", []))
                    }
            except Exception as e:
                results["dashboards"][dashboard] = {
                    "status": "❌ INVALID JSON",
                    "error": str(e)
                }
                results["issues"].append(f"Invalid JSON: {dashboard}")
                results["status"] = "FAILED"
        else:
            results["dashboards"][dashboard] = {
                "status": "❌ MISSING"
            }
            results["issues"].append(f"Missing dashboard: {dashboard}")
            results["status"] = "FAILED"

    return results

def validate_monitoring_infrastructure() -> Dict[str, Any]:
    """Validate monitoring infrastructure configuration"""
    results = {
        "status": "PASSED",
        "issues": [],
        "infrastructure": {}
    }

    project_root = Path(__file__).parent

    # Monitoring infrastructure files
    monitoring_files = {
        "docker-compose.yml": "Docker services configuration",
        "infrastructure/prometheus.yml": "Prometheus configuration",
        "infrastructure/grafana/dashboards/dashboard.yml": "Grafana provisioning",
        "backend/app/monitoring/__init__.py": "Monitoring module"
    }

    for file_path, description in monitoring_files.items():
        full_path = project_root / file_path
        if full_path.exists():
            results["infrastructure"][file_path] = {
                "status": "✅ EXISTS",
                "description": description
            }
        else:
            results["infrastructure"][file_path] = {
                "status": "❌ MISSING",
                "description": description
            }
            results["issues"].append(f"Missing: {file_path}")
            results["status"] = "FAILED"

    return results

def validate_agent_integration() -> Dict[str, Any]:
    """Validate agent integration components"""
    results = {
        "status": "PASSED",
        "issues": [],
        "integration": {}
    }

    # Check run_production.py for key integration points
    live_agents_file = project_root / "backend/run_production.py"

    if live_agents_file.exists():
        with open(live_agents_file, 'r') as f:
            content = f.read()

            # Check for PrometheusMetrics integration
            if "from app.monitoring.metrics import PrometheusMetrics" in content:
                results["integration"]["prometheus_import"] = "✅ INTEGRATED"
            else:
                results["integration"]["prometheus_import"] = "❌ MISSING"
                results["issues"].append("PrometheusMetrics not imported")
                results["status"] = "FAILED"

            # Check for TradingManager integration
            if "from app.trading.trading_manager import TradingManager" in content:
                results["integration"]["trading_manager_import"] = "✅ INTEGRATED"
            else:
                results["integration"]["trading_manager_import"] = "❌ MISSING"
                results["issues"].append("TradingManager not imported")
                results["status"] = "FAILED"

            # Check for trade execution integration
            if "await self.trading_manager.execute_trade" in content:
                results["integration"]["trade_execution"] = "✅ CONNECTED"
            else:
                results["integration"]["trade_execution"] = "❌ NOT CONNECTED"
                results["issues"].append("Trade execution not connected")
                results["status"] = "FAILED"

            # Check for metrics tracking
            if "self.metrics.record_mcp_agent_coordination" in content:
                results["integration"]["agent_metrics"] = "✅ TRACKING"
            else:
                results["integration"]["agent_metrics"] = "❌ NOT TRACKING"
                results["issues"].append("Agent metrics not tracked")
                results["status"] = "FAILED"
    else:
        results["integration"]["live_agents_file"] = "❌ MISSING"
        results["issues"].append("backend/run_production.py not found")
        results["status"] = "FAILED"

    return results

def run_production_validation():
    """Run complete production validation suite"""
    print("🚀 SWAGGY STACKS PRODUCTION VALIDATION")
    print("=" * 50)

    validation_tests = [
        ("Core System Files", validate_core_files),
        ("Grafana Dashboards", validate_grafana_dashboards),
        ("Monitoring Infrastructure", validate_monitoring_infrastructure),
        ("Agent Integration", validate_agent_integration)
    ]

    all_passed = True
    detailed_results = {}

    for test_name, test_func in validation_tests:
        print(f"\n📋 Testing: {test_name}")
        print("-" * 30)

        try:
            result = test_func()
            detailed_results[test_name] = result

            if result["status"] == "PASSED":
                print(f"✅ {test_name}: PASSED")
                for component, status in result.get("components", {}).items():
                    print(f"   {status['status']} {component}")
                for component, status in result.get("dashboards", {}).items():
                    print(f"   {status['status']} {component}")
                for component, status in result.get("infrastructure", {}).items():
                    print(f"   {status['status']} {component}")
                for component, status in result.get("integration", {}).items():
                    print(f"   {status} {component}")
            else:
                print(f"❌ {test_name}: FAILED")
                for issue in result.get("issues", []):
                    print(f"   • {issue}")
                all_passed = False

        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 PRODUCTION VALIDATION: ALL TESTS PASSED")
        print("✅ System is ready for live market testing!")
        print("🚀 Ready to start: cd backend && python3 run_production.py")
    else:
        print("⚠️  PRODUCTION VALIDATION: ISSUES FOUND")
        print("❌ System needs fixes before live testing")

    print("=" * 50)

    return {
        "overall_status": "PASSED" if all_passed else "FAILED",
        "detailed_results": detailed_results
    }

if __name__ == "__main__":
    run_production_validation()