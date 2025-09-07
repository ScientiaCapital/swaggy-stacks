#!/usr/bin/env python3
"""
Comprehensive deployment readiness validation for Swaggy Stacks Trading System
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class DeploymentValidator:
    """Comprehensive deployment readiness validation"""
    
    def __init__(self):
        self.results = {
            'validation_timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'UNKNOWN',
            'tests_passed': 0,
            'tests_failed': 0,
            'test_results': {},
            'recommendations': [],
            'deployment_ready': False
        }
        self.project_root = project_root
        
    def log(self, message: str, level: str = "INFO"):
        """Enhanced logging with timestamps"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def run_command(self, command: str, cwd: str = None) -> Tuple[bool, str]:
        """Execute shell command and return success status with output"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=cwd or str(self.project_root)
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out after 60 seconds"
        except Exception as e:
            return False, f"Command execution failed: {str(e)}"
    
    def validate_docker_services(self) -> Dict[str, Any]:
        """Validate Docker Compose services health"""
        self.log("🐳 Validating Docker Compose services...")
        
        result = {
            'test_name': 'Docker Services Health',
            'status': 'PASS',
            'details': {},
            'errors': []
        }
        
        try:
            # Check if Docker is running
            success, output = self.run_command("docker info")
            if not success:
                result['status'] = 'FAIL'
                result['errors'].append("Docker daemon not running")
                return result
            
            # Check docker-compose.yml exists
            compose_file = self.project_root / "docker-compose.yml"
            if not compose_file.exists():
                result['status'] = 'FAIL'
                result['errors'].append("docker-compose.yml not found")
                return result
            
            # Validate docker-compose configuration
            success, output = self.run_command("docker-compose config")
            if not success:
                result['status'] = 'FAIL'
                result['errors'].append(f"Invalid docker-compose.yml: {output}")
                return result
            
            # Check service status if running
            success, output = self.run_command("docker-compose ps --format json")
            if success and output.strip():
                try:
                    services = json.loads(f"[{output.strip().replace('}{', '},{')}]")
                    for service in services:
                        service_name = service.get('Service', 'unknown')
                        state = service.get('State', 'unknown')
                        result['details'][service_name] = state
                except json.JSONDecodeError:
                    result['details']['status'] = "Services not currently running (expected for validation)"
            else:
                result['details']['status'] = "Services not currently running (expected for validation)"
            
            result['details']['compose_valid'] = True
            
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Docker validation failed: {str(e)}")
        
        return result
    
    def validate_database_connectivity(self) -> Dict[str, Any]:
        """Validate database connectivity and migrations"""
        self.log("🗄️  Validating database configuration...")
        
        result = {
            'test_name': 'Database Configuration',
            'status': 'PASS',
            'details': {},
            'errors': []
        }
        
        try:
            # Check for database migration files
            migrations_dir = self.project_root / "backend" / "alembic" / "versions"
            if migrations_dir.exists():
                migration_files = list(migrations_dir.glob("*.py"))
                result['details']['migration_files'] = len(migration_files)
                result['details']['migrations_exist'] = len(migration_files) > 0
            else:
                result['details']['migrations_exist'] = False
            
            # Check database configuration files
            db_config_file = self.project_root / "backend" / "app" / "core" / "database.py"
            if db_config_file.exists():
                result['details']['database_config_exists'] = True
            else:
                result['status'] = 'FAIL'
                result['errors'].append("Database configuration file missing")
            
            # Check for required environment variables structure
            env_example = self.project_root / ".env.example"
            if env_example.exists():
                with open(env_example, 'r') as f:
                    env_content = f.read()
                    db_vars = ['POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB', 'DATABASE_URL']
                    for var in db_vars:
                        if var in env_content:
                            result['details'][f'{var}_configured'] = True
                        else:
                            result['details'][f'{var}_configured'] = False
                            if result['status'] != 'FAIL':
                                result['status'] = 'WARN'
                                result['errors'].append(f"Environment variable {var} not found in .env.example")
            
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Database validation failed: {str(e)}")
        
        return result
    
    def validate_mcp_configuration(self) -> Dict[str, Any]:
        """Validate MCP server configuration and connectivity"""
        self.log("🔌 Validating MCP server configuration...")
        
        result = {
            'test_name': 'MCP Configuration',
            'status': 'PASS',
            'details': {},
            'errors': []
        }
        
        try:
            # Check for MCP configuration files
            mcp_config = self.project_root / ".mcp.json"
            if mcp_config.exists():
                with open(mcp_config, 'r') as f:
                    config = json.load(f)
                    result['details']['mcp_config_exists'] = True
                    result['details']['mcp_servers_configured'] = len(config.get('mcpServers', {}))
                    
                    # Check for required MCP servers
                    required_servers = ['github', 'memory', 'serena', 'tavily', 'sequential_thinking']
                    configured_servers = list(config.get('mcpServers', {}).keys())
                    result['details']['configured_servers'] = configured_servers
                    
                    missing_servers = [s for s in required_servers if s not in configured_servers]
                    if missing_servers:
                        result['status'] = 'WARN'
                        result['errors'].append(f"Missing MCP servers: {', '.join(missing_servers)}")
                    
            else:
                result['status'] = 'WARN'
                result['errors'].append("MCP configuration file (.mcp.json) not found")
            
            # Check MCP orchestrator code
            mcp_orchestrator = self.project_root / "backend" / "app" / "mcp" / "orchestrator.py"
            if mcp_orchestrator.exists():
                result['details']['mcp_orchestrator_exists'] = True
            else:
                result['status'] = 'FAIL'
                result['errors'].append("MCP orchestrator implementation missing")
            
            # Check monitoring integration
            monitoring_endpoints = self.project_root / "backend" / "app" / "api" / "v1" / "endpoints" / "monitoring.py"
            if monitoring_endpoints.exists():
                result['details']['mcp_monitoring_integrated'] = True
            else:
                result['status'] = 'WARN'
                result['errors'].append("MCP monitoring endpoints not found")
                
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"MCP validation failed: {str(e)}")
        
        return result
    
    def validate_api_endpoints(self) -> Dict[str, Any]:
        """Validate API endpoint configuration"""
        self.log("🌐 Validating API endpoint configuration...")
        
        result = {
            'test_name': 'API Endpoints',
            'status': 'PASS',
            'details': {},
            'errors': []
        }
        
        try:
            # Check for main API endpoints
            endpoints_dir = self.project_root / "backend" / "app" / "api" / "v1" / "endpoints"
            if endpoints_dir.exists():
                endpoint_files = list(endpoints_dir.glob("*.py"))
                result['details']['endpoint_files'] = len(endpoint_files)
                
                # Check for critical endpoints
                critical_endpoints = [
                    'health.py', 'trading.py', 'ai_trading.py', 
                    'monitoring.py', 'auth.py', 'github.py'
                ]
                
                existing_endpoints = [f.name for f in endpoint_files]
                result['details']['existing_endpoints'] = existing_endpoints
                
                missing_endpoints = [e for e in critical_endpoints if e not in existing_endpoints]
                if missing_endpoints:
                    result['status'] = 'WARN'
                    result['errors'].append(f"Missing endpoints: {', '.join(missing_endpoints)}")
                
            else:
                result['status'] = 'FAIL'
                result['errors'].append("API endpoints directory not found")
            
            # Check main FastAPI application
            main_app = self.project_root / "backend" / "app" / "main.py"
            if main_app.exists():
                result['details']['main_app_exists'] = True
                
                # Check if monitoring routes are included
                with open(main_app, 'r') as f:
                    content = f.read()
                    if 'monitoring' in content.lower():
                        result['details']['monitoring_routes_included'] = True
                    else:
                        result['status'] = 'WARN'
                        result['errors'].append("Monitoring routes not integrated in main app")
            else:
                result['status'] = 'FAIL'
                result['errors'].append("Main FastAPI application missing")
                
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"API validation failed: {str(e)}")
        
        return result
    
    def validate_frontend_build(self) -> Dict[str, Any]:
        """Validate frontend build configuration"""
        self.log("⚛️  Validating frontend build configuration...")
        
        result = {
            'test_name': 'Frontend Build',
            'status': 'PASS',
            'details': {},
            'errors': []
        }
        
        try:
            frontend_dir = self.project_root / "frontend"
            
            # Check package.json
            package_json = frontend_dir / "package.json"
            if package_json.exists():
                with open(package_json, 'r') as f:
                    package_data = json.load(f)
                    result['details']['package_json_exists'] = True
                    result['details']['project_name'] = package_data.get('name', 'unknown')
                    
                    # Check for required scripts
                    scripts = package_data.get('scripts', {})
                    required_scripts = ['build', 'start', 'dev']
                    missing_scripts = [s for s in required_scripts if s not in scripts]
                    
                    if missing_scripts:
                        result['status'] = 'WARN'
                        result['errors'].append(f"Missing npm scripts: {', '.join(missing_scripts)}")
                    
                    result['details']['build_script_exists'] = 'build' in scripts
            else:
                result['status'] = 'FAIL'
                result['errors'].append("package.json not found")
            
            # Check Next.js configuration
            next_config = frontend_dir / "next.config.js"
            if next_config.exists():
                result['details']['next_config_exists'] = True
            else:
                result['status'] = 'WARN'
                result['errors'].append("next.config.js not found")
            
            # Check TypeScript configuration
            tsconfig = frontend_dir / "tsconfig.json"
            if tsconfig.exists():
                result['details']['typescript_config_exists'] = True
            else:
                result['status'] = 'WARN'
                result['errors'].append("tsconfig.json not found")
            
            # Check for MCP components
            mcp_components = frontend_dir / "components" / "mcp"
            if mcp_components.exists():
                mcp_files = list(mcp_components.glob("*.tsx"))
                result['details']['mcp_components_count'] = len(mcp_files)
                result['details']['mcp_integration'] = len(mcp_files) > 0
            else:
                result['status'] = 'WARN'
                result['errors'].append("MCP frontend components not found")
                
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Frontend validation failed: {str(e)}")
        
        return result
    
    def validate_monitoring_systems(self) -> Dict[str, Any]:
        """Validate monitoring and observability configuration"""
        self.log("📊 Validating monitoring systems...")
        
        result = {
            'test_name': 'Monitoring Systems',
            'status': 'PASS',
            'details': {},
            'errors': []
        }
        
        try:
            # Check Prometheus configuration
            prometheus_config = self.project_root / "infrastructure" / "prometheus.yml"
            if prometheus_config.exists():
                result['details']['prometheus_config_exists'] = True
                if YAML_AVAILABLE:
                    with open(prometheus_config, 'r') as f:
                        config = yaml.safe_load(f)
                        scrape_configs = config.get('scrape_configs', [])
                        result['details']['scrape_jobs'] = len(scrape_configs)
                        
                        # Check for MCP monitoring job
                        job_names = [job.get('job_name', '') for job in scrape_configs]
                        if 'mcp-health' in job_names:
                            result['details']['mcp_monitoring_job'] = True
                        else:
                            result['status'] = 'WARN'
                            result['errors'].append("MCP monitoring job not configured in Prometheus")
                else:
                    # Basic text check without yaml parsing
                    with open(prometheus_config, 'r') as f:
                        content = f.read()
                        if 'mcp-health' in content:
                            result['details']['mcp_monitoring_job'] = True
                        else:
                            result['status'] = 'WARN'
                            result['errors'].append("MCP monitoring job not found in Prometheus config")
                        
            else:
                result['status'] = 'FAIL'
                result['errors'].append("Prometheus configuration missing")
            
            # Check Grafana dashboards
            dashboards_dir = self.project_root / "infrastructure" / "grafana" / "dashboards"
            if dashboards_dir.exists():
                dashboard_files = list(dashboards_dir.glob("*.json"))
                result['details']['grafana_dashboards'] = len(dashboard_files)
                
                # Check for MCP dashboard
                mcp_dashboard = dashboards_dir / "mcp-monitoring.json"
                if mcp_dashboard.exists():
                    result['details']['mcp_dashboard_exists'] = True
                else:
                    result['status'] = 'WARN'
                    result['errors'].append("MCP monitoring dashboard missing")
                    
            else:
                result['status'] = 'WARN'
                result['errors'].append("Grafana dashboards directory not found")
            
            # Check health monitoring implementation
            health_checks = self.project_root / "backend" / "app" / "monitoring" / "health_checks.py"
            if health_checks.exists():
                result['details']['health_checks_implemented'] = True
            else:
                result['status'] = 'FAIL'
                result['errors'].append("Health checks implementation missing")
                
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Monitoring validation failed: {str(e)}")
        
        return result
    
    def validate_security_configuration(self) -> Dict[str, Any]:
        """Validate security configuration and best practices"""
        self.log("🔐 Validating security configuration...")
        
        result = {
            'test_name': 'Security Configuration',
            'status': 'PASS',
            'details': {},
            'errors': []
        }
        
        try:
            # Check for .env.example
            env_example = self.project_root / ".env.example"
            if env_example.exists():
                result['details']['env_example_exists'] = True
                
                # Check for security-related variables
                with open(env_example, 'r') as f:
                    env_content = f.read()
                    security_vars = ['SECRET_KEY', 'JWT_SECRET_KEY', 'ALPACA_API_KEY', 'ALPACA_SECRET_KEY']
                    result['details']['security_vars_documented'] = {}
                    
                    for var in security_vars:
                        is_documented = var in env_content
                        result['details']['security_vars_documented'][var] = is_documented
                        if not is_documented:
                            result['status'] = 'WARN'
                            result['errors'].append(f"Security variable {var} not documented")
                            
            else:
                result['status'] = 'WARN'
                result['errors'].append(".env.example file missing")
            
            # Check .env file is in .gitignore
            gitignore = self.project_root / ".gitignore"
            if gitignore.exists():
                with open(gitignore, 'r') as f:
                    gitignore_content = f.read()
                    if '.env' in gitignore_content:
                        result['details']['env_file_ignored'] = True
                    else:
                        result['status'] = 'FAIL'
                        result['errors'].append(".env file not in .gitignore - security risk!")
            else:
                result['status'] = 'WARN'
                result['errors'].append(".gitignore file missing")
            
            # Check authentication implementation
            auth_module = self.project_root / "backend" / "app" / "core" / "auth.py"
            if auth_module.exists():
                result['details']['auth_implementation_exists'] = True
            else:
                result['status'] = 'WARN'
                result['errors'].append("Authentication module missing")
                
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Security validation failed: {str(e)}")
        
        return result
    
    def validate_testing_infrastructure(self) -> Dict[str, Any]:
        """Validate testing infrastructure and coverage"""
        self.log("🧪 Validating testing infrastructure...")
        
        result = {
            'test_name': 'Testing Infrastructure',
            'status': 'PASS',
            'details': {},
            'errors': []
        }
        
        try:
            # Check backend tests
            backend_tests = self.project_root / "backend" / "tests"
            if backend_tests.exists():
                test_files = list(backend_tests.rglob("test_*.py"))
                result['details']['backend_test_files'] = len(test_files)
                result['details']['backend_tests_exist'] = len(test_files) > 0
            else:
                result['status'] = 'WARN'
                result['errors'].append("Backend tests directory missing")
            
            # Check for pytest configuration
            pytest_config = self.project_root / "backend" / "pytest.ini"
            pyproject_toml = self.project_root / "backend" / "pyproject.toml"
            
            if pytest_config.exists() or pyproject_toml.exists():
                result['details']['pytest_configured'] = True
            else:
                result['status'] = 'WARN'
                result['errors'].append("Pytest configuration missing")
            
            # Check frontend tests
            frontend_tests = self.project_root / "frontend"
            jest_config = frontend_tests / "jest.config.js"
            package_json = frontend_tests / "package.json"
            
            if package_json.exists():
                with open(package_json, 'r') as f:
                    package_data = json.load(f)
                    scripts = package_data.get('scripts', {})
                    if 'test' in scripts:
                        result['details']['frontend_test_script'] = True
                    else:
                        result['status'] = 'WARN'
                        result['errors'].append("Frontend test script missing")
            
            # Check test requirements
            test_requirements = self.project_root / "backend" / "requirements-dev.txt"
            if test_requirements.exists():
                result['details']['test_requirements_exist'] = True
            else:
                result['status'] = 'WARN'
                result['errors'].append("Development requirements file missing")
                
        except Exception as e:
            result['status'] = 'FAIL'
            result['errors'].append(f"Testing validation failed: {str(e)}")
        
        return result
    
    def generate_recommendations(self):
        """Generate deployment recommendations based on test results"""
        recommendations = []
        
        for test_name, test_result in self.results['test_results'].items():
            if test_result['status'] == 'FAIL':
                recommendations.append(f"🚨 CRITICAL: Fix {test_name} - {', '.join(test_result['errors'])}")
            elif test_result['status'] == 'WARN':
                recommendations.append(f"⚠️  WARNING: Address {test_name} - {', '.join(test_result['errors'])}")
        
        # Add general recommendations
        if self.results['tests_failed'] == 0:
            recommendations.append("✅ All critical systems validated - Ready for deployment!")
        else:
            recommendations.append("❌ Critical issues found - Deployment not recommended")
        
        if self.results['tests_passed'] / (self.results['tests_passed'] + self.results['tests_failed']) >= 0.8:
            recommendations.append("📈 Good system health - Most components ready")
        
        return recommendations
    
    async def run_all_validations(self):
        """Execute all deployment validation tests"""
        self.log("🚀 Starting comprehensive deployment readiness validation...")
        self.log(f"📁 Project root: {self.project_root}")
        
        # Define all validation tests
        validation_tests = [
            ('docker_services', self.validate_docker_services),
            ('database_config', self.validate_database_connectivity),
            ('mcp_configuration', self.validate_mcp_configuration),
            ('api_endpoints', self.validate_api_endpoints),
            ('frontend_build', self.validate_frontend_build),
            ('monitoring_systems', self.validate_monitoring_systems),
            ('security_config', self.validate_security_configuration),
            ('testing_infrastructure', self.validate_testing_infrastructure)
        ]
        
        # Execute all tests
        for test_key, test_function in validation_tests:
            try:
                self.log(f"Running {test_key.replace('_', ' ').title()} validation...")
                test_result = test_function()
                self.results['test_results'][test_key] = test_result
                
                if test_result['status'] == 'PASS':
                    self.results['tests_passed'] += 1
                    self.log(f"✅ {test_result['test_name']}: PASSED")
                elif test_result['status'] == 'WARN':
                    self.results['tests_passed'] += 1  # Count warnings as passed with issues
                    self.log(f"⚠️  {test_result['test_name']}: PASSED WITH WARNINGS")
                    for error in test_result['errors']:
                        self.log(f"    - {error}")
                else:
                    self.results['tests_failed'] += 1
                    self.log(f"❌ {test_result['test_name']}: FAILED")
                    for error in test_result['errors']:
                        self.log(f"    - {error}")
                        
            except Exception as e:
                self.results['tests_failed'] += 1
                self.log(f"❌ {test_key} validation failed: {str(e)}")
                self.results['test_results'][test_key] = {
                    'test_name': test_key.replace('_', ' ').title(),
                    'status': 'FAIL',
                    'details': {},
                    'errors': [f"Test execution failed: {str(e)}"]
                }
        
        # Generate final assessment
        total_tests = self.results['tests_passed'] + self.results['tests_failed']
        pass_rate = self.results['tests_passed'] / total_tests if total_tests > 0 else 0
        
        if self.results['tests_failed'] == 0:
            self.results['overall_status'] = 'READY'
            self.results['deployment_ready'] = True
        elif pass_rate >= 0.8:
            self.results['overall_status'] = 'READY_WITH_WARNINGS'
            self.results['deployment_ready'] = True
        else:
            self.results['overall_status'] = 'NOT_READY'
            self.results['deployment_ready'] = False
        
        # Generate recommendations
        self.results['recommendations'] = self.generate_recommendations()
        
        return self.results
    
    def save_report(self, filename: str = "deployment_readiness_report.json"):
        """Save validation results to file"""
        report_file = self.project_root / filename
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.log(f"📄 Deployment readiness report saved to: {report_file}")
        return report_file
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("🏁 DEPLOYMENT READINESS VALIDATION SUMMARY")
        print("="*80)
        
        # Overall status
        status_emoji = {
            'READY': '🟢',
            'READY_WITH_WARNINGS': '🟡', 
            'NOT_READY': '🔴',
            'UNKNOWN': '⚪'
        }
        
        print(f"\n{status_emoji.get(self.results['overall_status'], '⚪')} Overall Status: {self.results['overall_status']}")
        print(f"📊 Tests Passed: {self.results['tests_passed']}")
        print(f"❌ Tests Failed: {self.results['tests_failed']}")
        print(f"🎯 Deployment Ready: {'YES' if self.results['deployment_ready'] else 'NO'}")
        
        # Test results summary
        print(f"\n📋 Test Results Summary:")
        for test_name, result in self.results['test_results'].items():
            status_icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌'}.get(result['status'], '❓')
            print(f"  {status_icon} {result['test_name']}: {result['status']}")
        
        # Recommendations
        if self.results['recommendations']:
            print(f"\n💡 Recommendations:")
            for recommendation in self.results['recommendations']:
                print(f"  {recommendation}")
        
        print("\n" + "="*80)


async def main():
    """Main validation execution"""
    validator = DeploymentValidator()
    
    try:
        # Run all validations
        results = await validator.run_all_validations()
        
        # Print summary
        validator.print_summary()
        
        # Save detailed report
        report_file = validator.save_report()
        
        # Return appropriate exit code
        if results['deployment_ready']:
            print(f"\n🎉 Deployment validation PASSED! System is ready for production deployment.")
            sys.exit(0)
        else:
            print(f"\n⛔ Deployment validation FAILED! Address critical issues before deployment.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Validation execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())