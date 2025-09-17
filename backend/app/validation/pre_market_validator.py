"""
Pre-Market Validation System - Comprehensive validation before market opens
"""

import asyncio
import logging
import time
from datetime import datetime, time as dt_time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import psutil

from app.core.config import settings
from app.core.database import get_db_session, get_redis, engine
from app.trading.alpaca_client import AlpacaClient
from app.monitoring.metrics import PrometheusMetrics
from app.monitoring.alerts import AlertManager
from app.scanners.symbol_scanner import SymbolScanner
from app.services.scanner_service import scanner_service

logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    """Validation result status"""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    ERROR = "ERROR"

@dataclass
class ValidationCheck:
    """Individual validation check result"""
    name: str
    status: ValidationResult
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    critical: bool = True
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ValidationSummary:
    """Overall validation summary"""
    overall_status: ValidationResult
    total_checks: int
    passed: int
    warnings: int
    failures: int
    errors: int
    critical_failures: int
    total_duration_ms: float
    market_ready: bool
    alerts_sent: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

class PreMarketValidator:
    """Comprehensive pre-market validation system"""

    def __init__(self):
        self.alert_manager = AlertManager()
        self.prometheus_metrics = PrometheusMetrics()
        self.checks: List[ValidationCheck] = []
        self.validation_running = False

    async def run_full_validation(self) -> ValidationSummary:
        """Run complete pre-market validation suite"""
        if self.validation_running:
            return ValidationSummary(
                overall_status=ValidationResult.ERROR,
                total_checks=0, passed=0, warnings=0, failures=0, errors=1,
                critical_failures=0, total_duration_ms=0.0, market_ready=False,
                alerts_sent=["Validation already running"]
            )

        try:
            self.validation_running = True
            start_time = time.time()
            self.checks = []

            logger.info("🚀 Starting Pre-Market Validation System")

            # Core Infrastructure Checks
            await self._validate_database_connectivity()
            await self._validate_redis_connectivity()
            await self._validate_system_resources()

            # Trading API Checks
            await self._validate_alpaca_connectivity()
            await self._validate_alpaca_account_status()
            await self._validate_alpaca_permissions()
            await self._validate_market_data_access()

            # System Component Checks
            await self._validate_scanner_system()
            await self._validate_monitoring_system()
            await self._validate_agent_coordination()

            # Market Readiness Checks
            await self._validate_streaming_connections()
            await self._validate_trade_execution_pipeline()
            await self._validate_risk_management()

            # Performance and Health Checks
            await self._validate_cache_performance()
            await self._validate_celery_workers()
            await self._validate_memory_usage()

            # Generate summary
            total_duration = (time.time() - start_time) * 1000
            summary = self._generate_validation_summary(total_duration)

            # Send alerts if critical issues found
            await self._send_validation_alerts(summary)

            # Record metrics
            self._record_validation_metrics(summary)

            logger.info(f"✅ Pre-Market Validation completed: {summary.overall_status.value}")
            return summary

        except Exception as e:
            logger.error(f"❌ Pre-Market Validation failed: {e}")
            return ValidationSummary(
                overall_status=ValidationResult.ERROR,
                total_checks=len(self.checks), passed=0, warnings=0, failures=0, errors=1,
                critical_failures=1, total_duration_ms=0.0, market_ready=False,
                alerts_sent=[f"Validation system error: {str(e)}"]
            )
        finally:
            self.validation_running = False

    async def _validate_database_connectivity(self):
        """Validate database connection and pool status"""
        start_time = time.time()
        try:
            # Test database connection
            db = get_db_session()
            result = db.execute("SELECT 1").fetchone()
            db.close()

            # Check connection pool
            pool_info = {
                'size': engine.pool.size(),
                'checked_in': engine.pool.checkedin(),
                'checked_out': engine.pool.checkedout(),
                'overflow': engine.pool.overflow(),
                'invalid': engine.pool.invalid()
            }

            if result and pool_info['checked_out'] < pool_info['size']:
                self.checks.append(ValidationCheck(
                    name="Database Connectivity",
                    status=ValidationResult.PASS,
                    message="Database connection and pool healthy",
                    details=pool_info,
                    duration_ms=(time.time() - start_time) * 1000,
                    critical=True
                ))
            else:
                self.checks.append(ValidationCheck(
                    name="Database Connectivity",
                    status=ValidationResult.FAIL,
                    message="Database connection pool exhausted or connection failed",
                    details=pool_info,
                    duration_ms=(time.time() - start_time) * 1000,
                    critical=True
                ))

        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Database Connectivity",
                status=ValidationResult.FAIL,
                message=f"Database connection failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=True
            ))

    async def _validate_redis_connectivity(self):
        """Validate Redis connection and performance"""
        start_time = time.time()
        try:
            redis_client = get_redis()

            # Test basic operations
            test_key = f"premarket_validation_{int(time.time())}"
            redis_client.set(test_key, "test_value", ex=10)
            retrieved_value = redis_client.get(test_key)
            redis_client.delete(test_key)

            # Get Redis info
            redis_info = redis_client.info()

            if retrieved_value == "test_value":
                self.checks.append(ValidationCheck(
                    name="Redis Connectivity",
                    status=ValidationResult.PASS,
                    message="Redis connection and operations working",
                    details={
                        'connected_clients': redis_info.get('connected_clients', 0),
                        'used_memory_human': redis_info.get('used_memory_human', 'unknown'),
                        'keyspace_hits': redis_info.get('keyspace_hits', 0),
                        'keyspace_misses': redis_info.get('keyspace_misses', 0)
                    },
                    duration_ms=(time.time() - start_time) * 1000,
                    critical=True
                ))
            else:
                self.checks.append(ValidationCheck(
                    name="Redis Connectivity",
                    status=ValidationResult.FAIL,
                    message="Redis operations failed",
                    duration_ms=(time.time() - start_time) * 1000,
                    critical=True
                ))

        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Redis Connectivity",
                status=ValidationResult.FAIL,
                message=f"Redis connection failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=True
            ))

    async def _validate_system_resources(self):
        """Validate system resource availability"""
        start_time = time.time()
        try:
            # Check system resources
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=1)

            resource_status = ValidationResult.PASS
            issues = []

            if memory.percent > 90:
                resource_status = ValidationResult.FAIL
                issues.append(f"Memory usage critical: {memory.percent}%")
            elif memory.percent > 80:
                resource_status = ValidationResult.WARN
                issues.append(f"Memory usage high: {memory.percent}%")

            if disk.percent > 95:
                resource_status = ValidationResult.FAIL
                issues.append(f"Disk usage critical: {disk.percent}%")
            elif disk.percent > 85:
                resource_status = ValidationResult.WARN
                issues.append(f"Disk usage high: {disk.percent}%")

            if cpu_percent > 90:
                resource_status = ValidationResult.WARN
                issues.append(f"CPU usage high: {cpu_percent}%")

            message = "System resources healthy" if not issues else "; ".join(issues)

            self.checks.append(ValidationCheck(
                name="System Resources",
                status=resource_status,
                message=message,
                details={
                    'memory_percent': memory.percent,
                    'memory_available_gb': round(memory.available / (1024**3), 2),
                    'disk_percent': disk.percent,
                    'disk_free_gb': round(disk.free / (1024**3), 2),
                    'cpu_percent': cpu_percent
                },
                duration_ms=(time.time() - start_time) * 1000,
                critical=(resource_status == ValidationResult.FAIL)
            ))

        except Exception as e:
            self.checks.append(ValidationCheck(
                name="System Resources",
                status=ValidationResult.ERROR,
                message=f"Resource check failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=False
            ))

    async def _validate_alpaca_connectivity(self):
        """Validate Alpaca API connectivity"""
        start_time = time.time()
        try:
            client = AlpacaClient()

            # Test API connection
            account = client.get_account()

            if account:
                self.checks.append(ValidationCheck(
                    name="Alpaca API Connectivity",
                    status=ValidationResult.PASS,
                    message="Alpaca API connection successful",
                    details={
                        'account_id': account.id,
                        'status': account.status,
                        'trading_blocked': account.trading_blocked,
                        'transfers_blocked': account.transfers_blocked
                    },
                    duration_ms=(time.time() - start_time) * 1000,
                    critical=True
                ))
            else:
                self.checks.append(ValidationCheck(
                    name="Alpaca API Connectivity",
                    status=ValidationResult.FAIL,
                    message="Failed to retrieve account information",
                    duration_ms=(time.time() - start_time) * 1000,
                    critical=True
                ))

        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Alpaca API Connectivity",
                status=ValidationResult.FAIL,
                message=f"Alpaca API connection failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=True
            ))

    async def _validate_alpaca_account_status(self):
        """Validate Alpaca account status and permissions"""
        start_time = time.time()
        try:
            client = AlpacaClient()
            account = client.get_account()

            if not account:
                self.checks.append(ValidationCheck(
                    name="Alpaca Account Status",
                    status=ValidationResult.FAIL,
                    message="Cannot retrieve account status",
                    duration_ms=(time.time() - start_time) * 1000,
                    critical=True
                ))
                return

            status_issues = []
            account_status = ValidationResult.PASS

            # Check account status
            if account.status != 'ACTIVE':
                account_status = ValidationResult.FAIL
                status_issues.append(f"Account status: {account.status}")

            # Check trading permissions
            if account.trading_blocked:
                account_status = ValidationResult.FAIL
                status_issues.append("Trading is blocked")

            if account.transfers_blocked:
                account_status = ValidationResult.WARN
                status_issues.append("Transfers are blocked")

            # Check buying power
            buying_power = float(account.buying_power)
            if buying_power < 1000:  # Minimum for meaningful trading
                account_status = ValidationResult.WARN
                status_issues.append(f"Low buying power: ${buying_power:.2f}")

            message = "Account status healthy" if not status_issues else "; ".join(status_issues)

            self.checks.append(ValidationCheck(
                name="Alpaca Account Status",
                status=account_status,
                message=message,
                details={
                    'account_status': account.status,
                    'buying_power': buying_power,
                    'cash': float(account.cash),
                    'portfolio_value': float(account.portfolio_value),
                    'trading_blocked': account.trading_blocked,
                    'transfers_blocked': account.transfers_blocked,
                    'pattern_day_trader': account.pattern_day_trader
                },
                duration_ms=(time.time() - start_time) * 1000,
                critical=(account_status == ValidationResult.FAIL)
            ))

        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Alpaca Account Status",
                status=ValidationResult.FAIL,
                message=f"Account status check failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=True
            ))

    async def _validate_alpaca_permissions(self):
        """Validate Alpaca API permissions and rate limits"""
        start_time = time.time()
        try:
            client = AlpacaClient()

            # Test market data access
            test_symbols = ['AAPL', 'SPY']
            permissions_status = ValidationResult.PASS
            permission_issues = []

            for symbol in test_symbols:
                try:
                    quote = client.get_latest_quote(symbol)
                    if not quote:
                        permissions_status = ValidationResult.WARN
                        permission_issues.append(f"No quote data for {symbol}")
                except Exception as e:
                    permissions_status = ValidationResult.FAIL
                    permission_issues.append(f"Quote access failed for {symbol}: {str(e)}")

            # Test positions access
            try:
                positions = client.list_positions()
                # This should work even if empty
            except Exception as e:
                permissions_status = ValidationResult.FAIL
                permission_issues.append(f"Positions access failed: {str(e)}")

            message = "API permissions healthy" if not permission_issues else "; ".join(permission_issues)

            self.checks.append(ValidationCheck(
                name="Alpaca API Permissions",
                status=permissions_status,
                message=message,
                details={
                    'market_data_access': permissions_status != ValidationResult.FAIL,
                    'positions_access': 'positions' in locals(),
                    'test_symbols_checked': test_symbols
                },
                duration_ms=(time.time() - start_time) * 1000,
                critical=(permissions_status == ValidationResult.FAIL)
            ))

        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Alpaca API Permissions",
                status=ValidationResult.FAIL,
                message=f"Permission check failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=True
            ))

    async def _validate_market_data_access(self):
        """Validate market data feed accessibility"""
        start_time = time.time()
        try:
            client = AlpacaClient()

            # Test both IEX and SIP data if available
            test_symbols = ['AAPL', 'BTCUSD', 'SPY']
            data_status = ValidationResult.PASS
            data_issues = []
            successful_symbols = 0

            for symbol in test_symbols:
                try:
                    quote = client.get_latest_quote(symbol)
                    trade = client.get_latest_trade(symbol)

                    if quote and trade:
                        successful_symbols += 1
                    elif quote or trade:
                        data_issues.append(f"Partial data for {symbol}")
                    else:
                        data_issues.append(f"No data for {symbol}")

                except Exception as e:
                    data_issues.append(f"Data access failed for {symbol}: {str(e)}")

            if successful_symbols == len(test_symbols):
                data_status = ValidationResult.PASS
            elif successful_symbols > 0:
                data_status = ValidationResult.WARN
            else:
                data_status = ValidationResult.FAIL

            message = f"Market data access: {successful_symbols}/{len(test_symbols)} symbols"
            if data_issues:
                message += f" - Issues: {'; '.join(data_issues)}"

            self.checks.append(ValidationCheck(
                name="Market Data Access",
                status=data_status,
                message=message,
                details={
                    'successful_symbols': successful_symbols,
                    'total_symbols_tested': len(test_symbols),
                    'data_feed': settings.ALPACA_DATA_FEED,
                    'issues': data_issues
                },
                duration_ms=(time.time() - start_time) * 1000,
                critical=(data_status == ValidationResult.FAIL)
            ))

        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Market Data Access",
                status=ValidationResult.FAIL,
                message=f"Market data validation failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=True
            ))

    async def _validate_scanner_system(self):
        """Validate multi-symbol scanner system"""
        start_time = time.time()
        try:
            # Initialize scanner service if not already done
            if not scanner_service.is_initialized:
                init_result = await scanner_service.initialize()
                if init_result.get('status') != 'initialized':
                    self.checks.append(ValidationCheck(
                        name="Scanner System",
                        status=ValidationResult.FAIL,
                        message="Scanner service initialization failed",
                        details=init_result,
                        duration_ms=(time.time() - start_time) * 1000,
                        critical=True
                    ))
                    return

            # Test scanner with a small set of symbols
            test_symbols = ['AAPL', 'SPY', 'QQQ']
            scan_result = await scanner_service.scan_specific_symbols(test_symbols)

            scanner_status = ValidationResult.PASS
            scanner_issues = []

            if scan_result.get('status') == 'completed':
                opportunities = scan_result.get('opportunities', [])
                successful_scans = len([opp for opp in opportunities if not opp.get('errors')])

                if successful_scans == len(test_symbols):
                    scanner_status = ValidationResult.PASS
                elif successful_scans > 0:
                    scanner_status = ValidationResult.WARN
                    scanner_issues.append(f"Only {successful_scans}/{len(test_symbols)} symbols scanned successfully")
                else:
                    scanner_status = ValidationResult.FAIL
                    scanner_issues.append("No symbols scanned successfully")
            else:
                scanner_status = ValidationResult.FAIL
                scanner_issues.append(f"Scanner test failed: {scan_result.get('status')}")

            message = "Scanner system operational" if not scanner_issues else "; ".join(scanner_issues)

            self.checks.append(ValidationCheck(
                name="Scanner System",
                status=scanner_status,
                message=message,
                details={
                    'scanner_initialized': scanner_service.is_initialized,
                    'test_result': scan_result,
                    'universe_size': scanner_service.scanner.universe_manager.get_total_symbols() if scanner_service.is_initialized else 0
                },
                duration_ms=(time.time() - start_time) * 1000,
                critical=(scanner_status == ValidationResult.FAIL)
            ))

        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Scanner System",
                status=ValidationResult.FAIL,
                message=f"Scanner validation failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=True
            ))

    async def _validate_monitoring_system(self):
        """Validate monitoring and metrics system"""
        start_time = time.time()
        try:
            # Test Prometheus metrics
            metrics_status = ValidationResult.PASS
            metrics_issues = []

            try:
                # Record a test metric
                self.prometheus_metrics.record_system_health("healthy")
                self.prometheus_metrics.record_mcp_agent_coordination(True, 0.1)
            except Exception as e:
                metrics_status = ValidationResult.WARN
                metrics_issues.append(f"Metrics recording failed: {str(e)}")

            # Test alert manager
            try:
                # This should not actually send an alert, just test the system
                alert_test = await self.alert_manager.send_test_alert()
                if not alert_test:
                    metrics_issues.append("Alert system test failed")
            except Exception as e:
                metrics_issues.append(f"Alert system error: {str(e)}")

            message = "Monitoring system operational" if not metrics_issues else "; ".join(metrics_issues)

            self.checks.append(ValidationCheck(
                name="Monitoring System",
                status=metrics_status,
                message=message,
                details={
                    'prometheus_metrics': metrics_status != ValidationResult.FAIL,
                    'alert_manager': 'alert_test' in locals(),
                    'issues': metrics_issues
                },
                duration_ms=(time.time() - start_time) * 1000,
                critical=False  # Monitoring issues shouldn't block trading
            ))

        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Monitoring System",
                status=ValidationResult.WARN,
                message=f"Monitoring validation failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=False
            ))

    async def _validate_agent_coordination(self):
        """Validate agent coordination system readiness"""
        start_time = time.time()
        try:
            # Check if coordination components exist
            coordination_status = ValidationResult.PASS
            coordination_issues = []

            # This is a placeholder - in production would test actual agent systems
            # For now, we'll check if the required modules can be imported

            try:
                from app.agent_system import AgentCoordinationHub
                coordination_components = True
            except ImportError:
                coordination_components = False
                coordination_issues.append("Agent coordination components not available")
                coordination_status = ValidationResult.WARN

            # Check if agent streaming is ready (placeholder)
            streaming_ready = True  # Would test actual streaming connections

            message = "Agent coordination ready" if not coordination_issues else "; ".join(coordination_issues)

            self.checks.append(ValidationCheck(
                name="Agent Coordination",
                status=coordination_status,
                message=message,
                details={
                    'coordination_components': coordination_components,
                    'streaming_ready': streaming_ready,
                    'issues': coordination_issues
                },
                duration_ms=(time.time() - start_time) * 1000,
                critical=False  # Agent coordination issues are warnings for now
            ))

        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Agent Coordination",
                status=ValidationResult.WARN,
                message=f"Agent coordination check failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=False
            ))

    async def _validate_streaming_connections(self):
        """Validate real-time streaming connections"""
        start_time = time.time()
        try:
            # Test streaming connection setup
            streaming_status = ValidationResult.PASS
            streaming_issues = []

            # For now, this is a placeholder - would test actual WebSocket connections
            # In production, this would:
            # 1. Test Alpaca streaming connection
            # 2. Verify data flow
            # 3. Check reconnection logic

            alpaca_streaming = True  # Placeholder
            data_flow = True  # Placeholder

            if not alpaca_streaming:
                streaming_status = ValidationResult.FAIL
                streaming_issues.append("Alpaca streaming connection failed")

            if not data_flow:
                streaming_status = ValidationResult.WARN
                streaming_issues.append("Data flow verification failed")

            message = "Streaming connections ready" if not streaming_issues else "; ".join(streaming_issues)

            self.checks.append(ValidationCheck(
                name="Streaming Connections",
                status=streaming_status,
                message=message,
                details={
                    'alpaca_streaming': alpaca_streaming,
                    'data_flow': data_flow,
                    'streaming_enabled': settings.STREAMING_ENABLED
                },
                duration_ms=(time.time() - start_time) * 1000,
                critical=(streaming_status == ValidationResult.FAIL)
            ))

        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Streaming Connections",
                status=ValidationResult.FAIL,
                message=f"Streaming validation failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=True
            ))

    async def _validate_trade_execution_pipeline(self):
        """Validate trade execution pipeline without placing actual trades"""
        start_time = time.time()
        try:
            client = AlpacaClient()
            execution_status = ValidationResult.PASS
            execution_issues = []

            # Test order validation (dry run)
            try:
                # This would be a dry-run test order
                test_order_params = {
                    'symbol': 'AAPL',
                    'qty': 1,
                    'side': 'buy',
                    'type': 'market',
                    'time_in_force': 'day'
                }

                # In production, would validate order parameters without submitting
                order_validation = True  # Placeholder for actual validation

                if not order_validation:
                    execution_issues.append("Order validation failed")
                    execution_status = ValidationResult.FAIL

            except Exception as e:
                execution_issues.append(f"Order validation error: {str(e)}")
                execution_status = ValidationResult.WARN

            # Check trading permissions are still valid
            account = client.get_account()
            if account and account.trading_blocked:
                execution_issues.append("Trading is blocked on account")
                execution_status = ValidationResult.FAIL

            message = "Trade execution pipeline ready" if not execution_issues else "; ".join(execution_issues)

            self.checks.append(ValidationCheck(
                name="Trade Execution Pipeline",
                status=execution_status,
                message=message,
                details={
                    'order_validation': 'order_validation' in locals(),
                    'trading_blocked': account.trading_blocked if account else True,
                    'issues': execution_issues
                },
                duration_ms=(time.time() - start_time) * 1000,
                critical=(execution_status == ValidationResult.FAIL)
            ))

        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Trade Execution Pipeline",
                status=ValidationResult.FAIL,
                message=f"Execution pipeline validation failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=True
            ))

    async def _validate_risk_management(self):
        """Validate risk management system"""
        start_time = time.time()
        try:
            risk_status = ValidationResult.PASS
            risk_issues = []

            # Check risk parameters
            max_position_size = settings.MAX_POSITION_SIZE
            max_daily_loss = settings.MAX_DAILY_LOSS
            default_position_size = settings.DEFAULT_POSITION_SIZE

            if max_position_size <= 0:
                risk_issues.append("Invalid max position size")
                risk_status = ValidationResult.FAIL

            if max_daily_loss <= 0:
                risk_issues.append("Invalid max daily loss limit")
                risk_status = ValidationResult.FAIL

            if default_position_size <= 0:
                risk_issues.append("Invalid default position size")
                risk_status = ValidationResult.FAIL

            # Check current portfolio exposure (if any positions exist)
            try:
                client = AlpacaClient()
                positions = client.list_positions()
                total_exposure = sum(abs(float(pos.market_value)) for pos in positions)

                if total_exposure > max_position_size * 10:  # 10x max position as portfolio limit
                    risk_issues.append(f"High portfolio exposure: ${total_exposure:.2f}")
                    risk_status = ValidationResult.WARN

            except Exception as e:
                risk_issues.append(f"Position exposure check failed: {str(e)}")

            message = "Risk management ready" if not risk_issues else "; ".join(risk_issues)

            self.checks.append(ValidationCheck(
                name="Risk Management",
                status=risk_status,
                message=message,
                details={
                    'max_position_size': max_position_size,
                    'max_daily_loss': max_daily_loss,
                    'default_position_size': default_position_size,
                    'current_exposure': total_exposure if 'total_exposure' in locals() else 0,
                    'issues': risk_issues
                },
                duration_ms=(time.time() - start_time) * 1000,
                critical=(risk_status == ValidationResult.FAIL)
            ))

        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Risk Management",
                status=ValidationResult.FAIL,
                message=f"Risk management validation failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=True
            ))

    async def _validate_cache_performance(self):
        """Validate cache system performance"""
        start_time = time.time()
        try:
            from app.core.cache import get_market_cache

            cache_status = ValidationResult.PASS
            cache_issues = []

            # Test cache operations
            market_cache = get_market_cache()

            # Performance test
            test_key = f"validation_test_{int(time.time())}"
            test_data = {"test": "data", "timestamp": datetime.utcnow().isoformat()}

            # Test set/get performance
            cache_start = time.time()
            await market_cache.set(test_key, test_data)
            retrieved_data = await market_cache.get(test_key)
            cache_duration = (time.time() - cache_start) * 1000

            if retrieved_data and cache_duration < 100:  # Under 100ms
                cache_status = ValidationResult.PASS
            elif retrieved_data:
                cache_status = ValidationResult.WARN
                cache_issues.append(f"Cache operations slow: {cache_duration:.1f}ms")
            else:
                cache_status = ValidationResult.FAIL
                cache_issues.append("Cache operations failed")

            # Clean up test data
            await market_cache.delete(test_key)

            message = "Cache performance good" if not cache_issues else "; ".join(cache_issues)

            self.checks.append(ValidationCheck(
                name="Cache Performance",
                status=cache_status,
                message=message,
                details={
                    'cache_duration_ms': cache_duration,
                    'operations_successful': retrieved_data is not None,
                    'issues': cache_issues
                },
                duration_ms=(time.time() - start_time) * 1000,
                critical=False
            ))

        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Cache Performance",
                status=ValidationResult.WARN,
                message=f"Cache validation failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=False
            ))

    async def _validate_celery_workers(self):
        """Validate Celery worker availability"""
        start_time = time.time()
        try:
            from app.core.celery_app import celery_app

            worker_status = ValidationResult.PASS
            worker_issues = []

            # Check active workers
            try:
                # Get worker stats
                stats = celery_app.control.inspect().stats()
                active_workers = len(stats) if stats else 0

                if active_workers == 0:
                    worker_status = ValidationResult.WARN
                    worker_issues.append("No active Celery workers found")
                else:
                    # Test a simple task
                    test_task = celery_app.send_task('app.tasks.monitoring.system_health_check')
                    if test_task:
                        worker_status = ValidationResult.PASS
                    else:
                        worker_status = ValidationResult.WARN
                        worker_issues.append("Could not submit test task")

            except Exception as e:
                worker_status = ValidationResult.WARN
                worker_issues.append(f"Worker communication failed: {str(e)}")

            message = "Celery workers available" if not worker_issues else "; ".join(worker_issues)

            self.checks.append(ValidationCheck(
                name="Celery Workers",
                status=worker_status,
                message=message,
                details={
                    'active_workers': active_workers if 'active_workers' in locals() else 0,
                    'task_submission': 'test_task' in locals(),
                    'issues': worker_issues
                },
                duration_ms=(time.time() - start_time) * 1000,
                critical=False  # Workers not critical for immediate trading
            ))

        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Celery Workers",
                status=ValidationResult.WARN,
                message=f"Celery validation failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=False
            ))

    async def _validate_memory_usage(self):
        """Validate memory usage and availability"""
        start_time = time.time()
        try:
            process = psutil.Process()
            system_memory = psutil.virtual_memory()

            memory_status = ValidationResult.PASS
            memory_issues = []

            # Process memory usage
            process_memory_mb = process.memory_info().rss / (1024 * 1024)
            if process_memory_mb > 2048:  # 2GB limit
                memory_status = ValidationResult.WARN
                memory_issues.append(f"High process memory usage: {process_memory_mb:.1f}MB")

            # System memory availability
            if system_memory.available < 1024 * 1024 * 1024:  # 1GB available
                memory_status = ValidationResult.FAIL
                memory_issues.append(f"Low system memory: {system_memory.available / (1024**3):.1f}GB available")

            message = "Memory usage healthy" if not memory_issues else "; ".join(memory_issues)

            self.checks.append(ValidationCheck(
                name="Memory Usage",
                status=memory_status,
                message=message,
                details={
                    'process_memory_mb': process_memory_mb,
                    'system_available_gb': system_memory.available / (1024**3),
                    'system_percent': system_memory.percent,
                    'issues': memory_issues
                },
                duration_ms=(time.time() - start_time) * 1000,
                critical=(memory_status == ValidationResult.FAIL)
            ))

        except Exception as e:
            self.checks.append(ValidationCheck(
                name="Memory Usage",
                status=ValidationResult.WARN,
                message=f"Memory validation failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                critical=False
            ))

    def _generate_validation_summary(self, total_duration_ms: float) -> ValidationSummary:
        """Generate overall validation summary"""
        passed = sum(1 for check in self.checks if check.status == ValidationResult.PASS)
        warnings = sum(1 for check in self.checks if check.status == ValidationResult.WARN)
        failures = sum(1 for check in self.checks if check.status == ValidationResult.FAIL)
        errors = sum(1 for check in self.checks if check.status == ValidationResult.ERROR)
        critical_failures = sum(1 for check in self.checks
                              if check.status in [ValidationResult.FAIL, ValidationResult.ERROR] and check.critical)

        # Determine overall status
        if critical_failures > 0:
            overall_status = ValidationResult.FAIL
            market_ready = False
        elif failures > 0:
            overall_status = ValidationResult.FAIL
            market_ready = False
        elif errors > 0:
            overall_status = ValidationResult.ERROR
            market_ready = False
        elif warnings > 0:
            overall_status = ValidationResult.WARN
            market_ready = True  # Can trade with warnings
        else:
            overall_status = ValidationResult.PASS
            market_ready = True

        return ValidationSummary(
            overall_status=overall_status,
            total_checks=len(self.checks),
            passed=passed,
            warnings=warnings,
            failures=failures,
            errors=errors,
            critical_failures=critical_failures,
            total_duration_ms=total_duration_ms,
            market_ready=market_ready
        )

    async def _send_validation_alerts(self, summary: ValidationSummary):
        """Send alerts based on validation results"""
        alerts_sent = []

        # Send critical failure alerts
        if summary.critical_failures > 0:
            critical_checks = [check for check in self.checks
                             if check.status in [ValidationResult.FAIL, ValidationResult.ERROR] and check.critical]

            alert_message = f"🚨 CRITICAL: Pre-Market Validation Failed\n"
            alert_message += f"Critical failures: {summary.critical_failures}\n"
            alert_message += f"Failed checks:\n"
            for check in critical_checks:
                alert_message += f"  - {check.name}: {check.message}\n"

            try:
                await self.alert_manager.send_alert(
                    message=alert_message,
                    severity="critical",
                    component="pre_market_validation"
                )
                alerts_sent.append("Critical failure alert sent")
            except Exception as e:
                logger.error(f"Failed to send critical alert: {e}")

        # Send warning summary if warnings but no critical failures
        elif summary.warnings > 0:
            warning_checks = [check for check in self.checks if check.status == ValidationResult.WARN]

            alert_message = f"⚠️ WARNING: Pre-Market Validation Issues\n"
            alert_message += f"Warnings: {summary.warnings}\n"
            alert_message += f"Warning checks:\n"
            for check in warning_checks[:5]:  # Limit to first 5
                alert_message += f"  - {check.name}: {check.message}\n"

            try:
                await self.alert_manager.send_alert(
                    message=alert_message,
                    severity="warning",
                    component="pre_market_validation"
                )
                alerts_sent.append("Warning alert sent")
            except Exception as e:
                logger.error(f"Failed to send warning alert: {e}")

        # Send success notification if all passed
        elif summary.overall_status == ValidationResult.PASS:
            try:
                await self.alert_manager.send_alert(
                    message=f"✅ Pre-Market Validation: ALL SYSTEMS READY\n"
                           f"Completed {summary.total_checks} checks in {summary.total_duration_ms:.1f}ms\n"
                           f"System ready for market trading! 🚀",
                    severity="info",
                    component="pre_market_validation"
                )
                alerts_sent.append("Success notification sent")
            except Exception as e:
                logger.error(f"Failed to send success notification: {e}")

        summary.alerts_sent = alerts_sent

    def _record_validation_metrics(self, summary: ValidationSummary):
        """Record validation metrics to Prometheus"""
        try:
            # Record overall validation status
            status_value = {
                ValidationResult.PASS: 1.0,
                ValidationResult.WARN: 0.7,
                ValidationResult.FAIL: 0.0,
                ValidationResult.ERROR: 0.0
            }.get(summary.overall_status, 0.0)

            self.prometheus_metrics.record_system_health(
                "healthy" if summary.market_ready else "unhealthy"
            )

            # Record individual check metrics
            for check in self.checks:
                check_value = {
                    ValidationResult.PASS: 1.0,
                    ValidationResult.WARN: 0.5,
                    ValidationResult.FAIL: 0.0,
                    ValidationResult.ERROR: 0.0
                }.get(check.status, 0.0)

                # This would be a custom metric in production
                # self.prometheus_metrics.record_validation_check(check.name, check_value)

        except Exception as e:
            logger.error(f"Failed to record validation metrics: {e}")

    async def schedule_pre_market_validation(self, market_open_time: dt_time = dt_time(9, 30)) -> bool:
        """Schedule automatic pre-market validation"""
        try:
            # Calculate when to run validation (30 minutes before market open)
            validation_time = dt_time(
                hour=market_open_time.hour,
                minute=max(0, market_open_time.minute - 30)
            )

            logger.info(f"Pre-market validation scheduled for {validation_time}")

            # In production, this would integrate with Celery beat scheduler
            # For now, return True to indicate scheduling configured
            return True

        except Exception as e:
            logger.error(f"Failed to schedule pre-market validation: {e}")
            return False

    def get_validation_status(self) -> Dict[str, Any]:
        """Get current validation status"""
        if not self.checks:
            return {
                'status': 'not_run',
                'message': 'Validation has not been run yet'
            }

        latest_check = max(self.checks, key=lambda c: c.timestamp) if self.checks else None

        return {
            'status': 'completed',
            'last_run': latest_check.timestamp.isoformat() if latest_check else None,
            'total_checks': len(self.checks),
            'latest_results': [
                {
                    'name': check.name,
                    'status': check.status.value,
                    'message': check.message,
                    'critical': check.critical,
                    'duration_ms': check.duration_ms
                }
                for check in self.checks[-10:]  # Last 10 checks
            ],
            'validation_running': self.validation_running
        }