"""
Prometheus metrics collection for system monitoring.
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
from prometheus_client.core import REGISTRY

from app.core.logging import get_logger
from .health_checks import HealthChecker, HealthStatus, SystemHealthStatus

logger = get_logger(__name__)


@dataclass
class MetricConfig:
    """Configuration for a metric"""
    name: str
    description: str
    labels: list = None


class PrometheusMetrics:
    """Prometheus metrics for the trading system"""
    
    def __init__(self, registry: CollectorRegistry = None):
        self.registry = registry or REGISTRY
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Initialize all Prometheus metrics"""
        
        # System Health Metrics
        self.system_health_status = Gauge(
            'trading_system_health_status',
            'Overall system health status (0=critical, 1=degraded, 2=healthy)',
            registry=self.registry
        )
        
        self.component_health_status = Gauge(
            'trading_component_health_status',
            'Individual component health status',
            ['component', 'component_type'],
            registry=self.registry
        )
        
        self.system_uptime_seconds = Gauge(
            'trading_system_uptime_seconds',
            'System uptime in seconds',
            registry=self.registry
        )
        
        # MCP Metrics
        self.mcp_server_status = Gauge(
            'mcp_server_status',
            'MCP server availability status',
            ['server_name', 'server_type'],
            registry=self.registry
        )
        
        self.mcp_request_duration = Histogram(
            'mcp_request_duration_seconds',
            'MCP request duration in seconds',
            ['server_name', 'method'],
            registry=self.registry
        )
        
        self.mcp_request_total = Counter(
            'mcp_request_total',
            'Total MCP requests',
            ['server_name', 'method', 'status'],
            registry=self.registry
        )
        
        self.mcp_errors_total = Counter(
            'mcp_errors_total',
            'Total MCP errors',
            ['server_name', 'error_type'],
            registry=self.registry
        )
        
        # Trading System Metrics
        self.trading_orders_total = Counter(
            'trading_orders_total',
            'Total trading orders',
            ['symbol', 'side', 'status'],
            registry=self.registry
        )
        
        self.trading_positions_active = Gauge(
            'trading_positions_active',
            'Currently active trading positions',
            ['symbol'],
            registry=self.registry
        )
        
        self.trading_portfolio_value = Gauge(
            'trading_portfolio_value_usd',
            'Portfolio value in USD',
            registry=self.registry
        )
        
        self.trading_pnl_total = Gauge(
            'trading_pnl_total_usd',
            'Total profit/loss in USD',
            ['symbol'],
            registry=self.registry
        )
        
        # Database Metrics
        self.db_connection_pool_size = Gauge(
            'db_connection_pool_size',
            'Database connection pool size',
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'db_query_duration_seconds',
            'Database query duration in seconds',
            ['query_type'],
            registry=self.registry
        )
        
        # Redis Metrics
        self.redis_operations_total = Counter(
            'redis_operations_total',
            'Total Redis operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.redis_response_time = Histogram(
            'redis_response_time_seconds',
            'Redis operation response time',
            ['operation'],
            registry=self.registry
        )
        
        # Market Research Metrics
        self.market_research_requests_total = Counter(
            'market_research_requests_total',
            'Total market research requests',
            ['symbol', 'analysis_type', 'status'],
            registry=self.registry
        )
        
        self.market_sentiment_score = Gauge(
            'market_sentiment_score',
            'Market sentiment score (-1 to 1)',
            ['symbol', 'source'],
            registry=self.registry
        )
        
        # AI Insights Metrics
        self.ai_insights_generated_total = Counter(
            'ai_insights_generated_total',
            'Total AI insights generated',
            ['insight_type', 'status'],
            registry=self.registry
        )
        
        self.ai_processing_duration = Histogram(
            'ai_processing_duration_seconds',
            'AI processing duration in seconds',
            ['process_type'],
            registry=self.registry
        )
        
        # System Resource Metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # System Info
        self.system_info = Info(
            'trading_system_info',
            'Trading system information',
            registry=self.registry
        )
    
    def update_health_metrics(self, health_status: SystemHealthStatus):
        """Update health-related metrics"""
        
        # Overall system health
        health_value = {
            HealthStatus.CRITICAL: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.HEALTHY: 2
        }.get(health_status.overall_status, 0)
        
        self.system_health_status.set(health_value)
        self.system_uptime_seconds.set(health_status.uptime_seconds)
        
        # Component health
        for component in health_status.components:
            component_value = {
                HealthStatus.CRITICAL: 0,
                HealthStatus.DEGRADED: 1,
                HealthStatus.HEALTHY: 2,
                HealthStatus.UNKNOWN: 0
            }.get(component.status, 0)
            
            self.component_health_status.labels(
                component=component.component,
                component_type=component.component_type.value
            ).set(component_value)
    
    def record_mcp_request(self, server_name: str, method: str, duration: float, status: str):
        """Record MCP request metrics"""
        self.mcp_request_duration.labels(
            server_name=server_name,
            method=method
        ).observe(duration)
        
        self.mcp_request_total.labels(
            server_name=server_name,
            method=method,
            status=status
        ).inc()
    
    def record_mcp_error(self, server_name: str, error_type: str):
        """Record MCP error"""
        self.mcp_errors_total.labels(
            server_name=server_name,
            error_type=error_type
        ).inc()
    
    def update_mcp_server_status(self, server_name: str, server_type: str, available: bool):
        """Update MCP server status"""
        self.mcp_server_status.labels(
            server_name=server_name,
            server_type=server_type
        ).set(1 if available else 0)
    
    def record_trading_order(self, symbol: str, side: str, status: str):
        """Record trading order"""
        self.trading_orders_total.labels(
            symbol=symbol,
            side=side,
            status=status
        ).inc()
    
    def update_portfolio_metrics(self, portfolio_value: float, positions: Dict[str, float]):
        """Update portfolio metrics"""
        self.trading_portfolio_value.set(portfolio_value)
        
        for symbol, quantity in positions.items():
            self.trading_positions_active.labels(symbol=symbol).set(abs(quantity))
    
    def record_market_research(self, symbol: str, analysis_type: str, status: str):
        """Record market research request"""
        self.market_research_requests_total.labels(
            symbol=symbol,
            analysis_type=analysis_type,
            status=status
        ).inc()
    
    def update_sentiment_score(self, symbol: str, source: str, score: float):
        """Update market sentiment score"""
        self.market_sentiment_score.labels(
            symbol=symbol,
            source=source
        ).set(score)
    
    def record_ai_insight(self, insight_type: str, status: str, processing_time: float):
        """Record AI insight generation"""
        self.ai_insights_generated_total.labels(
            insight_type=insight_type,
            status=status
        ).inc()
        
        self.ai_processing_duration.labels(
            process_type=insight_type
        ).observe(processing_time)
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()
        
        self.http_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def set_system_info(self, version: str, environment: str, build_date: str):
        """Set system information"""
        self.system_info.info({
            'version': version,
            'environment': environment,
            'build_date': build_date,
            'system': 'swaggy-stacks-trading'
        })
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        return generate_latest(self.registry)


class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(self):
        self.prometheus_metrics = PrometheusMetrics()
        self.health_checker = HealthChecker()
        self._last_update = 0
        self._update_interval = 30  # Update every 30 seconds
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        
        # Check if we need to update (rate limiting)
        current_time = time.time()
        if current_time - self._last_update < self._update_interval:
            return await self.get_cached_metrics()
        
        try:
            # Get system health
            health_status = await self.health_checker.check_all_components()
            
            # Update Prometheus metrics
            self.prometheus_metrics.update_health_metrics(health_status)
            
            # Collect additional metrics
            metrics = {
                'health_status': health_status,
                'system_metrics': await self.health_checker.get_system_metrics(),
                'timestamp': current_time
            }
            
            self._last_update = current_time
            self._cached_metrics = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {
                'error': str(e),
                'timestamp': current_time
            }
    
    async def get_cached_metrics(self) -> Dict[str, Any]:
        """Get cached metrics to avoid frequent collection"""
        if hasattr(self, '_cached_metrics'):
            return self._cached_metrics
        return await self.collect_system_metrics()
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        return self.prometheus_metrics.get_metrics()
    
    async def update_mcp_metrics(self, server_statuses: Dict[str, bool]):
        """Update MCP server metrics"""
        for server_name, available in server_statuses.items():
            self.prometheus_metrics.update_mcp_server_status(
                server_name=server_name,
                server_type='mcp',
                available=available
            )
    
    def record_request_metrics(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        self.prometheus_metrics.record_http_request(method, endpoint, status_code, duration)