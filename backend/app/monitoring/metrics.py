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
        
        # MCP Agent Coordination Metrics
        self.mcp_agent_coordination_duration = Histogram(
            'mcp_agent_coordination_duration_seconds',
            'Duration of MCP agent coordination operations',
            ['coordination_type', 'source_agent', 'target_agent'],
            registry=self.registry
        )
        
        self.mcp_agent_queue_depth = Gauge(
            'mcp_agent_queue_depth',
            'Current queue depth for MCP agent operations',
            ['agent_name', 'queue_type'],
            registry=self.registry
        )
        
        self.mcp_cross_agent_requests_total = Counter(
            'mcp_cross_agent_requests_total',
            'Total cross-agent requests between MCP agents',
            ['source_agent', 'target_agent', 'request_type', 'status'],
            registry=self.registry
        )
        
        self.mcp_agent_coordination_success_rate = Gauge(
            'mcp_agent_coordination_success_rate',
            'Success rate of agent coordination operations (0-1)',
            ['coordination_type', 'agent_pair'],
            registry=self.registry
        )
        
        self.mcp_agent_response_time = Histogram(
            'mcp_agent_response_time_seconds',
            'Response time for MCP agent operations',
            ['agent_name', 'operation_type'],
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

        # CUSTOM TRADING STRATEGY PERFORMANCE METRICS - Task 1.1
        
        # Strategy Performance Metrics
        self.strategy_win_loss_ratio = Gauge(
            'trading_strategy_win_loss_ratio',
            'Win/loss ratio for trading strategies',
            ['strategy_name', 'symbol', 'timeframe'],
            registry=self.registry
        )
        
        self.strategy_average_profit_loss = Gauge(
            'trading_strategy_avg_profit_loss_usd',
            'Average profit/loss per trade in USD',
            ['strategy_name', 'symbol', 'trade_type'],
            registry=self.registry
        )
        
        self.strategy_total_trades = Counter(
            'trading_strategy_total_trades',
            'Total trades executed by strategy',
            ['strategy_name', 'symbol', 'outcome'],
            registry=self.registry
        )
        
        self.strategy_success_rate = Gauge(
            'trading_strategy_success_rate',
            'Strategy success rate (0-1)',
            ['strategy_name', 'symbol'],
            registry=self.registry
        )
        
        self.strategy_drawdown_current = Gauge(
            'trading_strategy_drawdown_current_pct',
            'Current drawdown percentage for strategy',
            ['strategy_name'],
            registry=self.registry
        )
        
        self.strategy_drawdown_max = Gauge(
            'trading_strategy_drawdown_max_pct',
            'Maximum drawdown percentage for strategy',
            ['strategy_name'],
            registry=self.registry
        )
        
        # Trade Execution Success/Failure Counters
        self.trade_execution_total = Counter(
            'trading_execution_total',
            'Total trade execution attempts',
            ['symbol', 'side', 'order_type', 'status'],
            registry=self.registry
        )
        
        self.trade_execution_failures = Counter(
            'trading_execution_failures_total',
            'Total trade execution failures',
            ['symbol', 'side', 'failure_reason', 'error_type'],
            registry=self.registry
        )
        
        self.trade_execution_success_rate = Gauge(
            'trading_execution_success_rate',
            'Trade execution success rate (0-1)',
            ['symbol', 'order_type'],
            registry=self.registry
        )
        
        # Portfolio Exposure and Risk Metrics Gauges
        self.portfolio_exposure_total = Gauge(
            'trading_portfolio_exposure_total_usd',
            'Total portfolio exposure in USD',
            registry=self.registry
        )
        
        self.portfolio_exposure_by_sector = Gauge(
            'trading_portfolio_exposure_by_sector_usd',
            'Portfolio exposure by sector in USD',
            ['sector'],
            registry=self.registry
        )
        
        self.portfolio_concentration_risk = Gauge(
            'trading_portfolio_concentration_risk',
            'Portfolio concentration risk score (0-1)',
            registry=self.registry
        )
        
        self.portfolio_var_daily = Gauge(
            'trading_portfolio_var_daily_usd',
            'Daily Value at Risk in USD',
            ['confidence_level'],
            registry=self.registry
        )
        
        self.portfolio_beta = Gauge(
            'trading_portfolio_beta',
            'Portfolio beta relative to market',
            ['benchmark'],
            registry=self.registry
        )
        
        self.position_size_risk = Gauge(
            'trading_position_size_risk_pct',
            'Position size as percentage of portfolio',
            ['symbol'],
            registry=self.registry
        )
        
        # Latency Metrics for Critical Trading Operations
        self.trade_execution_latency = Histogram(
            'trading_execution_latency_seconds',
            'Trade execution latency in seconds',
            ['operation_type', 'symbol', 'broker'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.market_data_latency = Histogram(
            'trading_market_data_latency_seconds',
            'Market data retrieval latency in seconds',
            ['data_type', 'symbol', 'source'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        self.strategy_analysis_latency = Histogram(
            'trading_strategy_analysis_latency_seconds',
            'Strategy analysis computation latency in seconds',
            ['strategy_name', 'analysis_type', 'symbol'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        self.order_book_latency = Histogram(
            'trading_order_book_latency_seconds',
            'Order book update latency in seconds',
            ['symbol', 'exchange'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            registry=self.registry
        )
        
        self.risk_check_latency = Histogram(
            'trading_risk_check_latency_seconds',
            'Risk management check latency in seconds',
            ['check_type', 'symbol'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
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

    def record_mcp_agent_coordination(self, coordination_type: str, source_agent: str, 
                                     target_agent: str, duration: float, success: bool):
        """Record MCP agent coordination metrics"""
        # Record coordination duration
        self.mcp_agent_coordination_duration.labels(
            coordination_type=coordination_type,
            source_agent=source_agent,
            target_agent=target_agent
        ).observe(duration)
        
        # Record cross-agent request
        status = "success" if success else "failed"
        self.mcp_cross_agent_requests_total.labels(
            source_agent=source_agent,
            target_agent=target_agent,
            request_type=coordination_type,
            status=status
        ).inc()
        
        # Update success rate (simplified calculation for demo)
        agent_pair = f"{source_agent}-{target_agent}"
        current_rate = 1.0 if success else 0.0
        self.mcp_agent_coordination_success_rate.labels(
            coordination_type=coordination_type,
            agent_pair=agent_pair
        ).set(current_rate)
    
    def record_mcp_agent_response_time(self, agent_name: str, operation_type: str, duration: float):
        """Record MCP agent response time"""
        self.mcp_agent_response_time.labels(
            agent_name=agent_name,
            operation_type=operation_type
        ).observe(duration)
    
    def update_mcp_agent_queue_depth(self, agent_name: str, queue_type: str, depth: int):
        """Update MCP agent queue depth"""
        self.mcp_agent_queue_depth.labels(
            agent_name=agent_name,
            queue_type=queue_type
        ).set(depth)
    
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

    
    # CUSTOM TRADING STRATEGY PERFORMANCE METHODS - Task 1.1 Implementation
    
    def record_strategy_performance(self, strategy_name: str, symbol: str, 
                                  win_loss_ratio: float, avg_profit: float, avg_loss: float, 
                                  total_trades: int, successful_trades: int, timeframe: str = "1D"):
        """Record comprehensive strategy performance metrics"""
        
        # Win/loss ratio
        self.strategy_win_loss_ratio.labels(
            strategy_name=strategy_name, 
            symbol=symbol, 
            timeframe=timeframe
        ).set(win_loss_ratio)
        
        # Average profit/loss
        self.strategy_average_profit_loss.labels(
            strategy_name=strategy_name, 
            symbol=symbol, 
            trade_type="profit"
        ).set(avg_profit)
        
        self.strategy_average_profit_loss.labels(
            strategy_name=strategy_name, 
            symbol=symbol, 
            trade_type="loss"
        ).set(avg_loss)
        
        # Success rate calculation
        success_rate = successful_trades / total_trades if total_trades > 0 else 0
        self.strategy_success_rate.labels(
            strategy_name=strategy_name, 
            symbol=symbol
        ).set(success_rate)
    
    def record_strategy_trade_outcome(self, strategy_name: str, symbol: str, outcome: str):
        """Record individual trade outcome for strategy"""
        self.strategy_total_trades.labels(
            strategy_name=strategy_name,
            symbol=symbol,
            outcome=outcome  # "win", "loss", "breakeven"
        ).inc()
    
    def update_strategy_drawdown(self, strategy_name: str, current_drawdown_pct: float, 
                               max_drawdown_pct: float):
        """Update strategy drawdown metrics"""
        self.strategy_drawdown_current.labels(strategy_name=strategy_name).set(current_drawdown_pct)
        self.strategy_drawdown_max.labels(strategy_name=strategy_name).set(max_drawdown_pct)
    
    def record_trade_execution_metrics(self, symbol: str, side: str, order_type: str, 
                                     status: str, execution_time: float, failure_reason: str = None):
        """Record trade execution performance and latency"""
        
        # Record execution attempt
        self.trade_execution_total.labels(
            symbol=symbol,
            side=side,
            order_type=order_type,
            status=status
        ).inc()
        
        # Record execution latency
        self.trade_execution_latency.labels(
            operation_type="order_placement",
            symbol=symbol,
            broker="alpaca"
        ).observe(execution_time)
        
        # Record failures if applicable
        if status == "failed" and failure_reason:
            self.trade_execution_failures.labels(
                symbol=symbol,
                side=side,
                failure_reason=failure_reason,
                error_type="execution_error"
            ).inc()
        
        # Update success rate (simplified calculation)
        if status == "filled":
            success_rate = 0.95  # This would be calculated from historical data
            self.trade_execution_success_rate.labels(
                symbol=symbol,
                order_type=order_type
            ).set(success_rate)
    
    def update_portfolio_risk_metrics(self, total_exposure: float, sector_exposures: Dict[str, float],
                                    concentration_risk: float, var_daily: float, beta: float,
                                    position_risks: Dict[str, float], benchmark: str = "SPY"):
        """Update comprehensive portfolio risk metrics"""
        
        # Total exposure
        self.portfolio_exposure_total.set(total_exposure)
        
        # Sector exposures
        for sector, exposure in sector_exposures.items():
            self.portfolio_exposure_by_sector.labels(sector=sector).set(exposure)
        
        # Risk metrics
        self.portfolio_concentration_risk.set(concentration_risk)
        self.portfolio_var_daily.labels(confidence_level="95").set(var_daily)
        self.portfolio_beta.labels(benchmark=benchmark).set(beta)
        
        # Position-specific risks
        for symbol, risk_pct in position_risks.items():
            self.position_size_risk.labels(symbol=symbol).set(risk_pct)
    
    def record_market_data_latency(self, data_type: str, symbol: str, source: str, latency: float):
        """Record market data retrieval latency"""
        self.market_data_latency.labels(
            data_type=data_type,  # "quote", "bars", "trades"
            symbol=symbol,
            source=source  # "alpaca", "polygon", "yahoo"
        ).observe(latency)
    
    def record_strategy_analysis_latency(self, strategy_name: str, analysis_type: str, 
                                       symbol: str, duration: float):
        """Record strategy analysis computation time"""
        self.strategy_analysis_latency.labels(
            strategy_name=strategy_name,
            analysis_type=analysis_type,  # "markov", "wyckoff", "fibonacci"
            symbol=symbol
        ).observe(duration)
    
    def record_order_book_latency(self, symbol: str, exchange: str, latency: float):
        """Record order book update latency"""
        self.order_book_latency.labels(
            symbol=symbol,
            exchange=exchange
        ).observe(latency)
    
    def record_risk_check_latency(self, check_type: str, symbol: str, duration: float):
        """Record risk management check latency"""
        self.risk_check_latency.labels(
            check_type=check_type,  # "position_size", "exposure", "drawdown"
            symbol=symbol
        ).observe(duration)
    
    def collect_trading_manager_metrics(self, trading_manager):
        """Collect metrics from TradingManager instance"""
        try:
            # Portfolio value and positions
            if trading_manager.performance_metrics:
                portfolio_value = trading_manager.performance_metrics.get("total_equity", 0)
                self.trading_portfolio_value.set(portfolio_value)
                
                daily_pnl = trading_manager.performance_metrics.get("daily_pnl", 0)
                self.trading_pnl_total.labels(symbol="PORTFOLIO").set(daily_pnl)
            
            # Active positions
            for symbol, position in trading_manager.active_positions.items():
                quantity = float(position.get("quantity", 0))
                self.trading_positions_active.labels(symbol=symbol).set(abs(quantity))
                
                unrealized_pnl = float(position.get("unrealized_pnl", 0))
                self.trading_pnl_total.labels(symbol=symbol).set(unrealized_pnl)
            
        except Exception as e:
            logger.warning("Failed to collect trading manager metrics", error=str(e))
    
    def collect_strategy_agent_metrics(self, strategy_agent, symbol: str):
        """Collect metrics from StrategyAgent instance"""
        try:
            # Get strategy performance if available
            if hasattr(strategy_agent, 'strategies'):
                for strategy_name in strategy_agent.strategies:
                    # Record strategy usage
                    self.strategy_total_trades.labels(
                        strategy_name=strategy_name,
                        symbol=symbol,
                        outcome="analysis"
                    ).inc()
            
            # Record analysis type
            if hasattr(strategy_agent, 'consensus_method'):
                consensus_method = strategy_agent.consensus_method
                logger.debug(f"Using consensus method: {consensus_method}")
                
        except Exception as e:
            logger.warning("Failed to collect strategy agent metrics", error=str(e))


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