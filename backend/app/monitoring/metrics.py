"""
Prometheus metrics collection for system monitoring.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from prometheus_client.core import REGISTRY

from app.core.logging import get_logger

if TYPE_CHECKING:
    from .health_checks import SystemHealthStatus

logger = get_logger(__name__)


@dataclass
class MetricConfig:
    """Configuration for a metric"""

    name: str
    description: str
    labels: list = None


class PrometheusMetrics:
    """Prometheus metrics for the trading system"""

    _instance = None
    _initialized = False

    def __new__(cls, registry: CollectorRegistry = None):
        if cls._instance is None:
            cls._instance = super(PrometheusMetrics, cls).__new__(cls)
        return cls._instance

    def __init__(self, registry: CollectorRegistry = None):
        if not self._initialized:
            self.registry = registry or REGISTRY
            self._setup_metrics()
            PrometheusMetrics._initialized = True

    def _setup_metrics(self):
        """Initialize all Prometheus metrics"""

        # System Health Metrics
        self.system_health_status = Gauge(
            "trading_system_health_status",
            "Overall system health status (0=critical, 1=degraded, 2=healthy)",
            registry=self.registry,
        )

        self.component_health_status = Gauge(
            "trading_component_health_status",
            "Individual component health status",
            ["component", "component_type"],
            registry=self.registry,
        )

        self.system_uptime_seconds = Gauge(
            "trading_system_uptime_seconds",
            "System uptime in seconds",
            registry=self.registry,
        )

        # MCP Metrics
        self.mcp_server_status = Gauge(
            "mcp_server_status",
            "MCP server availability status",
            ["server_name", "server_type"],
            registry=self.registry,
        )

        self.mcp_request_duration = Histogram(
            "mcp_request_duration_seconds",
            "MCP request duration in seconds",
            ["server_name", "method"],
            registry=self.registry,
        )

        self.mcp_request_total = Counter(
            "mcp_request_total",
            "Total MCP requests",
            ["server_name", "method", "status"],
            registry=self.registry,
        )

        self.mcp_errors_total = Counter(
            "mcp_errors_total",
            "Total MCP errors",
            ["server_name", "error_type"],
            registry=self.registry,
        )

        # MCP Agent Coordination Metrics
        self.mcp_agent_coordination_duration = Histogram(
            "mcp_agent_coordination_duration_seconds",
            "Duration of MCP agent coordination operations",
            ["coordination_type", "source_agent", "target_agent"],
            registry=self.registry,
        )

        self.mcp_agent_queue_depth = Gauge(
            "mcp_agent_queue_depth",
            "Current queue depth for MCP agent operations",
            ["agent_name", "queue_type"],
            registry=self.registry,
        )

        self.mcp_cross_agent_requests_total = Counter(
            "mcp_cross_agent_requests_total",
            "Total cross-agent requests between MCP agents",
            ["source_agent", "target_agent", "request_type", "status"],
            registry=self.registry,
        )

        self.mcp_agent_coordination_success_rate = Gauge(
            "mcp_agent_coordination_success_rate",
            "Success rate of agent coordination operations (0-1)",
            ["coordination_type", "agent_pair"],
            registry=self.registry,
        )

        self.mcp_agent_response_time = Histogram(
            "mcp_agent_response_time_seconds",
            "Response time for MCP agent operations",
            ["agent_name", "operation_type"],
            registry=self.registry,
        )

        # Trading System Metrics
        self.trading_orders_total = Counter(
            "trading_orders_total",
            "Total trading orders",
            ["symbol", "side", "status"],
            registry=self.registry,
        )

        self.trading_positions_active = Gauge(
            "trading_positions_active",
            "Currently active trading positions",
            ["symbol"],
            registry=self.registry,
        )

        self.trading_portfolio_value = Gauge(
            "trading_portfolio_value_usd",
            "Portfolio value in USD",
            registry=self.registry,
        )

        self.trading_pnl_total = Gauge(
            "trading_pnl_total_usd",
            "Total profit/loss in USD",
            ["symbol"],
            registry=self.registry,
        )

        # CUSTOM TRADING STRATEGY PERFORMANCE METRICS - Task 1.1

        # Strategy Performance Metrics
        self.strategy_win_loss_ratio = Gauge(
            "trading_strategy_win_loss_ratio",
            "Win/loss ratio for trading strategies",
            ["strategy_name", "symbol", "timeframe"],
            registry=self.registry,
        )

        self.strategy_average_profit_loss = Gauge(
            "trading_strategy_avg_profit_loss_usd",
            "Average profit/loss per trade in USD",
            ["strategy_name", "symbol", "trade_type"],
            registry=self.registry,
        )

        self.strategy_total_trades = Counter(
            "trading_strategy_total_trades",
            "Total trades executed by strategy",
            ["strategy_name", "symbol", "outcome"],
            registry=self.registry,
        )

        self.strategy_success_rate = Gauge(
            "trading_strategy_success_rate",
            "Strategy success rate (0-1)",
            ["strategy_name", "symbol"],
            registry=self.registry,
        )

        self.strategy_drawdown_current = Gauge(
            "trading_strategy_drawdown_current_pct",
            "Current drawdown percentage for strategy",
            ["strategy_name"],
            registry=self.registry,
        )

        self.strategy_drawdown_max = Gauge(
            "trading_strategy_drawdown_max_pct",
            "Maximum drawdown percentage for strategy",
            ["strategy_name"],
            registry=self.registry,
        )

        # Trade Execution Success/Failure Counters
        self.trade_execution_total = Counter(
            "trading_execution_total",
            "Total trade execution attempts",
            ["symbol", "side", "order_type", "status"],
            registry=self.registry,
        )

        self.trade_execution_failures = Counter(
            "trading_execution_failures_total",
            "Total trade execution failures",
            ["symbol", "side", "failure_reason", "error_type"],
            registry=self.registry,
        )

        self.trade_execution_success_rate = Gauge(
            "trading_execution_success_rate",
            "Trade execution success rate (0-1)",
            ["symbol", "order_type"],
            registry=self.registry,
        )

        # Portfolio Exposure and Risk Metrics Gauges
        self.portfolio_exposure_total = Gauge(
            "trading_portfolio_exposure_total_usd",
            "Total portfolio exposure in USD",
            registry=self.registry,
        )

        self.portfolio_exposure_by_sector = Gauge(
            "trading_portfolio_exposure_by_sector_usd",
            "Portfolio exposure by sector in USD",
            ["sector"],
            registry=self.registry,
        )

        self.portfolio_concentration_risk = Gauge(
            "trading_portfolio_concentration_risk",
            "Portfolio concentration risk score (0-1)",
            registry=self.registry,
        )

        self.portfolio_var_daily = Gauge(
            "trading_portfolio_var_daily_usd",
            "Daily Value at Risk in USD",
            ["confidence_level"],
            registry=self.registry,
        )

        self.portfolio_beta = Gauge(
            "trading_portfolio_beta",
            "Portfolio beta relative to market",
            ["benchmark"],
            registry=self.registry,
        )

        self.position_size_risk = Gauge(
            "trading_position_size_risk_pct",
            "Position size as percentage of portfolio",
            ["symbol"],
            registry=self.registry,
        )

        # Latency Metrics for Critical Trading Operations
        self.trade_execution_latency = Histogram(
            "trading_execution_latency_seconds",
            "Trade execution latency in seconds",
            ["operation_type", "symbol", "broker"],
            buckets=[
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
            ],
            registry=self.registry,
        )

        self.market_data_latency = Histogram(
            "trading_market_data_latency_seconds",
            "Market data retrieval latency in seconds",
            ["data_type", "symbol", "source"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry,
        )

        self.strategy_analysis_latency = Histogram(
            "trading_strategy_analysis_latency_seconds",
            "Strategy analysis computation latency in seconds",
            ["strategy_name", "analysis_type", "symbol"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            registry=self.registry,
        )

        self.order_book_latency = Histogram(
            "trading_order_book_latency_seconds",
            "Order book update latency in seconds",
            ["symbol", "exchange"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            registry=self.registry,
        )

        self.risk_check_latency = Histogram(
            "trading_risk_check_latency_seconds",
            "Risk management check latency in seconds",
            ["check_type", "symbol"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry,
        )

        # Database Metrics
        self.db_connection_pool_size = Gauge(
            "db_connection_pool_size",
            "Database connection pool size",
            registry=self.registry,
        )

        self.db_query_duration = Histogram(
            "db_query_duration_seconds",
            "Database query duration in seconds",
            ["query_type"],
            registry=self.registry,
        )

        # Redis Metrics
        self.redis_operations_total = Counter(
            "redis_operations_total",
            "Total Redis operations",
            ["operation", "status"],
            registry=self.registry,
        )

        self.redis_response_time = Histogram(
            "redis_response_time_seconds",
            "Redis operation response time",
            ["operation"],
            registry=self.registry,
        )

        # Market Research Metrics
        self.market_research_requests_total = Counter(
            "market_research_requests_total",
            "Total market research requests",
            ["symbol", "analysis_type", "status"],
            registry=self.registry,
        )

        self.market_sentiment_score = Gauge(
            "market_sentiment_score",
            "Market sentiment score (-1 to 1)",
            ["symbol", "source"],
            registry=self.registry,
        )

        # AI Insights Metrics
        self.ai_insights_generated_total = Counter(
            "ai_insights_generated_total",
            "Total AI insights generated",
            ["insight_type", "status"],
            registry=self.registry,
        )

        self.ai_processing_duration = Histogram(
            "ai_processing_duration_seconds",
            "AI processing duration in seconds",
            ["process_type"],
            registry=self.registry,
        )

        # System Resource Metrics
        self.http_requests_total = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.http_request_duration = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            registry=self.registry,
        )

        # System Info
        self.system_info = Info(
            "trading_system_info", "Trading system information", registry=self.registry
        )

        # UNSUPERVISED LEARNING METRICS - Advanced AI System Monitoring

        # Pattern Memory Metrics
        self.pattern_memory_total_patterns = Gauge(
            "unsupervised_pattern_memory_total_patterns",
            "Total patterns stored in pattern memory",
            ["symbol", "pattern_type"],
            registry=self.registry,
        )

        self.pattern_memory_cache_hit_rate = Gauge(
            "unsupervised_pattern_memory_cache_hit_rate",
            "Pattern memory cache hit rate (0-1)",
            registry=self.registry,
        )

        self.pattern_similarity_search_latency = Histogram(
            "unsupervised_pattern_similarity_search_latency_seconds",
            "Pattern similarity search latency in seconds",
            ["search_type", "pattern_count_bucket"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry,
        )

        self.pattern_compression_ratio = Gauge(
            "unsupervised_pattern_compression_ratio",
            "Pattern compression ratio using VAE (higher is better)",
            registry=self.registry,
        )

        # Market Regime Detection Metrics
        self.regime_detection_accuracy = Gauge(
            "unsupervised_regime_detection_accuracy",
            "Market regime detection accuracy (0-1)",
            ["regime_type"],
            registry=self.registry,
        )

        self.regime_transition_detection_latency = Histogram(
            "unsupervised_regime_transition_detection_latency_seconds",
            "Regime transition detection latency in seconds",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry,
        )

        self.regime_stability_score = Gauge(
            "unsupervised_regime_stability_score",
            "Current market regime stability score (0-1)",
            registry=self.registry,
        )

        self.regime_confidence_score = Gauge(
            "unsupervised_regime_confidence_score",
            "Market regime detection confidence (0-1)",
            ["detected_regime"],
            registry=self.registry,
        )

        # Anomaly Detection Metrics
        self.anomaly_detection_accuracy = Gauge(
            "unsupervised_anomaly_detection_accuracy",
            "Anomaly detection accuracy vs labeled data (0-1)",
            ["anomaly_type"],
            registry=self.registry,
        )

        self.anomaly_detection_latency = Histogram(
            "unsupervised_anomaly_detection_latency_seconds",
            "Anomaly detection computation latency in seconds",
            ["detector_type"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            registry=self.registry,
        )

        self.anomaly_score_distribution = Histogram(
            "unsupervised_anomaly_score_distribution",
            "Distribution of anomaly scores",
            ["symbol"],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry,
        )

        self.anomaly_alerts_total = Counter(
            "unsupervised_anomaly_alerts_total",
            "Total anomaly alerts generated",
            ["symbol", "anomaly_type", "severity"],
            registry=self.registry,
        )

        # Clustering Quality Metrics
        self.clustering_silhouette_score = Gauge(
            "unsupervised_clustering_silhouette_score",
            "Clustering silhouette score (-1 to 1, higher is better)",
            ["clustering_algorithm", "data_type"],
            registry=self.registry,
        )

        self.clustering_inertia = Gauge(
            "unsupervised_clustering_inertia",
            "K-means clustering inertia (lower is better)",
            ["data_type"],
            registry=self.registry,
        )

        self.clustering_iterations_to_convergence = Gauge(
            "unsupervised_clustering_iterations_to_convergence",
            "Iterations required for clustering convergence",
            ["clustering_algorithm"],
            registry=self.registry,
        )

        self.cluster_stability_score = Gauge(
            "unsupervised_cluster_stability_score",
            "Cluster stability across different runs (0-1)",
            ["clustering_algorithm"],
            registry=self.registry,
        )

        # Experience Clustering Metrics
        self.experience_clusters_total = Gauge(
            "unsupervised_experience_clusters_total",
            "Total experience clusters identified",
            ["agent_type", "cluster_type"],
            registry=self.registry,
        )

        self.experience_cluster_purity = Gauge(
            "unsupervised_experience_cluster_purity",
            "Experience cluster purity score (0-1)",
            ["agent_type", "cluster_id"],
            registry=self.registry,
        )

        self.experience_replay_effectiveness = Gauge(
            "unsupervised_experience_replay_effectiveness",
            "Experience replay learning effectiveness (0-1)",
            ["agent_type"],
            registry=self.registry,
        )

        # Strategy Evolution Metrics
        self.strategy_evolution_generation = Gauge(
            "unsupervised_strategy_evolution_generation",
            "Current strategy evolution generation",
            ["agent_type"],
            registry=self.registry,
        )

        self.strategy_variant_performance = Gauge(
            "unsupervised_strategy_variant_performance",
            "Strategy variant performance score (0-1)",
            ["agent_type", "variant_id"],
            registry=self.registry,
        )

        self.ab_test_statistical_significance = Gauge(
            "unsupervised_ab_test_statistical_significance",
            "A/B test statistical significance (0-1)",
            ["test_id", "metric_type"],
            registry=self.registry,
        )

        self.strategy_improvement_rate = Gauge(
            "unsupervised_strategy_improvement_rate",
            "Strategy improvement rate per evolution cycle",
            ["agent_type"],
            registry=self.registry,
        )

        # Market Basket Analysis Metrics
        self.association_rules_confidence = Gauge(
            "unsupervised_association_rules_confidence",
            "Market basket association rules confidence (0-1)",
            ["rule_id", "symbol_pair"],
            registry=self.registry,
        )

        self.association_rules_lift = Gauge(
            "unsupervised_association_rules_lift",
            "Market basket association rules lift score",
            ["rule_id", "symbol_pair"],
            registry=self.registry,
        )

        self.correlation_network_density = Gauge(
            "unsupervised_correlation_network_density",
            "Market correlation network density (0-1)",
            registry=self.registry,
        )

        # Learning Performance Metrics
        self.unsupervised_learning_accuracy = Gauge(
            "unsupervised_learning_accuracy_score",
            "Overall unsupervised learning accuracy (0-1)",
            ["learning_type"],
            registry=self.registry,
        )

        self.unsupervised_prediction_accuracy = Gauge(
            "unsupervised_prediction_accuracy",
            "Prediction accuracy using unsupervised insights (0-1)",
            ["prediction_type", "time_horizon"],
            registry=self.registry,
        )

        self.unsupervised_feature_importance = Gauge(
            "unsupervised_feature_importance_score",
            "Feature importance scores from unsupervised learning (0-1)",
            ["feature_name", "symbol"],
            registry=self.registry,
        )

        # System Resource Usage for Unsupervised Learning
        self.unsupervised_memory_usage = Gauge(
            "unsupervised_memory_usage_bytes",
            "Memory usage by unsupervised learning components",
            ["component_name"],
            registry=self.registry,
        )

        self.unsupervised_cpu_usage = Gauge(
            "unsupervised_cpu_usage_percent",
            "CPU usage by unsupervised learning components (0-100)",
            ["component_name"],
            registry=self.registry,
        )

        self.unsupervised_model_training_time = Histogram(

        # ===== OPTIONS TRADING METRICS - Task 10 Implementation =====

        # Options Greeks Portfolio Exposure Metrics
        self.options_portfolio_delta = Gauge(
            "options_portfolio_delta_total",
            "Total portfolio delta exposure",
            registry=self.registry,
        )

        self.options_portfolio_gamma = Gauge(
            "options_portfolio_gamma_total",
            "Total portfolio gamma exposure",
            registry=self.registry,
        )

        self.options_portfolio_theta = Gauge(
            "options_portfolio_theta_total",
            "Total portfolio theta (time decay) in USD per day",
            registry=self.registry,
        )

        self.options_portfolio_vega = Gauge(
            "options_portfolio_vega_total",
            "Total portfolio vega (volatility sensitivity)",
            registry=self.registry,
        )

        self.options_portfolio_rho = Gauge(
            "options_portfolio_rho_total",
            "Total portfolio rho (interest rate sensitivity)",
            registry=self.registry,
        )

        # Individual Position Greeks
        self.options_position_delta = Gauge(
            "options_position_delta",
            "Delta for individual options positions",
            ["symbol", "strategy", "expiration"],
            registry=self.registry,
        )

        self.options_position_gamma = Gauge(
            "options_position_gamma",
            "Gamma for individual options positions",
            ["symbol", "strategy", "expiration"],
            registry=self.registry,
        )

        # Strategy-Specific Performance Metrics
        self.options_strategy_pnl = Gauge(
            "options_strategy_pnl_usd",
            "Profit/loss by options strategy in USD",
            ["strategy_name", "symbol"],
            registry=self.registry,
        )

        self.options_strategy_success_rate = Gauge(
            "options_strategy_success_rate",
            "Success rate for options strategies (0-1)",
            ["strategy_name", "symbol"],
            registry=self.registry,
        )

        self.options_strategy_trades_total = Counter(
            "options_strategy_trades_total",
            "Total trades by options strategy",
            ["strategy_name", "symbol", "outcome"],
            registry=self.registry,
        )

        self.options_strategy_max_profit = Gauge(
            "options_strategy_max_profit_potential_usd",
            "Maximum profit potential for current strategy positions",
            ["strategy_name", "symbol"],
            registry=self.registry,
        )

        self.options_strategy_max_loss = Gauge(
            "options_strategy_max_loss_potential_usd",
            "Maximum loss potential for current strategy positions",
            ["strategy_name", "symbol"],
            registry=self.registry,
        )

        # Options Volume and Open Interest Tracking
        self.options_volume_daily = Gauge(
            "options_volume_daily_contracts",
            "Daily options volume in contracts",
            ["symbol", "option_type", "expiration"],
            registry=self.registry,
        )

        self.options_open_interest = Gauge(
            "options_open_interest_contracts",
            "Current open interest in contracts",
            ["symbol", "option_type", "strike", "expiration"],
            registry=self.registry,
        )

        self.options_volume_to_open_interest_ratio = Gauge(
            "options_volume_to_oi_ratio",
            "Volume to open interest ratio",
            ["symbol", "expiration"],
            registry=self.registry,
        )

        # Assignment Risk Metrics
        self.options_assignment_risk_score = Gauge(
            "options_assignment_risk_score",
            "Assignment risk score for short options positions (0-1)",
            ["symbol", "strike", "expiration", "option_type"],
            registry=self.registry,
        )

        self.options_time_to_expiry_hours = Gauge(
            "options_time_to_expiry_hours",
            "Hours until options expiration",
            ["symbol", "expiration"],
            registry=self.registry,
        )

        self.options_moneyness = Gauge(
            "options_moneyness_ratio",
            "Option moneyness (strike/spot) ratio",
            ["symbol", "strike", "expiration", "option_type"],
            registry=self.registry,
        )

        # Multi-Leg Execution Metrics
        self.options_multileg_execution_total = Counter(
            "options_multileg_execution_total",
            "Total multi-leg options executions",
            ["strategy_name", "leg_count", "status"],
            registry=self.registry,
        )

        self.options_multileg_execution_success_rate = Gauge(
            "options_multileg_execution_success_rate",
            "Multi-leg execution success rate (0-1)",
            ["strategy_name", "leg_count"],
            registry=self.registry,
        )

        self.options_multileg_execution_latency = Histogram(
            "options_multileg_execution_latency_seconds",
            "Multi-leg execution latency in seconds",
            ["strategy_name", "leg_count"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            registry=self.registry,
        )

        self.options_leg_fill_rate = Gauge(
            "options_leg_fill_rate",
            "Fill rate for individual legs in multi-leg orders (0-1)",
            ["strategy_name", "leg_number"],
            registry=self.registry,
        )

        # Volatility Prediction Accuracy Metrics
        self.options_volatility_prediction_accuracy = Gauge(
            "options_volatility_prediction_accuracy",
            "GARCH volatility prediction accuracy vs realized (0-1)",
            ["symbol", "time_horizon"],
            registry=self.registry,
        )

        self.options_implied_vs_realized_vol_diff = Gauge(
            "options_implied_vs_realized_vol_diff",
            "Difference between implied and realized volatility",
            ["symbol", "expiration"],
            registry=self.registry,
        )

        self.options_vol_smile_skew = Gauge(
            "options_vol_smile_skew",
            "Volatility smile skew measurement",
            ["symbol", "expiration"],
            registry=self.registry,
        )

        self.options_garch_model_confidence = Gauge(
            "options_garch_model_confidence_score",
            "GARCH model confidence score (0-1)",
            ["symbol"],
            registry=self.registry,
        )

        # Options Market Health Metrics
        self.options_market_bid_ask_spread = Gauge(
            "options_market_bid_ask_spread_pct",
            "Options bid-ask spread as percentage of mid price",
            ["symbol", "strike", "expiration", "option_type"],
            registry=self.registry,
        )

        self.options_liquidity_score = Gauge(
            "options_liquidity_score",
            "Options liquidity score based on volume and spread (0-1)",
            ["symbol", "expiration"],
            registry=self.registry,
        )

        self.options_pin_risk = Gauge(
            "options_pin_risk_score",
            "Pin risk score for options near strikes at expiration (0-1)",
            ["symbol", "strike", "expiration"],
            registry=self.registry,
        )

        # Risk Management Metrics
        self.options_max_loss_exposure = Gauge(
            "options_max_loss_exposure_usd",
            "Maximum potential loss exposure in USD",
            ["strategy_name"],
            registry=self.registry,
        )

        self.options_buying_power_used = Gauge(
            "options_buying_power_used_usd",
            "Buying power used for options positions in USD",
            registry=self.registry,
        )

        self.options_margin_requirement = Gauge(
            "options_margin_requirement_usd",
            "Margin requirement for options positions in USD",
            ["strategy_name"],
            registry=self.registry,
        )

        # Greeks Risk Limits Monitoring
        self.options_delta_limit_utilization = Gauge(
            "options_delta_limit_utilization_pct",
            "Portfolio delta limit utilization percentage",
            registry=self.registry,
        )

        self.options_gamma_limit_utilization = Gauge(
            "options_gamma_limit_utilization_pct",
            "Portfolio gamma limit utilization percentage",
            registry=self.registry,
        )

        self.options_vega_limit_utilization = Gauge(
            "options_vega_limit_utilization_pct",
            "Portfolio vega limit utilization percentage",
            registry=self.registry,
        )

        self.options_theta_daily_decay = Gauge(
            "options_theta_daily_decay_usd",
            "Expected daily theta decay in USD",
            registry=self.registry,
        )
            "unsupervised_model_training_time_seconds",
            "Time to train unsupervised models",
            ["model_type", "data_size_bucket"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0],
            registry=self.registry,
        )

    def update_health_metrics(self, health_status: "SystemHealthStatus"):
        """Update health-related metrics"""
        from .health_checks import HealthStatus

        # Overall system health
        health_value = {
            HealthStatus.CRITICAL: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.HEALTHY: 2,
        }.get(health_status.overall_status, 0)

        self.system_health_status.set(health_value)
        self.system_uptime_seconds.set(health_status.uptime_seconds)

        # Component health
        for component in health_status.components:
            component_value = {
                HealthStatus.CRITICAL: 0,
                HealthStatus.DEGRADED: 1,
                HealthStatus.HEALTHY: 2,
                HealthStatus.UNKNOWN: 0,
            }.get(component.status, 0)

            self.component_health_status.labels(
                component=component.component,
                component_type=component.component_type.value,
            ).set(component_value)

    def record_mcp_request(
        self, server_name: str, method: str, duration: float, status: str
    ):
        """Record MCP request metrics"""
        self.mcp_request_duration.labels(
            server_name=server_name, method=method
        ).observe(duration)

        self.mcp_request_total.labels(
            server_name=server_name, method=method, status=status
        ).inc()

    def record_mcp_error(self, server_name: str, error_type: str):
        """Record MCP error"""
        self.mcp_errors_total.labels(
            server_name=server_name, error_type=error_type
        ).inc()

    def update_mcp_server_status(
        self, server_name: str, server_type: str, available: bool
    ):
        """Update MCP server status"""
        self.mcp_server_status.labels(
            server_name=server_name, server_type=server_type
        ).set(1 if available else 0)

    def record_mcp_agent_coordination(
        self,
        coordination_type: str,
        source_agent: str,
        target_agent: str,
        duration: float,
        success: bool,
    ):
        """Record MCP agent coordination metrics"""
        # Record coordination duration
        self.mcp_agent_coordination_duration.labels(
            coordination_type=coordination_type,
            source_agent=source_agent,
            target_agent=target_agent,
        ).observe(duration)

        # Record cross-agent request
        status = "success" if success else "failed"
        self.mcp_cross_agent_requests_total.labels(
            source_agent=source_agent,
            target_agent=target_agent,
            request_type=coordination_type,
            status=status,
        ).inc()

        # Update success rate (simplified calculation for demo)
        agent_pair = f"{source_agent}-{target_agent}"
        current_rate = 1.0 if success else 0.0
        self.mcp_agent_coordination_success_rate.labels(
            coordination_type=coordination_type, agent_pair=agent_pair
        ).set(current_rate)

    def record_mcp_agent_response_time(
        self, agent_name: str, operation_type: str, duration: float
    ):
        """Record MCP agent response time"""
        self.mcp_agent_response_time.labels(
            agent_name=agent_name, operation_type=operation_type
        ).observe(duration)

    def update_mcp_agent_queue_depth(
        self, agent_name: str, queue_type: str, depth: int
    ):
        """Update MCP agent queue depth"""
        self.mcp_agent_queue_depth.labels(
            agent_name=agent_name, queue_type=queue_type
        ).set(depth)

    def record_trading_order(self, symbol: str, side: str, status: str):
        """Record trading order"""
        self.trading_orders_total.labels(symbol=symbol, side=side, status=status).inc()

    def update_portfolio_metrics(
        self, portfolio_value: float, positions: Dict[str, float]
    ):
        """Update portfolio metrics"""
        self.trading_portfolio_value.set(portfolio_value)

        for symbol, quantity in positions.items():
            self.trading_positions_active.labels(symbol=symbol).set(abs(quantity))

    def record_market_research(self, symbol: str, analysis_type: str, status: str):
        """Record market research request"""
        self.market_research_requests_total.labels(
            symbol=symbol, analysis_type=analysis_type, status=status
        ).inc()

    def update_sentiment_score(self, symbol: str, source: str, score: float):
        """Update market sentiment score"""
        self.market_sentiment_score.labels(symbol=symbol, source=source).set(score)

    def record_ai_insight(self, insight_type: str, status: str, processing_time: float):
        """Record AI insight generation"""
        self.ai_insights_generated_total.labels(
            insight_type=insight_type, status=status
        ).inc()

        self.ai_processing_duration.labels(process_type=insight_type).observe(
            processing_time
        )

    def record_http_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ):
        """Record HTTP request metrics"""
        self.http_requests_total.labels(
            method=method, endpoint=endpoint, status=str(status_code)
        ).inc()

        self.http_request_duration.labels(method=method, endpoint=endpoint).observe(
            duration
        )

    def set_system_info(self, version: str, environment: str, build_date: str):
        """Set system information"""
        self.system_info.info(
            {
                "version": version,
                "environment": environment,
                "build_date": build_date,
                "system": "swaggy-stacks-trading",
            }
        )

    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        return generate_latest(self.registry)

    # CUSTOM TRADING STRATEGY PERFORMANCE METHODS - Task 1.1 Implementation

    def record_strategy_performance(
        self,
        strategy_name: str,
        symbol: str,
        win_loss_ratio: float,
        avg_profit: float,
        avg_loss: float,
        total_trades: int,
        successful_trades: int,
        timeframe: str = "1D",
    ):
        """Record comprehensive strategy performance metrics"""

        # Win/loss ratio
        self.strategy_win_loss_ratio.labels(
            strategy_name=strategy_name, symbol=symbol, timeframe=timeframe
        ).set(win_loss_ratio)

        # Average profit/loss
        self.strategy_average_profit_loss.labels(
            strategy_name=strategy_name, symbol=symbol, trade_type="profit"
        ).set(avg_profit)

        self.strategy_average_profit_loss.labels(
            strategy_name=strategy_name, symbol=symbol, trade_type="loss"
        ).set(avg_loss)

        # Success rate calculation
        success_rate = successful_trades / total_trades if total_trades > 0 else 0
        self.strategy_success_rate.labels(
            strategy_name=strategy_name, symbol=symbol
        ).set(success_rate)

    def record_strategy_trade_outcome(
        self, strategy_name: str, symbol: str, outcome: str
    ):
        """Record individual trade outcome for strategy"""
        self.strategy_total_trades.labels(
            strategy_name=strategy_name,
            symbol=symbol,
            outcome=outcome,  # "win", "loss", "breakeven"
        ).inc()

    def update_strategy_drawdown(
        self, strategy_name: str, current_drawdown_pct: float, max_drawdown_pct: float
    ):
        """Update strategy drawdown metrics"""
        self.strategy_drawdown_current.labels(strategy_name=strategy_name).set(
            current_drawdown_pct
        )
        self.strategy_drawdown_max.labels(strategy_name=strategy_name).set(
            max_drawdown_pct
        )

    def record_trade_execution_metrics(
        self,
        symbol: str,
        side: str,
        order_type: str,
        status: str,
        execution_time: float,
        failure_reason: str = None,
    ):
        """Record trade execution performance and latency"""

        # Record execution attempt
        self.trade_execution_total.labels(
            symbol=symbol, side=side, order_type=order_type, status=status
        ).inc()

        # Record execution latency
        self.trade_execution_latency.labels(
            operation_type="order_placement", symbol=symbol, broker="alpaca"
        ).observe(execution_time)

        # Record failures if applicable
        if status == "failed" and failure_reason:
            self.trade_execution_failures.labels(
                symbol=symbol,
                side=side,
                failure_reason=failure_reason,
                error_type="execution_error",
            ).inc()

        # Update success rate (simplified calculation)
        if status == "filled":
            success_rate = 0.95  # This would be calculated from historical data
            self.trade_execution_success_rate.labels(
                symbol=symbol, order_type=order_type
            ).set(success_rate)

    def update_portfolio_risk_metrics(
        self,
        total_exposure: float,
        sector_exposures: Dict[str, float],
        concentration_risk: float,
        var_daily: float,
        beta: float,
        position_risks: Dict[str, float],
        benchmark: str = "SPY",
    ):
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

    def record_market_data_latency(
        self, data_type: str, symbol: str, source: str, latency: float
    ):
        """Record market data retrieval latency"""
        self.market_data_latency.labels(
            data_type=data_type,  # "quote", "bars", "trades"
            symbol=symbol,
            source=source,  # "alpaca", "polygon", "yahoo"
        ).observe(latency)

    def record_strategy_analysis_latency(
        self, strategy_name: str, analysis_type: str, symbol: str, duration: float
    ):
        """Record strategy analysis computation time"""
        self.strategy_analysis_latency.labels(
            strategy_name=strategy_name,
            analysis_type=analysis_type,  # "markov", "wyckoff", "fibonacci"
            symbol=symbol,
        ).observe(duration)

    def record_order_book_latency(self, symbol: str, exchange: str, latency: float):
        """Record order book update latency"""
        self.order_book_latency.labels(symbol=symbol, exchange=exchange).observe(
            latency
        )

    def record_risk_check_latency(self, check_type: str, symbol: str, duration: float):
        """Record risk management check latency"""
        self.risk_check_latency.labels(
            check_type=check_type,  # "position_size", "exposure", "drawdown"
            symbol=symbol,
        ).observe(duration)

    def collect_trading_manager_metrics(self, trading_manager):
        """Collect metrics from TradingManager instance"""
        try:
            # Portfolio value and positions
            if trading_manager.performance_metrics:
                portfolio_value = trading_manager.performance_metrics.get(
                    "total_equity", 0
                )
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
            if hasattr(strategy_agent, "strategies"):
                for strategy_name in strategy_agent.strategies:
                    # Record strategy usage
                    self.strategy_total_trades.labels(
                        strategy_name=strategy_name, symbol=symbol, outcome="analysis"
                    ).inc()

            # Record analysis type
            if hasattr(strategy_agent, "consensus_method"):
                consensus_method = strategy_agent.consensus_method
                logger.debug(f"Using consensus method: {consensus_method}")

        except Exception as e:
            logger.warning("Failed to collect strategy agent metrics", error=str(e))

    # Unsupervised Learning Metrics Update Methods
    def update_pattern_memory_metrics(
        self,
        symbol: str,
        pattern_type: str,
        total_patterns: int,
        cache_hit_rate: float,
        compression_ratio: float,
        memory_usage_mb: float,
        retrieval_latency_ms: float,
    ):
        """Update pattern memory performance metrics"""
        self.pattern_memory_total_patterns.labels(
            symbol=symbol, pattern_type=pattern_type
        ).set(total_patterns)

        self.pattern_memory_cache_hit_rate.set(cache_hit_rate)

        self.pattern_compression_ratio.set(compression_ratio)

        self.unsupervised_memory_usage.labels(component="pattern_memory").set(memory_usage_mb)

        self.pattern_similarity_search_latency.labels(
            search_type="retrieval", pattern_count_bucket="standard"
        ).observe(retrieval_latency_ms / 1000.0)  # Convert to seconds

    def update_regime_detection_metrics(
        self,
        detected_regime: str,
        regime_stability: float,
        prediction_accuracy: float,
        detection_latency_ms: float,
    ):
        """Update market regime detection metrics"""
        self.regime_detection_accuracy.labels(regime_type=detected_regime).set(prediction_accuracy)

        self.regime_stability_score.set(regime_stability)

        self.regime_confidence_score.labels(detected_regime=detected_regime).set(prediction_accuracy)

        self.regime_transition_detection_latency.observe(detection_latency_ms / 1000.0)  # Convert to seconds

    def update_anomaly_detection_metrics(
        self,
        symbol: str,
        anomaly_type: str,
        anomaly_score: float,
        detection_accuracy: float,
        detection_latency_ms: float,
        is_anomaly: bool,
        severity: str = "medium",
    ):
        """Update anomaly detection metrics"""
        self.anomaly_detection_accuracy.labels(anomaly_type=anomaly_type).set(detection_accuracy)

        self.anomaly_detection_latency.labels(detector_type=anomaly_type).observe(detection_latency_ms / 1000.0)

        self.anomaly_score_distribution.labels(symbol=symbol).observe(anomaly_score)

        if is_anomaly:
            self.anomaly_alerts_total.labels(
                symbol=symbol, anomaly_type=anomaly_type, severity=severity
            ).inc()

    def update_unsupervised_learning_accuracy(self, learning_type: str, accuracy: float):
        """Update overall unsupervised learning accuracy"""
        self.unsupervised_learning_accuracy.labels(learning_type=learning_type).set(accuracy)

    def update_prediction_accuracy(self, prediction_type: str, time_horizon: str, accuracy: float):
        """Update prediction accuracy metrics"""
        self.unsupervised_prediction_accuracy.labels(
            prediction_type=prediction_type, time_horizon=time_horizon
        ).set(accuracy)

    def update_feature_importance(self, feature_name: str, symbol: str, importance: float):
        """Update feature importance scores"""
        self.unsupervised_feature_importance.labels(
            feature_name=feature_name, symbol=symbol
        ).set(importance)

    def update_model_training_time(self, model_type: str, data_size_bucket: str, training_time_seconds: float):
        """Update unsupervised model training time"""
        self.unsupervised_model_training_time.labels(
            model_type=model_type, data_size_bucket=data_size_bucket
        ).observe(training_time_seconds)

    def update_unsupervised_resource_metrics(
        self,
        component_name: str,
        cpu_usage_percent: float,
        memory_usage_bytes: float,
    ):
        """Update resource usage for unsupervised components"""
        self.unsupervised_cpu_usage.labels(component=component_name).set(cpu_usage_percent)
        self.unsupervised_memory_usage.labels(component=component_name).set(memory_usage_bytes)

    async def collect_unsupervised_metrics(self):
        """Collect metrics from all unsupervised learning components"""
        try:
            if not UNSUPERVISED_AVAILABLE:
                logger.debug("Unsupervised components not available, skipping metrics collection")
                return

            # Basic resource monitoring for now
            # Individual components will update their own metrics when called
            logger.debug("Unsupervised metrics collection completed")

        except Exception as e:
            logger.warning("Failed to collect unsupervised metrics", error=str(e))

    # ===== OPTIONS MONITORING METHODS - Task 10 Implementation =====

    def update_options_portfolio_greeks(
        self,
        total_delta: float,
        total_gamma: float,
        total_theta: float,
        total_vega: float,
        total_rho: float
    ):
        """Update portfolio-wide Greeks exposure metrics"""
        self.options_portfolio_delta.set(total_delta)
        self.options_portfolio_gamma.set(total_gamma)
        self.options_portfolio_theta.set(total_theta)
        self.options_portfolio_vega.set(total_vega)
        self.options_portfolio_rho.set(total_rho)

        logger.debug(
            "Updated portfolio Greeks metrics",
            delta=total_delta,
            gamma=total_gamma,
            theta=total_theta,
            vega=total_vega
        )

    def update_options_position_greeks(
        self,
        symbol: str,
        strategy: str,
        expiration: str,
        delta: float,
        gamma: float
    ):
        """Update individual position Greeks"""
        self.options_position_delta.labels(
            symbol=symbol,
            strategy=strategy,
            expiration=expiration
        ).set(delta)

        self.options_position_gamma.labels(
            symbol=symbol,
            strategy=strategy,
            expiration=expiration
        ).set(gamma)

    def record_options_strategy_performance(
        self,
        strategy_name: str,
        symbol: str,
        pnl: float,
        success_rate: float,
        max_profit: float,
        max_loss: float
    ):
        """Record options strategy performance metrics"""
        self.options_strategy_pnl.labels(
            strategy_name=strategy_name,
            symbol=symbol
        ).set(pnl)

        self.options_strategy_success_rate.labels(
            strategy_name=strategy_name,
            symbol=symbol
        ).set(success_rate)

        self.options_strategy_max_profit.labels(
            strategy_name=strategy_name,
            symbol=symbol
        ).set(max_profit)

        self.options_strategy_max_loss.labels(
            strategy_name=strategy_name,
            symbol=symbol
        ).set(max_loss)

        logger.debug(
            "Updated options strategy performance",
            strategy=strategy_name,
            symbol=symbol,
            pnl=pnl,
            success_rate=success_rate
        )

    def record_options_strategy_trade(
        self,
        strategy_name: str,
        symbol: str,
        outcome: str
    ):
        """Record individual options strategy trade outcome"""
        self.options_strategy_trades_total.labels(
            strategy_name=strategy_name,
            symbol=symbol,
            outcome=outcome  # "win", "loss", "breakeven", "expired_worthless"
        ).inc()

    def update_options_volume_metrics(
        self,
        symbol: str,
        option_type: str,
        expiration: str,
        daily_volume: int,
        open_interest: int,
        strike: str = None
    ):
        """Update options volume and open interest metrics"""
        self.options_volume_daily.labels(
            symbol=symbol,
            option_type=option_type,
            expiration=expiration
        ).set(daily_volume)

        if strike:
            self.options_open_interest.labels(
                symbol=symbol,
                option_type=option_type,
                strike=strike,
                expiration=expiration
            ).set(open_interest)

        # Calculate volume to open interest ratio
        if open_interest > 0:
            vol_oi_ratio = daily_volume / open_interest
            self.options_volume_to_open_interest_ratio.labels(
                symbol=symbol,
                expiration=expiration
            ).set(vol_oi_ratio)

    def update_options_assignment_risk(
        self,
        symbol: str,
        strike: str,
        expiration: str,
        option_type: str,
        assignment_risk_score: float,
        time_to_expiry_hours: float,
        moneyness: float
    ):
        """Update assignment risk metrics for short options"""
        self.options_assignment_risk_score.labels(
            symbol=symbol,
            strike=strike,
            expiration=expiration,
            option_type=option_type
        ).set(assignment_risk_score)

        self.options_time_to_expiry_hours.labels(
            symbol=symbol,
            expiration=expiration
        ).set(time_to_expiry_hours)

        self.options_moneyness.labels(
            symbol=symbol,
            strike=strike,
            expiration=expiration,
            option_type=option_type
        ).set(moneyness)

    def record_options_multileg_execution(
        self,
        strategy_name: str,
        leg_count: int,
        status: str,
        execution_latency: float,
        leg_fill_rates: List[float] = None
    ):
        """Record multi-leg options execution metrics"""
        self.options_multileg_execution_total.labels(
            strategy_name=strategy_name,
            leg_count=str(leg_count),
            status=status  # "success", "partial_fill", "failed", "cancelled"
        ).inc()

        self.options_multileg_execution_latency.labels(
            strategy_name=strategy_name,
            leg_count=str(leg_count)
        ).observe(execution_latency)

        # Update success rate calculation (simplified)
        if status == "success":
            current_success_rate = 0.95  # This would be calculated from historical data
            self.options_multileg_execution_success_rate.labels(
                strategy_name=strategy_name,
                leg_count=str(leg_count)
            ).set(current_success_rate)

        # Record individual leg fill rates
        if leg_fill_rates:
            for leg_num, fill_rate in enumerate(leg_fill_rates, 1):
                self.options_leg_fill_rate.labels(
                    strategy_name=strategy_name,
                    leg_number=str(leg_num)
                ).set(fill_rate)

        logger.debug(
            "Recorded multi-leg execution",
            strategy=strategy_name,
            legs=leg_count,
            status=status,
            latency=execution_latency
        )

    def update_options_volatility_metrics(
        self,
        symbol: str,
        expiration: str,
        time_horizon: str,
        prediction_accuracy: float,
        implied_vol: float,
        realized_vol: float,
        vol_smile_skew: float,
        garch_confidence: float
    ):
        """Update volatility prediction and analysis metrics"""
        self.options_volatility_prediction_accuracy.labels(
            symbol=symbol,
            time_horizon=time_horizon
        ).set(prediction_accuracy)

        vol_diff = implied_vol - realized_vol
        self.options_implied_vs_realized_vol_diff.labels(
            symbol=symbol,
            expiration=expiration
        ).set(vol_diff)

        self.options_vol_smile_skew.labels(
            symbol=symbol,
            expiration=expiration
        ).set(vol_smile_skew)

        self.options_garch_model_confidence.labels(
            symbol=symbol
        ).set(garch_confidence)

        logger.debug(
            "Updated volatility metrics",
            symbol=symbol,
            prediction_accuracy=prediction_accuracy,
            vol_diff=vol_diff,
            garch_confidence=garch_confidence
        )

    def update_options_market_health(
        self,
        symbol: str,
        strike: str,
        expiration: str,
        option_type: str,
        bid_ask_spread_pct: float,
        liquidity_score: float,
        pin_risk_score: float = None
    ):
        """Update options market health and liquidity metrics"""
        self.options_market_bid_ask_spread.labels(
            symbol=symbol,
            strike=strike,
            expiration=expiration,
            option_type=option_type
        ).set(bid_ask_spread_pct)

        self.options_liquidity_score.labels(
            symbol=symbol,
            expiration=expiration
        ).set(liquidity_score)

        if pin_risk_score is not None:
            self.options_pin_risk.labels(
                symbol=symbol,
                strike=strike,
                expiration=expiration
            ).set(pin_risk_score)

    def update_options_risk_metrics(
        self,
        strategy_name: str,
        max_loss_exposure: float,
        margin_requirement: float,
        buying_power_used: float = None
    ):
        """Update options risk management metrics"""
        self.options_max_loss_exposure.labels(
            strategy_name=strategy_name
        ).set(max_loss_exposure)

        self.options_margin_requirement.labels(
            strategy_name=strategy_name
        ).set(margin_requirement)

        if buying_power_used is not None:
            self.options_buying_power_used.set(buying_power_used)

    def update_options_greeks_limits(
        self,
        delta_limit_utilization_pct: float,
        gamma_limit_utilization_pct: float,
        vega_limit_utilization_pct: float,
        theta_daily_decay: float
    ):
        """Update Greeks risk limits utilization"""
        self.options_delta_limit_utilization.set(delta_limit_utilization_pct)
        self.options_gamma_limit_utilization.set(gamma_limit_utilization_pct)
        self.options_vega_limit_utilization.set(vega_limit_utilization_pct)
        self.options_theta_daily_decay.set(theta_daily_decay)

        logger.debug(
            "Updated Greeks limits",
            delta_util=delta_limit_utilization_pct,
            gamma_util=gamma_limit_utilization_pct,
            vega_util=vega_limit_utilization_pct,
            theta_decay=theta_daily_decay
        )

    async def collect_options_metrics_from_greeks_manager(self, greeks_manager):
        """Collect comprehensive metrics from GreeksRiskManager"""
        try:
            if not hasattr(greeks_manager, 'portfolio_greeks'):
                logger.debug("GreeksRiskManager not initialized, skipping metrics collection")
                return

            # Get current portfolio Greeks
            portfolio_greeks = greeks_manager.portfolio_greeks
            
            self.update_options_portfolio_greeks(
                total_delta=portfolio_greeks.total_delta,
                total_gamma=portfolio_greeks.total_gamma,
                total_theta=portfolio_greeks.total_theta,
                total_vega=portfolio_greeks.total_vega,
                total_rho=portfolio_greeks.total_rho
            )

            # Get Greeks limits utilization
            limits = greeks_manager.greeks_limits
            if limits:
                delta_util = abs(portfolio_greeks.total_delta / limits.max_portfolio_delta) * 100
                gamma_util = abs(portfolio_greeks.total_gamma / limits.max_portfolio_gamma) * 100
                vega_util = abs(portfolio_greeks.total_vega / limits.max_portfolio_vega) * 100

                self.update_options_greeks_limits(
                    delta_limit_utilization_pct=delta_util,
                    gamma_limit_utilization_pct=gamma_util,
                    vega_limit_utilization_pct=vega_util,
                    theta_daily_decay=portfolio_greeks.total_theta
                )

            logger.debug("Options Greeks metrics collected successfully")

        except Exception as e:
            logger.warning("Failed to collect options Greeks metrics", error=str(e))

    async def collect_options_metrics_from_strategies(self, strategy_instances):
        """Collect metrics from options strategy instances"""
        try:
            strategy_mapping = {
                "ZeroDTEStrategy": "zero_dte",
                "WheelStrategy": "wheel",
                "IronCondorStrategy": "iron_condor",
                "GammaScalpingStrategy": "gamma_scalping"
            }

            for strategy_class_name, strategy_instance in strategy_instances.items():
                strategy_name = strategy_mapping.get(strategy_class_name, strategy_class_name.lower())

                if hasattr(strategy_instance, 'active_positions'):
                    for symbol, positions in strategy_instance.active_positions.items():
                        # Calculate strategy performance
                        total_pnl = sum(pos.unrealized_pnl for pos in positions)
                        
                        # Update strategy performance metrics
                        self.record_options_strategy_performance(
                            strategy_name=strategy_name,
                            symbol=symbol,
                            pnl=total_pnl,
                            success_rate=0.75,  # This would be calculated from historical data
                            max_profit=sum(pos.max_profit for pos in positions),
                            max_loss=sum(pos.max_loss for pos in positions)
                        )

                        # Update individual position Greeks
                        for position in positions:
                            if hasattr(position, 'greeks_data'):
                                self.update_options_position_greeks(
                                    symbol=symbol,
                                    strategy=strategy_name,
                                    expiration=position.expiration,
                                    delta=position.greeks_data.delta,
                                    gamma=position.greeks_data.gamma
                                )

            logger.debug("Options strategy metrics collected successfully")

        except Exception as e:
            logger.warning("Failed to collect options strategy metrics", error=str(e))

    async def collect_options_metrics_from_volatility_predictor(self, volatility_predictor):
        """Collect metrics from volatility prediction system"""
        try:
            # This would integrate with the VolatilityPredictor from Task 9
            if hasattr(volatility_predictor, 'cache'):
                cache_hit_rate = len(volatility_predictor.cache) / 100.0  # Simplified calculation
                
                for cache_key, (metrics, _) in volatility_predictor.cache.items():
                    symbol = cache_key.split('_')[0]
                    
                    self.update_options_volatility_metrics(
                        symbol=symbol,
                        expiration="default",
                        time_horizon="1D",
                        prediction_accuracy=metrics.confidence_score,
                        implied_vol=metrics.implied_vol or 0.0,
                        realized_vol=metrics.historical_vol,
                        vol_smile_skew=metrics.vol_smile_skew,
                        garch_confidence=metrics.confidence_score
                    )

            logger.debug("Volatility prediction metrics collected successfully")

        except Exception as e:
            logger.warning("Failed to collect volatility prediction metrics", error=str(e))


class MetricsCollector:
    """Collects and manages system metrics"""

    def __init__(self):
        self.prometheus_metrics = PrometheusMetrics()
        from .health_checks import HealthChecker

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

            # Collect unsupervised learning metrics
            await self.prometheus_metrics.collect_unsupervised_metrics()

            # Collect additional metrics
            metrics = {
                "health_status": health_status,
                "system_metrics": await self.health_checker.get_system_metrics(),
                "timestamp": current_time,
            }

            self._last_update = current_time
            self._cached_metrics = metrics

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {"error": str(e), "timestamp": current_time}

    async def get_cached_metrics(self) -> Dict[str, Any]:
        """Get cached metrics to avoid frequent collection"""
        if hasattr(self, "_cached_metrics"):
            return self._cached_metrics
        return await self.collect_system_metrics()

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        return self.prometheus_metrics.get_metrics()

    async def update_mcp_metrics(self, server_statuses: Dict[str, bool]):
        """Update MCP server metrics"""
        for server_name, available in server_statuses.items():
            self.prometheus_metrics.update_mcp_server_status(
                server_name=server_name, server_type="mcp", available=available
            )

    def record_request_metrics(
        self, method: str, endpoint: str, status_code: int, duration: float
    ):
        """Record HTTP request metrics"""
        self.prometheus_metrics.record_http_request(
            method, endpoint, status_code, duration
        )
