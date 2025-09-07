# Grafana Dashboard Architecture - Swaggy Stacks Monitoring

## Overview
Comprehensive Grafana dashboard configuration created for real-time monitoring of the Swaggy Stacks algorithmic trading system, consuming all Prometheus metrics from the enhanced monitoring infrastructure.

## Dashboard Structure

### 1. System Health Overview (Panels 1-4)
- **Overall Health Status**: Single stat panel displaying trading_system_health_status with color-coded thresholds (Critical/Degraded/Healthy)
- **System Uptime**: Pie chart visualization of trading_system_uptime_seconds
- **Component Health**: Table view of trading_component_health_status showing detailed breakdown by component and type

### 2. MCP Agent Coordination (Panels 5-9)
- **MCP Server Status**: Time series of mcp_server_status by server name and type
- **Agent Coordination Success Rate**: Gauge displaying mcp_agent_coordination_success_rate with thresholds (70% yellow, 90% green)
- **Response Time**: 95th percentile histogram of mcp_agent_response_time_seconds_bucket
- **Queue Depth**: Time series of mcp_agent_queue_depth by agent and queue type

### 3. Trading Performance (Panels 10-14)
- **Portfolio Value**: Stat panel showing trading_portfolio_value_usd with currency formatting
- **Active Positions**: Pie chart of trading_positions_active by symbol
- **Profit & Loss**: Time series of trading_pnl_total_usd by symbol over time
- **Trading Orders Rate**: Rate calculation of trading_orders_total over 5-minute intervals

### 4. Infrastructure Monitoring (Panels 15-18)
- **Database Connection Pool**: Stat panel of db_connection_pool_size
- **Redis Response Time**: 95th percentile of redis_response_time_seconds_bucket
- **HTTP Request Duration**: 95th percentile of http_request_duration_seconds_bucket by endpoint

### 5. AI & Market Research (Panels 19-21)
- **Market Sentiment Score**: Time series of market_sentiment_score with -1 to 1 range and color thresholds
- **AI Processing Duration**: 95th percentile of ai_processing_duration_seconds_bucket by process type

## Technical Configuration

### Refresh Rate
- **30-second refresh**: Matches MetricsCollector's rate-limited collection interval
- Real-time updates without overwhelming the system

### Data Source
- **Prometheus**: Connected to http://prometheus:9090 via docker network
- Automatic provisioning via datasources/prometheus.yml

### Query Patterns
- **Rate calculations**: `rate(metric_name[5m])` for throughput metrics
- **Percentile calculations**: `histogram_quantile(0.95, rate(metric_bucket[5m]))` for latency
- **Instant queries**: Direct metric values for current state panels

### Color Schemes and Thresholds
- **Health Status**: Red (Critical), Yellow (Degraded), Green (Healthy)
- **Success Rates**: Red (<70%), Yellow (70-90%), Green (>90%)
- **Sentiment**: Red (Negative), Yellow (Neutral), Green (Positive)

## Integration Points

### Provisioning Strategy
- **Auto-discovery**: Dashboard automatically provisioned via dashboard.yml
- **Volume mounting**: Configuration files mounted from host infrastructure/grafana/
- **No manual import**: Zero-touch deployment with docker-compose

### Metric Coverage
Consumes all 20+ metric types from enhanced PrometheusMetrics class:
- System health and component status
- MCP agent coordination metrics (5 new metrics)
- Trading performance and portfolio data
- Database and Redis infrastructure
- HTTP API performance
- AI processing and market research

## Performance Considerations
- **Efficient queries**: Using rate() and histogram_quantile() functions
- **Appropriate time ranges**: 5-minute windows for rate calculations
- **Legend optimization**: Minimal legends to reduce visual clutter
- **Panel organization**: Logical grouping with collapsible rows

This dashboard provides comprehensive real-time visibility into all aspects of the Swaggy Stacks trading system, from system health through MCP agent coordination to trading performance.