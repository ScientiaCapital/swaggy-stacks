# SwaggyStacks Metrics Catalog

**Version**: 1.0  
**Last Updated**: 2025-10-07  
**Total Metrics**: 64

## Quick Start Guide

### Accessing Metrics
- **Primary Endpoint**: `http://localhost:8000/api/v1/monitoring/metrics`
- **Prometheus UI**: `http://localhost:9090`
- **Grafana Dashboards**: `http://localhost:3001`

### Common PromQL Queries
```promql
# Portfolio value over time
trading_portfolio_value_usd

# Order success rate by symbol
sum by(symbol) (rate(trading_orders_total{status="filled"}[5m])) / sum by(symbol) (rate(trading_orders_total[5m]))

# P95 execution latency
histogram_quantile(0.95, rate(trading_execution_latency_seconds_bucket[5m]))

# MCP agent coordination success rate
avg(mcp_agent_coordination_success_rate) by (coordination_type)
```

---

## System Health Metrics

### trading_system_health_status
**Type**: Gauge  
**Description**: Overall system health status for monitoring and alerting  
**Labels**: None  
**Values**: 0=critical, 1=degraded, 2=healthy  
**Update Frequency**: Every health check cycle (10s)

**PromQL Examples**:
```promql
# Current health status
trading_system_health_status

# Alert on degraded/critical
trading_system_health_status < 2
```

**Grafana Integration**:
```promql
# Traffic light visualization
trading_system_health_status
# Use: Stat panel with thresholds (0=red, 1=orange, 2=green)
```

### trading_component_health_status
**Type**: Gauge  
**Description**: Individual component health for granular monitoring  
**Labels**: `component`, `component_type`  
**Values**: 0=critical, 1=degraded, 2=healthy, 0=unknown  
**Update Frequency**: Per-component health checks

**PromQL Examples**:
```promql
# Database component health
trading_component_health_status{component="database"}

# All degraded components
trading_component_health_status{} < 2
```

### trading_system_uptime_seconds
**Type**: Gauge  
**Description**: System uptime in seconds for availability tracking  
**Labels**: None  
**Update Frequency**: Continuous (updated with health checks)

**PromQL Examples**:
```promql
# Uptime in hours
trading_system_uptime_seconds / 3600

# Alert if system restarted (uptime < 5min)
trading_system_uptime_seconds < 300
```

---

## MCP Agent Coordination Metrics

### mcp_server_status
**Type**: Gauge  
**Description**: MCP server availability status for agent connectivity monitoring
  
**Labels**: `server_name`, `server_type`  
**Values**: 1=available, 0=unavailable  
**Update Frequency**: On MCP server status change

**PromQL Examples**:
```promql
# All unavailable MCP servers
mcp_server_status{} == 0

# Server availability rate over 5m
avg_over_time(mcp_server_status[5m])
```

### mcp_agent_coordination_success_rate
**Type**: Gauge  
**Description**: Success rate of agent coordination operations (0-1)  
**Labels**: `coordination_type`, `agent_pair`  
**Update Frequency**: After each coordination operation

**PromQL Examples**:
```promql
# Overall coordination success rate
avg(mcp_agent_coordination_success_rate)

# Success rate by coordination type
avg by(coordination_type) (mcp_agent_coordination_success_rate)

# Alert on low success rate
mcp_agent_coordination_success_rate < 0.8
```

### mcp_agent_response_time_seconds
**Type**: Histogram  
**Description**: Response time for MCP agent operations  
**Labels**: `agent_name`, `operation_type`  
**Update Frequency**: Per agent operation

**PromQL Examples**:
```promql
# P95 response time by agent
histogram_quantile(0.95, rate(mcp_agent_response_time_seconds_bucket[5m]))

# Average response time by operation
rate(mcp_agent_response_time_seconds_sum[5m]) / rate(mcp_agent_response_time_seconds_count[5m])
```

### mcp_agent_queue_depth
**Type**: Gauge  
**Description**: Current queue depth for MCP agent operations  
**Labels**: `agent_name`, `queue_type`  
**Update Frequency**: Real-time on queue changes

**PromQL Examples**:
```promql
# Current queue depths
mcp_agent_queue_depth

# Alert on high queue depth
mcp_agent_queue_depth > 50
```

---

## Trading Core Metrics

### trading_orders_total
**Type**: Counter  
**Description**: Total trading orders for order flow tracking  
**Labels**: `symbol`, `side`, `status`  
**Update Frequency**: Per order execution

**PromQL Examples**:
```promql
# Order rate by status
rate(trading_orders_total[5m])

# Order success rate
sum(rate(trading_orders_total{status="filled"}[5m])) / sum(rate(trading_orders_total[5m]))

# Orders by symbol
sum by(symbol) (rate(trading_orders_total[5m]))
```

**Grafana Integration**:
```promql
# Order flow heatmap by symbol and status
sum by(symbol, status) (increase(trading_orders_total[1h]))
```

### trading_portfolio_value_usd
**Type**: Gauge  
**Description**: Total portfolio value in USD for real-time P&L tracking  
**Labels**: None  
**Update Frequency**: Real-time on position changes

**PromQL Examples**:
```promql
# Current portfolio value
trading_portfolio_value_usd

# Portfolio value change over 1h
delta(trading_portfolio_value_usd[1h])

# Daily return percentage
(trading_portfolio_value_usd - trading_portfolio_value_usd offset 1d) / (trading_portfolio_value_usd offset 1d) * 100
```

### trading_positions_active
**Type**: Gauge  
**Description**: Currently active trading positions (absolute quantity)  
**Labels**: `symbol`  
**Update Frequency**: On position open/close

**PromQL Examples**:
```promql
# All active positions
trading_positions_active

# Position concentration
topk(5, trading_positions_active)

# Total positions count
count(trading_positions_active > 0)
```

### trading_pnl_total_usd
**Type**: Gauge  
**Description**: Total profit/loss in USD  
**Labels**: `symbol`  
**Update Frequency**: Real-time P&L calculation

**PromQL Examples**:
```promql
# Total P&L
sum(trading_pnl_total_usd)

# P&L by symbol
trading_pnl_total_usd

# Best/worst performers
topk(3, trading_pnl_total_usd)
bottomk(3, trading_pnl_total_usd)
```

---

## Strategy Performance Metrics

### trading_strategy_win_loss_ratio
**Type**: Gauge  
**Description**: Win/loss ratio for trading strategies  
**Labels**: `strategy_name`, `symbol`, `timeframe`  
**Update Frequency**: After strategy performance calculation

**PromQL Examples**:
```promql
# Win/loss ratio by strategy
trading_strategy_win_loss_ratio

# Best performing strategies
topk(3, avg by(strategy_name) (trading_strategy_win_loss_ratio))
```

### trading_strategy_success_rate
**Type**: Gauge  
**Description**: Strategy success rate (0-1)  
**Labels**: `strategy_name`, `symbol`  
**Update Frequency**: After each trade outcome

**PromQL Examples**:
```promql
# Success rate by strategy
avg by(strategy_name) (trading_strategy_success_rate)

# Alert on low success rate
trading_strategy_success_rate < 0.5
```

### trading_strategy_drawdown_current_pct
**Type**: Gauge  
**Description**: Current drawdown percentage from peak  
**Labels**: `strategy_name`  
**Update Frequency**: Real-time drawdown calculation

**PromQL Examples**:
```promql
# Current drawdowns
trading_strategy_drawdown_current_pct

# Alert on excessive drawdown
trading_strategy_drawdown_current_pct > 10
```

---

## Latency Metrics

### trading_execution_latency_seconds
**Type**: Histogram  
**Description**: Trade execution latency in seconds  
**Labels**: `operation_type`, `symbol`, `broker`  
**Buckets**: [1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s]  
**Update Frequency**: Per order execution

**PromQL Examples**:
```promql
# P95 execution latency
histogram_quantile(0.95, rate(trading_execution_latency_seconds_bucket[5m]))

# Average latency by broker
rate(trading_execution_latency_seconds_sum[5m]) / rate(trading_execution_latency_seconds_count[5m])

# Latency distribution
sum by(le) (rate(trading_execution_latency_seconds_bucket[5m]))
```

### trading_market_data_latency_seconds
**Type**: Histogram  
**Description**: Market data retrieval latency  
**Labels**: `data_type`, `symbol`, `source`  
**Buckets**: [1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s]

**PromQL Examples**:
```promql
# P99 data latency by type
histogram_quantile(0.99, rate(trading_market_data_latency_seconds_bucket[5m]))
```

---

## Portfolio Risk Metrics

### trading_portfolio_exposure_total_usd
**Type**: Gauge  
**Description**: Total portfolio exposure in USD  
**Labels**: None

**PromQL Examples**:
```promql
# Current exposure
trading_portfolio_exposure_total_usd

# Exposure as % of portfolio value
(trading_portfolio_exposure_total_usd / trading_portfolio_value_usd) * 100
```

### trading_portfolio_var_daily_usd
**Type**: Gauge  
**Description**: Daily Value at Risk in USD  
**Labels**: `confidence_level`

**PromQL Examples**:
```promql
# 95% confidence VaR
trading_portfolio_var_daily_usd{confidence_level="95"}

# VaR as % of portfolio
(trading_portfolio_var_daily_usd / trading_portfolio_value_usd) * 100
```

---

## Options Trading Metrics

### options_portfolio_delta_total
**Type**: Gauge  
**Description**: Total portfolio delta exposure  
**Labels**: None

**PromQL Examples**:
```promql
# Current delta exposure
options_portfolio_delta_total

# Delta-neutral check
abs(options_portfolio_delta_total) < 0.1
```

### options_strategy_pnl_usd
**Type**: Gauge  
**Description**: Profit/loss by options strategy in USD  
**Labels**: `strategy_name`, `symbol`

**PromQL Examples**:
```promql
# Strategy P&L
sum by(strategy_name) (options_strategy_pnl_usd)
```

---

## Unsupervised Learning Metrics

### unsupervised_regime_detection_accuracy
**Type**: Gauge  
**Description**: Market regime detection accuracy (0-1)  
**Labels**: `regime_type`

**PromQL Examples**:
```promql
# Detection accuracy by regime
avg by(regime_type) (unsupervised_regime_detection_accuracy)
```

### unsupervised_anomaly_alerts_total
**Type**: Counter  
**Description**: Total anomaly alerts generated  
**Labels**: `symbol`, `anomaly_type`, `severity`

**PromQL Examples**:
```promql
# Alert rate by severity
rate(unsupervised_anomaly_alerts_total[5m])
```

---

## Troubleshooting Guide

### Missing Metrics
**Problem**: Metrics not appearing in Prometheus  
**Solutions**:
1. Check backend health: `curl http://localhost:8000/api/v1/monitoring/metrics`
2. Verify Prometheus scrape: Check `http://localhost:9090/targets`
3. Review logs: `docker logs trading_backend`

### High Cardinality
**Problem**: Too many unique label combinations  
**Solutions**:
1. Use recording rules for pre-aggregation
2. Limit symbol labels to active positions only
3. Review `infrastructure/recording_rules.yml`

### Slow Queries
**Problem**: Grafana dashboard timeouts  
**Solutions**:
1. Use recording rules for complex queries
2. Reduce query time range
3. Add query caching in Grafana

---

## Dashboard Integration

### P&L Dashboard (pnl_dashboard.json)
**Key Metrics**:
- `trading_portfolio_value_usd`
- `trading_pnl_total_usd`
- `trading_positions_active`

### Strategy Performance (strategy_dashboard.json)
**Key Metrics**:
- `trading_strategy_win_loss_ratio`
- `trading_strategy_success_rate`
- `trading_strategy_drawdown_current_pct`

### Trade Execution (execution_dashboard.json)
**Key Metrics**:
- `trading_execution_latency_seconds`
- `trading_execution_success_rate`
- `trading_orders_total`

---

**For complete metric list**: See `backend/app/monitoring/metrics.py`  
**For enhanced method docs**: See audit report at `infrastructure/docs/metrics_audit_report.md`  
**For naming validation**: See `infrastructure/docs/metrics_naming_validation.md`
