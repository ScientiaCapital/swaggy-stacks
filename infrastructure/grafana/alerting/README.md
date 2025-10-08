# SwaggyStacks Grafana Unified Alerting System

## Overview

This directory contains the comprehensive Grafana 10.0+ unified alerting configuration for the SwaggyStacks trading system. The alerting system provides real-time notifications for:

- **System Health**: Infrastructure monitoring, uptime tracking, database/Redis performance
- **Trading Operations**: Portfolio value changes, order execution, strategy performance
- **Risk Management**: Portfolio exposure, position sizing, VaR thresholds, concentration risk
- **MCP Coordination**: AI agent coordination, response times, queue depth, server availability

## Directory Structure

```
infrastructure/grafana/alerting/
├── README.md                          # This file
├── alerting.yml                       # Main provisioning configuration
├── contact-points.yml                 # Notification channel definitions
├── notification-policies.yml          # Alert routing and grouping rules
└── rules/
    ├── system_health_alerts.yml      # Infrastructure & system alerts (6 rules)
    ├── trading_alerts.yml             # Trading operations alerts (7 rules)
    ├── risk_alerts.yml                # Risk management alerts (8 rules)
    └── mcp_alerts.yml                 # MCP coordination alerts (5 rules)
```

## Alert Rules Summary

### System Health Alerts (6 rules)

| Alert Name | Threshold | Severity | Dashboard Link |
|------------|-----------|----------|----------------|
| System Health Degraded | `trading_system_health_status < 2` | Critical | system-health (panel 1) |
| System Uptime Low | `uptime < 300s` | Critical | system-health (panel 2) |
| DB Connection Pool Exhausted | `available < 2` | Critical | system-health (panel 5) |
| Redis Response Time High | `> 100ms` | Warning | system-health (panel 6) |
| HTTP API Response Time High | `p95 > 5s` | Warning | system-health (panel 8) |
| AI Processing Duration High | `> 30s` | Warning | system-health (panel 10) |

### Trading Alerts (7 rules)

| Alert Name | Threshold | Severity | Dashboard Link |
|------------|-----------|----------|----------------|
| Portfolio Value Drop | `< -5% in 1h` | Critical | trading-pnl (panel 1) |
| Order Failure Rate High | `> 10%` | Critical | trading-execution (panel 2) |
| Strategy Drawdown | `> 5%` | Warning | trading-strategy (panel 3) |
| Trading API Latency High | `p95 > 500ms` | Warning | trading-execution (panel 5) |
| Daily P&L Negative | `< 0` | Info | trading-pnl (panel 2) |
| Position Count High | `> 50 positions` | Warning | trading-pnl (panel 4) |

### Risk Management Alerts (8 rules)

| Alert Name | Threshold | Severity | Dashboard Link |
|------------|-----------|----------|----------------|
| Portfolio Exposure Critical | `> 80% ($800k)` | Critical | trading-risk (panel 1) |
| Position Size Warning | `> 15% of portfolio` | Warning | trading-risk (panel 3) |
| Sector Concentration Risk | `> $200k per sector` | Warning | trading-risk (panel 5) |
| VaR Threshold Breach | `> $50k` | Critical | advanced-risk (panel 2) |
| Portfolio Beta Extreme High | `> 2.0` | Warning | advanced-risk (panel 4) |
| Portfolio Volatility Spike | `> 30%` | Warning | advanced-risk (panel 6) |
| Correlation Risk High | `avg > 0.75` | Warning | advanced-risk (panel 8) |

### MCP Coordination Alerts (5 rules)

| Alert Name | Threshold | Severity | Dashboard Link |
|------------|-----------|----------|----------------|
| MCP Agent Coordination Failure | `success rate < 80%` | Critical | system-health (panel 12) |
| MCP Agent Response Time High | `> 5s` | Warning | system-health (panel 13) |
| MCP Agent Queue Depth High | `> 50 operations` | Warning | system-health (panel 14) |
| MCP Server Unavailable | `status < 1` | Critical | system-health (panel 15) |
| MCP Agent Error Rate High | `> 5%` | Warning | system-health (panel 16) |

## Contact Points (Notification Channels)

### Primary Channels

1. **webhook-alerts** (Default)
   - Backend webhook endpoint: `http://backend:8000/api/v1/alerts/webhook`
   - All alerts delivered to backend AlertManager

2. **email-alerts**
   - SMTP email notifications
   - Recipients: `${ALERT_EMAIL_TO}`
   - Used for critical and warning alerts

3. **slack-critical**
   - Slack webhook for critical alerts
   - URL: `${SLACK_WEBHOOK_URL}`
   - Rich message formatting

### Additional Channels (Future Integration)

4. **discord-team** - Team notifications
5. **pagerduty-oncall** - On-call escalation
6. **opsgenie-oncall** - Alternative on-call system

## Notification Routing Logic

### Critical Alerts
- **Channels**: Slack + Email + Webhook
- **Repeat Interval**: 5 minutes
- **Group Wait**: 10 seconds
- **Components**: system, trading, risk, mcp

### Warning Alerts
- **Channels**: Webhook + Discord (performance/risk only)
- **Repeat Interval**: 15-30 minutes
- **Group Wait**: 1-2 minutes

### Info Alerts
- **Channels**: Webhook only
- **Repeat Interval**: 1 hour
- **Group Wait**: 5 minutes

### Team-Based Routing

| Team | Components | Channels | Severity |
|------|------------|----------|----------|
| Platform | system, database | Email + Webhook | Critical/Warning |
| Trading | trading, risk | Email + Webhook | Critical/Warning |
| AI/ML | mcp, ai | Email + Webhook | Critical/Warning |
| Backend | api, execution | Discord + Webhook | Warning |

## Environment Variables

Configure these in `.env` file:

```bash
# Required
ALERT_EMAIL_TO=alerts@swaggy-stacks.com

# Optional (for enhanced notifications)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR/WEBHOOK
PAGERDUTY_INTEGRATION_KEY=your_pagerduty_key
OPSGENIE_API_KEY=your_opsgenie_key
```

## Deployment

### 1. Start Services

```bash
# Start all services including Grafana with alerting
docker-compose up -d

# Check Grafana logs
docker-compose logs -f grafana
```

### 2. Verify Alert Configuration

```bash
# Access Grafana UI
open http://localhost:3001

# Navigate to: Alerting → Alert rules
# Verify all 26 rules are loaded

# Navigate to: Alerting → Contact points
# Verify contact points are configured

# Navigate to: Alerting → Notification policies
# Verify routing rules are active
```

### 3. Test Alerts

```bash
# Trigger test alert (system health)
curl -X POST http://localhost:8000/api/v1/alerts/test

# Check webhook endpoint
curl http://localhost:8000/api/v1/alerts/history
```

## Alert Rule Customization

### Modifying Thresholds

Edit the appropriate rule file in `rules/`:

```yaml
# Example: Change portfolio exposure threshold
expr: (trading_portfolio_total_exposure / 1000000) > 0.9  # Changed from 0.8 to 0.9
```

### Adding New Alerts

1. Choose appropriate file: `system_health_alerts.yml`, `trading_alerts.yml`, etc.
2. Add new rule following existing format:

```yaml
- uid: my_custom_alert
  title: My Custom Alert
  condition: B
  data:
    - refId: A
      datasourceUid: prometheus
      model:
        expr: my_metric > threshold
  annotations:
    summary: "Alert summary"
    description: "Detailed description with {{ $values.A.Value }}"
    dashboard_uid: dashboard-uid
  labels:
    severity: warning
    component: custom
```

3. Reload configuration:

```bash
docker-compose restart grafana
```

## Alert Labels

All alerts include these labels for routing:

- **severity**: `critical`, `warning`, `info`
- **component**: `system`, `trading`, `risk`, `mcp`, `database`, `redis`, `api`, `ai`
- **team**: `platform`, `trading`, `ai`, `ml`, `backend`
- **alert_type**: `health`, `performance`, `pnl`, `execution`, `exposure`, `var`, etc.

## Runbook Links

Each alert includes a `runbook_url` annotation pointing to:
`https://github.com/swaggy-stacks/runbooks/{alert-topic}`

Create corresponding runbooks for:
- System health troubleshooting
- Trading operations recovery
- Risk mitigation procedures
- MCP coordination debugging

## Monitoring & Debugging

### View Active Alerts

```bash
# Grafana UI
http://localhost:3001/alerting/list

# Prometheus UI (if enabled)
http://localhost:9090/alerts
```

### Alert History

```bash
# Backend alert history API
curl http://localhost:8000/api/v1/alerts/history?limit=50

# Filter by severity
curl http://localhost:8000/api/v1/alerts/history?severity=critical
```

### Webhook Debugging

```bash
# Check backend logs for webhook deliveries
docker-compose logs -f backend | grep "webhook"

# Test webhook endpoint
curl -X POST http://localhost:8000/api/v1/alerts/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "alerts": [{
      "labels": {"alertname": "test", "severity": "info"},
      "annotations": {"description": "Test alert"}
    }]
  }'
```

## Maintenance Windows

Suppress alerts during maintenance:

```yaml
# In notification-policies.yml, uncomment mute_time_intervals
mute_time_intervals:
  - maintenance-window  # Defined in alerting.yml
```

## Performance Considerations

- **Evaluation Interval**: 30 seconds (configurable per rule group)
- **For Duration**: 1-10 minutes (varies by alert criticality)
- **Group Wait**: 10s-5m (based on severity)
- **Repeat Interval**: 2m-1h (prevents alert fatigue)

## Integration with Backend AlertManager

The Grafana alerts integrate with the backend `app/monitoring/alerts.py` AlertManager:

1. **Webhook Delivery**: Grafana → `http://backend:8000/api/v1/alerts/webhook`
2. **Backend Processing**: AlertManager handles multi-channel delivery
3. **Cooldown Management**: Backend enforces alert cooldown periods
4. **Alert Deduplication**: Prevents duplicate notifications

## Troubleshooting

### Alerts Not Firing

1. Check Prometheus metrics availability:
   ```bash
   curl http://localhost:9090/api/v1/query?query=trading_system_health_status
   ```

2. Verify Grafana alerting is enabled:
   ```bash
   docker-compose logs grafana | grep "ngalert"
   ```

3. Check rule evaluation:
   ```bash
   # Grafana UI → Alerting → Alert rules → View details
   ```

### Webhook Failures

1. Verify backend is accessible from Grafana container:
   ```bash
   docker exec trading_grafana wget -O- http://backend:8000/health
   ```

2. Check backend webhook endpoint logs:
   ```bash
   docker-compose logs backend | grep "/api/v1/alerts/webhook"
   ```

### Missing Metrics

1. Ensure Prometheus is scraping backend:
   ```bash
   curl http://localhost:9090/api/v1/targets
   ```

2. Verify backend metrics endpoint:
   ```bash
   curl http://localhost:8000/metrics
   ```

## Future Enhancements

- [ ] Anomaly detection alerts using Prometheus ML
- [ ] Alert trend analysis and reporting
- [ ] Automated alert threshold tuning
- [ ] Integration with incident management systems
- [ ] Alert correlation and root cause analysis
- [ ] Custom Grafana alerting plugins

## References

- [Grafana Unified Alerting Documentation](https://grafana.com/docs/grafana/latest/alerting/unified-alerting/)
- [Prometheus Alert Rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)
- [Backend AlertManager Code](/backend/app/monitoring/alerts.py)
- [Dashboard Configurations](/infrastructure/grafana/dashboards/)

---

**Status**: Production Ready ✅
**Version**: 1.0.0
**Last Updated**: 2025-10-07
