# Grafana Unified Alerting - Deployment Guide

## Quick Start

### 1. Configure Environment Variables

```bash
# Copy example configuration
cp .env.example .env

# Edit .env and configure alert channels
vim .env
```

**Required Variables:**
```bash
ALERT_EMAIL_TO=your-email@company.com
```

**Optional Variables (for enhanced notifications):**
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR/WEBHOOK
PAGERDUTY_INTEGRATION_KEY=your_key
OPSGENIE_API_KEY=your_key
```

### 2. Validate Configuration

```bash
cd infrastructure/grafana/alerting
./validate_alerts.sh
```

Expected output:
```
✓ All validation checks passed!
24 unique alert UIDs verified
```

### 3. Deploy Services

```bash
# Start all services including Grafana with alerting
docker-compose up -d

# Check Grafana startup logs
docker-compose logs -f grafana
```

Look for these log lines:
```
ngalert.api: Alerting API enabled
provisioning.alerting: Provisioning alerting from configuration
```

### 4. Verify in Grafana UI

1. **Access Grafana**: http://localhost:3001
   - Username: `admin`
   - Password: `admin` (change on first login)

2. **Navigate to Alerting**:
   - Click: **Alerting** → **Alert rules**
   - Verify: 24 alert rules loaded across 4 groups

3. **Check Contact Points**:
   - Click: **Alerting** → **Contact points**
   - Verify: 6+ contact points configured

4. **Review Notification Policies**:
   - Click: **Alerting** → **Notification policies**
   - Verify: Routing rules active

### 5. Test Alerts

```bash
# Test webhook endpoint
curl -X POST http://localhost:8000/api/v1/alerts/test \
  -H "Content-Type: application/json" \
  -d '{
    "test_type": "system_health",
    "severity": "info"
  }'

# Check alert history
curl http://localhost:8000/api/v1/alerts/history | jq
```

## Alert Rules Summary

| Category | Rules | Critical | Warning | Info |
|----------|-------|----------|---------|------|
| System Health | 6 | 3 | 3 | 0 |
| Trading Operations | 7 | 2 | 4 | 1 |
| Risk Management | 8 | 2 | 6 | 0 |
| MCP Coordination | 5 | 2 | 3 | 0 |
| **Total** | **26** | **9** | **16** | **1** |

## Notification Channels

### Primary Channels (Always Active)

1. **webhook-alerts** (Default)
   - All alerts delivered to backend
   - URL: `http://backend:8000/api/v1/alerts/webhook`
   - Backend handles multi-channel distribution

2. **email-alerts**
   - Critical and warning alerts
   - SMTP configuration in .env
   - Recipient: `${ALERT_EMAIL_TO}`

3. **slack-critical**
   - Critical alerts only
   - Rich formatting with alert context
   - Requires: `${SLACK_WEBHOOK_URL}`

### Optional Channels

4. **discord-team**
   - Team notifications for performance/risk warnings
   - Requires: `${DISCORD_WEBHOOK_URL}`

5. **pagerduty-oncall**
   - On-call escalation (future)
   - Requires: `${PAGERDUTY_INTEGRATION_KEY}`

6. **opsgenie-oncall**
   - Incident management (future)
   - Requires: `${OPSGENIE_API_KEY}`

## Alert Routing Logic

### Critical Alerts (9 rules)
```
Severity: critical
Channels: Slack + Email + Webhook
Repeat: Every 5 minutes
Examples:
  - System health degraded
  - Portfolio value drop > 5%
  - Order failure rate > 10%
  - MCP coordination failure
```

### Warning Alerts (16 rules)
```
Severity: warning
Channels: Webhook + Discord (selected)
Repeat: Every 15-30 minutes
Examples:
  - Redis latency > 100ms
  - Strategy drawdown > 5%
  - Sector concentration > $200k
  - MCP response time > 5s
```

### Info Alerts (1 rule)
```
Severity: info
Channels: Webhook only
Repeat: Every 1 hour
Examples:
  - Daily P&L negative
```

## Alert Grouping Strategy

Alerts are grouped by:
- `alertname`: Prevents duplicate alerts
- `severity`: Groups by criticality
- `component`: Groups by system component

**Timing**:
- Group Wait: 10s-5m (based on severity)
- Group Interval: 2m-30m
- Repeat Interval: 5m-1h

## Dashboard Integration

All alerts link to specific dashboard panels:

| Dashboard UID | Panel Count | Alert Rules |
|---------------|-------------|-------------|
| system-health | 16 | 11 rules |
| trading-pnl | 4 | 3 rules |
| trading-execution | 5 | 2 rules |
| trading-strategy | 3 | 1 rule |
| trading-risk | 5 | 3 rules |
| advanced-risk | 8 | 6 rules |

## Troubleshooting

### Alerts Not Firing

**Check Prometheus metrics:**
```bash
# Verify metric availability
curl 'http://localhost:9090/api/v1/query?query=trading_system_health_status'

# Check specific alert expression
curl 'http://localhost:9090/api/v1/query?query=trading_portfolio_total_value'
```

**Check Grafana alert evaluation:**
```bash
# View Grafana logs
docker-compose logs grafana | grep -i "alert\|eval"

# Check alert state in UI
# Grafana → Alerting → Alert rules → [rule name] → View state
```

### Webhook Delivery Failures

**Verify backend connectivity:**
```bash
# Test from Grafana container
docker exec trading_grafana wget -O- http://backend:8000/health

# Check backend webhook logs
docker-compose logs backend | grep "/api/v1/alerts/webhook"
```

**Check backend alert endpoint:**
```bash
# Test webhook manually
curl -X POST http://localhost:8000/api/v1/alerts/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "receiver": "webhook-alerts",
    "status": "firing",
    "alerts": [{
      "labels": {
        "alertname": "test_alert",
        "severity": "info"
      },
      "annotations": {
        "description": "Test alert"
      }
    }]
  }'
```

### Email Delivery Issues

**Verify SMTP configuration:**
```bash
# Check environment variables
docker-compose exec grafana env | grep -E "EMAIL|ALERT"

# Test SMTP from backend
docker-compose exec backend python3 -c "
from app.core.config import settings
print(f'Email Host: {settings.EMAIL_HOST}')
print(f'Email Port: {settings.EMAIL_PORT}')
print(f'Email User: {settings.EMAIL_USERNAME}')
print(f'Alert Recipient: {settings.ALERT_EMAIL_TO}')
"
```

### Missing Metrics

**Check Prometheus scraping:**
```bash
# View Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job, health}'

# Check backend metrics endpoint
curl http://localhost:8000/metrics | grep trading_
```

**Verify metrics export:**
```bash
# Check backend metrics exposure
docker-compose logs backend | grep "metrics"

# Test specific metric
curl -s http://localhost:8000/metrics | grep trading_system_health_status
```

## Alert Threshold Tuning

### System Health

**Database Connections:**
```yaml
# Current: 2 available
# Adjust in: rules/system_health_alerts.yml
expr: trading_db_connections_available < 5  # Increase threshold
```

**API Latency:**
```yaml
# Current: p95 > 5s
# Adjust in: rules/system_health_alerts.yml
expr: histogram_quantile(0.95, ...) > 3  # Lower threshold for stricter SLA
```

### Trading Operations

**Portfolio Drop:**
```yaml
# Current: -5% in 1h
# Adjust in: rules/trading_alerts.yml
expr: ... < -3  # More sensitive to drops
```

**Order Failures:**
```yaml
# Current: 10%
# Adjust in: rules/trading_alerts.yml
expr: ... > 5  # Lower tolerance for failures
```

### Risk Management

**Exposure Limits:**
```yaml
# Current: 80% ($800k)
# Adjust in: rules/risk_alerts.yml
expr: (trading_portfolio_total_exposure / 1000000) > 0.7  # 70% threshold
```

**VaR Threshold:**
```yaml
# Current: $50k
# Adjust in: rules/risk_alerts.yml
expr: trading_portfolio_var_daily > 40000  # $40k threshold
```

## Maintenance Windows

To suppress alerts during scheduled maintenance:

1. **Edit notification-policies.yml:**
```yaml
mute_time_intervals:
  - maintenance-window
```

2. **Update alerting.yml:**
```yaml
muteTimings:
  - name: maintenance-window
    time_intervals:
      - times:
          - start_time: "02:00"
            end_time: "04:00"
        weekdays:
          - "saturday"
          - "sunday"
```

3. **Reload Grafana:**
```bash
docker-compose restart grafana
```

## Monitoring Alert System Health

### Alert Evaluation Metrics

Grafana exposes alerting metrics on port 3000:

```bash
# Alert evaluation duration
curl -s http://localhost:3001/metrics | grep grafana_alerting_rule_evaluation_duration

# Alert state
curl -s http://localhost:3001/metrics | grep grafana_alerting_alerts

# Notification delivery
curl -s http://localhost:3001/metrics | grep grafana_alerting_notifications
```

### Backend Alert Metrics

```bash
# Alert webhook deliveries
curl -s http://localhost:8000/metrics | grep alert_webhook

# Alert cooldown tracking
curl -s http://localhost:8000/metrics | grep alert_cooldown
```

## Production Deployment Checklist

- [ ] Configure `.env` with production email/webhook URLs
- [ ] Set appropriate alert thresholds for production SLAs
- [ ] Test all notification channels (email, Slack, Discord)
- [ ] Verify dashboard links are accessible
- [ ] Configure maintenance windows for scheduled downtime
- [ ] Set up runbook links for each alert type
- [ ] Test alert escalation flow
- [ ] Configure PagerDuty/OpsGenie for on-call rotation
- [ ] Document alert response procedures
- [ ] Train team on alert acknowledgment and resolution

## Next Steps

1. **Create Runbooks**: Document response procedures for each alert
2. **Set Up On-Call**: Configure PagerDuty/OpsGenie rotation
3. **Alert Analysis**: Review alert history and tune thresholds
4. **Incident Response**: Define escalation procedures
5. **Alert Fatigue Prevention**: Monitor alert frequency and adjust

## Resources

- [Grafana Unified Alerting Docs](https://grafana.com/docs/grafana/latest/alerting/unified-alerting/)
- [Backend AlertManager Code](/backend/app/monitoring/alerts.py)
- [Dashboard Configurations](/infrastructure/grafana/dashboards/)
- [Prometheus Recording Rules](/infrastructure/recording_rules.yml)

---

**Status**: Production Ready ✅
**Alert Rules**: 26 active
**Notification Channels**: 6 configured
**Last Updated**: 2025-10-07
