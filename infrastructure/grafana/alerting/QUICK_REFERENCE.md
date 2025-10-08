# Grafana Alerting - Quick Reference Card

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Configure environment
cp .env.example .env
vim .env  # Set ALERT_EMAIL_TO=your-email@company.com

# 2. Validate configuration
cd infrastructure/grafana/alerting && ./validate_alerts.sh

# 3. Start services
docker-compose up -d grafana

# 4. Access Grafana
open http://localhost:3001  # admin/admin
```

## 📊 Alert Rules at a Glance

### Critical Alerts (Immediate Action Required)

| Alert | Trigger | Action |
|-------|---------|--------|
| **system_health_degraded** | Status < 2 | Check system health dashboard |
| **system_uptime_low** | < 300s | System recently restarted |
| **db_connection_pool_exhausted** | < 2 available | Scale database connections |
| **trading_portfolio_value_drop** | -5% in 1h | Review positions immediately |
| **trading_orders_failure_rate_high** | > 10% failures | Check Alpaca API status |
| **portfolio_exposure_critical** | > 80% ($800k) | Reduce positions NOW |
| **var_threshold_breach** | > $50k daily | Emergency risk reduction |
| **mcp_agent_coordination_failure** | < 80% success | Check MCP server logs |
| **mcp_server_unavailable** | Status < 1 | Restart MCP services |

### Warning Alerts (Monitor & Investigate)

| Alert | Trigger | Review Dashboard |
|-------|---------|------------------|
| **redis_response_time_high** | > 100ms | system-health |
| **http_request_duration_high** | p95 > 5s | system-health |
| **ai_processing_duration_high** | > 30s | system-health |
| **strategy_drawdown_alert** | > 5% | trading-strategy |
| **trading_api_latency_high** | p95 > 500ms | trading-execution |
| **position_size_warning** | > 15% portfolio | trading-risk |
| **sector_concentration_risk** | > $200k sector | trading-risk |
| **portfolio_beta_extreme_high** | > 2.0 | advanced-risk |
| **volatility_spike_alert** | > 30% | advanced-risk |
| **correlation_risk_alert** | avg > 0.75 | advanced-risk |
| **mcp_agent_high_response_time** | > 5s | system-health |
| **mcp_agent_queue_depth_high** | > 50 ops | system-health |

## 🔔 Notification Channels

### Active Channels

```
Critical → Slack + Email + Webhook (Every 5min)
Warning → Webhook + Discord (Every 15-30min)
Info → Webhook only (Every 1h)
```

### Configure Channels

```bash
# .env configuration
ALERT_EMAIL_TO=alerts@swaggy-stacks.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR/WEBHOOK
```

## 🔍 Quick Troubleshooting

### Alerts Not Firing?

```bash
# Check Prometheus metrics
curl 'http://localhost:9090/api/v1/query?query=trading_system_health_status'

# Check Grafana alerting
docker-compose logs grafana | grep -i "alert\|eval"

# Verify backend metrics
curl http://localhost:8000/metrics | grep trading_
```

### Webhook Failures?

```bash
# Test backend connectivity
docker exec trading_grafana wget -O- http://backend:8000/health

# Check webhook logs
docker-compose logs backend | grep "/api/v1/alerts/webhook"

# Manual webhook test
curl -X POST http://localhost:8000/api/v1/alerts/webhook \
  -H "Content-Type: application/json" \
  -d '{"receiver":"webhook-alerts","status":"firing","alerts":[{"labels":{"alertname":"test"},"annotations":{"description":"Test"}}]}'
```

### Email Not Sending?

```bash
# Verify SMTP config
docker-compose exec grafana env | grep EMAIL

# Check backend email settings
docker-compose logs backend | grep -i "email\|smtp"
```

## 📈 Key Metrics to Monitor

### System Health
```promql
trading_system_health_status           # 0=DOWN, 1=DEGRADED, 2=HEALTHY
trading_system_uptime_seconds          # System uptime
trading_db_connections_available       # DB connections
trading_redis_operation_duration       # Redis latency
```

### Trading Performance
```promql
trading_portfolio_total_value          # Portfolio value
trading_portfolio_daily_pnl           # Daily P&L
trading_order_execution_total         # Order metrics
trading_strategy_drawdown_pct         # Strategy drawdown
```

### Risk Metrics
```promql
trading_portfolio_total_exposure      # Portfolio exposure
trading_portfolio_var_daily          # Daily VaR
trading_portfolio_beta               # Portfolio beta
trading_portfolio_volatility         # Volatility
```

### MCP Coordination
```promql
trading_mcp_agent_coordination_total  # Coordination events
trading_mcp_agent_response_time      # Response times
trading_mcp_agent_queue_depth        # Queue depth
trading_mcp_server_status            # Server status
```

## 🛠️ Common Operations

### View Active Alerts
```bash
# Grafana UI
http://localhost:3001/alerting/list

# Alert history API
curl http://localhost:8000/api/v1/alerts/history?limit=50
```

### Test Alert System
```bash
# Trigger test alert
curl -X POST http://localhost:8000/api/v1/alerts/test

# Check delivery
curl http://localhost:8000/api/v1/alerts/history | jq '.[] | select(.alertname=="test_alert")'
```

### Modify Alert Thresholds
```bash
# Edit alert rule
vim infrastructure/grafana/alerting/rules/trading_alerts.yml

# Restart Grafana
docker-compose restart grafana

# Verify changes
curl http://localhost:3001/api/alerting/rules | jq
```

### Silence Alerts (Maintenance)
```yaml
# Edit notification-policies.yml
mute_time_intervals:
  - maintenance-window

# Restart Grafana
docker-compose restart grafana
```

## 📋 Dashboard Quick Links

| Dashboard | URL | Alert Rules |
|-----------|-----|-------------|
| System Health | http://localhost:3001/d/system-health | 11 rules |
| P&L Dashboard | http://localhost:3001/d/trading-pnl | 3 rules |
| Execution | http://localhost:3001/d/trading-execution | 2 rules |
| Strategy | http://localhost:3001/d/trading-strategy | 1 rule |
| Risk | http://localhost:3001/d/trading-risk | 3 rules |
| Advanced Risk | http://localhost:3001/d/advanced-risk | 6 rules |

## 🎯 Alert Response Playbook

### Critical Alert Response

1. **Acknowledge Alert**
   - Check Grafana alert details
   - Review linked dashboard panel
   - Note alert timestamp and values

2. **Assess Impact**
   - Check affected systems
   - Determine blast radius
   - Estimate recovery time

3. **Take Action**
   - Follow runbook procedures
   - Implement mitigation steps
   - Document actions taken

4. **Verify Resolution**
   - Monitor alert state
   - Confirm metrics return to normal
   - Update incident log

### Escalation Path

```
Level 1: On-Call Engineer → Review alert, attempt resolution
Level 2: Team Lead → Complex issues, service degradation
Level 3: Principal Engineer → System-wide outages
Level 4: CTO/Management → Business impact escalation
```

## 📚 Essential Files

| File | Purpose | Location |
|------|---------|----------|
| README.md | Complete documentation | `infrastructure/grafana/alerting/` |
| DEPLOYMENT.md | Deployment guide | `infrastructure/grafana/alerting/` |
| validate_alerts.sh | Validation script | `infrastructure/grafana/alerting/` |
| .env.example | Config template | Project root |

## 🔐 Security Checklist

- [ ] Configure production email recipients
- [ ] Set up Slack webhook (HTTPS only)
- [ ] Rotate webhook secrets regularly
- [ ] Use environment variables (never hardcode)
- [ ] Enable Grafana authentication
- [ ] Configure alert cooldowns
- [ ] Review alert history for anomalies
- [ ] Implement alert rate limiting

## 📞 Emergency Contacts

```
Platform Team: platform-team@swaggy-stacks.com
Trading Team: trading-team@swaggy-stacks.com
AI/ML Team: ai-team@swaggy-stacks.com
On-Call: oncall@swaggy-stacks.com

PagerDuty: [Configure Integration]
OpsGenie: [Configure Integration]
```

## 💡 Pro Tips

- **Alert Fatigue**: Review and tune thresholds weekly
- **False Positives**: Document and adjust rules
- **Response Time**: Automate common fixes
- **Documentation**: Keep runbooks updated
- **Testing**: Regularly test notification channels
- **Metrics**: Use alert history for capacity planning

---

**Quick Help**: `./validate_alerts.sh` | **Status**: http://localhost:3001/alerting/list | **History**: http://localhost:8000/api/v1/alerts/history
