# RunPod Deployment Guide for Swaggy Stacks Trading System

## Overview

This guide covers deploying the Swaggy Stacks autonomous trading system to RunPod for 24/7 operation.

## Prerequisites

1. **RunPod Account**: Active account with credits
2. **Docker**: Installed locally for building images
3. **RunPod CLI**: Already installed and configured
4. **Environment Variables**: Configured in `.env` file

## Architecture

The deployment consists of:
- **FastAPI Backend**: Trading API server on port 8000
- **Trading Agents**: Autonomous AI agents monitoring markets
- **Supervisor**: Process manager ensuring both services run continuously
- **Health Checks**: Automatic monitoring and recovery

## Deployment Files

- `runpod.dockerfile`: Optimized Docker image for RunPod
- `docker-compose.runpod.yml`: Local testing configuration
- `deploy-to-runpod.sh`: Automated deployment script
- `live_trading_agents.py`: Trading agent coordinator

## Quick Deployment

### 1. Build and Test Locally

```bash
# Build the Docker image
docker build -f runpod.dockerfile -t swaggy-stacks-trading:latest .

# Test with Docker Compose (optional)
docker-compose -f docker-compose.runpod.yml up -d

# Check logs
docker-compose -f docker-compose.runpod.yml logs -f

# Stop local test
docker-compose -f docker-compose.runpod.yml down
```

### 2. Deploy to RunPod

```bash
# Run the deployment script
./deploy-to-runpod.sh

# The script will:
# - Build the Docker image
# - Configure RunPod deployment
# - Create a RunPod pod
# - Display connection details
```

## Manual Deployment Steps

If you prefer manual deployment:

### Step 1: Configure RunPod CLI

```bash
runpodctl config --apiKey $RUNPOD_API_KEY
```

### Step 2: Create RunPod Pod

```bash
runpodctl create pod \
    --name "swaggy-stacks-trading" \
    --imageName "swaggy-stacks-trading:latest" \
    --gpuType "NONE" \
    --containerDiskInGb 20 \
    --volumeInGb 10 \
    --minMemoryInGb 4 \
    --minVcpuCount 2 \
    --ports "8000/http,9090/http" \
    --env "ENVIRONMENT=production" \
    --env "ALPACA_API_KEY=${ALPACA_API_KEY}" \
    --env "ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}" \
    --bid 0.10
```

### Step 3: Access Your Deployment

After deployment, you'll receive endpoints like:
- API: `https://{pod-id}-8000.proxy.runpod.net`
- Metrics: `https://{pod-id}-9090.proxy.runpod.net`

## Environment Configuration

The system reads configuration from environment variables:

```env
# RunPod API
RUNPOD_API_KEY=your_runpod_api_key

# Trading API (Alpaca)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=trading_system

# Application
SECRET_KEY=your_secret_key
ENVIRONMENT=production
DEBUG=false
```

## Monitoring & Management

### View Logs

```bash
# Get pod ID from deployment
POD_ID=$(cat .runpod-deployment-id)

# View logs
runpodctl logs $POD_ID

# Follow logs
runpodctl logs $POD_ID --follow
```

### SSH Access

```bash
# SSH into the running pod
runpodctl ssh $POD_ID
```

### Health Check

```bash
# Check system health
curl https://{endpoint}-8000.proxy.runpod.net/health
```

### Stop/Start Pod

```bash
# Stop pod (preserves data)
runpodctl stop pod $POD_ID

# Start pod
runpodctl start pod $POD_ID

# Delete pod (removes everything)
runpodctl delete pod $POD_ID
```

## Trading System Features

Once deployed, the system provides:

1. **Real-Time Trading**: Monitors AAPL, MSFT, GOOGL, AMZN, TSLA
2. **WebSocket Streaming**: Real-time market data with fallback to polling
3. **AI Agent Coordination**: Multiple specialized agents analyzing markets
4. **Risk Management**: Portfolio limits and position sizing
5. **Paper Trading**: Safe testing with Alpaca paper account

## API Endpoints

- `GET /health`: System health status
- `GET /api/v1/portfolio`: Current portfolio status
- `GET /api/v1/positions`: Active positions
- `POST /api/v1/trading/start`: Start trading agents
- `POST /api/v1/trading/stop`: Stop trading agents
- `GET /docs`: Interactive API documentation

## Troubleshooting

### Pod Won't Start

1. Check RunPod credits/balance
2. Verify API key in `.env`
3. Check Docker image build succeeded
4. Review pod logs: `runpodctl logs $POD_ID`

### Connection Issues

1. Ensure pod status is "RUNNING"
2. Check endpoint URL format
3. Verify ports 8000/9090 are configured
4. Test health endpoint first

### Trading Not Working

1. Verify Alpaca API credentials
2. Check market hours (9:30 AM - 4:00 PM ET)
3. Ensure paper trading mode is enabled
4. Review agent logs in pod

## Cost Optimization

- **CPU-Only Pod**: No GPU needed, reduces costs
- **Spot Instances**: Use bid pricing for better rates
- **Auto-Stop**: Configure idle timeout to save credits
- **Resource Limits**: 2 vCPU, 4GB RAM is sufficient

## Security Considerations

1. **API Keys**: Store in `.env`, never commit to git
2. **Paper Trading**: Always use paper account for testing
3. **Access Control**: RunPod endpoints are public, use SECRET_KEY
4. **Monitoring**: Regular health checks and log reviews

## Support

- **RunPod Issues**: support@runpod.io
- **Trading System**: Check logs and health endpoints
- **Alpaca API**: https://alpaca.markets/support

## Next Steps

1. Monitor initial deployment for 24 hours
2. Review trading performance metrics
3. Adjust risk parameters as needed
4. Scale resources if required
5. Set up alerts for system issues

---

**Note**: This system is configured for paper trading. Never use real money without thorough testing and appropriate risk management.