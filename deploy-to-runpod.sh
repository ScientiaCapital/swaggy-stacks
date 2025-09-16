#!/bin/bash

# RunPod Deployment Script for Swaggy Stacks Trading System
# This script builds and deploys the trading system to RunPod

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check for required environment variables
if [ -z "$RUNPOD_API_KEY" ]; then
    echo -e "${RED}Error: RUNPOD_API_KEY not found in .env file${NC}"
    exit 1
fi

echo -e "${GREEN}=== Swaggy Stacks RunPod Deployment ===${NC}"
echo ""

# Function to check command availability
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        exit 1
    fi
}

# Check required tools
echo -e "${YELLOW}Checking required tools...${NC}"
check_command docker

# Use the local runpodctl
RUNPODCTL="./runpodctl"
if [ ! -f "$RUNPODCTL" ]; then
    echo -e "${RED}Error: $RUNPODCTL not found in current directory${NC}"
    exit 1
fi

# Step 1: Build Docker image locally
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -f runpod.dockerfile -t swaggy-stacks-trading:latest .

# Step 2: Test the image locally (optional)
read -p "Do you want to test the image locally first? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Starting local test...${NC}"
    docker-compose -f docker-compose.runpod.yml up -d
    echo -e "${GREEN}Local test environment started${NC}"
    echo "API available at: http://localhost:8000"
    echo "Run 'docker-compose -f docker-compose.runpod.yml logs -f' to see logs"
    echo ""
    read -p "Press any key to continue deployment (or Ctrl+C to cancel)..." -n 1 -r
    echo
    docker-compose -f docker-compose.runpod.yml down
fi

# Step 3: Tag image for RunPod registry
echo -e "${YELLOW}Preparing image for RunPod...${NC}"
RUNPOD_IMAGE_TAG="swaggy-stacks-trading:$(date +%Y%m%d-%H%M%S)"
docker tag swaggy-stacks-trading:latest $RUNPOD_IMAGE_TAG

# Step 4: Create RunPod configuration
echo -e "${YELLOW}Creating RunPod deployment configuration...${NC}"
cat > runpod-deployment.json <<EOF
{
  "name": "swaggy-stacks-trading",
  "imageName": "$RUNPOD_IMAGE_TAG",
  "gpuType": "NONE",
  "minMemoryInGb": 4,
  "minVcpuCount": 2,
  "containerDiskInGb": 20,
  "volumeInGb": 10,
  "volumeMountPath": "/app/data",
  "ports": "8000/http,9090/http",
  "env": [
    {
      "key": "ENVIRONMENT",
      "value": "production"
    },
    {
      "key": "DEBUG",
      "value": "false"
    },
    {
      "key": "SECRET_KEY",
      "value": "${SECRET_KEY}"
    },
    {
      "key": "POSTGRES_SERVER",
      "value": "localhost"
    },
    {
      "key": "POSTGRES_USER",
      "value": "${POSTGRES_USER}"
    },
    {
      "key": "POSTGRES_PASSWORD",
      "value": "${POSTGRES_PASSWORD}"
    },
    {
      "key": "POSTGRES_DB",
      "value": "${POSTGRES_DB}"
    },
    {
      "key": "REDIS_URL",
      "value": "redis://localhost:6379"
    },
    {
      "key": "ALPACA_API_KEY",
      "value": "${ALPACA_API_KEY}"
    },
    {
      "key": "ALPACA_SECRET_KEY",
      "value": "${ALPACA_SECRET_KEY}"
    },
    {
      "key": "ALPACA_BASE_URL",
      "value": "${ALPACA_BASE_URL}"
    },
    {
      "key": "EMAIL_HOST",
      "value": "${EMAIL_HOST}"
    },
    {
      "key": "EMAIL_PORT",
      "value": "${EMAIL_PORT}"
    },
    {
      "key": "EMAIL_USERNAME",
      "value": "${EMAIL_USERNAME}"
    },
    {
      "key": "EMAIL_PASSWORD",
      "value": "${EMAIL_PASSWORD}"
    },
    {
      "key": "ALERT_EMAIL_TO",
      "value": "${ALERT_EMAIL_TO}"
    }
  ],
  "startupCommand": "/app/startup.sh"
}
EOF

# Step 5: Deploy to RunPod
echo -e "${YELLOW}Deploying to RunPod...${NC}"

# Login to RunPod
$RUNPODCTL config --apiKey $RUNPOD_API_KEY

# Create a pod with the configuration
echo -e "${YELLOW}Creating RunPod instance...${NC}"
POD_ID=$($RUNPODCTL create pod \
    --name "swaggy-stacks-trading" \
    --imageName "swaggy-stacks-trading:latest" \
    --gpuCount 0 \
    --containerDiskSize 20 \
    --volumeSize 10 \
    --mem 4 \
    --vcpu 2 \
    --ports "8000/http,9090/http" \
    --env "ENVIRONMENT=production" \
    --env "ALPACA_API_KEY=${ALPACA_API_KEY}" \
    --env "ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}" \
    --env "POSTGRES_PASSWORD=${POSTGRES_PASSWORD}" \
    --env "SECRET_KEY=${SECRET_KEY}" \
    --cost 0.10 | grep -oP 'Pod ID: \K\S+')

if [ -z "$POD_ID" ]; then
    echo -e "${RED}Failed to create pod${NC}"
    exit 1
fi

echo -e "${GREEN}Pod created with ID: $POD_ID${NC}"

# Step 6: Wait for pod to be ready
echo -e "${YELLOW}Waiting for pod to be ready...${NC}"
MAX_WAIT=300  # 5 minutes
WAIT_TIME=0
while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    STATUS=$($RUNPODCTL get pod $POD_ID | grep -oP 'Status: \K\S+')
    if [ "$STATUS" = "RUNNING" ]; then
        echo -e "${GREEN}Pod is running!${NC}"
        break
    fi
    echo "Current status: $STATUS (waiting...)"
    sleep 10
    WAIT_TIME=$((WAIT_TIME + 10))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo -e "${RED}Timeout waiting for pod to start${NC}"
    exit 1
fi

# Step 7: Get pod endpoint
echo -e "${YELLOW}Getting pod endpoint...${NC}"
ENDPOINT=$($RUNPODCTL get pod $POD_ID | grep -oP 'Endpoint: \K\S+')

if [ -z "$ENDPOINT" ]; then
    echo -e "${RED}Failed to get endpoint${NC}"
    exit 1
fi

# Step 8: Display deployment information
echo ""
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo ""
echo "Pod ID: $POD_ID"
echo "Endpoint: $ENDPOINT"
echo ""
echo "API URL: https://$ENDPOINT-8000.proxy.runpod.net"
echo "Metrics URL: https://$ENDPOINT-9090.proxy.runpod.net"
echo ""
echo "To check logs:"
echo "  $RUNPODCTL logs $POD_ID"
echo ""
echo "To SSH into the pod:"
echo "  $RUNPODCTL ssh $POD_ID"
echo ""
echo "To stop the pod:"
echo "  $RUNPODCTL stop pod $POD_ID"
echo ""
echo "To delete the pod:"
echo "  $RUNPODCTL delete pod $POD_ID"
echo ""
echo -e "${GREEN}Your trading system is now running autonomously on RunPod!${NC}"

# Step 9: Health check
echo ""
echo -e "${YELLOW}Running health check...${NC}"
sleep 30  # Give the service time to start

HEALTH_URL="https://$ENDPOINT-8000.proxy.runpod.net/health"
if curl -f -s "$HEALTH_URL" > /dev/null; then
    echo -e "${GREEN}Health check passed! System is operational.${NC}"
else
    echo -e "${YELLOW}Health check failed. The system might still be starting up.${NC}"
    echo "Check the logs with: $RUNPODCTL logs $POD_ID"
fi

# Save deployment info
echo "$POD_ID" > .runpod-deployment-id
echo "$ENDPOINT" > .runpod-endpoint

echo ""
echo "Deployment information saved to .runpod-deployment-id and .runpod-endpoint"