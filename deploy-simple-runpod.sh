#!/bin/bash

# Simple RunPod deployment using pre-built image
set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check API key
if [ -z "$RUNPOD_API_KEY" ]; then
    echo -e "${RED}Error: RUNPOD_API_KEY not found in .env file${NC}"
    exit 1
fi

echo -e "${GREEN}=== Simple RunPod Deployment ===${NC}"
echo ""

# Configure RunPod CLI
./runpodctl config --apiKey $RUNPOD_API_KEY

# Create pod with Python base image
echo -e "${YELLOW}Creating RunPod pod with Python base image...${NC}"
POD_OUTPUT=$(./runpodctl create pod \
    --name "swaggy-test" \
    --imageName "python:3.11-slim" \
    --gpuType "NVIDIA GeForce RTX 3090" \
    --gpuCount 1 \
    --containerDiskSize 20 \
    --volumeSize 10 \
    --mem 8 \
    --vcpu 4 \
    --ports "8000/http" \
    --cost 0.50 \
    --communityCloud 2>&1)

echo "$POD_OUTPUT"

# Extract pod ID
POD_ID=$(echo "$POD_OUTPUT" | sed -n 's/.*pod \([a-zA-Z0-9]*\) created.*/\1/p')

if [ -z "$POD_ID" ]; then
    echo -e "${RED}Failed to create pod. Output:${NC}"
    echo "$POD_OUTPUT"
    exit 1
fi

echo -e "${GREEN}Pod created with ID: $POD_ID${NC}"
echo ""
echo "To access the pod:"
echo "  ./runpodctl ssh $POD_ID"
echo ""
echo "To check status:"
echo "  ./runpodctl get pod $POD_ID"
echo ""
echo "To stop the pod:"
echo "  ./runpodctl stop pod $POD_ID"
echo ""
echo "To delete the pod:"
echo "  ./runpodctl delete pod $POD_ID"