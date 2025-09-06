#!/bin/bash

# Swaggy Stacks Setup Script
echo "🚀 Setting up Swaggy Stacks - Advanced Markov Trading System"
echo "=============================================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp env.example .env
    echo "✅ .env file created. Please edit it with your API keys."
    echo ""
    echo "🔑 Required: Update ALPACA_SECRET_KEY in .env file"
    echo "   Your API Key: PKHUUXJV4V04PQ86MNPR"
    echo "   Get your Secret Key from: https://app.alpaca.markets/"
    echo ""
    read -p "Press Enter after updating your .env file..."
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "🐳 Starting Docker services..."
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check PostgreSQL
if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL is not ready"
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis is not ready"
fi

# Check Backend
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend API is ready"
else
    echo "⏳ Backend API is starting up..."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📊 Access your application:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo "   Grafana: http://localhost:3001 (admin/admin)"
echo ""
echo "🧪 Test your Alpaca connection:"
echo "   python scripts/test_alpaca.py"
echo ""
echo "📝 View logs:"
echo "   docker-compose logs -f"
