# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Local Development
```bash
# Backend (FastAPI)
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend (Next.js)  
cd frontend
npm install
npm run dev
```

### Docker Development
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Testing
```bash
# Backend tests
cd backend
pytest tests/ -v --cov=app

# Frontend tests
cd frontend  
npm test
```

### Code Quality
```bash
# Backend linting
cd backend
black app/ --check
isort app/ --check
flake8 app/
mypy app/

# Frontend linting
cd frontend
npm run lint
npm run type-check
```

## Architecture Overview

**Swaggy Stacks** is a production-ready algorithmic trading system built with a microservices architecture:

- **Backend**: FastAPI application with PostgreSQL, Redis, and Celery for background tasks
- **Frontend**: Next.js TypeScript app with Tailwind CSS and Shadcn UI components
- **Trading Engine**: Alpaca API integration for paper trading with Markov chain analysis
- **Monitoring**: Prometheus metrics with Grafana dashboards

### Key Backend Components

- `app/analysis/enhanced_markov_system.py` - Core Markov chain trading analysis engine
- `app/trading/alpaca_client.py` - Alpaca API integration for order execution
- `app/trading/risk_manager.py` - Risk management and position sizing
- `app/api/v1/endpoints/trading.py` - Trading API endpoints
- `app/core/config.py` - Configuration settings (database, trading parameters, API keys)

### Database Architecture

The system uses PostgreSQL for persistent data storage with SQLAlchemy ORM:
- Market data and historical prices
- User accounts and portfolios  
- Trading orders and execution history
- Strategy performance metrics

Redis is used for:
- Session storage and caching
- Real-time market data streaming
- Celery task queue for background jobs

### Trading System Architecture

The trading engine implements:
- **Enhanced Markov Analysis**: Multi-state regime detection with volatility and volume analysis
- **Risk Management**: Portfolio exposure limits, daily loss limits, position sizing
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Fibonacci levels integrated with Markov states
- **Paper Trading**: Alpaca API integration for safe strategy testing

### Environment Configuration

Required environment variables in `.env`:
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` - Alpaca trading API credentials
- `SECRET_KEY` - JWT token signing key
- Database credentials (POSTGRES_*)
- Redis URL

### Service Ports

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000  
- API Docs: http://localhost:8000/docs
- Grafana: http://localhost:3001
- Prometheus: http://localhost:9090
- PostgreSQL: localhost:5432
- Redis: localhost:6379

## Development Guidelines

- All trading operations default to paper trading mode for safety
- The system uses structured logging (structlog) with JSON output
- Database migrations are handled with Alembic
- Background tasks (market data updates, analysis) run via Celery workers
- API follows RESTful conventions with comprehensive OpenAPI documentation
- Frontend uses TypeScript with strict type checking enabled

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
