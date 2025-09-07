# Swaggy Stacks - Project Overview

## Purpose
Swaggy Stacks is a production-ready algorithmic trading system that integrates multiple advanced trading strategies including Markov chains, Fibonacci analysis, Elliott Wave theory, and Wyckoff method. The system provides real-time paper trading capabilities, comprehensive risk management, and API monetization features.

## Key Features
- **Real-time Market Data Integration**: Live market data processing with Redis caching
- **Advanced Trading Algorithms**: Enhanced Markov chain analysis, Elliott Wave, Fibonacci retracements, Wyckoff method
- **Paper Trading Engine**: Safe strategy testing using Alpaca API integration
- **Risk Management**: Portfolio exposure limits, position sizing, daily loss controls
- **Web Dashboard**: Real-time performance monitoring and trading interface
- **API Monetization**: Multi-tier subscription plans with usage-based pricing
- **Deep RL Components**: Reinforcement learning integration for strategy optimization

## System Architecture
**Microservices Architecture** with Docker Compose orchestration:
- **Backend**: FastAPI with PostgreSQL, Redis, Celery background tasks
- **Frontend**: Next.js TypeScript with Tailwind CSS and Shadcn UI
- **Trading Engine**: Alpaca API integration for order execution
- **Monitoring**: Prometheus metrics with Grafana dashboards
- **Background Processing**: Celery workers for market data updates and analysis

## Recent Major Improvements (Phase 1 Consolidation)
- **Code Consolidation**: Eliminated 1800+ lines of redundant code
- **Singleton TradingManager**: Centralized trading operations (~540 lines)
- **Plugin Strategy Agents**: Consolidated 4 separate agents into unified system (~660 lines)
- **Centralized Logging**: Structured logging with consistency across all modules
- **Shared API Models**: Eliminated duplicate Pydantic model definitions
- **Import Optimization**: Common imports module reducing redundancy

## Safety & Education Focus
The system is designed primarily for paper trading and educational purposes, emphasizing risk management and safe strategy testing before any real-money applications.