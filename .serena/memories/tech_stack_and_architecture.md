# Technology Stack and Architecture

## Core Technology Stack

### Backend Stack
- **Framework**: FastAPI (Python) - High-performance async API framework
- **Database**: PostgreSQL with pgvector extension for vector operations
- **Caching/Queue**: Redis for session storage, caching, and Celery message broker
- **Background Tasks**: Celery with Redis broker for market data updates and analysis
- **ORM**: SQLAlchemy with Alembic for database migrations
- **Async Support**: Full asyncio integration throughout the application

### Frontend Stack
- **Framework**: Next.js with TypeScript for type safety
- **Styling**: Tailwind CSS with Shadcn UI component library
- **Build**: Modern React with strict TypeScript configuration
- **Real-time**: WebSocket integration for live market data

### Infrastructure & DevOps
- **Containerization**: Docker with multi-stage builds
- **Orchestration**: Docker Compose for local development
- **CI/CD**: GitHub Actions with automated testing and deployment
- **Monitoring**: Prometheus for metrics collection, Grafana for dashboards
- **Security**: Structured logging with security considerations

### Trading & Market Data
- **Broker API**: Alpaca API for paper trading execution
- **Market Data**: Yahoo Finance (yfinance) and multiple fallback sources
- **Analysis**: Custom implementations of Markov chains, technical indicators
- **Risk Management**: Position sizing algorithms, exposure limits

## Architectural Patterns

### Microservices Architecture
Services communicate via API calls and message queues:
- **Backend API Service**: Core trading logic and API endpoints
- **Database Service**: PostgreSQL with connection pooling
- **Cache Service**: Redis for high-speed data access
- **Worker Service**: Celery for background processing
- **Monitoring Services**: Prometheus + Grafana stack

### Key Design Patterns (Post Phase 1 Consolidation)

#### Singleton Pattern
- **TradingManager**: Centralized trading operations management
- **MCPOrchestrator**: Planned central MCP server coordination

#### Plugin Pattern  
- **ConsolidatedStrategyAgent**: Pluggable trading strategies
- **Strategy Plugins**: Markov, Wyckoff, Fibonacci, Elliott Wave

#### Dependency Injection
- **FastAPI Dependencies**: Shared resources via Depends()
- **Database Sessions**: Automatic session management
- **Authentication**: Centralized user management

#### Repository Pattern
- **Data Access**: Clean separation between business logic and data access
- **Model Abstraction**: SQLAlchemy models with business logic separation

### Service Communication
- **Synchronous**: HTTP/REST API between frontend and backend
- **Asynchronous**: Celery tasks for heavy computations
- **Real-time**: WebSocket connections for live updates
- **Caching**: Redis for frequently accessed data