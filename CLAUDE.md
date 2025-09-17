# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Local Development
```bash
# Backend (Python 3.13 + FastAPI)
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Development server
uvicorn app.main:app --reload

# Production AI trading system
python3 run_production.py

# Frontend (Next.js 14 + TypeScript)
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

# Frontend tests (Jest + React Testing Library)
cd frontend
npm run test:ci              # Run all unit tests with coverage
npm run test                 # Watch mode for development

# End-to-end tests (Playwright)
npm run test:e2e            # Headless e2e tests
npm run test:e2e:headed     # Run with browser visible

# Run all tests
npm run test:all            # Unit tests + E2E tests
```

### Code Quality
```bash
# Backend linting
cd backend
black app/ --check
isort app/ --check
flake8 app/
mypy app/

# Structure validation (after cleanup)
python3 backend/scripts/verification/validate_structure.py

# Frontend linting
cd frontend
npm run lint
npm run type-check
```

## Architecture Overview

**Swaggy Stacks** is a production-ready algorithmic trading system built with a microservices architecture:

- **Backend**: FastAPI application with PostgreSQL, Redis, and Celery for background tasks
- **Frontend**: Next.js TypeScript app with Tailwind CSS and Shadcn UI components
- **Trading Engine**: Alpaca API integration for paper trading with enhanced Markov chain analysis
- **Monitoring**: Enterprise-grade monitoring with 6 comprehensive Grafana dashboards and 50+ Prometheus metrics
- **AI Integration**: Full MCP (Model Context Protocol) ecosystem with 4 specialized agents
- **Risk Management**: Advanced portfolio risk analysis with real-time alert notifications

### Key Backend Components

#### Core Trading & Analysis
- `app/analysis/consolidated_markov_system.py` - Enhanced Markov chain trading analysis engine
- `app/indicators/technical_indicators.py` - Technical analysis calculations
- `app/indicators/modern_indicators.py` - Advanced technical indicators
- `app/ml/llm_predictors.py` - LLM-based market predictions
- `app/analysis/pattern_validation_framework.py` - Pattern validation logic
- `app/analysis/integrated_alpha_detector.py` - Alpha signal detection

#### Unsupervised Learning System
- `app/ml/unsupervised/clustering/` - K-means, DBSCAN, hierarchical clustering algorithms
- `app/ml/unsupervised/dimensionality/` - PCA, autoencoders, t-SNE implementations
- `app/ml/unsupervised/market_regime_detector.py` - AI-powered market regime detection
- `app/ml/unsupervised/anomaly_detector.py` - DBSCAN-based anomaly detection for black swan events
- `app/ml/unsupervised/pattern_memory.py` - Pattern storage and mining with autoencoders
- `app/ml/unsupervised/self_learning_loop.py` - Autonomous learning and strategy evolution

#### Trading Execution & Risk
- `app/trading/alpaca_client.py` - Alpaca API integration for order execution
- `app/trading/risk_manager.py` - Risk management and position sizing
- `app/trading/trading_manager.py` - Centralized trading operations (Singleton pattern)
- `app/trading/order_manager.py` - Order lifecycle management
- `app/api/v1/endpoints/trading.py` - Trading API endpoints

#### Infrastructure & Monitoring
- `app/core/config.py` - Configuration settings (database, trading parameters, API keys)
- `app/core/database.py` - Optimized database configuration with connection pooling
- `app/monitoring/metrics.py` - Comprehensive Prometheus metrics (50+ trading-specific metrics)
- `app/monitoring/alerts.py` - Multi-channel alert system with Prometheus integration
- `app/monitoring/health_checks.py` - System-wide health monitoring and status compilation

### Module Architecture (Post-Refactoring)

The system follows clean separation of concerns with focused modules:

- **`app/indicators/`** - Pure technical analysis and indicator calculations
  - `technical_indicators.py` - Core indicators (RSI, MACD, Bollinger Bands)
  - `modern_indicators.py` - Advanced indicators from modern trading literature
  - `indicator_factory.py` - Factory pattern for indicator creation

- **`app/ml/`** - Machine learning and predictive models
  - `llm_predictors.py` - LLM-based market predictions with Chinese models
  - `prediction_prompts/` - Structured prompt templates for different models
  - **`unsupervised/`** - Comprehensive unsupervised learning infrastructure
    - `clustering/` - K-means, DBSCAN, hierarchical clustering with production optimizations
    - `dimensionality/` - PCA, autoencoders, t-SNE for feature engineering
    - `market_regime_detector.py` - AI-powered market regime identification
    - `anomaly_detector.py` - Real-time anomaly detection for risk management
    - `pattern_memory.py` - Pattern storage, retrieval, and mining system
    - `self_learning_loop.py` - Autonomous agent experience clustering and improvement

- **`app/analysis/`** - Pattern validation and alpha detection
  - `consolidated_markov_system.py` - Enhanced Markov chain analysis with technical indicators
  - `pattern_validation_framework.py` - Trading pattern validation logic
  - `integrated_alpha_detector.py` - Alpha signal detection and analysis
  - `tool_feedback_tracker.py` - Analysis feedback and learning system

- **`app/trading/`** - Order execution and risk management
  - `alpaca_client.py` - Alpaca API integration
  - `risk_manager.py` - Risk management and position sizing
  - `trading_manager.py` - Centralized trading operations
  - `order_manager.py` - Order lifecycle management

- **`app/core/`** - Shared utilities and optimized infrastructure
  - `database.py` - Optimized database configuration with QueuePool (20+30 connections)
  - `config.py` - Configuration settings and environment management
  - `exceptions.py` - Custom exception handling

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

### Database Performance Optimizations

Recent migration (003_optimize_database_performance.py) adds:
- **23+ Strategic Indexes**: Composite, partial, and GIN indexes for optimal query performance
- **Time-Series Optimization**: Specialized indexes for trading data queries (50-80% performance improvement)
- **16 Data Integrity Constraints**: Ensures data quality and consistency
- **2 Materialized Views**: Real-time P&L calculations and indicator performance summaries
- **Connection Pooling**: QueuePool with 20 base + 30 overflow connections for high-throughput trading
- **PostgreSQL Session Optimizations**: Trading workload-specific database settings

### Trading System Architecture

The trading engine implements:
- **Enhanced Markov Analysis**: Multi-state regime detection with volatility and volume analysis
- **Risk Management**: Portfolio exposure limits, daily loss limits, position sizing
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Fibonacci levels integrated with Markov states
- **Paper Trading**: Alpaca API integration for safe strategy testing

### Unsupervised Learning System Architecture

**SwaggyStacks** now features a comprehensive unsupervised learning infrastructure that enables autonomous market understanding and strategy evolution:

#### Core Clustering Algorithms (`app/ml/unsupervised/clustering/`)
- **K-means Clustering**: Market condition segmentation with production-ready optimizations
- **DBSCAN**: Density-based clustering for anomaly detection and regime identification
- **Hierarchical Clustering**: Multi-level market structure analysis with dendrogram visualization
- **Production Features**: Parallel processing, memory optimization, real-time streaming support

#### Dimensionality Reduction (`app/ml/unsupervised/dimensionality/`)
- **PCA (Principal Component Analysis)**: Feature reduction and noise filtering for market data
- **Autoencoders**: Deep learning-based feature extraction and pattern compression
- **t-SNE**: High-dimensional market data visualization and cluster analysis
- **Applications**: Feature engineering, data visualization, compression for real-time processing

#### Market Regime Detection (`market_regime_detector.py`)
- **AI-Powered Classification**: Automatic identification of bull, bear, sideways, and volatile markets
- **Multi-Factor Analysis**: Combines price action, volume, volatility, and sentiment indicators
- **Real-Time Processing**: Streaming regime detection with configurable update intervals
- **Trading Integration**: Direct integration with AI agents for regime-aware strategy selection

#### Anomaly Detection System (`anomaly_detector.py`)
- **DBSCAN-Based Detection**: Identification of unusual market conditions and potential black swan events
- **Risk Integration**: Automatic risk adjustment when anomalies are detected
- **Alert System**: Real-time notifications for significant market anomalies
- **Historical Analysis**: Pattern recognition for recurring anomalous conditions

#### Pattern Memory & Mining (`pattern_memory.py`)
- **Autoencoder Storage**: Compressed pattern storage using neural network representations
- **Pattern Retrieval**: Efficient similarity search for historical pattern matching
- **Market Basket Analysis**: Identification of co-occurring market conditions and events
- **Strategy Insights**: Pattern-based strategy recommendation and optimization

#### Self-Learning Feedback Loop (`self_learning_loop.py`)
- **Experience Clustering**: AI agents automatically cluster their trading experiences
- **Strategy Evolution**: Continuous improvement without manual intervention
- **Performance Tracking**: Automated assessment of strategy effectiveness
- **Adaptive Learning**: Dynamic adjustment of learning parameters based on market conditions

#### Integration with AI Trading System
- **Enhanced AI Agents**: All trading agents now consume unsupervised learning insights
- **Regime-Aware Strategies**: Automatic strategy selection based on detected market regimes
- **Anomaly Response**: Coordinated risk reduction during detected anomalous conditions
- **Pattern-Based Decisions**: Historical pattern matching influences current trading decisions

### Enhanced Monitoring & Alerting System

**SwaggyStacks** features a comprehensive monitoring ecosystem with real-time dashboards, proactive alerting, and AI agent coordination tracking:

#### Grafana Dashboard Ecosystem (6 Dashboards)
Comprehensive 360-degree trading system visibility with real-time updates:

1. **P&L Dashboard** (`pnl_dashboard.json`) - Real-time portfolio performance
   - Portfolio value tracking, daily P&L trends, position analysis
   - Dynamic symbol and strategy filtering

2. **Strategy Performance Dashboard** (`strategy_dashboard.json`) - Strategy analysis
   - Win/loss ratios, success rates, drawdown analysis
   - Multi-strategy comparison and performance metrics

3. **Trade Execution Dashboard** (`execution_dashboard.json`) - Execution monitoring
   - Success rates, latency metrics, failure analysis
   - Order type and symbol filtering capabilities

4. **Risk Dashboard** (`risk_dashboard.json`) - Portfolio risk monitoring
   - Exposure analysis, concentration risk, beta analysis
   - Sector and symbol-based risk breakdown

5. **System Health Dashboard** (`system_health_dashboard.json`) - Infrastructure monitoring
   - System health status, component monitoring, MCP agent coordination
   - Database performance, Redis operations, API latency tracking

6. **Advanced Risk Dashboard** (`advanced_risk_dashboard.json`) - Comprehensive risk analysis
   - VaR calculations, correlation analysis, volatility tracking
   - Risk-adjusted returns, concentration risk, alert threshold monitoring

#### Prometheus Metrics System (50+ Metrics)
- **System Health**: Overall system status, component health, uptime tracking
- **MCP Agent Coordination**: Success rates, response times, queue depth, server status
- **Trading Performance**: Portfolio values, P&L tracking, order execution metrics
- **Infrastructure**: Database connection pools, Redis performance, HTTP request metrics
- **AI Processing**: Market sentiment, processing durations, model performance

#### Enhanced AlertManager Integration
**Intelligent Prometheus-based alerting with 16 proactive alert rules:**

##### System Health Alerts (2 rules)
- **system_health_degraded**: System health status below healthy threshold
- **system_uptime_low**: System uptime below 5 minutes (critical restart indicator)

##### MCP Agent Coordination Alerts (4 rules)
- **mcp_agent_coordination_failure**: Success rate below 80%
- **mcp_agent_high_response_time**: Response time exceeding 5 seconds
- **mcp_agent_queue_depth_high**: Queue depth above 50 pending operations
- **mcp_server_unavailable**: MCP server connection failures

##### Trading System Alerts (2 rules)
- **trading_portfolio_value_drop**: Portfolio value decline exceeding 5%
- **trading_orders_failure_rate_high**: Order failure rate above 10%

##### Infrastructure Performance Alerts (6 rules)
- **db_connection_pool_exhausted**: Database connections below 2 available
- **redis_response_time_high**: Redis operations exceeding 100ms
- **http_request_duration_high**: API response times above 5 seconds
- **ai_processing_duration_high**: AI processing exceeding 30 seconds

#### Real-Time WebSocket Integration
- **Live System Health**: 30-second health status updates via WebSocket
- **Trading Dashboard**: Real-time portfolio, P&L, and position updates
- **Alert Streaming**: Immediate alert notifications to connected clients
- **Performance Metrics**: Live infrastructure and trading performance data

#### Alert Delivery Channels
- **Multi-Channel Support**: LOG, WEBHOOK, EMAIL notification methods
- **Intelligent Cooldowns**: Severity-based cooldown periods (2-60 minutes)
- **Alert Deduplication**: Prevents alert spam while maintaining visibility
- **Historical Tracking**: Complete alert history for post-incident analysis

**Features**: Real-time updates (5-10s refresh), template variables for filtering, cross-dashboard navigation, proactive alerting, PDF/PNG export capabilities

### Environment Configuration

Required environment variables in `.env`:
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` - Alpaca trading API credentials
- `SECRET_KEY` - JWT token signing key
- Database credentials (POSTGRES_*)
- Redis URL
- Email notification settings (EMAIL_HOST, EMAIL_USERNAME, EMAIL_PASSWORD, ALERT_EMAIL_TO)
- MCP server API keys (ANTHROPIC_API_KEY, PERPLEXITY_API_KEY for research features)

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

## Integrated MCP Workflow for Enhanced Development 🚀

**SwaggyStacks** now features a comprehensive AI-powered development ecosystem with four integrated MCP (Model Context Protocol) systems working together:

### 🎯 **TaskMaster-AI** - Strategic Project Management
- **Purpose**: High-level task planning, PRD parsing, and complexity analysis
- **Best For**: Breaking down features into tasks, analyzing project complexity
- **Key Commands**: `parse_prd`, `analyze_project_complexity`, `expand_task`, `get_tasks`
- **API Configuration**: Anthropic Claude models (configured with API key)

### 🦐 **Shrimp Task Manager** - Tactical Task Execution  
- **Purpose**: Granular task breakdown, dependency management, step-by-step execution
- **Best For**: Converting high-level tasks into executable subtasks with clear verification
- **Key Commands**: `plan_task`, `analyze_task`, `split_tasks`, `execute_task`, `verify_task`
- **Specialties**: Code implementation guidance, task verification, dependency tracking

### 🧠 **Serena** - Intelligent Codebase Navigation
- **Purpose**: Semantic code search, symbol-based editing, memory management
- **Best For**: Understanding existing code, making precise edits, architectural insights
- **Key Commands**: `find_symbol`, `get_symbols_overview`, `replace_symbol_body`, `search_for_pattern`
- **Memory System**: Project insights, patterns, and architectural decisions

### 💾 **MCP Memory** - Knowledge Graph Management
- **Purpose**: Entity-relationship tracking, cross-project insights, knowledge persistence
- **Best For**: Tracking project relationships, maintaining context across sessions
- **Key Commands**: `create_entities`, `add_observations`, `search_nodes`, `read_graph`

---

## 🔄 **Integrated Development Workflow**

### **Phase 1: Strategic Planning** 
```bash
# TaskMaster-AI: Parse requirements and create strategic tasks
mcp__taskmaster-ai__parse_prd
mcp__taskmaster-ai__analyze_project_complexity --research
mcp__taskmaster-ai__expand_all --research

# Review high-level tasks
mcp__taskmaster-ai__get_tasks --withSubtasks
```

### **Phase 2: Tactical Breakdown**
```bash
# Shrimp: Convert strategic tasks into executable subtasks
mcp__shrimp-task-manager__plan_task
mcp__shrimp-task-manager__analyze_task  
mcp__shrimp-task-manager__split_tasks

# Review detailed implementation plan
mcp__shrimp-task-manager__list_tasks --status=all
```

### **Phase 3: Code Implementation**
```bash
# Serena: Navigate and understand existing codebase
mcp__serena__get_symbols_overview
mcp__serena__find_symbol --include_body=true
mcp__serena__search_for_pattern

# Shrimp: Execute tasks with step-by-step guidance
mcp__shrimp-task-manager__execute_task
mcp__shrimp-task-manager__verify_task

# Serena: Make precise code changes
mcp__serena__replace_symbol_body
mcp__serena__insert_after_symbol
```

### **Phase 4: Knowledge Capture**
```bash
# Serena: Document architectural insights
mcp__serena__write_memory

# MCP Memory: Track project entities and relationships
mcp__memory__create_entities
mcp__memory__add_observations

# TaskMaster-AI: Update task progress
mcp__taskmaster-ai__set_task_status --status=done
```

---

## 🎯 **Best Practices for Team Development**

### **Morning Standup Workflow**
1. **TaskMaster-AI**: `get_tasks --status=pending` - Review today's strategic objectives
2. **Shrimp**: `list_tasks --status=pending` - Check tactical implementation tasks  
3. **Serena**: `list_memories` - Review recent architectural insights
4. **Start with**: `mcp__shrimp-task-manager__execute_task` for immediate guidance

### **Feature Development Cycle**
1. **Planning**: TaskMaster-AI parses requirements → generates strategic tasks
2. **Analysis**: Shrimp breaks down tasks → creates implementation plan
3. **Research**: Serena explores codebase → provides architectural context  
4. **Implementation**: Shrimp guides execution → Serena makes precise edits
5. **Documentation**: Serena captures insights → Memory tracks relationships

### **Code Review & Quality Assurance**
1. **Shrimp**: `verify_task` - Comprehensive task completion verification
2. **Serena**: `find_referencing_symbols` - Impact analysis of changes
3. **TaskMaster-AI**: `complexity_report` - Assess overall project health
4. **Memory**: `search_nodes` - Check for related architectural decisions

---

## 🛠 **Development Commands Integration**

### **TaskMaster-AI Configuration**
- **Configured**: ✅ Anthropic API key, Claude-3.5-Sonnet model
- **Ready**: Strategic planning, PRD parsing, complexity analysis
- **Location**: `.taskmaster/tasks/tasks.json` - Strategic task database

### **Advanced Features Available**
- **Research Mode**: Both TaskMaster-AI and Shrimp support research-backed analysis
- **Memory Integration**: Serena and MCP Memory provide persistent knowledge
- **Dependency Tracking**: Smart task dependencies across both systems
- **Verification System**: Comprehensive task completion validation

### **Team Coordination Commands**
```bash
# Quick status across all systems
mcp__taskmaster-ai__next_task          # Get next strategic task
mcp__shrimp-task-manager__list_tasks   # See tactical breakdown
mcp__serena__list_memories             # Review architectural insights
mcp__memory__read_graph                # Check knowledge graph

# Deep work session
mcp__shrimp-task-manager__execute_task # Get detailed implementation guidance
# Follow the step-by-step instructions provided
mcp__serena__find_symbol               # Navigate to specific code
mcp__serena__replace_symbol_body       # Make precise changes
mcp__shrimp-task-manager__verify_task  # Verify completion (80+ score)
```

---

## 📋 **Current Development Status**

### ✅ **Systems Configured & Production Ready**
- **TaskMaster-AI**: Initialized and configured with Claude-3.5-Sonnet model for strategic planning
- **Shrimp Task Manager**: Clean slate - ready for new tactical task breakdown and execution
- **Serena MCP**: Active with comprehensive project memory and architectural insights
- **MCP Memory**: Knowledge graph populated with project relationships and context
- **Database**: All 9 tables initialized with PostgreSQL + Redis operational
- **Testing**: Complete Jest + Playwright test infrastructure configured
- **Python Environment**: Python 3.13 virtual environment with updated requirements
- **Unsupervised Learning**: Complete AI-powered market analysis and self-learning system
- **Enhanced Monitoring**: 50+ Prometheus metrics with proactive AlertManager integration

### 🎯 **Development Workflow (Post-Implementation)**
**Recent Achievement**: Successfully completed comprehensive unsupervised learning system with 9 major tasks:
- ✅ Core clustering algorithms (K-means, DBSCAN, hierarchical)
- ✅ Dimensionality reduction (PCA, autoencoders, t-SNE)
- ✅ Market regime detection with AI-powered classification
- ✅ Anomaly detection for black swan event identification
- ✅ Pattern memory and mining with autoencoder storage
- ✅ Self-learning feedback loop with experience clustering
- ✅ AI agent integration with unsupervised insights
- ✅ Comprehensive monitoring and testing infrastructure

**Task Management Status**:
- **Shrimp Task Manager**: Cleared and ready for new projects (previous tasks backed up to `tasks_memory_2025-09-14T23-13-03.json`)
- **TaskMaster-AI**: Strategic task repository available for high-level planning
- **Serena Memory**: 10 comprehensive memory files with architectural insights and implementation patterns

### 🚀 **Ready for Next Development Phase**
All systems are configured for **advanced mode integration**:
- Cross-system task references and dependencies
- Shared context and memory between tools
- Research-backed task analysis and planning
- Comprehensive verification and quality assurance
- Complete unsupervised learning infrastructure
- Real-time monitoring with proactive alerting

**The system is production-ready with autonomous AI capabilities and comprehensive monitoring! 🎯**

---

## 🧠 **Memory Management & Knowledge Integration**

### **Serena MCP Memory System**
**SwaggyStacks** maintains comprehensive project knowledge through Serena's intelligent memory system:

#### **Current Memory Files (10 Active)**
1. **`project_overview`** - High-level project status and architectural achievements
2. **`tech_stack_and_architecture`** - Core technology stack and design patterns
3. **`development_commands`** - Essential development workflows and commands
4. **`code_organization_and_patterns`** - Project structure and implementation patterns
5. **`trading_algorithms_and_strategies`** - Trading system implementation details
6. **`monitoring-architecture-analysis`** - Real-time monitoring system analysis
7. **`advanced_task_management_integration`** - Multi-system task management workflows
8. **`grafana-dashboard-architecture`** - Dashboard configuration and metrics
9. **`enhanced-monitoring-architecture-analysis`** - Enhanced monitoring capabilities
10. **`alertmanager-prometheus-integration`** - AlertManager configuration and rules

#### **Memory Integration Workflow**
- **Automatic Knowledge Capture**: Serena automatically stores architectural insights during development
- **Cross-Session Context**: Memory files maintain context across different development sessions
- **Pattern Recognition**: Accumulated knowledge helps identify recurring patterns and solutions
- **Decision Support**: Historical insights inform current development decisions
- **Team Knowledge Sharing**: Memory files serve as living documentation for team collaboration

#### **MCP Memory Knowledge Graph**
- **Entity Tracking**: Project components, relationships, and dependencies stored as entities
- **Observation Management**: Implementation details and decisions tracked as observations
- **Relationship Mapping**: Cross-system connections and integration points documented
- **Context Preservation**: Project context maintained across task management systems

### **Memory-Driven Development Benefits**
- **Accelerated Onboarding**: New team members access comprehensive project knowledge
- **Consistent Architecture**: Previous decisions and patterns guide new implementations
- **Reduced Redundancy**: Avoid re-solving previously addressed problems
- **Quality Assurance**: Historical insights help identify potential issues early
- **Continuous Learning**: System learns from past implementations and decisions

---

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
