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
uvicorn app.main:app --reload

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

- `app/analysis/enhanced_markov_system.py` - Core Markov chain trading analysis engine
- `app/trading/alpaca_client.py` - Alpaca API integration for order execution
- `app/trading/risk_manager.py` - Risk management and position sizing
- `app/api/v1/endpoints/trading.py` - Trading API endpoints
- `app/core/config.py` - Configuration settings (database, trading parameters, API keys)
- `app/monitoring/metrics.py` - Comprehensive Prometheus metrics (50+ trading-specific metrics)
- `app/monitoring/alerts.py` - Multi-channel alert system (email notifications)

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

### Enterprise Monitoring Dashboard System

Comprehensive 6-dashboard ecosystem providing 360-degree trading system visibility:

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

**Features**: Real-time updates (5-10s refresh), template variables for filtering, cross-dashboard navigation, PDF/PNG export capabilities

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

## 📋 **Ready for Tomorrow's Development**

### ✅ **Systems Configured & Ready**
- **TaskMaster-AI**: Initialized and configured with Claude-3.5-Sonnet model
- **Shrimp Task Manager**: Ready for tactical task breakdown and execution
- **Serena MCP**: Active for intelligent codebase navigation and editing
- **MCP Memory**: Knowledge graph system ready for entity tracking
- **Database**: All 9 tables initialized with PostgreSQL + Redis operational
- **Testing**: Complete Jest + Playwright test infrastructure configured
- **Python Environment**: Python 3.13 virtual environment with updated requirements

### 🎯 **Recommended Tomorrow Workflow**
1. **Start**: `mcp__taskmaster-ai__next_task` - Get the next strategic objective
2. **Plan**: `mcp__shrimp-task-manager__plan_task` - Break it down tactically
3. **Execute**: `mcp__shrimp-task-manager__execute_task` - Get step-by-step guidance
4. **Code**: Use Serena for precise navigation and edits
5. **Verify**: `mcp__shrimp-task-manager__verify_task` - Ensure quality (80+ score)
6. **Document**: Capture insights in Serena memories for the team

### 🚀 **Advanced Mode Active**
All systems are configured for **advanced mode integration**:
- Cross-system task references and dependencies
- Shared context and memory between tools
- Research-backed task analysis and planning
- Comprehensive verification and quality assurance

**The team is ready to deliver high-quality, architected solutions with full AI assistance! 💪**

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
