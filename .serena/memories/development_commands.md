# Development Commands and Workflows

## Essential Development Commands

### Project Setup
```bash
# Clone and setup
git clone <repository-url>
cd swaggy-stacks

# Docker development (recommended)
docker-compose up -d                    # Start all services
docker-compose ps                       # Check service status  
docker-compose logs -f backend          # View backend logs
docker-compose logs -f frontend         # View frontend logs
docker-compose down                     # Stop all services

# Local development
cd backend && pip install -r requirements.txt
cd frontend && npm install
```

### Backend Development (FastAPI)
```bash
cd backend

# Start development server
uvicorn app.main:app --reload            # Hot reload enabled
uvicorn app.main:app --host 0.0.0.0 --port 8000  # Expose to network

# Virtual environment management
python -m venv venv
source venv/bin/activate                 # macOS/Linux
pip install -r requirements.txt

# Database operations
alembic upgrade head                     # Apply migrations
alembic revision --autogenerate -m "description"  # Create migration
```

### Frontend Development (Next.js)
```bash
cd frontend

# Development server
npm run dev                             # Start development server
npm run build                           # Production build
npm run start                           # Start production server

# Package management
npm install                             # Install dependencies
npm install <package>                   # Add new dependency
npm run type-check                      # TypeScript checking
```

### Code Quality and Linting

#### Backend Linting (Python)
```bash
cd backend

# Code formatting
black app/                              # Format code
black app/ --check                     # Check formatting without changes
isort app/                              # Sort imports  
isort app/ --check-only                 # Check import sorting

# Code quality checks
flake8 app/                             # PEP8 compliance
mypy app/                               # Static type checking

# All quality checks (recommended before commit)
black app/ --check && isort app/ --check && flake8 app/ && mypy app/
```

#### Frontend Linting (TypeScript)
```bash
cd frontend

# Linting and type checking
npm run lint                            # ESLint checking
npm run lint:fix                        # Fix ESLint issues
npm run type-check                      # TypeScript type checking

# All checks
npm run lint && npm run type-check
```

### Testing Commands

#### Backend Testing (pytest)
```bash
cd backend

# Run all tests
pytest tests/ -v                        # Verbose output
pytest tests/ -v --cov=app              # With coverage report
pytest tests/ -v --cov=app --cov-report=html  # HTML coverage report

# Test specific modules
pytest tests/test_trading.py -v         # Specific test file
pytest tests/test_trading.py::test_function -v  # Specific test function

# Integration tests (existing root-level tests)
python ../test_trading_simple.py        # Trading system tests
python ../test_ai_system.py             # AI system tests
python ../test_markov_mvp.py             # Markov analysis tests
```

#### Frontend Testing (Jest/React Testing Library)
```bash
cd frontend

# Run tests
npm test                                # Interactive test runner
npm test -- --coverage                 # With coverage
npm test -- --watchAll=false           # Single run (CI mode)
```

## Service Management

### Service Ports and Access
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (OpenAPI)
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

### Health Checks
```bash
# Service health
curl http://localhost:8000/health        # Backend health check
curl http://localhost:3000               # Frontend health

# Database connectivity
docker-compose exec backend python -c "from app.core.database import engine; print(engine.execute('SELECT 1').scalar())"

# Redis connectivity  
docker-compose exec redis redis-cli ping
```

## Git Workflow

### Standard Development Flow
```bash
# Start new feature
git checkout -b feature/your-feature-name
git push -u origin feature/your-feature-name

# Development cycle
git add .
git commit -m "feat: description of changes"
git push

# Code review and merge
gh pr create --title "Feature: Your Feature" --body "Description"
```

### Commit Message Conventions
```bash
feat: new feature
fix: bug fix  
docs: documentation changes
style: formatting changes
refactor: code restructuring
test: adding tests
chore: maintenance tasks
```

## Debugging and Troubleshooting

### Log Access
```bash
# Docker logs
docker-compose logs -f backend          # Backend application logs
docker-compose logs -f postgres         # Database logs
docker-compose logs -f redis            # Cache logs

# Direct log files (if running locally)
tail -f backend/logs/app.log            # Application logs
```

### Common Issues Resolution
```bash
# Database connection issues
docker-compose restart postgres
docker-compose exec postgres psql -U postgres -d trading_system

# Redis connection issues  
docker-compose restart redis
docker-compose exec redis redis-cli

# Port conflicts
lsof -i :8000                          # Check what's using port 8000
```