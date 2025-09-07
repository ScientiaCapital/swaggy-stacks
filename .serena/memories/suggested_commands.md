# Suggested Commands for Development

## Essential Daily Commands

### Quick Start Development
```bash
# Start development environment
docker-compose up -d                    # Start all services
docker-compose ps                       # Check service status

# Monitor logs  
docker-compose logs -f backend          # Backend application logs
docker-compose logs -f frontend         # Frontend development logs

# Stop services
docker-compose down                     # Stop all services
```

### Code Quality (Run Before Every Commit)
```bash
# Backend code quality
cd backend
black app/ && isort app/ && flake8 app/ && mypy app/

# Frontend code quality  
cd frontend
npm run lint && npm run type-check
```

### Testing Commands
```bash
# Backend testing
cd backend
pytest tests/ -v --cov=app              # Full test suite with coverage
python ../test_trading_simple.py        # Integration tests

# Frontend testing
cd frontend
npm test -- --coverage --watchAll=false # Full test suite
```

## Development Workflow Commands

### New Feature Development
```bash
# 1. Create feature branch
git checkout -b feature/your-feature-name
git push -u origin feature/your-feature-name

# 2. Development cycle
# ... make changes ...
black app/ && isort app/                 # Format code
pytest tests/test_your_module.py -v     # Test changes
git add . && git commit -m "feat: description"
git push

# 3. Create pull request  
gh pr create --title "Feature: Description" --body "Details"
```

### Backend Development
```bash
# Start backend development
cd backend
uvicorn app.main:app --reload           # Hot reload development server

# Database operations
alembic upgrade head                    # Apply latest migrations
alembic revision --autogenerate -m "description"  # Create new migration

# Dependencies
pip install <package>                   # Add new dependency
pip freeze > requirements.txt          # Update requirements
```

### Frontend Development  
```bash
# Start frontend development
cd frontend
npm run dev                            # Next.js development server

# Package management
npm install <package>                  # Add dependency
npm run build                          # Production build test
npm run start                          # Test production build
```

## Debugging and Troubleshooting

### Service Health Checks
```bash
# API health
curl http://localhost:8000/health       # Backend health check
curl http://localhost:8000/docs         # API documentation

# Database connectivity
docker-compose exec backend python -c "from app.core.database import engine; print('DB OK')"

# Redis connectivity
docker-compose exec redis redis-cli ping
```

### Log Investigation
```bash
# Real-time log monitoring
docker-compose logs -f backend | grep ERROR    # Backend errors
docker-compose logs -f postgres               # Database logs
docker-compose logs -f redis                  # Cache logs

# Log file access (if running locally)
tail -f backend/logs/app.log           # Application logs
```

### Common Issue Resolution
```bash
# Port conflicts
lsof -i :8000                          # Check port 8000 usage
lsof -i :3000                          # Check port 3000 usage

# Container issues
docker-compose restart backend          # Restart backend service
docker-compose restart postgres        # Restart database
docker system prune                    # Clean up Docker resources

# Database reset (development only)
docker-compose down -v                 # Remove volumes
docker-compose up -d                   # Restart with fresh DB
```

## Testing and Validation Commands

### Complete Testing Suite
```bash
# Run all backend tests
cd backend
pytest tests/ -v --cov=app --cov-report=html

# Run integration tests
python ../test_trading_simple.py       # Trading system
python ../test_ai_system.py           # AI components  
python ../test_markov_mvp.py           # Markov analysis
python ../test_live_trading_system.py  # Complete system

# Frontend tests
cd frontend
npm test -- --coverage
npm run build                          # Build validation
```

### Performance Monitoring
```bash
# System metrics
docker stats                           # Container resource usage
curl http://localhost:9090             # Prometheus metrics
curl http://localhost:3001             # Grafana dashboard

# Application metrics
curl http://localhost:8000/health      # Application health
```

## Utility Commands (macOS/Darwin)

### File Operations
```bash
# Find files (macOS compatible)
find . -name "*.py" -type f            # Find Python files
find . -path "*/tests/*" -name "*.py"  # Find test files

# Search in files
grep -r "TradingManager" app/          # Search for text
rg "async def" app/                    # Use ripgrep (faster)

# File permissions (if needed)
chmod +x script.sh                     # Make script executable
```

### Git Operations  
```bash
# Standard git workflow
git status                             # Check working directory status
git add .                              # Stage all changes
git commit -m "message"                # Commit changes
git push                               # Push to remote

# Branch management
git branch                             # List local branches
git checkout -b feature/name           # Create and switch to branch
git merge main                         # Merge main into current branch

# History and logs
git log --oneline                      # Compact commit history
git show HEAD                          # Show latest commit details
```

### Directory Navigation
```bash
# Quick navigation
cd backend && ls -la                   # Backend directory
cd frontend && ls -la                  # Frontend directory
cd scripts && ls -la                   # Scripts directory

# File inspection
ls -la                                 # Detailed file listing
tree -L 2                              # Directory tree (if installed)
du -sh *                               # Directory sizes
```

## MCP Integration Commands (Phase 2+)

### TaskMaster-AI Commands
```bash
# Task management (via MCP tools)
# task-master list                     # Show all tasks
# task-master next                     # Get next available task  
# task-master show <id>               # View task details
```

### System Integration
```bash
# MCP health checks (planned)
# Health checks for all MCP servers will be available
# Integration testing for MCP orchestration
```

## Emergency Commands

### Quick Problem Resolution
```bash
# Stop everything
docker-compose down                    # Stop all services
pkill -f uvicorn                      # Kill backend processes
pkill -f "npm run"                    # Kill frontend processes

# Clean restart
docker-compose down -v                # Remove all containers and volumes
docker-compose up -d                  # Fresh start

# Reset to clean state
git stash                             # Stash uncommitted changes
git checkout main                     # Switch to main branch
git pull                              # Get latest changes
```