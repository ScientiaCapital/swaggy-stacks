# System Dependencies

## Required

### Python
- **Python 3.13+** (required)
- Virtual environment recommended: `python3 -m venv venv`

### PostgreSQL
- **PostgreSQL 14+** (required for data persistence)
- Installation:
  - macOS: `brew install postgresql@14`
  - Ubuntu: `sudo apt-get install postgresql-14`
  - Docker: `docker run -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:14`

### Redis
- **Redis 6+** (required for caching and message queues)
- Installation:
  - macOS: `brew install redis`
  - Ubuntu: `sudo apt-get install redis-server`
  - Docker: `docker run -p 6379:6379 redis:6`

## Optional (for advanced indicators)

### TA-Lib (Technical Analysis Library)
**Note:** TA-Lib requires system-level installation before installing the Python wrapper.

#### macOS
```bash
brew install ta-lib
pip install talib
```

#### Ubuntu/Debian
```bash
sudo apt-get install libta-lib-dev
pip install talib
```

#### Windows
Download pre-built wheels from:
https://github.com/cgohlke/talib-build/releases

Or use conda:
```bash
conda install -c conda-forge ta-lib
```

#### Alternative
If you cannot install TA-Lib system library, the project will fall back to the pure Python `ta` library which is included in requirements.txt and requires no system dependencies.

## Development Tools

### Recommended
- **Docker & Docker Compose** (for containerized development)
- **Git** (version control)
- **Make** (for build automation, optional)

### Python Development
All Python dependencies are managed in `requirements.txt`:
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

## Verification

To verify your system dependencies are correctly installed:

```bash
# Check Python version
python3 --version  # Should be 3.13 or higher

# Check PostgreSQL
psql --version  # Should be 14 or higher

# Check Redis
redis-cli --version  # Should be 6 or higher

# Check TA-Lib (optional)
python3 -c "import talib; print('TA-Lib installed')" || echo "TA-Lib not installed (optional)"
```

## Environment Variables

Required environment variables (create a `.env` file in backend directory):

```bash
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=swaggy_stacks

# Redis
REDIS_URL=redis://localhost:6379/0

# Trading API (Alpaca)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading

# Security
SECRET_KEY=your_secret_key_here

# AI Providers (for LangGraph agents)
ANTHROPIC_API_KEY=sk-ant-...  # Claude API
OPENROUTER_API_KEY=sk-or-...  # OpenRouter (optional)
CEREBRAS_API_KEY=...          # Cerebras (optional)
```

## Troubleshooting

### Import Errors
If you encounter import errors after installation:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### TA-Lib Installation Issues
If TA-Lib fails to install, comment it out in requirements.txt. The system will use the pure Python `ta` library instead.

### PostgreSQL Connection Issues
- Ensure PostgreSQL service is running: `brew services start postgresql@14` (macOS)
- Check connection settings in `.env` file
- Verify user permissions: `psql -U postgres -c "\du"`

### Redis Connection Issues
- Ensure Redis service is running: `brew services start redis` (macOS)
- Test connection: `redis-cli ping` (should return PONG)
