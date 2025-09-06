# Swaggy Stacks - Advanced Markov Trading System

A production-ready algorithmic trading system integrating Markov chains, Fibonacci analysis, Elliott Wave theory, and Wyckoff method with real-time paper trading capabilities.

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd swaggy-stacks

# Start with Docker Compose
docker-compose up -d

# Or run locally
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

## 📋 Project Structure

```
swaggy-stacks/
├── backend/                 # FastAPI backend application
├── frontend/               # React TypeScript frontend
├── infrastructure/         # Docker, K8s, CI/CD configs
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── tests/                 # Integration tests
└── docker-compose.yml     # Local development setup
```

## 🛠 Technology Stack

- **Backend**: Python, FastAPI, PostgreSQL, Redis, Celery
- **Frontend**: React, TypeScript, Tailwind CSS, Shadcn UI
- **Infrastructure**: Docker, Kubernetes, GitHub Actions
- **Trading**: Alpaca API, Paper Trading
- **Monitoring**: Prometheus, Grafana

## 📊 Features

- Real-time market data integration
- Enhanced Markov analysis system
- Paper trading execution engine
- Portfolio management
- Risk management controls
- Web-based dashboard
- Real-time performance monitoring

## 🔧 Development

See [Development Guide](docs/development.md) for detailed setup instructions.

## 📈 Trading

This system is designed for paper trading and educational purposes. Always understand the risks before trading with real money.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.
