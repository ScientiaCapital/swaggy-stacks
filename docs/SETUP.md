# Swaggy Stacks - Setup Guide

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Git
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

### 1. Clone the Repository

```bash
git clone https://github.com/ScientiaCapital/swaggy-stacks.git
cd swaggy-stacks
```

### 2. Environment Setup

```bash
# Copy environment template
cp env.example .env

# Edit environment variables
nano .env
```

**Required Environment Variables:**
- `ALPACA_API_KEY` - Your Alpaca API key
- `ALPACA_SECRET_KEY` - Your Alpaca secret key
- `SECRET_KEY` - Random secret key for JWT tokens

### 3. Start with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090

## 🛠 Local Development

### Backend Development

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## 📊 Database Setup

### PostgreSQL

The application uses PostgreSQL for data persistence. With Docker Compose, it's automatically configured.

**Manual Setup:**
```bash
# Create database
createdb trading_system

# Run migrations
cd backend
alembic upgrade head
```

### Redis

Redis is used for caching and session storage. It's automatically configured with Docker Compose.

## 🔧 Configuration

### Alpaca API Setup

1. Sign up at [Alpaca Markets](https://alpaca.markets/)
2. Get your API keys from the dashboard
3. Add them to your `.env` file

### Trading Configuration

Edit `backend/app/core/config.py` to adjust:
- Position sizing limits
- Risk management parameters
- Market data update intervals

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v --cov=app
```

### Frontend Tests

```bash
cd frontend
npm test
```

## 📈 Monitoring

### Prometheus Metrics

The application exposes metrics at `/metrics` endpoint:
- Trading performance metrics
- System health metrics
- API response times

### Grafana Dashboards

Pre-configured dashboards are available at http://localhost:3001:
- Trading Performance
- System Health
- Market Data Quality

## 🚀 Deployment

### Production Deployment

1. **Set up production environment variables**
2. **Configure SSL certificates**
3. **Set up domain names**
4. **Deploy with Docker Compose or Kubernetes**

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f infrastructure/k8s/
```

## 🔒 Security

### Environment Variables

Never commit sensitive data to version control:
- API keys
- Database passwords
- JWT secrets

### API Security

- JWT token authentication
- Rate limiting
- CORS configuration
- Input validation

## 📝 API Documentation

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

- `POST /api/v1/trading/orders` - Create trading order
- `GET /api/v1/trading/orders` - Get orders
- `GET /api/v1/analysis/markov` - Get Markov analysis
- `GET /api/v1/portfolio/positions` - Get portfolio positions

## 🐛 Troubleshooting

### Common Issues

**Database Connection Error:**
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres
```

**Redis Connection Error:**
```bash
# Check if Redis is running
docker-compose ps redis

# Test connection
docker-compose exec redis redis-cli ping
```

**API Not Responding:**
```bash
# Check backend logs
docker-compose logs backend

# Restart services
docker-compose restart backend
```

### Logs

View application logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

## 📚 Additional Resources

- [Alpaca API Documentation](https://alpaca.markets/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Docker Documentation](https://docs.docker.com/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
