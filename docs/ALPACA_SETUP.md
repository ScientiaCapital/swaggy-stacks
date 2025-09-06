# Alpaca API Setup Guide

## 🔑 Your Alpaca Credentials

**API Key**: `PKHUUXJV4V04PQ86MNPR`  
**Endpoint**: `https://paper-api.alpaca.markets/v2`

## 📋 Setup Steps

### Step 1: Get Your Secret Key

1. Go to [Alpaca Dashboard](https://app.alpaca.markets/)
2. Log in to your account
3. Navigate to **"API Keys"** in the sidebar
4. Copy your **Secret Key** (it will look like: `abc123def456...`)

### Step 2: Configure Environment

```bash
# Navigate to your project
cd /Users/tmkipper/repos/swaggy-stacks

# Create .env file from template
cp env.example .env

# Edit the .env file
nano .env
```

Update these lines in your `.env` file:

```bash
# Alpaca API Configuration
ALPACA_API_KEY=PKHUUXJV4V04PQ86MNPR
ALPACA_SECRET_KEY=your-secret-key-here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets
```

### Step 3: Test Your Connection

```bash
# Install Python dependencies
cd backend
pip install -r requirements.txt

# Test Alpaca connection
python ../scripts/test_alpaca.py
```

Expected output:
```
🚀 Swaggy Stacks - Alpaca API Test
==================================================
🔧 Testing environment configuration...
✅ ALPACA_API_KEY: **********...MNPR
✅ ALPACA_SECRET_KEY: **********...xyz
✅ ALPACA_BASE_URL: https://paper-api.alpaca.markets
✅ All required environment variables are set

📊 Testing Alpaca API connection...
✅ Successfully connected to Alpaca API!
📈 Account Status: ACTIVE
💰 Portfolio Value: $100,000.00
💵 Cash: $100,000.00
📊 Buying Power: $100,000.00

📡 Testing market data access...
✅ Market data access successful!
🍎 AAPL Latest Price: $150.25

🎉 All tests passed! Your Alpaca integration is ready.
```

## 🚀 Quick Start

### Option 1: Automated Setup

```bash
# Run the setup script
./scripts/setup.sh
```

### Option 2: Manual Setup

```bash
# Start Docker services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

## 🔍 Verify Everything Works

### 1. Check Services

```bash
# Check all services are running
docker-compose ps

# Expected output:
# trading_postgres    Up
# trading_redis       Up  
# trading_backend     Up
# trading_frontend    Up
```

### 2. Test API Endpoints

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test Alpaca integration
curl http://localhost:8000/api/v1/trading/account
```

### 3. Access Web Interface

- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3001

## 📊 Paper Trading Features

With your Alpaca paper trading account, you can:

### ✅ Available Features
- **Real-time market data** for US stocks
- **Paper trading orders** (no real money)
- **Portfolio tracking** and performance metrics
- **Risk management** with position limits
- **Historical data** for backtesting
- **WebSocket streams** for live updates

### 📈 Supported Assets
- **US Stocks** (NYSE, NASDAQ)
- **ETFs** and **REITs**
- **Market hours**: 9:30 AM - 4:00 PM ET
- **Extended hours**: 4:00 AM - 8:00 PM ET

### 💰 Paper Trading Limits
- **Starting Balance**: $100,000 (virtual)
- **No minimum balance** requirements
- **No trading fees** (paper trading)
- **Real-time execution** simulation

## 🛠 Troubleshooting

### Common Issues

**1. "Invalid API Key" Error**
```bash
# Check your .env file
cat .env | grep ALPACA

# Verify API key format
# Should start with "PK" for paper trading
```

**2. "Connection Refused" Error**
```bash
# Check if Docker is running
docker info

# Restart services
docker-compose restart
```

**3. "Market Data Not Available"**
```bash
# Check if markets are open
# Paper trading works 24/7, but real data is limited to market hours
```

### Debug Commands

```bash
# View backend logs
docker-compose logs backend

# View all logs
docker-compose logs -f

# Check service health
docker-compose exec backend curl localhost:8000/health
```

## 🔒 Security Best Practices

### Environment Variables
- ✅ Never commit `.env` file to git
- ✅ Use strong, unique secret keys
- ✅ Rotate API keys regularly
- ✅ Use paper trading for development

### API Security
- ✅ Store credentials in environment variables
- ✅ Use HTTPS endpoints only
- ✅ Implement rate limiting
- ✅ Monitor API usage

## 📚 Next Steps

1. **Test Paper Trading**: Place a test order
2. **Configure Strategies**: Set up your trading parameters
3. **Monitor Performance**: Use the dashboard to track results
4. **Backtest Strategies**: Test on historical data
5. **Scale Up**: Move to live trading when ready

## 🆘 Support

If you encounter issues:

1. **Check the logs**: `docker-compose logs -f`
2. **Verify API keys**: Run the test script
3. **Check Alpaca status**: [Alpaca Status Page](https://status.alpaca.markets/)
4. **Review documentation**: [Alpaca API Docs](https://alpaca.markets/docs/)

## 🎯 Ready to Trade!

Your Swaggy Stacks system is now configured with Alpaca paper trading. You can:

- **Analyze markets** with advanced Markov chains
- **Execute trades** based on technical analysis
- **Monitor performance** in real-time
- **Backtest strategies** on historical data

Happy trading! 🚀📈
