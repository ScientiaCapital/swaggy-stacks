# Swaggy Stacks - API Documentation

## Overview

The Swaggy Stacks API provides comprehensive endpoints for algorithmic trading, market analysis, and portfolio management. Built with FastAPI, it offers real-time market data integration, advanced technical analysis, and risk management capabilities.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://api.swaggystacks.com`

## Authentication

The API uses JWT (JSON Web Tokens) for authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your-jwt-token>
```

## Rate Limiting

- **Standard endpoints**: 100 requests per minute
- **Trading endpoints**: 10 requests per minute
- **Market data endpoints**: 1000 requests per minute

## Response Format

All API responses follow this format:

```json
{
  "data": {},
  "message": "Success",
  "status": "success",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

Error responses:

```json
{
  "error": "Error message",
  "status": "error",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Endpoints

### Authentication

#### POST /api/v1/auth/login
Authenticate user and get JWT token.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

#### POST /api/v1/auth/register
Register a new user.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "password123",
  "full_name": "John Doe",
  "username": "johndoe"
}
```

### Trading

#### POST /api/v1/trading/orders
Create a new trading order.

**Request:**
```json
{
  "symbol": "AAPL",
  "quantity": 100,
  "side": "BUY",
  "order_type": "market",
  "time_in_force": "gtc",
  "strategy_id": 1
}
```

**Response:**
```json
{
  "order_id": "order_123",
  "symbol": "AAPL",
  "quantity": 100,
  "side": "BUY",
  "status": "submitted",
  "submitted_at": "2024-01-01T00:00:00Z",
  "message": "Order submitted successfully"
}
```

#### GET /api/v1/trading/orders
Get user's trading orders.

**Query Parameters:**
- `status` (optional): Filter by order status
- `limit` (optional): Number of orders to return (default: 50)

**Response:**
```json
[
  {
    "order_id": "order_123",
    "symbol": "AAPL",
    "quantity": 100,
    "side": "BUY",
    "status": "filled",
    "submitted_at": "2024-01-01T00:00:00Z",
    "filled_at": "2024-01-01T00:01:00Z",
    "filled_price": 150.25
  }
]
```

#### POST /api/v1/trading/orders/{order_id}/cancel
Cancel an existing order.

**Response:**
```json
{
  "message": "Order cancelled successfully"
}
```

### Market Analysis

#### GET /api/v1/analysis/markov/{symbol}
Get Markov chain analysis for a symbol.

**Response:**
```json
{
  "symbol": "AAPL",
  "analysis": {
    "signal": "BUY",
    "confidence": 0.85,
    "current_state": "UPTREND_HIGH_VOLUME",
    "next_state_prediction": {
      "state": "UPTREND_MEDIUM_VOLUME",
      "probability": 0.72
    },
    "expected_return": 0.025,
    "volatility": 0.18,
    "risk_score": 0.35
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### GET /api/v1/analysis/technical/{symbol}
Get technical analysis indicators.

**Response:**
```json
{
  "symbol": "AAPL",
  "indicators": {
    "rsi": 65.5,
    "macd": {
      "macd": 1.25,
      "signal": 1.10,
      "histogram": 0.15
    },
    "bollinger_bands": {
      "upper": 155.50,
      "middle": 150.25,
      "lower": 145.00,
      "position": 0.75
    },
    "fibonacci_levels": {
      "23.6": 148.50,
      "38.2": 146.75,
      "50.0": 145.00,
      "61.8": 143.25,
      "100": 140.00
    }
  },
  "composite_signals": {
    "trend": "BULLISH",
    "momentum": "NEUTRAL",
    "volatility": "NORMAL",
    "volume": "HIGH",
    "composite": "BUY",
    "signal_strength": 0.75
  }
}
```

#### GET /api/v1/analysis/regime/{symbol}
Get market regime analysis.

**Response:**
```json
{
  "symbol": "AAPL",
  "regime": "TRENDING_HIGH_VOLATILITY_HIGH_VOLUME",
  "hurst_exponent": 0.68,
  "volatility": 0.22,
  "volume_ratio": 1.8,
  "confidence": 0.82
}
```

### Portfolio Management

#### GET /api/v1/portfolio/positions
Get current portfolio positions.

**Response:**
```json
[
  {
    "symbol": "AAPL",
    "quantity": 100,
    "side": "LONG",
    "entry_price": 150.00,
    "current_price": 155.25,
    "market_value": 15525.00,
    "unrealized_pnl": 525.00,
    "unrealized_pnl_percent": 3.50
  }
]
```

#### GET /api/v1/portfolio/performance
Get portfolio performance metrics.

**Response:**
```json
{
  "total_value": 100000.00,
  "cash": 84475.00,
  "invested": 15525.00,
  "total_pnl": 525.00,
  "total_pnl_percent": 0.53,
  "daily_pnl": 125.00,
  "daily_pnl_percent": 0.13,
  "sharpe_ratio": 1.85,
  "max_drawdown": 0.08,
  "win_rate": 0.65
}
```

#### GET /api/v1/portfolio/risk
Get risk assessment for the portfolio.

**Response:**
```json
{
  "portfolio_exposure": 0.155,
  "max_portfolio_exposure": 0.95,
  "daily_loss": 0.0,
  "max_daily_loss": 500.0,
  "risk_score": 0.25,
  "warnings": [],
  "symbol_exposures": {
    "AAPL": 0.155
  }
}
```

### Market Data

#### GET /api/v1/market-data/quotes/{symbol}
Get latest quote for a symbol.

**Response:**
```json
{
  "symbol": "AAPL",
  "bid": 155.20,
  "ask": 155.25,
  "bid_size": 1000,
  "ask_size": 500,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### GET /api/v1/market-data/historical/{symbol}
Get historical market data.

**Query Parameters:**
- `start_date`: Start date (ISO format)
- `end_date`: End date (ISO format)
- `timeframe`: Data timeframe (1min, 5min, 1hour, 1day)

**Response:**
```json
[
  {
    "timestamp": "2024-01-01T00:00:00Z",
    "open": 150.00,
    "high": 155.50,
    "low": 149.75,
    "close": 155.25,
    "volume": 1000000
  }
]
```

### Strategies

#### GET /api/v1/strategies
Get user's trading strategies.

**Response:**
```json
[
  {
    "id": 1,
    "name": "Markov Momentum",
    "description": "Markov chain-based momentum strategy",
    "strategy_type": "MARKOV",
    "is_active": true,
    "total_trades": 45,
    "winning_trades": 30,
    "total_pnl": 2500.00,
    "win_rate": 0.67,
    "sharpe_ratio": 1.85
  }
]
```

#### POST /api/v1/strategies
Create a new trading strategy.

**Request:**
```json
{
  "name": "Fibonacci Retracement",
  "description": "Strategy based on Fibonacci retracement levels",
  "strategy_type": "FIBONACCI",
  "parameters": {
    "retracement_levels": [0.236, 0.382, 0.618],
    "stop_loss": 0.02,
    "take_profit": 0.15
  }
}
```

## WebSocket Endpoints

### Real-time Market Data

Connect to `ws://localhost:8000/ws/market-data` for real-time market data updates.

**Message Format:**
```json
{
  "type": "market_data",
  "symbol": "AAPL",
  "price": 155.25,
  "volume": 1000,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Trading Updates

Connect to `ws://localhost:8000/ws/trading` for real-time trading updates.

**Message Format:**
```json
{
  "type": "order_update",
  "order_id": "order_123",
  "status": "filled",
  "filled_price": 155.25,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 422 | Validation Error |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |

## SDKs and Libraries

### Python SDK

```python
from swaggy_stacks import SwaggyStacksClient

client = SwaggyStacksClient(
    api_key="your-api-key",
    base_url="http://localhost:8000"
)

# Get analysis
analysis = client.get_markov_analysis("AAPL")
print(f"Signal: {analysis['signal']}, Confidence: {analysis['confidence']}")

# Place order
order = client.create_order(
    symbol="AAPL",
    quantity=100,
    side="BUY"
)
```

### JavaScript SDK

```javascript
import { SwaggyStacksClient } from '@swaggystacks/sdk';

const client = new SwaggyStacksClient({
  apiKey: 'your-api-key',
  baseUrl: 'http://localhost:8000'
});

// Get analysis
const analysis = await client.getMarkovAnalysis('AAPL');
console.log(`Signal: ${analysis.signal}, Confidence: ${analysis.confidence}`);

// Place order
const order = await client.createOrder({
  symbol: 'AAPL',
  quantity: 100,
  side: 'BUY'
});
```

## Rate Limits

| Endpoint Category | Limit | Window |
|------------------|-------|--------|
| Authentication | 10 requests | 1 minute |
| Trading | 10 requests | 1 minute |
| Analysis | 100 requests | 1 minute |
| Market Data | 1000 requests | 1 minute |
| Portfolio | 50 requests | 1 minute |

## Support

For API support and questions:
- **Email**: support@swaggystacks.com
- **Documentation**: https://docs.swaggystacks.com
- **GitHub Issues**: https://github.com/ScientiaCapital/swaggy-stacks/issues
