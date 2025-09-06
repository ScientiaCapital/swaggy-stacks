"""
SwaggyStacks Trading Intelligence API
Comprehensive monetized API access to advanced AI trading system
"""

from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.security import APIKeyHeader, APIKeyQuery
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
import time
import asyncio
from datetime import datetime, timedelta
import json
from enum import Enum
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import your existing trading components
from deep_rl.models.enhanced_dqn_brain import EnhancedDQNBrain
from deep_rl.training.meta_orchestrator import MetaRLTradingOrchestrator
from deep_rl.validation.trading_validation_framework import TradingValidationFramework
from deep_rl.monitoring.trading_dashboard import TradingDashboard

# Initialize FastAPI application
app = FastAPI(
    title="SwaggyStacks Trading Intelligence API",
    description="Monetized API access to our advanced AI trading system with Markov chains, Fibonacci analysis, Elliott Wave theory, and Wyckoff method",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

# Database models (using your existing Convex infrastructure)
class APIUser(BaseModel):
    id: str
    email: str
    tier: str  # free, basic, pro, enterprise
    api_key: str
    monthly_quota: int
    requests_this_month: int
    balance: float
    rate_limit: int
    created_at: datetime
    updated_at: datetime

class APIRequestLog(BaseModel):
    id: str
    user_id: str
    endpoint: str
    parameters: Dict[str, Any]
    cost: float
    processing_time: float
    timestamp: datetime
    status: str  # success, failed, rate_limited

class SubscriptionPlan(BaseModel):
    id: str
    name: str
    price: float
    monthly_quota: int
    rate_limit: int  # requests per minute
    features: List[str]

# Request models
class StockAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol to analyze")
    timeframe: str = Field("1d", description="Timeframe for analysis")
    include_technical: bool = Field(True, description="Include technical analysis")
    include_sentiment: bool = Field(False, description="Include sentiment analysis")
    depth: str = Field("standard", description="Analysis depth: basic, standard, advanced")

class PortfolioAnalysisRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of stock symbols in portfolio")
    weights: Optional[Dict[str, float]] = Field(None, description="Portfolio weights")
    risk_tolerance: str = Field("medium", description="Risk tolerance level")
    investment_horizon: str = Field("medium", description="Investment horizon")

class TradingSignalRequest(BaseModel):
    symbol: str
    strategy: str = Field("multi_model", description="Trading strategy to use")

class BacktestRequest(BaseModel):
    strategy_config: Dict[str, Any]
    start_date: str
    end_date: str
    initial_capital: float = 10000

# Global trading system components (initialized on startup)
enhanced_dqn_brain = None
meta_orchestrator = None
validation_framework = None
trading_dashboard = None

# Authentication and rate limiting
async def get_api_user(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query)
):
    """Validate API key and return user object"""
    api_key = api_key_header or api_key_query
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    # Mock user for development (replace with actual Convex database query)
    if api_key == "demo_key_try_it_now":
        return APIUser(
            id="demo_user",
            email="demo@swaggystacks.com",
            tier="free",
            api_key=api_key,
            monthly_quota=100,
            requests_this_month=0,
            balance=10.0,
            rate_limit=10,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    # Query your Convex database for the user
    user = await query_convex_db("get_user_by_api_key", {"api_key": api_key})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return user

# Rate limiting storage (using Redis or similar)
import redis
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
except:
    # Fallback to in-memory storage for development
    redis_client = None
    rate_limit_storage = {}

def check_rate_limit(user: APIUser):
    """Implement rate limiting based on user tier"""
    if redis_client:
        current_minute = datetime.now().strftime("%Y-%m-%d-%H-%M")
        key = f"rate_limit:{user.id}:{current_minute}"
        
        # Get current count
        current_count = redis_client.get(key)
        if current_count is None:
            current_count = 0
            redis_client.setex(key, 60, 0)  # Expire in 60 seconds
        else:
            current_count = int(current_count)
        
        # Check against user's rate limit
        if current_count >= user.rate_limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Increment count
        redis_client.incr(key)
    else:
        # Fallback to in-memory storage
        current_minute = datetime.now().strftime("%Y-%m-%d-%H-%M")
        key = f"{user.id}:{current_minute}"
        
        if key not in rate_limit_storage:
            rate_limit_storage[key] = 0
        
        if rate_limit_storage[key] >= user.rate_limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        rate_limit_storage[key] += 1
    
    return True

# Usage tracking decorator
def track_usage(cost: float = 0.01):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            user = kwargs.get('user')
            
            # Check monthly quota
            if user.requests_this_month >= user.monthly_quota:
                raise HTTPException(
                    status_code=402, 
                    detail="Monthly quota exceeded. Please upgrade your plan."
                )
            
            # Check rate limit
            check_rate_limit(user)
            
            # Execute the endpoint
            try:
                result = await func(*args, **kwargs)
                processing_time = time.time() - start_time
                
                # Log the successful request
                await log_api_request(
                    user_id=user.id,
                    endpoint=func.__name__,
                    parameters=kwargs,
                    cost=cost,
                    processing_time=processing_time,
                    status="success"
                )
                
                # Update user's usage
                await update_user_usage(user.id, cost)
                
                return result
            except Exception as e:
                processing_time = time.time() - start_time
                await log_api_request(
                    user_id=user.id,
                    endpoint=func.__name__,
                    parameters=kwargs,
                    cost=0,  # No cost for failed requests
                    processing_time=processing_time,
                    status="failed"
                )
                raise e
        return wrapper
    return decorator

# Mock database functions (replace with actual Convex integration)
async def query_convex_db(query_name: str, params: Dict[str, Any]):
    """Mock Convex database query"""
    # Replace with actual Convex integration
    return None

async def log_api_request(user_id: str, endpoint: str, parameters: Dict[str, Any], 
                         cost: float, processing_time: float, status: str):
    """Log API request to database"""
    # Replace with actual Convex integration
    print(f"API Request: {endpoint} by {user_id}, cost: ${cost}, status: {status}")

async def update_user_usage(user_id: str, cost: float):
    """Update user's usage statistics"""
    # Replace with actual Convex integration
    print(f"Updated usage for {user_id}: ${cost}")

# Core Trading API Endpoints
@app.get("/")
async def root():
    return {
        "message": "SwaggyStacks Trading Intelligence API", 
        "version": "2.0.0",
        "features": [
            "AI-powered stock analysis",
            "Portfolio optimization",
            "Trading signal generation",
            "Backtesting capabilities",
            "Real-time market data",
            "Multi-model ensemble trading"
        ],
        "documentation": "/docs",
        "status": "operational"
    }

@app.get("/v1/analyze/stock/{symbol}")
@track_usage(cost=0.001)
async def analyze_stock(
    symbol: str,
    timeframe: str = "1d",
    depth: str = "standard",
    include_technical: bool = True,
    include_sentiment: bool = False,
    user: APIUser = Depends(get_api_user)
):
    """
    Analyze a stock using AI models
    
    **Costs:**
    - Basic: $0.001 per request
    - Standard: $0.003 per request  
    - Advanced: $0.005 per request
    
    **Features:**
    - Markov chain analysis
    - Fibonacci retracement levels
    - Elliott Wave patterns
    - Wyckoff accumulation/distribution
    - Technical indicators
    - Sentiment analysis (if enabled)
    """
    # Use your enhanced DQN brain and other models
    analysis = await perform_stock_analysis(symbol, timeframe, depth, include_technical, include_sentiment)
    
    # Calculate cost based on depth
    cost = 0.001 if depth == "basic" else (0.005 if depth == "advanced" else 0.003)
    
    return {
        "symbol": symbol,
        "analysis": analysis,
        "timestamp": datetime.now(),
        "credits_used": cost,
        "remaining_credits": user.balance - cost,
        "analysis_depth": depth,
        "features_used": {
            "technical_analysis": include_technical,
            "sentiment_analysis": include_sentiment,
            "markov_analysis": True,
            "fibonacci_levels": True,
            "elliott_wave": True,
            "wyckoff_phases": True
        }
    }

@app.post("/v1/analyze/portfolio")
@track_usage(cost=0.05)
async def analyze_portfolio(
    request: PortfolioAnalysisRequest,
    user: APIUser = Depends(get_api_user)
):
    """
    Analyze a portfolio using multi-model AI
    
    **Cost:** $0.05 per request
    
    **Features:**
    - Portfolio optimization
    - Risk assessment
    - Correlation analysis
    - Performance attribution
    - Rebalancing recommendations
    """
    # Use your meta-orchestrator for portfolio analysis
    portfolio_analysis = await perform_portfolio_analysis(
        request.symbols, 
        request.weights, 
        request.risk_tolerance, 
        request.investment_horizon
    )
    
    return {
        "portfolio_analysis": portfolio_analysis,
        "timestamp": datetime.now(),
        "credits_used": 0.05,
        "remaining_credits": user.balance - 0.05,
        "portfolio_size": len(request.symbols),
        "risk_tolerance": request.risk_tolerance,
        "investment_horizon": request.investment_horizon
    }

@app.post("/v1/signals/generate")
@track_usage(cost=0.01)
async def generate_trading_signals(
    request: TradingSignalRequest,
    user: APIUser = Depends(get_api_user)
):
    """
    Generate trading signals using specialized AI models
    
    **Cost:** $0.01 per request
    
    **Available Strategies:**
    - fibonacci: Fibonacci-based signals
    - elliott_wave: Elliott Wave analysis
    - wyckoff: Wyckoff method signals
    - markov: Markov chain predictions
    - multi_model: Ensemble of all models
    """
    # Use your specialized agents (Fibonacci, Elliott Wave, etc.)
    signals = await generate_trading_signals_internal(
        request.symbol, 
        request.strategy
    )
    
    return {
        "symbol": request.symbol,
        "strategy": request.strategy,
        "signals": signals,
        "timestamp": datetime.now(),
        "credits_used": 0.01,
        "remaining_credits": user.balance - 0.01,
        "signal_confidence": signals.get("confidence", 0.0),
        "recommended_actions": signals.get("actions", [])
    }

@app.post("/v1/backtest")
@track_usage(cost=0.10)
async def backtest_strategy(
    request: BacktestRequest,
    user: APIUser = Depends(get_api_user)
):
    """
    Backtest a trading strategy using historical data
    
    **Cost:** $0.10 per request
    
    **Features:**
    - Historical performance analysis
    - Risk metrics calculation
    - Drawdown analysis
    - Sharpe ratio calculation
    - Monte Carlo simulation
    """
    # Use your validation framework for backtesting
    backtest_results = await run_backtest_internal(request.strategy_config, request.start_date, request.end_date, request.initial_capital)
    
    return {
        "backtest_results": backtest_results,
        "timestamp": datetime.now(),
        "credits_used": 0.10,
        "remaining_credits": user.balance - 0.10,
        "strategy": request.strategy_config.get("name", "custom"),
        "period": f"{request.start_date} to {request.end_date}",
        "initial_capital": request.initial_capital
    }

@app.get("/v1/market/regime")
@track_usage(cost=0.002)
async def get_market_regime(
    user: APIUser = Depends(get_api_user)
):
    """
    Get current market regime analysis
    
    **Cost:** $0.002 per request
    
    **Features:**
    - Market regime classification
    - Turbulence index
    - Volatility regime
    - Trend analysis
    """
    regime_analysis = await analyze_market_regime()
    
    return {
        "market_regime": regime_analysis,
        "timestamp": datetime.now(),
        "credits_used": 0.002,
        "remaining_credits": user.balance - 0.002
    }

# Integration with Your Existing System
async def perform_stock_analysis(symbol: str, timeframe: str, depth: str, include_technical: bool, include_sentiment: bool):
    """Use your enhanced DQN brain and other models for analysis"""
    # Mock implementation - replace with actual integration
    analysis = {
        "symbol": symbol,
        "current_price": 150.25,
        "price_change": 2.15,
        "price_change_percent": 1.45,
        "volume": 45000000,
        "market_cap": 2500000000000,
        "pe_ratio": 28.5,
        "technical_analysis": {
            "rsi": 65.2,
            "macd": 0.85,
            "bollinger_bands": {
                "upper": 155.20,
                "middle": 150.25,
                "lower": 145.30
            },
            "support_levels": [148.50, 145.00, 142.00],
            "resistance_levels": [152.00, 155.00, 158.00]
        } if include_technical else None,
        "fibonacci_levels": {
            "23.6": 148.50,
            "38.2": 146.75,
            "50.0": 145.00,
            "61.8": 143.25,
            "78.6": 141.50
        },
        "elliott_wave": {
            "current_wave": 3,
            "wave_position": "middle",
            "target_price": 158.00,
            "confidence": 0.75
        },
        "wyckoff_phase": {
            "phase": "markup",
            "accumulation_complete": True,
            "distribution_start": False,
            "confidence": 0.68
        },
        "markov_analysis": {
            "regime": "bullish",
            "transition_probability": 0.15,
            "expected_return": 0.08,
            "volatility": 0.22
        },
        "sentiment_analysis": {
            "overall_sentiment": "positive",
            "news_sentiment": 0.65,
            "social_sentiment": 0.58,
            "analyst_rating": "buy"
        } if include_sentiment else None,
        "recommendation": {
            "action": "buy",
            "confidence": 0.72,
            "target_price": 158.00,
            "stop_loss": 145.00,
            "time_horizon": "3-6 months"
        }
    }
    
    return analysis

async def perform_portfolio_analysis(symbols: List[str], weights: Dict[str, float], risk_tolerance: str, investment_horizon: str):
    """Use your meta-orchestrator for portfolio analysis"""
    # Mock implementation - replace with actual integration
    analysis = {
        "portfolio_summary": {
            "total_symbols": len(symbols),
            "total_weight": sum(weights.values()) if weights else 1.0,
            "risk_tolerance": risk_tolerance,
            "investment_horizon": investment_horizon
        },
        "performance_metrics": {
            "expected_return": 0.12,
            "volatility": 0.18,
            "sharpe_ratio": 0.67,
            "max_drawdown": 0.15,
            "var_95": 0.08
        },
        "risk_analysis": {
            "concentration_risk": "low",
            "sector_diversification": "good",
            "correlation_risk": "moderate",
            "liquidity_risk": "low"
        },
        "optimization_suggestions": [
            {
                "action": "increase_weight",
                "symbol": "AAPL",
                "current_weight": 0.25,
                "suggested_weight": 0.30,
                "reason": "Strong momentum and technical indicators"
            },
            {
                "action": "decrease_weight",
                "symbol": "TSLA",
                "current_weight": 0.20,
                "suggested_weight": 0.15,
                "reason": "High volatility and overvaluation concerns"
            }
        ],
        "rebalancing_recommendations": {
            "frequency": "quarterly",
            "threshold": 0.05,
            "next_rebalance_date": "2024-03-31"
        }
    }
    
    return analysis

async def generate_trading_signals_internal(symbol: str, strategy: str):
    """Generate trading signals using specialized agents"""
    # Mock implementation - replace with actual integration
    signals = {
        "symbol": symbol,
        "strategy": strategy,
        "signals": [
            {
                "type": "buy",
                "price": 150.25,
                "confidence": 0.75,
                "reason": "Fibonacci retracement at 38.2% level with strong support",
                "timeframe": "1-2 weeks"
            },
            {
                "type": "hold",
                "price": 150.25,
                "confidence": 0.60,
                "reason": "Elliott Wave position suggests consolidation phase",
                "timeframe": "3-5 days"
            }
        ],
        "market_context": {
            "trend": "bullish",
            "volatility": "moderate",
            "volume": "above_average"
        },
        "risk_metrics": {
            "stop_loss": 145.00,
            "take_profit": 158.00,
            "risk_reward_ratio": 1.85
        }
    }
    
    return signals

async def run_backtest_internal(strategy_config: Dict[str, Any], start_date: str, end_date: str, initial_capital: float):
    """Run backtest using your validation framework"""
    # Mock implementation - replace with actual integration
    backtest_results = {
        "strategy": strategy_config.get("name", "custom"),
        "period": f"{start_date} to {end_date}",
        "initial_capital": initial_capital,
        "final_capital": initial_capital * 1.25,
        "total_return": 0.25,
        "annualized_return": 0.18,
        "volatility": 0.22,
        "sharpe_ratio": 0.82,
        "max_drawdown": 0.12,
        "win_rate": 0.65,
        "profit_factor": 1.85,
        "total_trades": 45,
        "winning_trades": 29,
        "losing_trades": 16,
        "average_win": 0.08,
        "average_loss": 0.04,
        "equity_curve": [10000, 10200, 10150, 10300, 10500, 10400, 10600, 10800, 10700, 11000, 11200, 11500, 12000, 11800, 12200, 12500],
        "monthly_returns": [0.02, -0.005, 0.015, 0.02, -0.01, 0.02, 0.02, -0.01, 0.03, 0.02, 0.03, 0.02, -0.02, 0.03, 0.02, 0.02]
    }
    
    return backtest_results

async def analyze_market_regime():
    """Analyze current market regime"""
    # Mock implementation - replace with actual integration
    regime_analysis = {
        "current_regime": "bull_market",
        "regime_confidence": 0.75,
        "turbulence_index": 25.5,
        "volatility_regime": "normal",
        "trend_strength": "strong",
        "market_phase": "expansion",
        "risk_level": "moderate",
        "recommended_allocation": {
            "stocks": 0.70,
            "bonds": 0.20,
            "cash": 0.10
        }
    }
    
    return regime_analysis

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "2.0.0",
        "services": {
            "api": "operational",
            "trading_engine": "operational",
            "auth_service": "operational",
            "database": "operational"
        },
        "uptime": "99.9%",
        "response_time": "< 100ms"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize trading system components on startup"""
    global enhanced_dqn_brain, meta_orchestrator, validation_framework, trading_dashboard
    
    print("Initializing SwaggyStacks Trading Intelligence API...")
    
    # Initialize your trading components
    try:
        # Initialize Enhanced DQN Brain
        enhanced_dqn_brain = EnhancedDQNBrain(
            state_size=20,
            action_size=3,
            hidden_size=128,
            num_lstm_layers=2
        )
        
        # Initialize Meta-Orchestrator
        specialized_agents = {
            'fibonacci': enhanced_dqn_brain,
            'elliott_wave': enhanced_dqn_brain,
            'wyckoff': enhanced_dqn_brain
        }
        
        meta_orchestrator = MetaRLTradingOrchestrator(
            specialized_agents=specialized_agents,
            state_size=20,
            action_size=3
        )
        
        # Initialize Validation Framework
        validation_framework = TradingValidationFramework(
            model=enhanced_dqn_brain,
            env_class=None,  # Mock for now
            data_sources={}
        )
        
        # Initialize Trading Dashboard
        trading_dashboard = TradingDashboard()
        
        print("Trading system components initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing trading components: {e}")
        print("API will run with mock implementations")

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=int(os.getenv("WORKERS", 1))
    )
