#!/usr/bin/env python3
"""
🚀 SIMPLE LIVE AGENTS WITH MOCK DATA
===================================
Working live agent server with real-time mock trading decisions
"""

import asyncio
import json
import time
import random
from datetime import datetime
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Live Trading Agents")

# Live agent data
agents_data = {
    "agents": {
        "analyst": {"status": "ACTIVE", "model": "llama3.2:3b", "decisions": 0},
        "risk": {"status": "ACTIVE", "model": "phi3:mini", "decisions": 0},
        "strategist": {"status": "ACTIVE", "model": "qwen2.5-coder:3b", "decisions": 0},
        "chat": {"status": "ACTIVE", "model": "gemma2:2b", "decisions": 0},
        "reasoning": {"status": "ACTIVE", "model": "deepseek-r1:1.5b", "decisions": 0}
    },
    "trades": [],
    "market_events": []
}

def generate_mock_market_data():
    """Generate realistic market events"""
    symbols = ["SPY", "AAPL", "TSLA", "QQQ", "MSFT", "NVDA"]
    symbol = random.choice(symbols)
    price = random.uniform(150, 450)
    change = random.uniform(-5, 5)

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "change": round(change, 2),
        "change_percent": round((change/price)*100, 2),
        "volume": random.randint(1000000, 15000000),
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

def generate_agent_decision(agent_type, market_data):
    """Generate agent-specific decisions"""

    if agent_type == "analyst":
        return {
            "recommendation": random.choice(["LONG", "SHORT", "NEUTRAL"]),
            "confidence": round(random.uniform(0.65, 0.95), 2),
            "target": round(market_data["price"] * random.uniform(1.02, 1.08), 2),
            "reasoning": f"Technical indicators suggest {random.choice(['bullish', 'bearish', 'neutral'])} momentum"
        }
    elif agent_type == "risk":
        return {
            "risk_level": random.choice(["LOW", "MODERATE", "HIGH"]),
            "approval": random.choice([True, True, False]),
            "max_position": random.randint(5000, 20000),
            "risk_score": round(random.uniform(2.5, 8.5), 1)
        }
    elif agent_type == "strategist":
        return {
            "strategy": random.choice(["Iron Condor", "Covered Call", "Protective Put", "Bull Spread"]),
            "max_profit": random.randint(300, 800),
            "max_loss": random.randint(150, 400),
            "win_probability": round(random.uniform(0.55, 0.78), 2)
        }
    elif agent_type == "chat":
        return {
            "message": f"Coordinating {market_data['symbol']} analysis",
            "participants": random.randint(3, 5),
            "status": random.choice(["analyzing", "consensus_building", "execution_ready"])
        }
    else:  # reasoning
        return {
            "pattern": random.choice(["Bullish Flag", "Bear Triangle", "Double Bottom", "Head & Shoulders"]),
            "confidence": round(random.uniform(0.6, 0.85), 2),
            "historical_success": round(random.uniform(0.65, 0.82), 2)
        }

async def trading_simulation():
    """Continuous trading simulation"""
    while True:
        try:
            # Generate market event
            market_data = generate_mock_market_data()
            agents_data["market_events"].append(market_data)

            print(f"📊 Market: {market_data['symbol']} @ ${market_data['price']} ({market_data['change']:+.2f})")

            # Each agent processes the market data
            agent_decisions = {}
            for agent_name in agents_data["agents"]:
                decision = generate_agent_decision(agent_name, market_data)
                agent_decisions[agent_name] = decision
                agents_data["agents"][agent_name]["decisions"] += 1

                print(f"   🤖 {agent_name}: {json.dumps(decision)[:60]}...")

            # Check for trade execution
            risk_approval = agent_decisions.get("risk", {}).get("approval", False)
            analyst_confidence = agent_decisions.get("analyst", {}).get("confidence", 0)

            if risk_approval and analyst_confidence > 0.75:
                trade = {
                    "symbol": market_data["symbol"],
                    "price": market_data["price"],
                    "strategy": agent_decisions.get("strategist", {}).get("strategy", "Market Order"),
                    "consensus_score": round((analyst_confidence + 0.8) / 2, 2),
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "trade_id": f"TRD_{len(agents_data['trades'])+1:03d}"
                }

                agents_data["trades"].append(trade)
                print(f"   🚀 TRADE EXECUTED: {trade['strategy']} on {trade['symbol']} @ ${trade['price']}")

            # Keep only recent data
            if len(agents_data["market_events"]) > 10:
                agents_data["market_events"] = agents_data["market_events"][-10:]
            if len(agents_data["trades"]) > 20:
                agents_data["trades"] = agents_data["trades"][-20:]

            await asyncio.sleep(random.uniform(3, 8))  # Random intervals

        except Exception as e:
            print(f"Error in trading simulation: {e}")
            await asyncio.sleep(2)

@app.get("/api/live")
async def get_live_data():
    """Get all live agent and trading data"""
    return {
        "agents": agents_data["agents"],
        "recent_trades": agents_data["trades"][-5:],
        "latest_market": agents_data["market_events"][-1] if agents_data["market_events"] else None,
        "total_decisions": sum(agent["decisions"] for agent in agents_data["agents"].values()),
        "total_trades": len(agents_data["trades"]),
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

@app.get("/api/agents")
async def get_agents():
    """Get agent status"""
    return agents_data["agents"]

@app.get("/api/trades")
async def get_trades():
    """Get recent trades"""
    return {"trades": agents_data["trades"], "count": len(agents_data["trades"])}

@app.get("/")
async def dashboard():
    """Live dashboard"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>🤖 Live Trading Agents</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #0a0a0a; color: #fff; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #00ff88; margin: 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px; }
        .card h3 { color: #00ff88; margin: 0 0 15px 0; }
        .agent { display: flex; justify-content: space-between; padding: 10px; margin: 5px 0; background: #2a2a2a; border-radius: 5px; }
        .active { color: #00ff88; }
        .trade { background: #1a3a1a; border-left: 4px solid #00ff88; padding: 10px; margin: 5px 0; border-radius: 4px; }
        .metrics { display: flex; justify-content: space-around; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #00ff88; }
        .status { padding: 10px; border-radius: 5px; margin-bottom: 20px; text-align: center; }
        .live { background: #1a3a1a; color: #00ff88; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 SwaggyStacks Live Trading Agents</h1>
        <p>Real-time AI agent coordination and decision making</p>
    </div>

    <div class="status live" id="status">🟢 All agents operational</div>

    <div class="grid">
        <div class="card">
            <h3>🤖 Active Agents</h3>
            <div id="agents"></div>
        </div>

        <div class="card">
            <h3>📊 Live Metrics</h3>
            <div class="metrics">
                <div>
                    <div class="metric-value" id="decisions">0</div>
                    <div>Total Decisions</div>
                </div>
                <div>
                    <div class="metric-value" id="trades">0</div>
                    <div>Trades Executed</div>
                </div>
                <div>
                    <div class="metric-value" id="timestamp">--:--:--</div>
                    <div>Last Update</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h3>🚀 Recent Trades</h3>
            <div id="recentTrades"></div>
        </div>

        <div class="card">
            <h3>📈 Latest Market</h3>
            <div id="latestMarket"></div>
        </div>
    </div>

    <script>
        async function updateData() {
            try {
                const response = await fetch('/api/live');
                const data = await response.json();

                // Update agents
                const agentsHtml = Object.entries(data.agents).map(([name, info]) =>
                    `<div class="agent">
                        <span><strong>${name}</strong> (${info.model})</span>
                        <span class="active">${info.status} | ${info.decisions} decisions</span>
                    </div>`
                ).join('');
                document.getElementById('agents').innerHTML = agentsHtml;

                // Update metrics
                document.getElementById('decisions').textContent = data.total_decisions;
                document.getElementById('trades').textContent = data.total_trades;
                document.getElementById('timestamp').textContent = data.timestamp;

                // Update recent trades
                const tradesHtml = data.recent_trades.map(trade =>
                    `<div class="trade">
                        <strong>${trade.strategy}</strong> on ${trade.symbol}<br>
                        Price: $${trade.price} | Score: ${trade.consensus_score} | ${trade.timestamp}
                    </div>`
                ).join('');
                document.getElementById('recentTrades').innerHTML = tradesHtml || '<div>No trades yet</div>';

                // Update latest market
                if (data.latest_market) {
                    const market = data.latest_market;
                    document.getElementById('latestMarket').innerHTML =
                        `<div style="font-size: 18px;">
                            <strong>${market.symbol}</strong> @ $${market.price}<br>
                            Change: ${market.change >= 0 ? '+' : ''}${market.change} (${market.change_percent}%)<br>
                            Volume: ${market.volume.toLocaleString()}<br>
                            Time: ${market.timestamp}
                        </div>`;
                }

            } catch (error) {
                console.error('Failed to update data:', error);
                document.getElementById('status').innerHTML = '❌ Connection error';
                document.getElementById('status').className = 'status';
            }
        }

        // Update every 2 seconds
        setInterval(updateData, 2000);
        updateData(); // Initial load
    </script>
</body>
</html>
    """)

@app.on_event("startup")
async def startup():
    """Start background trading simulation"""
    print("🚀 Starting Live Trading Agents...")
    print("🌐 Dashboard: http://localhost:8002")
    print("📡 API: http://localhost:8002/api/live")

    # Start trading simulation in background
    asyncio.create_task(trading_simulation())

    print("✅ All 5 agents are now LIVE and trading!")

if __name__ == "__main__":
    print("🚀 Starting Simple Live Agent Server...")
    print("📊 Dashboard: http://localhost:8002")

    uvicorn.run(
        "live_agents_simple:app",
        host="0.0.0.0",
        port=8002,
        log_level="info",
        reload=False
    )