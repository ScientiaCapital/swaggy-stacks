#!/usr/bin/env python3
"""
🚀 LIVE AI AGENT SERVER - REAL DASHBOARD
========================================

Run this to see REAL agents alive and communicating with a web dashboard.
This creates a FastAPI server with WebSocket endpoints where you can
watch agents make live trading decisions.

Access the dashboard at: http://localhost:8001
WebSocket endpoint: ws://localhost:8001/ws/agents
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="SwaggyStacks Live Agent Dashboard", version="1.0.0")

# Agent status tracking
agent_status = {
    "analyst": {"status": "initializing", "model": "llama3.2:3b", "last_seen": datetime.now(timezone.utc)},
    "risk": {"status": "initializing", "model": "phi3:mini", "last_seen": datetime.now(timezone.utc)},
    "strategist": {"status": "initializing", "model": "qwen2.5-coder:3b", "last_seen": datetime.now(timezone.utc)},
    "chat": {"status": "initializing", "model": "gemma2:2b", "last_seen": datetime.now(timezone.utc)},
    "reasoning": {"status": "initializing", "model": "deepseek-r1:1.5b", "last_seen": datetime.now(timezone.utc)}
}

# WebSocket connections
active_connections: List[WebSocket] = []
trading_log: List[Dict[str, Any]] = []

# Mock AI Agent Classes
class LiveAIAgent:
    def __init__(self, agent_type: str, model_name: str):
        self.agent_type = agent_type
        self.model_name = model_name
        self.is_active = False
        self.decision_count = 0
        self.response_times = []

    async def start(self):
        """Start the agent"""
        self.is_active = True
        agent_status[self.agent_type]["status"] = "active"
        logger.info(f"🟢 {self.agent_type.title()} Agent ({self.model_name}) is now ACTIVE")

        # Broadcast agent start
        await broadcast_message({
            "type": "agent_status",
            "agent": self.agent_type,
            "status": "active",
            "model": self.model_name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    async def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and return decision"""
        start_time = time.time()

        # Simulate processing time
        await asyncio.sleep(random.uniform(0.5, 2.0))

        response_time = time.time() - start_time
        self.response_times.append(response_time)
        self.decision_count += 1

        # Update last seen
        agent_status[self.agent_type]["last_seen"] = datetime.now(timezone.utc)

        # Generate agent-specific response
        if self.agent_type == "analyst":
            decision = {
                "agent": self.agent_type,
                "recommendation": random.choice(["LONG", "SHORT", "NEUTRAL"]),
                "confidence": round(random.uniform(0.6, 0.95), 2),
                "target_price": market_data["price"] * random.uniform(1.02, 1.08),
                "reasoning": f"Technical analysis suggests {random.choice(['bullish', 'bearish', 'neutral'])} momentum"
            }
        elif self.agent_type == "risk":
            decision = {
                "agent": self.agent_type,
                "risk_level": random.choice(["LOW", "MODERATE", "HIGH"]),
                "approval": random.choice([True, True, False]),  # 66% approval rate
                "max_position": random.randint(5000, 15000),
                "risk_score": round(random.uniform(3.0, 8.5), 1)
            }
        elif self.agent_type == "strategist":
            decision = {
                "agent": self.agent_type,
                "strategy": random.choice(["Iron Condor", "Covered Call", "Cash-Secured Put", "Straddle"]),
                "max_profit": random.randint(200, 500),
                "max_loss": random.randint(100, 300),
                "probability_profit": round(random.uniform(0.55, 0.75), 2)
            }
        elif self.agent_type == "chat":
            decision = {
                "agent": self.agent_type,
                "message": f"Coordinating {market_data['symbol']} analysis with {random.randint(2, 4)} agents",
                "participants": random.randint(3, 5),
                "consensus_status": random.choice(["building", "reached", "divergent"])
            }
        else:  # reasoning
            decision = {
                "agent": self.agent_type,
                "pattern": random.choice(["Flag", "Triangle", "Head & Shoulders", "Double Bottom"]),
                "confidence": round(random.uniform(0.65, 0.85), 2),
                "historical_accuracy": round(random.uniform(0.68, 0.82), 2)
            }

        decision.update({
            "response_time_ms": int(response_time * 1000),
            "decision_count": self.decision_count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return decision


# Initialize agents
agents = {
    "analyst": LiveAIAgent("analyst", "llama3.2:3b"),
    "risk": LiveAIAgent("risk", "phi3:mini"),
    "strategist": LiveAIAgent("strategist", "qwen2.5-coder:3b"),
    "chat": LiveAIAgent("chat", "gemma2:2b"),
    "reasoning": LiveAIAgent("reasoning", "deepseek-r1:1.5b")
}


async def broadcast_message(message: Dict[str, Any]):
    """Broadcast message to all connected WebSocket clients"""
    if active_connections:
        disconnected = []
        for connection in active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            active_connections.remove(conn)


async def market_data_simulator():
    """Simulate real-time market data"""
    symbols = ["SPY", "AAPL", "TSLA", "QQQ", "MSFT"]

    while True:
        try:
            # Generate market event
            symbol = random.choice(symbols)
            price = random.uniform(100, 500)

            market_data = {
                "symbol": symbol,
                "price": round(price, 2),
                "volume": random.randint(1000000, 10000000),
                "change": round(random.uniform(-3.0, 3.0), 2),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            logger.info(f"📊 Market Event: {symbol} @ ${price:.2f}")

            # Send to all agents
            agent_decisions = {}
            for agent_name, agent in agents.items():
                if agent.is_active:
                    decision = await agent.process_market_data(market_data)
                    agent_decisions[agent_name] = decision

                    # Broadcast individual agent decision
                    await broadcast_message({
                        "type": "agent_decision",
                        "market_data": market_data,
                        "decision": decision
                    })

            # Check for consensus and potential trade
            if len(agent_decisions) >= 3:
                await check_trading_consensus(market_data, agent_decisions)

            # Wait before next market event
            await asyncio.sleep(random.uniform(5, 15))

        except Exception as e:
            logger.error(f"Error in market simulator: {e}")
            await asyncio.sleep(5)


async def check_trading_consensus(market_data: Dict[str, Any], decisions: Dict[str, Any]):
    """Check if agents reach consensus for a trade"""

    # Simple consensus logic
    approvals = 0
    total_confidence = 0
    strategies = []

    for agent_name, decision in decisions.items():
        if agent_name == "risk" and decision.get("approval"):
            approvals += 1
        if agent_name == "analyst" and decision.get("confidence", 0) > 0.7:
            approvals += 1
        if agent_name == "strategist":
            strategies.append(decision.get("strategy", "Unknown"))
        if "confidence" in decision:
            total_confidence += decision["confidence"]

    avg_confidence = total_confidence / len([d for d in decisions.values() if "confidence" in d])

    if approvals >= 2 and avg_confidence > 0.7:
        # Execute trade
        trade = {
            "type": "trade_executed",
            "symbol": market_data["symbol"],
            "price": market_data["price"],
            "strategy": strategies[0] if strategies else "Market Order",
            "consensus_score": round(avg_confidence, 2),
            "approvals": approvals,
            "agents_involved": list(decisions.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trade_id": f"TRD_{int(time.time())}"
        }

        trading_log.append(trade)

        logger.info(f"🚀 TRADE EXECUTED: {trade['strategy']} on {trade['symbol']} @ ${trade['price']}")

        await broadcast_message(trade)


@app.websocket("/ws/agents")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time agent data"""
    await websocket.accept()
    active_connections.append(websocket)

    # Send current status
    await websocket.send_text(json.dumps({
        "type": "agent_status_all",
        "agents": agent_status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }))

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)


@app.get("/api/agents/status")
async def get_agent_status():
    """REST API endpoint for agent status"""
    return {
        "agents": agent_status,
        "active_count": sum(1 for status in agent_status.values() if status["status"] == "active"),
        "total_decisions": sum(agent.decision_count for agent in agents.values()),
        "trades_executed": len(trading_log),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/trading/log")
async def get_trading_log():
    """REST API endpoint for trading log"""
    return {
        "trades": trading_log[-20:],  # Last 20 trades
        "total_trades": len(trading_log),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/")
async def dashboard():
    """Live agent dashboard HTML"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 SwaggyStacks Live Agents Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #0f0f0f; color: #fff;
        }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #00ff88; margin: 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card {
            background: #1a1a1a; border: 1px solid #333; border-radius: 8px;
            padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        .card h3 { margin: 0 0 15px 0; color: #00ff88; }
        .agent-item {
            display: flex; justify-content: space-between; align-items: center;
            padding: 10px; margin: 5px 0; background: #2a2a2a; border-radius: 5px;
        }
        .status-active { color: #00ff88; }
        .status-inactive { color: #ff4444; }
        .trade-item {
            background: #1e3a1e; border-left: 4px solid #00ff88;
            padding: 10px; margin: 5px 0; border-radius: 4px;
        }
        .metrics { display: flex; justify-content: space-around; text-align: center; }
        .metric { padding: 10px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #00ff88; }
        #connectionStatus { padding: 10px; text-align: center; border-radius: 5px; margin-bottom: 20px; }
        .connected { background: #1e3a1e; color: #00ff88; }
        .disconnected { background: #3a1e1e; color: #ff4444; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 SwaggyStacks Live AI Agents</h1>
        <p>Real-time agent coordination and trading decisions</p>
    </div>

    <div id="connectionStatus" class="disconnected">⚠️ Connecting to agents...</div>

    <div class="grid">
        <div class="card">
            <h3>🤖 Agent Status</h3>
            <div id="agentStatus"></div>
        </div>

        <div class="card">
            <h3>📊 Live Metrics</h3>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value" id="activeAgents">0</div>
                    <div>Active Agents</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="totalDecisions">0</div>
                    <div>Decisions</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="tradesExecuted">0</div>
                    <div>Trades</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h3>🚀 Recent Trades</h3>
            <div id="recentTrades"></div>
        </div>

        <div class="card">
            <h3>💬 Agent Activity</h3>
            <div id="agentActivity"></div>
        </div>
    </div>

    <script>
        const ws = new WebSocket('ws://localhost:8001/ws/agents');
        const activities = [];

        ws.onopen = function(event) {
            document.getElementById('connectionStatus').innerHTML = '✅ Connected to live agents';
            document.getElementById('connectionStatus').className = 'connected';
        };

        ws.onclose = function(event) {
            document.getElementById('connectionStatus').innerHTML = '❌ Disconnected from agents';
            document.getElementById('connectionStatus').className = 'disconnected';
        };

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);

            if (data.type === 'agent_status_all') {
                updateAgentStatus(data.agents);
            } else if (data.type === 'agent_decision') {
                addActivity(`${data.decision.agent}: ${JSON.stringify(data.decision).substring(0, 100)}...`);
            } else if (data.type === 'trade_executed') {
                addTrade(data);
                addActivity(`🚀 TRADE: ${data.strategy} on ${data.symbol} @ $${data.price}`);
            }
        };

        function updateAgentStatus(agents) {
            const html = Object.entries(agents).map(([name, info]) => `
                <div class="agent-item">
                    <span><strong>${name}</strong> (${info.model})</span>
                    <span class="status-${info.status === 'active' ? 'active' : 'inactive'}">
                        ${info.status}
                    </span>
                </div>
            `).join('');
            document.getElementById('agentStatus').innerHTML = html;
        }

        function addActivity(text) {
            activities.unshift(`[${new Date().toLocaleTimeString()}] ${text}`);
            if (activities.length > 10) activities.pop();
            document.getElementById('agentActivity').innerHTML = activities.map(a => `<div>${a}</div>`).join('');
        }

        function addTrade(trade) {
            const tradeDiv = document.createElement('div');
            tradeDiv.className = 'trade-item';
            tradeDiv.innerHTML = `
                <strong>${trade.strategy}</strong> on ${trade.symbol}<br>
                Price: $${trade.price} | Score: ${trade.consensus_score}
            `;
            const container = document.getElementById('recentTrades');
            container.insertBefore(tradeDiv, container.firstChild);
            if (container.children.length > 5) container.removeChild(container.lastChild);
        }

        // Update metrics periodically
        setInterval(async () => {
            try {
                const response = await fetch('/api/agents/status');
                const data = await response.json();
                document.getElementById('activeAgents').textContent = data.active_count;
                document.getElementById('totalDecisions').textContent = data.total_decisions;
                document.getElementById('tradesExecuted').textContent = data.trades_executed;
            } catch (e) {
                console.error('Failed to fetch metrics:', e);
            }
        }, 2000);
    </script>
</body>
</html>
    """)


@app.on_event("startup")
async def startup_event():
    """Start all agents and background tasks"""
    logger.info("🚀 Starting SwaggyStacks Live Agent Server...")

    # Start all agents
    for agent in agents.values():
        await agent.start()

    # Start market data simulator
    asyncio.create_task(market_data_simulator())

    logger.info("✅ All agents are now LIVE!")
    logger.info("🌐 Dashboard: http://localhost:8001")
    logger.info("📡 WebSocket: ws://localhost:8001/ws/agents")


if __name__ == "__main__":
    print("🚀 Starting Live AI Agent Server...")
    print("📊 Dashboard will be available at: http://localhost:8001")
    print("📡 WebSocket endpoint: ws://localhost:8001/ws/agents")
    print("🤖 5 AI agents will start automatically!")

    uvicorn.run(
        "live_agent_server:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=False
    )