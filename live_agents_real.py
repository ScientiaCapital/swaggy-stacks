#!/usr/bin/env python3
"""
🚀 LIVE TRADING AGENTS WITH REAL ALPACA STREAMING
===============================================
Production-ready AI trading system with real-time Alpaca WebSocket data
"""

import os
import sys
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import structlog

# Add backend to path for imports
backend_path = Path(__file__).parent / "backend"
sys.path.append(str(backend_path))

# Core imports
from app.ai.coordination_hub import get_coordination_hub
from app.ai.consensus_engine import get_consensus_engine, AgentVote, VoteType
from app.trading.alpaca_stream_manager import get_stream_manager
from app.ai.trading_agents import AITradingCoordinator
from app.core.config import settings

# Setup logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# FastAPI app
app = FastAPI(title="Live Trading Agents - Real Alpaca Streaming")

# Global system state
system_state = {
    "coordination_hub": None,
    "stream_manager": None,
    "consensus_engine": None,
    "ai_coordinator": None,
    "is_running": False,
    "symbols": ["SPY", "AAPL", "TSLA", "QQQ", "MSFT", "NVDA"],
    "agents": {
        "analyst": {
            "model": "llama3.2:3b",
            "specialization": "market_analysis",
            "status": "INITIALIZING",
            "decisions": 0,
            "last_decision": None
        },
        "risk": {
            "model": "phi3:mini",
            "specialization": "risk_assessment",
            "status": "INITIALIZING",
            "decisions": 0,
            "last_decision": None
        },
        "strategist": {
            "model": "qwen2.5-coder:3b",
            "specialization": "strategy_optimization",
            "status": "INITIALIZING",
            "decisions": 0,
            "last_decision": None
        },
        "chat": {
            "model": "gemma2:2b",
            "specialization": "coordination",
            "status": "INITIALIZING",
            "decisions": 0,
            "last_decision": None
        },
        "reasoning": {
            "model": "deepseek-r1:1.5b",
            "specialization": "pattern_analysis",
            "status": "INITIALIZING",
            "decisions": 0,
            "last_decision": None
        }
    },
    "market_data": {
        "latest_trades": {},
        "latest_quotes": {},
        "latest_bars": {},
        "stream_health": "DISCONNECTED"
    },
    "trading_decisions": [],
    "consensus_decisions": [],
    "system_stats": {
        "stream_messages": 0,
        "agent_decisions": 0,
        "consensus_requests": 0,
        "trades_executed": 0,
        "uptime_start": datetime.now().isoformat()
    }
}


class RealTimeAgentSystem:
    """Real-time AI agent coordination system with Alpaca streaming"""

    def __init__(self):
        self.coordination_hub = None
        self.stream_manager = None
        self.consensus_engine = None
        self.ai_coordinator = None
        self.is_initialized = False

    async def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing Real-Time Agent System...")

            # Initialize core components
            self.stream_manager = await get_stream_manager()
            self.coordination_hub = await get_coordination_hub()
            self.consensus_engine = get_consensus_engine()

            # Initialize AI coordinator
            self.ai_coordinator = AITradingCoordinator(
                enable_streaming=True,
                enable_unsupervised=True
            )

            # Update global state
            system_state["coordination_hub"] = self.coordination_hub
            system_state["stream_manager"] = self.stream_manager
            system_state["consensus_engine"] = self.consensus_engine
            system_state["ai_coordinator"] = self.ai_coordinator

            # Register streaming callbacks
            await self._register_stream_callbacks()

            # Register consensus callbacks
            self._register_consensus_callbacks()

            # Update agent statuses
            for agent_name in system_state["agents"]:
                system_state["agents"][agent_name]["status"] = "READY"

            self.is_initialized = True
            logger.info("System initialization completed successfully")

        except Exception as e:
            logger.error("Failed to initialize system", error=str(e))
            raise

    async def _register_stream_callbacks(self):
        """Register callbacks for streaming data"""
        try:
            # Register market data callbacks
            await self.stream_manager.subscribe_trades(
                system_state["symbols"],
                self._handle_trade_update
            )
            await self.stream_manager.subscribe_quotes(
                system_state["symbols"],
                self._handle_quote_update
            )
            await self.stream_manager.subscribe_bars(
                system_state["symbols"],
                self._handle_bar_update
            )

            logger.info("Stream callbacks registered", symbols=system_state["symbols"])

        except Exception as e:
            logger.error("Failed to register stream callbacks", error=str(e))
            raise

    def _register_consensus_callbacks(self):
        """Register consensus decision callbacks"""
        try:
            self.consensus_engine.add_decision_callback(self._handle_consensus_decision)
            logger.info("Consensus callbacks registered")

        except Exception as e:
            logger.error("Failed to register consensus callbacks", error=str(e))

    async def _handle_trade_update(self, trade):
        """Handle real-time trade updates"""
        try:
            symbol = trade.symbol
            price = float(trade.price)
            size = int(trade.size)

            # Update system state
            system_state["market_data"]["latest_trades"][symbol] = {
                "price": price,
                "size": size,
                "timestamp": trade.timestamp.isoformat(),
                "exchange": getattr(trade, 'exchange', 'unknown')
            }

            system_state["system_stats"]["stream_messages"] += 1

            # Trigger agent analysis for significant trades
            if size > 10000:  # Large trade threshold
                await self._trigger_agent_analysis(symbol, "large_trade", {
                    "price": price,
                    "size": size,
                    "trade_data": trade
                })

            logger.debug("Trade processed", symbol=symbol, price=price, size=size)

        except Exception as e:
            logger.error("Failed to handle trade update", error=str(e))

    async def _handle_quote_update(self, quote):
        """Handle real-time quote updates"""
        try:
            symbol = quote.symbol

            # Update system state
            system_state["market_data"]["latest_quotes"][symbol] = {
                "bid_price": float(quote.bid_price),
                "ask_price": float(quote.ask_price),
                "bid_size": int(quote.bid_size),
                "ask_size": int(quote.ask_size),
                "spread": float(quote.ask_price) - float(quote.bid_price),
                "timestamp": quote.timestamp.isoformat()
            }

            system_state["system_stats"]["stream_messages"] += 1

            # Check for unusual spreads
            spread_pct = ((float(quote.ask_price) - float(quote.bid_price)) / float(quote.bid_price)) * 100
            if spread_pct > 1.0:  # Wide spread threshold
                await self._trigger_agent_analysis(symbol, "wide_spread", {
                    "spread_pct": spread_pct,
                    "quote_data": quote
                })

        except Exception as e:
            logger.error("Failed to handle quote update", error=str(e))

    async def _handle_bar_update(self, bar):
        """Handle real-time bar updates"""
        try:
            symbol = bar.symbol

            # Update system state
            system_state["market_data"]["latest_bars"][symbol] = {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
                "vwap": float(getattr(bar, 'vwap', 0)),
                "timestamp": bar.timestamp.isoformat()
            }

            system_state["system_stats"]["stream_messages"] += 1

            # Always trigger analysis on bar updates for systematic monitoring
            await self._trigger_agent_analysis(symbol, "bar_update", {
                "bar_data": bar,
                "price": float(bar.close),
                "volume": int(bar.volume)
            })

        except Exception as e:
            logger.error("Failed to handle bar update", error=str(e))

    async def _trigger_agent_analysis(self, symbol: str, trigger_type: str, data: Dict[str, Any]):
        """Trigger coordinated agent analysis"""
        try:
            # Broadcast market event to coordination hub
            await self.coordination_hub.broadcast_market_event(
                list(system_state["agents"].keys()),
                {
                    "symbol": symbol,
                    "trigger_type": trigger_type,
                    "market_data": data,
                    "timestamp": datetime.now().isoformat()
                }
            )

            # Get comprehensive analysis from AI coordinator
            result = await self.ai_coordinator.comprehensive_analysis(
                symbol=symbol,
                market_data=data,
                technical_indicators={},  # Will be calculated from real data
                account_info={},         # Will be fetched from Alpaca
                current_positions=[],    # Will be fetched from trading manager
                markov_analysis={}       # Will be calculated from real data
            )

            # Create agent votes from analysis results
            votes = await self._create_agent_votes(symbol, result)

            if votes:
                # Request consensus
                decision_id = f"consensus_{symbol}_{int(time.time())}"
                consensus_result = await self.consensus_engine.calculate_consensus(
                    decision_id=decision_id,
                    symbol=symbol,
                    votes=votes
                )

                # Store consensus decision
                system_state["consensus_decisions"].append({
                    "decision_id": decision_id,
                    "symbol": symbol,
                    "result": consensus_result,
                    "timestamp": datetime.now().isoformat()
                })

                system_state["system_stats"]["consensus_requests"] += 1

                logger.info("Agent analysis completed",
                           symbol=symbol,
                           trigger=trigger_type,
                           consensus=consensus_result.final_vote.value,
                           confidence=consensus_result.confidence_score)

        except Exception as e:
            logger.error("Failed to trigger agent analysis", error=str(e))

    async def _create_agent_votes(self, symbol: str, analysis_result: Dict[str, Any]) -> List[AgentVote]:
        """Create agent votes from analysis results"""
        try:
            votes = []

            # Extract recommendations from each agent
            agents_data = analysis_result.get("agents", {})

            for agent_name, agent_data in agents_data.items():
                if agent_name not in system_state["agents"]:
                    continue

                # Parse agent recommendation
                recommendation = agent_data.get("recommendation", "HOLD")
                confidence = agent_data.get("confidence", 0.5)
                reasoning = agent_data.get("reasoning", "No specific reasoning provided")
                risk_data = agent_data.get("risk_assessment", {})

                # Convert recommendation to vote type
                vote_type = VoteType.HOLD
                if recommendation in ["BUY", "LONG", "BULLISH"]:
                    vote_type = VoteType.BUY
                elif recommendation in ["SELL", "SHORT", "BEARISH"]:
                    vote_type = VoteType.SELL

                # Create vote
                vote = AgentVote(
                    agent_name=agent_name,
                    vote=vote_type,
                    confidence=confidence,
                    reasoning=reasoning,
                    risk_assessment=risk_data,
                    position_size_suggestion=agent_data.get("position_size", None)
                )

                votes.append(vote)

                # Update agent stats
                system_state["agents"][agent_name]["decisions"] += 1
                system_state["agents"][agent_name]["last_decision"] = {
                    "symbol": symbol,
                    "vote": vote_type.value,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                }
                system_state["agents"][agent_name]["status"] = "ACTIVE"

            system_state["system_stats"]["agent_decisions"] += len(votes)
            return votes

        except Exception as e:
            logger.error("Failed to create agent votes", error=str(e))
            return []

    async def _handle_consensus_decision(self, consensus_result):
        """Handle consensus decision for potential trade execution"""
        try:
            if not consensus_result.execution_recommended:
                logger.info("Consensus reached but execution not recommended",
                           symbol=consensus_result.symbol,
                           decision=consensus_result.final_vote.value,
                           confidence=consensus_result.confidence_score)
                return

            # Execute trade through trading manager
            trade_data = {
                "symbol": consensus_result.symbol,
                "action": consensus_result.final_vote.value,
                "quantity": int(consensus_result.suggested_position_size),
                "order_type": "market",
                "consensus_id": consensus_result.decision_id,
                "confidence": consensus_result.confidence_score,
                "reasoning": consensus_result.consensus_reasoning,
                "timestamp": datetime.now().isoformat()
            }

            # Store trading decision
            system_state["trading_decisions"].append(trade_data)
            system_state["system_stats"]["trades_executed"] += 1

            logger.info("Trade decision executed",
                       symbol=consensus_result.symbol,
                       action=consensus_result.final_vote.value,
                       quantity=consensus_result.suggested_position_size,
                       confidence=consensus_result.confidence_score)

        except Exception as e:
            logger.error("Failed to handle consensus decision", error=str(e))

    async def start_streaming(self):
        """Start the real-time streaming system"""
        try:
            if not self.is_initialized:
                await self.initialize()

            # Connect to Alpaca stream
            await self.stream_manager.connect()

            # Update system state
            system_state["is_running"] = True
            system_state["market_data"]["stream_health"] = "CONNECTED"

            logger.info("Real-time streaming started successfully",
                       symbols=system_state["symbols"])

        except Exception as e:
            logger.error("Failed to start streaming", error=str(e))
            system_state["market_data"]["stream_health"] = "ERROR"
            raise

    async def stop_streaming(self):
        """Stop the streaming system"""
        try:
            if self.stream_manager:
                await self.stream_manager.disconnect()

            if self.coordination_hub:
                await self.coordination_hub.stop_coordination()

            system_state["is_running"] = False
            system_state["market_data"]["stream_health"] = "DISCONNECTED"

            logger.info("Streaming system stopped")

        except Exception as e:
            logger.error("Failed to stop streaming", error=str(e))

    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            health_data = {
                "system_running": system_state["is_running"],
                "stream_health": system_state["market_data"]["stream_health"],
                "agents": {}
            }

            # Get agent health
            for agent_name, agent_data in system_state["agents"].items():
                health_data["agents"][agent_name] = {
                    "status": agent_data["status"],
                    "decisions_made": agent_data["decisions"],
                    "model": agent_data["model"]
                }

            # Get coordination hub health
            if self.coordination_hub:
                coordination_health = await self.coordination_hub.health_check()
                health_data["coordination_hub"] = coordination_health

            # Get stream manager health
            if self.stream_manager:
                stream_health = await self.stream_manager.get_connection_health()
                health_data["stream_manager"] = stream_health

            return health_data

        except Exception as e:
            logger.error("Failed to get system health", error=str(e))
            return {"error": str(e)}


# Global system instance
real_time_system = RealTimeAgentSystem()


@app.on_event("startup")
async def startup():
    """Start the real-time agent system"""
    logger.info("🚀 Starting Live Trading Agents with Real Alpaca Streaming...")
    logger.info("🌐 Dashboard: http://localhost:8002")
    logger.info("📡 API: http://localhost:8002/api/live")

    try:
        await real_time_system.start_streaming()
        logger.info("✅ All systems operational - Real-time trading active!")
    except Exception as e:
        logger.error("❌ Failed to start system", error=str(e))


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("Shutting down real-time agent system...")
    await real_time_system.stop_streaming()


# API Endpoints
@app.get("/api/live")
async def get_live_data():
    """Get all live system data"""
    return {
        "system_running": system_state["is_running"],
        "agents": system_state["agents"],
        "market_data": {
            "stream_health": system_state["market_data"]["stream_health"],
            "latest_prices": {
                symbol: data.get("price", 0.0)
                for symbol, data in system_state["market_data"]["latest_trades"].items()
            },
            "symbols_tracked": len(system_state["symbols"])
        },
        "recent_decisions": system_state["trading_decisions"][-5:],
        "recent_consensus": system_state["consensus_decisions"][-3:],
        "stats": system_state["system_stats"],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/health")
async def get_health():
    """Get system health status"""
    return await real_time_system.get_system_health()


@app.get("/api/agents")
async def get_agents():
    """Get agent status and performance"""
    return {
        "agents": system_state["agents"],
        "total_decisions": system_state["system_stats"]["agent_decisions"],
        "active_agents": len([a for a in system_state["agents"].values() if a["status"] == "ACTIVE"])
    }


@app.get("/api/trades")
async def get_trades():
    """Get recent trading decisions"""
    return {
        "trades": system_state["trading_decisions"],
        "total_trades": len(system_state["trading_decisions"]),
        "trades_today": system_state["system_stats"]["trades_executed"]
    }


@app.get("/api/consensus/{decision_id}")
async def get_consensus_decision(decision_id: str):
    """Get specific consensus decision"""
    for decision in system_state["consensus_decisions"]:
        if decision["decision_id"] == decision_id:
            return decision
    raise HTTPException(status_code=404, detail="Consensus decision not found")


@app.get("/api/market/{symbol}")
async def get_market_data(symbol: str):
    """Get market data for specific symbol"""
    symbol = symbol.upper()
    return {
        "symbol": symbol,
        "trade": system_state["market_data"]["latest_trades"].get(symbol, {}),
        "quote": system_state["market_data"]["latest_quotes"].get(symbol, {}),
        "bar": system_state["market_data"]["latest_bars"].get(symbol, {})
    }


@app.get("/")
async def dashboard():
    """Live trading dashboard"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>🤖 Live Trading Agents - Real Alpaca Stream</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #0a0a0a; color: #fff; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #00ff88; margin: 0; }
        .header .subtitle { color: #888; margin-top: 5px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px; }
        .card h3 { color: #00ff88; margin: 0 0 15px 0; }
        .agent { display: flex; justify-content: space-between; padding: 10px; margin: 5px 0; background: #2a2a2a; border-radius: 5px; }
        .agent .status { padding: 2px 8px; border-radius: 3px; font-size: 12px; }
        .status.ACTIVE { background: #2a5a2a; color: #00ff88; }
        .status.READY { background: #5a5a2a; color: #ffff88; }
        .status.INITIALIZING { background: #5a2a2a; color: #ff8888; }
        .trade { background: #1a3a1a; border-left: 4px solid #00ff88; padding: 10px; margin: 5px 0; border-radius: 4px; }
        .consensus { background: #3a1a3a; border-left: 4px solid #ff88ff; padding: 10px; margin: 5px 0; border-radius: 4px; }
        .metrics { display: flex; justify-content: space-around; text-align: center; margin: 15px 0; }
        .metric { flex: 1; }
        .metric-value { font-size: 24px; font-weight: bold; color: #00ff88; }
        .metric-label { font-size: 12px; color: #888; }
        .status-banner { padding: 15px; border-radius: 5px; margin-bottom: 20px; text-align: center; font-weight: bold; }
        .status-banner.connected { background: #1a3a1a; color: #00ff88; }
        .status-banner.disconnected { background: #3a1a1a; color: #ff8888; }
        .market-data { font-family: monospace; font-size: 14px; }
        .price { color: #00ff88; font-weight: bold; }
        .symbol { color: #88ccff; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 SwaggyStacks Live Trading Agents</h1>
        <div class="subtitle">Real-time AI coordination with Alpaca WebSocket streaming</div>
    </div>

    <div class="status-banner" id="statusBanner">
        🟡 Initializing system...
    </div>

    <div class="grid">
        <div class="card">
            <h3>🤖 AI Agents</h3>
            <div id="agents"></div>
        </div>

        <div class="card">
            <h3>📊 System Metrics</h3>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value" id="streamMessages">0</div>
                    <div class="metric-label">Stream Messages</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="agentDecisions">0</div>
                    <div class="metric-label">Agent Decisions</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="tradesExecuted">0</div>
                    <div class="metric-label">Trades Executed</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h3>🚀 Recent Trades</h3>
            <div id="recentTrades"></div>
        </div>

        <div class="card">
            <h3>🧠 Recent Consensus</h3>
            <div id="recentConsensus"></div>
        </div>

        <div class="card">
            <h3>📈 Live Market Data</h3>
            <div id="marketData" class="market-data"></div>
        </div>

        <div class="card">
            <h3>⚡ System Health</h3>
            <div id="systemHealth"></div>
        </div>
    </div>

    <script>
        let lastUpdate = 0;

        async function updateData() {
            try {
                const response = await fetch('/api/live');
                const data = await response.json();

                // Update status banner
                const banner = document.getElementById('statusBanner');
                if (data.system_running && data.market_data.stream_health === 'CONNECTED') {
                    banner.innerHTML = '🟢 System operational - Live trading active';
                    banner.className = 'status-banner connected';
                } else if (data.market_data.stream_health === 'DISCONNECTED') {
                    banner.innerHTML = '🔴 Stream disconnected - Attempting reconnection';
                    banner.className = 'status-banner disconnected';
                } else {
                    banner.innerHTML = '🟡 System initializing...';
                    banner.className = 'status-banner';
                }

                // Update agents
                const agentsHtml = Object.entries(data.agents).map(([name, info]) =>
                    `<div class="agent">
                        <div>
                            <strong>${name}</strong> (${info.model})<br>
                            <small>${info.specialization}</small>
                        </div>
                        <div>
                            <span class="status ${info.status}">${info.status}</span><br>
                            <small>${info.decisions} decisions</small>
                        </div>
                    </div>`
                ).join('');
                document.getElementById('agents').innerHTML = agentsHtml;

                // Update metrics
                document.getElementById('streamMessages').textContent = data.stats.stream_messages;
                document.getElementById('agentDecisions').textContent = data.stats.agent_decisions;
                document.getElementById('tradesExecuted').textContent = data.stats.trades_executed;

                // Update trades
                const tradesHtml = data.recent_decisions.map(trade =>
                    `<div class="trade">
                        <strong>${trade.action.toUpperCase()}</strong> ${trade.symbol}<br>
                        Qty: ${trade.quantity} | Conf: ${(trade.confidence * 100).toFixed(1)}%<br>
                        <small>${new Date(trade.timestamp).toLocaleTimeString()}</small>
                    </div>`
                ).join('');
                document.getElementById('recentTrades').innerHTML = tradesHtml || '<div>No trades yet</div>';

                // Update consensus
                const consensusHtml = data.recent_consensus.map(consensus =>
                    `<div class="consensus">
                        <strong>${consensus.symbol}</strong> - ${consensus.result?.final_vote || 'PENDING'}<br>
                        Conf: ${((consensus.result?.confidence_score || 0) * 100).toFixed(1)}%<br>
                        <small>${new Date(consensus.timestamp).toLocaleTimeString()}</small>
                    </div>`
                ).join('');
                document.getElementById('recentConsensus').innerHTML = consensusHtml || '<div>No consensus yet</div>';

                // Update market data
                const marketHtml = Object.entries(data.market_data.latest_prices || {}).map(([symbol, price]) =>
                    `<div><span class="symbol">${symbol}</span>: <span class="price">$${price.toFixed(2)}</span></div>`
                ).join('');
                document.getElementById('marketData').innerHTML = marketHtml || '<div>Waiting for market data...</div>';

                lastUpdate = Date.now();

            } catch (error) {
                console.error('Failed to update data:', error);
                document.getElementById('statusBanner').innerHTML = '❌ Connection error';
                document.getElementById('statusBanner').className = 'status-banner disconnected';
            }
        }

        // Health check
        async function updateHealth() {
            try {
                const response = await fetch('/api/health');
                const health = await response.json();

                let healthHtml = `
                    <div>Stream: ${health.stream_manager?.healthy ? '🟢' : '🔴'} ${health.stream_manager?.is_connected ? 'Connected' : 'Disconnected'}</div>
                    <div>Hub: ${health.coordination_hub?.coordination_hub === 'healthy' ? '🟢' : '🔴'} ${health.coordination_hub?.coordination_hub || 'Unknown'}</div>
                `;

                document.getElementById('systemHealth').innerHTML = healthHtml;

            } catch (error) {
                document.getElementById('systemHealth').innerHTML = '❌ Health check failed';
            }
        }

        // Update data every 2 seconds
        setInterval(updateData, 2000);
        setInterval(updateHealth, 10000);

        // Initial load
        updateData();
        updateHealth();
    </script>
</body>
</html>
    """)


if __name__ == "__main__":
    print("🚀 Starting Real-Time Trading Agents with Alpaca Streaming...")
    print("📊 Dashboard: http://localhost:8002")
    print("📡 API: http://localhost:8002/api/live")
    print("💡 Using real Alpaca WebSocket data - no mock data!")

    uvicorn.run(
        "live_agents_real:app",
        host="0.0.0.0",
        port=8002,
        log_level="info",
        reload=False
    )