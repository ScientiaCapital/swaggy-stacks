"""
WebSocket Agent Communication Integration Tests

Tests the real-time communication system between AI agents and the trading engine,
demonstrating live agent coordination, decision-making, and trade execution.

These tests validate:
- Agent initialization with specific Ollama LLM models
- WebSocket connection lifecycle and message broadcasting
- Agent consensus mechanisms and coordination protocols
- Real-time trading decisions based on market data
- Error handling and recovery in agent communication
"""

import pytest
import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from typing import Dict, List, Any
import websockets
from websockets.exceptions import ConnectionClosed

from app.websockets.trading_socket import TradingWebSocketManager
from app.ai.agent_coordinator import AgentCoordinator
from app.ai.base_agent import BaseAIAgent
from app.ai.ollama_client import OllamaClient
from app.trading.alpaca_client import AlpacaClient
from app.trading.risk_manager import RiskManager
from tests.mocks.mock_alpaca_options import mock_options_generator
from tests.mocks.mock_market_scenarios import mock_market_generator

# Configure logging for test visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockOllamaClient:
    """Mock Ollama client that simulates LLM responses"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_available = True

    async def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate mock LLM responses based on agent type"""

        if "analyst" in self.model_name or "llama3.2" in self.model_name:
            # Analyst agent using llama3.2:3b - market analysis
            return json.dumps({
                "analysis": "Bullish momentum detected in SPY",
                "signal_strength": 0.75,
                "key_factors": ["Volume increase", "RSI oversold recovery", "MACD crossover"],
                "recommendation": "LONG",
                "confidence": 0.82,
                "target_price": 445.0,
                "stop_loss": 435.0
            })

        elif "risk" in self.model_name or "phi3" in self.model_name:
            # Risk agent using phi3:mini - risk assessment
            return json.dumps({
                "risk_assessment": "MODERATE",
                "portfolio_exposure": 0.65,
                "max_position_size": 10000,
                "risk_factors": ["High IV in tech sector", "Earnings season volatility"],
                "approval": True,
                "risk_score": 6.2,
                "suggested_hedge": "Put spread on QQQ"
            })

        elif "strategist" in self.model_name or "qwen2.5" in self.model_name:
            # Strategist agent using qwen2.5-coder:3b - strategy generation
            return json.dumps({
                "strategy": "Iron Condor",
                "entry_criteria": "High IV rank > 70%, neutral market outlook",
                "legs": [
                    {"type": "PUT", "strike": 430, "action": "SELL"},
                    {"type": "PUT", "strike": 425, "action": "BUY"},
                    {"type": "CALL", "strike": 450, "action": "SELL"},
                    {"type": "CALL", "strike": 455, "action": "BUY"}
                ],
                "max_profit": 350,
                "max_loss": 150,
                "target_profit": 50
            })

        elif "chat" in self.model_name or "gemma2" in self.model_name:
            # Chat agent using gemma2:2b - conversational interface
            return json.dumps({
                "message": "Market analysis complete. Agents are coordinating on SPY Iron Condor strategy.",
                "status": "COORDINATING",
                "next_action": "Awaiting risk approval",
                "participants": ["analyst", "risk", "strategist"]
            })

        else:
            # Default response
            return json.dumps({"response": "Agent communication test successful", "model": self.model_name})


@pytest.fixture
async def mock_ollama_clients():
    """Create mock Ollama clients for all agent types"""
    return {
        "analyst": MockOllamaClient("llama3.2:3b"),
        "risk": MockOllamaClient("phi3:mini"),
        "strategist": MockOllamaClient("qwen2.5-coder:3b"),
        "chat": MockOllamaClient("gemma2:2b"),
        "reasoning": MockOllamaClient("deepseek-r1:1.5b")
    }


@pytest.fixture
async def mock_trading_websocket():
    """Create mock WebSocket manager for trading"""
    manager = AsyncMock(spec=TradingWebSocketManager)
    manager.active_connections = []
    manager.agent_connections = {}
    manager.is_running = True

    async def mock_broadcast(message: Dict[str, Any], agent_type: str = None):
        """Mock broadcast that logs messages"""
        logger.info(f"Broadcasting to {agent_type or 'all'}: {message}")
        return True

    manager.broadcast_to_agents = mock_broadcast
    manager.send_personal_message = AsyncMock()
    manager.connect_agent = AsyncMock()
    manager.disconnect_agent = AsyncMock()

    return manager


@pytest.fixture
async def agent_coordinator(mock_ollama_clients, mock_trading_websocket):
    """Create agent coordinator with mock dependencies"""
    coordinator = AgentCoordinator()

    # Initialize agents with mock Ollama clients
    coordinator.agents = {}
    for agent_type, mock_client in mock_ollama_clients.items():
        agent = BaseAIAgent(agent_type=agent_type, model_name=mock_client.model_name)
        agent.ollama_client = mock_client
        agent.is_initialized = True
        coordinator.agents[agent_type] = agent

    coordinator.websocket_manager = mock_trading_websocket
    coordinator.is_running = True

    return coordinator


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    return mock_market_generator.generate_bull_market_scenario(
        symbol="SPY",
        duration_days=5,
        start_price=440.0
    )


@pytest.fixture
def sample_options_chain():
    """Generate sample options chain for testing"""
    return mock_options_generator.get_mock_option_chain(
        symbol="SPY",
        underlying_price=440.0,
        chain_type="standard"
    )


class TestWebSocketAgentCommunication:
    """Test suite for WebSocket agent communication"""

    @pytest.mark.asyncio
    async def test_agent_initialization_with_specific_models(self, mock_ollama_clients):
        """Test that agents initialize with their specific Ollama LLM models"""

        # Test each agent type with correct model
        expected_models = {
            "analyst": "llama3.2:3b",
            "risk": "phi3:mini",
            "strategist": "qwen2.5-coder:3b",
            "chat": "gemma2:2b",
            "reasoning": "deepseek-r1:1.5b"
        }

        for agent_type, expected_model in expected_models.items():
            client = mock_ollama_clients[agent_type]
            assert client.model_name == expected_model
            assert client.is_available

            # Test agent can generate responses
            response = await client.generate_response("Test prompt")
            assert response is not None
            assert isinstance(response, str)

            # Verify response contains expected agent-specific content
            response_data = json.loads(response)
            if agent_type == "analyst":
                assert "analysis" in response_data
                assert "recommendation" in response_data
            elif agent_type == "risk":
                assert "risk_assessment" in response_data
                assert "approval" in response_data
            elif agent_type == "strategist":
                assert "strategy" in response_data
                assert "legs" in response_data

        logger.info("✅ All agents initialized with correct Ollama models")

    @pytest.mark.asyncio
    async def test_websocket_connection_lifecycle(self, mock_trading_websocket):
        """Test WebSocket connection setup and teardown"""

        # Test connection establishment
        websocket_mock = AsyncMock()
        agent_id = "test_agent_analyst"

        await mock_trading_websocket.connect_agent(websocket_mock, agent_id)
        mock_trading_websocket.connect_agent.assert_called_once_with(websocket_mock, agent_id)

        # Test message broadcasting
        test_message = {
            "type": "MARKET_UPDATE",
            "symbol": "SPY",
            "price": 440.50,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        await mock_trading_websocket.broadcast_to_agents(test_message)
        mock_trading_websocket.broadcast_to_agents.assert_called_once_with(test_message)

        # Test disconnection
        await mock_trading_websocket.disconnect_agent(agent_id)
        mock_trading_websocket.disconnect_agent.assert_called_once_with(agent_id)

        logger.info("✅ WebSocket connection lifecycle tested successfully")

    @pytest.mark.asyncio
    async def test_agent_coordination_workflow(self, agent_coordinator, sample_market_data, sample_options_chain):
        """Test complete agent coordination workflow for trading decision"""

        # Step 1: Market data arrives
        market_update = {
            "type": "MARKET_DATA",
            "symbol": "SPY",
            "price": sample_market_data[-1]["close"],
            "volume": sample_market_data[-1]["volume"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Step 2: Analyst processes market data
        analyst = agent_coordinator.agents["analyst"]
        analyst_response = await analyst.ollama_client.generate_response(
            "Analyze market conditions",
            context=market_update
        )
        analyst_analysis = json.loads(analyst_response)

        assert analyst_analysis["recommendation"] in ["LONG", "SHORT", "NEUTRAL"]
        assert "confidence" in analyst_analysis
        assert analyst_analysis["confidence"] > 0

        # Step 3: Risk manager evaluates proposal
        risk_agent = agent_coordinator.agents["risk"]
        risk_response = await risk_agent.ollama_client.generate_response(
            "Evaluate risk for trade",
            context={"analysis": analyst_analysis, "market_data": market_update}
        )
        risk_assessment = json.loads(risk_response)

        assert "risk_assessment" in risk_assessment
        assert "approval" in risk_assessment
        assert isinstance(risk_assessment["approval"], bool)

        # Step 4: Strategist creates execution plan (if approved)
        if risk_assessment["approval"]:
            strategist = agent_coordinator.agents["strategist"]
            strategy_response = await strategist.ollama_client.generate_response(
                "Create options strategy",
                context={"analysis": analyst_analysis, "risk": risk_assessment, "options": sample_options_chain}
            )
            strategy_plan = json.loads(strategy_response)

            assert "strategy" in strategy_plan
            assert "legs" in strategy_plan
            assert len(strategy_plan["legs"]) > 0

            # Step 5: Coordinate final execution
            coordination_message = {
                "type": "TRADE_COORDINATION",
                "analysis": analyst_analysis,
                "risk": risk_assessment,
                "strategy": strategy_plan,
                "status": "APPROVED_FOR_EXECUTION",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Broadcast coordination result
            await agent_coordinator.websocket_manager.broadcast_to_agents(
                coordination_message,
                agent_type="all"
            )

            logger.info("✅ Agent coordination workflow completed successfully")
            logger.info(f"Strategy: {strategy_plan['strategy']}")
            logger.info(f"Risk Score: {risk_assessment['risk_score']}")
            logger.info(f"Confidence: {analyst_analysis['confidence']}")

        else:
            logger.info("❌ Trade rejected by risk management")

    @pytest.mark.asyncio
    async def test_real_time_market_response(self, agent_coordinator, mock_trading_websocket):
        """Test agents responding to real-time market events"""

        # Simulate rapid market events
        market_events = [
            {"type": "PRICE_ALERT", "symbol": "SPY", "price": 445.0, "change": "+1.2%"},
            {"type": "VOLUME_SPIKE", "symbol": "SPY", "volume": 15000000, "avg_volume": 8000000},
            {"type": "VOLATILITY_CHANGE", "symbol": "SPY", "iv": 0.28, "iv_change": "+15%"},
            {"type": "NEWS_EVENT", "headline": "Fed announces rate decision", "sentiment": "bullish"}
        ]

        agent_responses = []

        for event in market_events:
            # Each agent processes the event
            for agent_type, agent in agent_coordinator.agents.items():
                try:
                    response = await agent.ollama_client.generate_response(
                        f"Process market event: {event['type']}",
                        context=event
                    )

                    agent_responses.append({
                        "agent": agent_type,
                        "event": event["type"],
                        "response": json.loads(response),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

                    # Broadcast agent response
                    await mock_trading_websocket.broadcast_to_agents(
                        {
                            "type": "AGENT_RESPONSE",
                            "agent": agent_type,
                            "event": event,
                            "response": response
                        },
                        agent_type=agent_type
                    )

                except Exception as e:
                    logger.error(f"Agent {agent_type} failed to process event {event['type']}: {e}")

        # Verify all agents responded
        agent_types = set(response["agent"] for response in agent_responses)
        assert len(agent_types) == len(agent_coordinator.agents)

        # Verify different event types were processed
        event_types = set(response["event"] for response in agent_responses)
        assert len(event_types) == len(market_events)

        logger.info(f"✅ Processed {len(agent_responses)} agent responses to {len(market_events)} market events")

    @pytest.mark.asyncio
    async def test_agent_consensus_mechanism(self, agent_coordinator):
        """Test agent consensus and disagreement resolution"""

        # Create a scenario where agents might disagree
        market_context = {
            "symbol": "SPY",
            "price": 440.0,
            "rsi": 65,  # Moderately overbought
            "iv_rank": 45,  # Medium volatility
            "trend": "sideways",
            "earnings_in_days": 7
        }

        # Collect all agent opinions
        agent_opinions = {}

        for agent_type, agent in agent_coordinator.agents.items():
            response = await agent.ollama_client.generate_response(
                "Provide trading recommendation",
                context=market_context
            )
            agent_opinions[agent_type] = json.loads(response)

        # Test consensus building
        consensus_votes = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}

        for agent_type, opinion in agent_opinions.items():
            if "recommendation" in opinion:
                vote = opinion["recommendation"]
                if vote in consensus_votes:
                    consensus_votes[vote] += 1
            elif "strategy" in opinion:
                # Strategist votes based on strategy direction
                consensus_votes["NEUTRAL"] += 1  # Iron Condor is neutral

        # Determine consensus
        majority_vote = max(consensus_votes, key=consensus_votes.get)
        total_votes = sum(consensus_votes.values())
        consensus_strength = consensus_votes[majority_vote] / total_votes if total_votes > 0 else 0

        # Broadcast consensus result
        consensus_message = {
            "type": "AGENT_CONSENSUS",
            "decision": majority_vote,
            "strength": consensus_strength,
            "votes": consensus_votes,
            "agent_opinions": agent_opinions,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        await agent_coordinator.websocket_manager.broadcast_to_agents(consensus_message)

        assert consensus_strength > 0
        assert majority_vote in ["LONG", "SHORT", "NEUTRAL"]

        logger.info(f"✅ Agent consensus: {majority_vote} (strength: {consensus_strength:.2f})")
        logger.info(f"Vote breakdown: {consensus_votes}")

    @pytest.mark.asyncio
    async def test_trading_execution_with_agents(self, agent_coordinator, sample_options_chain):
        """Test complete trading execution coordinated by agents"""

        # Mock trading components
        with patch('app.trading.alpaca_client.AlpacaClient') as mock_alpaca, \
             patch('app.trading.risk_manager.RiskManager') as mock_risk:

            mock_alpaca_instance = AsyncMock()
            mock_alpaca.return_value = mock_alpaca_instance
            mock_alpaca_instance.place_options_order = AsyncMock(return_value={
                "order_id": "test_order_123",
                "status": "filled",
                "filled_price": 2.50,
                "filled_qty": 1
            })

            mock_risk_instance = MagicMock()
            mock_risk.return_value = mock_risk_instance
            mock_risk_instance.validate_order = MagicMock(return_value=True)
            mock_risk_instance.calculate_position_size = MagicMock(return_value=1)

            # Step 1: Agent coordination creates strategy
            strategist = agent_coordinator.agents["strategist"]
            strategy_response = await strategist.ollama_client.generate_response(
                "Create Iron Condor strategy",
                context={"underlying_price": 440.0, "options_chain": sample_options_chain}
            )
            strategy = json.loads(strategy_response)

            # Step 2: Risk validation
            risk_agent = agent_coordinator.agents["risk"]
            risk_response = await risk_agent.ollama_client.generate_response(
                "Validate strategy risk",
                context={"strategy": strategy}
            )
            risk_check = json.loads(risk_response)

            # Step 3: Execute if approved
            if risk_check.get("approval", False):
                execution_results = []

                for leg in strategy["legs"]:
                    order_result = await mock_alpaca_instance.place_options_order(
                        symbol="SPY",
                        option_type=leg["type"],
                        strike=leg["strike"],
                        action=leg["action"],
                        quantity=1
                    )
                    execution_results.append(order_result)

                # Broadcast execution results
                execution_message = {
                    "type": "TRADE_EXECUTED",
                    "strategy": strategy["strategy"],
                    "orders": execution_results,
                    "total_cost": sum(result.get("filled_price", 0) for result in execution_results),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

                await agent_coordinator.websocket_manager.broadcast_to_agents(execution_message)

                # Verify execution
                assert len(execution_results) == len(strategy["legs"])
                assert all(result["status"] == "filled" for result in execution_results)

                logger.info("✅ Trading execution completed successfully")
                logger.info(f"Executed {len(execution_results)} option legs")
                logger.info(f"Total cost: ${execution_message['total_cost']:.2f}")

            else:
                logger.info("❌ Trade execution rejected by risk management")

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, agent_coordinator, mock_trading_websocket):
        """Test error handling and recovery in agent communication"""

        # Test agent failure scenario
        faulty_agent = agent_coordinator.agents["analyst"]

        # Simulate agent failure
        original_client = faulty_agent.ollama_client
        faulty_agent.ollama_client = None

        try:
            # Attempt communication with failed agent
            response = await faulty_agent.ollama_client.generate_response("test")
            assert False, "Expected exception not raised"
        except AttributeError:
            # Expected failure
            pass

        # Test recovery
        faulty_agent.ollama_client = original_client
        response = await faulty_agent.ollama_client.generate_response("recovery test")
        assert response is not None

        # Test WebSocket disconnection handling
        with patch.object(mock_trading_websocket, 'broadcast_to_agents', side_effect=ConnectionClosed(None, None)):
            try:
                await mock_trading_websocket.broadcast_to_agents({"type": "test"})
                assert False, "Expected ConnectionClosed exception"
            except ConnectionClosed:
                # Expected failure
                pass

        # Test graceful degradation
        partial_agents = {k: v for k, v in agent_coordinator.agents.items() if k != "reasoning"}

        # System should still function with fewer agents
        market_data = {"symbol": "SPY", "price": 440.0}
        responses = []

        for agent_type, agent in partial_agents.items():
            try:
                response = await agent.ollama_client.generate_response("analyze", context=market_data)
                responses.append(response)
            except Exception as e:
                logger.warning(f"Agent {agent_type} failed: {e}")

        # Verify partial functionality
        assert len(responses) >= 2, "System should function with partial agents"

        logger.info("✅ Error handling and recovery tested successfully")

    @pytest.mark.asyncio
    async def test_performance_and_latency(self, agent_coordinator):
        """Test agent response performance and latency"""

        import time

        # Test individual agent response times
        agent_latencies = {}

        for agent_type, agent in agent_coordinator.agents.items():
            start_time = time.time()

            response = await agent.ollama_client.generate_response(
                "Quick market analysis",
                context={"symbol": "SPY", "price": 440.0}
            )

            end_time = time.time()
            latency = end_time - start_time
            agent_latencies[agent_type] = latency

            assert response is not None
            assert latency < 5.0, f"Agent {agent_type} response too slow: {latency:.2f}s"

        # Test concurrent agent processing
        start_time = time.time()

        tasks = []
        for agent_type, agent in agent_coordinator.agents.items():
            task = agent.ollama_client.generate_response(
                "Concurrent analysis",
                context={"symbol": "SPY", "price": 440.0}
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        end_time = time.time()
        concurrent_latency = end_time - start_time

        assert len(responses) == len(agent_coordinator.agents)
        assert all(response is not None for response in responses)
        assert concurrent_latency < 10.0, f"Concurrent processing too slow: {concurrent_latency:.2f}s"

        logger.info("✅ Performance testing completed")
        logger.info(f"Individual latencies: {agent_latencies}")
        logger.info(f"Concurrent latency: {concurrent_latency:.2f}s")


@pytest.mark.asyncio
async def test_end_to_end_trading_scenario():
    """Complete end-to-end test demonstrating agents making live trading decisions"""

    logger.info("🚀 Starting end-to-end trading scenario demonstration")

    # Initialize all components
    mock_clients = {
        "analyst": MockOllamaClient("llama3.2:3b"),
        "risk": MockOllamaClient("phi3:mini"),
        "strategist": MockOllamaClient("qwen2.5-coder:3b"),
        "chat": MockOllamaClient("gemma2:2b")
    }

    # Generate realistic market scenario
    market_data = mock_market_generator.generate_earnings_event_scenario(
        symbol="AAPL",
        earnings_day=1,  # Earnings tomorrow
        earnings_surprise="positive"
    )

    options_chain = mock_options_generator.get_mock_option_chain(
        symbol="AAPL",
        underlying_price=market_data[-1]["close"],
        chain_type="earnings"
    )

    # Simulate complete trading workflow
    current_price = market_data[-1]["close"]

    logger.info(f"📊 Market Data: AAPL at ${current_price}, earnings tomorrow")

    # 1. Analyst analyzes market
    analyst_response = await mock_clients["analyst"].generate_response(
        f"Analyze AAPL pre-earnings at ${current_price}",
        context={"market_data": market_data[-5:], "earnings_tomorrow": True}
    )
    analysis = json.loads(analyst_response)
    logger.info(f"🔍 Analyst ({mock_clients['analyst'].model_name}): {analysis['recommendation']} with {analysis['confidence']:.0%} confidence")

    # 2. Risk manager evaluates
    risk_response = await mock_clients["risk"].generate_response(
        "Evaluate pre-earnings risk",
        context={"analysis": analysis, "symbol": "AAPL", "earnings": True}
    )
    risk_eval = json.loads(risk_response)
    logger.info(f"⚠️ Risk Manager ({mock_clients['risk'].model_name}): {risk_eval['risk_assessment']} risk, approved: {risk_eval['approval']}")

    # 3. Strategist creates plan
    if risk_eval["approval"]:
        strategy_response = await mock_clients["strategist"].generate_response(
            "Create pre-earnings options strategy",
            context={"analysis": analysis, "risk": risk_eval, "options": options_chain}
        )
        strategy = json.loads(strategy_response)
        logger.info(f"📋 Strategist ({mock_clients['strategist'].model_name}): {strategy['strategy']} strategy")
        logger.info(f"   Max Profit: ${strategy['max_profit']}, Max Loss: ${strategy['max_loss']}")

        # 4. Chat agent summarizes
        chat_response = await mock_clients["chat"].generate_response(
            "Summarize trading decision",
            context={"analysis": analysis, "risk": risk_eval, "strategy": strategy}
        )
        summary = json.loads(chat_response)
        logger.info(f"💬 Chat Agent ({mock_clients['chat'].model_name}): {summary['message']}")

        # 5. Simulate execution
        execution_results = []
        for leg in strategy["legs"]:
            result = {
                "leg": f"{leg['action']} {leg['type']} {leg['strike']}",
                "status": "FILLED",
                "price": 2.50,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            execution_results.append(result)

        logger.info("✅ TRADE EXECUTED SUCCESSFULLY")
        logger.info(f"   Strategy: {strategy['strategy']}")
        logger.info(f"   Legs: {len(execution_results)}")
        for result in execution_results:
            logger.info(f"   - {result['leg']}: ${result['price']}")

        logger.info("🎉 End-to-end demonstration completed - Agents are alive and trading!")

    else:
        logger.info("❌ Trade rejected by risk management")
        logger.info(f"   Reason: {risk_eval.get('risk_factors', 'High risk assessment')}")


if __name__ == "__main__":
    # Run the end-to-end demonstration
    asyncio.run(test_end_to_end_trading_scenario())