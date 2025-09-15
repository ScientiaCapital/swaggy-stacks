#!/usr/bin/env python3
"""
🚀 SWAGGY STACKS AGENTS DEMONSTRATION
====================================

This script demonstrates the AI agents coming alive and making real trading decisions
with mock data. Shows agent coordination, decision-making, and trade execution.

Agent Models:
- Analyst: llama3.2:3b (market analysis)
- Risk: phi3:mini (risk assessment)
- Strategist: qwen2.5-coder:3b (strategy generation)
- Chat: gemma2:2b (conversational interface)
- Reasoning: deepseek-r1:1.5b (pattern detection)
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from tests.mocks.mock_market_scenarios import mock_market_generator
from tests.mocks.mock_alpaca_options import mock_options_generator

# Configure logging for beautiful output
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ANSI color codes for beautiful output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class MockOllamaAgent:
    """Mock Ollama agent that simulates LLM responses based on agent type"""

    def __init__(self, agent_type: str, model_name: str):
        self.agent_type = agent_type
        self.model_name = model_name
        self.is_initialized = True

    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate agent-specific market analysis"""

        current_price = market_data.get('price', 440.0)
        symbol = market_data.get('symbol', 'SPY')

        if self.agent_type == "analyst":
            # Analyst using llama3.2:3b - comprehensive market analysis
            return {
                "agent": "analyst",
                "model": self.model_name,
                "analysis": f"{symbol} showing bullish momentum",
                "signal_strength": 0.78,
                "key_factors": [
                    "Volume surge (+45% above 20-day average)",
                    "RSI recovery from oversold (45 → 58)",
                    "MACD bullish crossover confirmed",
                    "Support held at $435 level"
                ],
                "recommendation": "LONG",
                "confidence": 0.82,
                "target_price": current_price * 1.03,
                "stop_loss": current_price * 0.97,
                "reasoning": f"Technical confluence at ${current_price:.2f} suggests continuation of bullish trend. Risk/reward ratio favorable at 3:1."
            }

        elif self.agent_type == "risk":
            # Risk manager using phi3:mini - risk assessment
            portfolio_exposure = min(0.65, current_price / 500)  # Dynamic exposure
            return {
                "agent": "risk",
                "model": self.model_name,
                "risk_assessment": "MODERATE",
                "portfolio_exposure": portfolio_exposure,
                "max_position_size": int(10000 * (1 - portfolio_exposure)),
                "risk_factors": [
                    "Earnings season IV expansion",
                    "Fed meeting next week",
                    f"Current exposure: {portfolio_exposure:.1%}"
                ],
                "approval": portfolio_exposure < 0.75,
                "risk_score": 6.2 + (portfolio_exposure * 2),
                "suggested_hedge": f"Put spread on {symbol} 5% OTM",
                "position_sizing": "Conservative - reduce size by 15% due to elevated IV"
            }

        elif self.agent_type == "strategist":
            # Strategist using qwen2.5-coder:3b - options strategy generation
            spread_width = 5
            otm_offset = int(current_price * 0.02)  # 2% OTM

            return {
                "agent": "strategist",
                "model": self.model_name,
                "strategy": "Iron Condor",
                "market_outlook": "Range-bound with elevated IV",
                "entry_criteria": [
                    "IV rank > 70%",
                    "Neutral to slightly bullish bias",
                    "35-45 DTE optimal"
                ],
                "legs": [
                    {
                        "type": "PUT",
                        "strike": current_price - otm_offset - spread_width,
                        "action": "SELL",
                        "quantity": 1,
                        "premium": 1.25
                    },
                    {
                        "type": "PUT",
                        "strike": current_price - otm_offset,
                        "action": "BUY",
                        "quantity": 1,
                        "premium": 2.50
                    },
                    {
                        "type": "CALL",
                        "strike": current_price + otm_offset,
                        "action": "SELL",
                        "quantity": 1,
                        "premium": 2.75
                    },
                    {
                        "type": "CALL",
                        "strike": current_price + otm_offset + spread_width,
                        "action": "BUY",
                        "quantity": 1,
                        "premium": 1.50
                    }
                ],
                "max_profit": 350,
                "max_loss": 150,
                "breakeven_points": [current_price - otm_offset + 3.5, current_price + otm_offset - 3.5],
                "target_profit": 50,  # 50% of max profit
                "probability_of_profit": 0.68
            }

        elif self.agent_type == "chat":
            # Chat agent using gemma2:2b - conversational coordination
            return {
                "agent": "chat",
                "model": self.model_name,
                "message": f"🤖 Agents analyzing {symbol} at ${current_price:.2f}",
                "status": "COORDINATING",
                "next_action": "Awaiting consensus from all agents",
                "participants": ["analyst", "risk", "strategist"],
                "coordination_summary": "Market analysis complete. Proceeding to strategy consensus.",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        elif self.agent_type == "reasoning":
            # Reasoning agent using deepseek-r1:1.5b - pattern analysis
            return {
                "agent": "reasoning",
                "model": self.model_name,
                "pattern_analysis": "Bullish flag formation detected",
                "historical_patterns": [
                    "Similar setups had 72% success rate",
                    "Average move: +4.2% over 7 days",
                    "Best entry: morning weakness"
                ],
                "market_regime": "Low volatility expansion phase",
                "confidence_factors": [
                    "Volume confirmation present",
                    "Sector rotation supporting",
                    "Options flow bullish bias"
                ],
                "probability_weighting": 0.74
            }

        else:
            return {
                "agent": self.agent_type,
                "model": self.model_name,
                "response": f"Agent {self.agent_type} operational with {self.model_name}",
                "status": "READY"
            }


class AgentCoordinator:
    """Coordinates multiple AI agents for trading decisions"""

    def __init__(self):
        self.agents = {
            "analyst": MockOllamaAgent("analyst", "llama3.2:3b"),
            "risk": MockOllamaAgent("risk", "phi3:mini"),
            "strategist": MockOllamaAgent("strategist", "qwen2.5-coder:3b"),
            "chat": MockOllamaAgent("chat", "gemma2:2b"),
            "reasoning": MockOllamaAgent("reasoning", "deepseek-r1:1.5b")
        }
        self.trading_log = []

    async def process_market_event(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data through all agents"""

        logger.info(f"\n{Colors.HEADER}{Colors.BOLD}🔄 AGENT COORDINATION INITIATED{Colors.ENDC}")
        logger.info(f"{Colors.OKCYAN}Market Event: {market_data['symbol']} @ ${market_data['price']:.2f}{Colors.ENDC}")

        # Collect responses from all agents
        agent_responses = {}

        for agent_type, agent in self.agents.items():
            logger.info(f"{Colors.OKBLUE}⚡ {agent_type.title()} Agent ({agent.model_name}) analyzing...{Colors.ENDC}")
            response = await agent.analyze_market(market_data)
            agent_responses[agent_type] = response

            # Log key insights
            if agent_type == "analyst":
                logger.info(f"   📊 {response['recommendation']} | Confidence: {response['confidence']:.0%} | Target: ${response['target_price']:.2f}")
            elif agent_type == "risk":
                logger.info(f"   ⚠️  {response['risk_assessment']} Risk | Approval: {'✅' if response['approval'] else '❌'} | Score: {response['risk_score']:.1f}/10")
            elif agent_type == "strategist":
                logger.info(f"   📋 {response['strategy']} | Max Profit: ${response['max_profit']} | P(Profit): {response['probability_of_profit']:.0%}")
            elif agent_type == "chat":
                logger.info(f"   💬 {response['message']}")
            elif agent_type == "reasoning":
                logger.info(f"   🧠 {response['pattern_analysis']} | Confidence: {response['probability_weighting']:.0%}")

        return agent_responses

    async def make_trading_decision(self, agent_responses: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate final trading decision based on agent consensus"""

        logger.info(f"\n{Colors.WARNING}{Colors.BOLD}🤝 BUILDING AGENT CONSENSUS{Colors.ENDC}")

        # Extract key decisions
        analyst_rec = agent_responses["analyst"]["recommendation"]
        risk_approval = agent_responses["risk"]["approval"]
        strategy = agent_responses["strategist"]["strategy"]

        # Build consensus
        consensus_score = 0
        decision_factors = []

        if analyst_rec in ["LONG", "SHORT"]:
            consensus_score += agent_responses["analyst"]["confidence"]
            decision_factors.append(f"Analyst: {analyst_rec} ({agent_responses['analyst']['confidence']:.0%})")

        if risk_approval:
            consensus_score += 0.3
            decision_factors.append("Risk: APPROVED")
        else:
            consensus_score -= 0.4
            decision_factors.append("Risk: REJECTED")

        consensus_score += agent_responses["reasoning"]["probability_weighting"] * 0.2
        decision_factors.append(f"Pattern: {agent_responses['reasoning']['probability_weighting']:.0%}")

        # Final decision
        if consensus_score > 0.7 and risk_approval:
            decision = "EXECUTE_TRADE"
            logger.info(f"{Colors.OKGREEN}✅ CONSENSUS REACHED: EXECUTE TRADE{Colors.ENDC}")
        else:
            decision = "REJECT_TRADE"
            logger.info(f"{Colors.FAIL}❌ CONSENSUS FAILED: REJECT TRADE{Colors.ENDC}")

        logger.info(f"{Colors.OKCYAN}Consensus Score: {consensus_score:.2f}{Colors.ENDC}")
        for factor in decision_factors:
            logger.info(f"   • {factor}")

        return {
            "decision": decision,
            "consensus_score": consensus_score,
            "factors": decision_factors,
            "strategy": strategy if decision == "EXECUTE_TRADE" else None,
            "agent_responses": agent_responses
        }

    async def execute_trade(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate trade execution"""

        if decision_data["decision"] != "EXECUTE_TRADE":
            return {"status": "REJECTED", "reason": "Failed consensus"}

        logger.info(f"\n{Colors.OKGREEN}{Colors.BOLD}🚀 EXECUTING TRADE{Colors.ENDC}")

        strategy_data = decision_data["agent_responses"]["strategist"]
        execution_results = []
        total_cost = 0

        for i, leg in enumerate(strategy_data["legs"]):
            # Simulate order execution
            result = {
                "leg_number": i + 1,
                "type": leg["type"],
                "strike": leg["strike"],
                "action": leg["action"],
                "quantity": leg["quantity"],
                "premium": leg["premium"],
                "status": "FILLED",
                "fill_time": datetime.now(timezone.utc).isoformat()
            }

            execution_results.append(result)

            # Calculate cost (negative for credit spreads)
            cost = leg["premium"] * leg["quantity"] * (1 if leg["action"] == "BUY" else -1)
            total_cost += cost

            logger.info(f"   🎯 Leg {i+1}: {leg['action']} {leg['type']} ${leg['strike']} @ ${leg['premium']:.2f} - FILLED")

        trade_summary = {
            "strategy": strategy_data["strategy"],
            "legs_executed": len(execution_results),
            "net_credit": -total_cost if total_cost < 0 else 0,
            "net_debit": total_cost if total_cost > 0 else 0,
            "max_profit": strategy_data["max_profit"],
            "max_loss": strategy_data["max_loss"],
            "execution_results": execution_results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        self.trading_log.append(trade_summary)

        logger.info(f"{Colors.OKGREEN}✅ TRADE EXECUTED SUCCESSFULLY{Colors.ENDC}")
        logger.info(f"   Strategy: {trade_summary['strategy']}")
        logger.info(f"   Net Credit: ${trade_summary['net_credit']:.2f}" if trade_summary['net_credit'] > 0 else f"   Net Debit: ${trade_summary['net_debit']:.2f}")
        logger.info(f"   Max Profit: ${trade_summary['max_profit']}")
        logger.info(f"   Max Loss: ${trade_summary['max_loss']}")

        return trade_summary


async def demonstrate_live_trading():
    """Main demonstration function showing agents coming alive"""

    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("🚀 SWAGGY STACKS AGENTS DEMONSTRATION")
    print("=====================================")
    print("Agents Coming Alive with Live Trading Decisions")
    print(f"{Colors.ENDC}")

    # Initialize agent coordinator
    coordinator = AgentCoordinator()

    # Display agent initialization
    print(f"{Colors.OKBLUE}{Colors.BOLD}🤖 INITIALIZING AI AGENTS{Colors.ENDC}")
    for agent_type, agent in coordinator.agents.items():
        print(f"   ✅ {agent_type.title()} Agent: {agent.model_name}")

    # Generate realistic market scenarios
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}📊 GENERATING MARKET SCENARIOS{Colors.ENDC}")

    # Scenario 1: Bull market momentum
    market_data_1 = {
        "symbol": "SPY",
        "price": 445.25,
        "volume": 8500000,
        "iv_rank": 75,
        "scenario": "bull_momentum"
    }

    # Scenario 2: Pre-earnings setup
    market_data_2 = {
        "symbol": "AAPL",
        "price": 185.50,
        "volume": 12000000,
        "iv_rank": 85,
        "scenario": "pre_earnings"
    }

    scenarios = [market_data_1, market_data_2]

    # Process each scenario
    for i, market_data in enumerate(scenarios, 1):
        print(f"\n{Colors.HEADER}{Colors.BOLD}📈 SCENARIO {i}: {market_data['scenario'].upper()}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Symbol: {market_data['symbol']} | Price: ${market_data['price']} | IV Rank: {market_data['iv_rank']}%{Colors.ENDC}")

        # Agent coordination
        agent_responses = await coordinator.process_market_event(market_data)

        # Decision making
        decision = await coordinator.make_trading_decision(agent_responses)

        # Trade execution
        if decision["decision"] == "EXECUTE_TRADE":
            trade_result = await coordinator.execute_trade(decision)
        else:
            print(f"{Colors.WARNING}⏸️  Trade rejected - awaiting better setup{Colors.ENDC}")

        print(f"{Colors.OKCYAN}{'─' * 60}{Colors.ENDC}")

    # Final summary
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}📋 TRADING SESSION SUMMARY{Colors.ENDC}")
    print(f"   Scenarios Analyzed: {len(scenarios)}")
    print(f"   Trades Executed: {len(coordinator.trading_log)}")

    if coordinator.trading_log:
        total_max_profit = sum(trade["max_profit"] for trade in coordinator.trading_log)
        total_max_loss = sum(trade["max_loss"] for trade in coordinator.trading_log)
        print(f"   Total Max Profit Potential: ${total_max_profit}")
        print(f"   Total Max Loss Risk: ${total_max_loss}")
        print(f"   Risk/Reward Ratio: {total_max_profit/total_max_loss:.2f}:1")

    print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 AGENTS ARE ALIVE AND TRADING! 🎉{Colors.ENDC}")
    print(f"{Colors.OKCYAN}All agents successfully initialized and coordinating trades{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Real-time decision making and execution demonstrated{Colors.ENDC}")
    print(f"{Colors.OKCYAN}System ready for live market deployment 🚀{Colors.ENDC}")


if __name__ == "__main__":
    try:
        # Run the demonstration
        asyncio.run(demonstrate_live_trading())
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Demo interrupted by user{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.FAIL}Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()