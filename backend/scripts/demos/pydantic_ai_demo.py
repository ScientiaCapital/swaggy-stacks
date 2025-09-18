#!/usr/bin/env python3
"""
PydanticAI Trading Agents Demo
=============================

Demonstration script showcasing the new type-safe, validated
PydanticAI trading agents with comprehensive error handling
and performance monitoring.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any

import structlog

# Import our new PydanticAI agents
from app.ai.pydantic_trading_agents import (
    PydanticTradingCoordinator,
    PydanticMarketAnalyst,
    PydanticRiskAdvisor,
    PydanticStrategyOptimizer,
)
from app.ai.pydantic_base_agent import AgentContext

logger = structlog.get_logger(__name__)


class PydanticAIDemo:
    """
    Demonstration of PydanticAI trading agents with comprehensive
    type safety, validation, and performance monitoring
    """

    def __init__(self):
        """Initialize demo with PydanticAI coordinator"""
        self.coordinator = PydanticTradingCoordinator(
            model_name="claude-3-5-sonnet-20241022"
        )

        # Demo symbols for testing
        self.demo_symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]

        # Mock current positions for risk assessment
        self.current_positions = [
            {"symbol": "SPY", "size": 15000, "entry_price": 400.0},
            {"symbol": "QQQ", "size": 8000, "entry_price": 350.0},
        ]

        logger.info("PydanticAI Demo initialized",
                   symbols=len(self.demo_symbols),
                   positions=len(self.current_positions))

    async def demo_individual_agents(self):
        """Demonstrate individual PydanticAI agents"""
        print("\n🤖 PydanticAI Individual Agents Demo")
        print("=" * 50)

        symbol = "AAPL"

        # Demo Market Analyst
        print(f"\n📊 Market Analyst Analysis for {symbol}")
        print("-" * 30)

        try:
            market_result = await self.coordinator.market_analyst.analyze_market(
                symbol=symbol,
                context={"market_regime": "trending", "volatility": "normal"},
                correlation_id=f"demo_market_{int(time.time())}",
            )

            print(f"✅ Market Analysis Completed:")
            print(f"   Sentiment: {market_result.data.sentiment.value}")
            print(f"   Confidence: {market_result.confidence:.2f}")
            print(f"   Risk Level: {market_result.data.risk_level.value}")
            print(f"   Execution Time: {market_result.execution_time_ms:.1f}ms")
            print(f"   Key Factors: {', '.join(market_result.data.key_factors[:3])}")
            print(f"   Technical Score: {market_result.data.technical_score:.1f}/10")

        except Exception as e:
            print(f"❌ Market Analysis Failed: {e}")

        # Demo Risk Advisor
        print(f"\n🛡️ Risk Advisor Assessment for {symbol}")
        print("-" * 30)

        try:
            risk_result = await self.coordinator.risk_advisor.assess_risk(
                symbol=symbol,
                position_size=10000.0,
                account_value=100000.0,
                current_positions=self.current_positions,
                correlation_id=f"demo_risk_{int(time.time())}",
            )

            print(f"✅ Risk Assessment Completed:")
            print(f"   Risk Level: {risk_result.data.risk_level.value}")
            print(f"   Portfolio Heat: {risk_result.data.portfolio_heat:.2f}")
            print(f"   Recommended Size: ${risk_result.data.recommended_position_size:,.2f}")
            print(f"   Stop Loss: {risk_result.data.stop_loss_percentage:.1%}")
            print(f"   Execution Time: {risk_result.execution_time_ms:.1f}ms")
            print(f"   Diversification Score: {risk_result.data.diversification_score:.1f}/10")

        except Exception as e:
            print(f"❌ Risk Assessment Failed: {e}")

        # Demo Strategy Optimizer
        print(f"\n🎯 Strategy Optimizer Signal for {symbol}")
        print("-" * 30)

        try:
            strategy_result = await self.coordinator.strategy_optimizer.generate_signal(
                symbol=symbol,
                market_context={
                    "regime": "trending",
                    "volatility": "normal",
                    "trend_strength": "strong"
                },
                correlation_id=f"demo_strategy_{int(time.time())}",
            )

            print(f"✅ Strategy Signal Generated:")
            print(f"   Action: {strategy_result.data.action.value}")
            print(f"   Confidence: {strategy_result.confidence:.2f}")
            print(f"   Position Size: ${strategy_result.data.position_size:,.2f}")
            print(f"   Time Horizon: {strategy_result.data.time_horizon}")
            print(f"   Execution Time: {strategy_result.execution_time_ms:.1f}ms")
            if strategy_result.data.entry_price:
                print(f"   Entry Price: ${strategy_result.data.entry_price:.2f}")
            print(f"   Risk/Reward: {strategy_result.data.risk_reward_ratio:.2f}")

        except Exception as e:
            print(f"❌ Strategy Signal Failed: {e}")

    async def demo_comprehensive_analysis(self):
        """Demonstrate comprehensive coordinated analysis"""
        print("\n🔄 PydanticAI Comprehensive Analysis Demo")
        print("=" * 50)

        for symbol in self.demo_symbols[:3]:  # Test first 3 symbols
            print(f"\n📈 Comprehensive Analysis: {symbol}")
            print("-" * 40)

            start_time = time.time()

            try:
                result = await self.coordinator.comprehensive_analysis(
                    symbol=symbol,
                    position_size=12000.0,
                    account_value=150000.0,
                    current_positions=self.current_positions,
                    market_context={
                        "market_session": "regular",
                        "economic_events": "none",
                        "sector_rotation": "tech_momentum"
                    },
                    correlation_id=f"demo_comprehensive_{symbol}_{int(time.time())}",
                )

                total_time = time.time() - start_time

                print(f"✅ Analysis Completed in {total_time:.2f}s")
                print(f"   Final Recommendation: {result['final_recommendation']}")
                print(f"   Total Execution Time: {result['execution_time_ms']:.1f}ms")
                print(f"   Agent Performance:")
                print(f"     Market Analyst: {result['market_analysis']['confidence']:.2f} confidence")
                print(f"     Risk Advisor: {result['risk_assessment']['result']['risk_level']}")
                print(f"     Strategy Optimizer: {result['strategy_signal']['result']['action']}")

                # Show coordinated decision making
                market_sentiment = result['market_analysis']['result']['sentiment']
                risk_level = result['risk_assessment']['result']['risk_level']
                strategy_action = result['strategy_signal']['result']['action']

                print(f"   Coordination Summary:")
                print(f"     Market: {market_sentiment} | Risk: {risk_level} | Signal: {strategy_action}")
                print(f"     → Final Decision: {result['final_recommendation']}")

            except Exception as e:
                print(f"❌ Comprehensive Analysis Failed: {e}")

            # Brief pause between analyses
            await asyncio.sleep(0.5)

    async def demo_performance_monitoring(self):
        """Demonstrate performance monitoring and health checks"""
        print("\n📊 PydanticAI Performance Monitoring Demo")
        print("=" * 50)

        # Run health check on all agents
        print("\n🏥 Agent Health Check")
        print("-" * 25)

        try:
            health = await self.coordinator.health_check()

            print(f"✅ Overall System Status: {health['status']}")
            print(f"   Model: {health['model']}")
            print(f"   Timestamp: {health['timestamp']}")
            print(f"\n   Individual Agent Health:")

            for agent_name, agent_health in health['agents'].items():
                status_emoji = "✅" if agent_health['status'] == 'healthy' else "⚠️"
                print(f"   {status_emoji} {agent_name.replace('_', ' ').title()}: {agent_health['status']}")

                if 'stats' in agent_health:
                    stats = agent_health['stats']
                    print(f"      Executions: {stats['total_executions']}")
                    print(f"      Error Rate: {stats['error_rate']:.1%}")
                    print(f"      Avg Time: {stats['average_execution_time_ms']:.1f}ms")
                    print(f"      Avg Confidence: {stats['average_confidence']:.2f}")

        except Exception as e:
            print(f"❌ Health Check Failed: {e}")

        # Demonstrate individual agent stats
        print("\n📈 Detailed Agent Statistics")
        print("-" * 30)

        agents = {
            "Market Analyst": self.coordinator.market_analyst,
            "Risk Advisor": self.coordinator.risk_advisor,
            "Strategy Optimizer": self.coordinator.strategy_optimizer,
        }

        for agent_name, agent in agents.items():
            stats = agent.get_stats()
            print(f"\n{agent_name}:")
            print(f"   Total Executions: {stats.total_executions}")
            print(f"   Success Rate: {((stats.successful_executions/max(stats.total_executions,1))*100):.1f}%")
            print(f"   Average Execution Time: {stats.average_execution_time_ms:.1f}ms")
            print(f"   Average Confidence: {stats.average_confidence:.2f}")
            if stats.last_execution:
                print(f"   Last Execution: {stats.last_execution.strftime('%H:%M:%S')}")

    async def demo_type_safety_validation(self):
        """Demonstrate type safety and validation features"""
        print("\n🔒 PydanticAI Type Safety & Validation Demo")
        print("=" * 50)

        print("\n✅ Valid Agent Context Creation")
        print("-" * 35)

        try:
            valid_context = AgentContext(
                agent_id="demo_agent_001",
                symbol="AAPL",
                risk_tolerance=0.05,  # Valid: 0-1 range
                max_position_size=25000.0,  # Valid: positive
            )
            print(f"   Agent ID: {valid_context.agent_id}")
            print(f"   Symbol: {valid_context.symbol}")
            print(f"   Risk Tolerance: {valid_context.risk_tolerance}")
            print(f"   Max Position: ${valid_context.max_position_size:,.2f}")
            print(f"   Timestamp: {valid_context.timestamp}")

        except Exception as e:
            print(f"   ❌ Validation Failed: {e}")

        print("\n❌ Invalid Agent Context Examples")
        print("-" * 35)

        # Test invalid risk tolerance
        try:
            invalid_context = AgentContext(
                agent_id="demo_agent_002",
                symbol="TSLA",
                risk_tolerance=1.5,  # Invalid: > 1.0
                max_position_size=10000.0,
            )
        except ValueError as e:
            print(f"   ✅ Caught Invalid Risk Tolerance: {e}")

        # Test invalid position size
        try:
            invalid_context = AgentContext(
                agent_id="demo_agent_003",
                symbol="MSFT",
                risk_tolerance=0.03,
                max_position_size=-5000.0,  # Invalid: negative
            )
        except ValueError as e:
            print(f"   ✅ Caught Invalid Position Size: {e}")

    async def run_full_demo(self):
        """Run complete PydanticAI demonstration"""
        print("🚀 PydanticAI Trading Agents - Full Demonstration")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Run all demo sections
            await self.demo_type_safety_validation()
            await self.demo_individual_agents()
            await self.demo_comprehensive_analysis()
            await self.demo_performance_monitoring()

            print("\n🎉 PydanticAI Demo Completed Successfully!")
            print("=" * 60)

        except Exception as e:
            logger.error("Demo failed", error=str(e))
            print(f"\n❌ Demo Failed: {e}")

        finally:
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


async def main():
    """Main entry point for PydanticAI demo"""
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    print("Initializing PydanticAI Demo...")
    demo = PydanticAIDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    asyncio.run(main())