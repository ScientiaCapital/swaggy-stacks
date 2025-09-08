#!/usr/bin/env python3
"""
Real-Time Agent System Demo
Demonstrates the complete event-driven AI agent architecture

Run this script to see the agent system in action:
python demo_agent_system.py
"""

import asyncio
import json
from datetime import datetime
from app.agent_system import agent_system
from app.testing.mock_data_generator import MarketRegime

async def demo_real_time_analysis():
    """Demo real-time analysis capabilities"""
    print("\n🔍 Demo: Real-Time Analysis")
    print("="*50)
    
    # Sample market data
    market_data = {
        "symbol": "AAPL",
        "price": 150.25,
        "volume": 2500000,
        "change": 2.15,
        "change_percent": 1.45,
        "timestamp": datetime.now().isoformat()
    }
    
    technical_indicators = {
        "rsi": 68.5,
        "macd": 1.85,
        "macd_signal": 1.2,
        "bb_upper": 155.0,
        "bb_lower": 145.0,
        "sma_20": 149.8,
        "ema_12": 150.1,
        "atr": 3.2
    }
    
    account_info = {
        "equity": 100000,
        "buying_power": 80000,
        "cash": 20000
    }
    
    print(f"📊 Analyzing {market_data['symbol']} at ${market_data['price']}")
    print(f"📈 RSI: {technical_indicators['rsi']}, MACD: {technical_indicators['macd']}")
    
    # Run comprehensive analysis
    result = await agent_system.run_real_time_analysis(
        symbol=market_data["symbol"],
        market_data=market_data,
        technical_indicators=technical_indicators,
        account_info=account_info
    )
    
    print(f"🎯 Final Decision: {result['final_recommendation']}")
    print(f"📝 Market Analysis: {result['market_analysis']['sentiment']} (confidence: {result['market_analysis']['confidence']:.2f})")
    print(f"⚠️  Risk Assessment: {result['risk_assessment']['risk_level']} (confidence: {result['risk_assessment']['confidence']:.2f})")
    print(f"🔄 Strategy Signal: {result['strategy_signal']['action']} (confidence: {result['strategy_signal']['confidence']:.2f})")
    
    return result

async def demo_multi_agent_consensus():
    """Demo multi-agent consensus mechanism"""
    print("\n🤝 Demo: Multi-Agent Consensus")
    print("="*50)
    
    context = {
        "market_condition": "trending_bullish",
        "volatility_level": "normal",
        "sector": "technology",
        "earnings_season": True,
        "fed_policy": "neutral"
    }
    
    print("📋 Requesting consensus from agents:")
    print("   • Market Analyst")
    print("   • Risk Advisor") 
    print("   • Strategy Optimizer")
    
    # Request consensus
    consensus_id = await agent_system.request_consensus(
        symbol="TSLA",
        context=context,
        required_agents=["market_analyst", "risk_advisor", "strategy_optimizer"],
        timeout_seconds=10
    )
    
    print(f"🔄 Consensus request submitted: {consensus_id}")
    print("⏳ Waiting for agent responses...")
    
    # Wait a bit for consensus processing
    await asyncio.sleep(12)  # Slightly longer than timeout
    
    # Get consensus result
    consensus_result = await agent_system.get_consensus_result(consensus_id)
    
    if consensus_result:
        print(f"✅ Consensus Result: {consensus_result['final_decision']}")
        print(f"🎯 Confidence: {consensus_result['confidence']:.2f}")
        print(f"👥 Participating Agents: {', '.join(consensus_result['participating_agents'])}")
        print(f"💭 Reasoning: {consensus_result['reasoning']}")
    else:
        print("⏰ Consensus timed out or is still processing")
    
    return consensus_result

async def demo_market_streaming():
    """Demo real-time market streaming simulation"""
    print("\n📡 Demo: Market Streaming Simulation")
    print("="*50)
    
    print("🎭 Simulating trending bullish market for SPY")
    print("📊 Streaming data every 5 seconds for 30 seconds...")
    
    # Start streaming simulation (short duration for demo)
    streaming_task = asyncio.create_task(
        agent_system.simulate_market_stream(
            symbol="SPY",
            regime=MarketRegime.TRENDING_BULLISH,
            duration_minutes=1,  # 1 minute for demo
            interval_seconds=5   # Updates every 5 seconds
        )
    )
    
    # Let it run for a bit
    await asyncio.sleep(15)
    
    # Stop streaming
    agent_system.stop_streaming()
    print("⏹️  Streaming stopped")
    
    # Wait for task to complete
    try:
        await asyncio.wait_for(streaming_task, timeout=5)
    except asyncio.TimeoutError:
        streaming_task.cancel()

async def demo_agent_testing():
    """Demo agent testing capabilities"""
    print("\n🧪 Demo: Agent Testing Suite")
    print("="*50)
    
    print("🔬 Running agent validation tests...")
    print("📋 Testing scenarios: Trending Bullish, High Volatility")
    
    # Run tests on specific regimes
    test_results = await agent_system.run_agent_tests(
        regime_filter=["trending_bullish", "high_volatility"],
        agent_types=["comprehensive", "market_analyst"]
    )
    
    # Display results
    summary = test_results["test_report"]["executive_summary"]
    print(f"✅ Tests Passed: {summary['passed_tests']}")
    print(f"❌ Tests Failed: {summary['failed_tests']}")
    print(f"⚡ Success Rate: {summary['overall_success_rate']:.1%}")
    
    # Show agent performance
    for agent_type, performance in test_results["performance_reports"].items():
        print(f"🤖 {agent_type}: {performance['success_rate']:.1%} success rate, {performance['average_response_time_ms']:.0f}ms avg response")
    
    return test_results

async def demo_system_monitoring():
    """Demo system status and monitoring"""
    print("\n📊 Demo: System Status & Monitoring")
    print("="*50)
    
    status = await agent_system.get_system_status()
    
    print("🖥️  System Status:")
    print(f"   • Initialized: {status['system_status']['initialized']}")
    print(f"   • Streaming Active: {status['system_status']['streaming_active']}")
    print(f"   • Real-time Callbacks: {status['system_status']['real_time_callbacks']}")
    
    print("\n💊 Component Health:")
    for component, health in status['component_health'].items():
        health_status = health.get('status', 'unknown')
        print(f"   • {component}: {health_status}")
    
    print("\n📈 Performance Summary:")
    perf = status['performance_summary']
    print(f"   • Active Agents: {perf['active_agents']}")
    print(f"   • Total Decisions: {perf['total_decisions']}")
    print(f"   • Consensus Processed: {perf['consensus_processed']}")
    print(f"   • Tool Executions: {perf['tool_executions']}")
    
    return status

async def main():
    """Main demo function"""
    print("🚀 Real-Time Event-Driven AI Agent System Demo")
    print("=" * 60)
    
    try:
        # Initialize the system
        print("\n⚡ Initializing Agent System...")
        success = await agent_system.initialize()
        
        if not success:
            print("❌ Failed to initialize agent system")
            return
        
        print("✅ Agent System initialized successfully!")
        
        # Set up real-time callback for demonstration
        async def demo_callback(event_type, data):
            print(f"📡 Real-time event: {event_type} - {data.get('message_type', 'N/A')}")
        
        agent_system.add_real_time_callback(demo_callback)
        
        # Run demonstrations
        await demo_real_time_analysis()
        await asyncio.sleep(1)  # Brief pause between demos
        
        await demo_multi_agent_consensus()
        await asyncio.sleep(1)
        
        await demo_market_streaming()
        await asyncio.sleep(1)
        
        await demo_agent_testing()
        await asyncio.sleep(1)
        
        await demo_system_monitoring()
        
        print("\n🎉 Demo completed successfully!")
        print("\n📝 Summary:")
        print("   ✅ Real-time analysis with streaming decisions")
        print("   ✅ Multi-agent consensus mechanisms")
        print("   ✅ Event-driven coordination via WebSocket & RabbitMQ")
        print("   ✅ Tool execution feedback tracking")
        print("   ✅ Comprehensive testing framework")
        print("   ✅ Market data simulation and streaming")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n🧹 Shutting down system...")
        await agent_system.shutdown()
        print("👋 Demo finished!")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())