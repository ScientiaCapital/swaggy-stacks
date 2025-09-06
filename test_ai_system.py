#!/usr/bin/env python3
"""
Test script for AI Trading System on M1 MacBook with 8GB RAM
"""

import asyncio
import sys
import os
import time
import psutil
from datetime import datetime

# Add backend to path
sys.path.append('/Users/tmkipper/repos/swaggy-stacks/backend')

from app.ai.ollama_client import OllamaClient
from app.ai.trading_agents import AIAgentCoordinator


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


async def test_ollama_health():
    """Test Ollama service health"""
    print("🔍 Testing Ollama service health...")
    
    try:
        client = OllamaClient()
        health = await client.health_check()
        
        print(f"✅ Ollama Status: {health['status']}")
        print(f"   Available models: {health.get('available_models', [])}")
        print(f"   Missing models: {health.get('missing_models', [])}")
        print(f"   Memory usage estimate: {health.get('memory_usage', 0)} MB")
        
        return health['status'] in ['healthy', 'partial']
        
    except Exception as e:
        print(f"❌ Ollama health check failed: {str(e)}")
        return False


async def test_basic_ai_chat():
    """Test basic AI chat functionality"""
    print("\n💬 Testing basic AI chat...")
    
    try:
        client = OllamaClient()
        
        # Simple test message
        response = await client.generate_response(
            prompt="Hello, can you briefly explain what you do?",
            model_key='chat',
            max_tokens=100
        )
        
        print(f"✅ AI Chat Response: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ AI chat test failed: {str(e)}")
        return False


async def test_market_analysis():
    """Test market analysis agent"""
    print("\n📊 Testing market analysis agent...")
    
    try:
        coordinator = AIAgentCoordinator()
        
        # Test market data
        market_data = {
            'current_price': 150.0,
            'volume': 50000000,
            'high_52w': 180.0,
            'low_52w': 120.0,
            'volatility': 0.25
        }
        
        technical_indicators = {
            'rsi': 65.0,
            'ma20': 148.0,
            'ma50': 145.0,
            'macd': 2.5,
            'atr': 3.2
        }
        
        analysis = await coordinator.market_analyst.analyze_market(
            symbol="AAPL",
            market_data=market_data,
            technical_indicators=technical_indicators,
            context="Testing M1 performance"
        )
        
        print(f"✅ Market Analysis Complete:")
        print(f"   Symbol: {analysis.symbol}")
        print(f"   Sentiment: {analysis.sentiment}")
        print(f"   Confidence: {analysis.confidence:.2f}")
        print(f"   Risk Level: {analysis.risk_level}")
        
        return True
        
    except Exception as e:
        print(f"❌ Market analysis test failed: {str(e)}")
        return False


async def test_comprehensive_analysis():
    """Test comprehensive analysis with all agents"""
    print("\n🧠 Testing comprehensive analysis...")
    
    try:
        coordinator = AIAgentCoordinator()
        
        market_data = {
            'current_price': 150.0,
            'volume': 50000000,
            'high_52w': 180.0,
            'low_52w': 120.0,
            'volatility': 0.25
        }
        
        technical_indicators = {
            'rsi': 65.0,
            'ma20': 148.0,
            'ma50': 145.0,
            'macd': 2.5,
            'atr': 3.2
        }
        
        markov_analysis = {
            'current_state': 'bullish',
            'confidence': 0.7,
            'direction': 'up',
            'transition_prob': 0.65
        }
        
        result = await coordinator.comprehensive_analysis(
            symbol="AAPL",
            market_data=market_data,
            technical_indicators=technical_indicators,
            account_info={'equity': 100000, 'cash': 50000},
            current_positions=[],
            markov_analysis=markov_analysis
        )
        
        print(f"✅ Comprehensive Analysis Complete:")
        print(f"   Symbol: {result['symbol']}")
        print(f"   Final Recommendation: {result['final_recommendation']}")
        print(f"   Market Sentiment: {result['market_analysis']['sentiment']}")
        print(f"   Strategy Action: {result['strategy_signal']['action']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive analysis test failed: {str(e)}")
        return False


async def test_memory_usage():
    """Test memory usage under load"""
    print("\n🧠 Testing memory usage...")
    
    initial_memory = get_memory_usage()
    print(f"   Initial memory usage: {initial_memory:.1f} MB")
    
    try:
        coordinator = AIAgentCoordinator()
        
        # Run multiple analyses to test memory usage
        for i in range(3):
            print(f"   Running analysis {i+1}/3...")
            
            market_data = {
                'current_price': 150.0 + i * 5,
                'volume': 50000000,
                'high_52w': 180.0,
                'low_52w': 120.0,
                'volatility': 0.25
            }
            
            technical_indicators = {
                'rsi': 65.0,
                'ma20': 148.0,
                'ma50': 145.0,
                'macd': 2.5,
                'atr': 3.2
            }
            
            await coordinator.market_analyst.analyze_market(
                symbol=f"TEST{i}",
                market_data=market_data,
                technical_indicators=technical_indicators
            )
            
            current_memory = get_memory_usage()
            print(f"   Memory after analysis {i+1}: {current_memory:.1f} MB")
            
            # Check if memory usage is reasonable for M1 8GB
            if current_memory > 4000:  # 4GB threshold
                print(f"⚠️  High memory usage detected: {current_memory:.1f} MB")
        
        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        print(f"   Final memory usage: {final_memory:.1f} MB")
        print(f"   Memory increase: {memory_increase:.1f} MB")
        
        if memory_increase < 500:  # Less than 500MB increase is good
            print("✅ Memory usage is within acceptable limits for M1 8GB")
            return True
        else:
            print("⚠️  Memory usage may be too high for sustained operation")
            return False
        
    except Exception as e:
        print(f"❌ Memory usage test failed: {str(e)}")
        return False


async def run_performance_test():
    """Run performance tests for M1 optimization"""
    print("\n⚡ Running M1 performance tests...")
    
    try:
        coordinator = AIAgentCoordinator()
        
        # Time a comprehensive analysis
        start_time = time.time()
        
        result = await coordinator.comprehensive_analysis(
            symbol="PERF_TEST",
            market_data={
                'current_price': 100.0,
                'volume': 1000000,
                'high_52w': 120.0,
                'low_52w': 80.0,
                'volatility': 0.2
            },
            technical_indicators={
                'rsi': 50.0,
                'ma20': 98.0,
                'ma50': 95.0,
                'macd': 1.0,
                'atr': 2.0
            },
            account_info={'equity': 100000, 'cash': 50000},
            current_positions=[],
            markov_analysis={
                'current_state': 'neutral',
                'confidence': 0.6,
                'direction': 'neutral',
                'transition_prob': 0.5
            }
        )
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        print(f"✅ Performance test completed:")
        print(f"   Analysis time: {analysis_time:.2f} seconds")
        
        if analysis_time < 10:  # Less than 10 seconds is good for M1
            print("✅ Performance is optimal for M1 MacBook")
            return True
        elif analysis_time < 30:
            print("⚠️  Performance is acceptable but could be improved")
            return True
        else:
            print("❌ Performance may be too slow for real-time trading")
            return False
        
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")
        return False


async def main():
    """Run all tests"""
    print("🤖 Swaggy Stacks AI System Test Suite")
    print("=" * 50)
    print(f"🖥️  System: M1 MacBook with 8GB RAM")
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🧠 Initial Memory: {get_memory_usage():.1f} MB")
    print()
    
    tests = [
        ("Ollama Health Check", test_ollama_health),
        ("Basic AI Chat", test_basic_ai_chat),
        ("Market Analysis Agent", test_market_analysis),
        ("Comprehensive Analysis", test_comprehensive_analysis),
        ("Memory Usage Test", test_memory_usage),
        ("Performance Test", run_performance_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"🧪 {test_name}...")
        try:
            success = await test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} crashed: {str(e)}")
            results[test_name] = False
        print()
    
    # Summary
    print("📊 Test Results Summary:")
    print("=" * 30)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🏆 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AI system is ready for M1 MacBook deployment.")
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed. System should work but may need optimization.")
    else:
        print("❌ Multiple tests failed. System needs debugging before deployment.")
    
    print(f"\n🧠 Final Memory Usage: {get_memory_usage():.1f} MB")


if __name__ == "__main__":
    asyncio.run(main())