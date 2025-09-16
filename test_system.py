#!/usr/bin/env python3
"""
Test Real-Time Alpaca Streaming System
=====================================
Simple test to verify system components work
"""

import os
import sys
from pathlib import Path
import asyncio

# Add backend to path for imports
backend_path = Path(__file__).parent / "backend"
sys.path.append(str(backend_path))

async def test_alpaca_connection():
    """Test basic Alpaca connection"""
    try:
        # Import core components
        from app.core.config import settings
        from app.trading.alpaca_stream_manager import AlpacaStreamManager

        print("✅ Core imports successful")

        # Test Alpaca credentials
        if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
            print("❌ Alpaca credentials not configured")
            return False

        print(f"✅ Alpaca credentials configured")
        print(f"   API Key: {settings.ALPACA_API_KEY[:8]}...")
        print(f"   Base URL: {settings.ALPACA_BASE_URL}")

        # Test stream manager creation
        stream_manager = AlpacaStreamManager(
            paper=True,
            data_feed="iex"
        )

        print("✅ Stream manager created successfully")

        # Initialize (but don't connect yet)
        await stream_manager.initialize()
        print("✅ Stream manager initialized")

        # Test connection health (before connecting)
        health = await stream_manager.get_connection_health()
        print(f"✅ Health check: {health}")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def test_consensus_engine():
    """Test consensus engine"""
    try:
        from app.ai.consensus_engine import ConsensusEngine, AgentVote, VoteType

        # Create consensus engine
        consensus_engine = ConsensusEngine()
        print("✅ Consensus engine created")

        # Create sample votes
        votes = [
            AgentVote(
                agent_name="analyst",
                vote=VoteType.BUY,
                confidence=0.8,
                reasoning="Strong upward momentum",
                risk_assessment={"risk_score": 0.3}
            ),
            AgentVote(
                agent_name="risk",
                vote=VoteType.BUY,
                confidence=0.6,
                reasoning="Acceptable risk level",
                risk_assessment={"risk_score": 0.4}
            )
        ]

        # Test consensus calculation
        result = await consensus_engine.calculate_consensus(
            decision_id="test_001",
            symbol="AAPL",
            votes=votes
        )

        print(f"✅ Consensus calculated: {result.final_vote.value} with {result.confidence_score:.2f} confidence")

        return True

    except Exception as e:
        print(f"❌ Consensus engine error: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Testing SwaggyStacks Real-Time System Components")
    print("=" * 50)

    # Test basic Alpaca connection
    print("\n1. Testing Alpaca Stream Manager...")
    alpaca_ok = await test_alpaca_connection()

    # Test consensus engine
    print("\n2. Testing Consensus Engine...")
    consensus_ok = await test_consensus_engine()

    # Summary
    print("\n" + "=" * 50)
    if alpaca_ok and consensus_ok:
        print("✅ All core components working!")
        print("🎯 Ready to deploy live agents system")
    else:
        print("❌ Some components need attention")

    return alpaca_ok and consensus_ok

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test stopped by user")
        exit(0)