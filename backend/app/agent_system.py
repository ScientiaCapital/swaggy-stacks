"""
Real-Time Event-Driven AI Agent System
Main integration point for the complete agent coordination architecture

This module provides the unified interface to:
- Real-time WebSocket agent coordination
- Event-driven agent communication via RabbitMQ  
- Multi-agent consensus mechanisms
- Tool execution feedback tracking
- Mock data generation for testing
- Comprehensive agent testing framework

Usage:
    from app.agent_system import AgentSystem
    
    # Initialize the complete system
    agent_system = AgentSystem()
    await agent_system.initialize()
    
    # Run real-time analysis with streaming
    result = await agent_system.run_real_time_analysis("AAPL", market_data)
    
    # Request multi-agent consensus
    consensus = await agent_system.request_consensus("TSLA", context, ["market_analyst", "risk_advisor"])
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import structlog

# Core agent components
from app.ai.trading_agents import AIAgentCoordinator
from app.websockets.agent_coordination_socket import (
    AgentCoordinationManager, 
    agent_coordination_manager,
    AgentDecisionUpdate,
    ToolExecutionResult,
    AgentCoordinationMessage,
    AgentStatusUpdate
)

# Event-driven architecture
from app.events.agent_event_bus import AgentEventBus, agent_event_bus
from app.events.multi_agent_coordinator import (
    MultiAgentCoordinator, 
    multi_agent_coordinator,
    ConsensusMethod,
    ConflictResolution
)

# Analysis and feedback
from app.analysis.tool_feedback_tracker import ToolFeedbackTracker, tool_feedback_tracker

# Testing components  
from app.testing.mock_data_generator import MockDataGenerator, MarketRegime, mock_data_generator
from app.testing.agent_testing_framework import AgentTestingFramework, agent_testing_framework

logger = structlog.get_logger(__name__)


class AgentSystem:
    """
    Unified Real-Time AI Agent System
    Coordinates all agent components for live trading decision making
    """
    
    def __init__(self):
        # Core components
        self.agent_coordinator = AIAgentCoordinator(enable_streaming=True)
        self.websocket_manager = agent_coordination_manager
        self.event_bus = agent_event_bus
        self.multi_coordinator = multi_agent_coordinator
        self.feedback_tracker = tool_feedback_tracker
        
        # Testing components
        self.mock_generator = mock_data_generator
        self.testing_framework = agent_testing_framework
        
        # System state
        self.initialized = False
        self.streaming_active = False
        self.real_time_callbacks: List[Callable] = []
        
    async def initialize(self) -> bool:
        """Initialize the complete agent system"""
        
        try:
            logger.info("Initializing Real-Time Agent System")
            
            # Initialize event bus
            await self.event_bus.initialize()
            logger.info("✅ Event bus initialized")
            
            # Initialize multi-agent coordinator
            await self.multi_coordinator.initialize()
            logger.info("✅ Multi-agent coordinator initialized")
            
            # Set up integrated callbacks
            await self._setup_integrated_callbacks()
            logger.info("✅ Integrated callbacks configured")
            
            # Start continuous feedback analysis
            await self.feedback_tracker.start_continuous_analysis(interval_minutes=15)
            logger.info("✅ Continuous feedback analysis started")
            
            self.initialized = True
            logger.info("🚀 Real-Time Agent System initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error("Failed to initialize agent system", error=str(e))
            return False
    
    async def _setup_integrated_callbacks(self):
        """Set up integrated callbacks between components"""
        
        # Connect agent coordinator to event bus and WebSocket
        async def stream_agent_decision(decision_update):
            # Stream to WebSocket clients
            await self.websocket_manager.broadcast_agent_decision(
                AgentDecisionUpdate(
                    agent_id=decision_update["agent_id"],
                    agent_type=decision_update["agent_type"],
                    symbol=decision_update["symbol"],
                    decision=decision_update["decision"],
                    confidence=decision_update["confidence"],
                    reasoning=decision_update["reasoning"],
                    timestamp=decision_update["timestamp"],
                    metadata=decision_update.get("metadata", {}),
                    tool_calls=decision_update.get("tool_calls", [])
                )
            )
            
            # Publish to event bus
            await self.event_bus.publish_decision_response(
                agent_id=decision_update["agent_id"],
                agent_type=decision_update["agent_type"],
                symbol=decision_update["symbol"],
                decision=decision_update["decision"],
                confidence=decision_update["confidence"],
                reasoning=decision_update["reasoning"],
                metadata=decision_update.get("metadata", {}),
                tool_calls=decision_update.get("tool_calls", []),
                correlation_id=decision_update.get("correlation_id")
            )
        
        async def stream_tool_execution(tool_result):
            # Stream to WebSocket clients
            await self.websocket_manager.broadcast_tool_execution_result(
                ToolExecutionResult(
                    agent_id=tool_result["agent_id"],
                    tool_name=tool_result["tool_name"],
                    execution_id=tool_result["execution_id"],
                    status=tool_result["status"],
                    result=tool_result["result"],
                    execution_time_ms=tool_result["execution_time_ms"],
                    timestamp=tool_result["timestamp"],
                    error_message=tool_result.get("error_message")
                )
            )
            
            # Record in feedback tracker
            await self.feedback_tracker.record_tool_execution(
                agent_id=tool_result["agent_id"],
                agent_type=tool_result.get("agent_type", "unknown"),
                tool_name=tool_result["tool_name"],
                input_params=tool_result.get("input_params", {}),
                output_result=tool_result["result"],
                execution_time_ms=tool_result["execution_time_ms"],
                success=tool_result["status"] == "success",
                error_message=tool_result.get("error_message"),
                context=tool_result.get("context", {})
            )
            
            # Publish to event bus
            await self.event_bus.publish_tool_execution(
                agent_id=tool_result["agent_id"],
                tool_name=tool_result["tool_name"],
                execution_id=tool_result["execution_id"],
                status=tool_result["status"],
                result=tool_result["result"],
                execution_time_ms=tool_result["execution_time_ms"],
                error_message=tool_result.get("error_message")
            )
        
        async def stream_coordination(coordination_update):
            # Stream to WebSocket clients
            await self.websocket_manager.broadcast_coordination_message(
                AgentCoordinationMessage(
                    sender_agent_id=coordination_update.get("sender_agent_id", "system"),
                    recipient_agent_id=coordination_update.get("recipient_agent_id"),
                    message_type=coordination_update["message_type"],
                    payload=coordination_update.get("payload", {}),
                    timestamp=coordination_update["timestamp"],
                    requires_response=coordination_update.get("requires_response", False)
                )
            )
            
            # Trigger real-time callbacks
            for callback in self.real_time_callbacks:
                try:
                    await callback("coordination", coordination_update)
                except Exception as e:
                    logger.warning("Real-time callback failed", error=str(e))
        
        # Register callbacks
        self.agent_coordinator.add_decision_callback(stream_agent_decision)
        self.agent_coordinator.add_tool_execution_callback(stream_tool_execution)
        self.agent_coordinator.add_coordination_callback(stream_coordination)
    
    def add_real_time_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for real-time system events"""
        self.real_time_callbacks.append(callback)
    
    async def run_real_time_analysis(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        account_info: Dict[str, Any],
        current_positions: List[Dict] = None,
        markov_analysis: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive real-time analysis with streaming
        This is the main entry point for live trading decisions
        """
        
        if not self.initialized:
            raise RuntimeError("Agent system not initialized. Call initialize() first.")
        
        logger.info("Running real-time analysis", symbol=symbol)
        
        # Use mock data if not provided (for testing)
        if markov_analysis is None:
            market_regime = MarketRegime.TRENDING_BULLISH  # Could be determined from market_data
            mock_data_point = self.mock_generator.generate_market_scenario(
                symbol=symbol, regime=market_regime, duration_minutes=5
            )[0]
            markov_analysis = self.mock_generator.generate_markov_analysis(
                symbol, mock_data_point, market_regime
            ).to_dict()
        
        # Run comprehensive analysis with real-time streaming
        result = await self.agent_coordinator.comprehensive_analysis(
            symbol=symbol,
            market_data=market_data,
            technical_indicators=technical_indicators,
            account_info=account_info or {"equity": 100000, "buying_power": 100000},
            current_positions=current_positions or [],
            markov_analysis=markov_analysis
        )
        
        # Update agent status
        for agent_type in ["market_analyst", "risk_advisor", "strategy_optimizer"]:
            await self.event_bus.publish_status_update(
                agent_id=f"{agent_type}_{symbol}",
                agent_type=agent_type,
                status="active",
                current_task=f"analyzed_{symbol}",
                performance_metrics={"last_analysis": datetime.now().isoformat()}
            )
        
        logger.info("Real-time analysis completed", 
                   symbol=symbol,
                   decision=result["final_recommendation"],
                   correlation_id=result.get("correlation_id"))
        
        return result
    
    async def request_consensus(
        self,
        symbol: str,
        context: Dict[str, Any],
        required_agents: List[str] = None,
        consensus_method: ConsensusMethod = ConsensusMethod.WEIGHTED_CONFIDENCE,
        timeout_seconds: int = 30
    ) -> str:
        """
        Request multi-agent consensus for trading decision
        Returns consensus_id for tracking the result
        """
        
        if not self.initialized:
            raise RuntimeError("Agent system not initialized. Call initialize() first.")
        
        if required_agents is None:
            required_agents = ["market_analyst", "risk_advisor", "strategy_optimizer"]
        
        logger.info("Requesting multi-agent consensus",
                   symbol=symbol,
                   required_agents=required_agents,
                   method=consensus_method.value)
        
        consensus_id = await self.multi_coordinator.request_consensus(
            symbol=symbol,
            context=context,
            required_agents=required_agents,
            consensus_method=consensus_method,
            timeout_seconds=timeout_seconds,
            requester_id="agent_system"
        )
        
        return consensus_id
    
    async def get_consensus_result(self, consensus_id: str) -> Optional[Dict[str, Any]]:
        """Get consensus result by ID"""
        
        result = self.multi_coordinator.get_consensus_result(consensus_id)
        if result:
            return {
                "consensus_id": result.consensus_id,
                "symbol": result.symbol,
                "final_decision": result.final_decision,
                "confidence": result.confidence,
                "consensus_achieved": result.consensus_achieved,
                "participating_agents": result.participating_agents,
                "reasoning": result.reasoning,
                "processing_time_ms": result.processing_time_ms,
                "timestamp": result.timestamp.isoformat()
            }
        return None
    
    async def run_agent_tests(
        self,
        regime_filter: Optional[List[str]] = None,
        agent_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive agent testing suite
        Returns detailed performance report
        """
        
        if agent_types is None:
            agent_types = ["comprehensive", "market_analyst", "risk_advisor", "strategy_optimizer"]
        
        logger.info("Running agent testing suite",
                   regime_filter=regime_filter,
                   agent_types=agent_types)
        
        # Run comprehensive tests
        performance_reports = await self.testing_framework.run_comprehensive_agent_test(
            regime_filter=regime_filter,
            agent_types=agent_types
        )
        
        # Generate final report
        test_report = self.testing_framework.generate_test_report()
        
        return {
            "performance_reports": {k: v.__dict__ for k, v in performance_reports.items()},
            "test_report": test_report,
            "timestamp": datetime.now().isoformat()
        }
    
    async def simulate_market_stream(
        self,
        symbol: str,
        regime: MarketRegime,
        duration_minutes: int = 60,
        interval_seconds: int = 30
    ):
        """
        Simulate real-time market data streaming for testing
        Continuously feeds mock data to agents
        """
        
        logger.info("Starting market stream simulation",
                   symbol=symbol,
                   regime=regime.value,
                   duration_minutes=duration_minutes)
        
        self.streaming_active = True
        
        # Stream callback for processing each data point
        async def process_stream_data(stream_data):
            if not self.streaming_active:
                return
            
            # Run real-time analysis for each stream update
            result = await self.run_real_time_analysis(
                symbol=symbol,
                market_data=stream_data["market_data"],
                technical_indicators=stream_data["technical_indicators"],
                account_info={"equity": 100000, "buying_power": 100000},
                markov_analysis=stream_data["markov_analysis"]
            )
            
            logger.info("Stream analysis completed",
                       symbol=symbol,
                       decision=result["final_recommendation"],
                       timestamp=stream_data["timestamp"])
        
        # Start streaming
        await self.mock_generator.stream_market_data(
            symbol=symbol,
            regime=regime,
            callback=process_stream_data,
            interval_seconds=interval_seconds,
            duration_minutes=duration_minutes
        )
        
        self.streaming_active = False
        logger.info("Market stream simulation completed", symbol=symbol)
    
    def stop_streaming(self):
        """Stop active market streaming"""
        self.streaming_active = False
        logger.info("Market streaming stopped")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        # Get health checks from all components
        agent_health = await self.agent_coordinator.health_check()
        event_health = await self.event_bus.health_check()
        multi_health = await self.multi_coordinator.health_check()
        feedback_health = self.feedback_tracker.get_health_check()
        
        return {
            "system_status": {
                "initialized": self.initialized,
                "streaming_active": self.streaming_active,
                "real_time_callbacks": len(self.real_time_callbacks)
            },
            "component_health": {
                "agent_coordinator": agent_health,
                "event_bus": event_health,
                "multi_coordinator": multi_health,
                "feedback_tracker": feedback_health
            },
            "performance_summary": {
                "active_agents": len(self.websocket_manager.get_active_agents()),
                "total_decisions": len(self.agent_coordinator.decision_history),
                "consensus_processed": multi_health.get("total_consensus_processed", 0),
                "tool_executions": feedback_health.get("total_executions", 0)
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Graceful shutdown of the agent system"""
        
        logger.info("Shutting down Real-Time Agent System")
        
        # Stop streaming
        self.stop_streaming()
        
        # Stop continuous analysis
        await self.feedback_tracker.stop_continuous_analysis()
        
        # Shutdown event bus
        await self.event_bus.shutdown()
        
        self.initialized = False
        logger.info("Agent system shutdown completed")


# Global agent system instance
agent_system = AgentSystem()


# Convenience functions for common operations
async def initialize_agent_system() -> bool:
    """Initialize the global agent system"""
    return await agent_system.initialize()


async def run_real_time_analysis(symbol: str, market_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Convenience function for real-time analysis"""
    return await agent_system.run_real_time_analysis(symbol, market_data, **kwargs)


async def request_agent_consensus(symbol: str, context: Dict[str, Any], **kwargs) -> str:
    """Convenience function for requesting consensus"""
    return await agent_system.request_consensus(symbol, context, **kwargs)


async def get_system_status() -> Dict[str, Any]:
    """Convenience function for system status"""
    return await agent_system.get_system_status()


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize system
        await initialize_agent_system()
        
        # Run sample analysis
        sample_market_data = {
            "symbol": "AAPL",
            "price": 150.0,
            "volume": 1000000,
            "timestamp": datetime.now().isoformat()
        }
        
        sample_tech_indicators = {
            "rsi": 65.0,
            "macd": 1.2,
            "bb_upper": 155.0,
            "bb_lower": 145.0
        }
        
        result = await run_real_time_analysis(
            symbol="AAPL",
            market_data=sample_market_data,
            technical_indicators=sample_tech_indicators
        )
        
        print(f"Analysis Result: {result['final_recommendation']}")
        
        # Get system status
        status = await get_system_status()
        print(f"System Status: {status['system_status']['initialized']}")
        
        # Shutdown
        await agent_system.shutdown()
    
    # Run example
    asyncio.run(main())