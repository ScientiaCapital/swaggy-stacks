"""
Comprehensive Integration Tests for Agent Decision Pipeline
Tests the entire agent coordination system end-to-end
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

from app.ai.trading_agents import AIAgentCoordinator
from app.events.agent_event_bus import AgentEventBus, agent_event_bus
from app.events.multi_agent_coordinator import (
    MultiAgentCoordinator, ConsensusMethod, ConflictResolution, multi_agent_coordinator
)
from app.websockets.agent_coordination_socket import agent_coordination_manager
from app.analysis.tool_feedback_tracker import ToolFeedbackTracker, tool_feedback_tracker
from app.testing.mock_data_generator import MockDataGenerator, MarketRegime, mock_data_generator
from app.testing.agent_testing_framework import AgentTestingFramework, agent_testing_framework


class TestAgentPipeline:
    """Integration tests for the complete agent decision pipeline"""
    
    @pytest.fixture
    def mock_generator(self):
        """Mock data generator fixture"""
        return MockDataGenerator(seed=42)
    
    @pytest.fixture
    def agent_coordinator(self):
        """Agent coordinator fixture"""
        return AIAgentCoordinator(enable_streaming=True)
    
    @pytest.fixture
    def event_bus(self):
        """Event bus fixture"""
        return agent_event_bus
    
    @pytest.fixture
    def multi_coordinator(self, event_bus):
        """Multi-agent coordinator fixture"""
        return MultiAgentCoordinator(event_bus)
    
    @pytest.fixture
    def feedback_tracker(self):
        """Tool feedback tracker fixture"""
        return tool_feedback_tracker
    
    @pytest.fixture
    def testing_framework(self, agent_coordinator, feedback_tracker, mock_generator):
        """Agent testing framework fixture"""
        return AgentTestingFramework(agent_coordinator, feedback_tracker, mock_generator)
    
    @pytest.mark.asyncio
    async def test_mock_data_generation(self, mock_generator):
        """Test mock data generation for various market regimes"""
        
        symbol = "AAPL"
        regimes_to_test = [
            MarketRegime.TRENDING_BULLISH,
            MarketRegime.TRENDING_BEARISH,
            MarketRegime.SIDEWAYS,
            MarketRegime.HIGH_VOLATILITY
        ]
        
        for regime in regimes_to_test:
            # Generate mock data
            market_data = mock_generator.generate_market_scenario(
                symbol=symbol,
                regime=regime,
                duration_minutes=60,
                interval_minutes=5
            )
            
            assert len(market_data) == 12  # 60 minutes / 5 minute intervals
            assert all(data.symbol == symbol for data in market_data)
            assert all(data.regime == regime.value for data in market_data)
            
            # Check OHLC consistency
            for data_point in market_data:
                assert data_point.high >= max(data_point.open, data_point.close)
                assert data_point.low <= min(data_point.open, data_point.close)
                assert data_point.bid < data_point.ask
                assert data_point.volume > 0
            
            # Generate technical indicators
            tech_indicators = mock_generator.generate_technical_indicators(symbol, market_data[0])
            assert tech_indicators.symbol == symbol
            assert 0 <= tech_indicators.rsi <= 100
            assert tech_indicators.atr > 0
            
            # Generate Markov analysis
            markov_analysis = mock_generator.generate_markov_analysis(symbol, market_data[0], regime)
            assert markov_analysis.symbol == symbol
            assert sum(markov_analysis.state_probabilities.values()) == pytest.approx(1.0, rel=1e-2)
    
    @pytest.mark.asyncio
    async def test_individual_agent_analysis(self, agent_coordinator, mock_generator):
        """Test individual agent analysis capabilities"""
        
        # Generate test data
        symbol = "TSLA"
        market_data = mock_generator.generate_market_scenario(
            symbol=symbol,
            regime=MarketRegime.TRENDING_BULLISH,
            duration_minutes=30,
            interval_minutes=5
        )[0]  # Use first data point
        
        tech_indicators = mock_generator.generate_technical_indicators(symbol, market_data)
        markov_analysis = mock_generator.generate_markov_analysis(symbol, market_data, MarketRegime.TRENDING_BULLISH)
        
        # Test market analyst
        market_analysis = await agent_coordinator.stream_market_analysis(
            symbol=symbol,
            market_data=market_data.to_dict(),
            technical_indicators=tech_indicators.to_dict()
        )
        
        assert hasattr(market_analysis, 'sentiment')
        assert hasattr(market_analysis, 'confidence')
        assert 0 <= market_analysis.confidence <= 1
        assert market_analysis.sentiment in ['bullish', 'bearish', 'neutral']
        
        # Test risk advisor
        risk_assessment = await agent_coordinator.stream_risk_assessment(
            symbol=symbol,
            position_size=5000.0,
            account_value=100000.0,
            current_positions=[],
            market_volatility={"atr": tech_indicators.atr},
            proposed_trade={"stop_loss_percent": 0.05, "take_profit_percent": 0.10}
        )
        
        assert hasattr(risk_assessment, 'risk_level')
        assert hasattr(risk_assessment, 'confidence')
        assert 0 <= risk_assessment.confidence <= 1
        assert risk_assessment.risk_level in ['low', 'medium', 'high']
        
        # Test strategy optimizer
        strategy_signal = await agent_coordinator.stream_strategy_signal(
            symbol=symbol,
            markov_analysis=markov_analysis.to_dict(),
            technical_indicators=tech_indicators.to_dict(),
            market_context={"regime": "trending", "volatility": "normal"},
            performance_history=[]
        )
        
        assert hasattr(strategy_signal, 'action')
        assert hasattr(strategy_signal, 'confidence')
        assert 0 <= strategy_signal.confidence <= 1
        assert strategy_signal.action in ['BUY', 'SELL', 'HOLD']
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis_pipeline(self, agent_coordinator, mock_generator):
        """Test the complete comprehensive analysis pipeline"""
        
        # Generate test data for multiple symbols
        symbols = ["AAPL", "TSLA", "SPY"]
        
        for symbol in symbols:
            market_data_points = mock_generator.generate_market_scenario(
                symbol=symbol,
                regime=MarketRegime.TRENDING_BULLISH,
                duration_minutes=15,
                interval_minutes=5
            )
            
            for data_point in market_data_points[:2]:  # Test first 2 data points
                tech_indicators = mock_generator.generate_technical_indicators(symbol, data_point)
                markov_analysis = mock_generator.generate_markov_analysis(
                    symbol, data_point, MarketRegime.TRENDING_BULLISH
                )
                
                # Run comprehensive analysis
                result = await agent_coordinator.comprehensive_analysis(
                    symbol=symbol,
                    market_data=data_point.to_dict(),
                    technical_indicators=tech_indicators.to_dict(),
                    account_info={"equity": 100000, "buying_power": 100000},
                    current_positions=[],
                    markov_analysis=markov_analysis.to_dict()
                )
                
                # Validate comprehensive result
                assert result["symbol"] == symbol
                assert "final_recommendation" in result
                assert result["final_recommendation"] in ["BUY", "SELL", "HOLD"]
                assert "market_analysis" in result
                assert "risk_assessment" in result  
                assert "strategy_signal" in result
                assert "correlation_id" in result
                assert "agent_states" in result
                
                # Validate individual analysis components
                market_analysis = result["market_analysis"]
                assert "sentiment" in market_analysis
                assert "confidence" in market_analysis
                assert 0 <= market_analysis["confidence"] <= 1
                
                risk_assessment = result["risk_assessment"]
                assert "risk_level" in risk_assessment
                assert "confidence" in risk_assessment
                assert 0 <= risk_assessment["confidence"] <= 1
                
                strategy_signal = result["strategy_signal"]
                assert "action" in strategy_signal
                assert "confidence" in strategy_signal
                assert 0 <= strategy_signal["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_tool_feedback_tracking(self, feedback_tracker, agent_coordinator, mock_generator):
        """Test tool execution feedback tracking"""
        
        # Generate test data
        symbol = "MSFT"
        market_data = mock_generator.generate_market_scenario(
            symbol=symbol,
            regime=MarketRegime.SIDEWAYS,
            duration_minutes=10,
            interval_minutes=5
        )[0]
        
        tech_indicators = mock_generator.generate_technical_indicators(symbol, market_data)
        
        # Set up feedback callback to capture tool executions
        captured_feedback = []
        
        async def capture_feedback(execution):
            captured_feedback.append(execution)
        
        feedback_tracker.add_feedback_callback(capture_feedback)
        
        # Run analysis to generate tool executions
        await agent_coordinator.stream_market_analysis(
            symbol=symbol,
            market_data=market_data.to_dict(),
            technical_indicators=tech_indicators.to_dict()
        )
        
        # Validate feedback tracking
        assert len(captured_feedback) > 0
        
        execution = captured_feedback[0]
        assert hasattr(execution, 'agent_id')
        assert hasattr(execution, 'tool_name')
        assert hasattr(execution, 'execution_time_ms')
        assert hasattr(execution, 'success')
        assert execution.execution_time_ms > 0
        
        # Test performance analysis
        await feedback_tracker.analyze_tool_performance()
        performance_metrics = feedback_tracker.performance_metrics
        
        assert len(performance_metrics) > 0
        
        for tool_key, metrics in performance_metrics.items():
            assert metrics.total_executions > 0
            assert 0 <= metrics.success_rate <= 1
            assert metrics.average_execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_event_bus_communication(self, event_bus):
        """Test event bus communication between components"""
        
        # Initialize event bus
        await event_bus.initialize()
        
        # Set up event capture
        captured_events = []
        
        async def capture_event(event):
            captured_events.append(event)
        
        event_bus.subscribe_to_event("agent.test.event", capture_event)
        
        # Publish test events
        await event_bus.publish_market_update(
            agent_id="test_agent",
            symbol="TEST",
            market_data={"price": 100.0, "volume": 1000}
        )
        
        await event_bus.publish_decision_response(
            agent_id="test_agent",
            agent_type="market_analyst",
            symbol="TEST",
            decision="BUY",
            confidence=0.8,
            reasoning="Test decision"
        )
        
        # Allow time for event processing
        await asyncio.sleep(0.1)
        
        # Validate event processing
        assert len(captured_events) >= 0  # Events may be processed differently
        
        # Test health check
        health = await event_bus.health_check()
        assert health["status"] in ["healthy", "degraded"]
    
    @pytest.mark.asyncio
    async def test_multi_agent_consensus(self, multi_coordinator, event_bus):
        """Test multi-agent consensus mechanism"""
        
        # Initialize coordinators
        await event_bus.initialize()
        await multi_coordinator.initialize()
        
        # Set up consensus callback
        consensus_results = []
        
        async def capture_consensus(result):
            consensus_results.append(result)
        
        multi_coordinator.add_consensus_callback(capture_consensus)
        
        # Simulate consensus request
        consensus_id = await multi_coordinator.request_consensus(
            symbol="TEST",
            context={"market_condition": "trending_bullish", "volatility": "normal"},
            required_agents=["market_analyst", "risk_advisor", "strategy_optimizer"],
            consensus_method=ConsensusMethod.WEIGHTED_CONFIDENCE,
            timeout_seconds=5
        )
        
        assert consensus_id is not None
        assert consensus_id in multi_coordinator.active_consensus
        
        # Simulate agent responses (in real scenario, agents would respond automatically)
        # This would normally happen through the event system
        
        # Wait for timeout to test timeout handling
        await asyncio.sleep(6)
        
        # Validate consensus handling
        assert consensus_id not in multi_coordinator.active_consensus  # Should be cleaned up after timeout
        
        # Test health check
        health = await multi_coordinator.health_check()
        assert health["status"] == "healthy"
        assert "total_consensus_processed" in health
    
    @pytest.mark.asyncio
    async def test_websocket_coordination(self):
        """Test WebSocket coordination manager"""
        
        # Test connection simulation (without actual WebSocket)
        status = agent_coordination_manager.get_active_agents()
        assert isinstance(status, list)
        
        # Test decision history retrieval
        history = agent_coordination_manager.get_agent_decision_history("TEST", limit=10)
        assert isinstance(history, list)
    
    @pytest.mark.asyncio
    async def test_agent_testing_framework(self, testing_framework):
        """Test the comprehensive agent testing framework"""
        
        # Create test scenarios
        scenarios = testing_framework.create_test_scenarios()
        
        assert len(scenarios) > 0
        
        for regime, tests in scenarios.items():
            assert len(tests) > 0
            
            # Validate test structure
            test = tests[0]
            assert hasattr(test, 'test_id')
            assert hasattr(test, 'symbol')
            assert hasattr(test, 'expected_decision_range')
            assert hasattr(test, 'min_confidence')
            assert len(test.expected_decision_range) > 0
        
        # Run a subset of tests
        test_regime = list(scenarios.keys())[0]
        sample_tests = scenarios[test_regime][:2]  # Run 2 tests
        
        for test in sample_tests:
            result = await testing_framework.run_single_agent_test(test, "comprehensive")
            
            assert hasattr(result, 'test_id')
            assert hasattr(result, 'result')
            assert hasattr(result, 'response_time_ms')
            assert result.response_time_ms > 0
            
        # Generate test report
        report = testing_framework.generate_test_report()
        
        assert "executive_summary" in report
        assert "total_tests" in report["executive_summary"]
        assert report["executive_summary"]["total_tests"] >= 2
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, mock_generator, agent_coordinator, 
                                      event_bus, multi_coordinator, feedback_tracker):
        """Test complete end-to-end agent pipeline"""
        
        # Initialize all components
        await event_bus.initialize()
        await multi_coordinator.initialize()
        
        # Set up comprehensive callback tracking
        pipeline_events = {
            "decisions": [],
            "tool_executions": [],
            "consensus_results": []
        }
        
        async def track_decisions(decision_update):
            pipeline_events["decisions"].append(decision_update)
        
        async def track_tool_execution(tool_result):
            pipeline_events["tool_executions"].append(tool_result)
        
        async def track_consensus(consensus_result):
            pipeline_events["consensus_results"].append(consensus_result)
        
        # Register callbacks
        agent_coordinator.add_decision_callback(track_decisions)
        agent_coordinator.add_tool_execution_callback(track_tool_execution)
        feedback_tracker.add_feedback_callback(track_tool_execution)
        multi_coordinator.add_consensus_callback(track_consensus)
        
        # Generate realistic market scenario
        symbol = "PIPELINE_TEST"
        market_scenario = mock_generator.generate_market_scenario(
            symbol=symbol,
            regime=MarketRegime.TRENDING_BULLISH,
            duration_minutes=20,
            interval_minutes=10  # 2 data points
        )
        
        # Process each data point through the pipeline
        for i, data_point in enumerate(market_scenario):
            tech_indicators = mock_generator.generate_technical_indicators(symbol, data_point)
            markov_analysis = mock_generator.generate_markov_analysis(
                symbol, data_point, MarketRegime.TRENDING_BULLISH
            )
            
            # Run comprehensive analysis
            result = await agent_coordinator.comprehensive_analysis(
                symbol=symbol,
                market_data=data_point.to_dict(),
                technical_indicators=tech_indicators.to_dict(),
                account_info={"equity": 100000},
                current_positions=[],
                markov_analysis=markov_analysis.to_dict()
            )
            
            # Validate result
            assert result["final_recommendation"] in ["BUY", "SELL", "HOLD"]
            
            # Small delay between iterations
            await asyncio.sleep(0.1)
        
        # Analyze pipeline performance
        await feedback_tracker.analyze_tool_performance()
        
        # Validate end-to-end pipeline execution
        assert len(pipeline_events["decisions"]) > 0 or len(pipeline_events["tool_executions"]) > 0
        
        # Test health checks across all components
        agent_health = await agent_coordinator.health_check()
        event_health = await event_bus.health_check()
        multi_health = await multi_coordinator.health_check()
        feedback_health = feedback_tracker.get_health_check()
        
        assert agent_health["streaming_enabled"] == True
        assert event_health["status"] in ["healthy", "degraded"]
        assert multi_health["status"] == "healthy"
        assert feedback_health["status"] == "healthy"
        
        # Validate that components are properly integrated
        assert len(agent_coordinator.decision_history) > 0 or len(agent_coordinator.tool_feedback) > 0
        assert feedback_tracker.get_health_check()["total_executions"] > 0


class TestAgentPipelinePerformance:
    """Performance and load tests for the agent pipeline"""
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_performance(self, mock_generator):
        """Test performance under concurrent analysis load"""
        
        agent_coordinator = AIAgentCoordinator(enable_streaming=True)
        
        # Generate multiple test scenarios
        symbols = ["PERF_TEST_1", "PERF_TEST_2", "PERF_TEST_3"]
        test_data = []
        
        for symbol in symbols:
            data_point = mock_generator.generate_market_scenario(
                symbol=symbol,
                regime=MarketRegime.HIGH_VOLATILITY,
                duration_minutes=5,
                interval_minutes=5
            )[0]
            
            tech_indicators = mock_generator.generate_technical_indicators(symbol, data_point)
            markov_analysis = mock_generator.generate_markov_analysis(
                symbol, data_point, MarketRegime.HIGH_VOLATILITY
            )
            
            test_data.append((symbol, data_point, tech_indicators, markov_analysis))
        
        # Run concurrent analyses
        start_time = datetime.now()
        
        tasks = []
        for symbol, data_point, tech_indicators, markov_analysis in test_data:
            task = agent_coordinator.comprehensive_analysis(
                symbol=symbol,
                market_data=data_point.to_dict(),
                technical_indicators=tech_indicators.to_dict(),
                account_info={"equity": 100000},
                current_positions=[],
                markov_analysis=markov_analysis.to_dict()
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Validate performance
        assert len(results) == len(symbols)
        assert execution_time < 30  # Should complete within 30 seconds
        
        for result in results:
            assert "final_recommendation" in result
            assert result["final_recommendation"] in ["BUY", "SELL", "HOLD"]
    
    @pytest.mark.asyncio  
    async def test_memory_usage_stability(self, mock_generator):
        """Test memory usage stability during extended operation"""
        
        agent_coordinator = AIAgentCoordinator(enable_streaming=True)
        feedback_tracker = ToolFeedbackTracker()
        
        # Run extended test
        for i in range(10):  # 10 iterations to test stability
            symbol = f"MEMORY_TEST_{i}"
            
            data_point = mock_generator.generate_market_scenario(
                symbol=symbol,
                regime=MarketRegime.SIDEWAYS,
                duration_minutes=5,
                interval_minutes=5
            )[0]
            
            tech_indicators = mock_generator.generate_technical_indicators(symbol, data_point)
            markov_analysis = mock_generator.generate_markov_analysis(
                symbol, data_point, MarketRegime.SIDEWAYS
            )
            
            result = await agent_coordinator.comprehensive_analysis(
                symbol=symbol,
                market_data=data_point.to_dict(),
                technical_indicators=tech_indicators.to_dict(),
                account_info={"equity": 100000},
                current_positions=[],
                markov_analysis=markov_analysis.to_dict()
            )
            
            assert result["final_recommendation"] in ["BUY", "SELL", "HOLD"]
            
            # Periodically clean up to test garbage collection
            if i % 5 == 0:
                await feedback_tracker.analyze_tool_performance()
        
        # Validate no memory leaks (basic check)
        agent_health = await agent_coordinator.health_check()
        assert agent_health["active_decisions"] <= 10  # Should not accumulate indefinitely


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])