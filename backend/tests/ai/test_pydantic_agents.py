"""
Tests for PydanticAI Trading Agents
===================================

Comprehensive test suite for type-safe, validated trading agents
using the PydanticAI framework.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from app.ai.pydantic_base_agent import (
    AgentContext,
    AgentResponse,
    AgentExecutionStats,
    PydanticBaseAgent,
)
from app.ai.pydantic_trading_agents import (
    PydanticMarketAnalyst,
    PydanticRiskAdvisor,
    PydanticStrategyOptimizer,
    PydanticTradingCoordinator,
    MarketAnalysisResult,
    RiskAssessmentResult,
    StrategySignalResult,
    MarketSentiment,
    RiskLevel,
    TradingAction,
)


class TestAgentContext:
    """Test AgentContext validation and functionality"""

    def test_agent_context_creation(self):
        """Test creating valid agent context"""
        context = AgentContext(
            agent_id="test_agent_001",
            symbol="AAPL",
            risk_tolerance=0.05,
            max_position_size=50000.0,
        )

        assert context.agent_id == "test_agent_001"
        assert context.symbol == "AAPL"
        assert context.risk_tolerance == 0.05
        assert context.max_position_size == 50000.0
        assert isinstance(context.timestamp, datetime)

    def test_agent_context_validation(self):
        """Test agent context field validation"""
        # Test risk tolerance validation
        with pytest.raises(ValueError):
            AgentContext(
                agent_id="test",
                symbol="AAPL",
                risk_tolerance=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValueError):
            AgentContext(
                agent_id="test",
                symbol="AAPL",
                risk_tolerance=-0.1,  # Invalid: < 0.0
            )

        # Test max position size validation
        with pytest.raises(ValueError):
            AgentContext(
                agent_id="test",
                symbol="AAPL",
                max_position_size=-1000.0,  # Invalid: <= 0
            )


class TestAgentExecutionStats:
    """Test AgentExecutionStats tracking and calculations"""

    def test_stats_initialization(self):
        """Test stats initialize with correct defaults"""
        stats = AgentExecutionStats()

        assert stats.total_executions == 0
        assert stats.successful_executions == 0
        assert stats.failed_executions == 0
        assert stats.average_execution_time_ms == 0.0
        assert stats.average_confidence == 0.0
        assert stats.last_execution is None
        assert stats.error_rate == 0.0

    def test_stats_update_success(self):
        """Test updating stats with successful execution"""
        stats = AgentExecutionStats()

        stats.update_stats(execution_time_ms=150.0, confidence=0.85, success=True)

        assert stats.total_executions == 1
        assert stats.successful_executions == 1
        assert stats.failed_executions == 0
        assert stats.average_execution_time_ms == 15.0  # alpha * 150 + (1-alpha) * 0
        assert stats.average_confidence == 0.085  # alpha * 0.85 + (1-alpha) * 0
        assert stats.error_rate == 0.0
        assert stats.last_execution is not None

    def test_stats_update_failure(self):
        """Test updating stats with failed execution"""
        stats = AgentExecutionStats()

        stats.update_stats(execution_time_ms=200.0, confidence=0.0, success=False)

        assert stats.total_executions == 1
        assert stats.successful_executions == 0
        assert stats.failed_executions == 1
        assert stats.error_rate == 1.0

    def test_stats_exponential_moving_average(self):
        """Test exponential moving average calculation"""
        stats = AgentExecutionStats()

        # First update
        stats.update_stats(100.0, 0.8, True)
        first_time = stats.average_execution_time_ms
        first_conf = stats.average_confidence

        # Second update
        stats.update_stats(200.0, 0.6, True)

        # Should be exponentially weighted
        alpha = 0.1
        expected_time = alpha * 200.0 + (1 - alpha) * first_time
        expected_conf = alpha * 0.6 + (1 - alpha) * first_conf

        assert abs(stats.average_execution_time_ms - expected_time) < 0.01
        assert abs(stats.average_confidence - expected_conf) < 0.01


class TestMarketAnalysisResult:
    """Test MarketAnalysisResult validation"""

    def test_valid_market_analysis_result(self):
        """Test creating valid market analysis result"""
        result = MarketAnalysisResult(
            sentiment=MarketSentiment.BULLISH,
            confidence=0.85,
            reasoning="Strong technical indicators support bullish sentiment",
            key_factors=["RSI oversold", "MACD bullish crossover", "Volume spike"],
            risk_level=RiskLevel.MEDIUM,
            price_target=155.0,
            support_level=145.0,
            resistance_level=160.0,
            technical_score=8.5,
            fundamental_score=7.2,
        )

        assert result.sentiment == MarketSentiment.BULLISH
        assert result.confidence == 0.85
        assert len(result.key_factors) == 3
        assert result.technical_score == 8.5
        assert result.fundamental_score == 7.2

    def test_market_analysis_validation_errors(self):
        """Test market analysis validation errors"""
        # Test invalid confidence
        with pytest.raises(ValueError):
            MarketAnalysisResult(
                sentiment=MarketSentiment.BULLISH,
                confidence=1.5,  # Invalid: > 1.0
                reasoning="Test",
                key_factors=["Factor 1"],
                risk_level=RiskLevel.LOW,
                technical_score=5.0,
                fundamental_score=5.0,
            )

        # Test invalid price levels
        with pytest.raises(ValueError):
            MarketAnalysisResult(
                sentiment=MarketSentiment.BULLISH,
                confidence=0.8,
                reasoning="Test",
                key_factors=["Factor 1"],
                risk_level=RiskLevel.LOW,
                price_target=-10.0,  # Invalid: negative
                technical_score=5.0,
                fundamental_score=5.0,
            )

        # Test empty key factors
        with pytest.raises(ValueError):
            MarketAnalysisResult(
                sentiment=MarketSentiment.BULLISH,
                confidence=0.8,
                reasoning="Test",
                key_factors=[],  # Invalid: empty
                risk_level=RiskLevel.LOW,
                technical_score=5.0,
                fundamental_score=5.0,
            )


class TestRiskAssessmentResult:
    """Test RiskAssessmentResult validation"""

    def test_valid_risk_assessment_result(self):
        """Test creating valid risk assessment result"""
        result = RiskAssessmentResult(
            risk_level=RiskLevel.MEDIUM,
            portfolio_heat=0.15,
            recommended_position_size=8500.0,
            max_position_risk=425.0,
            stop_loss_percentage=0.05,
            key_risk_factors=["High volatility", "Market uncertainty"],
            diversification_score=7.5,
        )

        assert result.risk_level == RiskLevel.MEDIUM
        assert result.portfolio_heat == 0.15
        assert result.recommended_position_size == 8500.0
        assert len(result.key_risk_factors) == 2

    def test_risk_assessment_validation_errors(self):
        """Test risk assessment validation errors"""
        # Test invalid portfolio heat
        with pytest.raises(ValueError):
            RiskAssessmentResult(
                risk_level=RiskLevel.HIGH,
                portfolio_heat=1.5,  # Invalid: > 1.0
                recommended_position_size=5000.0,
                max_position_risk=250.0,
                stop_loss_percentage=0.05,
                key_risk_factors=["Risk factor"],
                diversification_score=5.0,
            )

        # Test invalid position size
        with pytest.raises(ValueError):
            RiskAssessmentResult(
                risk_level=RiskLevel.LOW,
                portfolio_heat=0.1,
                recommended_position_size=-1000.0,  # Invalid: negative
                max_position_risk=50.0,
                stop_loss_percentage=0.03,
                key_risk_factors=["Risk factor"],
                diversification_score=8.0,
            )


class TestStrategySignalResult:
    """Test StrategySignalResult validation"""

    def test_valid_strategy_signal_result(self):
        """Test creating valid strategy signal result"""
        result = StrategySignalResult(
            action=TradingAction.BUY,
            confidence=0.92,
            reasoning="Strong bullish breakout with volume confirmation",
            entry_price=152.50,
            stop_loss=148.75,
            take_profit=159.00,
            position_size=10000.0,
            time_horizon="swing",
            technical_factors=["Bullish flag pattern", "Volume breakout", "RSI momentum"],
            risk_reward_ratio=1.73,
        )

        assert result.action == TradingAction.BUY
        assert result.confidence == 0.92
        assert result.entry_price == 152.50
        assert result.risk_reward_ratio == 1.73
        assert len(result.technical_factors) == 3

    def test_strategy_signal_validation_errors(self):
        """Test strategy signal validation errors"""
        # Test invalid price targets
        with pytest.raises(ValueError):
            StrategySignalResult(
                action=TradingAction.SELL,
                confidence=0.8,
                reasoning="Test",
                entry_price=-10.0,  # Invalid: negative
                position_size=5000.0,
                time_horizon="day",
                technical_factors=["Factor"],
                risk_reward_ratio=1.5,
            )


@pytest.mark.asyncio
class TestPydanticMarketAnalyst:
    """Test PydanticMarketAnalyst functionality"""

    @pytest.fixture
    def mock_model_response(self):
        """Mock model response for testing"""
        return MarketAnalysisResult(
            sentiment=MarketSentiment.BULLISH,
            confidence=0.85,
            reasoning="Technical analysis shows strong bullish signals",
            key_factors=["RSI momentum", "MACD crossover", "Volume spike"],
            risk_level=RiskLevel.MEDIUM,
            technical_score=8.0,
            fundamental_score=7.5,
        )

    async def test_market_analyst_initialization(self):
        """Test market analyst initializes correctly"""
        analyst = PydanticMarketAnalyst()

        assert analyst.agent_type == "market_analyst"
        assert analyst.enable_tools is True
        assert analyst.max_retries == 3
        assert isinstance(analyst.stats, AgentExecutionStats)

    async def test_market_analyst_system_prompt(self):
        """Test market analyst has appropriate system prompt"""
        analyst = PydanticMarketAnalyst()
        prompt = analyst._get_default_system_prompt()

        assert "market analyst" in prompt.lower()
        assert "technical" in prompt.lower()
        assert "fundamental" in prompt.lower()
        assert "sentiment" in prompt.lower()

    async def test_market_analyst_response_type(self):
        """Test market analyst returns correct response type"""
        analyst = PydanticMarketAnalyst()
        response_type = analyst._get_response_type()

        assert response_type == MarketAnalysisResult

    @patch('app.ai.pydantic_base_agent.Agent')
    async def test_market_analyst_analyze_market(self, mock_agent_class, mock_model_response):
        """Test market analyst analyze_market method"""
        # Setup mock
        mock_agent_instance = AsyncMock()
        mock_result = MagicMock()
        mock_result.data = mock_model_response
        mock_result.all_messages.return_value = ["initial", "tool_call"]
        mock_agent_instance.run.return_value = mock_result
        mock_agent_class.return_value = mock_agent_instance

        # Create analyst and run analysis
        analyst = PydanticMarketAnalyst()
        result = await analyst.analyze_market("AAPL", correlation_id="test_123")

        # Verify results
        assert isinstance(result, AgentResponse)
        assert isinstance(result.data, MarketAnalysisResult)
        assert result.data.sentiment == MarketSentiment.BULLISH
        assert result.confidence == 0.85
        assert result.agent_id == "market_analyst_AAPL"
        assert result.execution_time_ms > 0

        # Verify agent was called correctly
        mock_agent_instance.run.assert_called_once()


@pytest.mark.asyncio
class TestPydanticRiskAdvisor:
    """Test PydanticRiskAdvisor functionality"""

    @pytest.fixture
    def mock_risk_response(self):
        """Mock risk assessment response"""
        return RiskAssessmentResult(
            risk_level=RiskLevel.MEDIUM,
            portfolio_heat=0.12,
            recommended_position_size=8000.0,
            max_position_risk=400.0,
            stop_loss_percentage=0.05,
            key_risk_factors=["Market volatility", "Position concentration"],
            diversification_score=7.0,
        )

    async def test_risk_advisor_initialization(self):
        """Test risk advisor initializes correctly"""
        advisor = PydanticRiskAdvisor()

        assert advisor.agent_type == "risk_advisor"
        assert "risk management" in advisor._get_default_system_prompt().lower()
        assert advisor._get_response_type() == RiskAssessmentResult

    @patch('app.ai.pydantic_base_agent.Agent')
    async def test_risk_advisor_assess_risk(self, mock_agent_class, mock_risk_response):
        """Test risk advisor assess_risk method"""
        # Setup mock
        mock_agent_instance = AsyncMock()
        mock_result = MagicMock()
        mock_result.data = mock_risk_response
        mock_result.all_messages.return_value = ["initial"]
        mock_agent_instance.run.return_value = mock_result
        mock_agent_class.return_value = mock_agent_instance

        # Create advisor and run assessment
        advisor = PydanticRiskAdvisor()
        result = await advisor.assess_risk(
            symbol="TSLA",
            position_size=10000.0,
            account_value=100000.0,
            current_positions=[{"symbol": "AAPL", "size": 5000}],
        )

        # Verify results
        assert isinstance(result, AgentResponse)
        assert isinstance(result.data, RiskAssessmentResult)
        assert result.data.risk_level == RiskLevel.MEDIUM
        assert result.data.portfolio_heat == 0.12
        assert result.agent_id == "risk_advisor_TSLA"


@pytest.mark.asyncio
class TestPydanticStrategyOptimizer:
    """Test PydanticStrategyOptimizer functionality"""

    @pytest.fixture
    def mock_strategy_response(self):
        """Mock strategy signal response"""
        return StrategySignalResult(
            action=TradingAction.BUY,
            confidence=0.88,
            reasoning="Strong technical breakout pattern",
            entry_price=155.25,
            stop_loss=150.00,
            take_profit=162.50,
            position_size=7500.0,
            time_horizon="swing",
            technical_factors=["Bullish flag", "Volume confirmation"],
            risk_reward_ratio=1.38,
        )

    async def test_strategy_optimizer_initialization(self):
        """Test strategy optimizer initializes correctly"""
        optimizer = PydanticStrategyOptimizer()

        assert optimizer.agent_type == "strategy_optimizer"
        assert "strategy optimizer" in optimizer._get_default_system_prompt().lower()
        assert optimizer._get_response_type() == StrategySignalResult

    @patch('app.ai.pydantic_base_agent.Agent')
    async def test_strategy_optimizer_generate_signal(self, mock_agent_class, mock_strategy_response):
        """Test strategy optimizer generate_signal method"""
        # Setup mock
        mock_agent_instance = AsyncMock()
        mock_result = MagicMock()
        mock_result.data = mock_strategy_response
        mock_result.all_messages.return_value = ["initial", "tool1", "tool2"]
        mock_agent_instance.run.return_value = mock_result
        mock_agent_class.return_value = mock_agent_instance

        # Create optimizer and generate signal
        optimizer = PydanticStrategyOptimizer()
        result = await optimizer.generate_signal(
            symbol="MSFT",
            market_context={"regime": "trending", "volatility": "normal"},
        )

        # Verify results
        assert isinstance(result, AgentResponse)
        assert isinstance(result.data, StrategySignalResult)
        assert result.data.action == TradingAction.BUY
        assert result.data.confidence == 0.88
        assert result.agent_id == "strategy_optimizer_MSFT"


@pytest.mark.asyncio
class TestPydanticTradingCoordinator:
    """Test PydanticTradingCoordinator functionality"""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator for testing"""
        return PydanticTradingCoordinator()

    @pytest.fixture
    def mock_agent_responses(self):
        """Mock responses from all agents"""
        market_response = AgentResponse(
            data=MarketAnalysisResult(
                sentiment=MarketSentiment.BULLISH,
                confidence=0.85,
                reasoning="Strong bullish signals",
                key_factors=["Technical momentum"],
                risk_level=RiskLevel.MEDIUM,
                technical_score=8.0,
                fundamental_score=7.5,
            ),
            confidence=0.85,
            execution_time_ms=150.0,
            agent_id="market_analyst_AAPL",
            metadata={"tools_used": 2},
        )

        risk_response = AgentResponse(
            data=RiskAssessmentResult(
                risk_level=RiskLevel.LOW,
                portfolio_heat=0.08,
                recommended_position_size=9000.0,
                max_position_risk=450.0,
                stop_loss_percentage=0.04,
                key_risk_factors=["Low volatility"],
                diversification_score=8.5,
            ),
            confidence=0.95,
            execution_time_ms=100.0,
            agent_id="risk_advisor_AAPL",
            metadata={"tools_used": 1},
        )

        strategy_response = AgentResponse(
            data=StrategySignalResult(
                action=TradingAction.BUY,
                confidence=0.82,
                reasoning="Clear bullish breakout",
                entry_price=152.0,
                position_size=9000.0,
                time_horizon="day",
                technical_factors=["Breakout pattern"],
                risk_reward_ratio=1.5,
            ),
            confidence=0.82,
            execution_time_ms=175.0,
            agent_id="strategy_optimizer_AAPL",
            metadata={"tools_used": 3},
        )

        return market_response, risk_response, strategy_response

    async def test_coordinator_initialization(self, coordinator):
        """Test coordinator initializes all agents"""
        assert isinstance(coordinator.market_analyst, PydanticMarketAnalyst)
        assert isinstance(coordinator.risk_advisor, PydanticRiskAdvisor)
        assert isinstance(coordinator.strategy_optimizer, PydanticStrategyOptimizer)
        assert coordinator.model_name == "claude-3-5-sonnet-20241022"

    async def test_coordinator_comprehensive_analysis(self, coordinator, mock_agent_responses):
        """Test comprehensive analysis coordination"""
        market_response, risk_response, strategy_response = mock_agent_responses

        # Mock all agent methods
        coordinator.market_analyst.analyze_market = AsyncMock(return_value=market_response)
        coordinator.risk_advisor.assess_risk = AsyncMock(return_value=risk_response)
        coordinator.strategy_optimizer.generate_signal = AsyncMock(return_value=strategy_response)

        # Run comprehensive analysis
        result = await coordinator.comprehensive_analysis(
            symbol="AAPL",
            position_size=10000.0,
            account_value=100000.0,
            correlation_id="test_coordination",
        )

        # Verify structure
        assert result["symbol"] == "AAPL"
        assert result["correlation_id"] == "test_coordination"
        assert "market_analysis" in result
        assert "risk_assessment" in result
        assert "strategy_signal" in result
        assert "final_recommendation" in result
        assert "agent_metadata" in result

        # Verify agent results are included
        assert result["market_analysis"]["result"]["sentiment"] == "bullish"
        assert result["risk_assessment"]["result"]["risk_level"] == "low"
        assert result["strategy_signal"]["result"]["action"] == "BUY"

        # Verify all agents were called
        coordinator.market_analyst.analyze_market.assert_called_once()
        coordinator.risk_advisor.assess_risk.assert_called_once()
        coordinator.strategy_optimizer.generate_signal.assert_called_once()

    async def test_coordinator_synthesize_recommendation(self, coordinator):
        """Test recommendation synthesis logic"""
        # Test BUY recommendation
        market_result = MarketAnalysisResult(
            sentiment=MarketSentiment.BULLISH,
            confidence=0.85,
            reasoning="Strong bullish",
            key_factors=["Factor"],
            risk_level=RiskLevel.MEDIUM,
            technical_score=8.0,
            fundamental_score=7.0,
        )

        risk_result = RiskAssessmentResult(
            risk_level=RiskLevel.LOW,  # Low risk allows action
            portfolio_heat=0.05,
            recommended_position_size=5000.0,
            max_position_risk=250.0,
            stop_loss_percentage=0.03,
            key_risk_factors=["Factor"],
            diversification_score=8.0,
        )

        strategy_result = StrategySignalResult(
            action=TradingAction.BUY,
            confidence=0.80,  # Sufficient confidence
            reasoning="Buy signal",
            position_size=5000.0,
            time_horizon="day",
            technical_factors=["Factor"],
            risk_reward_ratio=1.5,
        )

        recommendation = coordinator._synthesize_recommendation(
            market_result, risk_result, strategy_result
        )
        assert recommendation == "BUY"

        # Test HOLD due to high risk
        risk_result.risk_level = RiskLevel.HIGH
        recommendation = coordinator._synthesize_recommendation(
            market_result, risk_result, strategy_result
        )
        assert recommendation == "HOLD"

        # Test HOLD due to low confidence
        risk_result.risk_level = RiskLevel.LOW
        market_result.confidence = 0.4  # Below threshold
        recommendation = coordinator._synthesize_recommendation(
            market_result, risk_result, strategy_result
        )
        assert recommendation == "HOLD"

    async def test_coordinator_health_check(self, coordinator):
        """Test coordinator health check"""
        # Mock agent health checks
        coordinator.market_analyst.health_check = AsyncMock(return_value={
            "status": "healthy", "error_rate": 0.02
        })
        coordinator.risk_advisor.health_check = AsyncMock(return_value={
            "status": "healthy", "error_rate": 0.01
        })
        coordinator.strategy_optimizer.health_check = AsyncMock(return_value={
            "status": "healthy", "error_rate": 0.03
        })

        health = await coordinator.health_check()

        assert health["status"] == "healthy"
        assert health["model"] == coordinator.model_name
        assert "agents" in health
        assert len(health["agents"]) == 3

    async def test_coordinator_error_handling(self, coordinator):
        """Test coordinator handles agent errors gracefully"""
        # Mock an agent to raise an exception
        coordinator.market_analyst.analyze_market = AsyncMock(
            side_effect=Exception("Market data unavailable")
        )

        result = await coordinator.comprehensive_analysis(
            symbol="ERROR_TEST",
            correlation_id="error_test",
        )

        # Should return error result instead of raising
        assert result["status"] == "failed"
        assert result["final_recommendation"] == "HOLD"
        assert "error" in result
        assert "Market data unavailable" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])