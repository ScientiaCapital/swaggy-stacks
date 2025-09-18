"""
PydanticAI Trading Agents
========================

Type-safe trading agents using PydanticAI framework for enhanced reliability
and validation in high-frequency trading operations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from enum import Enum

from pydantic import BaseModel, Field, validator
import structlog

from app.ai.pydantic_base_agent import (
    PydanticBaseAgent,
    AgentContext,
    AgentResponse,
    TradingToolResult,
)

logger = structlog.get_logger(__name__)


class MarketSentiment(str, Enum):
    """Market sentiment enumeration"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TradingAction(str, Enum):
    """Trading action enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class MarketAnalysisResult(BaseModel):
    """Structured market analysis response"""

    sentiment: MarketSentiment = Field(..., description="Overall market sentiment")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")
    reasoning: str = Field(..., min_length=10, description="Analysis reasoning")
    key_factors: List[str] = Field(..., min_items=1, description="Key influencing factors")
    risk_level: RiskLevel = Field(..., description="Associated risk level")
    price_target: Optional[float] = Field(None, gt=0, description="Price target if applicable")
    support_level: Optional[float] = Field(None, gt=0, description="Support price level")
    resistance_level: Optional[float] = Field(None, gt=0, description="Resistance price level")
    technical_score: float = Field(..., ge=0.0, le=10.0, description="Technical analysis score")
    fundamental_score: float = Field(..., ge=0.0, le=10.0, description="Fundamental analysis score")

    @validator('price_target', 'support_level', 'resistance_level')
    def validate_price_levels(cls, v, values):
        """Ensure price levels are reasonable"""
        if v is not None and v <= 0:
            raise ValueError("Price levels must be positive")
        return v

    @validator('key_factors')
    def validate_key_factors(cls, v):
        """Ensure key factors are meaningful"""
        if not v or len(v) == 0:
            raise ValueError("At least one key factor must be provided")
        for factor in v:
            if not factor or len(factor.strip()) < 3:
                raise ValueError("Key factors must be meaningful (at least 3 characters)")
        return v


class RiskAssessmentResult(BaseModel):
    """Structured risk assessment response"""

    risk_level: RiskLevel = Field(..., description="Overall risk level")
    portfolio_heat: float = Field(..., ge=0.0, le=1.0, description="Portfolio heat (0-1)")
    recommended_position_size: float = Field(..., gt=0, description="Recommended position size")
    max_position_risk: float = Field(..., ge=0.0, description="Maximum position risk")
    stop_loss_percentage: float = Field(..., ge=0.0, le=0.5, description="Recommended stop loss")
    key_risk_factors: List[str] = Field(..., min_items=1, description="Key risk factors")
    diversification_score: float = Field(..., ge=0.0, le=10.0, description="Portfolio diversification")
    leverage_ratio: float = Field(default=1.0, ge=1.0, description="Current leverage ratio")
    margin_usage: float = Field(default=0.0, ge=0.0, le=1.0, description="Margin utilization")

    @validator('recommended_position_size')
    def validate_position_size(cls, v):
        """Ensure recommended position size is reasonable"""
        if v <= 0:
            raise ValueError("Recommended position size must be positive")
        return v


class StrategySignalResult(BaseModel):
    """Structured strategy signal response"""

    action: TradingAction = Field(..., description="Recommended trading action")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence")
    reasoning: str = Field(..., min_length=10, description="Signal reasoning")
    entry_price: Optional[float] = Field(None, gt=0, description="Suggested entry price")
    stop_loss: Optional[float] = Field(None, gt=0, description="Stop loss price")
    take_profit: Optional[float] = Field(None, gt=0, description="Take profit price")
    position_size: float = Field(..., gt=0, description="Recommended position size")
    time_horizon: Literal["scalp", "day", "swing", "position"] = Field(..., description="Trade time horizon")
    technical_factors: List[str] = Field(..., description="Supporting technical factors")
    risk_reward_ratio: float = Field(..., gt=0, description="Risk/reward ratio")

    @validator('stop_loss', 'take_profit')
    def validate_price_targets(cls, v, values):
        """Validate price targets are reasonable"""
        if v is not None and v <= 0:
            raise ValueError("Price targets must be positive")
        return v


class PydanticMarketAnalyst(PydanticBaseAgent[MarketAnalysisResult]):
    """
    Type-safe Market Analyst using PydanticAI

    Provides comprehensive market analysis with structured validation
    and type safety for trading decisions.
    """

    def __init__(self, **kwargs):
        super().__init__(
            agent_type="market_analyst",
            system_prompt=self._get_default_system_prompt(),
            **kwargs
        )

    def _get_default_system_prompt(self) -> str:
        """Get market analyst system prompt"""
        return """You are an expert market analyst specializing in technical and fundamental analysis.

Your role:
- Analyze market data, price movements, and technical indicators
- Assess market sentiment and identify key driving factors
- Provide structured analysis with confidence scores
- Consider both technical and fundamental factors
- Focus on actionable insights for trading decisions

Always provide:
- Clear sentiment classification (bullish/bearish/neutral)
- Specific confidence level (0.0 to 1.0)
- Detailed reasoning with key factors
- Risk assessment and price levels
- Technical and fundamental scores

Use the available tools to gather market data and calculate indicators.
Be precise, objective, and focus on data-driven insights."""

    def _get_response_type(self) -> type[BaseModel]:
        """Return the expected response type"""
        return MarketAnalysisResult

    async def analyze_market(
        self,
        symbol: str,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> AgentResponse[MarketAnalysisResult]:
        """Analyze market conditions for a given symbol"""

        agent_context = AgentContext(
            agent_id=f"market_analyst_{symbol}",
            symbol=symbol,
            correlation_id=correlation_id,
        )

        prompt = f"""
        Analyze the market conditions for {symbol}.

        Please:
        1. Use get_market_data tool to fetch current market information
        2. Use calculate_technical_indicators tool to get technical analysis
        3. Assess the overall market sentiment and provide reasoning
        4. Identify key factors influencing the market
        5. Determine risk level and provide price targets if applicable
        6. Score both technical and fundamental aspects

        Consider current market context: {context or 'General market conditions'}

        Provide a comprehensive analysis with high confidence if data strongly supports conclusions.
        """

        return await self.execute(prompt, agent_context)


class PydanticRiskAdvisor(PydanticBaseAgent[RiskAssessmentResult]):
    """
    Type-safe Risk Advisor using PydanticAI

    Provides comprehensive risk assessment with portfolio management
    and position sizing recommendations.
    """

    def __init__(self, **kwargs):
        super().__init__(
            agent_type="risk_advisor",
            system_prompt=self._get_default_system_prompt(),
            **kwargs
        )

    def _get_default_system_prompt(self) -> str:
        """Get risk advisor system prompt"""
        return """You are an expert risk management advisor specializing in portfolio risk assessment.

Your role:
- Assess portfolio risk and position sizing
- Calculate appropriate stop losses and risk metrics
- Evaluate diversification and correlation risks
- Provide recommendations for risk mitigation
- Monitor portfolio heat and margin usage

Always provide:
- Clear risk level classification (low/medium/high)
- Specific portfolio heat calculation (0.0 to 1.0)
- Recommended position size based on risk tolerance
- Key risk factors and mitigation strategies
- Stop loss recommendations and diversification score

Use the assess_portfolio_risk tool to evaluate proposed positions.
Focus on capital preservation and risk-adjusted returns."""

    def _get_response_type(self) -> type[BaseModel]:
        """Return the expected response type"""
        return RiskAssessmentResult

    async def assess_risk(
        self,
        symbol: str,
        position_size: float,
        account_value: float,
        current_positions: Optional[List[Dict[str, Any]]] = None,
        correlation_id: Optional[str] = None,
    ) -> AgentResponse[RiskAssessmentResult]:
        """Assess risk for a proposed trading position"""

        agent_context = AgentContext(
            agent_id=f"risk_advisor_{symbol}",
            symbol=symbol,
            correlation_id=correlation_id,
            max_position_size=account_value * 0.1,  # Max 10% of account
        )

        prompt = f"""
        Assess the risk for a proposed position in {symbol}.

        Position Details:
        - Symbol: {symbol}
        - Proposed Position Size: ${position_size:,.2f}
        - Account Value: ${account_value:,.2f}
        - Current Positions: {len(current_positions or [])} positions

        Please:
        1. Use assess_portfolio_risk tool to evaluate the position
        2. Calculate portfolio heat and position percentage
        3. Assess diversification impact with current positions
        4. Recommend appropriate position size and stop loss
        5. Identify key risk factors
        6. Provide risk mitigation strategies

        Focus on capital preservation and ensure position sizing aligns with risk management principles.
        Consider correlation risks with existing positions.
        """

        return await self.execute(prompt, agent_context)


class PydanticStrategyOptimizer(PydanticBaseAgent[StrategySignalResult]):
    """
    Type-safe Strategy Optimizer using PydanticAI

    Generates optimized trading signals with entry/exit points
    and risk management parameters.
    """

    def __init__(self, **kwargs):
        super().__init__(
            agent_type="strategy_optimizer",
            system_prompt=self._get_default_system_prompt(),
            **kwargs
        )

    def _get_default_system_prompt(self) -> str:
        """Get strategy optimizer system prompt"""
        return """You are an expert trading strategy optimizer specializing in signal generation.

Your role:
- Generate optimal trading signals (BUY/SELL/HOLD)
- Determine precise entry and exit points
- Calculate risk/reward ratios
- Optimize position sizing for maximum return
- Consider technical factors and market conditions

Always provide:
- Clear trading action (BUY/SELL/HOLD)
- High confidence level for strong signals
- Specific entry, stop loss, and take profit levels
- Recommended position size
- Time horizon for the trade
- Supporting technical factors
- Risk/reward ratio calculation

Use all available tools to gather comprehensive market data.
Focus on high-probability, well-defined trading opportunities."""

    def _get_response_type(self) -> type[BaseModel]:
        """Return the expected response type"""
        return StrategySignalResult

    async def generate_signal(
        self,
        symbol: str,
        market_context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> AgentResponse[StrategySignalResult]:
        """Generate optimized trading signal for a symbol"""

        agent_context = AgentContext(
            agent_id=f"strategy_optimizer_{symbol}",
            symbol=symbol,
            correlation_id=correlation_id,
        )

        prompt = f"""
        Generate an optimized trading signal for {symbol}.

        Market Context: {market_context or 'Standard market conditions'}

        Please:
        1. Use get_market_data tool to fetch current market information
        2. Use calculate_technical_indicators tool for technical analysis
        3. Analyze price action and identify trading opportunities
        4. Determine optimal entry point and timing
        5. Calculate appropriate stop loss and take profit levels
        6. Recommend position size and time horizon
        7. Calculate risk/reward ratio

        Generate signals only for high-probability setups with clear technical confirmation.
        If conditions are unclear, recommend HOLD with detailed reasoning.

        Focus on:
        - Clear technical patterns and confirmations
        - Appropriate risk management
        - Realistic profit targets
        - Optimal entry timing
        """

        return await self.execute(prompt, agent_context)


class PydanticTradingCoordinator:
    """
    Coordinates PydanticAI trading agents for comprehensive market analysis

    Provides type-safe, validated trading intelligence with structured
    error handling and performance monitoring.
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022"):
        self.model_name = model_name

        # Initialize PydanticAI agents
        self.market_analyst = PydanticMarketAnalyst(model_name=model_name)
        self.risk_advisor = PydanticRiskAdvisor(model_name=model_name)
        self.strategy_optimizer = PydanticStrategyOptimizer(model_name=model_name)

        logger.info(
            "PydanticAI Trading Coordinator initialized",
            model=model_name,
            agents=["market_analyst", "risk_advisor", "strategy_optimizer"]
        )

    async def comprehensive_analysis(
        self,
        symbol: str,
        position_size: float = 10000.0,
        account_value: float = 100000.0,
        current_positions: Optional[List[Dict[str, Any]]] = None,
        market_context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive trading analysis using all PydanticAI agents

        Returns structured, validated results from all agents with
        synthesized recommendations and metadata.
        """
        start_time = datetime.now()

        try:
            logger.info(
                "Starting comprehensive PydanticAI analysis",
                symbol=symbol,
                correlation_id=correlation_id,
            )

            # Run all agents concurrently for maximum performance
            market_analysis_task = self.market_analyst.analyze_market(
                symbol=symbol,
                context=market_context,
                correlation_id=correlation_id,
            )

            risk_assessment_task = self.risk_advisor.assess_risk(
                symbol=symbol,
                position_size=position_size,
                account_value=account_value,
                current_positions=current_positions,
                correlation_id=correlation_id,
            )

            strategy_signal_task = self.strategy_optimizer.generate_signal(
                symbol=symbol,
                market_context=market_context,
                correlation_id=correlation_id,
            )

            # Wait for all agents to complete
            market_analysis, risk_assessment, strategy_signal = await asyncio.gather(
                market_analysis_task,
                risk_assessment_task,
                strategy_signal_task,
            )

            # Synthesize final recommendation
            final_recommendation = self._synthesize_recommendation(
                market_analysis.data,
                risk_assessment.data,
                strategy_signal.data,
            )

            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "correlation_id": correlation_id,
                "execution_time_ms": execution_time_ms,
                "market_analysis": {
                    "result": market_analysis.data.dict(),
                    "confidence": market_analysis.confidence,
                    "execution_time_ms": market_analysis.execution_time_ms,
                },
                "risk_assessment": {
                    "result": risk_assessment.data.dict(),
                    "confidence": risk_assessment.confidence,
                    "execution_time_ms": risk_assessment.execution_time_ms,
                },
                "strategy_signal": {
                    "result": strategy_signal.data.dict(),
                    "confidence": strategy_signal.confidence,
                    "execution_time_ms": strategy_signal.execution_time_ms,
                },
                "final_recommendation": final_recommendation,
                "agent_metadata": {
                    "model": self.model_name,
                    "total_execution_time_ms": execution_time_ms,
                    "agent_performance": {
                        "market_analyst": market_analysis.metadata,
                        "risk_advisor": risk_assessment.metadata,
                        "strategy_optimizer": strategy_signal.metadata,
                    }
                }
            }

            logger.info(
                "Comprehensive PydanticAI analysis completed",
                symbol=symbol,
                final_recommendation=final_recommendation,
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id,
            )

            return result

        except Exception as e:
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            logger.error(
                "Comprehensive PydanticAI analysis failed",
                symbol=symbol,
                error=str(e),
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id,
            )

            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "correlation_id": correlation_id,
                "execution_time_ms": execution_time_ms,
                "error": str(e),
                "final_recommendation": "HOLD",
                "status": "failed",
            }

    def _synthesize_recommendation(
        self,
        market_analysis: MarketAnalysisResult,
        risk_assessment: RiskAssessmentResult,
        strategy_signal: StrategySignalResult,
    ) -> str:
        """
        Synthesize final recommendation from all agent results

        Uses structured validation and type safety to ensure
        consistent decision making logic.
        """
        # Risk always trumps everything else
        if risk_assessment.risk_level == RiskLevel.HIGH:
            return "HOLD"

        # Need sufficient confidence for action
        min_confidence = 0.6
        if (market_analysis.confidence < min_confidence or
            strategy_signal.confidence < min_confidence):
            return "HOLD"

        # Check for aligned signals
        if (market_analysis.sentiment == MarketSentiment.BULLISH and
            strategy_signal.action == TradingAction.BUY):
            return "BUY"
        elif (market_analysis.sentiment == MarketSentiment.BEARISH and
              strategy_signal.action == TradingAction.SELL):
            return "SELL"
        else:
            return "HOLD"

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all PydanticAI agents"""
        market_health = await self.market_analyst.health_check()
        risk_health = await self.risk_advisor.health_check()
        strategy_health = await self.strategy_optimizer.health_check()

        overall_health = "healthy"
        if any(agent["status"] != "healthy" for agent in [market_health, risk_health, strategy_health]):
            overall_health = "degraded"

        return {
            "status": overall_health,
            "model": self.model_name,
            "agents": {
                "market_analyst": market_health,
                "risk_advisor": risk_health,
                "strategy_optimizer": strategy_health,
            },
            "timestamp": datetime.now().isoformat(),
        }