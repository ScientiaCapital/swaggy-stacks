"""
AI Trading Advisor Service
Leverages GPT-4 and Claude for advanced market analysis and trading insights
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from app.core.exceptions import MCPConnectionError, MCPError
from app.core.logging import get_logger, log_execution_time
from app.mcp.orchestrator import MCPOrchestrator, MCPServerType, get_mcp_orchestrator

logger = get_logger(__name__)


class AIModel(Enum):
    """Available AI models for trading analysis"""

    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"


class AnalysisType(Enum):
    """Types of AI analysis available"""

    MARKET_SENTIMENT = "market_sentiment"
    TECHNICAL_ANALYSIS = "technical_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    STRATEGY_GENERATION = "strategy_generation"
    TRADE_RECOMMENDATION = "trade_recommendation"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"


@dataclass
class AIAnalysisRequest:
    """Request structure for AI analysis"""

    analysis_type: AnalysisType
    symbol: str
    timeframe: str
    market_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    model_preference: Optional[AIModel] = None
    temperature: float = 0.3
    max_tokens: int = 1000


@dataclass
class AIAnalysisResult:
    """Result structure for AI analysis"""

    analysis_type: AnalysisType
    symbol: str
    model_used: AIModel
    confidence: float
    analysis: str
    recommendations: List[str]
    risk_level: str
    timestamp: datetime
    processing_time: float
    raw_response: Optional[Dict[str, Any]] = None


class AITradingAdvisor:
    """
    AI-powered trading advisor using GPT-4 and Claude
    Provides advanced market analysis and trading insights
    """

    def __init__(self):
        self._orchestrator: Optional[MCPOrchestrator] = None
        self._initialized = False

        # Model preferences and fallbacks
        self.default_models = {
            AnalysisType.MARKET_SENTIMENT: AIModel.CLAUDE_3_5_SONNET,
            AnalysisType.TECHNICAL_ANALYSIS: AIModel.GPT_4_TURBO,
            AnalysisType.RISK_ASSESSMENT: AIModel.CLAUDE_3_5_SONNET,
            AnalysisType.STRATEGY_GENERATION: AIModel.GPT_4_TURBO,
            AnalysisType.TRADE_RECOMMENDATION: AIModel.CLAUDE_3_5_SONNET,
            AnalysisType.PORTFOLIO_OPTIMIZATION: AIModel.GPT_4_TURBO,
        }

        # Performance tracking
        self.request_counts = {model: 0 for model in AIModel}
        self.success_rates = {model: 0.0 for model in AIModel}
        self.avg_response_times = {model: 0.0 for model in AIModel}

        logger.info("AITradingAdvisor initialized")

    async def initialize(self):
        """Initialize the AI trading advisor"""
        if self._initialized:
            return

        try:
            self._orchestrator = await get_mcp_orchestrator()

            # Check which AI servers are available
            available_servers = []

            if await self._orchestrator.is_server_available(MCPServerType.OPENAI_GPT):
                available_servers.append("OpenAI GPT")

            if await self._orchestrator.is_server_available(
                MCPServerType.ANTHROPIC_CLAUDE
            ):
                available_servers.append("Anthropic Claude")

            if not available_servers:
                logger.warning(
                    "No AI servers available - advisor will have limited functionality"
                )
            else:
                logger.info(
                    f"AI Trading Advisor initialized with servers: {', '.join(available_servers)}"
                )

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize AI Trading Advisor: {e}")
            raise MCPError(f"AI Trading Advisor initialization failed: {str(e)}")

    @log_execution_time()
    async def analyze_market(self, request: AIAnalysisRequest) -> AIAnalysisResult:
        """Perform AI-powered market analysis"""
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()

        # Determine which model to use
        model = request.model_preference or self.default_models.get(
            request.analysis_type, AIModel.CLAUDE_3_5_SONNET
        )

        try:
            # Prepare the analysis prompt
            prompt = self._build_analysis_prompt(request)

            # Call the appropriate AI server
            response = await self._call_ai_model(model, prompt, request)

            # Parse and structure the response
            result = await self._parse_ai_response(response, request, model, start_time)

            # Update performance metrics
            self._update_metrics(
                model, True, (datetime.now() - start_time).total_seconds()
            )

            logger.info(
                "AI analysis completed",
                symbol=request.symbol,
                analysis_type=request.analysis_type.value,
                model=model.value,
                confidence=result.confidence,
            )

            return result

        except Exception as e:
            self._update_metrics(
                model, False, (datetime.now() - start_time).total_seconds()
            )
            logger.error(
                "AI analysis failed",
                symbol=request.symbol,
                analysis_type=request.analysis_type.value,
                model=model.value,
                error=str(e),
            )
            raise

    async def batch_analyze(
        self, requests: List[AIAnalysisRequest]
    ) -> List[AIAnalysisResult]:
        """Perform batch AI analysis with concurrency control"""
        if not requests:
            return []

        # Limit concurrent requests to prevent rate limiting
        semaphore = asyncio.Semaphore(3)

        async def analyze_with_semaphore(request):
            async with semaphore:
                try:
                    return await self.analyze_market(request)
                except Exception as e:
                    logger.warning(f"Batch analysis failed for {request.symbol}: {e}")
                    return None

        # Run analyses concurrently
        tasks = [analyze_with_semaphore(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed analyses
        successful_results = [
            r for r in results if r is not None and not isinstance(r, Exception)
        ]

        logger.info(
            f"Batch analysis completed: {len(successful_results)}/{len(requests)} successful"
        )

        return successful_results

    async def generate_trading_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        portfolio_context: Optional[Dict[str, Any]] = None,
    ) -> AIAnalysisResult:
        """Generate comprehensive trading signal using AI analysis"""
        request = AIAnalysisRequest(
            analysis_type=AnalysisType.TRADE_RECOMMENDATION,
            symbol=symbol,
            timeframe="1D",
            market_data=market_data,
            context=portfolio_context or {},
            temperature=0.2,  # Lower temperature for more consistent signals
            max_tokens=1500,
        )

        return await self.analyze_market(request)

    async def assess_portfolio_risk(
        self, portfolio: Dict[str, Any], market_conditions: Dict[str, Any]
    ) -> AIAnalysisResult:
        """Assess portfolio risk using AI analysis"""
        request = AIAnalysisRequest(
            analysis_type=AnalysisType.RISK_ASSESSMENT,
            symbol="PORTFOLIO",
            timeframe="1D",
            market_data=market_conditions,
            context={"portfolio": portfolio},
            temperature=0.1,  # Very conservative for risk assessment
            max_tokens=2000,
        )

        return await self.analyze_market(request)

    def _build_analysis_prompt(self, request: AIAnalysisRequest) -> str:
        """Build analysis prompt for AI model"""
        base_prompts = {
            AnalysisType.MARKET_SENTIMENT: f"""
            Analyze the market sentiment for {request.symbol} based on the provided data.

            Market Data: {json.dumps(request.market_data, indent=2)}
            Timeframe: {request.timeframe}

            Please provide:
            1. Overall sentiment (Bullish/Bearish/Neutral) with confidence level
            2. Key factors driving the sentiment
            3. Potential catalysts to watch
            4. Risk factors and concerns
            5. Specific trading recommendations

            Format your response as structured analysis with clear recommendations.
            """,
            AnalysisType.TECHNICAL_ANALYSIS: f"""
            Perform technical analysis for {request.symbol} using the provided market data.

            Market Data: {json.dumps(request.market_data, indent=2)}
            Timeframe: {request.timeframe}

            Please analyze:
            1. Trend direction and strength
            2. Support and resistance levels
            3. Key technical indicators (RSI, MACD, etc.)
            4. Chart patterns and formations
            5. Entry/exit points and stop-loss levels

            Provide specific, actionable trading recommendations.
            """,
            AnalysisType.RISK_ASSESSMENT: f"""
            Assess the trading risks for {request.symbol} or portfolio based on current conditions.

            Market Data: {json.dumps(request.market_data, indent=2)}
            Context: {json.dumps(request.context or {}, indent=2)}

            Please evaluate:
            1. Market risk factors
            2. Position sizing recommendations
            3. Risk/reward ratios
            4. Potential drawdown scenarios
            5. Risk mitigation strategies

            Provide quantified risk assessments where possible.
            """,
            AnalysisType.STRATEGY_GENERATION: f"""
            Generate a trading strategy for {request.symbol} based on current market conditions.

            Market Data: {json.dumps(request.market_data, indent=2)}
            Context: {json.dumps(request.context or {}, indent=2)}

            Please create:
            1. Strategy overview and rationale
            2. Entry and exit rules
            3. Position sizing guidelines
            4. Risk management rules
            5. Performance expectations

            Make the strategy specific and actionable.
            """,
            AnalysisType.TRADE_RECOMMENDATION: f"""
            Provide comprehensive trading recommendations for {request.symbol}.

            Market Data: {json.dumps(request.market_data, indent=2)}
            Context: {json.dumps(request.context or {}, indent=2)}
            Timeframe: {request.timeframe}

            Please provide:
            1. Clear BUY/SELL/HOLD recommendation with confidence level
            2. Target price levels and timeline
            3. Stop-loss and take-profit levels
            4. Position size recommendations
            5. Risk factors and monitoring points

            Be specific and actionable in your recommendations.
            """,
            AnalysisType.PORTFOLIO_OPTIMIZATION: f"""
            Optimize portfolio allocation based on current market conditions.

            Market Data: {json.dumps(request.market_data, indent=2)}
            Portfolio Context: {json.dumps(request.context or {}, indent=2)}

            Please recommend:
            1. Optimal asset allocation percentages
            2. Rebalancing suggestions
            3. Risk-adjusted return expectations
            4. Diversification improvements
            5. Hedging strategies if needed

            Provide specific allocation percentages and rationale.
            """,
        }

        return base_prompts.get(
            request.analysis_type, "Provide market analysis for the given data."
        )

    async def _call_ai_model(
        self, model: AIModel, prompt: str, request: AIAnalysisRequest
    ) -> Dict[str, Any]:
        """Call the appropriate AI model based on preference"""

        # Prepare the request payload
        payload = {
            "prompt": prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "model": model.value,
        }

        try:
            if model in [AIModel.GPT_4, AIModel.GPT_4_TURBO]:
                # Call OpenAI GPT server
                response = await self._orchestrator.call_openai_gpt(
                    "chat_completion", **payload
                )
            else:
                # Call Anthropic Claude server
                response = await self._orchestrator.call_anthropic_claude(
                    "messages", **payload
                )

            return response

        except MCPConnectionError:
            # Try fallback model
            fallback_model = self._get_fallback_model(model)
            if fallback_model and fallback_model != model:
                logger.warning(
                    f"Falling back from {model.value} to {fallback_model.value}"
                )
                return await self._call_ai_model(fallback_model, prompt, request)
            else:
                raise

    def _get_fallback_model(self, failed_model: AIModel) -> Optional[AIModel]:
        """Get fallback model when primary model fails"""
        fallbacks = {
            AIModel.GPT_4: AIModel.CLAUDE_3_5_SONNET,
            AIModel.GPT_4_TURBO: AIModel.CLAUDE_3_5_SONNET,
            AIModel.CLAUDE_3_5_SONNET: AIModel.GPT_4_TURBO,
            AIModel.CLAUDE_3_OPUS: AIModel.GPT_4,
        }

        return fallbacks.get(failed_model)

    async def _parse_ai_response(
        self,
        response: Dict[str, Any],
        request: AIAnalysisRequest,
        model: AIModel,
        start_time: datetime,
    ) -> AIAnalysisResult:
        """Parse AI response into structured result"""

        # Extract the analysis text from response (model-specific parsing)
        if "choices" in response:
            # OpenAI format
            analysis_text = response["choices"][0]["message"]["content"]
        elif "content" in response:
            # Anthropic format
            if isinstance(response["content"], list):
                analysis_text = response["content"][0]["text"]
            else:
                analysis_text = response["content"]
        else:
            # Fallback to string representation
            analysis_text = str(response)

        # Extract confidence from text or assign default
        confidence = self._extract_confidence(analysis_text)

        # Extract recommendations
        recommendations = self._extract_recommendations(analysis_text)

        # Determine risk level
        risk_level = self._determine_risk_level(analysis_text)

        processing_time = (datetime.now() - start_time).total_seconds()

        return AIAnalysisResult(
            analysis_type=request.analysis_type,
            symbol=request.symbol,
            model_used=model,
            confidence=confidence,
            analysis=analysis_text,
            recommendations=recommendations,
            risk_level=risk_level,
            timestamp=datetime.now(),
            processing_time=processing_time,
            raw_response=response,
        )

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence level from analysis text"""
        # Simple pattern matching for confidence keywords
        text_lower = text.lower()

        if "high confidence" in text_lower or "very confident" in text_lower:
            return 0.9
        elif (
            "moderate confidence" in text_lower or "moderately confident" in text_lower
        ):
            return 0.7
        elif "low confidence" in text_lower or "uncertain" in text_lower:
            return 0.4
        else:
            return 0.6  # Default moderate confidence

    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract specific recommendations from analysis text"""
        recommendations = []

        # Simple pattern matching for recommendation sections
        lines = text.split("\n")
        in_recommendation_section = False

        for line in lines:
            line = line.strip()
            if any(
                keyword in line.lower()
                for keyword in ["recommend", "suggest", "advise"]
            ):
                in_recommendation_section = True
                recommendations.append(line)
            elif in_recommendation_section and line.startswith(
                ("-", "•", "*", str(len(recommendations) + 1))
            ):
                recommendations.append(line)
            elif line == "" and in_recommendation_section:
                in_recommendation_section = False

        return recommendations[:5]  # Limit to top 5 recommendations

    def _determine_risk_level(self, text: str) -> str:
        """Determine risk level from analysis text"""
        text_lower = text.lower()

        high_risk_keywords = [
            "high risk",
            "very risky",
            "significant risk",
            "dangerous",
            "volatile",
        ]
        low_risk_keywords = [
            "low risk",
            "safe",
            "conservative",
            "stable",
            "minimal risk",
        ]

        if any(keyword in text_lower for keyword in high_risk_keywords):
            return "HIGH"
        elif any(keyword in text_lower for keyword in low_risk_keywords):
            return "LOW"
        else:
            return "MEDIUM"

    def _update_metrics(self, model: AIModel, success: bool, response_time: float):
        """Update performance metrics for AI models"""
        self.request_counts[model] += 1

        # Update success rate
        current_successes = self.success_rates[model] * (self.request_counts[model] - 1)
        if success:
            current_successes += 1
        self.success_rates[model] = current_successes / self.request_counts[model]

        # Update average response time
        current_total_time = self.avg_response_times[model] * (
            self.request_counts[model] - 1
        )
        self.avg_response_times[model] = (
            current_total_time + response_time
        ) / self.request_counts[model]

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get AI advisor performance metrics"""
        return {
            "model_metrics": {
                model.value: {
                    "request_count": self.request_counts[model],
                    "success_rate": f"{self.success_rates[model] * 100:.1f}%",
                    "avg_response_time": f"{self.avg_response_times[model]:.3f}s",
                }
                for model in AIModel
            },
            "server_status": {
                "openai_available": (
                    await self._orchestrator.is_server_available(
                        MCPServerType.OPENAI_GPT
                    )
                    if self._orchestrator
                    else False
                ),
                "anthropic_available": (
                    await self._orchestrator.is_server_available(
                        MCPServerType.ANTHROPIC_CLAUDE
                    )
                    if self._orchestrator
                    else False
                ),
            },
            "initialized": self._initialized,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check AI advisor health"""
        try:
            if not self._initialized:
                await self.initialize()

            # Test both AI services if available
            health_data = {
                "status": "healthy",
                "initialized": self._initialized,
                "available_models": [],
            }

            if self._orchestrator:
                if await self._orchestrator.is_server_available(
                    MCPServerType.OPENAI_GPT
                ):
                    health_data["available_models"].extend(["gpt-4", "gpt-4-turbo"])

                if await self._orchestrator.is_server_available(
                    MCPServerType.ANTHROPIC_CLAUDE
                ):
                    health_data["available_models"].extend(
                        ["claude-3-5-sonnet", "claude-3-opus"]
                    )

            if not health_data["available_models"]:
                health_data["status"] = "degraded"
                health_data["warning"] = "No AI models available"

            return health_data

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self._initialized,
            }


# Singleton instance
_ai_trading_advisor: Optional[AITradingAdvisor] = None


async def get_ai_trading_advisor() -> AITradingAdvisor:
    """Get the singleton AI Trading Advisor instance"""
    global _ai_trading_advisor

    if _ai_trading_advisor is None:
        _ai_trading_advisor = AITradingAdvisor()
        await _ai_trading_advisor.initialize()

    return _ai_trading_advisor
