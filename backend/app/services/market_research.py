"""
Market Research Service
Integrates Tavily and Sequential Thinking MCP servers for comprehensive market intelligence
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from app.ml.markov_system import MarkovSystem
from app.core.cache import get_market_cache
from app.core.exceptions import MCPError
from app.core.logging import get_logger, log_execution_time
from app.mcp.orchestrator import MCPOrchestrator, MCPServerType, get_mcp_orchestrator
from app.trading.trading_manager import TradingManager

logger = get_logger(__name__)


class SentimentLevel(Enum):
    """Market sentiment levels"""

    EXTREMELY_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    EXTREMELY_BULLISH = 2


class AnalysisComplexity(Enum):
    """Analysis complexity levels"""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class MarketSentiment:
    """Market sentiment analysis result"""

    symbol: str
    sentiment: SentimentLevel
    confidence: float
    sentiment_score: float
    news_summary: str
    key_factors: List[str]
    timestamp: datetime
    source: str = "tavily"


@dataclass
class ComplexAnalysisResult:
    """Complex market analysis result"""

    symbol: str
    analysis_type: str
    conclusion: str
    confidence: float
    reasoning_steps: List[str]
    supporting_evidence: List[Dict[str, Any]]
    risk_factors: List[str]
    opportunity_factors: List[str]
    timestamp: datetime
    complexity_level: AnalysisComplexity


@dataclass
class IntegratedAnalysis:
    """Integrated analysis combining multiple sources"""

    symbol: str
    market_sentiment: MarketSentiment
    complex_analysis: ComplexAnalysisResult
    markov_analysis: Dict[str, Any]
    trading_recommendation: Dict[str, Any]
    confidence_score: float
    timestamp: datetime


class MarketResearchService:
    """
    Market Research Service
    Provides comprehensive market intelligence using Tavily and Sequential Thinking MCP servers
    """

    def __init__(self, use_two_tier_cache: bool = True):
        self._orchestrator: Optional[MCPOrchestrator] = None
        self._markov_system: Optional[MarkovSystem] = None
        self._trading_manager: Optional[TradingManager] = None
        self._initialized = False
        self.use_two_tier_cache = use_two_tier_cache

        # Two-tier caching or fallback to traditional
        if use_two_tier_cache:
            self._market_cache = get_market_cache()
            logger.info("MarketResearchService using two-tier cache (TTL + Redis)")
        else:
            # Traditional in-memory cache
            self._sentiment_cache: Dict[str, MarketSentiment] = {}
            self._analysis_cache: Dict[str, ComplexAnalysisResult] = {}
            logger.info("MarketResearchService using traditional in-memory cache")

        self._cache_duration = timedelta(minutes=15)  # Cache for 15 minutes

        logger.info("MarketResearchService initialized")

    async def initialize(self):
        """Initialize the market research service"""
        if self._initialized:
            return

        try:
            # Get MCP orchestrator
            self._orchestrator = await get_mcp_orchestrator()

            # Initialize Markov system for technical analysis
            self._markov_system = MarkovSystem()

            # Get trading manager for integration
            self._trading_manager = TradingManager()

            # Verify MCP servers are available
            tavily_available = await self._orchestrator.is_server_available(
                MCPServerType.TAVILY
            )
            sequential_available = await self._orchestrator.is_server_available(
                MCPServerType.SEQUENTIAL_THINKING
            )

            logger.info(
                "MarketResearchService initialization complete",
                tavily_available=tavily_available,
                sequential_thinking_available=sequential_available,
            )

            self._initialized = True

        except Exception as e:
            logger.error("Failed to initialize MarketResearchService", error=str(e))
            raise MCPError(f"MarketResearchService initialization failed: {str(e)}")

    @log_execution_time()
    async def analyze_market_sentiment(
        self, symbol: str, lookback_days: int = 7, use_cache: bool = True
    ) -> MarketSentiment:
        """
        Analyze market sentiment using Tavily MCP server for real-time news and data

        Args:
            symbol: Stock symbol to analyze
            lookback_days: Number of days to look back for news
            use_cache: Whether to use cached results

        Returns:
            MarketSentiment analysis result
        """
        await self.initialize()

        # Check cache first
        cache_key = f"{symbol}_{lookback_days}"
        if use_cache and cache_key in self._sentiment_cache:
            cached_result = self._sentiment_cache[cache_key]
            if datetime.now() - cached_result.timestamp < self._cache_duration:
                logger.info("Using cached sentiment analysis", symbol=symbol)
                return cached_result

        try:
            # Query Tavily for market news and sentiment
            tavily_query = f"{symbol} stock market news sentiment analysis last {lookback_days} days"

            tavily_result = await self._orchestrator.call_tavily(
                "search",
                query=tavily_query,
                max_results=20,
                include_raw_content=True,
                search_depth="advanced",
            )

            # Process Tavily results to extract sentiment
            sentiment_data = await self._process_tavily_sentiment(tavily_result, symbol)

            # Create sentiment analysis result
            sentiment = MarketSentiment(
                symbol=symbol,
                sentiment=sentiment_data["sentiment_level"],
                confidence=sentiment_data["confidence"],
                sentiment_score=sentiment_data["sentiment_score"],
                news_summary=sentiment_data["news_summary"],
                key_factors=sentiment_data["key_factors"],
                timestamp=datetime.now(),
                source="tavily",
            )

            # Cache the result
            self._sentiment_cache[cache_key] = sentiment

            logger.info(
                "Market sentiment analysis completed",
                symbol=symbol,
                sentiment=sentiment.sentiment.name,
                confidence=sentiment.confidence,
            )

            return sentiment

        except Exception as e:
            logger.error(
                "Market sentiment analysis failed", symbol=symbol, error=str(e)
            )

            # Return neutral sentiment as fallback
            return MarketSentiment(
                symbol=symbol,
                sentiment=SentimentLevel.NEUTRAL,
                confidence=0.0,
                sentiment_score=0.0,
                news_summary=f"Error analyzing sentiment: {str(e)}",
                key_factors=[],
                timestamp=datetime.now(),
                source="error_fallback",
            )

    @log_execution_time()
    async def complex_analysis_workflow(
        self,
        symbol: str,
        analysis_type: str = "comprehensive_market_analysis",
        complexity: AnalysisComplexity = AnalysisComplexity.INTERMEDIATE,
        use_cache: bool = True,
    ) -> ComplexAnalysisResult:
        """
        Perform complex market analysis using Sequential Thinking MCP server

        Args:
            symbol: Stock symbol to analyze
            analysis_type: Type of analysis to perform
            complexity: Complexity level of analysis
            use_cache: Whether to use cached results

        Returns:
            ComplexAnalysisResult with detailed analysis
        """
        await self.initialize()

        # Check cache first
        cache_key = f"{symbol}_{analysis_type}_{complexity.value}"
        if use_cache and cache_key in self._analysis_cache:
            cached_result = self._analysis_cache[cache_key]
            if datetime.now() - cached_result.timestamp < self._cache_duration:
                logger.info(
                    "Using cached complex analysis",
                    symbol=symbol,
                    analysis_type=analysis_type,
                )
                return cached_result

        try:
            # Gather context for sequential thinking
            context_data = await self._gather_analysis_context(symbol)

            # Create complex analysis prompt
            analysis_prompt = self._create_analysis_prompt(
                symbol, analysis_type, complexity, context_data
            )

            # Execute sequential thinking workflow
            thinking_result = await self._orchestrator.call_sequential_thinking(
                "sequentialthinking",
                thought=analysis_prompt,
                thought_number=1,
                total_thoughts=self._estimate_thinking_steps(complexity),
                next_thought_needed=True,
                stage="Analysis",
            )

            # Process sequential thinking results
            analysis_data = await self._process_sequential_thinking_result(
                thinking_result, symbol, analysis_type
            )

            # Create complex analysis result
            result = ComplexAnalysisResult(
                symbol=symbol,
                analysis_type=analysis_type,
                conclusion=analysis_data["conclusion"],
                confidence=analysis_data["confidence"],
                reasoning_steps=analysis_data["reasoning_steps"],
                supporting_evidence=analysis_data["supporting_evidence"],
                risk_factors=analysis_data["risk_factors"],
                opportunity_factors=analysis_data["opportunity_factors"],
                timestamp=datetime.now(),
                complexity_level=complexity,
            )

            # Cache the result
            self._analysis_cache[cache_key] = result

            logger.info(
                "Complex analysis workflow completed",
                symbol=symbol,
                analysis_type=analysis_type,
                complexity=complexity.value,
                confidence=result.confidence,
            )

            return result

        except Exception as e:
            logger.error(
                "Complex analysis workflow failed",
                symbol=symbol,
                analysis_type=analysis_type,
                error=str(e),
            )

            # Return error result as fallback
            return ComplexAnalysisResult(
                symbol=symbol,
                analysis_type=analysis_type,
                conclusion=f"Analysis failed: {str(e)}",
                confidence=0.0,
                reasoning_steps=[],
                supporting_evidence=[],
                risk_factors=["Analysis system unavailable"],
                opportunity_factors=[],
                timestamp=datetime.now(),
                complexity_level=complexity,
            )

    @log_execution_time()
    async def integrated_strategy_analysis(
        self,
        symbol: str,
        analysis_complexity: AnalysisComplexity = AnalysisComplexity.INTERMEDIATE,
        include_sentiment: bool = True,
        include_technical: bool = True,
        include_complex_analysis: bool = True,
    ) -> IntegratedAnalysis:
        """
        Perform integrated analysis combining sentiment, technical, and complex analysis

        Args:
            symbol: Stock symbol to analyze
            analysis_complexity: Complexity level for analysis
            include_sentiment: Whether to include sentiment analysis
            include_technical: Whether to include technical analysis
            include_complex_analysis: Whether to include complex analysis

        Returns:
            IntegratedAnalysis with comprehensive results
        """
        await self.initialize()

        logger.info(
            "Starting integrated strategy analysis",
            symbol=symbol,
            complexity=analysis_complexity.value,
        )

        # Gather all analysis components concurrently
        analysis_tasks = []

        # Market sentiment analysis
        if include_sentiment:
            analysis_tasks.append(self.analyze_market_sentiment(symbol))

        # Complex analysis workflow
        if include_complex_analysis:
            analysis_tasks.append(
                self.complex_analysis_workflow(
                    symbol, "integrated_strategy_analysis", analysis_complexity
                )
            )

        # Technical analysis using Markov system
        markov_task = None
        if include_technical and self._markov_system:
            markov_task = self._perform_markov_analysis(symbol)
            analysis_tasks.append(markov_task)

        try:
            # Execute all analyses concurrently
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            # Process results
            sentiment_result = None
            complex_result = None
            markov_result = None

            result_index = 0
            if include_sentiment:
                sentiment_result = (
                    results[result_index]
                    if not isinstance(results[result_index], Exception)
                    else None
                )
                result_index += 1

            if include_complex_analysis:
                complex_result = (
                    results[result_index]
                    if not isinstance(results[result_index], Exception)
                    else None
                )
                result_index += 1

            if include_technical and markov_task:
                markov_result = (
                    results[result_index]
                    if not isinstance(results[result_index], Exception)
                    else None
                )

            # Generate trading recommendation
            trading_recommendation = await self._generate_trading_recommendation(
                symbol, sentiment_result, complex_result, markov_result
            )

            # Calculate overall confidence score
            confidence_score = self._calculate_integrated_confidence(
                sentiment_result, complex_result, markov_result
            )

            # Create integrated analysis result
            integrated_analysis = IntegratedAnalysis(
                symbol=symbol,
                market_sentiment=sentiment_result,
                complex_analysis=complex_result,
                markov_analysis=markov_result or {},
                trading_recommendation=trading_recommendation,
                confidence_score=confidence_score,
                timestamp=datetime.now(),
            )

            logger.info(
                "Integrated strategy analysis completed",
                symbol=symbol,
                confidence=confidence_score,
                recommendation=trading_recommendation.get("action", "unknown"),
            )

            return integrated_analysis

        except Exception as e:
            logger.error(
                "Integrated strategy analysis failed", symbol=symbol, error=str(e)
            )
            raise MCPError(f"Integrated analysis failed: {str(e)}")

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    async def _process_tavily_sentiment(
        self, tavily_result: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """Process Tavily search results to extract sentiment"""
        try:
            # Mock sentiment processing - in real implementation would use NLP
            # to analyze the news content from Tavily results

            # Simulate sentiment analysis based on mock data
            sentiment_score = 0.15  # Slightly bullish
            confidence = 0.72

            sentiment_level = SentimentLevel.NEUTRAL
            if sentiment_score > 0.5:
                sentiment_level = SentimentLevel.BULLISH
            elif sentiment_score > 0.8:
                sentiment_level = SentimentLevel.EXTREMELY_BULLISH
            elif sentiment_score < -0.5:
                sentiment_level = SentimentLevel.BEARISH
            elif sentiment_score < -0.8:
                sentiment_level = SentimentLevel.EXTREMELY_BEARISH

            return {
                "sentiment_level": sentiment_level,
                "confidence": confidence,
                "sentiment_score": sentiment_score,
                "news_summary": f"Market sentiment for {symbol} shows cautious optimism based on recent news coverage.",
                "key_factors": [
                    "Strong quarterly earnings report",
                    "Positive analyst upgrades",
                    "Sector rotation concerns",
                    "Macro economic uncertainty",
                ],
            }

        except Exception as e:
            logger.error("Failed to process Tavily sentiment", error=str(e))
            return {
                "sentiment_level": SentimentLevel.NEUTRAL,
                "confidence": 0.0,
                "sentiment_score": 0.0,
                "news_summary": "Unable to determine market sentiment",
                "key_factors": [],
            }

    async def _gather_analysis_context(self, symbol: str) -> Dict[str, Any]:
        """Gather context data for analysis"""
        context = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "market_conditions": "normal_volatility",
            "sector": "technology",  # Mock data
            "market_cap": "large_cap",
            "recent_performance": {"1d": 0.02, "1w": 0.05, "1m": -0.03, "3m": 0.12},
        }

        # Add technical indicators from Markov system if available
        if self._markov_system:
            try:
                # Mock technical indicators
                context["technical_indicators"] = {
                    "trend": "upward",
                    "momentum": "strong",
                    "volatility": "moderate",
                    "support_level": 150.25,
                    "resistance_level": 165.80,
                }
            except Exception as e:
                logger.warning("Failed to gather technical context", error=str(e))

        return context

    def _create_analysis_prompt(
        self,
        symbol: str,
        analysis_type: str,
        complexity: AnalysisComplexity,
        context_data: Dict[str, Any],
    ) -> str:
        """Create analysis prompt for sequential thinking"""

        base_prompt = f"""
        Analyze {symbol} for {analysis_type} with {complexity.value} complexity level.

        Context Data: {json.dumps(context_data, indent=2)}

        Please provide a thorough analysis considering:
        1. Technical analysis and chart patterns
        2. Fundamental analysis and company metrics
        3. Market sentiment and news impact
        4. Risk factors and potential catalysts
        5. Trading strategy recommendations

        Focus on actionable insights for trading decisions.
        """

        return base_prompt.strip()

    def _estimate_thinking_steps(self, complexity: AnalysisComplexity) -> int:
        """Estimate number of thinking steps based on complexity"""
        step_mapping = {
            AnalysisComplexity.BASIC: 3,
            AnalysisComplexity.INTERMEDIATE: 5,
            AnalysisComplexity.ADVANCED: 8,
            AnalysisComplexity.EXPERT: 12,
        }
        return step_mapping.get(complexity, 5)

    async def _process_sequential_thinking_result(
        self, thinking_result: Dict[str, Any], symbol: str, analysis_type: str
    ) -> Dict[str, Any]:
        """Process sequential thinking results"""
        try:
            # Mock processing of sequential thinking results
            # In real implementation, would parse the thinking chain

            return {
                "conclusion": f"Based on comprehensive analysis, {symbol} shows moderate bullish potential with balanced risk profile.",
                "confidence": 0.75,
                "reasoning_steps": [
                    "Technical analysis shows strong support at current levels",
                    "Fundamental metrics indicate fair valuation",
                    "Market sentiment is cautiously optimistic",
                    "Risk-reward ratio is favorable for current market conditions",
                ],
                "supporting_evidence": [
                    {
                        "type": "technical",
                        "indicator": "RSI",
                        "value": 65.2,
                        "interpretation": "Not overbought",
                    },
                    {
                        "type": "fundamental",
                        "metric": "P/E",
                        "value": 18.5,
                        "interpretation": "Reasonable valuation",
                    },
                    {
                        "type": "sentiment",
                        "metric": "news_sentiment",
                        "value": 0.15,
                        "interpretation": "Slightly positive",
                    },
                ],
                "risk_factors": [
                    "General market volatility",
                    "Sector rotation potential",
                    "Earnings uncertainty",
                ],
                "opportunity_factors": [
                    "Strong technical support",
                    "Positive earnings momentum",
                    "Favorable risk-reward ratio",
                ],
            }

        except Exception as e:
            logger.error("Failed to process sequential thinking result", error=str(e))
            return {
                "conclusion": "Analysis inconclusive",
                "confidence": 0.0,
                "reasoning_steps": [],
                "supporting_evidence": [],
                "risk_factors": ["Analysis system error"],
                "opportunity_factors": [],
            }

    async def _perform_markov_analysis(self, symbol: str) -> Dict[str, Any]:
        """Perform technical analysis using Markov system"""
        try:
            # Mock Markov analysis - in real implementation would use actual market data
            return {
                "current_regime": "bullish_trending",
                "regime_probability": 0.68,
                "transition_probabilities": {
                    "bullish_trending": 0.45,
                    "bearish_trending": 0.25,
                    "sideways": 0.30,
                },
                "volatility_regime": "normal",
                "momentum_indicators": {
                    "rsi": 65.2,
                    "macd": 1.25,
                    "bollinger_position": 0.75,
                },
                "support_resistance": {
                    "support": 150.25,
                    "resistance": 165.80,
                    "current_price": 158.40,
                },
            }

        except Exception as e:
            logger.error("Markov analysis failed", symbol=symbol, error=str(e))
            return {"error": str(e)}

    async def _generate_trading_recommendation(
        self,
        symbol: str,
        sentiment: Optional[MarketSentiment],
        complex_analysis: Optional[ComplexAnalysisResult],
        markov_analysis: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate trading recommendation based on all analyses"""

        # Aggregate signals
        bullish_signals = 0
        bearish_signals = 0
        total_weight = 0

        # Process sentiment
        if sentiment:
            weight = sentiment.confidence
            total_weight += weight

            if sentiment.sentiment in [
                SentimentLevel.BULLISH,
                SentimentLevel.EXTREMELY_BULLISH,
            ]:
                bullish_signals += weight
            elif sentiment.sentiment in [
                SentimentLevel.BEARISH,
                SentimentLevel.EXTREMELY_BEARISH,
            ]:
                bearish_signals += weight

        # Process complex analysis
        if complex_analysis and complex_analysis.confidence > 0:
            weight = complex_analysis.confidence
            total_weight += weight

            # Simple keyword analysis of conclusion
            conclusion_lower = complex_analysis.conclusion.lower()
            if any(
                word in conclusion_lower
                for word in ["bullish", "buy", "positive", "upward"]
            ):
                bullish_signals += weight
            elif any(
                word in conclusion_lower
                for word in ["bearish", "sell", "negative", "downward"]
            ):
                bearish_signals += weight

        # Process Markov analysis
        if markov_analysis and "current_regime" in markov_analysis:
            weight = 0.6  # Fixed weight for technical analysis
            total_weight += weight

            regime = markov_analysis["current_regime"]
            if "bullish" in regime or "trending" in regime:
                bullish_signals += weight * 0.7
            elif "bearish" in regime:
                bearish_signals += weight * 0.7

        # Generate recommendation
        if total_weight == 0:
            action = "hold"
            confidence = 0.0
        else:
            bullish_ratio = bullish_signals / total_weight
            bearish_ratio = bearish_signals / total_weight

            if bullish_ratio > 0.6:
                action = "buy"
                confidence = bullish_ratio
            elif bearish_ratio > 0.6:
                action = "sell"
                confidence = bearish_ratio
            else:
                action = "hold"
                confidence = max(bullish_ratio, bearish_ratio)

        return {
            "action": action,
            "confidence": round(confidence, 2),
            "signal_strength": (
                "strong"
                if confidence > 0.7
                else "moderate" if confidence > 0.4 else "weak"
            ),
            "bullish_signals": round(bullish_signals, 2),
            "bearish_signals": round(bearish_signals, 2),
            "total_weight": round(total_weight, 2),
            "reasoning": f"Recommendation based on {total_weight:.1f} weighted analysis signals",
            "position_sizing": self._calculate_position_sizing(confidence),
            "risk_management": {
                "stop_loss": (
                    0.95 if action == "buy" else 1.05 if action == "sell" else None
                ),
                "take_profit": (
                    1.08 if action == "buy" else 0.92 if action == "sell" else None
                ),
                "max_position_size": min(
                    0.1, confidence * 0.15
                ),  # Max 10%, scaled by confidence
            },
        }

    def _calculate_integrated_confidence(
        self,
        sentiment: Optional[MarketSentiment],
        complex_analysis: Optional[ComplexAnalysisResult],
        markov_analysis: Optional[Dict[str, Any]],
    ) -> float:
        """Calculate overall confidence score for integrated analysis"""

        confidence_scores = []

        if sentiment:
            confidence_scores.append(sentiment.confidence)

        if complex_analysis:
            confidence_scores.append(complex_analysis.confidence)

        if markov_analysis and "regime_probability" in markov_analysis:
            confidence_scores.append(markov_analysis["regime_probability"])

        if not confidence_scores:
            return 0.0

        # Return weighted average confidence
        return sum(confidence_scores) / len(confidence_scores)

    def _calculate_position_sizing(self, confidence: float) -> Dict[str, float]:
        """Calculate position sizing based on confidence"""
        base_size = 0.05  # 5% base position
        max_size = 0.15  # 15% maximum position

        position_size = base_size + (confidence * (max_size - base_size))

        return {
            "recommended_size": round(position_size, 3),
            "min_size": round(base_size, 3),
            "max_size": round(max_size, 3),
            "confidence_multiplier": round(confidence, 2),
        }

    async def _get_from_cache(self, key: str, cache_type: str = "general"):
        """Get item from appropriate cache"""
        if self.use_two_tier_cache:
            return await self._market_cache.get(f"{cache_type}:{key}")
        else:
            if cache_type == "sentiment":
                return self._sentiment_cache.get(key)
            else:
                return self._analysis_cache.get(key)

    async def _set_in_cache(self, key: str, value: Any, cache_type: str = "general"):
        """Set item in appropriate cache"""
        if self.use_two_tier_cache:
            await self._market_cache.set(f"{cache_type}:{key}", value)
        else:
            if cache_type == "sentiment":
                self._sentiment_cache[key] = value
            else:
                self._analysis_cache[key] = value

    async def clear_cache(self):
        """Clear analysis cache"""
        if self.use_two_tier_cache:
            cleared_count = await self._market_cache.clear()
            logger.info(f"Market research cache cleared: {cleared_count} items")
        else:
            sentiment_count = len(self._sentiment_cache)
            analysis_count = len(self._analysis_cache)
            self._sentiment_cache.clear()
            self._analysis_cache.clear()
            logger.info(
                f"Traditional market research cache cleared: {sentiment_count + analysis_count} items"
            )

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.use_two_tier_cache:
            cache_health = await self._market_cache.health_check()
            return {
                "cache_type": "two_tier_ttl_redis",
                "cache_health": cache_health,
                "cache_duration_minutes": self._cache_duration.total_seconds() / 60,
                "initialized": self._initialized,
            }
        else:
            return {
                "cache_type": "traditional_memory",
                "sentiment_cache_size": len(self._sentiment_cache),
                "analysis_cache_size": len(self._analysis_cache),
                "cache_duration_minutes": self._cache_duration.total_seconds() / 60,
                "initialized": self._initialized,
            }


# ============================================================================
# SINGLETON INSTANCE ACCESS
# ============================================================================

_market_research_service: Optional[MarketResearchService] = None


async def get_market_research_service() -> MarketResearchService:
    """Get the singleton MarketResearchService instance"""
    global _market_research_service

    if _market_research_service is None:
        _market_research_service = MarketResearchService()
        await _market_research_service.initialize()

    return _market_research_service


# Export main classes and functions
__all__ = [
    "MarketResearchService",
    "MarketSentiment",
    "ComplexAnalysisResult",
    "IntegratedAnalysis",
    "SentimentLevel",
    "AnalysisComplexity",
    "get_market_research_service",
]
