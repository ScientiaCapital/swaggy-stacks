"""
Market Analyst Service - Extracted from trading_agents.py

Specialized service for market analysis and sentiment evaluation
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from .base_agent import BaseAIAgent
from .ollama_client import OllamaClient


@dataclass
class MarketAnalysis:
    """Market analysis result from AI agent"""
    symbol: str
    sentiment: str  # bullish, bearish, neutral
    confidence: float  # 0.0 to 1.0
    key_factors: List[str]
    recommendations: List[str]
    risk_level: str  # low, medium, high
    reasoning: str
    timestamp: datetime


class MarketAnalystService(BaseAIAgent):
    """AI agent specialized in market analysis and sentiment evaluation"""

    def __init__(self, ollama_client: OllamaClient):
        super().__init__(ollama_client, "analyst", "market_analysis.txt")

    def _get_default_prompt(self) -> str:
        return (
            "You are a market analysis expert. Provide clear, actionable insights "
            "based on market data and technical indicators. Focus on identifying "
            "trends, sentiment, and key risk factors."
        )

    async def analyze_market(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        context: str = "",
    ) -> MarketAnalysis:
        """Analyze market conditions for a specific symbol"""
        try:
            # Build data sections
            data_sections = {
                "Market Data": {
                    "Current Price": f"${market_data.get('current_price', 'N/A')}",
                    "Volume": market_data.get('volume', 'N/A'),
                    "52W High": f"${market_data.get('high_52w', 'N/A')}",
                    "52W Low": f"${market_data.get('low_52w', 'N/A')}",
                    "Market Cap": market_data.get('market_cap', 'N/A')
                },
                "Technical Indicators": {
                    "RSI": technical_indicators.get('rsi', 'N/A'),
                    "MACD": technical_indicators.get('macd', 'N/A'),
                    "MA20": f"${technical_indicators.get('ma20', 'N/A')}",
                    "MA50": f"${technical_indicators.get('ma50', 'N/A')}",
                    "Bollinger Bands": technical_indicators.get('bollinger_bands', 'N/A'),
                    "ATR": technical_indicators.get('atr', 'N/A')
                }
            }

            if context:
                data_sections["Additional Context"] = context

            # JSON schema
            json_schema = {
                "sentiment": "bullish|bearish|neutral",
                "confidence": "0.0-1.0",
                "key_factors": ["factor1", "factor2", "factor3"],
                "recommendations": ["rec1", "rec2"],
                "risk_level": "low|medium|high",
                "reasoning": "detailed explanation of your analysis"
            }

            # Build prompt
            instruction = "Provide your comprehensive market analysis"
            prompt = self._build_standard_prompt_template(
                symbol, data_sections, instruction, json_schema
            )

            # Generate response
            response = await self._generate_response(prompt, max_tokens=1024)

            # Parse response with fallback defaults
            default_values = {
                "sentiment": "neutral",
                "confidence": 0.5,
                "key_factors": ["Unable to analyze"],
                "recommendations": ["Review data quality"],
                "risk_level": "medium",
                "reasoning": "Analysis failed"
            }

            analysis_data = self._parse_json_response(response, default_values)

            # Validate and normalize data
            sentiment = self._validate_choice(
                analysis_data["sentiment"],
                ["bullish", "bearish", "neutral"],
                "neutral"
            )

            confidence = self._validate_confidence(analysis_data["confidence"])

            risk_level = self._validate_choice(
                analysis_data["risk_level"],
                ["low", "medium", "high"],
                "medium"
            )

            return MarketAnalysis(
                symbol=symbol,
                sentiment=sentiment,
                confidence=confidence,
                key_factors=analysis_data.get("key_factors", []),
                recommendations=analysis_data.get("recommendations", []),
                risk_level=risk_level,
                reasoning=analysis_data.get("reasoning", ""),
                timestamp=datetime.now()
            )

        except Exception as e:
            self.error_count += 1
            return MarketAnalysis(
                symbol=symbol,
                sentiment="neutral",
                confidence=0.0,
                key_factors=["Analysis error"],
                recommendations=["Retry analysis"],
                risk_level="high",
                reasoning=f"Error: {str(e)}",
                timestamp=datetime.now()
            )

    async def process(self, symbol: str, market_data: Dict[str, Any],
                     technical_indicators: Dict[str, Any], **kwargs) -> MarketAnalysis:
        """Main processing method for BaseAIAgent interface"""
        context = kwargs.get('context', '')
        return await self.analyze_market(symbol, market_data, technical_indicators, context)