"""
LLM-Based Prediction Models Using Chinese LLMs
Leverages the "secret sauce" of specialized Chinese LLMs for trading predictions
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import structlog

from app.ai.ollama_client import OllamaClient
from app.analysis.technical_indicators import TechnicalIndicators
from app.analysis.modern_indicators import ModernIndicators

logger = structlog.get_logger()


@dataclass
class PredictionResult:
    """Result from LLM ensemble prediction"""
    symbol: str
    prediction_type: str  # price_direction, volatility, trend_strength
    prediction_value: Union[str, float]  # BULLISH/BEARISH/NEUTRAL or numeric value
    confidence: float  # 0.0 to 1.0
    horizon_days: int
    reasoning: str
    model_contributions: Dict[str, Any]  # Individual model predictions
    market_context: Dict[str, Any]
    timestamp: datetime


@dataclass
class MarketContext:
    """Market context converted to natural language"""
    price_narrative: str
    trend_analysis: str
    volatility_assessment: str
    volume_story: str
    indicator_summary: str


class LLMPredictor:
    """
    Advanced LLM-based prediction system using ensemble of Chinese LLMs
    Each model is specialized for different aspects of market analysis:
    - Qwen2.5: Quantitative analysis with mathematical reasoning
    - Yi-6B: Technical pattern recognition
    - GLM-4-9B: Risk-aware predictions
    - DeepSeek: Strategy-aligned forecasting
    """

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.ollama_client = ollama_client or OllamaClient()
        self.technical_indicators = TechnicalIndicators()
        self.modern_indicators = ModernIndicators()

        # Model specializations with context limits
        self.model_config = {
            "qwen_quant": {
                "specialization": "quantitative_analysis",
                "context_limit": 32768,  # Largest context window
                "temperature": 0.3,
                "weight": 0.3,
                "strengths": ["mathematical_reasoning", "statistical_analysis", "price_calculations"]
            },
            "yi_technical": {
                "specialization": "technical_analysis",
                "context_limit": 4096,
                "temperature": 0.2,
                "weight": 0.25,
                "strengths": ["pattern_recognition", "chart_analysis", "technical_indicators"]
            },
            "glm_risk": {
                "specialization": "risk_management",
                "context_limit": 8192,
                "temperature": 0.1,
                "weight": 0.25,
                "strengths": ["risk_assessment", "volatility_analysis", "drawdown_prediction"]
            },
            "deepseek_lite": {
                "specialization": "strategy_development",
                "context_limit": 16384,
                "temperature": 0.2,
                "weight": 0.2,
                "strengths": ["strategy_alignment", "trend_analysis", "market_timing"]
            }
        }

        logger.info("🧠 LLM Predictor initialized with Chinese LLM ensemble")

    def _convert_market_data_to_context(
        self,
        symbol: str,
        historical_data: pd.DataFrame,
        indicators: Dict[str, Any]
    ) -> MarketContext:
        """
        Convert numerical market data into natural language context
        This is crucial - LLMs understand narratives better than raw numbers
        """
        try:
            # Ensure we have data
            if historical_data.empty:
                raise ValueError("No historical data provided")

            recent_prices = historical_data['close'].tail(10)
            current_price = recent_prices.iloc[-1]
            price_change_pct = ((current_price - recent_prices.iloc[0]) / recent_prices.iloc[0]) * 100

            # Create price narrative
            if price_change_pct > 5:
                price_narrative = f"{symbol} has shown strong bullish momentum with {price_change_pct:.1f}% gain over recent sessions. Current price ${current_price:.2f} represents significant upward movement."
            elif price_change_pct < -5:
                price_narrative = f"{symbol} has experienced bearish pressure with {abs(price_change_pct):.1f}% decline. Current price ${current_price:.2f} shows weakness in recent trading."
            else:
                price_narrative = f"{symbol} has been consolidating around ${current_price:.2f} with {price_change_pct:.1f}% change, showing neutral price action."

            # Analyze trend using moving averages
            ma20 = indicators.get('ma20', current_price)
            ma50 = indicators.get('ma50', current_price)

            if current_price > ma20 > ma50:
                trend_analysis = "Strong uptrend confirmed with price above both 20-day and 50-day moving averages. Technical structure supports bullish continuation."
            elif current_price < ma20 < ma50:
                trend_analysis = "Bearish trend established with price below key moving averages. Technical structure suggests further downside risk."
            else:
                trend_analysis = "Mixed trend signals with price action around moving averages. Market direction uncertain, awaiting clearer signals."

            # Volatility assessment
            atr = indicators.get('atr', 0)
            volatility_assessment = f"Average True Range of {atr:.2f} indicates {'high' if atr > current_price * 0.03 else 'moderate' if atr > current_price * 0.015 else 'low'} volatility environment."

            # Volume analysis
            recent_volume = historical_data['volume'].tail(5).mean()
            avg_volume = historical_data['volume'].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

            if volume_ratio > 1.5:
                volume_story = "Above-average volume confirms price movement with strong participation. High conviction in current direction."
            elif volume_ratio < 0.5:
                volume_story = "Below-average volume suggests lack of conviction. Price moves may not be sustainable without volume confirmation."
            else:
                volume_story = "Normal volume levels indicate steady participation without excessive speculation."

            # Indicator summary
            rsi = indicators.get('rsi', 50)
            macd_signal = indicators.get('macd_signal', 'NEUTRAL')
            bb_position = indicators.get('bollinger_position', 'MIDDLE')

            indicator_summary = f"RSI at {rsi:.1f} suggests {'overbought conditions' if rsi > 70 else 'oversold conditions' if rsi < 30 else 'neutral momentum'}. "
            indicator_summary += f"MACD shows {macd_signal.lower()} signal. "
            indicator_summary += f"Price is at {bb_position.lower()} of Bollinger Bands."

            return MarketContext(
                price_narrative=price_narrative,
                trend_analysis=trend_analysis,
                volatility_assessment=volatility_assessment,
                volume_story=volume_story,
                indicator_summary=indicator_summary
            )

        except Exception as e:
            logger.error(f"Error converting market data to context: {e}")
            # Return neutral context as fallback
            return MarketContext(
                price_narrative=f"{symbol} price data analysis unavailable",
                trend_analysis="Trend analysis inconclusive",
                volatility_assessment="Volatility assessment pending",
                volume_story="Volume analysis unavailable",
                indicator_summary="Technical indicators pending"
            )

    def _build_specialized_prompt(
        self,
        model_key: str,
        symbol: str,
        market_context: MarketContext,
        prediction_type: str,
        horizon_days: int
    ) -> str:
        """Build specialized prompts for each Chinese LLM based on their strengths"""

        base_context = f"""
Market Analysis for {symbol}:

Price Action: {market_context.price_narrative}
Trend Analysis: {market_context.trend_analysis}
Volatility: {market_context.volatility_assessment}
Volume: {market_context.volume_story}
Technical Indicators: {market_context.indicator_summary}

Prediction Task: {prediction_type} over {horizon_days} days
"""

        if model_key == "qwen_quant":
            return f"""{base_context}

As a quantitative analyst, analyze this market data using mathematical and statistical reasoning. Focus on:
1. Statistical probability of price movements based on historical patterns
2. Quantitative risk-reward ratios
3. Mathematical trend strength calculations
4. Statistical volatility projections

Provide your prediction in this JSON format:
{{
    "prediction": "BULLISH|BEARISH|NEUTRAL",
    "confidence": 0.0-1.0,
    "reasoning": "detailed mathematical and statistical analysis",
    "key_factors": ["factor1", "factor2", "factor3"],
    "probability_estimates": {{"up": 0.0-1.0, "down": 0.0-1.0, "sideways": 0.0-1.0}}
}}"""

        elif model_key == "yi_technical":
            return f"""{base_context}

As a technical analyst, focus on pattern recognition and chart analysis. Analyze:
1. Technical patterns and formations
2. Support and resistance levels
3. Indicator convergence/divergence
4. Classical technical analysis signals

Provide your prediction in this JSON format:
{{
    "prediction": "BULLISH|BEARISH|NEUTRAL",
    "confidence": 0.0-1.0,
    "reasoning": "technical pattern and chart analysis",
    "key_patterns": ["pattern1", "pattern2"],
    "support_levels": ["level1", "level2"],
    "resistance_levels": ["level1", "level2"]
}}"""

        elif model_key == "glm_risk":
            return f"""{base_context}

As a risk manager, assess this position from a risk-management perspective. Focus on:
1. Downside risk assessment
2. Volatility impact on position sizing
3. Maximum drawdown possibilities
4. Risk-adjusted return expectations

Provide your prediction in this JSON format:
{{
    "prediction": "BULLISH|BEARISH|NEUTRAL",
    "confidence": 0.0-1.0,
    "reasoning": "risk-focused analysis with capital preservation priority",
    "risk_factors": ["factor1", "factor2"],
    "max_drawdown_estimate": "percentage",
    "position_sizing_advice": "recommendation"
}}"""

        else:  # deepseek_lite
            return f"""{base_context}

As a strategy developer, analyze this from a systematic trading strategy perspective. Focus on:
1. Strategic entry/exit timing
2. Market regime identification
3. Strategy alignment with market conditions
4. Systematic approach to position management

Provide your prediction in this JSON format:
{{
    "prediction": "BULLISH|BEARISH|NEUTRAL",
    "confidence": 0.0-1.0,
    "reasoning": "strategic analysis for systematic implementation",
    "strategy_alignment": "how this fits systematic approach",
    "entry_signals": ["signal1", "signal2"],
    "exit_conditions": ["condition1", "condition2"]
}}"""

    async def _get_model_prediction(
        self,
        model_key: str,
        prompt: str
    ) -> Dict[str, Any]:
        """Get prediction from a specific LLM model"""
        try:
            # Ensure model is loaded
            if not await self.ollama_client.ensure_model_loaded(model_key):
                logger.warning(f"Failed to load model {model_key}")
                return {"error": f"Model {model_key} not available"}

            # Get model config
            config = self.model_config[model_key]

            # Adjust context if needed
            max_tokens = min(config["context_limit"] // 2, 2048)  # Reserve context for response

            response = await self.ollama_client.generate_response(
                prompt=prompt,
                model_key=model_key,
                max_tokens=max_tokens
            )

            # Parse JSON response
            try:
                prediction_data = json.loads(response.strip())
                prediction_data["model_used"] = model_key
                prediction_data["specialization"] = config["specialization"]
                return prediction_data
            except json.JSONDecodeError:
                # Fallback parsing for non-JSON responses
                logger.warning(f"Non-JSON response from {model_key}, attempting fallback parsing")
                return self._fallback_parse_response(response, model_key, config)

        except Exception as e:
            logger.error(f"Error getting prediction from {model_key}: {e}")
            return {"error": str(e), "model_used": model_key}

    def _fallback_parse_response(
        self,
        response: str,
        model_key: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback parsing for non-JSON LLM responses"""

        # Simple keyword-based parsing
        response_lower = response.lower()

        if "bullish" in response_lower or "buy" in response_lower or "up" in response_lower:
            prediction = "BULLISH"
        elif "bearish" in response_lower or "sell" in response_lower or "down" in response_lower:
            prediction = "BEARISH"
        else:
            prediction = "NEUTRAL"

        # Estimate confidence based on language strength
        if any(word in response_lower for word in ["strong", "very", "highly", "confident"]):
            confidence = 0.8
        elif any(word in response_lower for word in ["likely", "probable", "expect"]):
            confidence = 0.6
        else:
            confidence = 0.4

        return {
            "prediction": prediction,
            "confidence": confidence,
            "reasoning": response[:500],  # Truncate long responses
            "model_used": model_key,
            "specialization": config["specialization"],
            "parsed_fallback": True
        }

    def _combine_predictions(
        self,
        model_predictions: Dict[str, Any],
        symbol: str,
        prediction_type: str,
        horizon_days: int,
        market_context: MarketContext
    ) -> PredictionResult:
        """Combine individual model predictions into ensemble result"""

        # Filter out error predictions
        valid_predictions = {k: v for k, v in model_predictions.items()
                           if "error" not in v and "prediction" in v}

        if not valid_predictions:
            logger.error("No valid predictions from any model")
            return PredictionResult(
                symbol=symbol,
                prediction_type=prediction_type,
                prediction_value="NEUTRAL",
                confidence=0.0,
                horizon_days=horizon_days,
                reasoning="No valid predictions available from ensemble models",
                model_contributions=model_predictions,
                market_context=market_context.__dict__,
                timestamp=datetime.now()
            )

        # Weighted voting based on model configurations
        prediction_scores = {"BULLISH": 0.0, "BEARISH": 0.0, "NEUTRAL": 0.0}
        total_weighted_confidence = 0.0
        total_weight = 0.0

        for model_key, prediction_data in valid_predictions.items():
            model_weight = self.model_config[model_key]["weight"]
            model_prediction = prediction_data["prediction"]
            model_confidence = prediction_data.get("confidence", 0.5)

            # Apply weighted confidence to prediction scores
            weighted_confidence = model_confidence * model_weight
            prediction_scores[model_prediction] += weighted_confidence

            total_weighted_confidence += weighted_confidence
            total_weight += model_weight

        # Determine ensemble prediction
        final_prediction = max(prediction_scores, key=prediction_scores.get)
        final_confidence = prediction_scores[final_prediction]

        # Normalize confidence
        if total_weight > 0:
            final_confidence = final_confidence / total_weight

        # Build comprehensive reasoning
        reasoning_parts = []
        for model_key, prediction_data in valid_predictions.items():
            specialization = prediction_data.get("specialization", model_key)
            model_reasoning = prediction_data.get("reasoning", "No reasoning provided")
            reasoning_parts.append(f"{specialization.title()}: {model_reasoning}")

        ensemble_reasoning = f"Ensemble Analysis ({len(valid_predictions)} models): " + " | ".join(reasoning_parts)

        return PredictionResult(
            symbol=symbol,
            prediction_type=prediction_type,
            prediction_value=final_prediction,
            confidence=final_confidence,
            horizon_days=horizon_days,
            reasoning=ensemble_reasoning,
            model_contributions=model_predictions,
            market_context=market_context.__dict__,
            timestamp=datetime.now()
        )

    async def predict_price_direction(
        self,
        symbol: str,
        historical_data: pd.DataFrame,
        indicators: Optional[Dict[str, Any]] = None,
        horizon_days: int = 5
    ) -> PredictionResult:
        """
        Predict price direction using ensemble of Chinese LLMs

        Args:
            symbol: Trading symbol
            historical_data: DataFrame with OHLCV data
            indicators: Pre-calculated technical indicators
            horizon_days: Prediction horizon in days

        Returns:
            PredictionResult with ensemble prediction
        """
        try:
            # Calculate indicators if not provided
            if indicators is None:
                indicators = self.technical_indicators.calculate_all_indicators(historical_data)

            # Convert market data to natural language context
            market_context = self._convert_market_data_to_context(
                symbol, historical_data, indicators
            )

            # Get predictions from all models in parallel
            prediction_tasks = []
            for model_key in self.model_config.keys():
                prompt = self._build_specialized_prompt(
                    model_key, symbol, market_context, "price_direction", horizon_days
                )
                task = self._get_model_prediction(model_key, prompt)
                prediction_tasks.append((model_key, task))

            # Execute all predictions concurrently
            model_predictions = {}
            for model_key, task in prediction_tasks:
                try:
                    prediction = await task
                    model_predictions[model_key] = prediction
                except Exception as e:
                    logger.error(f"Error in {model_key} prediction: {e}")
                    model_predictions[model_key] = {"error": str(e)}

            # Combine predictions into ensemble result
            result = self._combine_predictions(
                model_predictions, symbol, "price_direction", horizon_days, market_context
            )

            logger.info(f"🔮 LLM Ensemble Prediction for {symbol}: {result.prediction_value} "
                       f"(confidence: {result.confidence:.2f})")

            return result

        except Exception as e:
            logger.error(f"Error in predict_price_direction: {e}")
            return PredictionResult(
                symbol=symbol,
                prediction_type="price_direction",
                prediction_value="NEUTRAL",
                confidence=0.0,
                horizon_days=horizon_days,
                reasoning=f"Error in prediction: {str(e)}",
                model_contributions={},
                market_context={},
                timestamp=datetime.now()
            )

    async def predict_volatility(
        self,
        symbol: str,
        historical_data: pd.DataFrame,
        indicators: Optional[Dict[str, Any]] = None,
        horizon_days: int = 5
    ) -> PredictionResult:
        """Predict volatility using ensemble focused on GLM and Qwen models"""

        # Similar implementation to predict_price_direction but focused on volatility
        # Using specialized prompts for volatility prediction
        try:
            if indicators is None:
                indicators = self.technical_indicators.calculate_all_indicators(historical_data)

            market_context = self._convert_market_data_to_context(
                symbol, historical_data, indicators
            )

            # Focus on models better suited for volatility (GLM for risk, Qwen for quant)
            priority_models = ["glm_risk", "qwen_quant"]
            model_predictions = {}

            for model_key in priority_models:
                prompt = self._build_specialized_prompt(
                    model_key, symbol, market_context, "volatility_forecast", horizon_days
                )
                prediction = await self._get_model_prediction(model_key, prompt)
                model_predictions[model_key] = prediction

            result = self._combine_predictions(
                model_predictions, symbol, "volatility_forecast", horizon_days, market_context
            )

            return result

        except Exception as e:
            logger.error(f"Error in predict_volatility: {e}")
            return PredictionResult(
                symbol=symbol,
                prediction_type="volatility_forecast",
                prediction_value="MODERATE",
                confidence=0.0,
                horizon_days=horizon_days,
                reasoning=f"Error in volatility prediction: {str(e)}",
                model_contributions={},
                market_context={},
                timestamp=datetime.now()
            )

    async def get_model_health(self) -> Dict[str, Any]:
        """Check health and availability of all Chinese LLM models"""
        health_status = {}

        for model_key, config in self.model_config.items():
            try:
                is_loaded = await self.ollama_client.ensure_model_loaded(model_key)
                model_info = self.ollama_client.get_model_info(model_key)

                health_status[model_key] = {
                    "available": is_loaded,
                    "specialization": config["specialization"],
                    "context_limit": config["context_limit"],
                    "memory_usage_mb": model_info.memory_usage_mb if model_info else 0,
                    "status": "healthy" if is_loaded else "unavailable"
                }

            except Exception as e:
                health_status[model_key] = {
                    "available": False,
                    "status": "error",
                    "error": str(e)
                }

        # Overall system health
        available_models = sum(1 for status in health_status.values() if status.get("available", False))
        total_models = len(self.model_config)

        return {
            "overall_status": "healthy" if available_models >= 2 else "degraded" if available_models >= 1 else "critical",
            "available_models": available_models,
            "total_models": total_models,
            "model_details": health_status,
            "memory_usage": self.ollama_client.get_total_memory_usage(),
            "timestamp": datetime.now()
        }


# Factory function for easy instantiation
async def get_llm_predictor() -> LLMPredictor:
    """Get configured LLM predictor instance"""
    predictor = LLMPredictor()

    # Verify model availability
    health = await predictor.get_model_health()
    if health["available_models"] == 0:
        logger.warning("No Chinese LLM models available - predictions will be limited")

    return predictor


# Convenience function for quick predictions
async def predict_symbol_direction(
    symbol: str,
    historical_data: pd.DataFrame,
    horizon_days: int = 5
) -> Dict[str, Any]:
    """Quick prediction for a symbol"""
    predictor = await get_llm_predictor()
    result = await predictor.predict_price_direction(symbol, historical_data, horizon_days=horizon_days)

    return {
        "symbol": result.symbol,
        "prediction": result.prediction_value,
        "confidence": result.confidence,
        "reasoning": result.reasoning[:200] + "..." if len(result.reasoning) > 200 else result.reasoning,
        "horizon_days": result.horizon_days,
        "timestamp": result.timestamp.isoformat()
    }


if __name__ == "__main__":
    # Example usage
    import yfinance as yf

    async def test_predictions():
        # Download sample data
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="3mo")
        data = data.reset_index()
        data.columns = [col.lower() for col in data.columns]

        # Test prediction
        result = await predict_symbol_direction("AAPL", data)
        print(json.dumps(result, indent=2, default=str))

    # Run test
    asyncio.run(test_predictions())