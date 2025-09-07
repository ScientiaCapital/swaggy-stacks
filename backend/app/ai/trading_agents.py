"""
Core AI trading agents for Swaggy Stacks Trading System
"""

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import structlog

from .ollama_client import OllamaClient

logger = structlog.get_logger()


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


@dataclass
class RiskAssessment:
    """Risk assessment result from AI agent"""

    symbol: str
    risk_level: str  # low, medium, high
    portfolio_heat: float  # current portfolio heat percentage
    recommended_position_size: float
    key_risk_factors: List[str]
    mitigation_strategies: List[str]
    exit_conditions: List[str]
    max_position_risk: float
    timestamp: datetime


@dataclass
class StrategySignal:
    """Strategy signal from AI agent"""

    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: Optional[float]
    reasoning: str
    technical_factors: List[str]
    timestamp: datetime


@dataclass
class TradeReview:
    """Trade review result from AI agent"""

    trade_id: str
    symbol: str
    performance_grade: str  # A, B, C, D, F
    execution_quality: str  # excellent, good, fair, poor
    key_learnings: List[str]
    improvement_suggestions: List[str]
    pattern_insights: List[str]
    systematic_improvements: List[str]
    timestamp: datetime


class MarketAnalystAgent:
    """AI agent specialized in market analysis and sentiment evaluation"""

    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.agent_type = "analyst"
        self.system_prompt = self._load_prompt("market_analysis.txt")

    def _load_prompt(self, filename: str) -> str:
        """Load system prompt from file"""
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), "prompts", filename)
            with open(prompt_path, "r") as f:
                return f.read().strip()
        except Exception as e:
            logger.error("Failed to load prompt", filename=filename, error=str(e))
            return (
                "You are a market analysis expert. Provide clear, actionable insights."
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
            prompt = f"""
            Analyze the market conditions for {symbol}:
            
            Market Data:
            - Current Price: ${market_data.get('current_price', 'N/A')}
            - Volume: {market_data.get('volume', 'N/A')}
            - 52W High: ${market_data.get('high_52w', 'N/A')}
            - 52W Low: ${market_data.get('low_52w', 'N/A')}
            - Market Cap: {market_data.get('market_cap', 'N/A')}
            
            Technical Indicators:
            - RSI: {technical_indicators.get('rsi', 'N/A')}
            - MACD: {technical_indicators.get('macd', 'N/A')}
            - MA20: ${technical_indicators.get('ma20', 'N/A')}
            - MA50: ${technical_indicators.get('ma50', 'N/A')}
            - Bollinger Bands: {technical_indicators.get('bollinger_bands', 'N/A')}
            - ATR: {technical_indicators.get('atr', 'N/A')}
            
            Additional Context:
            {context}
            
            Provide your analysis in this exact JSON format:
            {{
                "sentiment": "bullish|bearish|neutral",
                "confidence": 0.0-1.0,
                "key_factors": ["factor1", "factor2", "factor3"],
                "recommendations": ["rec1", "rec2"],
                "risk_level": "low|medium|high",
                "reasoning": "detailed explanation of your analysis"
            }}
            """

            response = await self.ollama_client.generate_response(
                prompt,
                model_key=self.agent_type,
                system_prompt=self.system_prompt,
                max_tokens=1024,
            )

            # Parse JSON response
            analysis_data = self._parse_json_response(
                response,
                {
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "key_factors": ["Unable to analyze"],
                    "recommendations": ["Review data quality"],
                    "risk_level": "medium",
                    "reasoning": "Analysis failed",
                },
            )

            return MarketAnalysis(
                symbol=symbol,
                sentiment=analysis_data["sentiment"],
                confidence=float(analysis_data["confidence"]),
                key_factors=analysis_data["key_factors"],
                recommendations=analysis_data["recommendations"],
                risk_level=analysis_data["risk_level"],
                reasoning=analysis_data["reasoning"],
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error("Market analysis failed", symbol=symbol, error=str(e))
            return MarketAnalysis(
                symbol=symbol,
                sentiment="neutral",
                confidence=0.0,
                key_factors=["Analysis error"],
                recommendations=["Retry analysis"],
                risk_level="high",
                reasoning=f"Error: {str(e)}",
                timestamp=datetime.now(),
            )

    def _parse_json_response(self, response: str, default_values: Dict) -> Dict:
        """Parse JSON response with fallback to default values"""
        try:
            # Clean response - remove markdown code blocks if present
            clean_response = response.strip()
            if clean_response.startswith("```"):
                lines = clean_response.split("\n")
                clean_response = "\n".join(lines[1:-1])

            return json.loads(clean_response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response", response=response[:200])
            return default_values


class RiskAdvisorAgent:
    """AI agent specialized in risk assessment and portfolio protection"""

    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.agent_type = "risk"
        self.system_prompt = self._load_prompt("risk_assessment.txt")

    def _load_prompt(self, filename: str) -> str:
        """Load system prompt from file"""
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), "prompts", filename)
            with open(prompt_path, "r") as f:
                return f.read().strip()
        except Exception as e:
            logger.error("Failed to load prompt", filename=filename, error=str(e))
            return "You are a risk management expert. Focus on capital preservation."

    async def assess_risk(
        self,
        symbol: str,
        position_size: float,
        account_value: float,
        current_positions: List[Dict],
        market_volatility: Dict[str, float],
        proposed_trade: Dict[str, Any],
    ) -> RiskAssessment:
        """Assess risk for a proposed trade"""

        try:
            # Calculate current portfolio heat
            total_risk = sum(pos.get("risk_amount", 0) for pos in current_positions)
            portfolio_heat = (
                (total_risk / account_value) * 100 if account_value > 0 else 0
            )

            # Calculate proposed position risk
            proposed_risk = position_size * proposed_trade.get(
                "stop_loss_percent", 0.05
            )
            position_risk_percent = (proposed_risk / account_value) * 100

            prompt = f"""
            Assess the risk for this proposed trade:
            
            Symbol: {symbol}
            Proposed Position Size: ${position_size:,.2f}
            Account Value: ${account_value:,.2f}
            Current Portfolio Heat: {portfolio_heat:.2f}%
            Proposed Position Risk: {position_risk_percent:.2f}%
            
            Current Positions:
            {json.dumps(current_positions, indent=2)}
            
            Market Volatility:
            - VIX: {market_volatility.get('vix', 'N/A')}
            - Symbol ATR: {market_volatility.get('atr', 'N/A')}
            - Historical Volatility: {market_volatility.get('hist_vol', 'N/A')}
            
            Proposed Trade Details:
            {json.dumps(proposed_trade, indent=2)}
            
            Provide risk assessment in this JSON format:
            {{
                "risk_level": "low|medium|high",
                "portfolio_heat": {portfolio_heat},
                "recommended_position_size": 0.0,
                "key_risk_factors": ["factor1", "factor2"],
                "mitigation_strategies": ["strategy1", "strategy2"],
                "exit_conditions": ["condition1", "condition2"],
                "max_position_risk": 0.5
            }}
            """

            response = await self.ollama_client.generate_response(
                prompt,
                model_key=self.agent_type,
                system_prompt=self.system_prompt,
                max_tokens=1024,
            )

            risk_data = self._parse_json_response(
                response,
                {
                    "risk_level": "high",
                    "portfolio_heat": portfolio_heat,
                    "recommended_position_size": position_size * 0.5,
                    "key_risk_factors": ["High uncertainty"],
                    "mitigation_strategies": ["Reduce position size"],
                    "exit_conditions": ["Stop loss triggered"],
                    "max_position_risk": 0.5,
                },
            )

            return RiskAssessment(
                symbol=symbol,
                risk_level=risk_data["risk_level"],
                portfolio_heat=float(risk_data["portfolio_heat"]),
                recommended_position_size=float(risk_data["recommended_position_size"]),
                key_risk_factors=risk_data["key_risk_factors"],
                mitigation_strategies=risk_data["mitigation_strategies"],
                exit_conditions=risk_data["exit_conditions"],
                max_position_risk=float(risk_data["max_position_risk"]),
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error("Risk assessment failed", symbol=symbol, error=str(e))
            return RiskAssessment(
                symbol=symbol,
                risk_level="high",
                portfolio_heat=portfolio_heat,
                recommended_position_size=position_size * 0.25,
                key_risk_factors=["Assessment error"],
                mitigation_strategies=["Manual review required"],
                exit_conditions=["Immediate exit if uncertain"],
                max_position_risk=0.25,
                timestamp=datetime.now(),
            )

    def _parse_json_response(self, response: str, default_values: Dict) -> Dict:
        """Parse JSON response with fallback"""
        try:
            clean_response = response.strip()
            if clean_response.startswith("```"):
                lines = clean_response.split("\n")
                clean_response = "\n".join(lines[1:-1])
            return json.loads(clean_response)
        except json.JSONDecodeError:
            return default_values


class StrategyOptimizerAgent:
    """AI agent specialized in strategy generation and optimization"""

    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.agent_type = "strategist"
        self.system_prompt = self._load_prompt("strategy_generation.txt")

    def _load_prompt(self, filename: str) -> str:
        """Load system prompt from file"""
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), "prompts", filename)
            with open(prompt_path, "r") as f:
                return f.read().strip()
        except Exception as e:
            logger.error("Failed to load prompt", filename=filename, error=str(e))
            return "You are a strategy optimization expert. Generate actionable trading signals."

    async def generate_signal(
        self,
        symbol: str,
        markov_analysis: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        market_context: Dict[str, Any],
        performance_history: List[Dict],
    ) -> StrategySignal:
        """Generate optimized trading signal"""

        try:
            prompt = f"""
            Generate an optimized trading signal for {symbol}:
            
            Enhanced Markov Analysis:
            - Current State: {markov_analysis.get('current_state', 'Unknown')}
            - Transition Probability: {markov_analysis.get('transition_prob', 0.0)}
            - Confidence: {markov_analysis.get('confidence', 0.0)}
            - Predicted Direction: {markov_analysis.get('direction', 'Neutral')}
            
            Technical Indicators:
            {json.dumps(technical_indicators, indent=2)}
            
            Market Context:
            - Market Regime: {market_context.get('regime', 'Unknown')}
            - Volatility Level: {market_context.get('volatility', 'Normal')}
            - Trend Strength: {market_context.get('trend_strength', 'Moderate')}
            
            Recent Performance History:
            {json.dumps(performance_history[-5:] if performance_history else [], indent=2)}
            
            Generate signal in this JSON format:
            {{
                "action": "BUY|SELL|HOLD",
                "confidence": 0.0-1.0,
                "entry_price": null,
                "stop_loss": null,
                "take_profit": null,
                "position_size": null,
                "reasoning": "detailed explanation",
                "technical_factors": ["factor1", "factor2", "factor3"]
            }}
            """

            response = await self.ollama_client.generate_response(
                prompt,
                model_key=self.agent_type,
                system_prompt=self.system_prompt,
                max_tokens=1024,
            )

            signal_data = self._parse_json_response(
                response,
                {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "entry_price": None,
                    "stop_loss": None,
                    "take_profit": None,
                    "position_size": None,
                    "reasoning": "Signal generation failed",
                    "technical_factors": ["Error in analysis"],
                },
            )

            return StrategySignal(
                symbol=symbol,
                action=signal_data["action"],
                confidence=float(signal_data["confidence"]),
                entry_price=signal_data.get("entry_price"),
                stop_loss=signal_data.get("stop_loss"),
                take_profit=signal_data.get("take_profit"),
                position_size=signal_data.get("position_size"),
                reasoning=signal_data["reasoning"],
                technical_factors=signal_data["technical_factors"],
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error("Signal generation failed", symbol=symbol, error=str(e))
            return StrategySignal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                position_size=None,
                reasoning=f"Error: {str(e)}",
                technical_factors=["Generation error"],
                timestamp=datetime.now(),
            )

    def _parse_json_response(self, response: str, default_values: Dict) -> Dict:
        """Parse JSON response with fallback"""
        try:
            clean_response = response.strip()
            if clean_response.startswith("```"):
                lines = clean_response.split("\n")
                clean_response = "\n".join(lines[1:-1])
            return json.loads(clean_response)
        except json.JSONDecodeError:
            return default_values


class PerformanceCoachAgent:
    """AI agent specialized in trade review and system improvement"""

    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self.agent_type = "analyst"  # Use analyst model for performance analysis
        self.system_prompt = self._load_prompt("trade_review.txt")

    def _load_prompt(self, filename: str) -> str:
        """Load system prompt from file"""
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), "prompts", filename)
            with open(prompt_path, "r") as f:
                return f.read().strip()
        except Exception as e:
            logger.error("Failed to load prompt", filename=filename, error=str(e))
            return "You are a performance coach. Review trades and provide improvement insights."

    async def review_trade(
        self,
        trade_data: Dict[str, Any],
        market_context: Dict[str, Any],
        system_performance: Dict[str, Any],
    ) -> TradeReview:
        """Review a completed trade and provide insights"""

        try:
            # Calculate trade metrics
            entry_price = trade_data.get("entry_price", 0)
            exit_price = trade_data.get("exit_price", 0)
            quantity = trade_data.get("quantity", 0)
            pnl = (exit_price - entry_price) * quantity
            pnl_percent = (
                ((exit_price - entry_price) / entry_price) * 100
                if entry_price > 0
                else 0
            )

            prompt = f"""
            Review this completed trade:
            
            Trade Details:
            - Symbol: {trade_data.get('symbol', 'Unknown')}
            - Entry Price: ${entry_price}
            - Exit Price: ${exit_price}
            - Quantity: {quantity}
            - P&L: ${pnl:.2f} ({pnl_percent:.2f}%)
            - Duration: {trade_data.get('duration', 'Unknown')}
            - Entry Signal Confidence: {trade_data.get('entry_confidence', 'N/A')}
            - Exit Reason: {trade_data.get('exit_reason', 'Unknown')}
            
            Market Context During Trade:
            {json.dumps(market_context, indent=2)}
            
            Current System Performance:
            - Win Rate: {system_performance.get('win_rate', 0.0)}%
            - Average Win: {system_performance.get('avg_win', 0.0)}%
            - Average Loss: {system_performance.get('avg_loss', 0.0)}%
            - Sharpe Ratio: {system_performance.get('sharpe', 0.0)}
            
            Provide structured review in this JSON format:
            {{
                "performance_grade": "A|B|C|D|F",
                "execution_quality": "excellent|good|fair|poor",
                "key_learnings": ["learning1", "learning2", "learning3"],
                "improvement_suggestions": ["suggestion1", "suggestion2"],
                "pattern_insights": ["insight1", "insight2"],
                "systematic_improvements": ["improvement1", "improvement2"]
            }}
            """

            response = await self.ollama_client.generate_response(
                prompt,
                model_key=self.agent_type,
                system_prompt=self.system_prompt,
                max_tokens=1024,
            )

            review_data = self._parse_json_response(
                response,
                {
                    "performance_grade": "C",
                    "execution_quality": "fair",
                    "key_learnings": ["Review analysis failed"],
                    "improvement_suggestions": ["Retry trade review"],
                    "pattern_insights": ["Unable to identify patterns"],
                    "systematic_improvements": ["Improve review process"],
                },
            )

            return TradeReview(
                trade_id=trade_data.get("id", "unknown"),
                symbol=trade_data.get("symbol", "Unknown"),
                performance_grade=review_data["performance_grade"],
                execution_quality=review_data["execution_quality"],
                key_learnings=review_data["key_learnings"],
                improvement_suggestions=review_data["improvement_suggestions"],
                pattern_insights=review_data["pattern_insights"],
                systematic_improvements=review_data["systematic_improvements"],
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(
                "Trade review failed", trade_id=trade_data.get("id"), error=str(e)
            )
            return TradeReview(
                trade_id=trade_data.get("id", "unknown"),
                symbol=trade_data.get("symbol", "Unknown"),
                performance_grade="F",
                execution_quality="poor",
                key_learnings=[f"Review error: {str(e)}"],
                improvement_suggestions=["Fix review process"],
                pattern_insights=["Unable to analyze"],
                systematic_improvements=["Debug review system"],
                timestamp=datetime.now(),
            )

    def _parse_json_response(self, response: str, default_values: Dict) -> Dict:
        """Parse JSON response with fallback"""
        try:
            clean_response = response.strip()
            if clean_response.startswith("```"):
                lines = clean_response.split("\n")
                clean_response = "\n".join(lines[1:-1])
            return json.loads(clean_response)
        except json.JSONDecodeError:
            return default_values


class AIAgentCoordinator:
    """Coordinates all AI agents for comprehensive trading intelligence"""

    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_client = OllamaClient(ollama_base_url)
        self.market_analyst = MarketAnalystAgent(self.ollama_client)
        self.risk_advisor = RiskAdvisorAgent(self.ollama_client)
        self.strategy_optimizer = StrategyOptimizerAgent(self.ollama_client)
        self.performance_coach = PerformanceCoachAgent(self.ollama_client)

    async def comprehensive_analysis(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        account_info: Dict[str, Any],
        current_positions: List[Dict],
        markov_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run comprehensive analysis using all agents"""

        try:
            logger.info("Starting comprehensive AI analysis", symbol=symbol)

            # Run market analysis
            market_analysis = await self.market_analyst.analyze_market(
                symbol=symbol,
                market_data=market_data,
                technical_indicators=technical_indicators,
            )

            # Calculate proposed position size (simplified)
            account_value = account_info.get("equity", 100000)
            proposed_position_size = account_value * 0.02  # 2% of account

            # Run risk assessment
            risk_assessment = await self.risk_advisor.assess_risk(
                symbol=symbol,
                position_size=proposed_position_size,
                account_value=account_value,
                current_positions=current_positions,
                market_volatility={
                    "atr": technical_indicators.get("atr", 0),
                    "hist_vol": market_data.get("volatility", 0),
                },
                proposed_trade={"stop_loss_percent": 0.05, "take_profit_percent": 0.10},
            )

            # Generate optimized signal
            strategy_signal = await self.strategy_optimizer.generate_signal(
                symbol=symbol,
                markov_analysis=markov_analysis,
                technical_indicators=technical_indicators,
                market_context={
                    "regime": "trending",  # This would come from regime detection
                    "volatility": "normal",
                    "trend_strength": "moderate",
                },
                performance_history=[],  # This would come from trade history
            )

            # Compile comprehensive result
            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "market_analysis": asdict(market_analysis),
                "risk_assessment": asdict(risk_assessment),
                "strategy_signal": asdict(strategy_signal),
                "final_recommendation": self._synthesize_recommendation(
                    market_analysis, risk_assessment, strategy_signal
                ),
            }

            logger.info("Comprehensive analysis completed", symbol=symbol)
            return result

        except Exception as e:
            logger.error("Comprehensive analysis failed", symbol=symbol, error=str(e))
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "final_recommendation": "HOLD",
            }

    def _synthesize_recommendation(
        self,
        market_analysis: MarketAnalysis,
        risk_assessment: RiskAssessment,
        strategy_signal: StrategySignal,
    ) -> str:
        """Synthesize final recommendation from all agents"""

        # Weight the recommendations
        if risk_assessment.risk_level == "high":
            return "HOLD"  # Risk trumps everything

        if market_analysis.confidence < 0.6 or strategy_signal.confidence < 0.6:
            return "HOLD"  # Need high confidence for action

        # If both market and strategy agree, follow their recommendation
        if market_analysis.sentiment in ["bullish"] and strategy_signal.action == "BUY":
            return "BUY"
        elif (
            market_analysis.sentiment in ["bearish"]
            and strategy_signal.action == "SELL"
        ):
            return "SELL"
        else:
            return "HOLD"

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all AI components"""
        return await self.ollama_client.health_check()
