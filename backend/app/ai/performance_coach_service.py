"""
Performance Coach Service - Extracted from trading_agents.py

Specialized service for trade review and system improvement
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from .base_agent import BaseAIAgent
from .ollama_client import OllamaClient


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


class PerformanceCoachService(BaseAIAgent):
    """AI agent specialized in trade review and system improvement"""

    def __init__(self, ollama_client: OllamaClient):
        super().__init__(ollama_client, "analyst", "trade_review.txt")

    def _get_default_prompt(self) -> str:
        return (
            "You are a performance coach. Review trades and provide "
            "improvement insights. Focus on identifying patterns, "
            "execution quality, and systematic improvements."
        )

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

            # Build data sections
            data_sections = {
                "Trade Details": {
                    "Symbol": trade_data.get("symbol", "Unknown"),
                    "Entry Price": f"${entry_price}",
                    "Exit Price": f"${exit_price}",
                    "Quantity": quantity,
                    "P&L": f"${pnl:.2f} ({pnl_percent:.2f}%)",
                    "Duration": trade_data.get("duration", "Unknown"),
                    "Entry Signal Confidence": trade_data.get(
                        "entry_confidence", "N/A"
                    ),
                    "Exit Reason": trade_data.get("exit_reason", "Unknown"),
                },
                "Market Context During Trade": json.dumps(market_context, indent=2),
                "Current System Performance": {
                    "Win Rate": f"{system_performance.get('win_rate', 0.0)}%",
                    "Average Win": f"{system_performance.get('avg_win', 0.0)}%",
                    "Average Loss": f"{system_performance.get('avg_loss', 0.0)}%",
                    "Sharpe Ratio": system_performance.get("sharpe", 0.0),
                },
            }

            # JSON schema
            json_schema = {
                "performance_grade": "A|B|C|D|F",
                "execution_quality": "excellent|good|fair|poor",
                "key_learnings": ["learning1", "learning2", "learning3"],
                "improvement_suggestions": ["suggestion1", "suggestion2"],
                "pattern_insights": ["insight1", "insight2"],
                "systematic_improvements": ["improvement1", "improvement2"],
            }

            # Build prompt
            instruction = (
                "Provide structured review and improvement insights for this trade"
            )
            prompt = self._build_standard_prompt_template(
                trade_data.get("symbol", "Unknown"),
                data_sections,
                instruction,
                json_schema,
            )

            # Generate response
            response = await self._generate_response(prompt, max_tokens=1024)

            # Parse response with fallback defaults
            default_values = {
                "performance_grade": "C",
                "execution_quality": "fair",
                "key_learnings": ["Review analysis failed"],
                "improvement_suggestions": ["Retry trade review"],
                "pattern_insights": ["Unable to identify patterns"],
                "systematic_improvements": ["Improve review process"],
            }

            review_data = self._parse_json_response(response, default_values)

            # Validate and normalize data
            performance_grade = self._validate_choice(
                review_data["performance_grade"], ["A", "B", "C", "D", "F"], "C"
            )

            execution_quality = self._validate_choice(
                review_data["execution_quality"],
                ["excellent", "good", "fair", "poor"],
                "fair",
            )

            return TradeReview(
                trade_id=trade_data.get("id", "unknown"),
                symbol=trade_data.get("symbol", "Unknown"),
                performance_grade=performance_grade,
                execution_quality=execution_quality,
                key_learnings=review_data.get("key_learnings", []),
                improvement_suggestions=review_data.get("improvement_suggestions", []),
                pattern_insights=review_data.get("pattern_insights", []),
                systematic_improvements=review_data.get("systematic_improvements", []),
                timestamp=datetime.now(),
            )

        except Exception as e:
            self.error_count += 1
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

    async def analyze_performance_patterns(
        self, trade_history: List[Dict[str, Any]], time_period_days: int = 30
    ) -> Dict[str, Any]:
        """Analyze performance patterns across multiple trades"""
        try:
            if not trade_history:
                return {"error": "No trade history provided"}

            # Aggregate trade data for pattern analysis
            total_trades = len(trade_history)
            winning_trades = [t for t in trade_history if t.get("pnl", 0) > 0]
            losing_trades = [t for t in trade_history if t.get("pnl", 0) < 0]

            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            avg_win = (
                sum(t.get("pnl", 0) for t in winning_trades) / len(winning_trades)
                if winning_trades
                else 0
            )
            avg_loss = (
                sum(t.get("pnl", 0) for t in losing_trades) / len(losing_trades)
                if losing_trades
                else 0
            )

            # Build analysis data
            analysis_data = {
                "Period Analysis": f"{time_period_days} days",
                "Trade Statistics": {
                    "Total Trades": total_trades,
                    "Winning Trades": len(winning_trades),
                    "Losing Trades": len(losing_trades),
                    "Win Rate": f"{win_rate:.2%}",
                    "Average Win": f"${avg_win:.2f}",
                    "Average Loss": f"${avg_loss:.2f}",
                },
                "Recent Trades Sample": json.dumps(
                    trade_history[-10:], indent=2, default=str
                ),
            }

            json_schema = {
                "performance_trends": ["trend1", "trend2"],
                "strength_areas": ["strength1", "strength2"],
                "weakness_areas": ["weakness1", "weakness2"],
                "strategic_recommendations": ["recommendation1", "recommendation2"],
                "risk_management_insights": ["insight1", "insight2"],
                "overall_assessment": "improving|stable|declining",
            }

            instruction = "Analyze performance patterns and provide strategic insights"
            prompt = self._build_standard_prompt_template(
                "Portfolio", analysis_data, instruction, json_schema
            )

            response = await self._generate_response(prompt, max_tokens=1024)

            default_values = {
                "performance_trends": ["Unable to identify trends"],
                "strength_areas": ["Analysis incomplete"],
                "weakness_areas": ["Analysis incomplete"],
                "strategic_recommendations": ["Require more data"],
                "risk_management_insights": ["Analysis incomplete"],
                "overall_assessment": "stable",
            }

            pattern_data = self._parse_json_response(response, default_values)

            return {
                "analysis_period_days": time_period_days,
                "trade_count": total_trades,
                "win_rate": win_rate,
                "performance_trends": pattern_data.get("performance_trends", []),
                "strength_areas": pattern_data.get("strength_areas", []),
                "weakness_areas": pattern_data.get("weakness_areas", []),
                "strategic_recommendations": pattern_data.get(
                    "strategic_recommendations", []
                ),
                "risk_management_insights": pattern_data.get(
                    "risk_management_insights", []
                ),
                "overall_assessment": self._validate_choice(
                    pattern_data.get("overall_assessment", "stable"),
                    ["improving", "stable", "declining"],
                    "stable",
                ),
                "timestamp": datetime.now(),
            }

        except Exception as e:
            self.error_count += 1
            return {
                "error": str(e),
                "analysis_period_days": time_period_days,
                "timestamp": datetime.now(),
            }

    async def process(
        self,
        trade_data: Dict[str, Any],
        market_context: Dict[str, Any],
        system_performance: Dict[str, Any],
        **kwargs,
    ) -> TradeReview:
        """Main processing method for BaseAIAgent interface"""
        return await self.review_trade(trade_data, market_context, system_performance)
