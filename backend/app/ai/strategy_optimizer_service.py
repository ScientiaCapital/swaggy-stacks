"""
Strategy Optimizer Service - Extracted from trading_agents.py

Specialized service for strategy generation and optimization
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_agent import BaseAIAgent
from .ollama_client import OllamaClient


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


class StrategyOptimizerService(BaseAIAgent):
    """AI agent specialized in strategy generation and optimization"""

    def __init__(self, ollama_client: OllamaClient):
        super().__init__(ollama_client, "strategist", "strategy_generation.txt")

    def _get_default_prompt(self) -> str:
        return (
            "You are a strategy optimization expert. Generate actionable "
            "trading signals based on technical analysis, market conditions, "
            "and historical performance patterns."
        )

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
            # Build data sections
            data_sections = {
                "Markov Analysis": {
                    "Current State": markov_analysis.get("current_state", "Unknown"),
                    "Transition Probability": markov_analysis.get(
                        "transition_prob", 0.0
                    ),
                    "Confidence": markov_analysis.get("confidence", 0.0),
                    "Predicted Direction": markov_analysis.get("direction", "Neutral"),
                },
                "Technical Indicators": json.dumps(technical_indicators, indent=2),
                "Market Context": {
                    "Market Regime": market_context.get("regime", "Unknown"),
                    "Volatility Level": market_context.get("volatility", "Normal"),
                    "Trend Strength": market_context.get("trend_strength", "Moderate"),
                },
                "Recent Performance History": json.dumps(
                    performance_history[-5:] if performance_history else [], indent=2
                ),
            }

            # JSON schema
            json_schema = {
                "action": "BUY|SELL|HOLD",
                "confidence": "0.0-1.0",
                "entry_price": "float or null",
                "stop_loss": "float or null",
                "take_profit": "float or null",
                "position_size": "float or null",
                "reasoning": "detailed explanation",
                "technical_factors": ["factor1", "factor2", "factor3"],
            }

            # Build prompt
            instruction = (
                "Generate an optimized trading signal based on all available data"
            )
            prompt = self._build_standard_prompt_template(
                symbol, data_sections, instruction, json_schema
            )

            # Generate response
            response = await self._generate_response(prompt, max_tokens=1024)

            # Parse response with fallback defaults
            default_values = {
                "action": "HOLD",
                "confidence": 0.0,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "position_size": None,
                "reasoning": "Signal generation failed",
                "technical_factors": ["Error in analysis"],
            }

            signal_data = self._parse_json_response(response, default_values)

            # Validate and normalize data
            action = self._validate_choice(
                signal_data["action"], ["BUY", "SELL", "HOLD"], "HOLD"
            )

            confidence = self._validate_confidence(signal_data["confidence"])

            # Validate numeric fields
            entry_price = self._parse_optional_float(signal_data.get("entry_price"))
            stop_loss = self._parse_optional_float(signal_data.get("stop_loss"))
            take_profit = self._parse_optional_float(signal_data.get("take_profit"))
            position_size = self._parse_optional_float(signal_data.get("position_size"))

            return StrategySignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=signal_data.get("reasoning", ""),
                technical_factors=signal_data.get("technical_factors", []),
                timestamp=datetime.now(),
            )

        except Exception as e:
            self.error_count += 1
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

    def _parse_optional_float(self, value: Any) -> Optional[float]:
        """Parse optional float value with proper None handling"""
        if value is None or value == "null" or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    async def optimize_strategy_parameters(
        self,
        symbol: str,
        current_parameters: Dict[str, Any],
        performance_metrics: Dict[str, float],
        optimization_target: str = "sharpe_ratio",
    ) -> Dict[str, Any]:
        """Optimize strategy parameters based on performance feedback"""
        try:
            # Build data sections for parameter optimization
            data_sections = {
                "Current Parameters": json.dumps(current_parameters, indent=2),
                "Performance Metrics": {
                    "Win Rate": f"{performance_metrics.get('win_rate', 0):.2%}",
                    "Sharpe Ratio": f"{performance_metrics.get('sharpe_ratio', 0):.2f}",
                    "Max Drawdown": f"{performance_metrics.get('max_drawdown', 0):.2%}",
                    "Total Return": f"{performance_metrics.get('total_return', 0):.2%}",
                },
                "Optimization Target": optimization_target,
            }

            # JSON schema for parameter optimization
            json_schema = {
                "optimized_parameters": {"parameter_name": "new_value"},
                "expected_improvement": "0.0-1.0",
                "optimization_reasoning": "explanation of changes",
                "risk_assessment": "low|medium|high",
            }

            instruction = (
                f"Optimize strategy parameters to improve {optimization_target}"
            )
            prompt = self._build_standard_prompt_template(
                symbol, data_sections, instruction, json_schema
            )

            response = await self._generate_response(prompt, max_tokens=1024)

            default_values = {
                "optimized_parameters": current_parameters,
                "expected_improvement": 0.0,
                "optimization_reasoning": "No optimization performed",
                "risk_assessment": "medium",
            }

            optimization_data = self._parse_json_response(response, default_values)

            return {
                "symbol": symbol,
                "original_parameters": current_parameters,
                "optimized_parameters": optimization_data.get(
                    "optimized_parameters", {}
                ),
                "expected_improvement": self._validate_confidence(
                    optimization_data.get("expected_improvement", 0.0)
                ),
                "reasoning": optimization_data.get("optimization_reasoning", ""),
                "risk_level": self._validate_choice(
                    optimization_data.get("risk_assessment", "medium"),
                    ["low", "medium", "high"],
                    "medium",
                ),
                "timestamp": datetime.now(),
            }

        except Exception as e:
            self.error_count += 1
            return {
                "symbol": symbol,
                "error": str(e),
                "optimized_parameters": current_parameters,
                "expected_improvement": 0.0,
                "timestamp": datetime.now(),
            }

    async def process(
        self,
        symbol: str,
        markov_analysis: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        market_context: Dict[str, Any],
        performance_history: List[Dict],
        **kwargs,
    ) -> StrategySignal:
        """Main processing method for BaseAIAgent interface"""
        return await self.generate_signal(
            symbol,
            markov_analysis,
            technical_indicators,
            market_context,
            performance_history,
        )
