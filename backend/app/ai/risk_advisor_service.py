"""
Risk Advisor Service - Extracted from trading_agents.py

Specialized service for risk assessment and portfolio protection
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from .base_agent import BaseAIAgent
from .ollama_client import OllamaClient


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


class RiskAdvisorService(BaseAIAgent):
    """AI agent specialized in risk assessment and portfolio protection"""

    def __init__(self, ollama_client: OllamaClient):
        super().__init__(ollama_client, "risk", "risk_assessment.txt")

    def _get_default_prompt(self) -> str:
        return (
            "You are a risk management expert. Focus on capital preservation "
            "and identify potential risks in trading positions. Provide "
            "actionable risk mitigation strategies."
        )

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
            # Calculate current portfolio metrics
            total_risk = sum(pos.get("risk_amount", 0) for pos in current_positions)
            portfolio_heat = (total_risk / account_value) * 100 if account_value > 0 else 0

            # Calculate proposed position risk
            proposed_risk = position_size * proposed_trade.get("stop_loss_percent", 0.05)
            position_risk_percent = (proposed_risk / account_value) * 100

            # Build data sections
            data_sections = {
                "Position Details": {
                    "Symbol": symbol,
                    "Proposed Position Size": f"${position_size:,.2f}",
                    "Account Value": f"${account_value:,.2f}",
                    "Current Portfolio Heat": f"{portfolio_heat:.2f}%",
                    "Proposed Position Risk": f"{position_risk_percent:.2f}%"
                },
                "Current Positions": json.dumps(current_positions, indent=2),
                "Market Volatility": {
                    "VIX": market_volatility.get('vix', 'N/A'),
                    "Symbol ATR": market_volatility.get('atr', 'N/A'),
                    "Historical Volatility": market_volatility.get('hist_vol', 'N/A')
                },
                "Proposed Trade Details": json.dumps(proposed_trade, indent=2)
            }

            # JSON schema
            json_schema = {
                "risk_level": "low|medium|high",
                "portfolio_heat": portfolio_heat,
                "recommended_position_size": 0.0,
                "key_risk_factors": ["factor1", "factor2"],
                "mitigation_strategies": ["strategy1", "strategy2"],
                "exit_conditions": ["condition1", "condition2"],
                "max_position_risk": 0.5
            }

            # Build prompt
            instruction = "Provide comprehensive risk assessment for this proposed trade"
            prompt = self._build_standard_prompt_template(
                symbol, data_sections, instruction, json_schema
            )

            # Generate response
            response = await self._generate_response(prompt, max_tokens=1024)

            # Parse response with fallback defaults
            default_values = {
                "risk_level": "high",
                "portfolio_heat": portfolio_heat,
                "recommended_position_size": position_size * 0.5,
                "key_risk_factors": ["High uncertainty"],
                "mitigation_strategies": ["Reduce position size"],
                "exit_conditions": ["Stop loss triggered"],
                "max_position_risk": 0.5,
            }

            risk_data = self._parse_json_response(response, default_values)

            # Validate and normalize data
            risk_level = self._validate_choice(
                risk_data["risk_level"],
                ["low", "medium", "high"],
                "high"
            )

            # Ensure reasonable position sizing recommendations
            recommended_size = float(risk_data.get("recommended_position_size", position_size * 0.5))
            recommended_size = max(0, min(recommended_size, account_value * 0.25))  # Cap at 25% of account

            max_risk = float(risk_data.get("max_position_risk", 0.5))
            max_risk = max(0.01, min(max_risk, 0.1))  # Between 1% and 10%

            return RiskAssessment(
                symbol=symbol,
                risk_level=risk_level,
                portfolio_heat=float(risk_data.get("portfolio_heat", portfolio_heat)),
                recommended_position_size=recommended_size,
                key_risk_factors=risk_data.get("key_risk_factors", []),
                mitigation_strategies=risk_data.get("mitigation_strategies", []),
                exit_conditions=risk_data.get("exit_conditions", []),
                max_position_risk=max_risk,
                timestamp=datetime.now()
            )

        except Exception as e:
            self.error_count += 1
            return RiskAssessment(
                symbol=symbol,
                risk_level="high",
                portfolio_heat=portfolio_heat if 'portfolio_heat' in locals() else 0.0,
                recommended_position_size=position_size * 0.25,
                key_risk_factors=["Assessment error"],
                mitigation_strategies=["Manual review required"],
                exit_conditions=["Immediate exit if uncertain"],
                max_position_risk=0.25,
                timestamp=datetime.now()
            )

    async def process(self, symbol: str, position_size: float, account_value: float,
                     current_positions: List[Dict], market_volatility: Dict[str, float],
                     proposed_trade: Dict[str, Any], **kwargs) -> RiskAssessment:
        """Main processing method for BaseAIAgent interface"""
        return await self.assess_risk(
            symbol, position_size, account_value, current_positions,
            market_volatility, proposed_trade
        )