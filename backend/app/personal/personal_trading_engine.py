"""
Personal Trading Engine - Simplified interface for personal trading decisions

Wraps the complex AI system with a simple, personal-focused interface that answers:
- Should I trade right now?
- What's my risk?
- What is the AI learning about my patterns?
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from ..ai.trading_agents import AITradingCoordinator
from ..trading.risk_manager import RiskManager

logger = structlog.get_logger(__name__)


@dataclass
class PersonalTradingDecision:
    """Simple trading decision for personal use"""

    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-1 scale
    reasoning: str  # Simple explanation
    position_size_usd: float
    risk_level: str  # LOW, MEDIUM, HIGH

    # Key metrics
    expected_return: float
    max_loss: float
    time_horizon: str

    # Learning insights
    pattern_match: Optional[str]
    market_regime: str
    anomaly_alert: Optional[str]

    timestamp: datetime


@dataclass
class PersonalPortfolioSummary:
    """Simple portfolio overview for personal tracking"""

    total_value: float
    daily_pnl: float
    daily_pnl_pct: float

    active_positions: int
    cash_available: float
    portfolio_risk: str  # LOW, MEDIUM, HIGH

    # AI insights
    ai_recommendation: str
    learning_status: str
    pattern_confidence: float

    last_updated: datetime


class PersonalTradingEngine:
    """Simplified trading engine optimized for personal use"""

    def __init__(self,
                 user_id: int,
                 ai_coordinator: Optional[AITradingCoordinator] = None,
                 risk_manager: Optional[RiskManager] = None,
                 personal_mode: bool = True):

        self.user_id = user_id
        self.personal_mode = personal_mode

        # Initialize with simplified configs for personal use
        self.ai_coordinator = ai_coordinator or AITradingCoordinator(
            enable_streaming=False,  # Reduce overhead
            enable_unsupervised=True  # Keep learning but lightweight
        )

        self.risk_manager = risk_manager or RiskManager(
            user_id=user_id,
            user_risk_params={
                'max_position_size': 5000,  # Conservative for personal trading
                'max_daily_loss': 500,
                'anomaly_risk_adjustment_enabled': True
            }
        )

        # Personal trading preferences
        self.personal_preferences = {
            'risk_tolerance': 'conservative',
            'explanation_detail': 'simple',  # vs 'detailed'
            'learning_transparency': True,
            'max_concurrent_positions': 3
        }

        # Simple metrics tracking
        self.decision_history: List[PersonalTradingDecision] = []

        logger.info("Personal Trading Engine initialized",
                   user_id=user_id,
                   personal_mode=personal_mode)

    async def should_i_trade(self,
                           symbol: str,
                           market_data: Dict[str, Any],
                           technical_indicators: Dict[str, Any],
                           account_info: Dict[str, Any],
                           current_positions: List[Dict] = None) -> PersonalTradingDecision:
        """Simple interface: Should I trade this symbol right now?"""

        try:
            current_positions = current_positions or []

            # Get AI analysis (but simplify the output)
            ai_analysis = await self.ai_coordinator.comprehensive_analysis(
                symbol=symbol,
                market_data=market_data,
                technical_indicators=technical_indicators,
                account_info=account_info,
                current_positions=current_positions,
                markov_analysis={}  # Simplified
            )

            # Extract key insights in personal-friendly format
            decision = self._translate_ai_to_personal_decision(
                symbol, ai_analysis, account_info, current_positions
            )

            # Store for learning
            self.decision_history.append(decision)

            # Keep only recent decisions (memory efficient)
            if len(self.decision_history) > 100:
                self.decision_history = self.decision_history[-100:]

            logger.info("Personal trading decision made",
                       symbol=symbol,
                       action=decision.action,
                       confidence=decision.confidence)

            return decision

        except Exception as e:
            logger.error("Failed to generate personal trading decision",
                        symbol=symbol, error=str(e))

            # Safe fallback
            return PersonalTradingDecision(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                reasoning="System error - staying safe with HOLD",
                position_size_usd=0.0,
                risk_level="HIGH",
                expected_return=0.0,
                max_loss=0.0,
                time_horizon="unknown",
                pattern_match=None,
                market_regime="unknown",
                anomaly_alert="system_error",
                timestamp=datetime.now()
            )

    def _translate_ai_to_personal_decision(self,
                                         symbol: str,
                                         ai_analysis: Dict[str, Any],
                                         account_info: Dict[str, Any],
                                         current_positions: List[Dict]) -> PersonalTradingDecision:
        """Translate complex AI analysis into simple personal decision"""

        # Extract core recommendation
        action = ai_analysis.get('final_recommendation', 'HOLD')

        # Calculate simple confidence (0-1 scale)
        market_confidence = ai_analysis.get('market_analysis', {}).get('confidence', 0.5)
        strategy_confidence = ai_analysis.get('strategy_signal', {}).get('confidence', 0.5)
        overall_confidence = (market_confidence + strategy_confidence) / 2

        # Simple reasoning
        reasoning_parts = []
        market_sentiment = ai_analysis.get('market_analysis', {}).get('sentiment', 'neutral')
        risk_level = ai_analysis.get('risk_assessment', {}).get('risk_level', 'medium')

        if action == "BUY":
            reasoning_parts.append(f"Market looks {market_sentiment}")
        elif action == "SELL":
            reasoning_parts.append(f"Market appears {market_sentiment}")
        else:
            reasoning_parts.append("Staying cautious")

        reasoning_parts.append(f"Risk: {risk_level}")

        # Add anomaly info if available
        if ai_analysis.get('unsupervised_insights', {}).get('anomaly_detection'):
            anomaly_score = ai_analysis['unsupervised_insights']['anomaly_detection'].get('max_score', 0)
            if anomaly_score > 0.7:
                reasoning_parts.append("⚠️ Unusual market activity detected")

        reasoning = " | ".join(reasoning_parts)

        # Calculate position size (simplified)
        account_value = account_info.get('equity', 100000)
        if action == "BUY":
            base_size = account_value * 0.02  # 2% of account
            # Adjust for confidence and risk
            if overall_confidence > 0.8:
                position_size = base_size * 1.5
            elif overall_confidence < 0.4:
                position_size = base_size * 0.5
            else:
                position_size = base_size
        else:
            position_size = 0.0

        # Simplified risk assessment
        if risk_level == "high" or overall_confidence < 0.3:
            simple_risk = "HIGH"
        elif risk_level == "low" and overall_confidence > 0.7:
            simple_risk = "LOW"
        else:
            simple_risk = "MEDIUM"

        # Extract learning insights
        pattern_match = None
        if ai_analysis.get('unsupervised_insights', {}).get('similar_patterns'):
            patterns = ai_analysis['unsupervised_insights']['similar_patterns']
            if patterns:
                pattern_match = f"Similar to {len(patterns)} past patterns"

        market_regime = ai_analysis.get('enhanced_market_context', {}).get('regime', 'unknown')

        anomaly_alert = None
        if ai_analysis.get('unsupervised_insights', {}).get('anomaly_detection'):
            max_score = ai_analysis['unsupervised_insights']['anomaly_detection'].get('max_score', 0)
            if max_score > 0.8:
                anomaly_alert = "high"
            elif max_score > 0.6:
                anomaly_alert = "medium"

        # Simple expected return and max loss
        if action == "BUY":
            expected_return = position_size * 0.02 * overall_confidence  # Rough estimate
            max_loss = position_size * 0.05  # 5% max loss
            time_horizon = "1-3 days"
        else:
            expected_return = 0.0
            max_loss = 0.0
            time_horizon = "N/A"

        return PersonalTradingDecision(
            symbol=symbol,
            action=action,
            confidence=overall_confidence,
            reasoning=reasoning,
            position_size_usd=position_size,
            risk_level=simple_risk,
            expected_return=expected_return,
            max_loss=max_loss,
            time_horizon=time_horizon,
            pattern_match=pattern_match,
            market_regime=market_regime,
            anomaly_alert=anomaly_alert,
            timestamp=datetime.now()
        )

    async def get_portfolio_summary(self,
                                  account_info: Dict[str, Any],
                                  current_positions: List[Dict]) -> PersonalPortfolioSummary:
        """Get simple portfolio overview"""

        try:
            # Basic portfolio metrics
            total_value = account_info.get('equity', 0)
            daily_pnl = account_info.get('daily_pnl', 0)
            daily_pnl_pct = (daily_pnl / total_value * 100) if total_value > 0 else 0

            active_positions = len(current_positions)
            cash_available = account_info.get('cash', 0)

            # Simple risk assessment
            risk_summary = self.risk_manager.get_risk_summary(current_positions, total_value, daily_pnl)
            portfolio_risk = "LOW"
            if risk_summary.get('portfolio_heat', 0) > 0.7:
                portfolio_risk = "HIGH"
            elif risk_summary.get('portfolio_heat', 0) > 0.4:
                portfolio_risk = "MEDIUM"

            # AI insights summary
            recent_decisions = self.decision_history[-5:] if self.decision_history else []

            if recent_decisions:
                avg_confidence = sum(d.confidence for d in recent_decisions) / len(recent_decisions)

                # Simple recommendation based on recent patterns
                buy_signals = sum(1 for d in recent_decisions if d.action == "BUY")
                sell_signals = sum(1 for d in recent_decisions if d.action == "SELL")

                if buy_signals > sell_signals and avg_confidence > 0.6:
                    ai_recommendation = "BULLISH - Consider new positions"
                elif sell_signals > buy_signals:
                    ai_recommendation = "BEARISH - Consider reducing exposure"
                else:
                    ai_recommendation = "NEUTRAL - Monitor for opportunities"

                pattern_confidence = avg_confidence
                learning_status = f"Learning from {len(self.decision_history)} decisions"
            else:
                ai_recommendation = "COLLECTING DATA - No recent decisions"
                pattern_confidence = 0.0
                learning_status = "Starting to learn..."

            return PersonalPortfolioSummary(
                total_value=total_value,
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                active_positions=active_positions,
                cash_available=cash_available,
                portfolio_risk=portfolio_risk,
                ai_recommendation=ai_recommendation,
                learning_status=learning_status,
                pattern_confidence=pattern_confidence,
                last_updated=datetime.now()
            )

        except Exception as e:
            logger.error("Failed to generate portfolio summary", error=str(e))

            # Safe fallback
            return PersonalPortfolioSummary(
                total_value=account_info.get('equity', 0),
                daily_pnl=0,
                daily_pnl_pct=0,
                active_positions=0,
                cash_available=0,
                portfolio_risk="UNKNOWN",
                ai_recommendation="System error - check manually",
                learning_status="Error",
                pattern_confidence=0.0,
                last_updated=datetime.now()
            )

    def get_learning_insights(self) -> Dict[str, Any]:
        """Simple learning insights for personal review"""

        if not self.decision_history:
            return {
                "status": "just_starting",
                "total_decisions": 0,
                "insights": []
            }

        # Simple pattern analysis
        total_decisions = len(self.decision_history)
        buy_decisions = sum(1 for d in self.decision_history if d.action == "BUY")
        sell_decisions = sum(1 for d in self.decision_history if d.action == "SELL")
        hold_decisions = total_decisions - buy_decisions - sell_decisions

        # Calculate average confidence by action
        action_confidence = {}
        for action in ["BUY", "SELL", "HOLD"]:
            action_decisions = [d for d in self.decision_history if d.action == action]
            if action_decisions:
                action_confidence[action] = sum(d.confidence for d in action_decisions) / len(action_decisions)

        # Simple insights
        insights = []

        if buy_decisions > sell_decisions:
            insights.append(f"Tendency: More bullish signals ({buy_decisions} buy vs {sell_decisions} sell)")
        elif sell_decisions > buy_decisions:
            insights.append(f"Tendency: More bearish signals ({sell_decisions} sell vs {buy_decisions} buy)")
        else:
            insights.append("Balanced approach between buying and selling")

        if action_confidence.get("BUY", 0) > 0.7:
            insights.append("High confidence in buy signals")
        if action_confidence.get("SELL", 0) > 0.7:
            insights.append("High confidence in sell signals")

        # Pattern matching insights
        pattern_matches = [d for d in self.decision_history if d.pattern_match]
        if pattern_matches:
            insights.append(f"Pattern recognition active: {len(pattern_matches)} matches found")

        return {
            "status": "learning",
            "total_decisions": total_decisions,
            "decision_breakdown": {
                "BUY": buy_decisions,
                "SELL": sell_decisions,
                "HOLD": hold_decisions
            },
            "confidence_by_action": action_confidence,
            "insights": insights,
            "last_decision": self.decision_history[-1].timestamp.isoformat() if self.decision_history else None
        }

    def enable_simple_mode(self):
        """Enable simplified mode for even faster decisions"""
        self.personal_preferences['explanation_detail'] = 'minimal'
        self.ai_coordinator.enable_streaming = False
        logger.info("Simple mode enabled")

    def get_personal_stats(self) -> Dict[str, Any]:
        """Get personal trading statistics"""
        return {
            "total_decisions": len(self.decision_history),
            "recent_activity": len([d for d in self.decision_history
                                  if (datetime.now() - d.timestamp).days <= 7]),
            "learning_status": self.get_learning_insights()["status"],
            "personal_mode": self.personal_mode,
            "risk_tolerance": self.personal_preferences["risk_tolerance"],
            "ai_enabled": self.ai_coordinator.enable_unsupervised
        }