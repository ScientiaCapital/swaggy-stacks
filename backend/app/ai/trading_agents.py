"""
AI Trading Coordinator - Refactored from trading_agents.py

Lightweight orchestrator coordinating specialized AI agent services
"""

from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import structlog
import numpy as np

from .market_analyst_service import MarketAnalysis, MarketAnalystService
from .ollama_client import OllamaClient
from .performance_coach_service import PerformanceCoachService, TradeReview
from .risk_advisor_service import RiskAdvisorService, RiskAssessment
from .strategy_optimizer_service import StrategyOptimizerService, StrategySignal

# Enhanced imports for unsupervised learning integration
try:
    from ..ml.unsupervised.pattern_memory import PatternMemory
    from ..ml.unsupervised.pattern_mining import MarketBasketAnalyzer
    from ..ml.unsupervised.market_regime import MarketRegimeDetector
    from ..ml.unsupervised.anomaly_detector import AnomalyDetector
    UNSUPERVISED_AVAILABLE = True
except ImportError as e:
    UNSUPERVISED_AVAILABLE = False
    structlog.get_logger().warning("Unsupervised learning modules not available", error=str(e))

logger = structlog.get_logger()


class AITradingCoordinator:
    """
    Coordinates all AI agent services for comprehensive trading intelligence

    This refactored coordinator now delegates to specialized services:
    - MarketAnalystService: Market analysis and sentiment
    - RiskAdvisorService: Risk assessment and portfolio protection
    - StrategyOptimizerService: Strategy generation and optimization
    - PerformanceCoachService: Trade review and system improvement
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        enable_streaming: bool = True,
        enable_unsupervised: bool = True,
    ):
        self.ollama_client = OllamaClient(ollama_base_url)

        # Initialize specialized services
        self.market_analyst = MarketAnalystService(self.ollama_client)
        self.risk_advisor = RiskAdvisorService(self.ollama_client)
        self.strategy_optimizer = StrategyOptimizerService(self.ollama_client)
        self.performance_coach = PerformanceCoachService(self.ollama_client)

        # Initialize unsupervised learning components
        self.enable_unsupervised = enable_unsupervised and UNSUPERVISED_AVAILABLE
        if self.enable_unsupervised:
            try:
                self.pattern_memory = PatternMemory()
                self.basket_analyzer = MarketBasketAnalyzer()
                self.regime_detector = MarketRegimeDetector()
                self.anomaly_detector = AnomalyDetector()
                logger.info("Unsupervised learning components initialized successfully")
            except Exception as e:
                logger.warning("Failed to initialize unsupervised components", error=str(e))
                self.enable_unsupervised = False
        else:
            self.pattern_memory = None
            self.basket_analyzer = None
            self.regime_detector = None
            self.anomaly_detector = None

        # Real-time streaming configuration
        self.enable_streaming = enable_streaming
        self.decision_callbacks: List[Callable] = []
        self.tool_execution_callbacks: List[Callable] = []
        self.coordination_callbacks: List[Callable] = []

        # Decision history and feedback tracking
        self.decision_history: Dict[str, List[Dict[str, Any]]] = {}
        self.tool_feedback: Dict[str, List[Dict[str, Any]]] = {}
        self.active_decisions: Dict[str, Dict[str, Any]] = {}

        logger.info("AITradingCoordinator initialized with specialized services",
                   unsupervised_enabled=self.enable_unsupervised)

    def add_decision_callback(self, callback: Callable):
        """Add callback for real-time decision streaming"""
        self.decision_callbacks.append(callback)

    def add_tool_execution_callback(self, callback: Callable):
        """Add callback for tool execution feedback"""
        self.tool_execution_callbacks.append(callback)

    def add_coordination_callback(self, callback: Callable):
        """Add callback for agent coordination events"""
        self.coordination_callbacks.append(callback)

    async def _stream_decision(
        self, agent_type: str, symbol: str, decision_data: Dict[str, Any]
    ):
        """Stream agent decision in real-time"""
        if not self.enable_streaming:
            return

        decision_update = {
            "agent_id": f"{agent_type}_{symbol}",
            "agent_type": agent_type,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            **decision_data,
        }

        # Cache decision
        if symbol not in self.decision_history:
            self.decision_history[symbol] = []
        self.decision_history[symbol].append(decision_update)

        # Trigger callbacks
        for callback in self.decision_callbacks:
            try:
                await callback(decision_update)
            except Exception as e:
                logger.warning("Decision callback failed", error=str(e))

    async def _track_tool_execution(
        self,
        agent_type: str,
        tool_name: str,
        execution_id: str,
        start_time: float,
        result: Any,
        error: Optional[str] = None,
    ):
        """Track tool execution for feedback loops"""
        execution_time_ms = (datetime.now().timestamp() - start_time) * 1000

        tool_result = {
            "agent_id": f"{agent_type}_{execution_id}",
            "tool_name": tool_name,
            "execution_id": execution_id,
            "status": "success" if error is None else "failed",
            "result": result,
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.now().isoformat(),
            "error_message": error,
        }

        # Cache feedback
        if agent_type not in self.tool_feedback:
            self.tool_feedback[agent_type] = []
        self.tool_feedback[agent_type].append(tool_result)

        # Trigger callbacks
        for callback in self.tool_execution_callbacks:
            try:
                await callback(tool_result)
            except Exception as e:
                logger.warning("Tool execution callback failed", error=str(e))

    async def stream_market_analysis(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        technical_indicators: Dict[str, Any],
    ) -> MarketAnalysis:
        """Run market analysis with real-time decision streaming"""
        agent_type = "market_analyst"
        execution_id = f"market_analysis_{symbol}_{datetime.now().timestamp()}"
        start_time = datetime.now().timestamp()

        try:
            # Execute analysis
            result = await self.market_analyst.analyze_market(
                symbol=symbol,
                market_data=market_data,
                technical_indicators=technical_indicators,
            )

            # Track tool execution
            await self._track_tool_execution(
                agent_type, "analyze_market", execution_id, start_time, asdict(result)
            )

            # Stream decision
            await self._stream_decision(
                agent_type,
                symbol,
                {
                    "decision": result.sentiment,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "metadata": {
                        "key_factors": result.key_factors,
                        "risk_level": result.risk_level,
                        "recommendations": result.recommendations,
                    },
                },
            )

            return result

        except Exception as e:
            await self._track_tool_execution(
                agent_type, "analyze_market", execution_id, start_time, None, str(e)
            )
            raise

    async def stream_risk_assessment(
        self,
        symbol: str,
        position_size: float,
        account_value: float,
        current_positions: List[Dict],
        market_volatility: Dict[str, Any],
        proposed_trade: Dict[str, Any],
    ) -> RiskAssessment:
        """Run risk assessment with real-time streaming"""
        agent_type = "risk_advisor"
        execution_id = f"risk_assessment_{symbol}_{datetime.now().timestamp()}"
        start_time = datetime.now().timestamp()

        try:
            result = await self.risk_advisor.assess_risk(
                symbol=symbol,
                position_size=position_size,
                account_value=account_value,
                current_positions=current_positions,
                market_volatility=market_volatility,
                proposed_trade=proposed_trade,
            )

            # Track tool execution
            await self._track_tool_execution(
                agent_type, "assess_risk", execution_id, start_time, asdict(result)
            )

            # Stream decision
            await self._stream_decision(
                agent_type,
                symbol,
                {
                    "decision": f"RISK_{result.risk_level.upper()}",
                    "confidence": 1.0,  # Risk assessments are definitive
                    "reasoning": f"Risk level: {result.risk_level}",
                    "metadata": {
                        "portfolio_heat": result.portfolio_heat,
                        "recommended_position_size": result.recommended_position_size,
                        "key_risk_factors": result.key_risk_factors,
                        "max_position_risk": result.max_position_risk,
                    },
                },
            )

            return result

        except Exception as e:
            await self._track_tool_execution(
                agent_type, "assess_risk", execution_id, start_time, None, str(e)
            )
            raise

    async def stream_strategy_signal(
        self,
        symbol: str,
        markov_analysis: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        market_context: Dict[str, Any],
        performance_history: List[Dict],
    ) -> StrategySignal:
        """Generate strategy signal with real-time streaming"""
        agent_type = "strategy_optimizer"
        execution_id = f"strategy_signal_{symbol}_{datetime.now().timestamp()}"
        start_time = datetime.now().timestamp()

        try:
            result = await self.strategy_optimizer.generate_signal(
                symbol=symbol,
                markov_analysis=markov_analysis,
                technical_indicators=technical_indicators,
                market_context=market_context,
                performance_history=performance_history,
            )

            # Track tool execution
            await self._track_tool_execution(
                agent_type, "generate_signal", execution_id, start_time, asdict(result)
            )

            # Stream decision
            await self._stream_decision(
                agent_type,
                symbol,
                {
                    "decision": result.action,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "metadata": {
                        "entry_price": result.entry_price,
                        "stop_loss": result.stop_loss,
                        "take_profit": result.take_profit,
                        "position_size": result.position_size,
                        "technical_factors": result.technical_factors,
                    },
                },
            )

            return result

        except Exception as e:
            await self._track_tool_execution(
                agent_type, "generate_signal", execution_id, start_time, None, str(e)
            )
            raise

    async def comprehensive_analysis(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        account_info: Dict[str, Any],
        current_positions: List[Dict],
        markov_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run comprehensive analysis using all agent services and unsupervised insights"""
        try:
            logger.info("Starting comprehensive AI analysis", symbol=symbol)
            correlation_id = f"analysis_{symbol}_{datetime.now().timestamp()}"

            # Track active decision
            self.active_decisions[correlation_id] = {
                "symbol": symbol,
                "start_time": datetime.now().isoformat(),
                "status": "in_progress",
            }

            # Generate unsupervised insights if enabled
            unsupervised_insights = {}
            enhanced_market_context = {
                "regime": "trending",
                "volatility": "normal",
                "trend_strength": "moderate",
            }

            if self.enable_unsupervised:
                try:
                    # Detect market regime
                    if self.regime_detector:
                        regime_result = self.regime_detector.detect_regime(market_data, technical_indicators)
                        enhanced_market_context.update({
                            "regime": regime_result.get("current_regime", "trending"),
                            "regime_confidence": regime_result.get("confidence", 0.5),
                            "volatility": regime_result.get("volatility_regime", "normal"),
                        })
                        unsupervised_insights["regime_analysis"] = regime_result

                    # Check for anomalies
                    if self.anomaly_detector:
                        anomaly_scores = self.anomaly_detector.detect_anomalies(market_data, technical_indicators)
                        unsupervised_insights["anomaly_detection"] = anomaly_scores

                        # Adjust market context based on anomalies
                        max_anomaly_score = max(anomaly_scores.get("scores", [0]))
                        if max_anomaly_score > 0.8:
                            enhanced_market_context["anomaly_alert"] = "high"
                        elif max_anomaly_score > 0.6:
                            enhanced_market_context["anomaly_alert"] = "medium"

                    # Find similar patterns
                    if self.pattern_memory:
                        current_pattern = np.array([
                            technical_indicators.get("rsi", 50),
                            technical_indicators.get("macd", 0),
                            technical_indicators.get("bb_position", 0.5),
                            market_data.get("price_change_pct", 0),
                            market_data.get("volume_ratio", 1.0),
                        ])

                        similar_patterns = await self.pattern_memory.find_similar_patterns(
                            current_pattern, symbol, top_k=5
                        )
                        unsupervised_insights["similar_patterns"] = similar_patterns

                    # Analyze market correlations
                    if self.basket_analyzer and len(current_positions) > 1:
                        correlation_rules = self.basket_analyzer.generate_association_rules(
                            [pos.get("symbol") for pos in current_positions]
                        )
                        unsupervised_insights["correlation_analysis"] = correlation_rules

                    logger.info("Unsupervised insights generated", symbol=symbol,
                              insights_count=len(unsupervised_insights))

                except Exception as e:
                    logger.warning("Failed to generate unsupervised insights", symbol=symbol, error=str(e))

            # Run streaming market analysis
            market_analysis = await self.stream_market_analysis(
                symbol=symbol,
                market_data=market_data,
                technical_indicators=technical_indicators,
            )

            # Calculate proposed position size
            account_value = account_info.get("equity", 100000)
            base_position_size = account_value * 0.02  # 2% of account

            # Adjust position size based on anomaly detection
            if unsupervised_insights.get("anomaly_detection", {}).get("max_score", 0) > 0.7:
                proposed_position_size = base_position_size * 0.5  # Reduce size during anomalies
            else:
                proposed_position_size = base_position_size

            # Enhanced volatility calculation with regime context
            market_volatility = {
                "atr": technical_indicators.get("atr", 0),
                "hist_vol": market_data.get("volatility", 0),
                "regime_vol": enhanced_market_context.get("volatility", "normal"),
            }

            # Run streaming risk assessment
            risk_assessment = await self.stream_risk_assessment(
                symbol=symbol,
                position_size=proposed_position_size,
                account_value=account_value,
                current_positions=current_positions,
                market_volatility=market_volatility,
                proposed_trade={"stop_loss_percent": 0.05, "take_profit_percent": 0.10},
            )

            # Generate streaming optimized signal with enhanced context
            strategy_signal = await self.stream_strategy_signal(
                symbol=symbol,
                markov_analysis=markov_analysis,
                technical_indicators=technical_indicators,
                market_context=enhanced_market_context,
                performance_history=[],
            )

            # Synthesize final recommendation with unsupervised insights
            final_recommendation = self._synthesize_recommendation(
                market_analysis, risk_assessment, strategy_signal, unsupervised_insights
            )

            # Compile comprehensive result
            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "correlation_id": correlation_id,
                "market_analysis": asdict(market_analysis),
                "risk_assessment": asdict(risk_assessment),
                "strategy_signal": asdict(strategy_signal),
                "unsupervised_insights": unsupervised_insights,
                "enhanced_market_context": enhanced_market_context,
                "final_recommendation": final_recommendation,
                "agent_performance": self.get_all_agent_stats(),
                "unsupervised_enabled": self.enable_unsupervised,
            }

            # Store pattern for future learning if enabled
            if self.enable_unsupervised and self.pattern_memory:
                try:
                    outcome_pattern = {
                        "symbol": symbol,
                        "features": current_pattern.tolist() if 'current_pattern' in locals() else [],
                        "recommendation": final_recommendation,
                        "confidence": strategy_signal.confidence,
                        "timestamp": datetime.now().isoformat(),
                    }
                    await self.pattern_memory.store_pattern(outcome_pattern, symbol)
                except Exception as e:
                    logger.warning("Failed to store pattern for learning", error=str(e))

            # Update active decision status
            self.active_decisions[correlation_id].update(
                {
                    "status": "completed",
                    "end_time": datetime.now().isoformat(),
                    "final_recommendation": final_recommendation,
                }
            )

            # Stream final coordinated decision
            await self._stream_coordinated_decision(symbol, result)

            logger.info("Comprehensive analysis completed", symbol=symbol,
                       unsupervised_insights_used=bool(unsupervised_insights))
            return result

        except Exception as e:
            logger.error("Comprehensive analysis failed", symbol=symbol, error=str(e))

            # Update active decision with error
            if correlation_id in self.active_decisions:
                self.active_decisions[correlation_id].update(
                    {
                        "status": "failed",
                        "error": str(e),
                        "end_time": datetime.now().isoformat(),
                    }
                )

            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "correlation_id": correlation_id,
                "error": str(e),
                "final_recommendation": "HOLD",
                "unsupervised_enabled": self.enable_unsupervised,
            }

    async def _stream_coordinated_decision(
        self, symbol: str, analysis_result: Dict[str, Any]
    ):
        """Stream final coordinated decision from all agents"""
        coordination_update = {
            "message_type": "final_decision",
            "symbol": symbol,
            "final_recommendation": analysis_result["final_recommendation"],
            "agent_consensus": {
                "market_analyst": analysis_result["market_analysis"]["sentiment"],
                "risk_advisor": analysis_result["risk_assessment"]["risk_level"],
                "strategy_optimizer": analysis_result["strategy_signal"]["action"],
            },
            "confidence_scores": {
                "market_analyst": analysis_result["market_analysis"]["confidence"],
                "risk_advisor": 1.0,  # Risk assessments are definitive
                "strategy_optimizer": analysis_result["strategy_signal"]["confidence"],
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Trigger coordination callbacks
        for callback in self.coordination_callbacks:
            try:
                await callback(coordination_update)
            except Exception as e:
                logger.warning("Coordination callback failed", error=str(e))

    def _synthesize_recommendation(
        self,
        market_analysis: MarketAnalysis,
        risk_assessment: RiskAssessment,
        strategy_signal: StrategySignal,
        unsupervised_insights: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Synthesize final recommendation from all agent services and unsupervised insights"""
        # Risk trumps everything
        if risk_assessment.risk_level == "high":
            return "HOLD"

        # Check for high anomaly alerts - override normal signals
        if unsupervised_insights and unsupervised_insights.get("anomaly_detection", {}).get("max_score", 0) > 0.8:
            logger.info("High anomaly detected - forcing HOLD recommendation")
            return "HOLD"

        # Check regime stability - avoid trading during regime transitions
        if unsupervised_insights:
            regime_analysis = unsupervised_insights.get("regime_analysis", {})
            regime_confidence = regime_analysis.get("confidence", 1.0)
            if regime_confidence < 0.5:
                logger.info("Low regime confidence - forcing HOLD recommendation")
                return "HOLD"

        # Adjust confidence thresholds based on similar patterns
        base_confidence_threshold = 0.6
        if unsupervised_insights and "similar_patterns" in unsupervised_insights:
            similar_patterns = unsupervised_insights["similar_patterns"]
            if len(similar_patterns) > 0:
                # If we have strong historical evidence, lower the threshold
                avg_success_rate = np.mean([p.get("success_rate", 0.5) for p in similar_patterns])
                if avg_success_rate > 0.7:
                    base_confidence_threshold = 0.5  # More aggressive with good patterns
                elif avg_success_rate < 0.3:
                    base_confidence_threshold = 0.8  # More conservative with bad patterns

        # Need sufficient confidence for action
        if market_analysis.confidence < base_confidence_threshold or strategy_signal.confidence < base_confidence_threshold:
            return "HOLD"

        # Check correlation rules if available
        if unsupervised_insights and "correlation_analysis" in unsupervised_insights:
            correlation_rules = unsupervised_insights["correlation_analysis"]
            # If strong negative correlations exist with current positions, be more cautious
            for rule in correlation_rules:
                if rule.get("confidence", 0) > 0.8 and rule.get("lift", 1) < 0.5:
                    logger.info("Strong negative correlation detected - adjusting to HOLD")
                    return "HOLD"

        # If both market and strategy agree, follow their recommendation
        if market_analysis.sentiment == "bullish" and strategy_signal.action == "BUY":
            return "BUY"
        elif (
            market_analysis.sentiment == "bearish" and strategy_signal.action == "SELL"
        ):
            return "SELL"
        else:
            return "HOLD"

    def get_all_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get execution statistics for all agent services"""
        stats = {
            "market_analyst": self.market_analyst.get_agent_stats(),
            "risk_advisor": self.risk_advisor.get_agent_stats(),
            "strategy_optimizer": self.strategy_optimizer.get_agent_stats(),
            "performance_coach": self.performance_coach.get_agent_stats(),
        }

        # Add unsupervised learning stats if enabled
        if self.enable_unsupervised:
            stats["unsupervised"] = self._get_unsupervised_stats()

        return stats

    def _get_unsupervised_stats(self) -> Dict[str, Any]:
        """Get statistics from unsupervised learning components"""
        stats = {
            "enabled": self.enable_unsupervised,
            "components_available": UNSUPERVISED_AVAILABLE,
        }

        if self.enable_unsupervised:
            try:
                if self.pattern_memory:
                    stats["pattern_memory"] = {
                        "total_patterns": getattr(self.pattern_memory, 'pattern_count', 0),
                        "cache_hit_rate": getattr(self.pattern_memory, 'cache_hit_rate', 0.0),
                    }

                if self.regime_detector:
                    stats["regime_detector"] = {
                        "last_regime": getattr(self.regime_detector, 'current_regime', 'unknown'),
                        "regime_confidence": getattr(self.regime_detector, 'regime_confidence', 0.0),
                    }

                if self.anomaly_detector:
                    stats["anomaly_detector"] = {
                        "detection_count": getattr(self.anomaly_detector, 'detection_count', 0),
                        "avg_anomaly_score": getattr(self.anomaly_detector, 'avg_score', 0.0),
                    }

                if self.basket_analyzer:
                    stats["basket_analyzer"] = {
                        "rules_generated": getattr(self.basket_analyzer, 'rules_count', 0),
                        "avg_confidence": getattr(self.basket_analyzer, 'avg_confidence', 0.0),
                    }

            except Exception as e:
                stats["error"] = str(e)

        return stats

    def get_decision_history(
        self, symbol: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent decision history for a symbol"""
        if symbol not in self.decision_history:
            return []
        return self.decision_history[symbol][-limit:]

    def get_tool_feedback(
        self, agent_type: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent tool execution feedback for an agent"""
        if agent_type not in self.tool_feedback:
            return []
        return self.tool_feedback[agent_type][-limit:]

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all AI components"""
        base_health = await self.ollama_client.health_check()

        return {
            **base_health,
            "agent_stats": self.get_all_agent_stats(),
            "streaming_enabled": self.enable_streaming,
            "active_decisions": len(self.active_decisions),
            "decision_callbacks": len(self.decision_callbacks),
            "tool_callbacks": len(self.tool_execution_callbacks),
            "coordination_callbacks": len(self.coordination_callbacks),
        }

    # Legacy compatibility methods
    async def review_trade(
        self,
        trade_data: Dict[str, Any],
        market_context: Dict[str, Any],
        system_performance: Dict[str, Any],
    ) -> TradeReview:
        """Legacy compatibility for trade review"""
        return await self.performance_coach.review_trade(
            trade_data, market_context, system_performance
        )
