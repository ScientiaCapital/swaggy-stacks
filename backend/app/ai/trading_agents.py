"""
AI Trading Coordinator - Refactored from trading_agents.py

Lightweight orchestrator coordinating specialized AI agent services
"""

from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import structlog

from .market_analyst_service import MarketAnalysis, MarketAnalystService
from .ollama_client import OllamaClient
from .performance_coach_service import PerformanceCoachService, TradeReview
from .risk_advisor_service import RiskAdvisorService, RiskAssessment
from .strategy_optimizer_service import StrategyOptimizerService, StrategySignal

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
    ):
        self.ollama_client = OllamaClient(ollama_base_url)

        # Initialize specialized services
        self.market_analyst = MarketAnalystService(self.ollama_client)
        self.risk_advisor = RiskAdvisorService(self.ollama_client)
        self.strategy_optimizer = StrategyOptimizerService(self.ollama_client)
        self.performance_coach = PerformanceCoachService(self.ollama_client)

        # Real-time streaming configuration
        self.enable_streaming = enable_streaming
        self.decision_callbacks: List[Callable] = []
        self.tool_execution_callbacks: List[Callable] = []
        self.coordination_callbacks: List[Callable] = []

        # Decision history and feedback tracking
        self.decision_history: Dict[str, List[Dict[str, Any]]] = {}
        self.tool_feedback: Dict[str, List[Dict[str, Any]]] = {}
        self.active_decisions: Dict[str, Dict[str, Any]] = {}

        logger.info("AITradingCoordinator initialized with specialized services")

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
        """Run comprehensive analysis using all agent services"""
        try:
            logger.info("Starting comprehensive AI analysis", symbol=symbol)
            correlation_id = f"analysis_{symbol}_{datetime.now().timestamp()}"

            # Track active decision
            self.active_decisions[correlation_id] = {
                "symbol": symbol,
                "start_time": datetime.now().isoformat(),
                "status": "in_progress",
            }

            # Run streaming market analysis
            market_analysis = await self.stream_market_analysis(
                symbol=symbol,
                market_data=market_data,
                technical_indicators=technical_indicators,
            )

            # Calculate proposed position size
            account_value = account_info.get("equity", 100000)
            proposed_position_size = account_value * 0.02  # 2% of account

            # Run streaming risk assessment
            risk_assessment = await self.stream_risk_assessment(
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

            # Generate streaming optimized signal
            strategy_signal = await self.stream_strategy_signal(
                symbol=symbol,
                markov_analysis=markov_analysis,
                technical_indicators=technical_indicators,
                market_context={
                    "regime": "trending",
                    "volatility": "normal",
                    "trend_strength": "moderate",
                },
                performance_history=[],
            )

            # Synthesize final recommendation
            final_recommendation = self._synthesize_recommendation(
                market_analysis, risk_assessment, strategy_signal
            )

            # Compile comprehensive result
            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "correlation_id": correlation_id,
                "market_analysis": asdict(market_analysis),
                "risk_assessment": asdict(risk_assessment),
                "strategy_signal": asdict(strategy_signal),
                "final_recommendation": final_recommendation,
                "agent_performance": self.get_all_agent_stats(),
            }

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

            logger.info("Comprehensive analysis completed", symbol=symbol)
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
    ) -> str:
        """Synthesize final recommendation from all agent services"""
        # Risk trumps everything
        if risk_assessment.risk_level == "high":
            return "HOLD"

        # Need high confidence for action
        if market_analysis.confidence < 0.6 or strategy_signal.confidence < 0.6:
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
        return {
            "market_analyst": self.market_analyst.get_agent_stats(),
            "risk_advisor": self.risk_advisor.get_agent_stats(),
            "strategy_optimizer": self.strategy_optimizer.get_agent_stats(),
            "performance_coach": self.performance_coach.get_agent_stats(),
        }

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
