"""
LangGraph Trading Workflow

This module implements a sophisticated multi-step trading decision workflow using LangGraph,
providing state management, conditional routing, and parallel strategy evaluation.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph

from app.monitoring.metrics import PrometheusMetrics
from app.rag.agents.base_agent import TradingSignal
from app.rag.agents.strategy_agent import (
    FibonacciStrategy,
    MarkovStrategy,
    StrategyPlugin,
    WyckoffStrategy,
)
from app.rag.services.context_builder import (
    TradingContextBuilder,
    TradingState,
    get_context_builder,
)
from app.rag.services.tool_registry import get_tool_registry

logger = logging.getLogger(__name__)


class TradingWorkflow:
    """
    Advanced LangGraph trading workflow implementing multi-step decision making
    with state management, risk gates, and parallel strategy evaluation
    """

    def __init__(self):
        self.context_builder: Optional[TradingContextBuilder] = None
        self.tool_registry = None
        self.strategies: Dict[str, StrategyPlugin] = {}
        self.metrics = PrometheusMetrics()
        self.graph: Optional[CompiledGraph] = None

    async def initialize(self) -> None:
        """Initialize the trading workflow"""
        # Initialize dependencies
        self.context_builder = await get_context_builder()
        self.tool_registry = await get_tool_registry()

        # Initialize strategy plugins
        self.strategies = {
            "markov": MarkovStrategy(),
            "wyckoff": WyckoffStrategy(),
            "fibonacci": FibonacciStrategy(),
        }

        # Build the workflow graph
        self.graph = self._build_workflow_graph()

        logger.info("Trading Workflow initialized with LangGraph")

    def _build_workflow_graph(self) -> CompiledGraph:
        """Build the LangGraph workflow"""
        # Create state graph
        workflow = StateGraph(TradingState)

        # Add nodes for each workflow step
        workflow.add_node("market_analysis", self._market_analysis_node)
        workflow.add_node("memory_retrieval", self._memory_retrieval_node)
        workflow.add_node("strategy_evaluation", self._strategy_evaluation_node)
        workflow.add_node("risk_assessment", self._risk_assessment_node)
        workflow.add_node("decision_synthesis", self._decision_synthesis_node)
        workflow.add_node("execution_planning", self._execution_planning_node)
        workflow.add_node("monitoring", self._monitoring_node)

        # Define the workflow edges and conditional routing
        workflow.set_entry_point("market_analysis")

        # Market analysis -> Memory retrieval (always)
        workflow.add_edge("market_analysis", "memory_retrieval")

        # Memory retrieval -> Strategy evaluation (always)
        workflow.add_edge("memory_retrieval", "strategy_evaluation")

        # Strategy evaluation -> Risk assessment (always)
        workflow.add_edge("strategy_evaluation", "risk_assessment")

        # Risk assessment -> Decision synthesis or END (conditional)
        workflow.add_conditional_edges(
            "risk_assessment",
            self._risk_gate_condition,
            {"proceed": "decision_synthesis", "halt": END},
        )

        # Decision synthesis -> Execution planning or END (conditional)
        workflow.add_conditional_edges(
            "decision_synthesis",
            self._decision_condition,
            {"execute": "execution_planning", "hold": END},
        )

        # Execution planning -> Monitoring or END (conditional)
        workflow.add_conditional_edges(
            "execution_planning",
            self._execution_condition,
            {"monitor": "monitoring", "complete": END},
        )

        # Monitoring -> END (always)
        workflow.add_edge("monitoring", END)

        # Compile with checkpointer for state persistence
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def execute_workflow(
        self,
        symbol: str,
        user_message: str = "",
        workflow_type: str = "standard_analysis",
    ) -> Dict[str, Any]:
        """Execute the complete trading workflow"""
        try:
            # Create initial state
            initial_state = await self.context_builder.create_initial_state(
                symbol=symbol, workflow_type=workflow_type
            )

            # Add user message if provided
            if user_message:
                await self.context_builder.add_user_message(initial_state, user_message)

            # Execute workflow
            config = {"configurable": {"thread_id": initial_state["session_id"]}}

            start_time = datetime.now()

            # Run the workflow
            result = await self.graph.ainvoke(initial_state, config)

            execution_time = (datetime.now() - start_time).total_seconds()

            # Record metrics
            self.metrics.record_workflow_execution(
                workflow_type=workflow_type,
                symbol=symbol,
                execution_time=execution_time,
                success=len(result["errors"]) == 0,
            )

            return {
                "success": True,
                "session_id": result["session_id"],
                "symbol": symbol,
                "decision": result.get("final_decision"),
                "execution_plan": result.get("execution_plan"),
                "confidence": result["decision_context"].confidence_level,
                "strategy_signals": len(result["strategy_signals"]),
                "risk_approved": result["risk_approval"],
                "execution_time": execution_time,
                "errors": result["errors"],
                "warnings": result["warnings"],
            }

        except Exception as e:
            error_msg = f"Workflow execution failed for {symbol}: {str(e)}"
            logger.error(error_msg)

            return {"success": False, "error": error_msg, "symbol": symbol}

    async def _market_analysis_node(self, state: TradingState) -> TradingState:
        """Node 1: Market Analysis - Gather and analyze market data"""
        logger.info(f"Starting market analysis for {state['symbol']}")

        # Update workflow step
        await self.context_builder.update_workflow_step(
            state, "market_analysis", ["memory_retrieval"]
        )

        # Update market context
        state = await self.context_builder.update_market_context(state)

        # Add analysis message
        if state["market_data"].current_price:
            price_info = f"Current price: ${state['market_data'].current_price:.2f}"
            if state["market_data"].volatility:
                price_info += f", Volatility: {state['market_data'].volatility:.1%}"
        else:
            price_info = "Price data unavailable"

        await self.context_builder.add_ai_message(
            state, f"Market analysis completed for {state['symbol']} - {price_info}"
        )

        return state

    async def _memory_retrieval_node(self, state: TradingState) -> TradingState:
        """Node 2: Memory Retrieval - Find similar patterns from history"""
        logger.info(f"Retrieving memory patterns for {state['symbol']}")

        # Update workflow step
        await self.context_builder.update_workflow_step(
            state, "memory_retrieval", ["strategy_evaluation"]
        )

        # Build query for similar patterns
        query_text = f"trading pattern analysis {state['symbol']} "
        if state["market_data"].trend_direction:
            query_text += f"trend {state['market_data'].trend_direction}"

        # Add memory context
        state = await self.context_builder.add_memory_context(state, query_text)

        # Add memory retrieval message
        memory_count = len(state["memory_context"])
        await self.context_builder.add_ai_message(
            state,
            f"Memory retrieval completed - Found {memory_count} similar historical patterns",
        )

        return state

    async def _strategy_evaluation_node(self, state: TradingState) -> TradingState:
        """Node 3: Strategy Evaluation - Run multiple strategy analyses in parallel"""
        logger.info(f"Running strategy evaluation for {state['symbol']}")

        # Update workflow step
        await self.context_builder.update_workflow_step(
            state, "strategy_evaluation", ["risk_assessment"]
        )

        # Prepare market data for strategies
        market_data = {
            "symbol": state["symbol"],
            "prices": (
                [state["market_data"].current_price]
                if state["market_data"].current_price
                else []
            ),
            "volumes": [],  # Would need historical data
        }

        # Run strategies in parallel
        strategy_tasks = []
        for strategy_name, strategy in self.strategies.items():
            task = self._run_strategy_analysis(strategy_name, strategy, market_data)
            strategy_tasks.append(task)

        # Execute all strategies
        strategy_results = await asyncio.gather(*strategy_tasks, return_exceptions=True)

        # Process strategy results
        for i, result in enumerate(strategy_results):
            strategy_name = list(self.strategies.keys())[i]

            if isinstance(result, Exception):
                logger.warning(f"Strategy {strategy_name} failed: {result}")
                continue

            if result:
                # Create trading signal
                signal = TradingSignal(
                    agent_type="strategy",
                    strategy_name=strategy_name,
                    symbol=state["symbol"],
                    action=result.get("action", "HOLD"),
                    confidence=result.get("confidence", 0.0),
                    reasoning=result.get("reasoning", "Strategy analysis"),
                    metadata={"strategy_data": result},
                )

                # Add signal to state
                state = await self.context_builder.add_strategy_signal(state, signal)

        # Add strategy evaluation message
        signal_count = len(state["strategy_signals"])
        await self.context_builder.add_ai_message(
            state,
            f"Strategy evaluation completed - Generated {signal_count} trading signals",
        )

        return state

    async def _risk_assessment_node(self, state: TradingState) -> TradingState:
        """Node 4: Risk Assessment - Evaluate position and portfolio risk"""
        logger.info(f"Assessing risk for {state['symbol']}")

        # Update workflow step
        await self.context_builder.update_workflow_step(
            state, "risk_assessment", ["decision_synthesis"]
        )

        # Update risk context
        state = await self.context_builder.update_risk_context(state)

        # Add risk assessment message
        risk_status = "approved" if state["risk_approval"] else "requires review"
        risk_score = state["risk_metrics"].risk_score

        await self.context_builder.add_ai_message(
            state,
            f"Risk assessment completed - Status: {risk_status} (Risk Score: {risk_score:.1%})",
        )

        return state

    async def _decision_synthesis_node(self, state: TradingState) -> TradingState:
        """Node 5: Decision Synthesis - Combine all inputs for decision"""
        logger.info(f"Synthesizing decision for {state['symbol']}")

        # Update workflow step
        await self.context_builder.update_workflow_step(
            state, "decision_synthesis", ["execution_planning"]
        )

        # Create decision context
        state = await self.context_builder.create_decision_context(state)

        # Make final decision based on consensus
        decision_context = state["decision_context"]

        if decision_context.confidence_level > 0.7 and state["risk_approval"]:
            # Find most confident signal for action
            best_signal = max(
                state["strategy_signals"], key=lambda s: s.confidence, default=None
            )
            final_decision = best_signal.action if best_signal else "HOLD"
        else:
            final_decision = "HOLD"

        state["final_decision"] = final_decision

        # Add decision message
        await self.context_builder.add_ai_message(
            state,
            f"Decision synthesis completed - Final decision: {final_decision} "
            f"(Confidence: {decision_context.confidence_level:.1%})",
        )

        return state

    async def _execution_planning_node(self, state: TradingState) -> TradingState:
        """Node 6: Execution Planning - Plan order execution"""
        logger.info(f"Planning execution for {state['symbol']}")

        # Update workflow step
        await self.context_builder.update_workflow_step(
            state, "execution_planning", ["monitoring"]
        )

        # Create execution plan
        if state["final_decision"] and state["final_decision"] != "HOLD":
            state = await self.context_builder.create_execution_plan(
                state, state["final_decision"]
            )

            # Add execution planning message
            if state["execution_plan"]:
                plan = state["execution_plan"]
                await self.context_builder.add_ai_message(
                    state,
                    f"Execution plan created - {plan.action} {plan.quantity:.2f} shares "
                    f"via {plan.order_type} order",
                )
            else:
                await self.context_builder.add_ai_message(
                    state, "No execution plan needed (HOLD decision)"
                )
        else:
            await self.context_builder.add_ai_message(
                state, "No execution required - Decision: HOLD"
            )

        return state

    async def _monitoring_node(self, state: TradingState) -> TradingState:
        """Node 7: Monitoring - Track execution and outcomes"""
        logger.info(f"Setting up monitoring for {state['symbol']}")

        # Update workflow step
        await self.context_builder.update_workflow_step(state, "monitoring", [])

        # Set execution status
        if state["execution_plan"]:
            state["execution_status"] = "planned"
            await self.context_builder.add_ai_message(
                state, "Execution monitoring configured - Ready for trade execution"
            )
        else:
            state["execution_status"] = "no_action"
            await self.context_builder.add_ai_message(
                state, "Monitoring configured - No execution required"
            )

        return state

    def _risk_gate_condition(self, state: TradingState) -> str:
        """Risk gate condition - determines if workflow should proceed"""
        # Check for critical errors
        critical_errors = [
            error for error in state["errors"] if "critical" in error.lower()
        ]
        if critical_errors:
            return "halt"

        # Check risk approval
        if not state["risk_approval"] and any(
            signal.confidence > 0.5 and signal.action != "HOLD"
            for signal in state["strategy_signals"]
        ):
            # Only halt if we have actionable signals but no risk approval
            state["warnings"].append(
                "Risk gate: High-confidence signals require risk approval"
            )
            return "halt"

        return "proceed"

    def _decision_condition(self, state: TradingState) -> str:
        """Decision condition - determines execution path"""
        if state["final_decision"] and state["final_decision"] != "HOLD":
            return "execute"
        return "hold"

    def _execution_condition(self, state: TradingState) -> str:
        """Execution condition - determines if monitoring is needed"""
        if state["execution_plan"] and state["execution_plan"].risk_validated:
            return "monitor"
        return "complete"

    async def _run_strategy_analysis(
        self, strategy_name: str, strategy: StrategyPlugin, market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Run individual strategy analysis"""
        try:
            # Perform analysis
            analysis = strategy.analyze_market(market_data)

            # Generate signal
            signal = strategy.generate_signal(analysis, market_data)

            return signal

        except Exception as e:
            logger.error(f"Strategy {strategy_name} analysis failed: {e}")
            return None


# Global workflow instance
_workflow_instance: Optional[TradingWorkflow] = None


async def get_trading_workflow() -> TradingWorkflow:
    """Get the global trading workflow instance"""
    global _workflow_instance

    if _workflow_instance is None:
        _workflow_instance = TradingWorkflow()
        await _workflow_instance.initialize()

    return _workflow_instance


async def execute_trading_analysis(
    symbol: str, user_message: str = "", workflow_type: str = "standard_analysis"
) -> Dict[str, Any]:
    """Convenience function to execute trading analysis workflow"""
    workflow = await get_trading_workflow()
    return await workflow.execute_workflow(symbol, user_message, workflow_type)


__all__ = ["TradingWorkflow", "get_trading_workflow", "execute_trading_analysis"]
