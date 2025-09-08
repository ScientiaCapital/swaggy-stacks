"""
Agent Testing Framework - Comprehensive validation system for AI agent decision-making
Tests agents across various market scenarios with performance metrics and validation
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import traceback

import structlog

from .mock_data_generator import MockDataGenerator, MockMarketData, MarketRegime, mock_data_generator
from app.ai.trading_agents import AIAgentCoordinator
from app.analysis.tool_feedback_tracker import ToolFeedbackTracker, tool_feedback_tracker

logger = structlog.get_logger(__name__)


class TestResult(Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class AgentDecisionTest:
    """Individual agent decision test case"""
    test_id: str
    test_name: str
    symbol: str
    regime: MarketRegime
    market_data: MockMarketData
    technical_indicators: Dict[str, Any]
    markov_analysis: Dict[str, Any]
    expected_decision_range: List[str]  # Acceptable decisions (BUY, SELL, HOLD)
    min_confidence: float
    max_response_time_ms: float
    test_context: Dict[str, Any]


@dataclass
class AgentTestResult:
    """Result of agent decision test"""
    test_id: str
    agent_type: str
    test_name: str
    result: TestResult
    actual_decision: Optional[str]
    confidence: Optional[float]
    response_time_ms: float
    expected_decisions: List[str]
    decision_correct: bool
    confidence_adequate: bool
    response_time_acceptable: bool
    reasoning: str
    error_message: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class AgentPerformanceReport:
    """Comprehensive agent performance report"""
    agent_type: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    success_rate: float
    average_confidence: float
    average_response_time_ms: float
    decision_accuracy_by_regime: Dict[str, float]
    common_failures: List[str]
    performance_trends: Dict[str, Any]
    recommendations: List[str]
    test_duration_minutes: float
    timestamp: datetime


class AgentTestingFramework:
    """Comprehensive testing framework for AI agents"""
    
    def __init__(self, 
                 agent_coordinator: Optional[AIAgentCoordinator] = None,
                 feedback_tracker: Optional[ToolFeedbackTracker] = None,
                 mock_generator: Optional[MockDataGenerator] = None):
        
        self.agent_coordinator = agent_coordinator or AIAgentCoordinator(enable_streaming=False)  # Disable streaming for tests
        self.feedback_tracker = feedback_tracker or tool_feedback_tracker
        self.mock_generator = mock_generator or mock_data_generator
        
        # Test configuration
        self.test_results: List[AgentTestResult] = []
        self.test_callbacks: List[Callable[[AgentTestResult], None]] = []
        self.performance_reports: Dict[str, AgentPerformanceReport] = {}
        
        # Test scenarios
        self.test_scenarios: Dict[str, List[AgentDecisionTest]] = {}
        
    def add_test_callback(self, callback: Callable[[AgentTestResult], None]):
        """Add callback for real-time test result notifications"""
        self.test_callbacks.append(callback)
    
    def create_test_scenarios(self) -> Dict[str, List[AgentDecisionTest]]:
        """Create comprehensive test scenarios for different market regimes"""
        
        logger.info("Creating test scenarios for agent validation")
        
        test_scenarios = {}
        test_symbols = ["AAPL", "TSLA", "SPY"]
        
        for regime in MarketRegime:
            scenario_tests = []
            
            for symbol in test_symbols:
                # Generate mock data for the regime
                market_data_points = self.mock_generator.generate_market_scenario(
                    symbol=symbol,
                    regime=regime,
                    duration_minutes=60,  # 1 hour of data
                    interval_minutes=5
                )
                
                for i, market_data in enumerate(market_data_points[::3]):  # Every 3rd point
                    # Generate technical indicators
                    tech_indicators = self.mock_generator.generate_technical_indicators(symbol, market_data)
                    markov_analysis = self.mock_generator.generate_markov_analysis(symbol, market_data, regime)
                    
                    # Define expected decision ranges based on regime
                    expected_decisions = self._get_expected_decisions_for_regime(regime, tech_indicators.to_dict())
                    
                    test = AgentDecisionTest(
                        test_id=f"test_{regime.value}_{symbol}_{i}",
                        test_name=f"{regime.value.title()} Market - {symbol}",
                        symbol=symbol,
                        regime=regime,
                        market_data=market_data,
                        technical_indicators=tech_indicators.to_dict(),
                        markov_analysis=markov_analysis.to_dict(),
                        expected_decision_range=expected_decisions,
                        min_confidence=0.5,
                        max_response_time_ms=5000,  # 5 seconds
                        test_context={
                            "regime": regime.value,
                            "market_condition": self._assess_market_condition(tech_indicators.to_dict()),
                            "volatility_level": "high" if regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.BREAKOUT_BULLISH, MarketRegime.BREAKOUT_BEARISH] else "normal"
                        }
                    )
                    
                    scenario_tests.append(test)
            
            test_scenarios[regime.value] = scenario_tests
        
        self.test_scenarios = test_scenarios
        
        total_tests = sum(len(tests) for tests in test_scenarios.values())
        logger.info("Test scenarios created", 
                   regimes=len(test_scenarios),
                   total_tests=total_tests)
        
        return test_scenarios
    
    def _get_expected_decisions_for_regime(self, regime: MarketRegime, tech_indicators: Dict[str, Any]) -> List[str]:
        """Determine expected decision range for a market regime"""
        
        rsi = tech_indicators.get("rsi", 50)
        macd = tech_indicators.get("macd", 0)
        
        if regime == MarketRegime.TRENDING_BULLISH:
            return ["BUY", "HOLD"] if rsi < 70 else ["HOLD"]
        elif regime == MarketRegime.TRENDING_BEARISH:
            return ["SELL", "HOLD"] if rsi > 30 else ["HOLD"]
        elif regime == MarketRegime.BREAKOUT_BULLISH:
            return ["BUY"]
        elif regime == MarketRegime.BREAKOUT_BEARISH:
            return ["SELL"]
        elif regime == MarketRegime.HIGH_VOLATILITY:
            return ["HOLD", "BUY", "SELL"]  # Any decision acceptable in high volatility
        elif regime == MarketRegime.LOW_VOLATILITY:
            return ["HOLD"]
        elif regime in [MarketRegime.SIDEWAYS, MarketRegime.MEAN_REVERSION]:
            if rsi > 70:
                return ["SELL", "HOLD"]
            elif rsi < 30:
                return ["BUY", "HOLD"]  
            else:
                return ["HOLD"]
        
        return ["BUY", "SELL", "HOLD"]  # Default: any decision acceptable
    
    def _assess_market_condition(self, tech_indicators: Dict[str, Any]) -> str:
        """Assess market condition from technical indicators"""
        
        rsi = tech_indicators.get("rsi", 50)
        macd = tech_indicators.get("macd", 0)
        bb_position = self._calculate_bollinger_position(tech_indicators)
        
        if rsi > 70 and macd > 0:
            return "overbought_bullish"
        elif rsi < 30 and macd < 0:
            return "oversold_bearish"
        elif bb_position > 0.8:
            return "near_upper_band"
        elif bb_position < 0.2:
            return "near_lower_band"
        elif abs(macd) < 0.5:
            return "consolidating"
        else:
            return "neutral"
    
    def _calculate_bollinger_position(self, tech_indicators: Dict[str, Any]) -> float:
        """Calculate position within Bollinger Bands (0-1 scale)"""
        
        price = tech_indicators.get("close", 100)
        bb_upper = tech_indicators.get("bb_upper", 105)
        bb_lower = tech_indicators.get("bb_lower", 95)
        
        if bb_upper == bb_lower:
            return 0.5
        
        position = (price - bb_lower) / (bb_upper - bb_lower)
        return max(0, min(1, position))
    
    async def run_single_agent_test(self, test: AgentDecisionTest, agent_type: str) -> AgentTestResult:
        """Run a single test case for a specific agent"""
        
        start_time = datetime.now()
        
        try:
            # Prepare test data
            market_data = test.market_data.to_dict()
            technical_indicators = test.technical_indicators
            markov_analysis = test.markov_analysis
            
            # Mock account info
            account_info = {
                "equity": 100000,
                "buying_power": 100000,
                "cash": 50000
            }
            
            current_positions = []  # No existing positions for tests
            
            # Execute agent analysis
            if agent_type == "comprehensive":
                # Test full comprehensive analysis
                result = await self.agent_coordinator.comprehensive_analysis(
                    symbol=test.symbol,
                    market_data=market_data,
                    technical_indicators=technical_indicators,
                    account_info=account_info,
                    current_positions=current_positions,
                    markov_analysis=markov_analysis
                )
                
                actual_decision = result.get("final_recommendation", "UNKNOWN")
                confidence = self._extract_confidence_from_comprehensive(result)
                reasoning = self._extract_reasoning_from_comprehensive(result)
                
            elif agent_type == "market_analyst":
                # Test market analyst specifically
                result = await self.agent_coordinator.stream_market_analysis(
                    symbol=test.symbol,
                    market_data=market_data,
                    technical_indicators=technical_indicators
                )
                
                actual_decision = result.sentiment
                confidence = result.confidence
                reasoning = result.reasoning
                
            elif agent_type == "risk_advisor":
                # Test risk advisor specifically
                result = await self.agent_coordinator.stream_risk_assessment(
                    symbol=test.symbol,
                    position_size=2000.0,  # $2k position
                    account_value=100000.0,
                    current_positions=current_positions,
                    market_volatility={"atr": technical_indicators.get("atr", 2.0)},
                    proposed_trade={"stop_loss_percent": 0.05, "take_profit_percent": 0.10}
                )
                
                actual_decision = f"RISK_{result.risk_level.upper()}"
                confidence = result.confidence
                reasoning = result.reasoning
                
            elif agent_type == "strategy_optimizer":
                # Test strategy optimizer specifically
                result = await self.agent_coordinator.stream_strategy_signal(
                    symbol=test.symbol,
                    markov_analysis=markov_analysis,
                    technical_indicators=technical_indicators,
                    market_context=test.test_context,
                    performance_history=[]
                )
                
                actual_decision = result.action
                confidence = result.confidence
                reasoning = result.reasoning
            
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Calculate response time
            response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Validate results
            decision_correct = actual_decision in test.expected_decision_range
            confidence_adequate = confidence >= test.min_confidence
            response_time_acceptable = response_time_ms <= test.max_response_time_ms
            
            # Determine test result
            if decision_correct and confidence_adequate and response_time_acceptable:
                test_result = TestResult.PASSED
            else:
                test_result = TestResult.FAILED
            
            # Create test result
            agent_test_result = AgentTestResult(
                test_id=test.test_id,
                agent_type=agent_type,
                test_name=test.test_name,
                result=test_result,
                actual_decision=actual_decision,
                confidence=confidence,
                response_time_ms=response_time_ms,
                expected_decisions=test.expected_decision_range,
                decision_correct=decision_correct,
                confidence_adequate=confidence_adequate,
                response_time_acceptable=response_time_acceptable,
                reasoning=reasoning,
                error_message=None,
                timestamp=datetime.now(),
                metadata={
                    "symbol": test.symbol,
                    "regime": test.regime.value,
                    "market_condition": test.test_context.get("market_condition"),
                    "volatility_level": test.test_context.get("volatility_level")
                }
            )
            
        except Exception as e:
            # Handle test errors
            response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            agent_test_result = AgentTestResult(
                test_id=test.test_id,
                agent_type=agent_type,
                test_name=test.test_name,
                result=TestResult.ERROR,
                actual_decision=None,
                confidence=None,
                response_time_ms=response_time_ms,
                expected_decisions=test.expected_decision_range,
                decision_correct=False,
                confidence_adequate=False,
                response_time_acceptable=False,
                reasoning="Test execution failed",
                error_message=str(e),
                timestamp=datetime.now(),
                metadata={
                    "symbol": test.symbol,
                    "regime": test.regime.value,
                    "error_traceback": traceback.format_exc()
                }
            )
        
        # Store result and trigger callbacks
        self.test_results.append(agent_test_result)
        
        for callback in self.test_callbacks:
            try:
                await callback(agent_test_result)
            except Exception as e:
                logger.warning("Test callback failed", error=str(e))
        
        return agent_test_result
    
    def _extract_confidence_from_comprehensive(self, result: Dict[str, Any]) -> float:
        """Extract overall confidence from comprehensive analysis result"""
        
        confidences = []
        
        if "market_analysis" in result and "confidence" in result["market_analysis"]:
            confidences.append(result["market_analysis"]["confidence"])
            
        if "risk_assessment" in result and "confidence" in result["risk_assessment"]:
            confidences.append(result["risk_assessment"]["confidence"])
            
        if "strategy_signal" in result and "confidence" in result["strategy_signal"]:
            confidences.append(result["strategy_signal"]["confidence"])
        
        return statistics.mean(confidences) if confidences else 0.5
    
    def _extract_reasoning_from_comprehensive(self, result: Dict[str, Any]) -> str:
        """Extract combined reasoning from comprehensive analysis result"""
        
        reasoning_parts = []
        
        if "market_analysis" in result and "reasoning" in result["market_analysis"]:
            reasoning_parts.append(f"Market: {result['market_analysis']['reasoning']}")
            
        if "risk_assessment" in result and "reasoning" in result["risk_assessment"]:
            reasoning_parts.append(f"Risk: {result['risk_assessment']['reasoning']}")
            
        if "strategy_signal" in result and "reasoning" in result["strategy_signal"]:
            reasoning_parts.append(f"Strategy: {result['strategy_signal']['reasoning']}")
        
        return " | ".join(reasoning_parts) if reasoning_parts else "Comprehensive analysis"
    
    async def run_comprehensive_agent_test(
        self, 
        regime_filter: Optional[List[str]] = None,
        agent_types: List[str] = ["comprehensive", "market_analyst", "risk_advisor", "strategy_optimizer"]
    ) -> Dict[str, AgentPerformanceReport]:
        """Run comprehensive test suite across all agents and scenarios"""
        
        start_time = datetime.now()
        
        logger.info("Starting comprehensive agent testing",
                   agent_types=agent_types,
                   regime_filter=regime_filter)
        
        # Create test scenarios if not already created
        if not self.test_scenarios:
            self.create_test_scenarios()
        
        # Filter scenarios if requested
        test_scenarios = self.test_scenarios
        if regime_filter:
            test_scenarios = {k: v for k, v in self.test_scenarios.items() if k in regime_filter}
        
        # Run tests for each agent type
        for agent_type in agent_types:
            logger.info(f"Testing agent type: {agent_type}")
            
            agent_test_results = []
            
            for regime, tests in test_scenarios.items():
                logger.info(f"Running {len(tests)} tests for regime: {regime}")
                
                # Run tests in batches to avoid overwhelming the system
                batch_size = 5
                for i in range(0, len(tests), batch_size):
                    batch = tests[i:i + batch_size]
                    
                    # Run batch concurrently
                    tasks = [self.run_single_agent_test(test, agent_type) for test in batch]
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in batch_results:
                        if isinstance(result, AgentTestResult):
                            agent_test_results.append(result)
                        else:
                            logger.error("Test batch execution failed", error=str(result))
                    
                    # Small delay between batches
                    await asyncio.sleep(1)
            
            # Generate performance report for this agent
            performance_report = self._generate_performance_report(agent_type, agent_test_results)
            self.performance_reports[agent_type] = performance_report
        
        test_duration = (datetime.now() - start_time).total_seconds() / 60
        
        logger.info("Comprehensive agent testing completed",
                   agent_types=len(agent_types),
                   total_tests=len(self.test_results),
                   duration_minutes=test_duration)
        
        return self.performance_reports
    
    def _generate_performance_report(self, agent_type: str, test_results: List[AgentTestResult]) -> AgentPerformanceReport:
        """Generate performance report for an agent"""
        
        if not test_results:
            return AgentPerformanceReport(
                agent_type=agent_type,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                error_tests=0,
                success_rate=0.0,
                average_confidence=0.0,
                average_response_time_ms=0.0,
                decision_accuracy_by_regime={},
                common_failures=[],
                performance_trends={},
                recommendations=[],
                test_duration_minutes=0.0,
                timestamp=datetime.now()
            )
        
        # Basic metrics
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results if result.result == TestResult.PASSED)
        failed_tests = sum(1 for result in test_results if result.result == TestResult.FAILED)
        error_tests = sum(1 for result in test_results if result.result == TestResult.ERROR)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Confidence and response time metrics
        valid_results = [r for r in test_results if r.confidence is not None]
        average_confidence = statistics.mean([r.confidence for r in valid_results]) if valid_results else 0.0
        average_response_time_ms = statistics.mean([r.response_time_ms for r in test_results])
        
        # Decision accuracy by regime
        decision_accuracy_by_regime = {}
        regime_groups = {}
        for result in test_results:
            regime = result.metadata.get("regime", "unknown")
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append(result)
        
        for regime, regime_results in regime_groups.items():
            correct_decisions = sum(1 for r in regime_results if r.decision_correct)
            decision_accuracy_by_regime[regime] = correct_decisions / len(regime_results)
        
        # Common failure patterns
        common_failures = []
        failure_reasons = {}
        for result in test_results:
            if result.result == TestResult.FAILED:
                reasons = []
                if not result.decision_correct:
                    reasons.append(f"Wrong decision: got {result.actual_decision}, expected {result.expected_decisions}")
                if not result.confidence_adequate:
                    reasons.append(f"Low confidence: {result.confidence:.2f}")
                if not result.response_time_acceptable:
                    reasons.append(f"Slow response: {result.response_time_ms:.0f}ms")
                
                for reason in reasons:
                    if reason not in failure_reasons:
                        failure_reasons[reason] = 0
                    failure_reasons[reason] += 1
        
        # Top 5 most common failures
        common_failures = sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
        common_failures = [f"{reason} ({count} times)" for reason, count in common_failures]
        
        # Performance trends (simplified)
        performance_trends = {
            "confidence_trend": "stable",  # Could analyze trend over time
            "response_time_trend": "stable",
            "accuracy_trend": "stable"
        }
        
        # Generate recommendations
        recommendations = []
        if success_rate < 0.8:
            recommendations.append(f"Overall success rate is low ({success_rate:.1%})")
        if average_confidence < 0.6:
            recommendations.append(f"Average confidence is low ({average_confidence:.1%})")
        if average_response_time_ms > 3000:
            recommendations.append(f"Average response time is high ({average_response_time_ms:.0f}ms)")
        
        # Regime-specific recommendations
        for regime, accuracy in decision_accuracy_by_regime.items():
            if accuracy < 0.7:
                recommendations.append(f"Poor performance in {regime} regime ({accuracy:.1%} accuracy)")
        
        return AgentPerformanceReport(
            agent_type=agent_type,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            success_rate=success_rate,
            average_confidence=average_confidence,
            average_response_time_ms=average_response_time_ms,
            decision_accuracy_by_regime=decision_accuracy_by_regime,
            common_failures=common_failures,
            performance_trends=performance_trends,
            recommendations=recommendations,
            test_duration_minutes=0.0,  # Would be calculated from test execution
            timestamp=datetime.now()
        )
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive testing report"""
        
        total_tests = len(self.test_results)
        if total_tests == 0:
            return {"status": "no_tests_run", "message": "No tests have been executed"}
        
        # Overall statistics
        passed_tests = sum(1 for result in self.test_results if result.result == TestResult.PASSED)
        failed_tests = sum(1 for result in self.test_results if result.result == TestResult.FAILED)
        error_tests = sum(1 for result in self.test_results if result.result == TestResult.ERROR)
        
        # Agent-specific statistics
        agent_stats = {}
        for agent_type, performance_report in self.performance_reports.items():
            agent_stats[agent_type] = asdict(performance_report)
        
        # Test execution summary
        report = {
            "executive_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "overall_success_rate": passed_tests / total_tests,
                "agents_tested": len(self.performance_reports),
                "regimes_tested": len(set(r.metadata.get("regime") for r in self.test_results))
            },
            "agent_performance": agent_stats,
            "test_execution_details": {
                "start_time": min(r.timestamp for r in self.test_results).isoformat(),
                "end_time": max(r.timestamp for r in self.test_results).isoformat(),
                "test_results": [asdict(result) for result in self.test_results[-50:]]  # Last 50 results
            },
            "recommendations": self._generate_overall_recommendations(),
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall testing recommendations"""
        
        recommendations = []
        
        if not self.performance_reports:
            return ["No performance reports available for analysis"]
        
        # Overall performance assessment
        avg_success_rate = statistics.mean(report.success_rate for report in self.performance_reports.values())
        
        if avg_success_rate < 0.7:
            recommendations.append("Overall agent performance is below acceptable threshold (70%)")
        elif avg_success_rate > 0.9:
            recommendations.append("Excellent overall agent performance - consider more challenging test scenarios")
        
        # Agent comparison
        agent_success_rates = {agent: report.success_rate for agent, report in self.performance_reports.items()}
        best_agent = max(agent_success_rates.items(), key=lambda x: x[1])
        worst_agent = min(agent_success_rates.items(), key=lambda x: x[1])
        
        if len(agent_success_rates) > 1:
            recommendations.append(f"Best performing agent: {best_agent[0]} ({best_agent[1]:.1%} success rate)")
            recommendations.append(f"Worst performing agent: {worst_agent[0]} ({worst_agent[1]:.1%} success rate)")
        
        return recommendations
    
    async def run_stress_test(
        self,
        concurrent_agents: int = 5,
        test_duration_minutes: int = 10,
        requests_per_minute: int = 30
    ) -> Dict[str, Any]:
        """Run stress test to validate agent performance under load"""
        
        logger.info("Starting agent stress test",
                   concurrent_agents=concurrent_agents,
                   duration_minutes=test_duration_minutes,
                   requests_per_minute=requests_per_minute)
        
        start_time = datetime.now()
        stress_test_results = []
        
        # Create a subset of test scenarios for stress testing
        if not self.test_scenarios:
            self.create_test_scenarios()
        
        # Use a smaller set of representative tests
        stress_tests = []
        for regime, tests in list(self.test_scenarios.items())[:3]:  # First 3 regimes
            stress_tests.extend(tests[:5])  # 5 tests per regime
        
        async def stress_worker(worker_id: int):
            """Individual stress test worker"""
            worker_results = []
            end_time = start_time + timedelta(minutes=test_duration_minutes)
            request_interval = 60.0 / requests_per_minute  # seconds between requests
            
            while datetime.now() < end_time:
                test = random.choice(stress_tests)
                
                try:
                    result = await self.run_single_agent_test(test, "comprehensive")
                    result.metadata["worker_id"] = worker_id
                    worker_results.append(result)
                except Exception as e:
                    logger.error(f"Stress test worker {worker_id} failed", error=str(e))
                
                await asyncio.sleep(request_interval)
            
            return worker_results
        
        # Run concurrent stress workers
        tasks = [stress_worker(i) for i in range(concurrent_agents)]
        worker_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for results in worker_results:
            if isinstance(results, list):
                stress_test_results.extend(results)
        
        # Analyze stress test performance
        duration_minutes = (datetime.now() - start_time).total_seconds() / 60
        total_requests = len(stress_test_results)
        successful_requests = sum(1 for r in stress_test_results if r.result == TestResult.PASSED)
        
        # Calculate performance metrics
        response_times = [r.response_time_ms for r in stress_test_results]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = sorted(response_times)[int(0.95 * len(response_times))] if len(response_times) >= 20 else max(response_times, default=0)
        
        stress_report = {
            "stress_test_config": {
                "concurrent_agents": concurrent_agents,
                "duration_minutes": test_duration_minutes,
                "target_requests_per_minute": requests_per_minute
            },
            "performance_metrics": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
                "actual_duration_minutes": duration_minutes,
                "actual_requests_per_minute": total_requests / duration_minutes if duration_minutes > 0 else 0,
                "average_response_time_ms": avg_response_time,
                "p95_response_time_ms": p95_response_time
            },
            "performance_assessment": "healthy" if successful_requests / total_requests > 0.9 and avg_response_time < 3000 else "degraded",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Stress test completed",
                   total_requests=total_requests,
                   success_rate=successful_requests / total_requests if total_requests > 0 else 0,
                   avg_response_time_ms=avg_response_time)
        
        return stress_report


# Global testing framework instance
agent_testing_framework = AgentTestingFramework()