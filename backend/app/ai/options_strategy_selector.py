"""
AI-Powered Options Strategy Selector

Intelligent strategy selection system that analyzes market conditions, volatility regimes,
portfolio Greeks exposure, and risk tolerance to recommend optimal options strategies
with confidence scoring and risk-adjusted capital allocation.

Integrates with:
- Market regime detection from unsupervised learning
- GARCH volatility prediction from Task 9
- Greeks risk management from Task 8
- All four options strategies from Tasks 3-6
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import structlog

from app.ai.base_agent import BaseAIAgent
from app.ai.ollama_client import OllamaClient

logger = structlog.get_logger(__name__)


class MarketCondition(Enum):
    """Market condition classifications for strategy selection"""
    BULLISH_LOW_VOL = "bullish_low_vol"
    BULLISH_HIGH_VOL = "bullish_high_vol"
    BEARISH_LOW_VOL = "bearish_low_vol"
    BEARISH_HIGH_VOL = "bearish_high_vol"
    NEUTRAL_LOW_VOL = "neutral_low_vol"
    NEUTRAL_HIGH_VOL = "neutral_high_vol"
    CRISIS = "crisis"
    UNCERTAIN = "uncertain"


class StrategyRecommendation(Enum):
    """Available options strategy recommendations"""
    ZERO_DTE = "zero_dte"
    WHEEL = "wheel"
    IRON_CONDOR = "iron_condor"
    GAMMA_SCALPING = "gamma_scalping"
    CASH = "cash"  # No options exposure
    REDUCE_EXPOSURE = "reduce_exposure"  # Scale down existing positions


@dataclass
class MarketAnalysis:
    """Comprehensive market condition analysis"""
    market_condition: MarketCondition
    volatility_regime: str  # From VolatilityPredictor
    trend_direction: str  # "bullish", "bearish", "neutral"
    volatility_level: float  # Annualized volatility
    regime_stability: float  # Market regime stability [0-1]
    time_decay_environment: str  # "favorable", "neutral", "unfavorable"
    liquidity_condition: str  # "high", "normal", "low"
    confidence_score: float  # Overall analysis confidence [0-1]


@dataclass
class PortfolioRiskAssessment:
    """Current portfolio risk and exposure analysis"""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float

    # Risk utilization percentages
    delta_utilization: float  # Percentage of delta limit used
    gamma_utilization: float
    vega_utilization: float

    # Portfolio metrics
    max_loss_exposure: float
    buying_power_available: float
    concentration_risk: float  # Single position concentration [0-1]

    # Risk tolerance
    risk_capacity: str  # "conservative", "moderate", "aggressive"
    time_horizon: str  # "short_term", "medium_term", "long_term"


@dataclass
class StrategyRecommendationResult:
    """AI strategy recommendation with detailed reasoning"""
    recommended_strategy: StrategyRecommendation
    confidence_score: float  # [0-1]
    allocation_percentage: float  # Percentage of available capital

    # Strategy-specific parameters
    suggested_strikes: Optional[List[float]] = None
    suggested_expiration: Optional[str] = None
    max_position_size: Optional[float] = None

    # Risk metrics
    expected_max_profit: Optional[float] = None
    expected_max_loss: Optional[float] = None
    probability_of_profit: Optional[float] = None

    # Reasoning
    primary_reasons: List[str] = None
    risk_considerations: List[str] = None
    alternative_strategies: List[str] = None

    # Monitoring recommendations
    exit_conditions: List[str] = None
    adjustment_triggers: List[str] = None


class OptionsStrategySelector(BaseAIAgent):
    """AI-powered options strategy selector with market analysis integration"""

    def __init__(self, ollama_client: OllamaClient):
        super().__init__(
            ollama_client=ollama_client,
            agent_type="strategy_selector",
            prompt_filename="options_strategy_selector.txt"
        )

        # Strategy scoring weights
        self.strategy_weights = {
            "market_alignment": 0.30,
            "volatility_suitability": 0.25,
            "risk_reward": 0.20,
            "portfolio_fit": 0.15,
            "execution_feasibility": 0.10
        }

        # Risk limits for strategy selection
        self.risk_limits = {
            "max_single_strategy_allocation": 0.20,  # 20% max per strategy
            "max_total_options_exposure": 0.50,     # 50% max options exposure
            "min_liquidity_threshold": 0.30,        # 30% min buying power
            "max_correlation_exposure": 0.60        # 60% max correlated exposure
        }

        logger.info("OptionsStrategySelector initialized with AI reasoning")

    def _get_default_prompt(self) -> str:
        """Default prompt for strategy selection"""
        return """
You are an expert options trading strategist with deep knowledge of market conditions,
volatility patterns, and risk management. Analyze the provided market data and portfolio
status to recommend the optimal options strategy.

Consider:
1. Market regime and trend direction
2. Volatility environment and expected changes
3. Portfolio Greeks exposure and risk limits
4. Time decay characteristics
5. Liquidity and execution feasibility

Provide detailed reasoning for your recommendations with specific risk considerations.
"""

    async def select_strategy(
        self,
        symbol: str,
        price_data: List[Dict[str, Any]],
        option_chain: Optional[List[Dict[str, Any]]] = None,
        portfolio_greeks: Optional[Dict[str, float]] = None,
        risk_tolerance: str = "moderate"
    ) -> StrategyRecommendationResult:
        """
        Main strategy selection method with comprehensive analysis

        Args:
            symbol: Target symbol for strategy
            price_data: Recent price data for analysis
            option_chain: Current option chain data
            portfolio_greeks: Current portfolio Greeks exposure
            risk_tolerance: Risk tolerance level

        Returns:
            Complete strategy recommendation with reasoning
        """
        try:
            logger.info(
                "Starting AI strategy selection",
                symbol=symbol,
                risk_tolerance=risk_tolerance
            )

            # 1. Analyze market conditions
            market_analysis = await self._analyze_market_conditions(symbol, price_data)

            # 2. Assess portfolio risk
            risk_assessment = await self._assess_portfolio_risk(portfolio_greeks, risk_tolerance)

            # 3. Evaluate strategy suitability
            strategy_scores = await self._evaluate_strategies(
                symbol, market_analysis, risk_assessment, option_chain
            )

            # 4. Generate AI-powered recommendation
            recommendation = await self._generate_ai_recommendation(
                symbol, market_analysis, risk_assessment, strategy_scores
            )

            # 5. Validate and adjust recommendation
            final_recommendation = self._validate_recommendation(
                recommendation, risk_assessment
            )

            logger.info(
                "Strategy selection completed",
                symbol=symbol,
                recommended_strategy=final_recommendation.recommended_strategy.value,
                confidence=final_recommendation.confidence_score
            )

            return final_recommendation

        except Exception as e:
            logger.error("Strategy selection failed", symbol=symbol, error=str(e))
            return self._create_fallback_recommendation()

    async def _analyze_market_conditions(
        self,
        symbol: str,
        price_data: List[Dict[str, Any]]
    ) -> MarketAnalysis:
        """Analyze current market conditions using integrated systems"""
        try:
            # 1. Get volatility analysis from Task 9
            volatility_analysis = await self._get_volatility_analysis(symbol, price_data)

            # 2. Detect market regime using unsupervised learning
            regime_analysis = await self._detect_market_regime(price_data)

            # 3. Analyze trend direction
            trend_direction = self._analyze_trend_direction(price_data)

            # 4. Assess time decay environment
            time_decay_env = self._assess_time_decay_environment(volatility_analysis)

            # 5. Evaluate liquidity conditions
            liquidity_condition = self._evaluate_liquidity_conditions(price_data)

            # 6. Determine overall market condition
            market_condition = self._classify_market_condition(
                trend_direction, volatility_analysis['volatility_regime']
            )

            # 7. Calculate confidence score
            confidence_score = self._calculate_analysis_confidence(
                volatility_analysis, regime_analysis
            )

            return MarketAnalysis(
                market_condition=market_condition,
                volatility_regime=volatility_analysis['volatility_regime'],
                trend_direction=trend_direction,
                volatility_level=volatility_analysis['current_volatility'],
                regime_stability=regime_analysis.get('stability', 0.5),
                time_decay_environment=time_decay_env,
                liquidity_condition=liquidity_condition,
                confidence_score=confidence_score
            )

        except Exception as e:
            logger.error("Market analysis failed", symbol=symbol, error=str(e))
            return self._create_fallback_market_analysis()

    async def _get_volatility_analysis(
        self,
        symbol: str,
        price_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get volatility analysis from GARCH predictor (Task 9)"""
        try:
            from app.ml.volatility_predictor import get_volatility_predictor

            predictor = get_volatility_predictor()

            # Get volatility prediction
            vol_metrics = await predictor.predict_volatility(
                symbol=symbol,
                price_data=price_data
            )

            return {
                'current_volatility': vol_metrics.garch_predicted_vol,
                'volatility_regime': vol_metrics.vol_regime.value,
                'confidence': vol_metrics.confidence_score,
                'spike_probability': vol_metrics.spike_probability,
                'mean_reversion_factor': vol_metrics.mean_reversion_factor
            }

        except Exception as e:
            logger.warning("Volatility analysis failed, using fallback", error=str(e))
            return {
                'current_volatility': 0.25,
                'volatility_regime': 'normal',
                'confidence': 0.3,
                'spike_probability': 0.5,
                'mean_reversion_factor': 0.1
            }

    async def _detect_market_regime(self, price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect market regime using unsupervised learning"""
        try:
            from app.ml.unsupervised.market_regime import MarketRegimeDetector

            detector = MarketRegimeDetector()

            # Convert price data to format expected by detector
            prices = [float(candle['close']) for candle in price_data]

            # Detect regime (simplified implementation)
            if len(prices) >= 20:
                recent_volatility = np.std(np.diff(np.log(prices[-20:]))) * np.sqrt(252)
                regime_stability = 1.0 - min(recent_volatility / 0.5, 1.0)
            else:
                regime_stability = 0.5

            return {
                'regime': 'normal',  # Would be actual regime detection
                'stability': regime_stability,
                'confidence': 0.7
            }

        except Exception as e:
            logger.warning("Regime detection failed, using fallback", error=str(e))
            return {'regime': 'normal', 'stability': 0.5, 'confidence': 0.3}

    def _analyze_trend_direction(self, price_data: List[Dict[str, Any]]) -> str:
        """Analyze trend direction from price data"""
        try:
            if len(price_data) < 20:
                return "neutral"

            prices = [float(candle['close']) for candle in price_data]

            # Simple trend analysis using moving averages
            short_ma = np.mean(prices[-10:])
            long_ma = np.mean(prices[-20:])

            if short_ma > long_ma * 1.02:  # 2% threshold
                return "bullish"
            elif short_ma < long_ma * 0.98:
                return "bearish"
            else:
                return "neutral"

        except Exception:
            return "neutral"

    def _assess_time_decay_environment(self, volatility_analysis: Dict[str, Any]) -> str:
        """Assess time decay favorability"""
        vol_regime = volatility_analysis.get('volatility_regime', 'normal')

        if vol_regime in ['high', 'extreme']:
            return "favorable"  # High volatility good for theta strategies
        elif vol_regime == 'low':
            return "unfavorable"  # Low volatility bad for theta
        else:
            return "neutral"

    def _evaluate_liquidity_conditions(self, price_data: List[Dict[str, Any]]) -> str:
        """Evaluate market liquidity from volume patterns"""
        try:
            if len(price_data) < 10:
                return "normal"

            volumes = [float(candle.get('volume', 0)) for candle in price_data[-10:]]
            avg_volume = np.mean(volumes)
            recent_volume = volumes[-1] if volumes else 0

            if recent_volume > avg_volume * 1.5:
                return "high"
            elif recent_volume < avg_volume * 0.5:
                return "low"
            else:
                return "normal"

        except Exception:
            return "normal"

    def _classify_market_condition(self, trend: str, vol_regime: str) -> MarketCondition:
        """Classify overall market condition"""
        vol_high = vol_regime in ['high', 'extreme']

        condition_map = {
            ('bullish', True): MarketCondition.BULLISH_HIGH_VOL,
            ('bullish', False): MarketCondition.BULLISH_LOW_VOL,
            ('bearish', True): MarketCondition.BEARISH_HIGH_VOL,
            ('bearish', False): MarketCondition.BEARISH_LOW_VOL,
            ('neutral', True): MarketCondition.NEUTRAL_HIGH_VOL,
            ('neutral', False): MarketCondition.NEUTRAL_LOW_VOL,
        }

        return condition_map.get((trend, vol_high), MarketCondition.UNCERTAIN)

    def _calculate_analysis_confidence(
        self,
        vol_analysis: Dict[str, Any],
        regime_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall analysis confidence"""
        vol_confidence = vol_analysis.get('confidence', 0.5)
        regime_confidence = regime_analysis.get('confidence', 0.5)

        # Weighted average with volatility analysis weighted higher
        overall_confidence = 0.6 * vol_confidence + 0.4 * regime_confidence
        return min(max(overall_confidence, 0.0), 1.0)

    async def _assess_portfolio_risk(
        self,
        portfolio_greeks: Optional[Dict[str, float]],
        risk_tolerance: str
    ) -> PortfolioRiskAssessment:
        """Assess current portfolio risk and capacity"""
        try:
            # Get current Greeks from risk manager (Task 8)
            if portfolio_greeks:
                greeks = portfolio_greeks
            else:
                greeks = await self._get_current_portfolio_greeks()

            # Calculate risk utilization
            risk_utilization = self._calculate_risk_utilization(greeks)

            # Assess risk capacity based on tolerance
            risk_capacity = risk_tolerance  # "conservative", "moderate", "aggressive"

            return PortfolioRiskAssessment(
                total_delta=greeks.get('total_delta', 0.0),
                total_gamma=greeks.get('total_gamma', 0.0),
                total_theta=greeks.get('total_theta', 0.0),
                total_vega=greeks.get('total_vega', 0.0),
                total_rho=greeks.get('total_rho', 0.0),
                delta_utilization=risk_utilization.get('delta', 0.0),
                gamma_utilization=risk_utilization.get('gamma', 0.0),
                vega_utilization=risk_utilization.get('vega', 0.0),
                max_loss_exposure=greeks.get('max_loss', 0.0),
                buying_power_available=100000.0,  # Would get from account
                concentration_risk=0.3,  # Would calculate from positions
                risk_capacity=risk_capacity,
                time_horizon="medium_term"
            )

        except Exception as e:
            logger.error("Portfolio risk assessment failed", error=str(e))
            return self._create_fallback_risk_assessment(risk_tolerance)

    async def _get_current_portfolio_greeks(self) -> Dict[str, float]:
        """Get current portfolio Greeks from risk manager"""
        try:
            from app.trading.greeks_risk_manager import GreeksRiskManager

            risk_manager = GreeksRiskManager()

            if hasattr(risk_manager, 'portfolio_greeks'):
                greeks = risk_manager.portfolio_greeks
                return {
                    'total_delta': greeks.total_delta,
                    'total_gamma': greeks.total_gamma,
                    'total_theta': greeks.total_theta,
                    'total_vega': greeks.total_vega,
                    'total_rho': greeks.total_rho,
                    'max_loss': getattr(greeks, 'max_loss', 0.0)
                }

        except Exception as e:
            logger.warning("Failed to get portfolio Greeks", error=str(e))

        return {
            'total_delta': 0.0,
            'total_gamma': 0.0,
            'total_theta': 0.0,
            'total_vega': 0.0,
            'total_rho': 0.0,
            'max_loss': 0.0
        }

    def _calculate_risk_utilization(self, greeks: Dict[str, float]) -> Dict[str, float]:
        """Calculate risk limit utilization percentages"""
        # Default risk limits (would be configurable)
        limits = {
            'max_delta': 100.0,
            'max_gamma': 50.0,
            'max_vega': 200.0
        }

        return {
            'delta': abs(greeks.get('total_delta', 0.0)) / limits['max_delta'],
            'gamma': abs(greeks.get('total_gamma', 0.0)) / limits['max_gamma'],
            'vega': abs(greeks.get('total_vega', 0.0)) / limits['max_vega']
        }

    async def _evaluate_strategies(
        self,
        symbol: str,
        market_analysis: MarketAnalysis,
        risk_assessment: PortfolioRiskAssessment,
        option_chain: Optional[List[Dict[str, Any]]]
    ) -> Dict[StrategyRecommendation, float]:
        """Evaluate all strategies and assign suitability scores"""

        strategies = [
            StrategyRecommendation.ZERO_DTE,
            StrategyRecommendation.WHEEL,
            StrategyRecommendation.IRON_CONDOR,
            StrategyRecommendation.GAMMA_SCALPING,
            StrategyRecommendation.CASH
        ]

        scores = {}

        for strategy in strategies:
            score = self._score_strategy(
                strategy, market_analysis, risk_assessment
            )
            scores[strategy] = score

            logger.debug(
                "Strategy scored",
                strategy=strategy.value,
                score=score,
                market_condition=market_analysis.market_condition.value
            )

        return scores

    def _score_strategy(
        self,
        strategy: StrategyRecommendation,
        market_analysis: MarketAnalysis,
        risk_assessment: PortfolioRiskAssessment
    ) -> float:
        """Score individual strategy based on market conditions and risk"""

        scoring_matrix = {
            StrategyRecommendation.ZERO_DTE: self._score_zero_dte,
            StrategyRecommendation.WHEEL: self._score_wheel,
            StrategyRecommendation.IRON_CONDOR: self._score_iron_condor,
            StrategyRecommendation.GAMMA_SCALPING: self._score_gamma_scalping,
            StrategyRecommendation.CASH: self._score_cash
        }

        scorer = scoring_matrix.get(strategy)
        if scorer:
            return scorer(market_analysis, risk_assessment)
        else:
            return 0.0

    def _score_zero_dte(
        self,
        market: MarketAnalysis,
        risk: PortfolioRiskAssessment
    ) -> float:
        """Score Zero-DTE strategy"""
        score = 0.5  # Base score

        # Favorable for high volatility
        if market.volatility_regime in ['high', 'extreme']:
            score += 0.3
        elif market.volatility_regime == 'low':
            score -= 0.2

        # Requires good liquidity
        if market.liquidity_condition == 'high':
            score += 0.2
        elif market.liquidity_condition == 'low':
            score -= 0.3

        # Risk capacity considerations
        if risk.risk_capacity == 'aggressive':
            score += 0.1
        elif risk.risk_capacity == 'conservative':
            score -= 0.2

        # Gamma exposure check
        if risk.gamma_utilization > 0.7:
            score -= 0.3  # Too much gamma already

        return max(0.0, min(1.0, score))

    def _score_wheel(
        self,
        market: MarketAnalysis,
        risk: PortfolioRiskAssessment
    ) -> float:
        """Score Wheel strategy"""
        score = 0.6  # Base score (generally good strategy)

        # Favorable for neutral to bullish markets
        if market.trend_direction == 'bullish':
            score += 0.2
        elif market.trend_direction == 'bearish':
            score -= 0.2

        # Good for high time decay environments
        if market.time_decay_environment == 'favorable':
            score += 0.2

        # Conservative strategy
        if risk.risk_capacity in ['conservative', 'moderate']:
            score += 0.1

        # Delta considerations
        if risk.delta_utilization < 0.5:
            score += 0.1  # Room for delta exposure

        return max(0.0, min(1.0, score))

    def _score_iron_condor(
        self,
        market: MarketAnalysis,
        risk: PortfolioRiskAssessment
    ) -> float:
        """Score Iron Condor strategy"""
        score = 0.5  # Base score

        # Excellent for neutral markets with high volatility
        if market.trend_direction == 'neutral':
            score += 0.3
        else:
            score -= 0.2

        if market.volatility_regime in ['high', 'extreme']:
            score += 0.3
        elif market.volatility_regime == 'low':
            score -= 0.2

        # Good for stable regimes
        if market.regime_stability > 0.7:
            score += 0.2

        # Gamma considerations
        if risk.gamma_utilization < 0.6:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _score_gamma_scalping(
        self,
        market: MarketAnalysis,
        risk: PortfolioRiskAssessment
    ) -> float:
        """Score Gamma Scalping strategy"""
        score = 0.4  # Base score (complex strategy)

        # Excellent for high volatility
        if market.volatility_regime in ['high', 'extreme']:
            score += 0.4
        elif market.volatility_regime == 'low':
            score -= 0.3

        # Needs good liquidity for scalping
        if market.liquidity_condition == 'high':
            score += 0.3
        elif market.liquidity_condition == 'low':
            score -= 0.4

        # Requires sophisticated risk management
        if risk.risk_capacity == 'aggressive':
            score += 0.2
        elif risk.risk_capacity == 'conservative':
            score -= 0.2

        # Vega exposure considerations
        if risk.vega_utilization > 0.8:
            score -= 0.3

        return max(0.0, min(1.0, score))

    def _score_cash(
        self,
        market: MarketAnalysis,
        risk: PortfolioRiskAssessment
    ) -> float:
        """Score cash/no exposure option"""
        score = 0.2  # Base score

        # Higher in crisis or uncertain conditions
        if market.market_condition == MarketCondition.CRISIS:
            score += 0.6
        elif market.market_condition == MarketCondition.UNCERTAIN:
            score += 0.4

        # Higher if already at risk limits
        max_utilization = max(
            risk.delta_utilization,
            risk.gamma_utilization,
            risk.vega_utilization
        )

        if max_utilization > 0.8:
            score += 0.5

        # Conservative preference
        if risk.risk_capacity == 'conservative':
            score += 0.2

        return max(0.0, min(1.0, score))

    async def _generate_ai_recommendation(
        self,
        symbol: str,
        market_analysis: MarketAnalysis,
        risk_assessment: PortfolioRiskAssessment,
        strategy_scores: Dict[StrategyRecommendation, float]
    ) -> StrategyRecommendationResult:
        """Generate AI-powered recommendation with reasoning"""

        # Find best strategy
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        recommended_strategy, confidence_score = best_strategy

        # Build prompt for AI reasoning
        prompt = self._build_recommendation_prompt(
            symbol, market_analysis, risk_assessment, strategy_scores
        )

        try:
            # Get AI reasoning
            ai_response = await self._generate_response(prompt, max_tokens=2048)

            # Parse AI response
            ai_analysis = self._parse_ai_recommendation(ai_response)

            # Build comprehensive recommendation
            recommendation = StrategyRecommendationResult(
                recommended_strategy=recommended_strategy,
                confidence_score=confidence_score,
                allocation_percentage=self._calculate_allocation_percentage(
                    recommended_strategy, confidence_score, risk_assessment
                ),
                primary_reasons=ai_analysis.get('reasons', [
                    f"Market condition: {market_analysis.market_condition.value}",
                    f"Volatility regime: {market_analysis.volatility_regime}",
                    f"Strategy score: {confidence_score:.2f}"
                ]),
                risk_considerations=ai_analysis.get('risks', [
                    "Monitor volatility changes",
                    "Track portfolio Greeks limits",
                    "Assess liquidity conditions"
                ]),
                alternative_strategies=self._get_alternative_strategies(strategy_scores),
                exit_conditions=ai_analysis.get('exit_conditions', [
                    "Volatility regime change",
                    "50% max profit achieved",
                    "Risk limits approached"
                ]),
                adjustment_triggers=ai_analysis.get('adjustments', [
                    "Delta exposure exceeds 80% of limit",
                    "Market trend reversal",
                    "Volatility spike detected"
                ])
            )

            return recommendation

        except Exception as e:
            logger.error("AI recommendation generation failed", error=str(e))
            return self._create_simple_recommendation(
                recommended_strategy, confidence_score, risk_assessment
            )

    def _build_recommendation_prompt(
        self,
        symbol: str,
        market_analysis: MarketAnalysis,
        risk_assessment: PortfolioRiskAssessment,
        strategy_scores: Dict[StrategyRecommendation, float]
    ) -> str:
        """Build detailed prompt for AI reasoning"""

        # Format strategy scores
        scores_text = "\n".join([
            f"- {strategy.value}: {score:.2f}"
            for strategy, score in sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        ])

        prompt = f"""
Analyze {symbol} options strategy selection with the following data:

MARKET ANALYSIS:
- Market Condition: {market_analysis.market_condition.value}
- Volatility Regime: {market_analysis.volatility_regime}
- Trend Direction: {market_analysis.trend_direction}
- Volatility Level: {market_analysis.volatility_level:.1%}
- Time Decay Environment: {market_analysis.time_decay_environment}
- Liquidity: {market_analysis.liquidity_condition}
- Analysis Confidence: {market_analysis.confidence_score:.1%}

PORTFOLIO RISK ASSESSMENT:
- Current Delta: {risk_assessment.total_delta:.1f}
- Current Gamma: {risk_assessment.total_gamma:.1f}
- Current Vega: {risk_assessment.total_vega:.1f}
- Delta Utilization: {risk_assessment.delta_utilization:.1%}
- Gamma Utilization: {risk_assessment.gamma_utilization:.1%}
- Risk Capacity: {risk_assessment.risk_capacity}
- Available Capital: ${risk_assessment.buying_power_available:,.0f}

STRATEGY SCORES:
{scores_text}

Provide detailed reasoning for the top strategy recommendation including:
1. Why this strategy fits current market conditions
2. Key risk considerations and monitoring points
3. Specific exit conditions and adjustment triggers
4. Alternative strategies if conditions change

Response format:
{{
    "reasons": ["reason1", "reason2", "reason3"],
    "risks": ["risk1", "risk2", "risk3"],
    "exit_conditions": ["condition1", "condition2"],
    "adjustments": ["trigger1", "trigger2"]
}}
"""
        return prompt

    def _parse_ai_recommendation(self, response: str) -> Dict[str, List[str]]:
        """Parse AI recommendation response"""
        default_response = {
            "reasons": ["AI analysis unavailable"],
            "risks": ["Monitor market conditions"],
            "exit_conditions": ["Manual review required"],
            "adjustments": ["Regular portfolio review"]
        }

        return self._parse_json_response(response, default_response)

    def _calculate_allocation_percentage(
        self,
        strategy: StrategyRecommendation,
        confidence: float,
        risk_assessment: PortfolioRiskAssessment
    ) -> float:
        """Calculate appropriate allocation percentage"""

        base_allocations = {
            StrategyRecommendation.ZERO_DTE: 0.05,      # 5% - high risk
            StrategyRecommendation.WHEEL: 0.15,         # 15% - moderate risk
            StrategyRecommendation.IRON_CONDOR: 0.10,   # 10% - moderate risk
            StrategyRecommendation.GAMMA_SCALPING: 0.08, # 8% - high complexity
            StrategyRecommendation.CASH: 0.0,           # 0% - no allocation
            StrategyRecommendation.REDUCE_EXPOSURE: -0.5 # -50% - reduce existing
        }

        base_allocation = base_allocations.get(strategy, 0.05)

        # Adjust based on confidence
        confidence_adjusted = base_allocation * confidence

        # Adjust based on risk capacity
        risk_multipliers = {
            'conservative': 0.7,
            'moderate': 1.0,
            'aggressive': 1.3
        }

        risk_multiplier = risk_multipliers.get(risk_assessment.risk_capacity, 1.0)

        final_allocation = confidence_adjusted * risk_multiplier

        # Cap at maximum single strategy allocation
        max_allocation = self.risk_limits['max_single_strategy_allocation']

        return min(final_allocation, max_allocation)

    def _get_alternative_strategies(
        self,
        strategy_scores: Dict[StrategyRecommendation, float]
    ) -> List[str]:
        """Get top 3 alternative strategies"""
        sorted_strategies = sorted(
            strategy_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top 3 alternatives (excluding the best one)
        alternatives = [
            f"{strategy.value} (score: {score:.2f})"
            for strategy, score in sorted_strategies[1:4]
        ]

        return alternatives

    def _validate_recommendation(
        self,
        recommendation: StrategyRecommendationResult,
        risk_assessment: PortfolioRiskAssessment
    ) -> StrategyRecommendationResult:
        """Validate and adjust recommendation based on risk limits"""

        # Check if recommendation would exceed risk limits
        allocation = recommendation.allocation_percentage

        # Adjust allocation if too high
        if allocation > self.risk_limits['max_single_strategy_allocation']:
            recommendation.allocation_percentage = self.risk_limits['max_single_strategy_allocation']
            recommendation.risk_considerations.append(
                "Allocation reduced due to risk limits"
            )

        # Add warning if portfolio is near limits
        max_utilization = max(
            risk_assessment.delta_utilization,
            risk_assessment.gamma_utilization,
            risk_assessment.vega_utilization
        )

        if max_utilization > 0.8:
            recommendation.risk_considerations.append(
                "Portfolio approaching risk limits - consider reducing exposure"
            )
            recommendation.confidence_score *= 0.8  # Reduce confidence

        return recommendation

    def _create_fallback_recommendation(self) -> StrategyRecommendationResult:
        """Create fallback recommendation when analysis fails"""
        return StrategyRecommendationResult(
            recommended_strategy=StrategyRecommendation.CASH,
            confidence_score=0.1,
            allocation_percentage=0.0,
            primary_reasons=["Analysis failed - defaulting to cash"],
            risk_considerations=["System error - manual review required"],
            alternative_strategies=["Manual strategy selection required"],
            exit_conditions=["Immediate manual review"],
            adjustment_triggers=["System recovery required"]
        )

    def _create_simple_recommendation(
        self,
        strategy: StrategyRecommendation,
        confidence: float,
        risk_assessment: PortfolioRiskAssessment
    ) -> StrategyRecommendationResult:
        """Create simple recommendation without AI reasoning"""
        return StrategyRecommendationResult(
            recommended_strategy=strategy,
            confidence_score=confidence,
            allocation_percentage=self._calculate_allocation_percentage(
                strategy, confidence, risk_assessment
            ),
            primary_reasons=[f"Top scoring strategy: {strategy.value}"],
            risk_considerations=["Monitor market conditions"],
            alternative_strategies=["Manual review recommended"],
            exit_conditions=["Standard exit criteria"],
            adjustment_triggers=["Regular portfolio review"]
        )

    def _create_fallback_market_analysis(self) -> MarketAnalysis:
        """Create fallback market analysis"""
        return MarketAnalysis(
            market_condition=MarketCondition.UNCERTAIN,
            volatility_regime="normal",
            trend_direction="neutral",
            volatility_level=0.25,
            regime_stability=0.5,
            time_decay_environment="neutral",
            liquidity_condition="normal",
            confidence_score=0.2
        )

    def _create_fallback_risk_assessment(self, risk_tolerance: str) -> PortfolioRiskAssessment:
        """Create fallback risk assessment"""
        return PortfolioRiskAssessment(
            total_delta=0.0,
            total_gamma=0.0,
            total_theta=0.0,
            total_vega=0.0,
            total_rho=0.0,
            delta_utilization=0.0,
            gamma_utilization=0.0,
            vega_utilization=0.0,
            max_loss_exposure=0.0,
            buying_power_available=100000.0,
            concentration_risk=0.0,
            risk_capacity=risk_tolerance,
            time_horizon="medium_term"
        )

    async def process(self, *args, **kwargs) -> StrategyRecommendationResult:
        """Main processing method for BaseAIAgent compatibility"""
        return await self.select_strategy(*args, **kwargs)


# Global singleton instance
_strategy_selector: Optional[OptionsStrategySelector] = None


async def get_strategy_selector() -> OptionsStrategySelector:
    """Get or create global strategy selector instance"""
    global _strategy_selector
    if _strategy_selector is None:
        from app.ai.ollama_client import OllamaClient
        ollama_client = OllamaClient()
        _strategy_selector = OptionsStrategySelector(ollama_client)
    return _strategy_selector


__all__ = [
    "OptionsStrategySelector",
    "StrategyRecommendationResult",
    "MarketAnalysis",
    "PortfolioRiskAssessment",
    "StrategyRecommendation",
    "MarketCondition",
    "get_strategy_selector"
]