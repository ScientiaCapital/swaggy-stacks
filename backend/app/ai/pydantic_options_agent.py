"""
PydanticAI Options Trading Agent - Type-safe options analysis and strategy recommendation
using the enhanced options trading system with advanced volatility modeling and Greeks monitoring.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator
import structlog

from app.ai.pydantic_base_agent import PydanticBaseAgent, AgentContext, AgentResponse
from app.trading.options_trading import (
    OptionsTrader, OptionStrategy, OptionType, OptionContract,
    get_options_trader, get_enhanced_calculator
)
from app.trading.greeks_monitor import GreeksMonitor, GreeksLimits, get_greeks_monitor
from app.ai.options_strategy_selector import OptionsStrategySelector, StrategyRecommendation

logger = structlog.get_logger(__name__)


class OptionsAction(Enum):
    BUY_CALL = "buy_call"
    SELL_CALL = "sell_call"
    BUY_PUT = "buy_put"
    SELL_PUT = "sell_put"
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    IRON_CONDOR = "iron_condor"
    CLOSE_POSITION = "close_position"
    HOLD = "hold"


class VolatilityEnvironment(Enum):
    LOW_VOL = "low_volatility"
    NORMAL_VOL = "normal_volatility"
    HIGH_VOL = "high_volatility"
    EXTREME_VOL = "extreme_volatility"


class MarketRegime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"
    TRENDING = "trending"


class OptionsAnalysisRequest(BaseModel):
    """Request for options analysis"""
    symbol: str = Field(..., description="Stock symbol to analyze")
    analysis_type: str = Field("comprehensive", description="Type of analysis: strategy, greeks, risk, comprehensive")
    strategy_focus: Optional[str] = Field(None, description="Specific strategy to focus on")
    risk_tolerance: str = Field("moderate", description="Risk tolerance: conservative, moderate, aggressive")
    investment_horizon: str = Field("medium_term", description="Time horizon: short_term, medium_term, long_term")
    portfolio_context: Optional[Dict[str, Any]] = Field(None, description="Current portfolio context")
    market_view: Optional[str] = Field(None, description="Market outlook: bullish, bearish, neutral")

    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v) > 10:
            raise ValueError("Symbol must be provided and under 10 characters")
        return v.upper()


class OptionsRecommendation(BaseModel):
    """Options strategy recommendation"""
    recommended_action: OptionsAction
    strategy_name: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    target_strikes: List[float] = Field(default_factory=list)
    target_expiration: Optional[str] = None
    position_size: float = Field(..., gt=0)
    max_risk: float
    max_reward: Optional[float] = None
    breakeven_points: List[float] = Field(default_factory=list)
    probability_of_profit: float = Field(..., ge=0.0, le=1.0)

    # Greeks exposure
    delta_exposure: float = 0.0
    gamma_exposure: float = 0.0
    theta_exposure: float = 0.0
    vega_exposure: float = 0.0

    # Risk metrics
    max_loss_1pct_move: float = 0.0
    max_loss_5pct_move: float = 0.0
    portfolio_heat: float = 0.0  # Percentage of portfolio at risk


class VolatilityAnalysis(BaseModel):
    """Volatility environment analysis"""
    current_iv: float = Field(..., description="Current implied volatility")
    historical_iv: float = Field(..., description="Historical implied volatility")
    iv_percentile: float = Field(..., ge=0.0, le=100.0, description="IV percentile rank")
    volatility_environment: VolatilityEnvironment
    skew_analysis: Dict[str, float] = Field(default_factory=dict)
    term_structure: Dict[str, float] = Field(default_factory=dict)
    vol_forecast: float = Field(..., description="Predicted volatility")
    vol_regime: str = Field(..., description="Volatility regime classification")


class GreeksAnalysis(BaseModel):
    """Portfolio Greeks analysis"""
    current_portfolio_delta: float = 0.0
    current_portfolio_gamma: float = 0.0
    current_portfolio_theta: float = 0.0
    current_portfolio_vega: float = 0.0
    current_portfolio_rho: float = 0.0

    # Risk utilization
    delta_utilization_pct: float = Field(..., ge=0.0, le=200.0)
    gamma_utilization_pct: float = Field(..., ge=0.0, le=200.0)
    vega_utilization_pct: float = Field(..., ge=0.0, le=200.0)

    # Projected Greeks after recommendation
    projected_portfolio_delta: float = 0.0
    projected_portfolio_gamma: float = 0.0
    projected_portfolio_theta: float = 0.0
    projected_portfolio_vega: float = 0.0

    risk_warnings: List[str] = Field(default_factory=list)
    position_limits_exceeded: bool = False


class MarketAnalysisResult(BaseModel):
    """Market analysis for options context"""
    market_regime: MarketRegime
    trend_direction: str = Field(..., description="Current trend direction")
    volatility_analysis: VolatilityAnalysis
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    key_support_levels: List[float] = Field(default_factory=list)
    key_resistance_levels: List[float] = Field(default_factory=list)
    expected_move_1week: float = 0.0
    expected_move_1month: float = 0.0
    earnings_announcement: Optional[str] = None
    liquidity_assessment: str = "normal"  # low, normal, high


class OptionsAnalysisResult(BaseModel):
    """Complete options analysis result"""
    symbol: str
    analysis_timestamp: datetime
    current_price: float

    # Core analysis
    market_analysis: MarketAnalysisResult
    greeks_analysis: GreeksAnalysis
    primary_recommendation: OptionsRecommendation
    alternative_recommendations: List[OptionsRecommendation] = Field(default_factory=list)

    # Risk assessment
    overall_risk_level: str = Field(..., description="low, moderate, high, extreme")
    portfolio_impact: str = Field(..., description="Impact on overall portfolio")
    risk_warnings: List[str] = Field(default_factory=list)

    # Strategy insights
    strategy_rationale: List[str] = Field(default_factory=list)
    key_considerations: List[str] = Field(default_factory=list)
    exit_conditions: List[str] = Field(default_factory=list)
    monitoring_points: List[str] = Field(default_factory=list)

    # Performance tracking
    analysis_confidence: float = Field(..., ge=0.0, le=1.0)
    model_accuracy_note: str = ""


class PydanticOptionsAgent(PydanticBaseAgent[OptionsAnalysisResult]):
    """
    PydanticAI Options Trading Agent

    Provides comprehensive options analysis with:
    - Enhanced volatility modeling
    - Real-time Greeks monitoring
    - Advanced strategy selection
    - Risk-aware position sizing
    - Market regime awareness
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022"):
        super().__init__(model_name=model_name)

        # Initialize components
        self.options_trader = get_options_trader()
        self.enhanced_calculator = get_enhanced_calculator()
        self.greeks_monitor = get_greeks_monitor()
        self.strategy_selector = None  # Will initialize with Ollama when available

        # Agent configuration
        self.agent_type = "options_analyst"
        self.risk_limits = GreeksLimits()

        logger.info("PydanticOptionsAgent initialized with enhanced components")

    async def analyze_options(
        self,
        request: OptionsAnalysisRequest,
        context: AgentContext,
        correlation_id: Optional[str] = None
    ) -> AgentResponse[OptionsAnalysisResult]:
        """
        Main options analysis method
        """
        start_time = datetime.now()

        try:
            logger.info(
                "Starting options analysis",
                symbol=request.symbol,
                analysis_type=request.analysis_type,
                agent_id=context.agent_id,
                correlation_id=correlation_id
            )

            # Get current market data
            current_price = await self._get_current_price(request.symbol)
            if not current_price:
                raise ValueError(f"Could not get current price for {request.symbol}")

            # Perform comprehensive analysis
            market_analysis = await self._analyze_market(request, current_price)
            greeks_analysis = await self._analyze_current_greeks(request)

            # Get strategy recommendations
            recommendations = await self._get_strategy_recommendations(
                request, market_analysis, greeks_analysis, current_price
            )

            if not recommendations:
                raise ValueError("No viable strategy recommendations found")

            primary_recommendation = recommendations[0]
            alternative_recommendations = recommendations[1:3]  # Top 3 alternatives

            # Assess overall risk
            risk_assessment = self._assess_overall_risk(
                primary_recommendation, greeks_analysis, market_analysis
            )

            # Generate strategic insights
            insights = await self._generate_strategy_insights(
                request, market_analysis, primary_recommendation
            )

            # Build final result
            result = OptionsAnalysisResult(
                symbol=request.symbol,
                analysis_timestamp=datetime.now(),
                current_price=current_price,
                market_analysis=market_analysis,
                greeks_analysis=greeks_analysis,
                primary_recommendation=primary_recommendation,
                alternative_recommendations=alternative_recommendations,
                overall_risk_level=risk_assessment["risk_level"],
                portfolio_impact=risk_assessment["portfolio_impact"],
                risk_warnings=risk_assessment["warnings"],
                strategy_rationale=insights["rationale"],
                key_considerations=insights["considerations"],
                exit_conditions=insights["exit_conditions"],
                monitoring_points=insights["monitoring_points"],
                analysis_confidence=self._calculate_confidence(market_analysis, greeks_analysis),
                model_accuracy_note="Analysis based on enhanced volatility models and real-time Greeks monitoring"
            )

            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Update agent statistics
            self._update_execution_stats(execution_time, True, result.analysis_confidence)

            logger.info(
                "Options analysis completed",
                symbol=request.symbol,
                recommended_action=primary_recommendation.recommended_action.value,
                confidence=result.analysis_confidence,
                execution_time_ms=execution_time,
                correlation_id=correlation_id
            )

            return AgentResponse[OptionsAnalysisResult](
                data=result,
                confidence=result.analysis_confidence,
                execution_time_ms=execution_time,
                agent_id=context.agent_id,
                correlation_id=correlation_id or f"options_{int(start_time.timestamp())}"
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_execution_stats(execution_time, False, 0.0)

            logger.error(
                "Options analysis failed",
                symbol=request.symbol,
                error=str(e),
                execution_time_ms=execution_time,
                correlation_id=correlation_id
            )
            raise

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current stock price"""
        try:
            price_data = await self.options_trader.alpaca_client.get_latest_price(symbol)
            return price_data.get("price") if price_data else None
        except Exception as e:
            logger.error("Failed to get current price", symbol=symbol, error=str(e))
            return None

    async def _analyze_market(
        self,
        request: OptionsAnalysisRequest,
        current_price: float
    ) -> MarketAnalysisResult:
        """Analyze market conditions for options context"""
        try:
            # Get recent price data for analysis
            price_data = await self.options_trader.alpaca_client.get_historical_data(
                request.symbol, days=30
            )

            if not price_data or len(price_data) < 10:
                # Fallback to basic analysis
                return self._create_fallback_market_analysis(current_price)

            # Get volatility analysis using enhanced calculator
            vol_metrics = await self.enhanced_calculator.volatility_predictor.predict_volatility(
                symbol=request.symbol,
                price_data=price_data
            )

            # Determine market regime
            market_regime = self._classify_market_regime(price_data, vol_metrics)

            # Calculate expected moves
            iv = vol_metrics.implied_vol or vol_metrics.garch_predicted_vol
            expected_move_1week = current_price * iv * (7/365)**0.5
            expected_move_1month = current_price * iv * (30/365)**0.5

            # Build volatility analysis
            volatility_analysis = VolatilityAnalysis(
                current_iv=iv,
                historical_iv=vol_metrics.historical_vol,
                iv_percentile=50.0,  # Would need historical IV data
                volatility_environment=self._classify_vol_environment(vol_metrics.vol_regime.value),
                vol_forecast=vol_metrics.garch_predicted_vol,
                vol_regime=vol_metrics.vol_regime.value
            )

            # Support/resistance levels (simplified)
            support_resistance = self._calculate_support_resistance(price_data, current_price)

            return MarketAnalysisResult(
                market_regime=market_regime,
                trend_direction=self._analyze_trend(price_data),
                volatility_analysis=volatility_analysis,
                sentiment_score=0.0,  # Would integrate with sentiment analysis
                key_support_levels=support_resistance["support"],
                key_resistance_levels=support_resistance["resistance"],
                expected_move_1week=expected_move_1week,
                expected_move_1month=expected_move_1month,
                liquidity_assessment="normal"  # Would assess from volume data
            )

        except Exception as e:
            logger.error("Market analysis failed", error=str(e))
            return self._create_fallback_market_analysis(current_price)

    async def _analyze_current_greeks(self, request: OptionsAnalysisRequest) -> GreeksAnalysis:
        """Analyze current portfolio Greeks and capacity"""
        try:
            # Get current portfolio Greeks from monitor
            await self.greeks_monitor.update_portfolio_greeks()
            current_greeks = self.greeks_monitor.current_greeks

            if not current_greeks:
                return GreeksAnalysis(
                    delta_utilization_pct=0.0,
                    gamma_utilization_pct=0.0,
                    vega_utilization_pct=0.0
                )

            # Calculate utilization percentages
            delta_util = (abs(current_greeks.total_delta) / self.risk_limits.max_delta) * 100
            gamma_util = (abs(current_greeks.total_gamma) / self.risk_limits.max_gamma) * 100
            vega_util = (abs(current_greeks.total_vega) / self.risk_limits.max_vega) * 100

            # Check for warnings
            warnings = []
            limits_exceeded = False

            if delta_util > 90:
                warnings.append("Delta utilization above 90% - limited directional capacity")
                limits_exceeded = True
            if gamma_util > 90:
                warnings.append("Gamma utilization above 90% - limited volatility capacity")
                limits_exceeded = True
            if vega_util > 90:
                warnings.append("Vega utilization above 90% - limited volatility exposure capacity")
                limits_exceeded = True

            return GreeksAnalysis(
                current_portfolio_delta=current_greeks.total_delta,
                current_portfolio_gamma=current_greeks.total_gamma,
                current_portfolio_theta=current_greeks.total_theta,
                current_portfolio_vega=current_greeks.total_vega,
                current_portfolio_rho=current_greeks.total_rho,
                delta_utilization_pct=delta_util,
                gamma_utilization_pct=gamma_util,
                vega_utilization_pct=vega_util,
                risk_warnings=warnings,
                position_limits_exceeded=limits_exceeded
            )

        except Exception as e:
            logger.error("Greeks analysis failed", error=str(e))
            return GreeksAnalysis(
                delta_utilization_pct=0.0,
                gamma_utilization_pct=0.0,
                vega_utilization_pct=0.0
            )

    async def _get_strategy_recommendations(
        self,
        request: OptionsAnalysisRequest,
        market_analysis: MarketAnalysisResult,
        greeks_analysis: GreeksAnalysis,
        current_price: float
    ) -> List[OptionsRecommendation]:
        """Get strategy recommendations based on analysis"""
        try:
            recommendations = []

            # Get option chain for analysis
            option_chain = await self.options_trader.get_option_chain(request.symbol)

            if not option_chain:
                logger.warning("No option chain available", symbol=request.symbol)
                return []

            # Analyze different strategies based on market conditions
            strategies_to_analyze = self._select_strategies_to_analyze(
                market_analysis, greeks_analysis, request
            )

            for strategy in strategies_to_analyze:
                try:
                    recommendation = await self._analyze_strategy(
                        strategy, request.symbol, option_chain, current_price,
                        market_analysis, greeks_analysis
                    )
                    if recommendation:
                        recommendations.append(recommendation)
                except Exception as e:
                    logger.warning(f"Strategy analysis failed for {strategy}", error=str(e))
                    continue

            # Sort by confidence score
            recommendations.sort(key=lambda x: x.confidence_score, reverse=True)

            return recommendations[:5]  # Return top 5 recommendations

        except Exception as e:
            logger.error("Strategy recommendations failed", error=str(e))
            return []

    def _select_strategies_to_analyze(
        self,
        market_analysis: MarketAnalysisResult,
        greeks_analysis: GreeksAnalysis,
        request: OptionsAnalysisRequest
    ) -> List[OptionStrategy]:
        """Select which strategies to analyze based on conditions"""

        vol_env = market_analysis.volatility_analysis.volatility_environment
        market_regime = market_analysis.market_regime

        strategies = []

        # High volatility strategies
        if vol_env in [VolatilityEnvironment.HIGH_VOL, VolatilityEnvironment.EXTREME_VOL]:
            strategies.extend([
                OptionStrategy.LONG_STRADDLE,
                OptionStrategy.PROTECTIVE_PUT
            ])

        # Low/normal volatility strategies
        if vol_env in [VolatilityEnvironment.LOW_VOL, VolatilityEnvironment.NORMAL_VOL]:
            strategies.extend([
                OptionStrategy.COVERED_CALL,
                OptionStrategy.LONG_CALL if market_regime == MarketRegime.BULLISH else OptionStrategy.LONG_PUT
            ])

        # Add basic strategies
        strategies.extend([OptionStrategy.LONG_CALL, OptionStrategy.LONG_PUT])

        return list(set(strategies))  # Remove duplicates

    async def _analyze_strategy(
        self,
        strategy: OptionStrategy,
        symbol: str,
        option_chain: List[OptionContract],
        current_price: float,
        market_analysis: MarketAnalysisResult,
        greeks_analysis: GreeksAnalysis
    ) -> Optional[OptionsRecommendation]:
        """Analyze a specific strategy and create recommendation"""
        try:
            # Use the OptionsTrader to analyze the strategy
            analysis_result = await self.options_trader.analyze_option_strategy(
                strategy, symbol, {"strike_price": current_price * 1.05}
            )

            if "error" in analysis_result:
                return None

            # Extract key metrics
            metrics = analysis_result.get("metrics", {})
            risk_analysis = analysis_result.get("risk_analysis", {})

            # Map strategy to action
            action_mapping = {
                OptionStrategy.LONG_CALL: OptionsAction.BUY_CALL,
                OptionStrategy.COVERED_CALL: OptionsAction.COVERED_CALL,
                OptionStrategy.PROTECTIVE_PUT: OptionsAction.PROTECTIVE_PUT,
                OptionStrategy.STRADDLE: OptionsAction.LONG_STRADDLE,
            }

            action = action_mapping.get(strategy, OptionsAction.HOLD)

            # Calculate confidence based on market alignment
            confidence = self._calculate_strategy_confidence(
                strategy, market_analysis, greeks_analysis
            )

            # Determine position size based on risk management
            position_size = self._calculate_position_size(
                metrics.get("cost", 0), greeks_analysis
            )

            return OptionsRecommendation(
                recommended_action=action,
                strategy_name=strategy.value,
                confidence_score=confidence,
                target_strikes=[analysis_result.get("selected_option", {}).get("strike", current_price)],
                position_size=position_size,
                max_risk=metrics.get("max_loss", 0),
                max_reward=metrics.get("max_profit") if isinstance(metrics.get("max_profit"), (int, float)) else None,
                breakeven_points=[metrics.get("breakeven", current_price)],
                probability_of_profit=metrics.get("probability_of_profit", 50) / 100,
                delta_exposure=risk_analysis.get("delta_exposure", 0),
                gamma_exposure=0.0,  # Would extract from Greeks if available
                theta_exposure=risk_analysis.get("theta_decay", 0),
                vega_exposure=risk_analysis.get("vega_risk", 0),
                portfolio_heat=min(metrics.get("cost", 0) / 10000, 0.1)  # Assume $10k portfolio
            )

        except Exception as e:
            logger.error(f"Strategy analysis failed for {strategy.value}", error=str(e))
            return None

    def _calculate_strategy_confidence(
        self,
        strategy: OptionStrategy,
        market_analysis: MarketAnalysisResult,
        greeks_analysis: GreeksAnalysis
    ) -> float:
        """Calculate confidence score for a strategy"""

        base_confidence = 0.5
        vol_env = market_analysis.volatility_analysis.volatility_environment
        market_regime = market_analysis.market_regime

        # Volatility alignment
        if strategy == OptionStrategy.LONG_STRADDLE:
            if vol_env in [VolatilityEnvironment.HIGH_VOL, VolatilityEnvironment.EXTREME_VOL]:
                base_confidence += 0.3
            else:
                base_confidence -= 0.2

        elif strategy == OptionStrategy.COVERED_CALL:
            if vol_env == VolatilityEnvironment.HIGH_VOL:
                base_confidence += 0.2
            if market_regime in [MarketRegime.NEUTRAL, MarketRegime.BULLISH]:
                base_confidence += 0.2

        elif strategy == OptionStrategy.LONG_CALL:
            if market_regime == MarketRegime.BULLISH:
                base_confidence += 0.3
            elif market_regime == MarketRegime.BEARISH:
                base_confidence -= 0.3

        # Greeks capacity check
        if greeks_analysis.position_limits_exceeded:
            base_confidence -= 0.2

        return max(0.0, min(1.0, base_confidence))

    def _calculate_position_size(self, strategy_cost: float, greeks_analysis: GreeksAnalysis) -> float:
        """Calculate appropriate position size"""
        # Simple position sizing - 2% of assumed portfolio
        portfolio_value = 100000  # Assume $100k portfolio
        max_risk_per_trade = portfolio_value * 0.02

        if strategy_cost > 0:
            position_size = min(1.0, max_risk_per_trade / strategy_cost)
        else:
            position_size = 1.0

        # Adjust for Greeks utilization
        max_utilization = max(
            greeks_analysis.delta_utilization_pct,
            greeks_analysis.gamma_utilization_pct,
            greeks_analysis.vega_utilization_pct
        )

        if max_utilization > 80:
            position_size *= 0.5  # Reduce size if near limits

        return max(0.1, position_size)

    # Additional helper methods...
    def _classify_market_regime(self, price_data: List[Dict], vol_metrics) -> MarketRegime:
        """Classify current market regime"""
        if not price_data or len(price_data) < 10:
            return MarketRegime.NEUTRAL

        # Simple trend analysis
        recent_prices = [float(candle["close"]) for candle in price_data[-10:]]
        older_prices = [float(candle["close"]) for candle in price_data[-20:-10]]

        recent_avg = sum(recent_prices) / len(recent_prices)
        older_avg = sum(older_prices) / len(older_prices)

        change_pct = (recent_avg - older_avg) / older_avg

        if change_pct > 0.05:
            return MarketRegime.BULLISH
        elif change_pct < -0.05:
            return MarketRegime.BEARISH
        else:
            return MarketRegime.NEUTRAL

    def _classify_vol_environment(self, vol_regime: str) -> VolatilityEnvironment:
        """Classify volatility environment"""
        regime_mapping = {
            "low": VolatilityEnvironment.LOW_VOL,
            "normal": VolatilityEnvironment.NORMAL_VOL,
            "high": VolatilityEnvironment.HIGH_VOL,
            "extreme": VolatilityEnvironment.EXTREME_VOL
        }
        return regime_mapping.get(vol_regime, VolatilityEnvironment.NORMAL_VOL)

    def _analyze_trend(self, price_data: List[Dict]) -> str:
        """Analyze price trend direction"""
        if len(price_data) < 5:
            return "neutral"

        prices = [float(candle["close"]) for candle in price_data[-5:]]
        if prices[-1] > prices[0] * 1.02:
            return "uptrend"
        elif prices[-1] < prices[0] * 0.98:
            return "downtrend"
        else:
            return "sideways"

    def _calculate_support_resistance(self, price_data: List[Dict], current_price: float) -> Dict[str, List[float]]:
        """Calculate basic support and resistance levels"""
        if len(price_data) < 20:
            return {"support": [current_price * 0.95], "resistance": [current_price * 1.05]}

        highs = [float(candle["high"]) for candle in price_data[-20:]]
        lows = [float(candle["low"]) for candle in price_data[-20:]]

        resistance = [max(highs)]
        support = [min(lows)]

        return {"support": support, "resistance": resistance}

    def _assess_overall_risk(self, recommendation: OptionsRecommendation, greeks_analysis: GreeksAnalysis, market_analysis: MarketAnalysisResult) -> Dict[str, Any]:
        """Assess overall risk level"""
        risk_factors = []
        risk_score = 0

        # Strategy risk
        if recommendation.recommended_action in [OptionsAction.LONG_STRADDLE, OptionsAction.SHORT_STRADDLE]:
            risk_score += 2
            risk_factors.append("High volatility sensitivity")

        # Greeks capacity
        if greeks_analysis.position_limits_exceeded:
            risk_score += 2
            risk_factors.append("Portfolio risk limits near capacity")

        # Market volatility
        if market_analysis.volatility_analysis.volatility_environment == VolatilityEnvironment.EXTREME_VOL:
            risk_score += 1
            risk_factors.append("Extreme volatility environment")

        # Determine risk level
        if risk_score >= 4:
            risk_level = "extreme"
        elif risk_score >= 3:
            risk_level = "high"
        elif risk_score >= 2:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return {
            "risk_level": risk_level,
            "portfolio_impact": "Limited to position size" if recommendation.portfolio_heat < 0.05 else "Moderate portfolio impact",
            "warnings": risk_factors
        }

    async def _generate_strategy_insights(self, request: OptionsAnalysisRequest, market_analysis: MarketAnalysisResult, recommendation: OptionsRecommendation) -> Dict[str, List[str]]:
        """Generate strategic insights for the recommendation"""
        return {
            "rationale": [
                f"Strategy aligns with {market_analysis.market_regime.value} market regime",
                f"Volatility environment ({market_analysis.volatility_analysis.volatility_environment.value}) supports strategy",
                f"Risk-reward profile matches {request.risk_tolerance} risk tolerance"
            ],
            "considerations": [
                "Monitor implied volatility changes",
                "Watch for trend reversals",
                "Manage position size according to portfolio heat"
            ],
            "exit_conditions": [
                "50% profit target achieved",
                "Market regime change detected",
                "Risk limits approached"
            ],
            "monitoring_points": [
                "Daily Greeks exposure check",
                "Weekly volatility reassessment",
                "Position P&L tracking"
            ]
        }

    def _calculate_confidence(self, market_analysis: MarketAnalysisResult, greeks_analysis: GreeksAnalysis) -> float:
        """Calculate overall analysis confidence"""
        base_confidence = 0.7

        # Reduce confidence if near risk limits
        if greeks_analysis.position_limits_exceeded:
            base_confidence -= 0.2

        # Reduce confidence in extreme volatility
        if market_analysis.volatility_analysis.volatility_environment == VolatilityEnvironment.EXTREME_VOL:
            base_confidence -= 0.1

        return max(0.3, min(1.0, base_confidence))

    def _create_fallback_market_analysis(self, current_price: float) -> MarketAnalysisResult:
        """Create fallback market analysis when data is unavailable"""
        return MarketAnalysisResult(
            market_regime=MarketRegime.NEUTRAL,
            trend_direction="neutral",
            volatility_analysis=VolatilityAnalysis(
                current_iv=0.25,
                historical_iv=0.25,
                iv_percentile=50.0,
                volatility_environment=VolatilityEnvironment.NORMAL_VOL,
                vol_forecast=0.25,
                vol_regime="normal"
            ),
            sentiment_score=0.0,
            key_support_levels=[current_price * 0.95],
            key_resistance_levels=[current_price * 1.05],
            expected_move_1week=current_price * 0.02,
            expected_move_1month=current_price * 0.05
        )


# Global instance for singleton pattern
_pydantic_options_agent: Optional[PydanticOptionsAgent] = None


def get_pydantic_options_agent() -> PydanticOptionsAgent:
    """Get global PydanticOptionsAgent instance"""
    global _pydantic_options_agent
    if _pydantic_options_agent is None:
        _pydantic_options_agent = PydanticOptionsAgent()
    return _pydantic_options_agent


# Export key classes
__all__ = [
    "PydanticOptionsAgent",
    "OptionsAnalysisRequest",
    "OptionsAnalysisResult",
    "OptionsRecommendation",
    "VolatilityAnalysis",
    "GreeksAnalysis",
    "MarketAnalysisResult",
    "get_pydantic_options_agent"
]