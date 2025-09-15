"""
Unit tests for OptionsStrategySelector

Tests AI-powered options strategy selection with market condition analysis,
portfolio risk assessment, and comprehensive decision logic validation.

Test Coverage:
- Market condition analysis and classification
- Portfolio risk assessment and utilization
- Strategy scoring and recommendation logic
- AI reasoning generation and parsing
- Risk limit validation and enforcement
- Alternative strategy suggestions
- Error handling and fallback mechanisms
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timezone
import json
import numpy as np

from app.ai.options_strategy_selector import (
    OptionsStrategySelector,
    MarketCondition,
    StrategyRecommendation,
    MarketAnalysis,
    PortfolioRiskAssessment,
    StrategyRecommendationResult
)
from app.ai.ollama_client import OllamaClient


class TestOptionsStrategySelector:
    """Test suite for OptionsStrategySelector AI decision making"""

    @pytest.fixture
    def mock_ollama_client(self):
        """Mock OllamaClient for testing"""
        client = AsyncMock(spec=OllamaClient)
        client.generate_response = AsyncMock()
        return client

    @pytest.fixture
    def strategy_selector(self, mock_ollama_client):
        """OptionsStrategySelector instance with mocked dependencies"""
        return OptionsStrategySelector(ollama_client=mock_ollama_client)

    @pytest.fixture
    def sample_price_data(self):
        """Sample price data for market analysis"""
        base_price = 100.0
        dates = [datetime.now(timezone.utc) for _ in range(60)]

        # Generate realistic price movement with trend
        price_data = []
        for i, date in enumerate(dates):
            # Simulate upward trend with volatility
            trend_factor = 1 + (i * 0.001)  # Slight upward trend
            noise = np.random.normal(0, 0.02)  # 2% daily volatility
            price = base_price * trend_factor * (1 + noise)

            price_data.append({
                'timestamp': date,
                'open': price * 0.999,
                'high': price * 1.015,
                'low': price * 0.985,
                'close': price,
                'volume': np.random.randint(1000000, 5000000)
            })

        return price_data

    @pytest.fixture
    def sample_option_chain(self):
        """Sample option chain data"""
        current_price = 100.0
        strikes = np.arange(90, 111, 2.5)

        option_chain = []
        for strike in strikes:
            moneyness = strike / current_price

            # Simulate realistic implied volatility
            iv = 0.25 + 0.1 * (moneyness - 1)**2  # Volatility smile

            option_chain.append({
                'strike_price': float(strike),
                'expiry_date': datetime.now(timezone.utc),
                'option_type': 'call' if strike >= current_price else 'put',
                'implied_volatility': iv,
                'bid': max(0.05, iv * 10),
                'ask': iv * 12,
                'volume': np.random.randint(0, 1000),
                'open_interest': np.random.randint(100, 5000)
            })

        return option_chain

    @pytest.fixture
    def sample_portfolio_greeks(self):
        """Sample portfolio Greeks data"""
        return {
            'total_delta': 25.5,
            'total_gamma': 15.2,
            'total_theta': -8.7,
            'total_vega': 120.3,
            'total_rho': 5.1,
            'max_loss': 5000.0
        }

    @pytest.fixture
    def sample_market_analysis(self):
        """Sample market analysis result"""
        return MarketAnalysis(
            market_condition=MarketCondition.BULLISH_LOW_VOL,
            volatility_regime="normal",
            trend_direction="bullish",
            volatility_level=0.22,
            regime_stability=0.75,
            time_decay_environment="neutral",
            liquidity_condition="normal",
            confidence_score=0.8
        )

    @pytest.fixture
    def sample_risk_assessment(self):
        """Sample portfolio risk assessment"""
        return PortfolioRiskAssessment(
            total_delta=25.5,
            total_gamma=15.2,
            total_theta=-8.7,
            total_vega=120.3,
            total_rho=5.1,
            delta_utilization=0.35,
            gamma_utilization=0.25,
            vega_utilization=0.45,
            max_loss_exposure=5000.0,
            buying_power_available=100000.0,
            concentration_risk=0.30,
            risk_capacity="moderate",
            time_horizon="medium_term"
        )

    @pytest.mark.asyncio
    async def test_select_strategy_full_workflow(
        self, strategy_selector, sample_price_data, sample_option_chain, sample_portfolio_greeks
    ):
        """Test complete strategy selection workflow"""
        # Mock AI response
        ai_response = json.dumps({
            "reasons": [
                "Bullish market trend supports wheel strategy",
                "Normal volatility regime favorable for theta collection",
                "Portfolio has capacity for additional delta exposure"
            ],
            "risks": [
                "Monitor for volatility spike",
                "Watch delta concentration risk",
                "Assess liquidity during execution"
            ],
            "exit_conditions": [
                "Volatility regime change to high",
                "50% max profit achieved"
            ],
            "adjustments": [
                "Reduce position size if delta utilization exceeds 70%",
                "Roll positions if approaching expiration"
            ]
        })

        strategy_selector.ollama_client.generate_response.return_value = ai_response

        # Mock volatility analysis
        with patch.object(strategy_selector, '_get_volatility_analysis') as mock_vol:
            mock_vol.return_value = {
                'current_volatility': 0.22,
                'volatility_regime': 'normal',
                'confidence': 0.8,
                'spike_probability': 0.2,
                'mean_reversion_factor': 0.1
            }

            result = await strategy_selector.select_strategy(
                symbol="AAPL",
                price_data=sample_price_data,
                option_chain=sample_option_chain,
                portfolio_greeks=sample_portfolio_greeks,
                risk_tolerance="moderate"
            )

        # Verify recommendation structure
        assert isinstance(result, StrategyRecommendationResult)
        assert result.recommended_strategy in StrategyRecommendation
        assert 0.0 <= result.confidence_score <= 1.0
        assert 0.0 <= result.allocation_percentage <= 1.0
        assert len(result.primary_reasons) > 0
        assert len(result.risk_considerations) > 0

    @pytest.mark.asyncio
    async def test_analyze_market_conditions_bullish_low_vol(
        self, strategy_selector, sample_price_data
    ):
        """Test market condition analysis for bullish low volatility"""
        with patch.object(strategy_selector, '_get_volatility_analysis') as mock_vol:
            mock_vol.return_value = {
                'current_volatility': 0.18,  # Low volatility
                'volatility_regime': 'low',
                'confidence': 0.85,
                'spike_probability': 0.1,
                'mean_reversion_factor': 0.15
            }

            # Modify price data for clear bullish trend
            bullish_data = sample_price_data.copy()
            for i, candle in enumerate(bullish_data):
                candle['close'] = 100 + (i * 0.5)  # Clear upward trend

            analysis = await strategy_selector._analyze_market_conditions("AAPL", bullish_data)

        assert analysis.market_condition == MarketCondition.BULLISH_LOW_VOL
        assert analysis.trend_direction == "bullish"
        assert analysis.volatility_regime == "low"
        assert analysis.volatility_level == 0.18
        assert analysis.confidence_score > 0.5

    @pytest.mark.asyncio
    async def test_analyze_market_conditions_neutral_high_vol(
        self, strategy_selector, sample_price_data
    ):
        """Test market condition analysis for neutral high volatility"""
        with patch.object(strategy_selector, '_get_volatility_analysis') as mock_vol:
            mock_vol.return_value = {
                'current_volatility': 0.45,  # High volatility
                'volatility_regime': 'high',
                'confidence': 0.75,
                'spike_probability': 0.7,
                'mean_reversion_factor': 0.05
            }

            # Modify price data for sideways movement
            neutral_data = sample_price_data.copy()
            for i, candle in enumerate(neutral_data):
                # Sideways with high volatility
                noise = np.random.normal(0, 0.05)  # 5% daily volatility
                candle['close'] = 100 + noise

            analysis = await strategy_selector._analyze_market_conditions("AAPL", neutral_data)

        assert analysis.market_condition == MarketCondition.NEUTRAL_HIGH_VOL
        assert analysis.volatility_regime == "high"
        assert analysis.volatility_level == 0.45

    @pytest.mark.asyncio
    async def test_assess_portfolio_risk_moderate_tolerance(
        self, strategy_selector, sample_portfolio_greeks
    ):
        """Test portfolio risk assessment with moderate risk tolerance"""
        assessment = await strategy_selector._assess_portfolio_risk(
            portfolio_greeks=sample_portfolio_greeks,
            risk_tolerance="moderate"
        )

        assert assessment.risk_capacity == "moderate"
        assert assessment.total_delta == 25.5
        assert assessment.total_gamma == 15.2
        assert assessment.total_vega == 120.3
        assert 0.0 <= assessment.delta_utilization <= 1.0
        assert 0.0 <= assessment.gamma_utilization <= 1.0
        assert 0.0 <= assessment.vega_utilization <= 1.0

    @pytest.mark.asyncio
    async def test_assess_portfolio_risk_conservative_tolerance(
        self, strategy_selector
    ):
        """Test portfolio risk assessment with conservative risk tolerance"""
        assessment = await strategy_selector._assess_portfolio_risk(
            portfolio_greeks=None,  # Will use defaults
            risk_tolerance="conservative"
        )

        assert assessment.risk_capacity == "conservative"
        assert assessment.total_delta == 0.0  # Default values
        assert assessment.buying_power_available == 100000.0

    def test_score_zero_dte_high_volatility(self, strategy_selector, sample_risk_assessment):
        """Test Zero-DTE strategy scoring in high volatility environment"""
        # Create high volatility market
        high_vol_market = MarketAnalysis(
            market_condition=MarketCondition.NEUTRAL_HIGH_VOL,
            volatility_regime="high",
            trend_direction="neutral",
            volatility_level=0.45,
            regime_stability=0.6,
            time_decay_environment="favorable",
            liquidity_condition="high",
            confidence_score=0.8
        )

        score = strategy_selector._score_zero_dte(high_vol_market, sample_risk_assessment)

        # Should score well in high volatility with high liquidity
        assert score > 0.7
        assert score <= 1.0

    def test_score_zero_dte_low_volatility(self, strategy_selector, sample_risk_assessment):
        """Test Zero-DTE strategy scoring in low volatility environment"""
        # Create low volatility market
        low_vol_market = MarketAnalysis(
            market_condition=MarketCondition.BULLISH_LOW_VOL,
            volatility_regime="low",
            trend_direction="bullish",
            volatility_level=0.15,
            regime_stability=0.8,
            time_decay_environment="unfavorable",
            liquidity_condition="normal",
            confidence_score=0.7
        )

        score = strategy_selector._score_zero_dte(low_vol_market, sample_risk_assessment)

        # Should score poorly in low volatility
        assert score < 0.5

    def test_score_wheel_bullish_market(self, strategy_selector, sample_risk_assessment):
        """Test Wheel strategy scoring in bullish market"""
        bullish_market = MarketAnalysis(
            market_condition=MarketCondition.BULLISH_HIGH_VOL,
            volatility_regime="high",
            trend_direction="bullish",
            volatility_level=0.35,
            regime_stability=0.7,
            time_decay_environment="favorable",
            liquidity_condition="normal",
            confidence_score=0.75
        )

        score = strategy_selector._score_wheel(bullish_market, sample_risk_assessment)

        # Wheel should score well in bullish markets with favorable time decay
        assert score > 0.7

    def test_score_wheel_bearish_market(self, strategy_selector, sample_risk_assessment):
        """Test Wheel strategy scoring in bearish market"""
        bearish_market = MarketAnalysis(
            market_condition=MarketCondition.BEARISH_HIGH_VOL,
            volatility_regime="high",
            trend_direction="bearish",
            volatility_level=0.4,
            regime_stability=0.5,
            time_decay_environment="favorable",
            liquidity_condition="normal",
            confidence_score=0.6
        )

        score = strategy_selector._score_wheel(bearish_market, sample_risk_assessment)

        # Wheel should score lower in bearish markets
        assert score < 0.6

    def test_score_iron_condor_neutral_market(self, strategy_selector, sample_risk_assessment):
        """Test Iron Condor strategy scoring in neutral market"""
        neutral_market = MarketAnalysis(
            market_condition=MarketCondition.NEUTRAL_HIGH_VOL,
            volatility_regime="high",
            trend_direction="neutral",
            volatility_level=0.38,
            regime_stability=0.8,
            time_decay_environment="favorable",
            liquidity_condition="normal",
            confidence_score=0.85
        )

        score = strategy_selector._score_iron_condor(neutral_market, sample_risk_assessment)

        # Iron Condor should score very well in neutral, high volatility markets
        assert score > 0.8

    def test_score_iron_condor_trending_market(self, strategy_selector, sample_risk_assessment):
        """Test Iron Condor strategy scoring in trending market"""
        trending_market = MarketAnalysis(
            market_condition=MarketCondition.BULLISH_HIGH_VOL,
            volatility_regime="high",
            trend_direction="bullish",
            volatility_level=0.35,
            regime_stability=0.6,
            time_decay_environment="favorable",
            liquidity_condition="normal",
            confidence_score=0.7
        )

        score = strategy_selector._score_iron_condor(trending_market, sample_risk_assessment)

        # Iron Condor should score poorly in trending markets
        assert score < 0.5

    def test_score_gamma_scalping_high_volatility(self, strategy_selector):
        """Test Gamma Scalping strategy scoring in high volatility"""
        # Create aggressive risk profile suitable for gamma scalping
        aggressive_risk = PortfolioRiskAssessment(
            total_delta=10.0,
            total_gamma=5.0,
            total_theta=-2.0,
            total_vega=50.0,
            total_rho=1.0,
            delta_utilization=0.2,
            gamma_utilization=0.15,
            vega_utilization=0.3,
            max_loss_exposure=2000.0,
            buying_power_available=100000.0,
            concentration_risk=0.2,
            risk_capacity="aggressive",
            time_horizon="short_term"
        )

        high_vol_market = MarketAnalysis(
            market_condition=MarketCondition.NEUTRAL_HIGH_VOL,
            volatility_regime="extreme",
            trend_direction="neutral",
            volatility_level=0.55,
            regime_stability=0.4,
            time_decay_environment="neutral",
            liquidity_condition="high",
            confidence_score=0.7
        )

        score = strategy_selector._score_gamma_scalping(high_vol_market, aggressive_risk)

        # Should score very well with extreme volatility and aggressive profile
        assert score > 0.8

    def test_score_gamma_scalping_low_volatility(self, strategy_selector, sample_risk_assessment):
        """Test Gamma Scalping strategy scoring in low volatility"""
        low_vol_market = MarketAnalysis(
            market_condition=MarketCondition.BULLISH_LOW_VOL,
            volatility_regime="low",
            trend_direction="bullish",
            volatility_level=0.12,
            regime_stability=0.9,
            time_decay_environment="unfavorable",
            liquidity_condition="low",
            confidence_score=0.8
        )

        score = strategy_selector._score_gamma_scalping(low_vol_market, sample_risk_assessment)

        # Should score very poorly in low volatility
        assert score < 0.3

    def test_score_cash_crisis_conditions(self, strategy_selector, sample_risk_assessment):
        """Test cash strategy scoring during crisis conditions"""
        crisis_market = MarketAnalysis(
            market_condition=MarketCondition.CRISIS,
            volatility_regime="extreme",
            trend_direction="bearish",
            volatility_level=0.8,
            regime_stability=0.1,
            time_decay_environment="unfavorable",
            liquidity_condition="low",
            confidence_score=0.3
        )

        score = strategy_selector._score_cash(crisis_market, sample_risk_assessment)

        # Cash should score very high during crisis
        assert score > 0.7

    def test_score_cash_normal_conditions(self, strategy_selector, sample_risk_assessment):
        """Test cash strategy scoring during normal conditions"""
        normal_market = MarketAnalysis(
            market_condition=MarketCondition.BULLISH_LOW_VOL,
            volatility_regime="normal",
            trend_direction="bullish",
            volatility_level=0.2,
            regime_stability=0.8,
            time_decay_environment="neutral",
            liquidity_condition="normal",
            confidence_score=0.8
        )

        score = strategy_selector._score_cash(normal_market, sample_risk_assessment)

        # Cash should score low during normal conditions
        assert score < 0.4

    @pytest.mark.asyncio
    async def test_evaluate_strategies_comprehensive(
        self, strategy_selector, sample_market_analysis, sample_risk_assessment, sample_option_chain
    ):
        """Test comprehensive strategy evaluation and scoring"""
        scores = await strategy_selector._evaluate_strategies(
            symbol="AAPL",
            market_analysis=sample_market_analysis,
            risk_assessment=sample_risk_assessment,
            option_chain=sample_option_chain
        )

        # Verify all strategies are scored
        expected_strategies = [
            StrategyRecommendation.ZERO_DTE,
            StrategyRecommendation.WHEEL,
            StrategyRecommendation.IRON_CONDOR,
            StrategyRecommendation.GAMMA_SCALPING,
            StrategyRecommendation.CASH
        ]

        for strategy in expected_strategies:
            assert strategy in scores
            assert 0.0 <= scores[strategy] <= 1.0

        # For bullish low vol market, Wheel should score well
        assert scores[StrategyRecommendation.WHEEL] > 0.6

    def test_calculate_allocation_percentage_moderate_confidence(self, strategy_selector, sample_risk_assessment):
        """Test allocation percentage calculation with moderate confidence"""
        allocation = strategy_selector._calculate_allocation_percentage(
            strategy=StrategyRecommendation.WHEEL,
            confidence=0.75,
            risk_assessment=sample_risk_assessment
        )

        # Should be reasonable allocation for moderate risk with good confidence
        assert 0.05 <= allocation <= 0.20
        assert allocation <= strategy_selector.risk_limits['max_single_strategy_allocation']

    def test_calculate_allocation_percentage_low_confidence(self, strategy_selector, sample_risk_assessment):
        """Test allocation percentage calculation with low confidence"""
        allocation = strategy_selector._calculate_allocation_percentage(
            strategy=StrategyRecommendation.IRON_CONDOR,
            confidence=0.25,
            risk_assessment=sample_risk_assessment
        )

        # Should be small allocation for low confidence
        assert allocation < 0.05

    def test_calculate_allocation_percentage_conservative_risk(self, strategy_selector):
        """Test allocation percentage calculation with conservative risk tolerance"""
        conservative_risk = PortfolioRiskAssessment(
            total_delta=5.0,
            total_gamma=2.0,
            total_theta=-1.0,
            total_vega=20.0,
            total_rho=0.5,
            delta_utilization=0.1,
            gamma_utilization=0.05,
            vega_utilization=0.15,
            max_loss_exposure=1000.0,
            buying_power_available=100000.0,
            concentration_risk=0.1,
            risk_capacity="conservative",
            time_horizon="long_term"
        )

        allocation = strategy_selector._calculate_allocation_percentage(
            strategy=StrategyRecommendation.WHEEL,
            confidence=0.8,
            risk_assessment=conservative_risk
        )

        # Conservative should get reduced allocation
        assert allocation < 0.12  # Below base 15% * 0.8 confidence

    def test_build_recommendation_prompt_comprehensive(
        self, strategy_selector, sample_market_analysis, sample_risk_assessment
    ):
        """Test comprehensive recommendation prompt building"""
        strategy_scores = {
            StrategyRecommendation.WHEEL: 0.85,
            StrategyRecommendation.IRON_CONDOR: 0.65,
            StrategyRecommendation.ZERO_DTE: 0.45,
            StrategyRecommendation.GAMMA_SCALPING: 0.35,
            StrategyRecommendation.CASH: 0.15
        }

        prompt = strategy_selector._build_recommendation_prompt(
            symbol="AAPL",
            market_analysis=sample_market_analysis,
            risk_assessment=sample_risk_assessment,
            strategy_scores=strategy_scores
        )

        # Verify prompt contains key information
        assert "AAPL" in prompt
        assert "MARKET ANALYSIS" in prompt
        assert "PORTFOLIO RISK ASSESSMENT" in prompt
        assert "STRATEGY SCORES" in prompt
        assert "wheel: 0.85" in prompt.lower()
        assert sample_market_analysis.market_condition.value in prompt
        assert str(sample_risk_assessment.risk_capacity) in prompt

    def test_parse_ai_recommendation_valid_json(self, strategy_selector):
        """Test AI recommendation parsing with valid JSON"""
        valid_response = json.dumps({
            "reasons": ["Market condition favorable", "Low risk utilization"],
            "risks": ["Volatility spike risk", "Liquidity concern"],
            "exit_conditions": ["50% profit", "Vol regime change"],
            "adjustments": ["Reduce size", "Roll positions"]
        })

        parsed = strategy_selector._parse_ai_recommendation(valid_response)

        assert len(parsed["reasons"]) == 2
        assert len(parsed["risks"]) == 2
        assert len(parsed["exit_conditions"]) == 2
        assert len(parsed["adjustments"]) == 2
        assert "Market condition favorable" in parsed["reasons"]

    def test_parse_ai_recommendation_invalid_json(self, strategy_selector):
        """Test AI recommendation parsing with invalid JSON"""
        invalid_response = "This is not valid JSON"

        parsed = strategy_selector._parse_ai_recommendation(invalid_response)

        # Should return default response
        assert "AI analysis unavailable" in parsed["reasons"]
        assert "Monitor market conditions" in parsed["risks"]

    def test_get_alternative_strategies(self, strategy_selector):
        """Test alternative strategies generation"""
        strategy_scores = {
            StrategyRecommendation.WHEEL: 0.85,
            StrategyRecommendation.IRON_CONDOR: 0.65,
            StrategyRecommendation.ZERO_DTE: 0.45,
            StrategyRecommendation.GAMMA_SCALPING: 0.35,
            StrategyRecommendation.CASH: 0.15
        }

        alternatives = strategy_selector._get_alternative_strategies(strategy_scores)

        # Should return top 3 alternatives (excluding best)
        assert len(alternatives) == 3
        assert "iron_condor" in alternatives[0].lower()  # Second best
        assert "zero_dte" in alternatives[1].lower()     # Third best
        assert "gamma_scalping" in alternatives[2].lower()  # Fourth best

    def test_validate_recommendation_within_limits(self, strategy_selector, sample_risk_assessment):
        """Test recommendation validation within risk limits"""
        recommendation = StrategyRecommendationResult(
            recommended_strategy=StrategyRecommendation.WHEEL,
            confidence_score=0.8,
            allocation_percentage=0.15,  # Within limits
            primary_reasons=["Test reason"],
            risk_considerations=["Test risk"],
            alternative_strategies=["iron_condor"],
            exit_conditions=["Test exit"],
            adjustment_triggers=["Test adjustment"]
        )

        validated = strategy_selector._validate_recommendation(recommendation, sample_risk_assessment)

        # Should remain unchanged
        assert validated.allocation_percentage == 0.15
        assert validated.confidence_score == 0.8

    def test_validate_recommendation_exceeds_limits(self, strategy_selector, sample_risk_assessment):
        """Test recommendation validation when exceeding risk limits"""
        recommendation = StrategyRecommendationResult(
            recommended_strategy=StrategyRecommendation.GAMMA_SCALPING,
            confidence_score=0.9,
            allocation_percentage=0.35,  # Exceeds 20% limit
            primary_reasons=["Test reason"],
            risk_considerations=["Test risk"],
            alternative_strategies=["wheel"],
            exit_conditions=["Test exit"],
            adjustment_triggers=["Test adjustment"]
        )

        validated = strategy_selector._validate_recommendation(recommendation, sample_risk_assessment)

        # Should be reduced to limit
        assert validated.allocation_percentage == 0.20  # Max limit
        assert "Allocation reduced due to risk limits" in validated.risk_considerations

    def test_validate_recommendation_near_risk_limits(self, strategy_selector):
        """Test recommendation validation when portfolio is near risk limits"""
        high_risk_assessment = PortfolioRiskAssessment(
            total_delta=80.0,
            total_gamma=40.0,
            total_theta=-25.0,
            total_vega=180.0,
            total_rho=15.0,
            delta_utilization=0.85,  # Near limit
            gamma_utilization=0.82,  # Near limit
            vega_utilization=0.88,   # Near limit
            max_loss_exposure=8000.0,
            buying_power_available=50000.0,
            concentration_risk=0.6,
            risk_capacity="aggressive",
            time_horizon="short_term"
        )

        recommendation = StrategyRecommendationResult(
            recommended_strategy=StrategyRecommendation.ZERO_DTE,
            confidence_score=0.8,
            allocation_percentage=0.10,
            primary_reasons=["Test reason"],
            risk_considerations=["Test risk"],
            alternative_strategies=["wheel"],
            exit_conditions=["Test exit"],
            adjustment_triggers=["Test adjustment"]
        )

        validated = strategy_selector._validate_recommendation(recommendation, high_risk_assessment)

        # Should add warning and reduce confidence
        assert "Portfolio approaching risk limits" in validated.risk_considerations
        assert validated.confidence_score < 0.8  # Reduced confidence

    def test_analyze_trend_direction_bullish(self, strategy_selector):
        """Test trend direction analysis for bullish trend"""
        # Create clearly bullish price data
        bullish_data = []
        for i in range(25):
            bullish_data.append({
                'close': 100 + (i * 2)  # Strong upward trend
            })

        trend = strategy_selector._analyze_trend_direction(bullish_data)
        assert trend == "bullish"

    def test_analyze_trend_direction_bearish(self, strategy_selector):
        """Test trend direction analysis for bearish trend"""
        # Create clearly bearish price data
        bearish_data = []
        for i in range(25):
            bearish_data.append({
                'close': 120 - (i * 2)  # Strong downward trend
            })

        trend = strategy_selector._analyze_trend_direction(bearish_data)
        assert trend == "bearish"

    def test_analyze_trend_direction_neutral(self, strategy_selector):
        """Test trend direction analysis for neutral/sideways trend"""
        # Create sideways price data
        neutral_data = []
        for i in range(25):
            neutral_data.append({
                'close': 100 + np.sin(i * 0.5)  # Sideways oscillation
            })

        trend = strategy_selector._analyze_trend_direction(neutral_data)
        assert trend == "neutral"

    def test_analyze_trend_direction_insufficient_data(self, strategy_selector):
        """Test trend direction analysis with insufficient data"""
        short_data = [{'close': 100}, {'close': 101}]  # Only 2 data points

        trend = strategy_selector._analyze_trend_direction(short_data)
        assert trend == "neutral"  # Default for insufficient data

    def test_assess_time_decay_environment(self, strategy_selector):
        """Test time decay environment assessment"""
        # High volatility should be favorable for theta strategies
        high_vol_analysis = {'volatility_regime': 'high'}
        result = strategy_selector._assess_time_decay_environment(high_vol_analysis)
        assert result == "favorable"

        # Low volatility should be unfavorable
        low_vol_analysis = {'volatility_regime': 'low'}
        result = strategy_selector._assess_time_decay_environment(low_vol_analysis)
        assert result == "unfavorable"

        # Normal volatility should be neutral
        normal_vol_analysis = {'volatility_regime': 'normal'}
        result = strategy_selector._assess_time_decay_environment(normal_vol_analysis)
        assert result == "neutral"

    def test_evaluate_liquidity_conditions(self, strategy_selector):
        """Test liquidity condition evaluation"""
        # High volume should indicate high liquidity
        high_volume_data = [{'volume': 5000000} for _ in range(10)]
        high_volume_data[-1]['volume'] = 8000000  # Recent spike

        result = strategy_selector._evaluate_liquidity_conditions(high_volume_data)
        assert result == "high"

        # Low volume should indicate low liquidity
        low_volume_data = [{'volume': 2000000} for _ in range(10)]
        low_volume_data[-1]['volume'] = 500000  # Recent drop

        result = strategy_selector._evaluate_liquidity_conditions(low_volume_data)
        assert result == "low"

    def test_classify_market_condition_all_combinations(self, strategy_selector):
        """Test market condition classification for all trend/volatility combinations"""
        test_cases = [
            ("bullish", "high", MarketCondition.BULLISH_HIGH_VOL),
            ("bullish", "low", MarketCondition.BULLISH_LOW_VOL),
            ("bearish", "high", MarketCondition.BEARISH_HIGH_VOL),
            ("bearish", "low", MarketCondition.BEARISH_LOW_VOL),
            ("neutral", "high", MarketCondition.NEUTRAL_HIGH_VOL),
            ("neutral", "low", MarketCondition.NEUTRAL_LOW_VOL),
        ]

        for trend, vol_regime, expected_condition in test_cases:
            result = strategy_selector._classify_market_condition(trend, vol_regime)
            assert result == expected_condition

    def test_create_fallback_recommendation(self, strategy_selector):
        """Test fallback recommendation creation"""
        fallback = strategy_selector._create_fallback_recommendation()

        assert fallback.recommended_strategy == StrategyRecommendation.CASH
        assert fallback.confidence_score == 0.1
        assert fallback.allocation_percentage == 0.0
        assert "Analysis failed" in fallback.primary_reasons[0]

    @pytest.mark.asyncio
    async def test_error_handling_volatility_analysis_failure(
        self, strategy_selector, sample_price_data
    ):
        """Test error handling when volatility analysis fails"""
        with patch.object(strategy_selector, '_get_volatility_analysis') as mock_vol:
            mock_vol.side_effect = Exception("Volatility analysis failed")

            analysis = await strategy_selector._analyze_market_conditions("AAPL", sample_price_data)

            # Should return fallback analysis
            assert analysis.market_condition == MarketCondition.UNCERTAIN
            assert analysis.confidence_score == 0.2

    @pytest.mark.asyncio
    async def test_deterministic_behavior_fixed_inputs(
        self, strategy_selector, sample_price_data, sample_portfolio_greeks
    ):
        """Test that AI decisions are deterministic with fixed inputs"""
        # Mock consistent AI response
        consistent_response = json.dumps({
            "reasons": ["Consistent reason 1", "Consistent reason 2"],
            "risks": ["Consistent risk 1"],
            "exit_conditions": ["Consistent exit 1"],
            "adjustments": ["Consistent adjustment 1"]
        })

        strategy_selector.ollama_client.generate_response.return_value = consistent_response

        with patch.object(strategy_selector, '_get_volatility_analysis') as mock_vol:
            mock_vol.return_value = {
                'current_volatility': 0.25,
                'volatility_regime': 'normal',
                'confidence': 0.8,
                'spike_probability': 0.3,
                'mean_reversion_factor': 0.1
            }

            # Run selection multiple times with same inputs
            result1 = await strategy_selector.select_strategy(
                symbol="AAPL",
                price_data=sample_price_data,
                portfolio_greeks=sample_portfolio_greeks,
                risk_tolerance="moderate"
            )

            result2 = await strategy_selector.select_strategy(
                symbol="AAPL",
                price_data=sample_price_data,
                portfolio_greeks=sample_portfolio_greeks,
                risk_tolerance="moderate"
            )

        # Results should be identical (deterministic)
        assert result1.recommended_strategy == result2.recommended_strategy
        assert result1.confidence_score == result2.confidence_score
        assert result1.allocation_percentage == result2.allocation_percentage
        assert result1.primary_reasons == result2.primary_reasons