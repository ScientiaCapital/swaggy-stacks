"""
Opportunity Ranking System for intelligent scoring of trading opportunities
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math

from app.core.cache import strategy_cache

logger = logging.getLogger(__name__)

@dataclass
class RankingCriteria:
    """Criteria for ranking trading opportunities"""
    momentum_weight: float = 0.25
    technical_weight: float = 0.30
    volume_weight: float = 0.20
    volatility_weight: float = 0.15
    liquidity_weight: float = 0.10

class OpportunityRanker:
    """Advanced opportunity ranking system with multiple scoring algorithms"""

    def __init__(self):
        self.criteria = RankingCriteria()
        self.historical_performance = {}
        self.market_regime = "normal"  # normal, volatile, trending
        self.last_regime_update = None

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the opportunity ranker"""
        logger.info("Initializing Opportunity Ranker...")

        # Initialize performance tracking
        self.historical_performance = {}

        # Detect current market regime
        await self._detect_market_regime()

        # Adjust criteria based on market regime
        self._adjust_criteria_for_regime()

        result = {
            'status': 'initialized',
            'market_regime': self.market_regime,
            'criteria': {
                'momentum_weight': self.criteria.momentum_weight,
                'technical_weight': self.criteria.technical_weight,
                'volume_weight': self.criteria.volume_weight,
                'volatility_weight': self.criteria.volatility_weight,
                'liquidity_weight': self.criteria.liquidity_weight
            },
            'initialized_at': datetime.utcnow().isoformat()
        }

        logger.info(f"Opportunity Ranker initialized: {result}")
        return result

    async def rank_opportunity(self, symbol: str, data: Dict[str, Any]) -> Dict[str, float]:
        """Rank a trading opportunity and return comprehensive scoring"""
        try:
            # Extract data with defaults
            rsi = data.get('rsi', 50.0)
            macd_signal = data.get('macd_signal', 'neutral')
            bollinger_position = data.get('bollinger_position', 'middle')
            volume_spike = data.get('volume_spike', False)
            volatility = data.get('volatility', 0.2)
            price_change_percent = data.get('price_change_percent', 0.0)
            current_price = data.get('current_price', 0.0)
            volume = data.get('volume', 0)

            # Calculate individual scores
            momentum_score = self._calculate_momentum_score(price_change_percent, rsi)
            technical_score = self._calculate_technical_score(rsi, macd_signal, bollinger_position)
            volume_score = self._calculate_volume_score(volume_spike, volume)
            volatility_score = self._calculate_volatility_score(volatility)
            liquidity_score = self._calculate_liquidity_score(symbol, current_price, volume)

            # Weighted composite score
            composite_score = (
                momentum_score * self.criteria.momentum_weight +
                technical_score * self.criteria.technical_weight +
                volume_score * self.criteria.volume_weight +
                volatility_score * self.criteria.volatility_weight +
                liquidity_score * self.criteria.liquidity_weight
            )

            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(data)

            # Apply market regime adjustments
            adjusted_score = self._apply_regime_adjustments(composite_score, symbol, data)

            # Normalize to 0-1 range
            final_score = max(0.0, min(1.0, adjusted_score))

            result = {
                'score': final_score,
                'confidence': confidence,
                'liquidity_score': liquidity_score,
                'component_scores': {
                    'momentum': momentum_score,
                    'technical': technical_score,
                    'volume': volume_score,
                    'volatility': volatility_score,
                    'liquidity': liquidity_score
                },
                'market_regime': self.market_regime,
                'ranked_at': datetime.utcnow().isoformat()
            }

            # Cache the ranking for performance tracking
            await self._cache_ranking(symbol, result)

            return result

        except Exception as e:
            logger.error(f"Error ranking opportunity for {symbol}: {e}")
            return {
                'score': 0.0,
                'confidence': 0.0,
                'liquidity_score': 0.0,
                'error': str(e),
                'ranked_at': datetime.utcnow().isoformat()
            }

    def _calculate_momentum_score(self, price_change_percent: float, rsi: Optional[float]) -> float:
        """Calculate momentum-based score"""
        score = 0.5  # Neutral baseline

        # Price momentum component (40% of momentum score)
        if price_change_percent > 0:
            # Positive momentum is good, but not too extreme
            momentum_component = min(price_change_percent / 5.0, 1.0) * 0.4
            score += momentum_component
        else:
            # Negative momentum reduces score
            momentum_component = max(price_change_percent / 5.0, -1.0) * 0.4
            score += momentum_component

        # RSI momentum component (60% of momentum score)
        if rsi is not None:
            if 40 <= rsi <= 60:
                # Neutral RSI is good for momentum continuation
                rsi_component = 0.3
            elif 30 <= rsi < 40:
                # Oversold can be good for reversal momentum
                rsi_component = 0.25
            elif 60 < rsi <= 70:
                # Slight overbought can indicate strong momentum
                rsi_component = 0.25
            elif rsi < 30:
                # Very oversold - potential reversal
                rsi_component = 0.15
            elif rsi > 70:
                # Overbought - momentum may be exhausted
                rsi_component = 0.1
            else:
                rsi_component = 0.0

            score += rsi_component

        return max(0.0, min(1.0, score))

    def _calculate_technical_score(self, rsi: Optional[float], macd_signal: str,
                                 bollinger_position: str) -> float:
        """Calculate technical analysis based score"""
        score = 0.0

        # RSI technical score (40% of technical score)
        if rsi is not None:
            if 30 <= rsi <= 70:
                # RSI in healthy range
                rsi_score = 0.4
            elif 20 <= rsi < 30 or 70 < rsi <= 80:
                # Approaching extremes - caution
                rsi_score = 0.2
            else:
                # Extreme RSI values
                rsi_score = 0.1

            score += rsi_score

        # MACD signal score (35% of technical score)
        macd_scores = {
            'bullish': 0.35,
            'neutral': 0.15,
            'bearish': 0.0
        }
        score += macd_scores.get(macd_signal, 0.15)

        # Bollinger Bands position score (25% of technical score)
        bollinger_scores = {
            'oversold': 0.25,      # Good buying opportunity
            'lower_half': 0.20,    # Approaching support
            'upper_half': 0.15,    # Above middle line
            'overbought': 0.05,    # Potential selling pressure
            'middle': 0.15         # Neutral position
        }
        score += bollinger_scores.get(bollinger_position, 0.10)

        return max(0.0, min(1.0, score))

    def _calculate_volume_score(self, volume_spike: bool, volume: int) -> float:
        """Calculate volume-based score"""
        score = 0.3  # Base score for normal volume

        # Volume spike bonus
        if volume_spike:
            score += 0.4

        # Volume magnitude score
        if volume > 1000000:  # High volume
            score += 0.3
        elif volume > 100000:  # Medium volume
            score += 0.2
        elif volume > 10000:   # Low but acceptable volume
            score += 0.1
        else:
            # Very low volume - penalty
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _calculate_volatility_score(self, volatility: Optional[float]) -> float:
        """Calculate volatility-based score"""
        if volatility is None:
            return 0.5  # Neutral score for unknown volatility

        # Optimal volatility range for trading opportunities
        if 0.15 <= volatility <= 0.35:
            # Good volatility for trading
            return 1.0
        elif 0.10 <= volatility < 0.15:
            # Low volatility - limited profit potential
            return 0.6
        elif 0.35 < volatility <= 0.50:
            # High volatility - higher risk but good potential
            return 0.7
        elif volatility > 0.50:
            # Very high volatility - risky
            return 0.3
        else:
            # Very low volatility
            return 0.4

    def _calculate_liquidity_score(self, symbol: str, current_price: float, volume: int) -> float:
        """Calculate liquidity-based score"""
        score = 0.5  # Base liquidity score

        # Symbol-based liquidity adjustments
        high_liquidity_symbols = {
            'SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL', 'AMZN',
            'META', 'BTCUSD', 'ETHUSD', 'IWM', 'GLD', 'SLV'
        }

        if symbol in high_liquidity_symbols:
            score += 0.3

        # Volume-based liquidity
        if volume > 1000000:
            score += 0.2
        elif volume > 100000:
            score += 0.1

        # Price-based liquidity (higher priced stocks often more liquid)
        if current_price > 100:
            score += 0.1
        elif current_price > 50:
            score += 0.05

        # Crypto gets special treatment for 24/7 liquidity
        if symbol.endswith('USD'):
            score += 0.15

        return max(0.0, min(1.0, score))

    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence in the ranking based on data quality"""
        confidence = 1.0

        # Penalize missing data
        required_fields = ['rsi', 'current_price', 'volume', 'volatility']
        missing_fields = sum(1 for field in required_fields if data.get(field) is None)
        confidence -= (missing_fields / len(required_fields)) * 0.4

        # Penalize extreme or unrealistic values
        rsi = data.get('rsi')
        if rsi is not None and (rsi < 0 or rsi > 100):
            confidence -= 0.2

        volatility = data.get('volatility')
        if volatility is not None and volatility > 1.0:  # > 100% volatility
            confidence -= 0.1

        volume = data.get('volume', 0)
        if volume <= 0:
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))

    async def _detect_market_regime(self):
        """Detect current market regime"""
        try:
            # This is a simplified regime detection
            # In production, this would analyze market indices, VIX, etc.

            # For now, use a simple heuristic based on cached data
            cached_regime = strategy_cache.cache.get('market_regime')
            if cached_regime:
                self.market_regime = cached_regime.get('regime', 'normal')
            else:
                # Default to normal regime
                self.market_regime = "normal"

            self.last_regime_update = datetime.utcnow()

        except Exception as e:
            logger.warning(f"Error detecting market regime: {e}")
            self.market_regime = "normal"

    def _adjust_criteria_for_regime(self):
        """Adjust ranking criteria based on market regime"""
        if self.market_regime == "volatile":
            # In volatile markets, emphasize technical analysis and reduce momentum weight
            self.criteria.technical_weight = 0.35
            self.criteria.momentum_weight = 0.20
            self.criteria.volatility_weight = 0.20
            self.criteria.volume_weight = 0.15
            self.criteria.liquidity_weight = 0.10

        elif self.market_regime == "trending":
            # In trending markets, emphasize momentum
            self.criteria.momentum_weight = 0.35
            self.criteria.technical_weight = 0.25
            self.criteria.volume_weight = 0.20
            self.criteria.volatility_weight = 0.10
            self.criteria.liquidity_weight = 0.10

        else:  # normal market
            # Balanced approach
            self.criteria.momentum_weight = 0.25
            self.criteria.technical_weight = 0.30
            self.criteria.volume_weight = 0.20
            self.criteria.volatility_weight = 0.15
            self.criteria.liquidity_weight = 0.10

    def _apply_regime_adjustments(self, score: float, symbol: str, data: Dict[str, Any]) -> float:
        """Apply market regime specific adjustments to the score"""
        adjusted_score = score

        if self.market_regime == "volatile":
            # In volatile markets, prefer more stable, liquid symbols
            if symbol in ['SPY', 'QQQ', 'GLD', 'TLT']:
                adjusted_score += 0.1

            # Reduce score for extremely volatile individual stocks
            volatility = data.get('volatility', 0.2)
            if volatility > 0.5:
                adjusted_score -= 0.15

        elif self.market_regime == "trending":
            # In trending markets, boost momentum-based opportunities
            price_change = data.get('price_change_percent', 0.0)
            if abs(price_change) > 2.0:  # Strong price movement
                adjusted_score += 0.1

        # Always boost crypto in 24/7 markets
        if symbol.endswith('USD'):
            adjusted_score += 0.05

        return adjusted_score

    async def _cache_ranking(self, symbol: str, ranking_result: Dict[str, Any]):
        """Cache ranking result for performance tracking"""
        try:
            cache_key = f"ranking:{symbol}:{datetime.utcnow().strftime('%Y%m%d')}"
            strategy_cache.cache.set(cache_key, ranking_result, ttl=86400)  # 24 hours
        except Exception as e:
            logger.warning(f"Error caching ranking for {symbol}: {e}")

    async def get_top_opportunities(self, analyses: List[Dict[str, Any]],
                                  limit: int = 20) -> List[Dict[str, Any]]:
        """Get top opportunities from a list of analyses"""
        try:
            # Rank all opportunities
            ranked_opportunities = []
            for analysis in analyses:
                symbol = analysis.get('symbol')
                if symbol:
                    ranking = await self.rank_opportunity(symbol, analysis)
                    ranked_opportunities.append({
                        'symbol': symbol,
                        'ranking': ranking,
                        'analysis': analysis
                    })

            # Sort by score and confidence
            ranked_opportunities.sort(
                key=lambda x: (x['ranking']['score'], x['ranking']['confidence']),
                reverse=True
            )

            return ranked_opportunities[:limit]

        except Exception as e:
            logger.error(f"Error getting top opportunities: {e}")
            return []

    def get_ranking_criteria(self) -> Dict[str, float]:
        """Get current ranking criteria weights"""
        return {
            'momentum_weight': self.criteria.momentum_weight,
            'technical_weight': self.criteria.technical_weight,
            'volume_weight': self.criteria.volume_weight,
            'volatility_weight': self.criteria.volatility_weight,
            'liquidity_weight': self.criteria.liquidity_weight,
            'market_regime': self.market_regime
        }

    async def update_market_regime(self, new_regime: str):
        """Manually update market regime"""
        if new_regime in ['normal', 'volatile', 'trending']:
            self.market_regime = new_regime
            self._adjust_criteria_for_regime()
            self.last_regime_update = datetime.utcnow()

            # Cache the regime
            try:
                regime_data = {
                    'regime': new_regime,
                    'updated_at': self.last_regime_update.isoformat()
                }
                strategy_cache.cache.set('market_regime', regime_data, ttl=3600)
            except Exception as e:
                logger.warning(f"Error caching market regime: {e}")

            logger.info(f"Market regime updated to: {new_regime}")
            return True
        else:
            logger.warning(f"Invalid market regime: {new_regime}")
            return False