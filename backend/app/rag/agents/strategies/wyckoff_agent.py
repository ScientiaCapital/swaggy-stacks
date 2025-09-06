"""
Wyckoff Method Trading Agent - Enhanced with RAG and Pattern Learning
Integrates Wyckoff accumulation/distribution analysis with LangChain agents
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

from langchain.agents import Tool
from langchain.schema import HumanMessage

from backend.app.rag.agents.base_agent import BaseTradingAgent, TradingSignal

logger = logging.getLogger(__name__)

class WyckoffPhase(Enum):
    """Wyckoff market cycle phases"""
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"

@dataclass
class WyckoffSignal:
    """Represents a Wyckoff trading signal"""
    phase: WyckoffPhase
    strength: float
    volume_confirmation: bool
    price_action_confirmation: bool
    effort_vs_result: str  # "normal", "weak", "strong"
    supply_demand_balance: str  # "supply", "demand", "neutral"

@dataclass
class EffortVsResult:
    """Analyzes effort (volume) vs result (price movement)"""
    volume_effort: float
    price_result: float
    ratio: float
    interpretation: str
    bullish_score: float

class WyckoffAgent(BaseTradingAgent):
    """
    Wyckoff Method trading agent with RAG capabilities
    Focuses on supply/demand analysis and market structure
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            agent_name="wyckoff_agent",
            strategy_type="wyckoff_method",
            **kwargs
        )
        
        # Wyckoff analysis parameters
        self.volume_lookback = 20
        self.price_lookback = 50
        self.phase_detection_periods = 100
        
        # Volume analysis thresholds
        self.high_volume_threshold = 1.5  # 1.5x average volume
        self.low_volume_threshold = 0.7   # 0.7x average volume
        
        # Effort vs Result thresholds
        self.strong_effort_threshold = 1.3
        self.weak_effort_threshold = 0.7
        
        # Phase transition confidence
        self.phase_confidence_threshold = 0.7
        
        logger.info("WyckoffAgent initialized with market structure focus")
    
    async def _create_tools(self) -> List[Tool]:
        """Create LangChain tools specific to Wyckoff analysis"""
        return [
            Tool(
                name="analyze_wyckoff_phase",
                func=self._analyze_wyckoff_phase,
                description="Identify current Wyckoff market cycle phase"
            ),
            Tool(
                name="calculate_effort_vs_result",
                func=self._calculate_effort_vs_result,
                description="Analyze volume (effort) vs price movement (result)"
            ),
            Tool(
                name="detect_supply_demand_zones",
                func=self._detect_supply_demand_zones,
                description="Identify key supply and demand zones"
            ),
            Tool(
                name="analyze_volume_patterns",
                func=self._analyze_volume_patterns,
                description="Analyze volume patterns for Wyckoff confirmation"
            ),
            Tool(
                name="find_wyckoff_patterns",
                func=self._find_similar_wyckoff_patterns,
                description="Find similar Wyckoff patterns in historical data"
            )
        ]
    
    def _analyze_wyckoff_phase(self, price_volume_data: str) -> str:
        """Analyze current Wyckoff phase"""
        try:
            # Parse combined price and volume data
            data_points = price_volume_data.strip().split('|')
            if len(data_points) < 2:
                return "Insufficient data format. Expected: prices|volumes"
            
            prices = [float(x.strip()) for x in data_points[0].split(',')]
            volumes = [float(x.strip()) for x in data_points[1].split(',')]
            
            if len(prices) < self.phase_detection_periods or len(volumes) < self.phase_detection_periods:
                return "Insufficient data for Wyckoff phase analysis"
            
            phase = self._identify_wyckoff_phase(prices, volumes)
            
            result = f"Wyckoff Phase Analysis:\n"
            result += f"  Current Phase: {phase.phase.value.upper()}\n"
            result += f"  Phase Strength: {phase.strength:.2f}\n"
            result += f"  Volume Confirmation: {phase.volume_confirmation}\n"
            result += f"  Price Action Confirmation: {phase.price_action_confirmation}\n"
            result += f"  Effort vs Result: {phase.effort_vs_result}\n"
            result += f"  Supply/Demand Balance: {phase.supply_demand_balance}\n"
            
            return result
            
        except Exception as e:
            return f"Error analyzing Wyckoff phase: {str(e)}"
    
    def _calculate_effort_vs_result(self, price_volume_data: str) -> str:
        """Calculate effort vs result analysis"""
        try:
            data_points = price_volume_data.strip().split('|')
            prices = [float(x.strip()) for x in data_points[0].split(',')]
            volumes = [float(x.strip()) for x in data_points[1].split(',')]
            
            effort_result = self._analyze_effort_vs_result(prices, volumes)
            
            result = f"Effort vs Result Analysis:\n"
            result += f"  Volume Effort: {effort_result.volume_effort:.2f}\n"
            result += f"  Price Result: {effort_result.price_result:.2f}\n"
            result += f"  Effort/Result Ratio: {effort_result.ratio:.2f}\n"
            result += f"  Interpretation: {effort_result.interpretation}\n"
            result += f"  Bullish Score: {effort_result.bullish_score:.2f}\n"
            
            return result
            
        except Exception as e:
            return f"Error calculating effort vs result: {str(e)}"
    
    def _detect_supply_demand_zones(self, price_data: str) -> str:
        """Detect supply and demand zones"""
        try:
            prices = [float(x.strip()) for x in price_data.split(',')]
            
            zones = self._find_supply_demand_zones(prices)
            
            if not zones:
                return "No significant supply/demand zones detected"
            
            result = f"Supply/Demand Zones:\n"
            for zone in zones[:5]:  # Top 5 zones
                result += f"  {zone['type'].upper()}: {zone['price_range'][0]:.4f} - {zone['price_range'][1]:.4f} "
                result += f"(strength: {zone['strength']:.2f})\n"
            
            return result
            
        except Exception as e:
            return f"Error detecting supply/demand zones: {str(e)}"
    
    def _analyze_volume_patterns(self, volume_data: str) -> str:
        """Analyze volume patterns for Wyckoff confirmation"""
        try:
            volumes = [float(x.strip()) for x in volume_data.split(',')]
            
            patterns = self._identify_volume_patterns(volumes)
            
            result = f"Volume Pattern Analysis:\n"
            result += f"  Average Volume: {np.mean(volumes):.0f}\n"
            result += f"  Volume Trend: {patterns['trend']}\n"
            result += f"  High Volume Events: {patterns['high_volume_count']}\n"
            result += f"  Low Volume Events: {patterns['low_volume_count']}\n"
            result += f"  Volume Volatility: {patterns['volatility']:.2f}\n"
            
            return result
            
        except Exception as e:
            return f"Error analyzing volume patterns: {str(e)}"
    
    async def _find_similar_wyckoff_patterns(self, pattern_description: str) -> str:
        """Find similar Wyckoff patterns in historical data"""
        try:
            features = {
                'pattern_description': pattern_description,
                'strategy': 'wyckoff_method',
                'pattern_type': 'wyckoff_structure'
            }
            
            similar_patterns = await self.find_similar_patterns(features, similarity_threshold=0.7)
            
            if not similar_patterns:
                return "No similar Wyckoff patterns found"
            
            result = f"Found {len(similar_patterns)} similar patterns:\n"
            for i, pattern in enumerate(similar_patterns[:3], 1):
                result += f"  {i}. {pattern.pattern_name} "
                result += f"(similarity: {pattern.similarity:.2f}, success: {pattern.success_rate:.1%})\n"
                
            return result
            
        except Exception as e:
            return f"Error finding similar patterns: {str(e)}"
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> TradingSignal:
        """
        Main analysis method - combines Wyckoff analysis with RAG enhancement
        """
        try:
            prices = market_data.get('prices', [])
            volumes = market_data.get('volumes', [])
            symbol = market_data.get('symbol', 'UNKNOWN')
            current_price = market_data.get('current_price', 0.0)
            
            if len(prices) < self.phase_detection_periods or len(volumes) < self.volume_lookback:
                return TradingSignal(
                    agent_type=self.agent_name,
                    strategy_name=self.strategy_type,
                    symbol=symbol,
                    action="HOLD",
                    confidence=0.0,
                    reasoning="Insufficient price/volume data for Wyckoff analysis"
                )
            
            # Extract market features
            features = self._extract_market_features(market_data)
            
            # Perform Wyckoff analysis
            wyckoff_result = self._perform_wyckoff_analysis(prices, volumes, current_price)
            
            # Find similar historical patterns
            similar_patterns = await self.find_similar_patterns(features)
            
            # Get contextual information
            pattern_context = await self.get_pattern_context(features)
            
            # Generate enhanced signal with RAG context
            signal = self._generate_enhanced_signal(
                wyckoff_result,
                similar_patterns,
                pattern_context,
                symbol,
                current_price
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in Wyckoff agent analysis: {e}")
            return TradingSignal(
                agent_type=self.agent_name,
                strategy_name=self.strategy_type,
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                reasoning=f"Analysis error: {str(e)}"
            )
    
    def _extract_market_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Wyckoff-specific features from market data"""
        prices = market_data.get('prices', [])
        volumes = market_data.get('volumes', [])
        
        if len(prices) < self.phase_detection_periods:
            return {'error': 'insufficient_data'}
        
        # Wyckoff phase analysis
        wyckoff_signal = self._identify_wyckoff_phase(prices, volumes)
        
        # Effort vs Result analysis
        effort_result = self._analyze_effort_vs_result(prices, volumes)
        
        # Supply/Demand zones
        zones = self._find_supply_demand_zones(prices)
        
        # Volume characteristics
        volume_patterns = self._identify_volume_patterns(volumes)
        
        features = {
            'wyckoff_phase': wyckoff_signal.phase.value,
            'phase_strength': wyckoff_signal.strength,
            'volume_confirmation': wyckoff_signal.volume_confirmation,
            'price_confirmation': wyckoff_signal.price_action_confirmation,
            'effort_vs_result': effort_result.interpretation,
            'effort_result_ratio': effort_result.ratio,
            'bullish_score': effort_result.bullish_score,
            'supply_demand_balance': wyckoff_signal.supply_demand_balance,
            'supply_zones_count': len([z for z in zones if z['type'] == 'supply']),
            'demand_zones_count': len([z for z in zones if z['type'] == 'demand']),
            'volume_trend': volume_patterns['trend'],
            'volume_volatility': volume_patterns['volatility'],
            'high_volume_events': volume_patterns['high_volume_count'],
        }
        
        # Overall market characteristics
        returns = np.diff(np.log(prices))
        features.update({
            'volatility': np.std(returns) * np.sqrt(252),
            'trend_strength': self._calculate_trend_strength(prices),
            'momentum': np.mean(returns[-5:]) if len(returns) >= 5 else 0.0,
        })
        
        return features
    
    def _perform_wyckoff_analysis(self, prices: List[float], volumes: List[float], current_price: float) -> Dict[str, Any]:
        """Perform core Wyckoff analysis"""
        try:
            # Identify Wyckoff phase
            wyckoff_signal = self._identify_wyckoff_phase(prices, volumes)
            
            # Analyze effort vs result
            effort_result = self._analyze_effort_vs_result(prices, volumes)
            
            # Find supply/demand zones
            zones = self._find_supply_demand_zones(prices)
            
            # Volume pattern analysis
            volume_patterns = self._identify_volume_patterns(volumes)
            
            # Generate trading decision
            action, confidence = self._wyckoff_trading_decision(
                wyckoff_signal, effort_result, zones, current_price
            )
            
            return {
                'wyckoff_phase': wyckoff_signal.phase,
                'phase_strength': wyckoff_signal.strength,
                'effort_result': effort_result,
                'supply_demand_zones': zones,
                'volume_patterns': volume_patterns,
                'action': action,
                'confidence': confidence,
                'volume_confirmation': wyckoff_signal.volume_confirmation,
                'price_confirmation': wyckoff_signal.price_action_confirmation
            }
            
        except Exception as e:
            logger.error(f"Error in Wyckoff analysis: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _identify_wyckoff_phase(self, prices: List[float], volumes: List[float]) -> WyckoffSignal:
        """Identify current Wyckoff market cycle phase"""
        
        # Calculate price and volume trends
        price_trend = self._calculate_trend_strength(prices[-self.price_lookback:])
        volume_trend = self._calculate_volume_trend(volumes[-self.volume_lookback:])
        
        # Calculate effort vs result
        effort_result = self._analyze_effort_vs_result(prices, volumes)
        
        # Determine phase based on price action and volume
        phase = WyckoffPhase.UNKNOWN
        strength = 0.0
        
        # Phase identification logic
        if price_trend < -0.3:  # Strong downtrend
            if volume_trend > 0 and effort_result.ratio < 0.8:  # High volume, small price moves
                phase = WyckoffPhase.ACCUMULATION
                strength = min(1.0, abs(volume_trend) * 0.8)
            else:
                phase = WyckoffPhase.MARKDOWN
                strength = min(1.0, abs(price_trend) * 0.7)
                
        elif price_trend > 0.3:  # Strong uptrend
            if volume_trend < 0 and effort_result.ratio < 0.8:  # Low volume, small price moves  
                phase = WyckoffPhase.DISTRIBUTION
                strength = min(1.0, abs(volume_trend) * 0.8)
            else:
                phase = WyckoffPhase.MARKUP
                strength = min(1.0, price_trend * 0.7)
                
        elif abs(price_trend) < 0.1:  # Sideways movement
            if volume_trend > 0.2:  # High volume during consolidation
                phase = WyckoffPhase.ACCUMULATION
                strength = volume_trend * 0.6
            elif volume_trend < -0.2:  # Low volume during consolidation
                phase = WyckoffPhase.DISTRIBUTION
                strength = abs(volume_trend) * 0.6
        
        # Volume and price action confirmation
        volume_confirmation = self._check_volume_confirmation(phase, volumes, prices)
        price_confirmation = self._check_price_confirmation(phase, prices)
        
        # Supply/demand balance
        supply_demand = self._assess_supply_demand_balance(effort_result, phase)
        
        return WyckoffSignal(
            phase=phase,
            strength=strength,
            volume_confirmation=volume_confirmation,
            price_action_confirmation=price_confirmation,
            effort_vs_result=effort_result.interpretation,
            supply_demand_balance=supply_demand
        )
    
    def _analyze_effort_vs_result(self, prices: List[float], volumes: List[float]) -> EffortVsResult:
        """Analyze volume (effort) vs price movement (result)"""
        
        if len(prices) < 10 or len(volumes) < 10:
            return EffortVsResult(0, 0, 0, "insufficient_data", 0.5)
        
        # Calculate recent volume effort (normalized)
        recent_volumes = volumes[-10:]
        avg_volume = np.mean(volumes[-50:]) if len(volumes) >= 50 else np.mean(volumes)
        volume_effort = np.mean(recent_volumes) / avg_volume
        
        # Calculate recent price result (absolute percentage change)
        price_result = abs(prices[-1] - prices[-10]) / prices[-10]
        
        # Calculate effort vs result ratio
        if price_result == 0:
            ratio = volume_effort * 2  # High effort with no result is bearish
        else:
            ratio = volume_effort / (price_result * 100)  # Normalize price result
        
        # Interpret the relationship
        if ratio > self.strong_effort_threshold:
            if price_result < 0.01:  # Small price movement despite high volume
                interpretation = "weak_hands_selling" if prices[-1] < prices[-5] else "absorption"
                bullish_score = 0.7 if prices[-1] >= prices[-5] else 0.3
            else:
                interpretation = "strong_trend"
                bullish_score = 0.8 if prices[-1] > prices[-10] else 0.2
        elif ratio < self.weak_effort_threshold:
            interpretation = "natural_movement"
            bullish_score = 0.6 if prices[-1] > prices[-10] else 0.4
        else:
            interpretation = "normal_activity"
            bullish_score = 0.5
        
        return EffortVsResult(
            volume_effort=volume_effort,
            price_result=price_result,
            ratio=ratio,
            interpretation=interpretation,
            bullish_score=bullish_score
        )
    
    def _find_supply_demand_zones(self, prices: List[float]) -> List[Dict[str, Any]]:
        """Find significant supply and demand zones"""
        zones = []
        
        if len(prices) < 20:
            return zones
        
        # Find swing highs and lows
        for i in range(10, len(prices) - 10):
            # Potential supply zone (swing high)
            if (prices[i] == max(prices[i-5:i+6])):  # Local high
                strength = self._calculate_zone_strength(prices, i, "supply")
                if strength > 0.3:  # Only significant zones
                    zones.append({
                        'type': 'supply',
                        'price_range': (prices[i] * 0.995, prices[i] * 1.005),  # 0.5% range
                        'center_price': prices[i],
                        'strength': strength,
                        'index': i
                    })
            
            # Potential demand zone (swing low)
            elif (prices[i] == min(prices[i-5:i+6])):  # Local low
                strength = self._calculate_zone_strength(prices, i, "demand")
                if strength > 0.3:  # Only significant zones
                    zones.append({
                        'type': 'demand',
                        'price_range': (prices[i] * 0.995, prices[i] * 1.005),  # 0.5% range
                        'center_price': prices[i],
                        'strength': strength,
                        'index': i
                    })
        
        # Sort by strength and return top zones
        zones.sort(key=lambda x: x['strength'], reverse=True)
        return zones[:10]  # Top 10 zones
    
    def _calculate_zone_strength(self, prices: List[float], index: int, zone_type: str) -> float:
        """Calculate the strength of a supply/demand zone"""
        if index < 10 or index >= len(prices) - 10:
            return 0.0
        
        zone_price = prices[index]
        
        # Check how many times price reacted from this level
        reactions = 0
        for i in range(index + 1, min(index + 50, len(prices))):
            if zone_type == "supply" and prices[i] <= zone_price * 1.01 and prices[i] >= zone_price * 0.99:
                # Price came back to supply zone
                if i + 5 < len(prices) and max(prices[i:i+5]) < prices[i]:
                    reactions += 1
            elif zone_type == "demand" and prices[i] <= zone_price * 1.01 and prices[i] >= zone_price * 0.99:
                # Price came back to demand zone
                if i + 5 < len(prices) and min(prices[i:i+5]) > prices[i]:
                    reactions += 1
        
        # Base strength on reactions and distance from current price
        base_strength = min(1.0, reactions * 0.3)
        current_distance = abs(prices[-1] - zone_price) / zone_price
        distance_factor = max(0.1, 1.0 - current_distance * 5)  # Closer zones are stronger
        
        return base_strength * distance_factor
    
    def _identify_volume_patterns(self, volumes: List[float]) -> Dict[str, Any]:
        """Identify volume patterns"""
        if len(volumes) < self.volume_lookback:
            return {'trend': 'unknown', 'volatility': 0.0, 'high_volume_count': 0, 'low_volume_count': 0}
        
        recent_volumes = volumes[-self.volume_lookback:]
        avg_volume = np.mean(volumes)
        
        # Volume trend
        volume_trend = self._calculate_volume_trend(recent_volumes)
        
        # Volume volatility
        volume_volatility = np.std(recent_volumes) / np.mean(recent_volumes)
        
        # Count significant volume events
        high_volume_count = len([v for v in recent_volumes if v > avg_volume * self.high_volume_threshold])
        low_volume_count = len([v for v in recent_volumes if v < avg_volume * self.low_volume_threshold])
        
        trend_description = "increasing" if volume_trend > 0.1 else "decreasing" if volume_trend < -0.1 else "stable"
        
        return {
            'trend': trend_description,
            'volatility': volume_volatility,
            'high_volume_count': high_volume_count,
            'low_volume_count': low_volume_count,
            'avg_volume': avg_volume
        }
    
    def _calculate_volume_trend(self, volumes: List[float]) -> float:
        """Calculate volume trend using linear regression"""
        if len(volumes) < 5:
            return 0.0
        
        x = np.arange(len(volumes))
        slope = np.polyfit(x, volumes, 1)[0]
        
        # Normalize by average volume
        return slope / np.mean(volumes)
    
    def _check_volume_confirmation(self, phase: WyckoffPhase, volumes: List[float], prices: List[float]) -> bool:
        """Check if volume confirms the identified Wyckoff phase"""
        if len(volumes) < self.volume_lookback:
            return False
        
        recent_volumes = volumes[-self.volume_lookback:]
        avg_volume = np.mean(volumes)
        volume_trend = self._calculate_volume_trend(recent_volumes)
        
        if phase == WyckoffPhase.ACCUMULATION:
            # Should see increasing volume on up moves, decreasing on down moves
            return volume_trend > 0.1
        elif phase == WyckoffPhase.DISTRIBUTION:
            # Should see high volume on distribution moves
            return np.mean(recent_volumes) > avg_volume * 1.2
        elif phase == WyckoffPhase.MARKUP:
            # Should see increasing volume supporting the trend
            return volume_trend > 0.0 and prices[-1] > prices[-10]
        elif phase == WyckoffPhase.MARKDOWN:
            # Should see volume accompanying the decline
            return np.mean(recent_volumes) > avg_volume * 0.8
        
        return False
    
    def _check_price_confirmation(self, phase: WyckoffPhase, prices: List[float]) -> bool:
        """Check if price action confirms the identified Wyckoff phase"""
        if len(prices) < 20:
            return False
        
        trend = self._calculate_trend_strength(prices[-20:])
        
        if phase == WyckoffPhase.ACCUMULATION:
            # Should see sideways to slight upward movement
            return -0.1 <= trend <= 0.3
        elif phase == WyckoffPhase.DISTRIBUTION:
            # Should see sideways to slight downward movement  
            return -0.3 <= trend <= 0.1
        elif phase == WyckoffPhase.MARKUP:
            # Should see strong upward movement
            return trend > 0.2
        elif phase == WyckoffPhase.MARKDOWN:
            # Should see strong downward movement
            return trend < -0.2
        
        return False
    
    def _assess_supply_demand_balance(self, effort_result: EffortVsResult, phase: WyckoffPhase) -> str:
        """Assess overall supply/demand balance"""
        
        if effort_result.bullish_score > 0.6:
            if phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP]:
                return "demand"
            else:
                return "neutral"  # Mixed signals
        elif effort_result.bullish_score < 0.4:
            if phase in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN]:
                return "supply"
            else:
                return "neutral"  # Mixed signals
        else:
            return "neutral"
    
    def _wyckoff_trading_decision(
        self, 
        wyckoff_signal: WyckoffSignal, 
        effort_result: EffortVsResult, 
        zones: List[Dict[str, Any]], 
        current_price: float
    ) -> Tuple[str, float]:
        """Generate trading decision based on Wyckoff analysis"""
        
        base_confidence = 0.5
        action = "HOLD"
        
        # Phase-based decision
        if wyckoff_signal.phase == WyckoffPhase.ACCUMULATION and wyckoff_signal.strength > 0.5:
            action = "BUY"
            base_confidence = 0.7
        elif wyckoff_signal.phase == WyckoffPhase.MARKUP and wyckoff_signal.strength > 0.6:
            action = "BUY"
            base_confidence = 0.8
        elif wyckoff_signal.phase == WyckoffPhase.DISTRIBUTION and wyckoff_signal.strength > 0.5:
            action = "SELL"
            base_confidence = 0.7
        elif wyckoff_signal.phase == WyckoffPhase.MARKDOWN and wyckoff_signal.strength > 0.6:
            action = "SELL"
            base_confidence = 0.8
        
        # Confirmation adjustments
        confirmations = 0
        if wyckoff_signal.volume_confirmation:
            confirmations += 1
        if wyckoff_signal.price_action_confirmation:
            confirmations += 1
        
        confirmation_bonus = confirmations * 0.1
        base_confidence += confirmation_bonus
        
        # Effort vs Result adjustment
        if effort_result.interpretation in ["absorption", "strong_trend"]:
            base_confidence += 0.1
        elif effort_result.interpretation == "weak_hands_selling":
            base_confidence += 0.05
        
        # Supply/Demand zone proximity
        for zone in zones:
            zone_distance = abs(current_price - zone['center_price']) / zone['center_price']
            if zone_distance < 0.02:  # Within 2% of zone
                if zone['type'] == 'demand' and action == "BUY":
                    base_confidence += zone['strength'] * 0.1
                elif zone['type'] == 'supply' and action == "SELL":
                    base_confidence += zone['strength'] * 0.1
        
        # Cap confidence
        final_confidence = min(1.0, base_confidence)
        
        return action, final_confidence
    
    def _generate_enhanced_signal(
        self,
        wyckoff_result: Dict[str, Any],
        similar_patterns: List,
        pattern_context: str,
        symbol: str,
        current_price: float
    ) -> TradingSignal:
        """Generate enhanced trading signal with RAG context"""
        
        action = wyckoff_result.get('action', 'HOLD')
        base_confidence = wyckoff_result.get('confidence', 0.0)
        
        # Adjust confidence based on similar patterns
        confidence_adjustment = 0.0
        if similar_patterns:
            avg_success_rate = np.mean([p.success_rate for p in similar_patterns])
            confidence_adjustment = (avg_success_rate - 0.5) * 0.3
        
        final_confidence = min(1.0, max(0.0, base_confidence + confidence_adjustment))
        
        # Generate reasoning
        reasoning_parts = []
        
        wyckoff_phase = wyckoff_result.get('wyckoff_phase', WyckoffPhase.UNKNOWN)
        phase_strength = wyckoff_result.get('phase_strength', 0.0)
        reasoning_parts.append(f"Wyckoff Phase: {wyckoff_phase.value.upper()} (strength: {phase_strength:.2f})")
        
        if wyckoff_result.get('volume_confirmation'):
            reasoning_parts.append("Volume confirms phase")
        
        if wyckoff_result.get('price_confirmation'):
            reasoning_parts.append("Price action confirms phase")
        
        effort_result = wyckoff_result.get('effort_result')
        if effort_result:
            reasoning_parts.append(f"Effort vs Result: {effort_result.interpretation}")
        
        zones = wyckoff_result.get('supply_demand_zones', [])
        if zones:
            reasoning_parts.append(f"Found {len(zones)} supply/demand zones")
        
        if similar_patterns:
            reasoning_parts.append(f"Historical patterns: {len(similar_patterns)} similar cases "
                                 f"with {avg_success_rate:.1%} success rate")
        
        metadata = {
            'wyckoff_phase': wyckoff_phase.value,
            'phase_strength': phase_strength,
            'volume_confirmation': wyckoff_result.get('volume_confirmation', False),
            'price_confirmation': wyckoff_result.get('price_confirmation', False),
            'effort_vs_result': effort_result.interpretation if effort_result else 'unknown',
            'supply_zones_count': len([z for z in zones if z['type'] == 'supply']),
            'demand_zones_count': len([z for z in zones if z['type'] == 'demand']),
            'confidence_adjustment': confidence_adjustment,
            'pattern_context_available': bool(pattern_context and pattern_context != "No similar patterns found.")
        }
        
        return TradingSignal(
            agent_type=self.agent_name,
            strategy_name=self.strategy_type,
            symbol=symbol,
            action=action,
            confidence=final_confidence,
            reasoning=" | ".join(reasoning_parts),
            entry_price=current_price if action != "HOLD" else None,
            metadata=metadata
        )
    
    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength indicator"""
        if len(prices) < 10:
            return 0.0
        
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        normalized_slope = slope / np.mean(prices)
        
        return np.tanh(normalized_slope * 100)

# Test function
async def test_wyckoff_agent():
    """Test the Wyckoff Trading Agent"""
    print("🧪 Testing Wyckoff Trading Agent...")
    
    agent = WyckoffAgent()
    await agent.initialize()
    
    # Test with sample market data showing accumulation pattern
    sample_data = {
        'symbol': 'TEST',
        'current_price': 102.0,
        'prices': [100, 99, 98, 99, 100, 101, 100, 99, 101, 102, 101, 100, 102, 103, 102, 101, 102, 103, 102, 102] * 5,  # Sideways pattern
        'volumes': [800, 1200, 1500, 1100, 900, 1000, 1300, 1600, 1200, 1000, 1100, 900, 1200, 1400, 1100, 1000, 1200, 1300, 1100, 1000] * 5
    }
    
    # Generate signal
    signal = await agent.analyze_market(sample_data)
    
    print(f"✅ Generated signal:")
    print(f"   Action: {signal.action}")
    print(f"   Confidence: {signal.confidence:.2f}")
    print(f"   Reasoning: {signal.reasoning}")
    print(f"   Metadata: {signal.metadata}")
    
    # Test health check
    health = await agent.health_check()
    print(f"✅ Health check: {health['is_initialized']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_wyckoff_agent())