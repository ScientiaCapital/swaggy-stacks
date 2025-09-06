"""
Fibonacci Golden Zone Trading Agent - Enhanced with RAG and Pattern Learning
Integrates Fibonacci retracement and extension analysis with LangChain agents
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass

from langchain.agents import Tool
from langchain.schema import HumanMessage

from backend.app.rag.agents.base_agent import BaseTradingAgent, TradingSignal

logger = logging.getLogger(__name__)

@dataclass
class FibonacciLevel:
    """Represents a Fibonacci level with price and ratio"""
    ratio: float
    price: float
    level_type: str  # 'retracement' or 'extension'
    strength: float  # How strong/significant this level is
    distance_from_current: float
    
@dataclass
class GoldenZone:
    """Represents the Fibonacci Golden Zone (61.8% - 78.6%)"""
    upper_bound: float
    lower_bound: float
    center: float
    strength: float
    in_zone: bool
    zone_type: str  # 'retracement' or 'extension'

class FibonacciAgent(BaseTradingAgent):
    """
    Fibonacci Golden Zone trading agent with RAG capabilities
    Focuses on key Fibonacci levels and golden zone entries
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            agent_name="fibonacci_agent",
            strategy_type="fibonacci_golden_zone",
            **kwargs
        )
        
        # Fibonacci ratios
        self.retracement_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.extension_ratios = [1.0, 1.272, 1.382, 1.618, 2.618]
        
        # Golden Zone parameters (key Fibonacci levels)
        self.golden_zone_lower = 0.618  # 61.8%
        self.golden_zone_upper = 0.786  # 78.6%
        
        # Trading parameters
        self.min_swing_percentage = 0.02  # Minimum 2% swing for valid Fibonacci
        self.proximity_threshold = 0.005  # 0.5% proximity to Fibonacci level
        self.confluence_bonus = 0.2  # Confidence boost for multiple level confluence
        
        # Lookback periods for swing detection
        self.swing_lookback = 20
        self.trend_lookback = 50
        
        logger.info("FibonacciAgent initialized with golden zone focus")
    
    async def _create_tools(self) -> List[Tool]:
        """Create LangChain tools specific to Fibonacci analysis"""
        return [
            Tool(
                name="calculate_fibonacci_levels",
                func=self._calculate_fibonacci_levels,
                description="Calculate Fibonacci retracement and extension levels"
            ),
            Tool(
                name="identify_golden_zone",
                func=self._identify_golden_zone,
                description="Identify the Fibonacci Golden Zone (61.8% - 78.6%)"
            ),
            Tool(
                name="find_swing_points", 
                func=self._find_swing_points,
                description="Find significant swing highs and lows for Fibonacci analysis"
            ),
            Tool(
                name="check_level_confluence",
                func=self._check_level_confluence,
                description="Check for confluence between multiple Fibonacci levels"
            ),
            Tool(
                name="find_fibonacci_patterns",
                func=self._find_similar_fibonacci_setups,
                description="Find similar Fibonacci setups in historical data"
            )
        ]
    
    def _calculate_fibonacci_levels(self, price_data: str) -> str:
        """Calculate Fibonacci levels from price data"""
        try:
            if isinstance(price_data, str):
                prices = [float(x.strip()) for x in price_data.split(',')]
            else:
                prices = price_data
                
            if len(prices) < self.swing_lookback:
                return "Insufficient price data for Fibonacci analysis"
            
            # Find recent swing points
            swing_high, swing_low = self._find_recent_swing_points(prices)
            
            if swing_high is None or swing_low is None:
                return "No valid swing points found for Fibonacci analysis"
            
            # Calculate retracement levels
            fib_levels = self._calculate_fib_levels(swing_high, swing_low)
            
            result = f"Fibonacci levels calculated from swing high {swing_high:.4f} to low {swing_low:.4f}:\n"
            for level in fib_levels:
                result += f"  {level.ratio:.1%}: {level.price:.4f} ({level.level_type})\n"
                
            return result
            
        except Exception as e:
            return f"Error calculating Fibonacci levels: {str(e)}"
    
    def _identify_golden_zone(self, swing_high: str, swing_low: str, current_price: str) -> str:
        """Identify the Golden Zone and current position"""
        try:
            high = float(swing_high)
            low = float(swing_low)
            current = float(current_price)
            
            golden_zone = self._get_golden_zone(high, low)
            
            result = f"Golden Zone Analysis:\n"
            result += f"  Zone: {golden_zone.lower_bound:.4f} - {golden_zone.upper_bound:.4f}\n"
            result += f"  Center: {golden_zone.center:.4f}\n"
            result += f"  Current Price: {current:.4f}\n"
            result += f"  In Golden Zone: {golden_zone.in_zone}\n"
            result += f"  Zone Strength: {golden_zone.strength:.2f}\n"
            
            return result
            
        except Exception as e:
            return f"Error identifying Golden Zone: {str(e)}"
    
    def _find_swing_points(self, price_data: str) -> str:
        """Find significant swing highs and lows"""
        try:
            if isinstance(price_data, str):
                prices = [float(x.strip()) for x in price_data.split(',')]
            else:
                prices = price_data
                
            swings = self._detect_swing_points(prices)
            
            if not swings:
                return "No significant swing points detected"
            
            result = "Recent swing points:\n"
            for i, swing in enumerate(swings[-5:], 1):  # Last 5 swings
                result += f"  {i}. {swing['type']} at {swing['price']:.4f} (index {swing['index']})\n"
                
            return result
            
        except Exception as e:
            return f"Error finding swing points: {str(e)}"
    
    def _check_level_confluence(self, levels_data: str) -> str:
        """Check for confluence between multiple Fibonacci levels"""
        try:
            # Parse levels (assuming format: "level1,level2,level3,...")
            levels = [float(x.strip()) for x in levels_data.split(',')]
            
            confluences = self._find_level_confluences(levels)
            
            if not confluences:
                return "No significant level confluences found"
            
            result = f"Found {len(confluences)} confluence zones:\n"
            for i, confluence in enumerate(confluences, 1):
                result += f"  {i}. Zone around {confluence['price']:.4f} "
                result += f"({confluence['count']} levels, strength: {confluence['strength']:.2f})\n"
                
            return result
            
        except Exception as e:
            return f"Error checking level confluence: {str(e)}"
    
    async def _find_similar_fibonacci_setups(self, setup_description: str) -> str:
        """Find similar Fibonacci setups in historical data"""
        try:
            # Convert setup description to features for similarity search
            features = {
                'setup_description': setup_description,
                'strategy': 'fibonacci_golden_zone',
                'pattern_type': 'fibonacci_setup'
            }
            
            similar_patterns = await self.find_similar_patterns(features, similarity_threshold=0.7)
            
            if not similar_patterns:
                return "No similar Fibonacci setups found"
            
            result = f"Found {len(similar_patterns)} similar setups:\n"
            for i, pattern in enumerate(similar_patterns[:3], 1):
                result += f"  {i}. {pattern.pattern_name} "
                result += f"(similarity: {pattern.similarity:.2f}, success: {pattern.success_rate:.1%})\n"
                
            return result
            
        except Exception as e:
            return f"Error finding similar setups: {str(e)}"
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> TradingSignal:
        """
        Main analysis method - combines Fibonacci analysis with RAG enhancement
        """
        try:
            prices = market_data.get('prices', [])
            symbol = market_data.get('symbol', 'UNKNOWN')
            current_price = market_data.get('current_price', 0.0)
            
            if len(prices) < self.swing_lookback:
                return TradingSignal(
                    agent_type=self.agent_name,
                    strategy_name=self.strategy_type,
                    symbol=symbol,
                    action="HOLD",
                    confidence=0.0,
                    reasoning="Insufficient price data for Fibonacci analysis"
                )
            
            # Extract market features
            features = self._extract_market_features(market_data)
            
            # Perform Fibonacci analysis
            fibonacci_result = self._perform_fibonacci_analysis(prices, current_price)
            
            # Find similar historical patterns
            similar_patterns = await self.find_similar_patterns(features)
            
            # Get contextual information
            pattern_context = await self.get_pattern_context(features)
            
            # Generate enhanced signal with RAG context
            signal = self._generate_enhanced_signal(
                fibonacci_result,
                similar_patterns,
                pattern_context,
                symbol,
                current_price
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in Fibonacci agent analysis: {e}")
            return TradingSignal(
                agent_type=self.agent_name,
                strategy_name=self.strategy_type,
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                reasoning=f"Analysis error: {str(e)}"
            )
    
    def _extract_market_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Fibonacci-specific features from market data"""
        prices = market_data.get('prices', [])
        
        if len(prices) < self.swing_lookback:
            return {'error': 'insufficient_data'}
        
        # Find swing points
        swing_high, swing_low = self._find_recent_swing_points(prices)
        current_price = prices[-1]
        
        features = {}
        
        if swing_high and swing_low:
            # Calculate Fibonacci levels
            fib_levels = self._calculate_fib_levels(swing_high, swing_low)
            
            # Golden Zone analysis
            golden_zone = self._get_golden_zone(swing_high, swing_low)
            
            # Find closest Fibonacci levels
            closest_levels = self._find_closest_levels(current_price, fib_levels)
            
            features.update({
                'swing_high': swing_high,
                'swing_low': swing_low,
                'swing_range': swing_high - swing_low,
                'swing_range_pct': (swing_high - swing_low) / swing_low,
                'golden_zone_upper': golden_zone.upper_bound,
                'golden_zone_lower': golden_zone.lower_bound,
                'in_golden_zone': golden_zone.in_zone,
                'golden_zone_strength': golden_zone.strength,
                'closest_fib_level': closest_levels[0]['price'] if closest_levels else 0,
                'distance_to_closest_fib': closest_levels[0]['distance'] if closest_levels else 1.0,
                'confluence_count': len(self._find_level_confluences([level.price for level in fib_levels])),
            })
        
        # Overall market characteristics
        returns = np.diff(np.log(prices))
        features.update({
            'volatility': np.std(returns) * np.sqrt(252),
            'trend_strength': self._calculate_trend_strength(prices),
            'momentum': np.mean(returns[-5:]) if len(returns) >= 5 else 0.0,
            'price_position': self._calculate_price_position(prices)
        })
        
        return features
    
    def _perform_fibonacci_analysis(self, prices: List[float], current_price: float) -> Dict[str, Any]:
        """Perform core Fibonacci analysis"""
        try:
            # Find swing points
            swing_high, swing_low = self._find_recent_swing_points(prices)
            
            if not swing_high or not swing_low:
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'error': 'No valid swing points found'
                }
            
            # Calculate Fibonacci levels
            fib_levels = self._calculate_fib_levels(swing_high, swing_low)
            
            # Golden Zone analysis
            golden_zone = self._get_golden_zone(swing_high, swing_low)
            
            # Check for level proximity and confluence
            closest_levels = self._find_closest_levels(current_price, fib_levels)
            confluences = self._find_level_confluences([level.price for level in fib_levels])
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(prices)
            
            # Generate trading decision
            action, confidence = self._fibonacci_trading_decision(
                current_price, golden_zone, closest_levels, confluences, trend_direction
            )
            
            return {
                'swing_high': swing_high,
                'swing_low': swing_low,
                'golden_zone': golden_zone,
                'closest_levels': closest_levels,
                'confluences': confluences,
                'trend_direction': trend_direction,
                'action': action,
                'confidence': confidence,
                'fib_levels_count': len(fib_levels),
                'in_golden_zone': golden_zone.in_zone
            }
            
        except Exception as e:
            logger.error(f"Error in Fibonacci analysis: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _find_recent_swing_points(self, prices: List[float]) -> Tuple[Optional[float], Optional[float]]:
        """Find the most recent significant swing high and low"""
        if len(prices) < self.swing_lookback:
            return None, None
        
        # Look for swing points in recent data
        lookback_prices = prices[-self.swing_lookback:]
        
        # Simple swing detection using local maxima/minima
        swing_high = max(lookback_prices)
        swing_low = min(lookback_prices)
        
        # Validate swing significance
        swing_range = (swing_high - swing_low) / swing_low
        if swing_range < self.min_swing_percentage:
            return None, None
        
        return swing_high, swing_low
    
    def _calculate_fib_levels(self, swing_high: float, swing_low: float) -> List[FibonacciLevel]:
        """Calculate Fibonacci retracement and extension levels"""
        levels = []
        swing_range = swing_high - swing_low
        
        # Retracement levels (from high to low)
        for ratio in self.retracement_ratios:
            price = swing_high - (swing_range * ratio)
            levels.append(FibonacciLevel(
                ratio=ratio,
                price=price,
                level_type='retracement',
                strength=self._calculate_level_strength(ratio),
                distance_from_current=0  # Will be updated later
            ))
        
        # Extension levels (beyond the swing range)
        for ratio in self.extension_ratios:
            price = swing_low - (swing_range * (ratio - 1))  # Below swing low
            levels.append(FibonacciLevel(
                ratio=ratio,
                price=price,
                level_type='extension',
                strength=self._calculate_level_strength(ratio, is_extension=True),
                distance_from_current=0  # Will be updated later
            ))
        
        return levels
    
    def _get_golden_zone(self, swing_high: float, swing_low: float) -> GoldenZone:
        """Get the Fibonacci Golden Zone (61.8% - 78.6% retracement)"""
        swing_range = swing_high - swing_low
        
        upper_bound = swing_high - (swing_range * self.golden_zone_lower)  # 61.8%
        lower_bound = swing_high - (swing_range * self.golden_zone_upper)  # 78.6%
        center = (upper_bound + lower_bound) / 2
        
        # Calculate zone strength based on range and market conditions
        zone_strength = min(1.0, swing_range / swing_low * 10)  # Normalize strength
        
        return GoldenZone(
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            center=center,
            strength=zone_strength,
            in_zone=False,  # Will be updated with current price
            zone_type='retracement'
        )
    
    def _calculate_level_strength(self, ratio: float, is_extension: bool = False) -> float:
        """Calculate the strength/importance of a Fibonacci level"""
        # Key Fibonacci levels have higher strength
        key_levels = {0.382: 0.8, 0.5: 0.7, 0.618: 1.0, 0.786: 0.9}  # Golden ratio is strongest
        extension_levels = {1.0: 0.6, 1.272: 0.7, 1.618: 1.0, 2.618: 0.8}
        
        if is_extension:
            return extension_levels.get(ratio, 0.5)
        else:
            return key_levels.get(ratio, 0.5)
    
    def _find_closest_levels(self, current_price: float, fib_levels: List[FibonacciLevel]) -> List[Dict[str, Any]]:
        """Find the closest Fibonacci levels to current price"""
        level_distances = []
        
        for level in fib_levels:
            distance = abs(current_price - level.price) / current_price
            level_distances.append({
                'level': level,
                'price': level.price,
                'ratio': level.ratio,
                'distance': distance,
                'type': level.level_type,
                'strength': level.strength
            })
        
        # Sort by distance and return closest levels
        level_distances.sort(key=lambda x: x['distance'])
        return level_distances[:3]  # Return top 3 closest levels
    
    def _find_level_confluences(self, levels: List[float]) -> List[Dict[str, Any]]:
        """Find confluence zones where multiple Fibonacci levels cluster"""
        confluences = []
        tolerance = self.proximity_threshold
        
        for i, level1 in enumerate(levels):
            cluster = [level1]
            for j, level2 in enumerate(levels[i+1:], i+1):
                if abs(level1 - level2) / level1 <= tolerance:
                    cluster.append(level2)
            
            if len(cluster) >= 2:  # At least 2 levels in confluence
                avg_price = sum(cluster) / len(cluster)
                strength = len(cluster) * 0.3  # Strength based on number of confluent levels
                
                confluences.append({
                    'price': avg_price,
                    'count': len(cluster),
                    'strength': min(1.0, strength),
                    'levels': cluster
                })
        
        return confluences
    
    def _determine_trend_direction(self, prices: List[float]) -> str:
        """Determine overall trend direction"""
        if len(prices) < self.trend_lookback:
            return 'NEUTRAL'
        
        recent_prices = prices[-self.trend_lookback:]
        
        # Simple trend using linear regression slope
        x = np.arange(len(recent_prices))
        slope = np.polyfit(x, recent_prices, 1)[0]
        
        # Normalize slope by price level
        normalized_slope = slope / np.mean(recent_prices)
        
        if normalized_slope > 0.001:  # 0.1% per period
            return 'BULLISH'
        elif normalized_slope < -0.001:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _fibonacci_trading_decision(
        self, 
        current_price: float, 
        golden_zone: GoldenZone, 
        closest_levels: List[Dict[str, Any]], 
        confluences: List[Dict[str, Any]], 
        trend_direction: str
    ) -> Tuple[str, float]:
        """Generate trading decision based on Fibonacci analysis"""
        
        base_confidence = 0.5
        action = "HOLD"
        
        # Check if in Golden Zone
        golden_zone.in_zone = golden_zone.lower_bound <= current_price <= golden_zone.upper_bound
        
        if golden_zone.in_zone:
            if trend_direction == 'BULLISH':
                action = "BUY"
                base_confidence = 0.7
            elif trend_direction == 'BEARISH':
                action = "SELL"
                base_confidence = 0.7
        
        # Proximity to key Fibonacci levels
        if closest_levels:
            closest = closest_levels[0]
            if closest['distance'] < self.proximity_threshold:
                # Near a key level, adjust confidence
                level_strength_bonus = closest['strength'] * 0.2
                base_confidence += level_strength_bonus
                
                # Direction based on level type and trend
                if closest['type'] == 'retracement' and trend_direction == 'BULLISH':
                    if action == "HOLD":
                        action = "BUY"
                elif closest['type'] == 'extension' and trend_direction == 'BEARISH':
                    if action == "HOLD":
                        action = "SELL"
        
        # Confluence bonus
        if confluences:
            for confluence in confluences:
                distance_to_confluence = abs(current_price - confluence['price']) / current_price
                if distance_to_confluence < self.proximity_threshold:
                    base_confidence += confluence['strength'] * self.confluence_bonus
        
        # Cap confidence at 1.0
        final_confidence = min(1.0, base_confidence)
        
        return action, final_confidence
    
    def _generate_enhanced_signal(
        self,
        fibonacci_result: Dict[str, Any],
        similar_patterns: List,
        pattern_context: str,
        symbol: str,
        current_price: float
    ) -> TradingSignal:
        """Generate enhanced trading signal with RAG context"""
        
        action = fibonacci_result.get('action', 'HOLD')
        base_confidence = fibonacci_result.get('confidence', 0.0)
        
        # Adjust confidence based on similar patterns
        confidence_adjustment = 0.0
        if similar_patterns:
            avg_success_rate = np.mean([p.success_rate for p in similar_patterns])
            confidence_adjustment = (avg_success_rate - 0.5) * 0.3
        
        final_confidence = min(1.0, max(0.0, base_confidence + confidence_adjustment))
        
        # Generate reasoning
        reasoning_parts = []
        
        if fibonacci_result.get('in_golden_zone'):
            reasoning_parts.append("Price is in Fibonacci Golden Zone (61.8%-78.6%)")
        
        closest_levels = fibonacci_result.get('closest_levels', [])
        if closest_levels:
            closest = closest_levels[0]
            reasoning_parts.append(f"Near {closest['ratio']:.1%} Fibonacci level at {closest['price']:.4f}")
        
        confluences = fibonacci_result.get('confluences', [])
        if confluences:
            reasoning_parts.append(f"Found {len(confluences)} confluence zones")
        
        trend = fibonacci_result.get('trend_direction', 'NEUTRAL')
        reasoning_parts.append(f"Trend: {trend}")
        
        if similar_patterns:
            reasoning_parts.append(f"Historical patterns: {len(similar_patterns)} similar cases "
                                 f"with {avg_success_rate:.1%} success rate")
        
        # Calculate position sizing and risk levels
        swing_high = fibonacci_result.get('swing_high')
        swing_low = fibonacci_result.get('swing_low')
        
        stop_loss = None
        take_profit = None
        
        if action == "BUY" and swing_low:
            stop_loss = swing_low * 0.99  # 1% below swing low
            if swing_high:
                take_profit = swing_high * 1.01  # 1% above swing high
        elif action == "SELL" and swing_high:
            stop_loss = swing_high * 1.01  # 1% above swing high
            if swing_low:
                take_profit = swing_low * 0.99  # 1% below swing low
        
        metadata = {
            'in_golden_zone': fibonacci_result.get('in_golden_zone', False),
            'closest_fib_ratio': closest_levels[0]['ratio'] if closest_levels else None,
            'confluence_count': len(confluences),
            'trend_direction': trend,
            'swing_range_pct': ((swing_high - swing_low) / swing_low) if swing_high and swing_low else 0,
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
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata
        )
    
    def _detect_swing_points(self, prices: List[float]) -> List[Dict[str, Any]]:
        """Detect swing highs and lows in price data"""
        swings = []
        
        if len(prices) < 5:
            return swings
        
        for i in range(2, len(prices) - 2):
            # Check for swing high
            if (prices[i] > prices[i-1] and prices[i] > prices[i-2] and 
                prices[i] > prices[i+1] and prices[i] > prices[i+2]):
                swings.append({
                    'type': 'HIGH',
                    'price': prices[i],
                    'index': i
                })
            
            # Check for swing low
            elif (prices[i] < prices[i-1] and prices[i] < prices[i-2] and 
                  prices[i] < prices[i+1] and prices[i] < prices[i+2]):
                swings.append({
                    'type': 'LOW',
                    'price': prices[i],
                    'index': i
                })
        
        return swings
    
    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength indicator"""
        if len(prices) < 20:
            return 0.0
        
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        normalized_slope = slope / np.mean(prices)
        
        return np.tanh(normalized_slope * 100)
    
    def _calculate_price_position(self, prices: List[float]) -> float:
        """Calculate where current price sits in recent range"""
        if len(prices) < 20:
            return 0.5
        
        recent_prices = prices[-20:]
        current_price = prices[-1]
        price_range = max(recent_prices) - min(recent_prices)
        
        if price_range == 0:
            return 0.5
        
        return (current_price - min(recent_prices)) / price_range

# Test function
async def test_fibonacci_agent():
    """Test the Fibonacci Trading Agent"""
    print("🧪 Testing Fibonacci Trading Agent...")
    
    agent = FibonacciAgent()
    await agent.initialize()
    
    # Test with sample market data showing a clear swing
    sample_data = {
        'symbol': 'TEST',
        'current_price': 105.0,
        'prices': [100, 102, 98, 95, 92, 89, 91, 94, 97, 100, 103, 106, 109, 107, 105, 103, 101, 102, 104, 105],
        'volumes': [1000] * 20
    }
    
    # Generate signal
    signal = await agent.analyze_market(sample_data)
    
    print(f"✅ Generated signal:")
    print(f"   Action: {signal.action}")
    print(f"   Confidence: {signal.confidence:.2f}")
    print(f"   Reasoning: {signal.reasoning}")
    print(f"   Entry Price: {signal.entry_price}")
    print(f"   Stop Loss: {signal.stop_loss}")
    print(f"   Take Profit: {signal.take_profit}")
    print(f"   Metadata: {signal.metadata}")
    
    # Test health check
    health = await agent.health_check()
    print(f"✅ Health check: {health['is_initialized']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fibonacci_agent())