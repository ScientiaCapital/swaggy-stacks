"""
Elliott Wave Trading Agent - Enhanced with RAG and Pattern Recognition
Converts Elliott Wave analysis into an intelligent agent
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from scipy import signal as scipy_signal
from scipy.stats import linregress

from langchain.agents import Tool
from langchain.schema import HumanMessage

from backend.app.rag.agents.base_agent import BaseTradingAgent, TradingSignal

logger = logging.getLogger(__name__)

class ElliottWaveAgent(BaseTradingAgent):
    """
    Elliott Wave theory agent with advanced pattern recognition
    Identifies wave patterns and projects targets using Fibonacci ratios
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            agent_name="elliott_wave_agent",
            strategy_type="elliott_wave",
            **kwargs
        )
        
        # Elliott Wave specific parameters
        self.wave_patterns = {
            'impulse': [1, 2, 3, 4, 5],
            'corrective_abc': ['A', 'B', 'C'],
            'corrective_wxy': ['W', 'X', 'Y'],
            'triangle': ['A', 'B', 'C', 'D', 'E']
        }
        
        # Fibonacci ratios for wave projections
        self.fibonacci_ratios = {
            'retracement': [0.236, 0.382, 0.5, 0.618, 0.786],
            'extension': [1.0, 1.272, 1.382, 1.618, 2.618],
            'projection': [0.618, 1.0, 1.382, 1.618]
        }
        
        # Wave rules and guidelines
        self.wave_rules = {
            'wave2_max_retrace': 0.99,  # Wave 2 cannot retrace more than 99% of Wave 1
            'wave4_max_retrace': 0.618,  # Wave 4 typically retraces less than Wave 1
            'wave3_min_extension': 1.0,   # Wave 3 must be longer than Wave 1
            'alternation_factor': 0.3     # Waves 2 and 4 should alternate in character
        }
        
        self.min_wave_points = 5  # Minimum points needed to identify a pattern
        self.confidence_threshold = 0.65
        
        logger.info("ElliottWaveAgent initialized")
    
    async def _create_tools(self) -> List[Tool]:
        """Create LangChain tools for Elliott Wave analysis"""
        return [
            Tool(
                name="identify_wave_pattern",
                func=self._identify_wave_pattern,
                description="Identify current Elliott Wave pattern in price data"
            ),
            Tool(
                name="calculate_wave_targets",
                func=self._calculate_wave_targets,
                description="Calculate Fibonacci-based wave targets"
            ),
            Tool(
                name="validate_wave_count",
                func=self._validate_wave_count,
                description="Validate wave count against Elliott Wave rules"
            ),
            Tool(
                name="get_wave_personality",
                func=self._get_wave_personality,
                description="Get expected characteristics of current wave"
            ),
            Tool(
                name="find_similar_wave_patterns",
                func=self._find_similar_wave_patterns,
                description="Find similar Elliott Wave patterns in history"
            )
        ]
    
    def _identify_wave_pattern(self, price_data: str) -> str:
        """Identify Elliott Wave pattern from price data"""
        try:
            prices = [float(x.strip()) for x in price_data.split(',')]
            if len(prices) < self.min_wave_points:
                return f"Need at least {self.min_wave_points} price points for wave analysis"
            
            pattern_info = self._detect_wave_pattern(prices)
            return f"Wave pattern: {pattern_info['pattern_type']}, Current wave: {pattern_info['current_wave']}, Progress: {pattern_info['completion']:.1%}"
            
        except Exception as e:
            return f"Error identifying wave pattern: {str(e)}"
    
    def _calculate_wave_targets(self, current_wave: str, wave_data: str) -> str:
        """Calculate Fibonacci-based wave targets"""
        try:
            # Parse wave data (high, low points)
            points = [float(x.strip()) for x in wave_data.split(',')]
            wave_num = int(current_wave) if current_wave.isdigit() else 3
            
            targets = self._compute_fibonacci_targets(points, wave_num)
            
            result = f"Wave {wave_num} targets:\n"
            for level, target in targets.items():
                result += f"  {level}: {target:.2f}\n"
                
            return result
            
        except Exception as e:
            return f"Error calculating wave targets: {str(e)}"
    
    def _validate_wave_count(self, wave_sequence: str) -> str:
        """Validate wave count against Elliott Wave rules"""
        try:
            waves = wave_sequence.split(',')
            violations = self._check_wave_rules(waves)
            
            if not violations:
                return "Wave count is valid according to Elliott Wave rules"
            else:
                return f"Wave rule violations: {', '.join(violations)}"
                
        except Exception as e:
            return f"Error validating wave count: {str(e)}"
    
    def _get_wave_personality(self, wave_number: str) -> str:
        """Get expected characteristics of the specified wave"""
        try:
            wave_num = int(wave_number) if wave_number.isdigit() else 1
            personality = self._wave_personalities.get(wave_num, "Unknown wave characteristics")
            return f"Wave {wave_num} personality: {personality}"
            
        except Exception as e:
            return f"Error getting wave personality: {str(e)}"
    
    async def _find_similar_wave_patterns(self, pattern_description: str) -> str:
        """Find similar Elliott Wave patterns"""
        try:
            features = {
                'pattern_description': pattern_description,
                'wave_type': 'elliott_wave',
                'strategy': 'wave_pattern_matching'
            }
            
            similar_patterns = await self.find_similar_patterns(features, similarity_threshold=0.75)
            
            if not similar_patterns:
                return "No similar Elliott Wave patterns found"
            
            result = f"Found {len(similar_patterns)} similar wave patterns:\n"
            for i, pattern in enumerate(similar_patterns[:3], 1):
                result += f"  {i}. {pattern.pattern_name} (similarity: {pattern.similarity:.2f})\n"
                
            return result
            
        except Exception as e:
            return f"Error finding similar patterns: {str(e)}"
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> TradingSignal:
        """Main Elliott Wave analysis method"""
        try:
            prices = market_data.get('prices', [])
            symbol = market_data.get('symbol', 'UNKNOWN')
            current_price = market_data.get('current_price', 0.0)
            
            if len(prices) < self.min_wave_points * 2:
                return TradingSignal(
                    agent_type=self.agent_name,
                    strategy_name=self.strategy_type,
                    symbol=symbol,
                    action="HOLD",
                    confidence=0.0,
                    reasoning="Insufficient price data for Elliott Wave analysis"
                )
            
            # Extract market features
            features = self._extract_market_features(market_data)
            
            # Perform Elliott Wave analysis
            wave_analysis = self._perform_wave_analysis(prices)
            
            # Find similar historical patterns
            similar_patterns = await self.find_similar_patterns(features)
            
            # Generate enhanced signal
            signal = self._generate_wave_signal(
                wave_analysis,
                similar_patterns,
                symbol,
                current_price
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in Elliott Wave analysis: {e}")
            return TradingSignal(
                agent_type=self.agent_name,
                strategy_name=self.strategy_type,
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                reasoning=f"Elliott Wave analysis error: {str(e)}"
            )
    
    def _extract_market_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Elliott Wave specific features"""
        prices = market_data.get('prices', [])
        
        if len(prices) < 10:
            return {'error': 'insufficient_data'}
        
        # Find pivot points (highs and lows)
        pivots = self._find_pivot_points(prices)
        
        # Identify current wave pattern
        pattern_info = self._detect_wave_pattern(prices)
        
        # Calculate wave relationships
        wave_ratios = self._calculate_wave_ratios(pivots)
        
        features = {
            'pivot_count': len(pivots),
            'wave_pattern': pattern_info['pattern_type'],
            'current_wave': pattern_info['current_wave'],
            'pattern_completion': pattern_info['completion'],
            'wave_ratios': wave_ratios,
            'trend_direction': self._determine_trend_direction(prices),
            'wave_quality_score': pattern_info['quality_score']
        }
        
        return features
    
    def _perform_wave_analysis(self, prices: List[float]) -> Dict[str, Any]:
        """Perform comprehensive Elliott Wave analysis"""
        try:
            # Find significant pivot points
            pivots = self._find_pivot_points(prices)
            
            if len(pivots) < 3:
                return {
                    'pattern_type': 'insufficient_data',
                    'confidence': 0.0,
                    'action': 'HOLD'
                }
            
            # Detect wave pattern
            pattern_info = self._detect_wave_pattern(prices)
            
            # Calculate targets for next wave
            targets = self._compute_fibonacci_targets(pivots, pattern_info['current_wave'])
            
            # Validate against Elliott Wave rules
            rule_validation = self._validate_elliott_rules(pivots, pattern_info)
            
            # Determine trading action
            action, confidence = self._determine_wave_action(pattern_info, rule_validation, targets)
            
            return {
                'pattern_type': pattern_info['pattern_type'],
                'current_wave': pattern_info['current_wave'],
                'completion': pattern_info['completion'],
                'quality_score': pattern_info['quality_score'],
                'targets': targets,
                'rule_validation': rule_validation,
                'action': action,
                'confidence': confidence,
                'pivot_points': pivots
            }
            
        except Exception as e:
            logger.error(f"Error in wave analysis: {e}")
            return {
                'pattern_type': 'error',
                'confidence': 0.0,
                'action': 'HOLD',
                'error': str(e)
            }
    
    def _find_pivot_points(self, prices: List[float], window: int = 5) -> List[Dict]:
        """Find significant pivot highs and lows"""
        pivots = []
        prices_array = np.array(prices)
        
        # Find local maxima and minima
        max_indices = scipy_signal.argrelextrema(prices_array, np.greater, order=window)[0]
        min_indices = scipy_signal.argrelextrema(prices_array, np.less, order=window)[0]
        
        # Combine and sort by index
        all_pivots = []
        for idx in max_indices:
            all_pivots.append({'index': idx, 'price': prices[idx], 'type': 'high'})
        
        for idx in min_indices:
            all_pivots.append({'index': idx, 'price': prices[idx], 'type': 'low'})
        
        # Sort by index
        all_pivots.sort(key=lambda x: x['index'])
        
        # Filter out insignificant pivots (less than 2% price move)
        filtered_pivots = []
        for i, pivot in enumerate(all_pivots):
            if i == 0:
                filtered_pivots.append(pivot)
                continue
                
            prev_pivot = filtered_pivots[-1]
            price_change = abs(pivot['price'] - prev_pivot['price']) / prev_pivot['price']
            
            if price_change > 0.02:  # 2% minimum price change
                filtered_pivots.append(pivot)
        
        return filtered_pivots[-20:]  # Keep last 20 significant pivots
    
    def _detect_wave_pattern(self, prices: List[float]) -> Dict[str, Any]:
        """Detect Elliott Wave pattern in price data"""
        pivots = self._find_pivot_points(prices)
        
        if len(pivots) < 5:
            return {
                'pattern_type': 'insufficient_pivots',
                'current_wave': 1,
                'completion': 0.0,
                'quality_score': 0.0
            }
        
        # Analyze last 9 pivots for complete 5-3 wave pattern
        recent_pivots = pivots[-9:] if len(pivots) >= 9 else pivots
        
        # Determine if we're in impulse or corrective phase
        pattern_type, quality_score = self._classify_wave_pattern(recent_pivots)
        
        # Determine current wave number
        current_wave = self._identify_current_wave(recent_pivots, pattern_type)
        
        # Calculate completion percentage
        completion = self._calculate_pattern_completion(recent_pivots, pattern_type)
        
        return {
            'pattern_type': pattern_type,
            'current_wave': current_wave,
            'completion': completion,
            'quality_score': quality_score
        }
    
    def _classify_wave_pattern(self, pivots: List[Dict]) -> Tuple[str, float]:
        """Classify the wave pattern type and quality"""
        if len(pivots) < 5:
            return 'incomplete', 0.0
        
        # Check for impulse pattern (5 waves)
        impulse_score = self._score_impulse_pattern(pivots)
        
        # Check for corrective pattern (3 waves)
        corrective_score = self._score_corrective_pattern(pivots)
        
        if impulse_score > corrective_score and impulse_score > 0.6:
            return 'impulse', impulse_score
        elif corrective_score > 0.6:
            return 'corrective', corrective_score
        else:
            return 'complex', max(impulse_score, corrective_score)
    
    def _score_impulse_pattern(self, pivots: List[Dict]) -> float:
        """Score how well pivots match impulse pattern"""
        if len(pivots) < 5:
            return 0.0
        
        score = 0.0
        total_checks = 0
        
        # Take last 5 significant moves for impulse analysis
        last_5_pivots = pivots[-5:]
        
        # Check alternation (waves should alternate high-low-high-low-high for uptrend)
        alternation_correct = True
        for i in range(1, len(last_5_pivots)):
            if i % 2 == 1:  # Odd indices should be opposite of previous
                if last_5_pivots[i]['type'] == last_5_pivots[i-1]['type']:
                    alternation_correct = False
                    break
        
        if alternation_correct:
            score += 0.4
        total_checks += 1
        
        # Check wave 3 extension (should be longest)
        if len(last_5_pivots) >= 5:
            wave_lengths = []
            for i in range(1, len(last_5_pivots)):
                length = abs(last_5_pivots[i]['price'] - last_5_pivots[i-1]['price'])
                wave_lengths.append(length)
            
            if len(wave_lengths) >= 3 and wave_lengths[2] == max(wave_lengths):
                score += 0.3  # Wave 3 is longest
                total_checks += 1
        
        # Check Fibonacci relationships
        fib_score = self._check_fibonacci_relationships(last_5_pivots)
        score += fib_score * 0.3
        total_checks += 1
        
        return score / max(1, total_checks) if total_checks > 0 else 0.0
    
    def _score_corrective_pattern(self, pivots: List[Dict]) -> float:
        """Score how well pivots match corrective pattern"""
        if len(pivots) < 3:
            return 0.0
        
        # Simple 3-wave corrective pattern check
        last_3_pivots = pivots[-3:]
        
        score = 0.0
        
        # Check for A-B-C structure
        if len(last_3_pivots) == 3:
            # A and C should be in same direction, B opposite
            a_to_b = last_3_pivots[1]['price'] - last_3_pivots[0]['price']
            b_to_c = last_3_pivots[2]['price'] - last_3_pivots[1]['price']
            
            # A and C should be similar in magnitude
            if abs(abs(a_to_b) - abs(b_to_c)) / max(abs(a_to_b), abs(b_to_c)) < 0.5:
                score += 0.5
            
            # B should be smaller retracement
            if abs(a_to_b) > abs(b_to_c):
                score += 0.3
        
        return min(1.0, score)
    
    def _identify_current_wave(self, pivots: List[Dict], pattern_type: str) -> int:
        """Identify which wave we're currently in"""
        if pattern_type == 'impulse':
            # For impulse, we have waves 1-5
            wave_count = min(len(pivots), 5)
            return wave_count
        elif pattern_type == 'corrective':
            # For corrective, we have waves A, B, C (return as 1, 2, 3)
            wave_count = min(len(pivots), 3)
            return wave_count
        else:
            return 1
    
    def _calculate_pattern_completion(self, pivots: List[Dict], pattern_type: str) -> float:
        """Calculate how complete the current pattern is"""
        if pattern_type == 'impulse':
            expected_waves = 5
            current_waves = min(len(pivots), 5)
            return current_waves / expected_waves
        elif pattern_type == 'corrective':
            expected_waves = 3
            current_waves = min(len(pivots), 3)
            return current_waves / expected_waves
        else:
            return 0.5  # Unknown pattern, assume 50% complete
    
    def _compute_fibonacci_targets(self, pivots: List[Dict], current_wave: int) -> Dict[str, float]:
        """Compute Fibonacci-based price targets"""
        if len(pivots) < 2:
            return {}
        
        targets = {}
        
        # Get the last significant swing
        last_high = max([p['price'] for p in pivots if p['type'] == 'high'])
        last_low = min([p['price'] for p in pivots if p['type'] == 'low'])
        swing_range = last_high - last_low
        
        # Calculate retracement levels
        for ratio in self.fibonacci_ratios['retracement']:
            targets[f'retracement_{ratio*100:.1f}%'] = last_high - (swing_range * ratio)
        
        # Calculate extension levels
        for ratio in self.fibonacci_ratios['extension']:
            targets[f'extension_{ratio*100:.1f}%'] = last_high + (swing_range * (ratio - 1))
        
        return targets
    
    def _validate_elliott_rules(self, pivots: List[Dict], pattern_info: Dict) -> Dict[str, bool]:
        """Validate against core Elliott Wave rules"""
        validation = {
            'wave_2_rule': True,  # Wave 2 never retraces more than 100% of Wave 1
            'wave_3_rule': True,  # Wave 3 is never the shortest
            'wave_4_rule': True,  # Wave 4 never enters Wave 1 price territory
            'alternation_rule': True  # Waves 2 and 4 alternate in character
        }
        
        if len(pivots) >= 5 and pattern_info['pattern_type'] == 'impulse':
            # Implement specific rule checks here
            # This is a simplified version - full implementation would be more complex
            pass
        
        return validation
    
    def _determine_wave_action(self, pattern_info: Dict, rule_validation: Dict, targets: Dict) -> Tuple[str, float]:
        """Determine trading action based on wave analysis"""
        
        # Base confidence on pattern quality and rule validation
        base_confidence = pattern_info['quality_score']
        
        # Reduce confidence if rules are violated
        rule_violations = sum(1 for valid in rule_validation.values() if not valid)
        confidence_penalty = rule_violations * 0.15
        confidence = max(0.0, base_confidence - confidence_penalty)
        
        if confidence < self.confidence_threshold:
            return "HOLD", confidence
        
        # Determine action based on wave type and position
        pattern_type = pattern_info['pattern_type']
        current_wave = pattern_info['current_wave']
        completion = pattern_info['completion']
        
        if pattern_type == 'impulse':
            if current_wave in [1, 3, 5] and completion < 0.8:  # Motive waves
                return "BUY", confidence
            elif current_wave in [2, 4] and completion < 0.8:  # Corrective waves
                return "SELL", confidence * 0.8  # Lower confidence for counter-trend
            else:
                return "HOLD", confidence * 0.5  # Pattern near completion
        
        elif pattern_type == 'corrective':
            if completion > 0.8:  # Correction nearly complete
                return "BUY", confidence * 0.9  # Prepare for next impulse
            else:
                return "HOLD", confidence * 0.6  # Wait for correction to complete
        
        return "HOLD", confidence * 0.5
    
    def _generate_wave_signal(
        self,
        wave_analysis: Dict[str, Any],
        similar_patterns: List,
        symbol: str,
        current_price: float
    ) -> TradingSignal:
        """Generate trading signal based on Elliott Wave analysis"""
        
        action = wave_analysis.get('action', 'HOLD')
        confidence = wave_analysis.get('confidence', 0.0)
        
        # Adjust confidence based on similar patterns
        if similar_patterns:
            avg_success_rate = np.mean([p.success_rate for p in similar_patterns])
            confidence_adjustment = (avg_success_rate - 0.5) * 0.2
            confidence = min(1.0, max(0.0, confidence + confidence_adjustment))
        
        # Build reasoning
        reasoning_parts = [
            f"Elliott Wave: {wave_analysis.get('pattern_type', 'unknown')} pattern, "
            f"Wave {wave_analysis.get('current_wave', '?')}, "
            f"{wave_analysis.get('completion', 0)*100:.0f}% complete"
        ]
        
        if wave_analysis.get('quality_score', 0) > 0:
            reasoning_parts.append(f"Pattern quality: {wave_analysis['quality_score']:.1%}")
        
        if similar_patterns:
            reasoning_parts.append(f"Found {len(similar_patterns)} similar wave patterns")
        
        # Calculate targets
        targets = wave_analysis.get('targets', {})
        entry_price = current_price if action != "HOLD" else None
        
        metadata = {
            'pattern_type': wave_analysis.get('pattern_type'),
            'current_wave': wave_analysis.get('current_wave'),
            'completion': wave_analysis.get('completion'),
            'quality_score': wave_analysis.get('quality_score'),
            'targets': targets,
            'rule_validation': wave_analysis.get('rule_validation', {}),
            'similar_patterns_count': len(similar_patterns)
        }
        
        return TradingSignal(
            agent_type=self.agent_name,
            strategy_name=self.strategy_type,
            symbol=symbol,
            action=action,
            confidence=confidence,
            reasoning=" | ".join(reasoning_parts),
            entry_price=entry_price,
            metadata=metadata
        )
    
    def _check_fibonacci_relationships(self, pivots: List[Dict]) -> float:
        """Check for Fibonacci relationships between waves"""
        if len(pivots) < 4:
            return 0.0
        
        score = 0.0
        checks = 0
        
        # Check common Fibonacci relationships
        for i in range(len(pivots) - 3):
            wave1_length = abs(pivots[i+1]['price'] - pivots[i]['price'])
            wave2_length = abs(pivots[i+2]['price'] - pivots[i+1]['price'])
            
            if wave1_length > 0:
                ratio = wave2_length / wave1_length
                
                # Check if ratio matches common Fibonacci ratios
                for fib_ratio in [0.382, 0.5, 0.618, 1.0, 1.272, 1.618]:
                    if abs(ratio - fib_ratio) / fib_ratio < 0.1:  # Within 10%
                        score += 1.0
                        break
                
                checks += 1
        
        return score / max(1, checks)
    
    def _determine_trend_direction(self, prices: List[float]) -> str:
        """Determine overall trend direction"""
        if len(prices) < 10:
            return 'unknown'
        
        # Use linear regression on recent prices
        recent_prices = prices[-20:] if len(prices) >= 20 else prices
        x = np.arange(len(recent_prices))
        slope, _, r_value, _, _ = linregress(x, recent_prices)
        
        # Strong trend if R-squared > 0.6
        if r_value ** 2 > 0.6:
            if slope > 0:
                return 'uptrend'
            else:
                return 'downtrend'
        else:
            return 'sideways'
    
    # Wave personalities for educational purposes
    @property
    def _wave_personalities(self) -> Dict[int, str]:
        return {
            1: "Initial impulse, often with skepticism from market participants",
            2: "Corrective, can retrace 50-78.6% of Wave 1, often sharp",
            3: "Strongest impulse, usually the longest, broad market participation",
            4: "Corrective, usually a sideways consolidation, alternates with Wave 2",
            5: "Final impulse, often with divergences and less conviction",
        }

# Test function
async def test_elliott_wave_agent():
    """Test the Elliott Wave Agent"""
    print("🧪 Testing Elliott Wave Agent...")
    
    agent = ElliottWaveAgent()
    await agent.initialize()
    
    # Test with sample price data showing a potential wave pattern
    sample_data = {
        'symbol': 'TEST',
        'current_price': 110.0,
        'prices': [
            100, 105, 102, 108, 106, 115, 110, 120, 118, 125,  # Potential waves 1-3
            120, 130, 125, 135, 130, 140, 135, 145, 140, 150   # Continue pattern
        ]
    }
    
    # Generate signal
    signal = await agent.analyze_market(sample_data)
    
    print(f"✅ Elliott Wave signal generated:")
    print(f"   Action: {signal.action}")
    print(f"   Confidence: {signal.confidence:.2f}")
    print(f"   Reasoning: {signal.reasoning}")
    print(f"   Pattern: {signal.metadata.get('pattern_type')}")
    print(f"   Current Wave: {signal.metadata.get('current_wave')}")
    
    # Test health check
    health = await agent.health_check()
    print(f"✅ Health check: {health['is_initialized']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_elliott_wave_agent())