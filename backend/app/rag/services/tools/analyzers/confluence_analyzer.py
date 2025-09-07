"""
Confluence analyzer for combining multiple technical analysis methods
"""

from typing import List, Dict, Any, Optional
from .base_analyzer import BaseAnalyzer
from .fibonacci_analyzer import FibonacciAnalyzer
from .elliott_wave_analyzer import ElliottWaveAnalyzer
from .wyckoff_analyzer import WyckoffAnalyzer


class ConfluenceAnalyzer(BaseAnalyzer):
    """Analyzer that combines multiple technical analysis methods for high-probability setups"""
    
    def __init__(self):
        super().__init__()
        self.fibonacci_analyzer = FibonacciAnalyzer()
        self.elliott_analyzer = ElliottWaveAnalyzer()
        self.wyckoff_analyzer = WyckoffAnalyzer()
        
        # Confidence weights for different analysis methods
        self.method_weights = {
            'fibonacci': 0.4,
            'elliott_wave': 0.3,
            'wyckoff': 0.3
        }
    
    async def analyze(self, data: List[Dict], lookback_period: int, min_strength: float, **kwargs) -> Dict[str, Any]:
        """Analyze confluence patterns combining multiple methods"""
        try:
            if not self._validate_data(data, min_length=50):
                return {"patterns": [], "error": "Insufficient data for confluence analysis"}
            
            symbol = kwargs.get('symbol', 'UNKNOWN')
            
            # Run individual analyses
            fib_result = await self.fibonacci_analyzer.analyze(data, lookback_period, min_strength)
            elliott_result = await self.elliott_analyzer.analyze(data, lookback_period, min_strength)
            wyckoff_result = await self.wyckoff_analyzer.analyze(data, lookback_period, min_strength)
            
            # Extract patterns from each method
            fib_patterns = fib_result.get('patterns', [])
            elliott_patterns = elliott_result.get('patterns', [])
            wyckoff_patterns = wyckoff_result.get('patterns', [])
            
            # Analyze confluence
            confluence_patterns = self._find_confluence_setups(
                fib_result, elliott_result, wyckoff_result, min_strength
            )
            
            # Calculate overall market sentiment
            market_sentiment = self._calculate_market_sentiment(
                fib_patterns, elliott_patterns, wyckoff_patterns
            )
            
            return {
                "patterns": confluence_patterns,
                "individual_methods": {
                    "fibonacci": {"patterns": len(fib_patterns), "data": fib_result},
                    "elliott_wave": {"patterns": len(elliott_patterns), "data": elliott_result},
                    "wyckoff": {"patterns": len(wyckoff_patterns), "data": wyckoff_result}
                },
                "market_sentiment": market_sentiment,
                "confluence_summary": self._generate_confluence_summary(confluence_patterns),
                "trading_recommendations": self._generate_trading_recommendations(confluence_patterns)
            }
            
        except Exception as e:
            return {"patterns": [], "error": f"Confluence analysis failed: {str(e)}"}
    
    def _find_confluence_setups(self, fib_data: Dict, elliott_data: Dict, wyckoff_data: Dict, min_strength: float) -> List[Dict]:
        """Find high-probability setups where multiple methods agree"""
        confluence_patterns = []
        
        # Golden Zone Confluence Analysis
        if fib_data.get("in_golden_zone"):
            golden_zone_level = fib_data["golden_zone_level"]
            confluence_factors = ["fibonacci_golden_zone"]
            confidence_score = 0.9  # Start with high Fibonacci confidence
            
            # Check Elliott Wave confirmation
            elliott_patterns = elliott_data.get('patterns', [])
            corrective_waves = [p for p in elliott_patterns if p.get("wave_type") == "3_wave_correction"]
            if corrective_waves:
                confluence_factors.append("elliott_wave_correction")
                confidence_score += 0.1
            
            # Check Wyckoff confirmation
            wyckoff_patterns = wyckoff_data.get('patterns', [])
            accumulation_signals = [p for p in wyckoff_patterns if "accumulation" in p.get("name", "").lower()]
            if accumulation_signals:
                confluence_factors.append("wyckoff_accumulation")
                confidence_score += 0.1
            
            # Check for spring confirmation
            springs = [p for p in wyckoff_patterns if p.get("type") == "spring"]
            if springs:
                confluence_factors.append("wyckoff_spring")
                confidence_score += 0.15
            
            if len(confluence_factors) >= 2 and confidence_score >= min_strength:
                confluence_patterns.append({
                    "name": "Golden Zone Confluence Setup",
                    "type": "high_probability_reversal",
                    "confluence_factors": confluence_factors,
                    "confidence_score": min(confidence_score, 1.0),
                    "entry_zone": golden_zone_level,
                    "strength": confidence_score,
                    "signal": "strong_buy_opportunity" if fib_data.get("trend_direction") == "bullish" else "strong_sell_opportunity",
                    "setup_quality": "excellent" if len(confluence_factors) >= 3 else "good",
                    "risk_reward_ratio": self._calculate_risk_reward(fib_data, elliott_data, wyckoff_data),
                    "trading_plan": self._create_trading_plan(fib_data, elliott_data, wyckoff_data, "golden_zone")
                })
        
        # Elliott Wave + Wyckoff Confluence
        elliott_impulses = [p for p in elliott_data.get('patterns', []) if p.get('type') == 'impulse']
        wyckoff_markup = [p for p in wyckoff_data.get('patterns', []) if p.get('phase') == 'markup']
        
        if elliott_impulses and wyckoff_markup:
            confluence_factors = ["elliott_wave_impulse", "wyckoff_markup"]
            confidence_score = 0.85
            
            confluence_patterns.append({
                "name": "Elliott Wave Impulse + Wyckoff Markup",
                "type": "trend_continuation",
                "confluence_factors": confluence_factors,
                "confidence_score": confidence_score,
                "strength": confidence_score,
                "signal": "strong_trend_continuation",
                "setup_quality": "excellent",
                "trading_plan": self._create_trading_plan(fib_data, elliott_data, wyckoff_data, "trend_continuation")
            })
        
        # Triple Confluence (all three methods agree)
        fib_signals = self._extract_signals(fib_data.get('patterns', []))
        elliott_signals = self._extract_signals(elliott_data.get('patterns', []))
        wyckoff_signals = self._extract_signals(wyckoff_data.get('patterns', []))
        
        common_signals = set(fib_signals) & set(elliott_signals) & set(wyckoff_signals)
        
        if common_signals:
            for signal in common_signals:
                confluence_patterns.append({
                    "name": f"Triple Confluence - {signal.title()}",
                    "type": "triple_confluence",
                    "confluence_factors": ["fibonacci", "elliott_wave", "wyckoff"],
                    "confidence_score": 0.95,
                    "strength": 0.95,
                    "signal": signal,
                    "setup_quality": "exceptional",
                    "rarity": "rare_setup",
                    "trading_plan": self._create_trading_plan(fib_data, elliott_data, wyckoff_data, "triple_confluence")
                })
        
        return confluence_patterns
    
    def _extract_signals(self, patterns: List[Dict]) -> List[str]:
        """Extract trading signals from pattern list"""
        signals = []
        for pattern in patterns:
            signal = pattern.get('signal', '')
            if signal:
                signals.append(signal)
        return signals
    
    def _calculate_market_sentiment(self, fib_patterns: List, elliott_patterns: List, wyckoff_patterns: List) -> Dict[str, Any]:
        """Calculate overall market sentiment based on all methods"""
        bullish_signals = 0
        bearish_signals = 0
        neutral_signals = 0
        
        all_patterns = fib_patterns + elliott_patterns + wyckoff_patterns
        
        for pattern in all_patterns:
            signal = pattern.get('signal', '').lower()
            
            if any(word in signal for word in ['buy', 'bullish', 'support', 'accumulation', 'markup']):
                bullish_signals += 1
            elif any(word in signal for word in ['sell', 'bearish', 'resistance', 'distribution', 'markdown']):
                bearish_signals += 1
            else:
                neutral_signals += 1
        
        total_signals = bullish_signals + bearish_signals + neutral_signals
        
        if total_signals == 0:
            return {"sentiment": "unknown", "confidence": 0.0}
        
        bullish_ratio = bullish_signals / total_signals
        bearish_ratio = bearish_signals / total_signals
        
        if bullish_ratio > 0.6:
            sentiment = "bullish"
            confidence = bullish_ratio
        elif bearish_ratio > 0.6:
            sentiment = "bearish"
            confidence = bearish_ratio
        else:
            sentiment = "neutral"
            confidence = 1 - max(bullish_ratio, bearish_ratio)
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
            "neutral_signals": neutral_signals,
            "total_patterns": total_signals
        }
    
    def _generate_confluence_summary(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Generate summary of confluence analysis"""
        if not patterns:
            return {"quality": "no_confluence", "setups_found": 0}
        
        high_quality_setups = len([p for p in patterns if p.get('setup_quality') in ['excellent', 'exceptional']])
        avg_confidence = sum(p.get('confidence_score', 0) for p in patterns) / len(patterns)
        
        return {
            "setups_found": len(patterns),
            "high_quality_setups": high_quality_setups,
            "average_confidence": round(avg_confidence, 2),
            "best_setup": max(patterns, key=lambda x: x.get('confidence_score', 0)) if patterns else None,
            "recommendation": self._get_overall_recommendation(patterns)
        }
    
    def _get_overall_recommendation(self, patterns: List[Dict]) -> str:
        """Get overall trading recommendation"""
        if not patterns:
            return "no_clear_direction"
        
        excellent_setups = [p for p in patterns if p.get('setup_quality') in ['excellent', 'exceptional']]
        
        if excellent_setups:
            signals = [p.get('signal', '') for p in excellent_setups]
            buy_signals = len([s for s in signals if 'buy' in s.lower() or 'bullish' in s.lower()])
            sell_signals = len([s for s in signals if 'sell' in s.lower() or 'bearish' in s.lower()])
            
            if buy_signals > sell_signals:
                return "strong_buy_recommendation"
            elif sell_signals > buy_signals:
                return "strong_sell_recommendation"
        
        return "mixed_signals"
    
    def _calculate_risk_reward(self, fib_data: Dict, elliott_data: Dict, wyckoff_data: Dict) -> Optional[float]:
        """Calculate risk-reward ratio based on multiple methods"""
        try:
            # Use Fibonacci levels for risk/reward calculation
            fib_patterns = fib_data.get('patterns', [])
            golden_zone_patterns = [p for p in fib_patterns if p.get('is_golden_zone')]
            
            if golden_zone_patterns:
                pattern = golden_zone_patterns[0]
                entry = pattern.get('entry_zone', 0)
                stop_loss = pattern.get('stop_loss', 0)
                target = pattern.get('target', 0)
                
                if entry > 0 and stop_loss > 0 and target > 0:
                    risk = abs(entry - stop_loss)
                    reward = abs(target - entry)
                    
                    if risk > 0:
                        return round(reward / risk, 2)
            
            return None
            
        except (ZeroDivisionError, KeyError):
            return None
    
    def _create_trading_plan(self, fib_data: Dict, elliott_data: Dict, wyckoff_data: Dict, setup_type: str) -> Dict[str, Any]:
        """Create detailed trading plan based on confluence analysis"""
        plan = {
            "setup_type": setup_type,
            "entry_strategy": [],
            "risk_management": [],
            "profit_targets": [],
            "invalidation_levels": []
        }
        
        # Add Fibonacci-based levels
        fib_patterns = fib_data.get('patterns', [])
        golden_zone_patterns = [p for p in fib_patterns if p.get('is_golden_zone')]
        
        if golden_zone_patterns:
            pattern = golden_zone_patterns[0]
            plan["entry_strategy"].append(f"Enter near Golden Zone: {pattern.get('entry_zone', 'N/A')}")
            plan["risk_management"].append(f"Stop loss: {pattern.get('stop_loss', 'N/A')}")
            plan["profit_targets"].append(f"Primary target: {pattern.get('target', 'N/A')}")
        
        # Add Elliott Wave considerations
        elliott_patterns = elliott_data.get('patterns', [])
        if elliott_patterns:
            wave_position = elliott_data.get('probable_wave_position', '')
            plan["entry_strategy"].append(f"Elliott Wave context: {wave_position}")
        
        # Add Wyckoff considerations
        wyckoff_phase = wyckoff_data.get('current_phase', '')
        if wyckoff_phase:
            plan["entry_strategy"].append(f"Wyckoff phase: {wyckoff_phase}")
        
        return plan
    
    def _generate_trading_recommendations(self, patterns: List[Dict]) -> List[Dict]:
        """Generate specific trading recommendations"""
        recommendations = []
        
        for pattern in patterns:
            setup_quality = pattern.get('setup_quality', 'unknown')
            confidence = pattern.get('confidence_score', 0)
            
            if setup_quality in ['excellent', 'exceptional'] and confidence >= 0.8:
                recommendations.append({
                    "action": pattern.get('signal', '').replace('_', ' ').title(),
                    "setup": pattern.get('name', ''),
                    "confidence": confidence,
                    "quality": setup_quality,
                    "position_size": self._suggest_position_size(confidence, setup_quality),
                    "time_horizon": self._suggest_time_horizon(pattern),
                    "key_levels": self._extract_key_levels(pattern)
                })
        
        return recommendations
    
    def _suggest_position_size(self, confidence: float, setup_quality: str) -> str:
        """Suggest position size based on setup quality"""
        if setup_quality == 'exceptional' and confidence >= 0.9:
            return "large_position"
        elif setup_quality == 'excellent' and confidence >= 0.8:
            return "medium_position"
        else:
            return "small_position"
    
    def _suggest_time_horizon(self, pattern: Dict) -> str:
        """Suggest appropriate time horizon for the trade"""
        setup_type = pattern.get('type', '')
        
        if 'reversal' in setup_type:
            return "medium_term"
        elif 'continuation' in setup_type:
            return "short_to_medium_term"
        else:
            return "flexible"
    
    def _extract_key_levels(self, pattern: Dict) -> Dict[str, Any]:
        """Extract key price levels from pattern"""
        levels = {}
        
        if 'entry_zone' in pattern:
            levels['entry'] = pattern['entry_zone']
        
        if 'trading_plan' in pattern:
            plan = pattern['trading_plan']
            # Extract levels from trading plan if available
            
        return levels