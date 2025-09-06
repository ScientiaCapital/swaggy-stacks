"""
Markov Trading Agent - Enhanced with RAG and Pattern Learning
Integrates existing Markov Chain system with LangChain agents
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from langchain.agents import Tool
from langchain.schema import HumanMessage

from backend.app.rag.agents.base_agent import BaseTradingAgent, TradingSignal
from backend.app.analysis.enhanced_markov_system import EnhancedMarkovSystem

logger = logging.getLogger(__name__)

class MarkovTradingAgent(BaseTradingAgent):
    """
    Enhanced Markov Chain trading agent with RAG capabilities
    Converts the existing Markov system into an intelligent agent
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            agent_name="markov_agent",
            strategy_type="markov_chain",
            **kwargs
        )
        
        # Initialize the existing Markov system
        self.markov_system = EnhancedMarkovSystem()
        
        # Markov-specific parameters
        self.state_transition_threshold = 0.6
        self.confidence_adjustment_factor = 1.2
        self.lookback_periods = [5, 10, 20, 50]  # For multi-timeframe analysis
        
        logger.info("MarkovTradingAgent initialized")
    
    async def _create_tools(self) -> List[Tool]:
        """Create LangChain tools specific to Markov analysis"""
        return [
            Tool(
                name="calculate_markov_states",
                func=self._calculate_markov_states,
                description="Calculate current Markov chain states from price data"
            ),
            Tool(
                name="get_transition_probabilities", 
                func=self._get_transition_probabilities,
                description="Get state transition probability matrix"
            ),
            Tool(
                name="predict_next_state",
                func=self._predict_next_state,
                description="Predict the most likely next market state"
            ),
            Tool(
                name="get_state_persistence",
                func=self._get_state_persistence,
                description="Analyze how long states typically persist"
            ),
            Tool(
                name="find_markov_patterns",
                func=self._find_similar_markov_patterns,
                description="Find similar Markov state patterns in history"
            )
        ]
    
    def _calculate_markov_states(self, price_data: str) -> str:
        """Calculate Markov states from price data"""
        try:
            # Parse price data (assuming JSON string or comma-separated)
            if isinstance(price_data, str):
                prices = [float(x.strip()) for x in price_data.split(',')]
            else:
                prices = price_data
                
            if len(prices) < 10:
                return "Insufficient price data for Markov analysis"
            
            # Use the enhanced Markov system
            current_state = self.markov_system.calculate_state(prices)
            state_info = self.markov_system.get_state_info(current_state)
            
            return f"Current Markov state: {current_state}, Characteristics: {state_info}"
            
        except Exception as e:
            return f"Error calculating Markov states: {str(e)}"
    
    def _get_transition_probabilities(self, current_state: str) -> str:
        """Get transition probabilities from current state"""
        try:
            state_idx = int(current_state) if current_state.isdigit() else 0
            transition_probs = self.markov_system.get_transition_probabilities(state_idx)
            
            result = f"Transition probabilities from state {state_idx}:\n"
            for next_state, prob in enumerate(transition_probs):
                result += f"  To state {next_state}: {prob:.3f}\n"
                
            return result
            
        except Exception as e:
            return f"Error getting transition probabilities: {str(e)}"
    
    def _predict_next_state(self, current_state: str) -> str:
        """Predict the next most likely state"""
        try:
            state_idx = int(current_state) if current_state.isdigit() else 0
            next_state, probability = self.markov_system.predict_next_state(state_idx)
            
            return f"Most likely next state: {next_state} (probability: {probability:.3f})"
            
        except Exception as e:
            return f"Error predicting next state: {str(e)}"
    
    def _get_state_persistence(self, state: str) -> str:
        """Analyze state persistence"""
        try:
            state_idx = int(state) if state.isdigit() else 0
            persistence_info = self.markov_system.get_state_persistence(state_idx)
            
            return f"State {state_idx} persistence: avg duration {persistence_info.get('avg_duration', 0):.1f} periods"
            
        except Exception as e:
            return f"Error analyzing state persistence: {str(e)}"
    
    async def _find_similar_markov_patterns(self, state_sequence: str) -> str:
        """Find similar Markov state sequences"""
        try:
            # Convert state sequence to features for similarity search
            features = {
                'state_sequence': state_sequence,
                'sequence_length': len(state_sequence.split()),
                'strategy': 'markov_pattern_matching'
            }
            
            similar_patterns = await self.find_similar_patterns(features, similarity_threshold=0.7)
            
            if not similar_patterns:
                return "No similar Markov patterns found"
            
            result = f"Found {len(similar_patterns)} similar patterns:\n"
            for i, pattern in enumerate(similar_patterns[:3], 1):
                result += f"  {i}. {pattern.pattern_name} (similarity: {pattern.similarity:.2f}, success: {pattern.success_rate:.1%})\n"
                
            return result
            
        except Exception as e:
            return f"Error finding similar patterns: {str(e)}"
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> TradingSignal:
        """
        Main analysis method - combines Markov analysis with RAG enhancement
        """
        try:
            # Extract price data
            prices = market_data.get('prices', [])
            symbol = market_data.get('symbol', 'UNKNOWN')
            current_price = market_data.get('current_price', 0.0)
            
            if len(prices) < 20:
                return TradingSignal(
                    agent_type=self.agent_name,
                    strategy_name=self.strategy_type,
                    symbol=symbol,
                    action="HOLD",
                    confidence=0.0,
                    reasoning="Insufficient price data for Markov analysis"
                )
            
            # Extract market features
            features = self._extract_market_features(market_data)
            
            # Perform Markov analysis
            markov_result = self._perform_markov_analysis(prices)
            
            # Find similar historical patterns
            similar_patterns = await self.find_similar_patterns(features)
            
            # Get contextual information
            pattern_context = await self.get_pattern_context(features)
            
            # Generate enhanced signal with RAG context
            signal = self._generate_enhanced_signal(
                markov_result, 
                similar_patterns, 
                pattern_context,
                symbol,
                current_price
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in Markov agent analysis: {e}")
            return TradingSignal(
                agent_type=self.agent_name,
                strategy_name=self.strategy_type,
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                reasoning=f"Analysis error: {str(e)}"
            )
    
    def _extract_market_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Markov-specific features from market data"""
        prices = market_data.get('prices', [])
        
        if len(prices) < 10:
            return {'error': 'insufficient_data'}
        
        # Calculate multi-timeframe Markov states
        features = {}
        
        for period in self.lookback_periods:
            if len(prices) >= period:
                period_prices = prices[-period:]
                state = self.markov_system.calculate_state(period_prices)
                features[f'markov_state_{period}'] = state
                
                # Get transition information
                transition_matrix = self.markov_system.get_transition_probabilities(state)
                features[f'max_transition_prob_{period}'] = max(transition_matrix)
                features[f'entropy_{period}'] = self._calculate_entropy(transition_matrix)
        
        # Overall market characteristics
        returns = np.diff(np.log(prices))
        features.update({
            'volatility': np.std(returns) * np.sqrt(252),
            'trend_strength': self._calculate_trend_strength(prices),
            'momentum': np.mean(returns[-5:]) if len(returns) >= 5 else 0.0,
            'price_position': (prices[-1] - min(prices[-20:])) / (max(prices[-20:]) - min(prices[-20:])) if len(prices) >= 20 else 0.5
        })
        
        return features
    
    def _perform_markov_analysis(self, prices: List[float]) -> Dict[str, Any]:
        """Perform core Markov analysis"""
        try:
            # Calculate current state
            current_state = self.markov_system.calculate_state(prices)
            
            # Get transition probabilities
            transition_probs = self.markov_system.get_transition_probabilities(current_state)
            
            # Predict next state
            next_state, next_state_prob = self.markov_system.predict_next_state(current_state)
            
            # Calculate confidence based on transition probability and state persistence
            confidence = min(1.0, next_state_prob * self.confidence_adjustment_factor)
            
            # Determine action based on state characteristics
            state_info = self.markov_system.get_state_info(next_state)
            action = self._state_to_action(next_state, state_info, confidence)
            
            return {
                'current_state': current_state,
                'next_state': next_state,
                'next_state_probability': next_state_prob,
                'action': action,
                'confidence': confidence,
                'transition_matrix': transition_probs.tolist(),
                'state_info': state_info
            }
            
        except Exception as e:
            logger.error(f"Error in Markov analysis: {e}")
            return {
                'current_state': 0,
                'action': 'HOLD',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _state_to_action(self, state: int, state_info: Dict, confidence: float) -> str:
        """Convert Markov state to trading action"""
        # This maps Markov states to trading actions based on the enhanced system
        # States are typically: 0=strong_bear, 1=bear, 2=neutral, 3=bull, 4=strong_bull
        
        if confidence < self.state_transition_threshold:
            return "HOLD"
        
        if state >= 3:  # Bull or strong bull
            return "BUY"
        elif state <= 1:  # Bear or strong bear
            return "SELL"
        else:  # Neutral
            return "HOLD"
    
    def _generate_enhanced_signal(
        self, 
        markov_result: Dict[str, Any], 
        similar_patterns: List,
        pattern_context: str,
        symbol: str,
        current_price: float
    ) -> TradingSignal:
        """Generate enhanced trading signal with RAG context"""
        
        action = markov_result.get('action', 'HOLD')
        base_confidence = markov_result.get('confidence', 0.0)
        
        # Adjust confidence based on similar patterns
        confidence_adjustment = 0.0
        if similar_patterns:
            avg_success_rate = np.mean([p.success_rate for p in similar_patterns])
            confidence_adjustment = (avg_success_rate - 0.5) * 0.3  # Max ±15% adjustment
        
        final_confidence = min(1.0, max(0.0, base_confidence + confidence_adjustment))
        
        # Generate reasoning with pattern context
        reasoning_parts = [
            f"Markov Analysis: Current state {markov_result.get('current_state')}, "
            f"predicted next state {markov_result.get('next_state')} "
            f"(probability: {markov_result.get('next_state_probability', 0):.2f})"
        ]
        
        if similar_patterns:
            reasoning_parts.append(f"Historical patterns: Found {len(similar_patterns)} similar cases "
                                 f"with {avg_success_rate:.1%} average success rate")
        
        if confidence_adjustment != 0:
            reasoning_parts.append(f"Confidence adjusted by {confidence_adjustment:+.1%} based on pattern analysis")
        
        # Calculate position sizing and risk levels
        metadata = {
            'markov_state': markov_result.get('current_state'),
            'predicted_state': markov_result.get('next_state'),
            'transition_probability': markov_result.get('next_state_probability'),
            'similar_patterns_count': len(similar_patterns),
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
    
    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate entropy of probability distribution"""
        # Avoid log(0) by adding small epsilon
        eps = 1e-10
        probabilities = np.array(probabilities) + eps
        probabilities = probabilities / np.sum(probabilities)  # Normalize
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength indicator"""
        if len(prices) < 20:
            return 0.0
        
        # Use linear regression slope as trend strength
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        
        # Normalize by price level
        normalized_slope = slope / np.mean(prices)
        
        # Return bounded value between -1 and 1
        return np.tanh(normalized_slope * 100)

# Test function
async def test_markov_agent():
    """Test the Markov Trading Agent"""
    print("🧪 Testing Markov Trading Agent...")
    
    agent = MarkovTradingAgent()
    await agent.initialize()
    
    # Test with sample market data
    sample_data = {
        'symbol': 'TEST',
        'current_price': 100.0,
        'prices': [95, 96, 97, 99, 100, 101, 103, 102, 104, 105, 107, 106, 108, 110, 109, 111, 113, 112, 114, 115],
        'volumes': [1000] * 20
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
    
    # Test performance summary
    performance = await agent.get_performance_summary()
    print(f"✅ Performance: {performance}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_markov_agent())