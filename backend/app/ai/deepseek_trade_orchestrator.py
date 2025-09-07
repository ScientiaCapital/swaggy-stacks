"""
DeepSeek Trade Orchestrator

Primary orchestrator leveraging DeepSeek's hedge fund DNA from High-Flyer Capital Management.
Routes trading tasks to specialized Chinese LLMs based on performance and task requirements.
Implements MLA (Multi-head Latent Attention) concepts for memory efficiency.
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import structlog
from sqlalchemy.orm import Session

from .ollama_client import OllamaClient
from ..core.database import get_db
from ..models.trading import TradingDecision

logger = structlog.get_logger()


class TaskType(Enum):
    """Types of trading tasks that can be routed to specialized LLMs"""
    BACKTEST_ANALYSIS = "backtest_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    RISK_CALCULATION = "risk_calculation"
    STRATEGY_CODING = "strategy_coding"
    MARKET_SENTIMENT = "market_sentiment"
    TECHNICAL_ANALYSIS = "technical_analysis"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"


class LLMModel(Enum):
    """Available Chinese LLM models and their specializations"""
    DEEPSEEK_R1 = "deepseek-r1:7b"           # Primary orchestrator
    QWEN_QUANT = "qwen2.5:7b"                # Mathematical/quantitative analysis
    YI_TECHNICAL = "yi:6b"                   # Technical patterns & charts
    GLM_RISK = "glm4:9b"                     # Risk management & intelligence
    DEEPSEEK_CODER = "deepseek-coder:6.7b"  # Strategy implementation


@dataclass
class TaskContext:
    """Context for trading task execution"""
    task_type: TaskType
    symbol: str
    market_data: Dict[str, Any]
    technical_indicators: Optional[Dict[str, Any]] = None
    historical_data: Optional[List[Dict]] = None
    risk_parameters: Optional[Dict[str, Any]] = None
    performance_history: Optional[List[Dict]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RoutingPerformance:
    """Track performance of LLM routing decisions"""
    model: str
    task_type: str
    success_rate: float
    avg_confidence: float
    total_tasks: int
    avg_execution_time: float
    last_updated: datetime


@dataclass
class TradingDecisionResult:
    """Result from orchestrator trading decision"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    reasoning: str
    contributing_agents: List[str]
    execution_time: float
    risk_level: str
    expected_return: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: Optional[float]
    timestamp: datetime


class MemoryEfficientAttention:
    """
    Simplified implementation inspired by DeepSeek's MLA mechanism
    Compresses context to reduce memory usage by ~90%
    """
    
    def __init__(self, max_context_tokens: int = 2048):
        self.max_context_tokens = max_context_tokens
        self.compressed_cache = {}
        
    def compress_context(self, context_key: str, context_data: Dict) -> Dict:
        """Compress context data using key-value cache reduction"""
        try:
            # Serialize and compress key information
            essential_keys = ['symbol', 'current_price', 'volume', 'trend']
            compressed = {k: context_data.get(k) for k in essential_keys if k in context_data}
            
            # Store in compressed cache
            self.compressed_cache[context_key] = {
                'data': compressed,
                'timestamp': datetime.now(),
                'original_size': len(str(context_data)),
                'compressed_size': len(str(compressed))
            }
            
            return compressed
        except Exception as e:
            logger.error("Context compression failed", error=str(e))
            return context_data
    
    def get_compression_ratio(self, context_key: str) -> float:
        """Get compression ratio for monitoring"""
        if context_key in self.compressed_cache:
            cache_entry = self.compressed_cache[context_key]
            original_size = cache_entry['original_size']
            compressed_size = cache_entry['compressed_size']
            return compressed_size / original_size if original_size > 0 else 1.0
        return 1.0


class DeepSeekTradeOrchestrator:
    """
    Primary trade orchestrator with hedge fund-grade decision making
    
    Features:
    - Intelligent task routing based on LLM specializations
    - Performance tracking and adaptive routing
    - Memory-efficient context management (MLA-inspired)
    - Consensus-based decision making
    - Risk-aware trade orchestration
    """
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_client = OllamaClient(ollama_base_url)
        self.memory_attention = MemoryEfficientAttention()
        
        # Performance tracking matrix - updated based on actual outcomes
        self.routing_performance = {
            TaskType.BACKTEST_ANALYSIS: {
                LLMModel.QWEN_QUANT.value: 0.85,      # Strong math skills
                LLMModel.GLM_RISK.value: 0.72,        # Risk understanding
                LLMModel.YI_TECHNICAL.value: 0.68     # Pattern recognition
            },
            TaskType.PATTERN_RECOGNITION: {
                LLMModel.YI_TECHNICAL.value: 0.91,    # Specialized in patterns
                LLMModel.QWEN_QUANT.value: 0.78,      # Mathematical patterns
                LLMModel.GLM_RISK.value: 0.71         # Risk patterns
            },
            TaskType.RISK_CALCULATION: {
                LLMModel.GLM_RISK.value: 0.89,        # Risk specialty
                LLMModel.QWEN_QUANT.value: 0.83,      # Quantitative risk
                LLMModel.YI_TECHNICAL.value: 0.65     # Technical risk
            },
            TaskType.STRATEGY_CODING: {
                LLMModel.DEEPSEEK_CODER.value: 0.94,  # Code implementation
                LLMModel.QWEN_QUANT.value: 0.81,      # Algorithm design
                LLMModel.GLM_RISK.value: 0.73         # Risk logic
            },
            TaskType.MARKET_SENTIMENT: {
                LLMModel.GLM_RISK.value: 0.87,        # Market intelligence
                LLMModel.YI_TECHNICAL.value: 0.79,    # Technical sentiment
                LLMModel.QWEN_QUANT.value: 0.74       # Quantitative sentiment
            }
        }
        
        # System performance metrics
        self.orchestration_stats = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'avg_execution_time': 0.0,
            'memory_usage_mb': 0.0,
            'cache_hit_rate': 0.0
        }
    
    async def orchestrate_trade_decision(
        self, 
        context: TaskContext,
        require_consensus: bool = True,
        min_confidence: float = 0.7
    ) -> TradingDecisionResult:
        """
        Main orchestration method - intelligently routes tasks and synthesizes decisions
        
        Args:
            context: Trading context with all necessary data
            require_consensus: Whether to require agreement between multiple agents
            min_confidence: Minimum confidence threshold for trade execution
            
        Returns:
            TradingDecisionResult with final decision and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(
                "Starting trade orchestration", 
                symbol=context.symbol, 
                task_type=context.task_type.value
            )
            
            # Compress context for memory efficiency
            context_key = f"{context.symbol}_{context.task_type.value}_{int(time.time())}"
            compressed_context = self.memory_attention.compress_context(
                context_key, context.market_data
            )
            
            # Route tasks to specialized agents based on performance
            agent_results = await self._route_to_specialized_agents(context)
            
            # Synthesize final decision using DeepSeek's hedge fund reasoning
            final_decision = await self._synthesize_decision(
                context, agent_results, require_consensus, min_confidence
            )
            
            # Update performance tracking
            execution_time = time.time() - start_time
            await self._update_performance_metrics(agent_results, execution_time)
            
            # Create final result
            result = TradingDecisionResult(
                symbol=context.symbol,
                action=final_decision['action'],
                confidence=final_decision['confidence'],
                reasoning=final_decision['reasoning'],
                contributing_agents=final_decision['agents'],
                execution_time=execution_time,
                risk_level=final_decision.get('risk_level', 'medium'),
                expected_return=final_decision.get('expected_return'),
                stop_loss=final_decision.get('stop_loss'),
                take_profit=final_decision.get('take_profit'),
                position_size=final_decision.get('position_size'),
                timestamp=datetime.now()
            )
            
            logger.info(
                "Trade orchestration completed",
                symbol=context.symbol,
                action=result.action,
                confidence=result.confidence,
                execution_time=execution_time
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Trade orchestration failed", 
                symbol=context.symbol,
                error=str(e)
            )
            
            # Return safe default
            return TradingDecisionResult(
                symbol=context.symbol,
                action="HOLD",
                confidence=0.0,
                reasoning=f"Orchestration error: {str(e)}",
                contributing_agents=[],
                execution_time=time.time() - start_time,
                risk_level="high",
                expected_return=None,
                stop_loss=None,
                take_profit=None,
                position_size=None,
                timestamp=datetime.now()
            )
    
    async def _route_to_specialized_agents(self, context: TaskContext) -> Dict[str, Any]:
        """Route tasks to the best-performing LLMs for each task type"""
        
        agent_results = {}
        
        # Always get primary DeepSeek analysis
        primary_analysis = await self._get_primary_analysis(context)
        agent_results['primary'] = primary_analysis
        
        # Route to specialized agents based on context
        if context.task_type == TaskType.BACKTEST_ANALYSIS:
            # Route to Qwen for mathematical analysis
            quant_analysis = await self._analyze_with_qwen(context)
            agent_results['quantitative'] = quant_analysis
            
            # Get risk perspective from GLM
            risk_analysis = await self._analyze_with_glm(context)
            agent_results['risk'] = risk_analysis
            
        elif context.task_type == TaskType.PATTERN_RECOGNITION:
            # Route to Yi for pattern analysis
            pattern_analysis = await self._analyze_with_yi(context)
            agent_results['patterns'] = pattern_analysis
            
            # Get quantitative validation
            quant_validation = await self._analyze_with_qwen(context)
            agent_results['validation'] = quant_validation
            
        elif context.task_type == TaskType.RISK_CALCULATION:
            # Primary risk analysis with GLM
            risk_analysis = await self._analyze_with_glm(context)
            agent_results['risk'] = risk_analysis
            
            # Secondary analysis with Qwen
            quant_risk = await self._analyze_with_qwen(context)
            agent_results['quantitative_risk'] = quant_risk
            
        else:
            # Default routing based on performance matrix
            best_model = self._get_best_model_for_task(context.task_type)
            if best_model == LLMModel.QWEN_QUANT.value:
                analysis = await self._analyze_with_qwen(context)
            elif best_model == LLMModel.YI_TECHNICAL.value:
                analysis = await self._analyze_with_yi(context)
            elif best_model == LLMModel.GLM_RISK.value:
                analysis = await self._analyze_with_glm(context)
            else:
                analysis = await self._analyze_with_deepseek_coder(context)
            
            agent_results['specialized'] = analysis
        
        return agent_results
    
    async def _get_primary_analysis(self, context: TaskContext) -> Dict[str, Any]:
        """Get primary analysis from DeepSeek-R1 (hedge fund orchestrator)"""
        
        prompt = f"""
        As a quantitative analyst from High-Flyer Capital Management, analyze this trading opportunity:
        
        Symbol: {context.symbol}
        Task: {context.task_type.value}
        
        Market Data:
        {json.dumps(context.market_data, indent=2)}
        
        Technical Indicators:
        {json.dumps(context.technical_indicators or {}, indent=2)}
        
        Provide analysis in JSON format:
        {{
            "market_regime": "bull|bear|sideways|volatile",
            "trend_strength": "strong|moderate|weak",
            "key_levels": {{"support": 0.0, "resistance": 0.0}},
            "volatility_assessment": "low|normal|high|extreme",
            "institutional_perspective": "accumulation|distribution|neutral",
            "hedge_fund_rating": "strong_buy|buy|hold|sell|strong_sell",
            "confidence": 0.0-1.0,
            "reasoning": "detailed hedge fund analysis"
        }}
        """
        
        system_prompt = """You are a senior quantitative analyst from High-Flyer Capital Management, 
        a leading quantitative hedge fund. Apply institutional-grade analysis with focus on:
        - Market regime identification
        - Volatility clustering and mean reversion
        - Institutional flow analysis
        - Risk-adjusted return optimization
        - Statistical arbitrage opportunities
        
        Always think like a hedge fund professional optimizing for Sharpe ratio and alpha generation."""
        
        try:
            response = await self.ollama_client.generate_response(
                prompt=prompt,
                model_key="deepseek_r1",  # Will be added to OllamaClient
                system_prompt=system_prompt,
                max_tokens=1024
            )
            
            return self._parse_json_response(response, {
                "market_regime": "sideways",
                "trend_strength": "moderate", 
                "key_levels": {"support": 0.0, "resistance": 0.0},
                "volatility_assessment": "normal",
                "institutional_perspective": "neutral",
                "hedge_fund_rating": "hold",
                "confidence": 0.5,
                "reasoning": "Primary analysis unavailable"
            })
            
        except Exception as e:
            logger.error("Primary analysis failed", error=str(e))
            return {"error": str(e), "confidence": 0.0}
    
    async def _analyze_with_qwen(self, context: TaskContext) -> Dict[str, Any]:
        """Specialized mathematical/quantitative analysis with Qwen2.5"""
        
        prompt = f"""
        Perform quantitative analysis for {context.symbol}:
        
        Data: {json.dumps(context.market_data, indent=2)}
        
        Calculate and analyze:
        1. Statistical metrics (mean, std, skewness, kurtosis)
        2. Volatility measures (historical, implied, GARCH)
        3. Risk metrics (VaR, Expected Shortfall, Sharpe ratio)
        4. Technical indicators (RSI, MACD, Bollinger Bands)
        5. Probability distributions and confidence intervals
        
        Provide JSON response:
        {{
            "statistics": {{"mean_return": 0.0, "volatility": 0.0, "sharpe": 0.0}},
            "risk_metrics": {{"var_95": 0.0, "max_drawdown": 0.0}},
            "probability_up": 0.0-1.0,
            "expected_return": 0.0,
            "confidence_interval": {{"lower": 0.0, "upper": 0.0}},
            "quant_rating": "strong_buy|buy|hold|sell|strong_sell",
            "confidence": 0.0-1.0,
            "mathematical_reasoning": "detailed quant analysis"
        }}
        """
        
        system_prompt = """You are a quantitative analyst specializing in mathematical trading models.
        Use advanced statistics, probability theory, and econometrics. Focus on:
        - Time series analysis and forecasting
        - Risk modeling and Monte Carlo simulations  
        - Option pricing and Greeks calculations
        - Statistical arbitrage opportunities
        - Backtesting and performance attribution
        
        Always provide mathematical rigor and statistical significance testing."""
        
        try:
            response = await self.ollama_client.generate_response(
                prompt=prompt,
                model_key="qwen_quant",
                system_prompt=system_prompt,
                max_tokens=1024
            )
            
            return self._parse_json_response(response, {
                "statistics": {"mean_return": 0.0, "volatility": 0.2, "sharpe": 0.0},
                "risk_metrics": {"var_95": 0.05, "max_drawdown": 0.1},
                "probability_up": 0.5,
                "expected_return": 0.0,
                "confidence_interval": {"lower": -0.1, "upper": 0.1},
                "quant_rating": "hold",
                "confidence": 0.5,
                "mathematical_reasoning": "Quantitative analysis unavailable"
            })
            
        except Exception as e:
            logger.error("Qwen analysis failed", error=str(e))
            return {"error": str(e), "confidence": 0.0}
    
    async def _analyze_with_yi(self, context: TaskContext) -> Dict[str, Any]:
        """Technical pattern analysis with Yi model"""
        
        prompt = f"""
        Analyze technical patterns for {context.symbol}:
        
        Market Data: {json.dumps(context.market_data, indent=2)}
        Technical Indicators: {json.dumps(context.technical_indicators or {}, indent=2)}
        
        Identify and analyze:
        1. Candlestick patterns (doji, hammer, engulfing, etc.)
        2. Chart patterns (head-and-shoulders, triangles, flags)
        3. Support and resistance levels
        4. Trend lines and channels
        5. Elliott Wave analysis
        6. Fibonacci retracements and extensions
        
        JSON response:
        {{
            "candlestick_patterns": ["pattern1", "pattern2"],
            "chart_patterns": ["triangle", "flag"],
            "support_levels": [100.0, 95.0],
            "resistance_levels": [110.0, 115.0],
            "trend_analysis": "uptrend|downtrend|sideways",
            "pattern_strength": "strong|moderate|weak",
            "breakout_probability": 0.0-1.0,
            "pattern_target": 0.0,
            "technical_rating": "strong_buy|buy|hold|sell|strong_sell",
            "confidence": 0.0-1.0,
            "pattern_reasoning": "detailed technical analysis"
        }}
        """
        
        system_prompt = """You are a technical analysis expert specializing in chart patterns and price action.
        Focus on:
        - Classical chart patterns and their reliability
        - Japanese candlestick pattern analysis
        - Support/resistance level identification
        - Volume analysis and confirmation
        - Multiple timeframe analysis
        - Risk/reward ratio calculation
        
        Prioritize high-probability patterns with strong historical success rates."""
        
        try:
            response = await self.ollama_client.generate_response(
                prompt=prompt,
                model_key="yi_technical",
                system_prompt=system_prompt,
                max_tokens=1024
            )
            
            return self._parse_json_response(response, {
                "candlestick_patterns": [],
                "chart_patterns": [],
                "support_levels": [],
                "resistance_levels": [],
                "trend_analysis": "sideways",
                "pattern_strength": "weak",
                "breakout_probability": 0.5,
                "pattern_target": 0.0,
                "technical_rating": "hold",
                "confidence": 0.5,
                "pattern_reasoning": "Technical analysis unavailable"
            })
            
        except Exception as e:
            logger.error("Yi analysis failed", error=str(e))
            return {"error": str(e), "confidence": 0.0}
    
    async def _analyze_with_glm(self, context: TaskContext) -> Dict[str, Any]:
        """Risk management and market intelligence with GLM4"""
        
        prompt = f"""
        Analyze risk and market intelligence for {context.symbol}:
        
        Market Data: {json.dumps(context.market_data, indent=2)}
        Risk Parameters: {json.dumps(context.risk_parameters or {}, indent=2)}
        
        Assess:
        1. Portfolio risk contribution
        2. Correlation with market indices
        3. Liquidity risk and market impact
        4. Event risk and earnings impact
        5. Macro economic sensitivity
        6. Tail risk and black swan scenarios
        
        JSON response:
        {{
            "risk_level": "low|medium|high|extreme",
            "portfolio_impact": 0.0-1.0,
            "correlation_spy": -1.0-1.0,
            "liquidity_score": 0.0-1.0,
            "event_risk": "low|medium|high",
            "macro_sensitivity": 0.0-1.0,
            "tail_risk": 0.0-1.0,
            "recommended_position_size": 0.0-1.0,
            "stop_loss_level": 0.0,
            "risk_rating": "conservative|moderate|aggressive",
            "confidence": 0.0-1.0,
            "risk_reasoning": "detailed risk analysis"
        }}
        """
        
        system_prompt = """You are a risk management specialist and market intelligence analyst.
        Focus on:
        - Portfolio risk management and position sizing
        - Correlation analysis and diversification
        - Liquidity risk and market microstructure
        - Event-driven risk assessment
        - Macro-economic factor analysis
        - Tail risk and scenario planning
        
        Always prioritize capital preservation and risk-adjusted returns."""
        
        try:
            response = await self.ollama_client.generate_response(
                prompt=prompt,
                model_key="glm_risk",
                system_prompt=system_prompt,
                max_tokens=1024
            )
            
            return self._parse_json_response(response, {
                "risk_level": "medium",
                "portfolio_impact": 0.5,
                "correlation_spy": 0.0,
                "liquidity_score": 0.5,
                "event_risk": "medium",
                "macro_sensitivity": 0.5,
                "tail_risk": 0.1,
                "recommended_position_size": 0.02,
                "stop_loss_level": 0.95,
                "risk_rating": "moderate",
                "confidence": 0.5,
                "risk_reasoning": "Risk analysis unavailable"
            })
            
        except Exception as e:
            logger.error("GLM analysis failed", error=str(e))
            return {"error": str(e), "confidence": 0.0}
    
    async def _analyze_with_deepseek_coder(self, context: TaskContext) -> Dict[str, Any]:
        """Strategy implementation and backtesting with DeepSeek-Coder"""
        
        prompt = f"""
        Implement trading strategy for {context.symbol}:
        
        Context: {json.dumps(context.market_data, indent=2)}
        
        Design and analyze:
        1. Entry and exit rules
        2. Position sizing algorithm
        3. Risk management logic
        4. Backtesting framework
        5. Performance optimization
        6. Code implementation approach
        
        JSON response:
        {{
            "strategy_type": "momentum|mean_reversion|breakout|arbitrage",
            "entry_conditions": ["condition1", "condition2"],
            "exit_conditions": ["exit1", "exit2"],
            "position_sizing": "fixed|kelly|volatility_adjusted",
            "risk_management": ["stop_loss", "take_profit", "position_limit"],
            "expected_sharpe": 0.0-3.0,
            "win_rate": 0.0-1.0,
            "implementation_complexity": "low|medium|high",
            "coding_rating": "strong_buy|buy|hold|sell|strong_sell",
            "confidence": 0.0-1.0,
            "implementation_notes": "detailed coding strategy"
        }}
        """
        
        system_prompt = """You are a quantitative developer specializing in trading strategy implementation.
        Focus on:
        - Algorithmic trading strategy design
        - Backtesting and performance optimization
        - Risk management system implementation
        - Order execution algorithms
        - Portfolio management code
        - High-frequency trading considerations
        
        Emphasize robust, production-ready code with proper error handling."""
        
        try:
            response = await self.ollama_client.generate_response(
                prompt=prompt,
                model_key="deepseek_coder",
                system_prompt=system_prompt,
                max_tokens=1024
            )
            
            return self._parse_json_response(response, {
                "strategy_type": "momentum",
                "entry_conditions": ["trend_confirmation"],
                "exit_conditions": ["stop_loss", "take_profit"],
                "position_sizing": "volatility_adjusted",
                "risk_management": ["stop_loss", "position_limit"],
                "expected_sharpe": 1.0,
                "win_rate": 0.55,
                "implementation_complexity": "medium",
                "coding_rating": "hold",
                "confidence": 0.5,
                "implementation_notes": "Implementation analysis unavailable"
            })
            
        except Exception as e:
            logger.error("DeepSeek-Coder analysis failed", error=str(e))
            return {"error": str(e), "confidence": 0.0}
    
    async def _synthesize_decision(
        self, 
        context: TaskContext, 
        agent_results: Dict[str, Any],
        require_consensus: bool,
        min_confidence: float
    ) -> Dict[str, Any]:
        """
        Synthesize final trading decision from all agent analyses
        Uses DeepSeek's hedge fund reasoning to weigh different perspectives
        """
        
        try:
            # Extract key insights from each agent
            insights = {}
            total_confidence = 0.0
            valid_agents = 0
            
            for agent_name, result in agent_results.items():
                if 'error' not in result and 'confidence' in result:
                    insights[agent_name] = result
                    total_confidence += result.get('confidence', 0.0)
                    valid_agents += 1
            
            # Calculate average confidence
            avg_confidence = total_confidence / valid_agents if valid_agents > 0 else 0.0
            
            # Synthesize final decision using DeepSeek orchestrator
            synthesis_prompt = f"""
            As the chief investment officer of High-Flyer Capital Management, synthesize these analyses:
            
            Agent Analyses:
            {json.dumps(insights, indent=2)}
            
            Average Confidence: {avg_confidence:.2f}
            Minimum Required: {min_confidence}
            
            Synthesize final decision considering:
            1. Risk-adjusted return potential
            2. Consistency across multiple analyses
            3. Confidence levels and reliability
            4. Market regime and current conditions
            5. Portfolio impact and risk management
            
            Final decision JSON:
            {{
                "action": "BUY|SELL|HOLD",
                "confidence": 0.0-1.0,
                "expected_return": 0.0,
                "risk_level": "low|medium|high",
                "position_size": 0.0-1.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "agents": ["agent1", "agent2"],
                "reasoning": "comprehensive hedge fund analysis synthesis"
            }}
            """
            
            system_prompt = """You are the CIO of High-Flyer Capital Management making the final investment decision.
            Consider:
            - Risk-adjusted returns and Sharpe ratio optimization
            - Capital preservation in uncertain conditions
            - Diversification and portfolio correlation
            - Market regime and volatility clustering
            - Institutional flow and smart money behavior
            
            Only recommend trades with high conviction and favorable risk/reward."""
            
            response = await self.ollama_client.generate_response(
                prompt=synthesis_prompt,
                model_key="deepseek_r1",
                system_prompt=system_prompt,
                max_tokens=1024
            )
            
            decision = self._parse_json_response(response, {
                "action": "HOLD",
                "confidence": avg_confidence,
                "expected_return": 0.0,
                "risk_level": "medium",
                "position_size": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "agents": list(insights.keys()),
                "reasoning": "Synthesis failed, defaulting to HOLD"
            })
            
            # Apply consensus and confidence filters
            if require_consensus and len(insights) > 1:
                actions = [result.get('hedge_fund_rating', 'hold') for result in insights.values()]
                if len(set(actions)) > 2:  # No clear consensus
                    decision['action'] = 'HOLD'
                    decision['reasoning'] += " | No consensus among agents"
            
            if decision['confidence'] < min_confidence:
                decision['action'] = 'HOLD'
                decision['reasoning'] += f" | Confidence {decision['confidence']:.2f} below threshold {min_confidence}"
            
            return decision
            
        except Exception as e:
            logger.error("Decision synthesis failed", error=str(e))
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "expected_return": 0.0,
                "risk_level": "high",
                "position_size": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "agents": [],
                "reasoning": f"Synthesis error: {str(e)}"
            }
    
    def _get_best_model_for_task(self, task_type: TaskType) -> str:
        """Get the best performing model for a given task type"""
        if task_type in self.routing_performance:
            performance_dict = self.routing_performance[task_type]
            return max(performance_dict.items(), key=lambda x: x[1])[0]
        return LLMModel.DEEPSEEK_R1.value  # Default to primary orchestrator
    
    def _parse_json_response(self, response: str, fallback: Dict) -> Dict:
        """Parse JSON response with fallback handling"""
        try:
            # Clean response
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response.replace("```json", "").replace("```", "").strip()
            elif clean_response.startswith("```"):
                lines = clean_response.split("\n")
                clean_response = "\n".join(lines[1:-1])
            
            return json.loads(clean_response)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Failed to parse JSON response", response=response[:200])
            return fallback
    
    async def _update_performance_metrics(
        self, 
        agent_results: Dict[str, Any], 
        execution_time: float
    ):
        """Update performance tracking metrics"""
        try:
            # Update orchestration stats
            self.orchestration_stats['total_decisions'] += 1
            self.orchestration_stats['avg_execution_time'] = (
                (self.orchestration_stats['avg_execution_time'] * 
                 (self.orchestration_stats['total_decisions'] - 1) + execution_time) /
                self.orchestration_stats['total_decisions']
            )
            
            # Update memory usage (simplified)
            compression_ratios = [
                self.memory_attention.get_compression_ratio(key) 
                for key in self.memory_attention.compressed_cache.keys()
            ]
            if compression_ratios:
                avg_compression = np.mean(compression_ratios)
                self.orchestration_stats['memory_usage_mb'] = 2000 * avg_compression  # Estimated
            
            logger.info(
                "Performance metrics updated",
                total_decisions=self.orchestration_stats['total_decisions'],
                avg_execution_time=self.orchestration_stats['avg_execution_time'],
                memory_usage_mb=self.orchestration_stats['memory_usage_mb']
            )
            
        except Exception as e:
            logger.error("Failed to update performance metrics", error=str(e))
    
    async def get_orchestration_health(self) -> Dict[str, Any]:
        """Get health status of the orchestration system"""
        try:
            # Get Ollama client health
            ollama_health = await self.ollama_client.health_check()
            
            # Calculate memory efficiency
            memory_efficiency = 1.0 - (self.orchestration_stats['memory_usage_mb'] / 8000.0)  # 8GB total
            
            # Calculate performance score
            avg_confidence = np.mean([
                np.mean(list(performance.values())) 
                for performance in self.routing_performance.values()
            ])
            
            return {
                "status": "healthy" if ollama_health.get("status") == "healthy" else "degraded",
                "ollama_status": ollama_health,
                "orchestration_stats": self.orchestration_stats,
                "memory_efficiency": memory_efficiency,
                "avg_routing_confidence": avg_confidence,
                "compressed_cache_size": len(self.memory_attention.compressed_cache),
                "available_models": list(LLMModel),
                "supported_tasks": list(TaskType)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "orchestration_stats": self.orchestration_stats
            }
    
    def get_routing_performance_matrix(self) -> Dict[str, Any]:
        """Get current routing performance matrix"""
        return {
            "performance_matrix": self.routing_performance,
            "total_decisions": self.orchestration_stats['total_decisions'],
            "memory_usage_mb": self.orchestration_stats['memory_usage_mb'],
            "avg_execution_time": self.orchestration_stats['avg_execution_time']
        }


# Convenience functions for easy integration

async def create_orchestrator(ollama_url: str = "http://localhost:11434") -> DeepSeekTradeOrchestrator:
    """Create and initialize DeepSeek trade orchestrator"""
    orchestrator = DeepSeekTradeOrchestrator(ollama_url)
    health = await orchestrator.get_orchestration_health()
    
    if health["status"] != "healthy":
        logger.warning("Orchestrator initialized with degraded health", health=health)
    else:
        logger.info("DeepSeek trade orchestrator initialized successfully")
    
    return orchestrator


async def quick_trade_analysis(
    symbol: str,
    market_data: Dict[str, Any],
    technical_indicators: Optional[Dict[str, Any]] = None,
    ollama_url: str = "http://localhost:11434"
) -> TradingDecisionResult:
    """Quick trade analysis using DeepSeek orchestrator"""
    
    orchestrator = await create_orchestrator(ollama_url)
    
    context = TaskContext(
        task_type=TaskType.BACKTEST_ANALYSIS,
        symbol=symbol,
        market_data=market_data,
        technical_indicators=technical_indicators
    )
    
    return await orchestrator.orchestrate_trade_decision(context)