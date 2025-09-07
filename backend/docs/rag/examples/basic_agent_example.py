"""
Basic Agent Intelligence Infrastructure Example

This example demonstrates how to set up and use the complete Agent Intelligence
Infrastructure for a trading agent that can:

1. Store and retrieve memories
2. Search knowledge base for trading insights
3. Execute trading tools
4. Build intelligent context for decisions
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import all the infrastructure components
from app.rag.services.memory_manager import (
    AgentMemoryManager, Memory, MemoryType, MemoryQuery
)
from app.rag.services.rag_service import (
    TradingRAGService, RAGQuery, DocumentType
)
from app.rag.services.tool_registry import (
    TradingToolRegistry, ToolDefinition, ToolCategory, 
    PermissionLevel, ToolExecutionContext
)
from app.rag.services.context_builder import (
    TradingContextBuilder, ContextComponent, ContextTemplate
)
from app.rag.services.embedding_factory import EmbeddingFactory


class SmartTradingAgent:
    """
    Example trading agent using the Agent Intelligence Infrastructure.
    
    This agent can learn from experience, search for relevant information,
    execute trading tools, and make informed decisions based on comprehensive context.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Infrastructure components (will be initialized)
        self.memory_manager = None
        self.rag_service = None
        self.tool_registry = None
        self.context_builder = None
    
    async def initialize(self):
        """Initialize all infrastructure components."""
        print(f"🚀 Initializing Smart Trading Agent: {self.agent_id}")
        
        # 1. Set up embedding service (using mock for demo)
        print("📊 Setting up embedding service...")
        embedding_service = await EmbeddingFactory.create_service("mock")
        
        # 2. Initialize Memory Manager
        print("🧠 Initializing Memory Manager...")
        self.memory_manager = AgentMemoryManager(
            embedding_service=embedding_service,
            vector_dimension=384
        )
        await self.memory_manager.initialize()
        
        # 3. Initialize RAG Service  
        print("📚 Initializing RAG Service...")
        self.rag_service = TradingRAGService(
            embedding_service=embedding_service,
            memory_manager=self.memory_manager,
            chunk_size=500,
            chunk_overlap=50
        )
        await self.rag_service.initialize()
        
        # 4. Initialize Tool Registry
        print("🔧 Initializing Tool Registry...")
        self.tool_registry = TradingToolRegistry()
        await self.tool_registry.initialize()
        
        # 5. Register custom trading tools
        await self._register_trading_tools()
        
        # 6. Initialize Context Builder
        print("🎯 Initializing Context Builder...")
        self.context_builder = TradingContextBuilder(
            memory_manager=self.memory_manager,
            rag_service=self.rag_service,
            tool_registry=self.tool_registry,
            max_context_tokens=4000
        )
        await self.context_builder.initialize()
        
        # 7. Set up knowledge base
        await self._populate_knowledge_base()
        
        print("✅ Smart Trading Agent initialized successfully!")
    
    async def _register_trading_tools(self):
        """Register custom trading tools."""
        
        # RSI Calculator Tool
        def calculate_rsi(prices: List[float], period: int = 14) -> Dict[str, Any]:
            """Calculate RSI (Relative Strength Index) indicator."""
            if len(prices) < period:
                return {"error": "Insufficient price data"}
            
            # Simple RSI calculation (simplified for demo)
            gains = []
            losses = []
            
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # Generate signal
            if rsi > 70:
                signal = "overbought"
            elif rsi < 30:
                signal = "oversold"
            else:
                signal = "neutral"
            
            return {
                "rsi": round(rsi, 2),
                "signal": signal,
                "avg_gain": round(avg_gain, 4),
                "avg_loss": round(avg_loss, 4)
            }
        
        rsi_tool = ToolDefinition(
            name="rsi_calculator",
            description="Calculate RSI technical indicator for price analysis",
            category=ToolCategory.TECHNICAL_INDICATORS,
            parameters={
                "prices": {
                    "type": "array",
                    "items": {"type": "number"},
                    "required": True,
                    "description": "Array of historical prices"
                },
                "period": {
                    "type": "integer",
                    "default": 14,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "RSI calculation period"
                }
            },
            permissions=[PermissionLevel.CALCULATE_INDICATORS],
            implementation=calculate_rsi,
            async_capable=False
        )
        
        # Market Data Tool (Mock)
        def get_market_data(symbol: str) -> Dict[str, Any]:
            """Get current market data for a symbol."""
            # Mock market data (in real implementation, this would call actual API)
            mock_data = {
                "AAPL": {"price": 150.25, "volume": 45000000, "change": 0.75},
                "GOOGL": {"price": 2800.50, "volume": 1200000, "change": -5.20},
                "MSFT": {"price": 380.75, "volume": 25000000, "change": 2.10},
                "TSLA": {"price": 205.30, "volume": 78000000, "change": -8.45}
            }
            
            if symbol not in mock_data:
                return {"error": f"No data available for symbol {symbol}"}
            
            data = mock_data[symbol]
            return {
                "symbol": symbol,
                "current_price": data["price"],
                "volume": data["volume"],
                "change": data["change"],
                "change_percent": round((data["change"] / (data["price"] - data["change"])) * 100, 2),
                "timestamp": datetime.now().isoformat()
            }
        
        market_data_tool = ToolDefinition(
            name="get_market_data",
            description="Retrieve current market data for a stock symbol",
            category=ToolCategory.MARKET_DATA,
            parameters={
                "symbol": {
                    "type": "string",
                    "required": True,
                    "description": "Stock symbol (e.g., AAPL, GOOGL)"
                }
            },
            permissions=[PermissionLevel.READ_MARKET_DATA],
            implementation=get_market_data,
            async_capable=False
        )
        
        # Risk Assessment Tool
        def assess_position_risk(symbol: str, position_size: int, 
                               current_price: float, portfolio_value: float) -> Dict[str, Any]:
            """Assess risk for a potential position."""
            position_value = position_size * current_price
            position_weight = position_value / portfolio_value
            
            # Risk calculation (simplified)
            if position_weight > 0.2:
                risk_level = "high"
            elif position_weight > 0.1:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "symbol": symbol,
                "position_value": round(position_value, 2),
                "position_weight": round(position_weight * 100, 2),
                "risk_level": risk_level,
                "max_loss_estimate": round(position_value * 0.02, 2),  # 2% max loss estimate
                "recommendations": [
                    f"Position represents {position_weight*100:.1f}% of portfolio",
                    f"Risk level: {risk_level}",
                    "Consider position sizing based on risk tolerance"
                ]
            }
        
        risk_tool = ToolDefinition(
            name="assess_position_risk",
            description="Assess risk for a potential trading position",
            category=ToolCategory.RISK_MANAGEMENT,
            parameters={
                "symbol": {"type": "string", "required": True},
                "position_size": {"type": "integer", "required": True, "minimum": 1},
                "current_price": {"type": "number", "required": True, "minimum": 0},
                "portfolio_value": {"type": "number", "required": True, "minimum": 0}
            },
            permissions=[PermissionLevel.RISK_ANALYSIS],
            implementation=assess_position_risk,
            async_capable=False
        )
        
        # Register all tools
        await self.tool_registry.register_tool(rsi_tool)
        await self.tool_registry.register_tool(market_data_tool)
        await self.tool_registry.register_tool(risk_tool)
        
        print(f"✅ Registered {len([rsi_tool, market_data_tool, risk_tool])} trading tools")
    
    async def _populate_knowledge_base(self):
        """Populate the RAG knowledge base with trading information."""
        
        # Technical analysis document
        tech_analysis = """
        Apple Inc. (AAPL) Technical Analysis - January 15, 2024
        
        Current Price: $150.25
        Support Levels: $148.00, $145.50, $142.00
        Resistance Levels: $152.00, $155.00, $158.50
        
        Technical Indicators:
        - RSI (14): 65.5 - Approaching overbought but still in bullish range
        - MACD: Bullish crossover confirmed with histogram expanding
        - Volume: 45M shares (20% above 20-day average)
        - Bollinger Bands: Price near upper band with band expansion
        - Moving Averages: Price above 20, 50, and 200-day MAs
        
        Pattern Analysis:
        Strong bullish momentum pattern with volume confirmation. 
        The stock has broken through previous resistance at $148 with conviction.
        Similar pattern in October 2023 led to 8% rally over 2 weeks.
        
        Risk Factors:
        - High RSI suggests potential pullback
        - Earnings announcement in 2 weeks could create volatility
        - Broader market conditions remain uncertain
        
        Trading Strategy:
        Consider entry on any pullback to $148-149 support area.
        Target: $155-158 (first resistance zone)
        Stop Loss: Below $145.50 (major support)
        """
        
        await self.rag_service.add_document(
            content=tech_analysis,
            document_type=DocumentType.TECHNICAL_ANALYSIS,
            metadata={
                "symbol": "AAPL",
                "analyst": "TradingBot",
                "confidence": 0.85,
                "date": "2024-01-15",
                "timeframe": "swing_trading"
            }
        )
        
        # Market sentiment document
        sentiment_analysis = """
        Technology Sector Sentiment Analysis - January 2024
        
        Overall Sentiment: Bullish (7.5/10)
        
        Key Factors:
        - Strong Q4 earnings reports from major tech companies
        - AI and automation trends driving growth expectations
        - Federal Reserve policy stabilization reducing uncertainty
        - Institutional buying pressure in mega-cap tech stocks
        
        Sector Rotation:
        Money flowing from defensive sectors into growth and technology.
        FAANG stocks showing renewed investor interest after 2023 correction.
        
        Risk Considerations:
        - Valuation concerns at current levels
        - Regulatory scrutiny on big tech continues
        - Interest rate sensitivity remains elevated
        
        Symbols to Watch:
        - AAPL: Strong fundamentals, new product cycle
        - GOOGL: AI leadership position, reasonable valuation  
        - MSFT: Cloud growth, AI integration
        - TSLA: EV market dynamics, execution risk
        """
        
        await self.rag_service.add_document(
            content=sentiment_analysis,
            document_type=DocumentType.SENTIMENT_ANALYSIS,
            metadata={
                "sector": "technology",
                "sentiment_score": 7.5,
                "date": "2024-01-15",
                "source": "MarketAnalytics"
            }
        )
        
        print("✅ Knowledge base populated with trading documents")
    
    async def store_trading_memory(self, content: str, memory_type: MemoryType, 
                                 metadata: Dict[str, Any], importance: float = 0.5):
        """Store a trading-related memory."""
        memory = Memory(
            memory_id=f"{self.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            agent_id=self.agent_id,
            content=content,
            memory_type=memory_type,
            metadata=metadata,
            importance=importance
        )
        
        success = await self.memory_manager.store_memory(memory)
        if success:
            print(f"💾 Stored memory: {content[:50]}...")
        return success
    
    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a trading symbol.
        
        This method demonstrates the full power of the Agent Intelligence Infrastructure:
        1. Execute tools to get current data
        2. Search memories for past experiences
        3. Query knowledge base for relevant information
        4. Build comprehensive context
        5. Make informed trading decision
        """
        print(f"\n🔍 Analyzing symbol: {symbol}")
        
        # Step 1: Execute tools to get current market data and technical analysis
        execution_context = ToolExecutionContext(
            agent_id=self.agent_id,
            session_id=self.session_id,
            permissions=[
                PermissionLevel.READ_MARKET_DATA,
                PermissionLevel.CALCULATE_INDICATORS,
                PermissionLevel.RISK_ANALYSIS
            ]
        )
        
        # Get market data
        market_data_result = await self.tool_registry.execute_tool(
            tool_name="get_market_data",
            parameters={"symbol": symbol},
            context=execution_context
        )
        
        if not market_data_result.success:
            return {"error": f"Failed to get market data: {market_data_result.error}"}
        
        market_data = market_data_result.data
        current_price = market_data["current_price"]
        
        # Calculate RSI using historical prices (mock data for demo)
        historical_prices = [current_price * (1 + (i-10)/100) for i in range(20)]
        rsi_result = await self.tool_registry.execute_tool(
            tool_name="rsi_calculator",
            parameters={"prices": historical_prices, "period": 14},
            context=execution_context
        )
        
        # Assess position risk (assuming $100k portfolio, 100 shares)
        risk_result = await self.tool_registry.execute_tool(
            tool_name="assess_position_risk",
            parameters={
                "symbol": symbol,
                "position_size": 100,
                "current_price": current_price,
                "portfolio_value": 100000
            },
            context=execution_context
        )
        
        # Step 2: Search memories for past experiences with this symbol
        memory_query = MemoryQuery(
            agent_id=self.agent_id,
            query_text=f"{symbol} trading patterns and outcomes",
            memory_types=[MemoryType.PATTERN, MemoryType.DECISION, MemoryType.OUTCOME],
            limit=5,
            time_range_days=90
        )
        
        relevant_memories = await self.memory_manager.search_memories(memory_query)
        
        # Step 3: Search knowledge base for relevant information
        rag_query = RAGQuery(
            query_text=f"{symbol} technical analysis momentum patterns",
            context={"symbol": symbol, "analysis_type": "technical"},
            document_types=[DocumentType.TECHNICAL_ANALYSIS, DocumentType.SENTIMENT_ANALYSIS],
            max_results=5,
            min_relevance_score=0.6
        )
        
        knowledge_results = await self.rag_service.search(rag_query)
        
        # Step 4: Build comprehensive decision context
        context = await self.context_builder.build_context(
            agent_id=self.agent_id,
            session_id=self.session_id,
            decision_type="symbol_analysis",
            market_data=market_data,
            tool_results=[market_data_result, rsi_result, risk_result]
        )
        
        # Step 5: Synthesize all information for analysis
        analysis_result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "market_data": market_data,
            "technical_indicators": {
                "rsi": rsi_result.data if rsi_result.success else None
            },
            "risk_assessment": risk_result.data if risk_result.success else None,
            "relevant_memories": [
                {
                    "content": mem.memory.content,
                    "similarity": mem.similarity_score,
                    "type": mem.memory.memory_type.value
                }
                for mem in relevant_memories
            ],
            "knowledge_insights": [
                {
                    "content": result.content[:200] + "...",
                    "relevance": result.relevance_score,
                    "type": result.document_type.value
                }
                for result in knowledge_results
            ],
            "context_summary": context.generate_summary()
        }
        
        # Generate trading recommendation based on all available information
        recommendation = self._generate_recommendation(analysis_result)
        analysis_result["recommendation"] = recommendation
        
        # Step 6: Store this analysis as a memory for future reference
        await self.store_trading_memory(
            content=f"Analyzed {symbol}: Price ${current_price}, RSI {rsi_result.data.get('rsi', 'N/A')}, Recommendation: {recommendation['action']}",
            memory_type=MemoryType.DECISION,
            metadata={
                "symbol": symbol,
                "price": current_price,
                "action": recommendation["action"],
                "confidence": recommendation["confidence"]
            },
            importance=0.7
        )
        
        return analysis_result
    
    def _generate_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading recommendation based on comprehensive analysis."""
        symbol = analysis["symbol"]
        market_data = analysis["market_data"]
        rsi_data = analysis["technical_indicators"]["rsi"]
        risk_data = analysis["risk_assessment"]
        
        # Simple decision logic (in practice, this would be much more sophisticated)
        factors = []
        score = 0
        
        # Price momentum factor
        if market_data["change_percent"] > 2:
            factors.append("Strong positive momentum")
            score += 2
        elif market_data["change_percent"] > 0:
            factors.append("Positive momentum")
            score += 1
        elif market_data["change_percent"] < -2:
            factors.append("Strong negative momentum") 
            score -= 2
        else:
            factors.append("Negative momentum")
            score -= 1
        
        # RSI factor
        if rsi_data:
            rsi = rsi_data["rsi"]
            if rsi < 30:
                factors.append("RSI oversold - potential bounce")
                score += 1
            elif rsi > 70:
                factors.append("RSI overbought - potential pullback")
                score -= 1
            else:
                factors.append("RSI neutral")
        
        # Risk factor
        if risk_data:
            risk_level = risk_data["risk_level"]
            if risk_level == "low":
                factors.append("Low position risk")
                score += 1
            elif risk_level == "high":
                factors.append("High position risk")
                score -= 1
        
        # Memory-based factor
        positive_memories = len([m for m in analysis["relevant_memories"] 
                               if "successful" in m["content"].lower() or "profit" in m["content"].lower()])
        if positive_memories > 0:
            factors.append(f"Found {positive_memories} positive historical patterns")
            score += positive_memories * 0.5
        
        # Generate final recommendation
        if score >= 2:
            action = "BUY"
            confidence = min(0.9, 0.5 + score * 0.1)
        elif score <= -2:
            action = "SELL"
            confidence = min(0.9, 0.5 + abs(score) * 0.1)
        else:
            action = "HOLD"
            confidence = 0.5
        
        return {
            "action": action,
            "confidence": round(confidence, 2),
            "score": score,
            "reasoning_factors": factors,
            "recommendation": f"{action} {symbol} with {confidence:.0%} confidence"
        }
    
    async def simulate_trading_session(self):
        """Simulate a complete trading session with multiple symbol analyses."""
        print("\n" + "="*60)
        print("🎯 SMART TRADING AGENT - SIMULATION SESSION")
        print("="*60)
        
        symbols_to_analyze = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        session_results = {}
        
        for symbol in symbols_to_analyze:
            try:
                analysis = await self.analyze_symbol(symbol)
                session_results[symbol] = analysis
                
                if "error" not in analysis:
                    rec = analysis["recommendation"]
                    print(f"📊 {symbol}: {rec['action']} (Confidence: {rec['confidence']:.0%})")
                    print(f"   Reasoning: {', '.join(rec['reasoning_factors'][:2])}")
                else:
                    print(f"❌ {symbol}: {analysis['error']}")
                    
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {str(e)}")
                session_results[symbol] = {"error": str(e)}
        
        # Session summary
        print(f"\n📈 Session Summary:")
        buy_recommendations = [s for s, r in session_results.items() 
                             if r.get("recommendation", {}).get("action") == "BUY"]
        sell_recommendations = [s for s, r in session_results.items() 
                              if r.get("recommendation", {}).get("action") == "SELL"]
        
        print(f"   BUY signals: {', '.join(buy_recommendations) if buy_recommendations else 'None'}")
        print(f"   SELL signals: {', '.join(sell_recommendations) if sell_recommendations else 'None'}")
        
        # Store session summary as memory
        await self.store_trading_memory(
            content=f"Trading session analyzed {len(symbols_to_analyze)} symbols. BUY: {len(buy_recommendations)}, SELL: {len(sell_recommendations)}",
            memory_type=MemoryType.OUTCOME,
            metadata={
                "session_id": self.session_id,
                "symbols_analyzed": symbols_to_analyze,
                "buy_signals": buy_recommendations,
                "sell_signals": sell_recommendations
            },
            importance=0.8
        )
        
        return session_results


async def main():
    """
    Main example function demonstrating the Agent Intelligence Infrastructure.
    """
    print("🚀 Agent Intelligence Infrastructure - Basic Example")
    print("="*60)
    
    try:
        # Create and initialize the smart trading agent
        agent = SmartTradingAgent("demo_agent_001")
        await agent.initialize()
        
        # Run a complete trading session simulation
        results = await agent.simulate_trading_session()
        
        # Display detailed results for one symbol
        if "AAPL" in results and "error" not in results["AAPL"]:
            print(f"\n📋 Detailed Analysis for AAPL:")
            aapl_analysis = results["AAPL"]
            print(f"   Current Price: ${aapl_analysis['market_data']['current_price']}")
            print(f"   Change: {aapl_analysis['market_data']['change_percent']}%")
            if aapl_analysis['technical_indicators']['rsi']:
                print(f"   RSI: {aapl_analysis['technical_indicators']['rsi']['rsi']}")
            print(f"   Risk Level: {aapl_analysis['risk_assessment']['risk_level']}")
            print(f"   Memories Found: {len(aapl_analysis['relevant_memories'])}")
            print(f"   Knowledge Insights: {len(aapl_analysis['knowledge_insights'])}")
        
        print(f"\n✅ Example completed successfully!")
        print(f"   Agent ID: {agent.agent_id}")
        print(f"   Session ID: {agent.session_id}")
        print(f"   Symbols analyzed: {len(results)}")
        
    except Exception as e:
        print(f"❌ Example failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())