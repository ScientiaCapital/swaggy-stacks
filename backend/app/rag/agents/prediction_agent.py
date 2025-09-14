"""
LangChain Prediction Agent
Integrates LLM predictions with LangChain's agent framework for sophisticated workflows
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

from langchain.schema import AIMessage, HumanMessage
from langchain.tools import BaseTool, tool
from langchain.agents import Tool, AgentType, initialize_agent, AgentExecutor
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from app.analysis.llm_predictors import LLMPredictor, get_llm_predictor
from app.ai.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class PredictionTool(BaseTool):
    """LangChain tool wrapper for LLM predictions"""
    name = "llm_prediction"
    description = """
    Provides AI-powered trading predictions using ensemble of Chinese LLMs.
    Input format: 'symbol:SYMBOL,days:N,type:TYPE' where TYPE is price_direction or volatility
    Example: 'symbol:AAPL,days:5,type:price_direction'
    """

    def __init__(self, predictor: LLMPredictor):
        super().__init__()
        self.predictor = predictor

    def _run(self, query: str) -> str:
        """Synchronous wrapper for async prediction"""
        try:
            # Parse query
            params = {}
            for param in query.split(','):
                key, value = param.split(':')
                params[key.strip()] = value.strip()

            symbol = params.get('symbol', 'AAPL')
            days = int(params.get('days', 5))
            pred_type = params.get('type', 'price_direction')

            # For demo, create sample data (in production, would fetch from database)
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")
            data = data.reset_index()
            data.columns = [col.lower() for col in data.columns]

            # Get prediction
            if pred_type == 'price_direction':
                result = asyncio.run(self.predictor.predict_price_direction(
                    symbol, data, horizon_days=days
                ))
            elif pred_type == 'volatility':
                result = asyncio.run(self.predictor.predict_volatility(
                    symbol, data, horizon_days=days
                ))
            else:
                return f"Unknown prediction type: {pred_type}"

            return f"🔮 {symbol} Prediction: {result.prediction_value} (confidence: {result.confidence:.2f}) - {result.reasoning[:150]}..."

        except Exception as e:
            logger.error(f"Error in prediction tool: {e}")
            return f"Error generating prediction: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)


@tool
def market_context_tool(symbol: str) -> str:
    """Extract market context for a symbol"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info

        context = f"{symbol} Market Context: "
        context += f"Sector: {info.get('sector', 'Unknown')}, "
        context += f"Market Cap: ${info.get('marketCap', 0):,}, "
        context += f"P/E: {info.get('trailingPE', 'N/A')}, "
        context += f"52W Range: ${info.get('fiftyTwoWeekLow', 0):.2f} - ${info.get('fiftyTwoWeekHigh', 0):.2f}"

        return context
    except Exception as e:
        return f"Error getting market context for {symbol}: {str(e)}"


@tool
def model_health_tool(query: str) -> str:
    """Check health of Chinese LLM models"""
    try:
        predictor = asyncio.run(get_llm_predictor())
        health = asyncio.run(predictor.get_model_health())

        status = f"🏥 Model Health Status: {health['overall_status'].upper()}\n"
        status += f"Available Models: {health['available_models']}/{health['total_models']}\n"

        for model, details in health['model_details'].items():
            status += f"- {model}: {details['status']} ({details.get('specialization', 'unknown')})\n"

        return status
    except Exception as e:
        return f"Error checking model health: {str(e)}"


class PredictionAgent:
    """
    LangChain agent for sophisticated prediction workflows
    Combines Chinese LLMs with agent reasoning and memory
    """

    def __init__(self):
        self.predictor: Optional[LLMPredictor] = None
        self.agent: Optional[AgentExecutor] = None
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.ollama_client = OllamaClient()

    async def initialize(self):
        """Initialize the prediction agent with tools and memory"""
        # Initialize LLM predictor
        self.predictor = await get_llm_predictor()

        # Create tools
        prediction_tool = PredictionTool(self.predictor)

        tools = [
            Tool(
                name="llm_prediction",
                func=prediction_tool._run,
                description=prediction_tool.description
            ),
            Tool(
                name="market_context",
                func=market_context_tool,
                description="Get market context and fundamental information for a stock symbol"
            ),
            Tool(
                name="model_health",
                func=model_health_tool,
                description="Check the health and availability of Chinese LLM models"
            )
        ]

        # Create LLM for agent reasoning (using one of our Chinese models)
        # We'll use Yi model for conversational agent as it's good at technical analysis

        # Initialize agent with conversational type for memory support
        self.agent = initialize_agent(
            tools=tools,
            llm=self.ollama_client,  # This will need to be adapted for LangChain
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True
        )

        logger.info("🤖 Prediction Agent initialized with LangChain integration")

    async def predict_with_context(
        self,
        symbol: str,
        user_query: str = "",
        horizon_days: int = 5
    ) -> Dict[str, Any]:
        """
        Make predictions with full context and reasoning

        Args:
            symbol: Stock symbol
            user_query: User's specific question or context
            horizon_days: Prediction horizon

        Returns:
            Comprehensive prediction with reasoning
        """
        try:
            if not self.agent:
                await self.initialize()

            # Build comprehensive query for agent
            agent_query = f"""
            I need a comprehensive trading analysis for {symbol}.

            User Context: {user_query if user_query else 'General analysis requested'}

            Please:
            1. Get market context for {symbol}
            2. Generate price direction prediction for {horizon_days} days
            3. Check model health to ensure prediction reliability
            4. Provide actionable insights based on the analysis

            Focus on practical trading implications and risk considerations.
            """

            # Execute agent workflow
            response = await self.agent.arun(input=agent_query)

            # Also get direct LLM prediction for detailed results
            direct_prediction = await self.predictor.predict_price_direction(
                symbol=symbol,
                historical_data=self._get_sample_data(symbol),
                horizon_days=horizon_days
            )

            return {
                "agent_analysis": response,
                "direct_prediction": {
                    "prediction": direct_prediction.prediction_value,
                    "confidence": direct_prediction.confidence,
                    "reasoning": direct_prediction.reasoning,
                    "model_contributions": len(direct_prediction.model_contributions),
                },
                "symbol": symbol,
                "horizon_days": horizon_days,
                "timestamp": datetime.now().isoformat(),
                "conversation_memory": self.memory.load_memory_variables({})
            }

        except Exception as e:
            logger.error(f"Error in predict_with_context: {e}")
            return {
                "error": str(e),
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }

    def _get_sample_data(self, symbol: str) -> pd.DataFrame:
        """Get sample data for predictions (in production, would use proper data source)"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")
            data = data.reset_index()
            data.columns = [col.lower() for col in data.columns]
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            # Return empty DataFrame with required columns
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    async def analyze_portfolio_symbols(
        self,
        symbols: List[str],
        user_context: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze multiple symbols for portfolio insights

        Args:
            symbols: List of stock symbols
            user_context: User's portfolio context or questions

        Returns:
            Portfolio-wide analysis
        """
        try:
            if not self.agent:
                await self.initialize()

            results = {}
            for symbol in symbols:
                prediction = await self.predictor.predict_price_direction(
                    symbol=symbol,
                    historical_data=self._get_sample_data(symbol)
                )

                results[symbol] = {
                    "prediction": prediction.prediction_value,
                    "confidence": prediction.confidence,
                    "reasoning": prediction.reasoning[:100] + "..."  # Truncated
                }

            # Agent analysis of portfolio
            portfolio_query = f"""
            Portfolio Analysis Request:

            Symbols: {', '.join(symbols)}
            Context: {user_context}

            Based on individual predictions for each symbol, provide:
            1. Overall portfolio outlook
            2. Risk diversification assessment
            3. Correlation considerations
            4. Recommended actions (buy/sell/hold/rebalance)

            Individual predictions: {json.dumps(results, indent=2)}
            """

            portfolio_analysis = await self.agent.arun(input=portfolio_query)

            return {
                "portfolio_analysis": portfolio_analysis,
                "individual_predictions": results,
                "symbols_analyzed": len(symbols),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in analyze_portfolio_symbols: {e}")
            return {
                "error": str(e),
                "symbols": symbols,
                "timestamp": datetime.now().isoformat()
            }

    async def get_agent_conversation_history(self) -> Dict[str, Any]:
        """Get the agent's conversation history"""
        try:
            return self.memory.load_memory_variables({})
        except Exception as e:
            return {"error": str(e)}

    async def clear_agent_memory(self):
        """Clear agent conversation memory"""
        self.memory.clear()
        logger.info("Agent memory cleared")


# Global agent instance
_agent_instance: Optional[PredictionAgent] = None


async def get_prediction_agent() -> PredictionAgent:
    """Get the global prediction agent instance"""
    global _agent_instance

    if _agent_instance is None:
        _agent_instance = PredictionAgent()
        await _agent_instance.initialize()

    return _agent_instance


# Convenience functions for API integration
async def quick_symbol_analysis(symbol: str, user_query: str = "") -> Dict[str, Any]:
    """Quick analysis for a single symbol"""
    agent = await get_prediction_agent()
    return await agent.predict_with_context(symbol, user_query)


async def portfolio_analysis(symbols: List[str], context: str = "") -> Dict[str, Any]:
    """Portfolio-wide analysis"""
    agent = await get_prediction_agent()
    return await agent.analyze_portfolio_symbols(symbols, context)


# Example LangChain chain for prediction reasoning
class PredictionReasoningChain(LLMChain):
    """
    Custom LangChain chain for prediction reasoning
    Uses Chinese LLMs for enhanced analysis
    """

    @classmethod
    def create_reasoning_chain(cls, ollama_client: OllamaClient):
        """Create a reasoning chain for predictions"""

        prompt = PromptTemplate(
            input_variables=["prediction_data", "market_context", "user_query"],
            template="""
作为一个专业的量化交易分析师，请基于以下信息提供深度分析：

预测数据: {prediction_data}
市场背景: {market_context}
用户问题: {user_query}

请提供：
1. 预测结果的置信度分析
2. 关键风险因素识别
3. 具体的交易建议
4. 潜在的市场催化剂

请用专业但易懂的语言回答。
            """
        )

        return cls(llm=ollama_client, prompt=prompt)


if __name__ == "__main__":
    # Example usage
    async def test_prediction_agent():
        agent = await get_prediction_agent()

        # Test single symbol analysis
        result = await agent.predict_with_context(
            symbol="AAPL",
            user_query="Is Apple a good buy for a growth portfolio?",
            horizon_days=10
        )

        print("Single Symbol Analysis:")
        print(json.dumps(result, indent=2, default=str))

        # Test portfolio analysis
        portfolio_result = await agent.analyze_portfolio_symbols(
            symbols=["AAPL", "GOOGL", "MSFT"],
            user_context="Tech-focused growth portfolio"
        )

        print("\nPortfolio Analysis:")
        print(json.dumps(portfolio_result, indent=2, default=str))

    # Run test
    asyncio.run(test_prediction_agent())