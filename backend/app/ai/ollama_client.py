"""
Ollama client for AI model interaction optimized for M1 MacBook with 8GB RAM
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class ModelConfig:
    """Configuration for different models optimized for M1 performance"""

    name: str
    context_length: int
    memory_usage_mb: int
    use_case: str
    temperature: float = 0.7


class OllamaClient:
    """
    Optimized Ollama client for M1 MacBook with intelligent model management
    """

    # Models optimized for 8GB M1 MacBook
    MODELS = {
        # Public models for general trading tasks
        "analyst": ModelConfig(
            name="llama3.2:3b",
            context_length=4096,
            memory_usage_mb=2048,
            use_case="market_analysis",
            temperature=0.3,
        ),
        "risk": ModelConfig(
            name="phi3:mini",
            context_length=2048,
            memory_usage_mb=1536,
            use_case="risk_assessment",
            temperature=0.1,
        ),
        "strategist": ModelConfig(
            name="qwen2.5-coder:3b",
            context_length=8192,
            memory_usage_mb=2560,
            use_case="strategy_generation",
            temperature=0.5,
        ),
        "chat": ModelConfig(
            name="llama3.2:3b",
            context_length=4096,
            memory_usage_mb=2048,
            use_case="conversational",
            temperature=0.7,
        ),
    }

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.current_model = None
        self.context_history = {}
        self.max_context_tokens = 2048  # Conservative for 8GB RAM
        self.active_models = set()  # Track which models are loaded

    async def ensure_model_loaded(self, model_key: str) -> bool:
        """Ensure the specified model is loaded and ready"""
        try:
            model_config = self.MODELS.get(model_key)
            if not model_config:
                logger.error("Unknown model key", key=model_key)
                return False

            # Check if model is available
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    logger.error(
                        "Failed to get model list", status=response.status_code
                    )
                    return False

                models = response.json()
                model_names = [model["name"] for model in models.get("models", [])]

                if model_config.name not in model_names:
                    logger.warning(
                        "Model not found, attempting to pull", model=model_config.name
                    )
                    # Note: In production, you'd want to pull models beforehand
                    return False

                self.current_model = model_config
                self.active_models.add(model_key)  # Track active model
                logger.info(
                    "Model ready",
                    model=model_config.name,
                    use_case=model_config.use_case,
                    memory_mb=model_config.memory_usage_mb
                )
                return True

        except Exception as e:
            logger.error(
                "Error ensuring model loaded", error=str(e), model_key=model_key
            )
            return False

    async def generate_response(
        self,
        prompt: str,
        model_key: str = "analyst",
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """Generate response from specified model"""
        try:
            if not await self.ensure_model_loaded(model_key):
                raise Exception(f"Failed to load model: {model_key}")

            model_config = self.MODELS[model_key]

            # Prepare the request
            request_data = {
                "model": model_config.name,
                "prompt": prompt,
                "system": system_prompt or self._get_default_system_prompt(model_key),
                "stream": stream,
                "options": {
                    "temperature": model_config.temperature,
                    "num_ctx": min(
                        model_config.context_length, self.max_context_tokens
                    ),
                    "num_predict": max_tokens or 512,
                },
            }

            # Manage context to prevent memory issues
            context_key = f"{model_key}_{hash(system_prompt or '')}"
            if context_key in self.context_history:
                request_data["context"] = self.context_history[context_key]

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate", json=request_data
                )

                if response.status_code != 200:
                    logger.error(
                        "Ollama request failed",
                        status=response.status_code,
                        text=response.text,
                    )
                    return "Error: Failed to generate response"

                if stream:
                    return await self._handle_stream_response(response, context_key)
                else:
                    result = response.json()

                    # Store context for next interaction (memory management)
                    if (
                        "context" in result and len(self.context_history) < 5
                    ):  # Limit context storage
                        self.context_history[context_key] = result["context"]

                    return result.get("response", "No response generated")

        except Exception as e:
            logger.error("Error generating response", error=str(e), model_key=model_key)
            return f"Error: {str(e)}"

    async def generate_streaming_response(
        self,
        prompt: str,
        model_key: str = "analyst",
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response for real-time interaction"""
        try:
            if not await self.ensure_model_loaded(model_key):
                yield "Error: Failed to load model"
                return

            model_config = self.MODELS[model_key]

            request_data = {
                "model": model_config.name,
                "prompt": prompt,
                "system": system_prompt or self._get_default_system_prompt(model_key),
                "stream": True,
                "options": {
                    "temperature": model_config.temperature,
                    "num_ctx": self.max_context_tokens,
                },
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/generate", json=request_data
                ) as response:

                    if response.status_code != 200:
                        yield f"Error: HTTP {response.status_code}"
                        return

                    async for chunk in response.aiter_lines():
                        if chunk:
                            try:
                                data = json.loads(chunk)
                                if "response" in data:
                                    yield data["response"]
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error("Error in streaming response", error=str(e))
            yield f"Error: {str(e)}"

    async def analyze_market_data(
        self, market_data: Dict, signals: Dict, context: str = ""
    ) -> Dict[str, Any]:
        """Specialized method for market analysis"""

        prompt = f"""
        Analyze this market data and trading signals:
        
        Market Data:
        {json.dumps(market_data, indent=2)}
        
        Current Signals:
        {json.dumps(signals, indent=2)}
        
        Additional Context:
        {context}
        
        Provide analysis in this JSON format:
        {{
            "market_sentiment": "bullish|bearish|neutral",
            "confidence": 0.0-1.0,
            "key_factors": ["factor1", "factor2"],
            "recommendations": ["rec1", "rec2"],
            "risk_level": "low|medium|high",
            "reasoning": "detailed explanation"
        }}
        """

        system_prompt = """You are an expert quantitative analyst specializing in algorithmic trading. 
        Analyze market data objectively and provide actionable insights. Focus on statistical patterns and risk assessment.
        Always respond in valid JSON format."""

        response = await self.generate_response(
            prompt, model_key="analyst", system_prompt=system_prompt, max_tokens=1024
        )

        try:
            # Attempt to parse JSON response
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Fallback to structured text response
            return {
                "market_sentiment": "neutral",
                "confidence": 0.5,
                "key_factors": ["Unable to parse AI response"],
                "recommendations": ["Review AI model output"],
                "risk_level": "medium",
                "reasoning": response,
            }

    def _get_default_system_prompt(self, model_key: str) -> str:
        """Get default system prompt for each model type"""
        prompts = {
            # Original models (backward compatibility)
            "analyst": """You are a senior quantitative analyst for a hedge fund specializing in algorithmic trading. 
            Analyze market data with statistical rigor and provide clear, actionable insights. Be concise and precise.""",
            "risk": """You are a risk management expert focused on portfolio protection and capital preservation. 
            Evaluate trades for potential risks and suggest appropriate safeguards. Prioritize downside protection.""",
            "strategist": """You are a quantitative strategist who develops algorithmic trading strategies. 
            Generate Python code and mathematical models for trading systems. Focus on statistical edge and risk-adjusted returns.""",
            "chat": """You are an intelligent trading assistant. Help users understand their trading system, 
            explain market conditions, and provide guidance in plain English. Be helpful but honest about limitations.""",
            
            - Performance attribution and strategy diagnostics
            - High-frequency trading and low-latency system design
            - Portfolio management and rebalancing algorithms
            Write robust, production-ready code with comprehensive error handling and logging.""",
        }

        return prompts.get(model_key, prompts["chat"])

    async def _handle_stream_response(self, response, context_key: str) -> str:
        """Handle streaming response and collect full text"""
        full_response = ""

        async for chunk in response.aiter_lines():
            if chunk:
                try:
                    data = json.loads(chunk)
                    if "response" in data:
                        full_response += data["response"]
                    if data.get("done", False):
                        # Store context for memory management
                        if "context" in data and len(self.context_history) < 5:
                            self.context_history[context_key] = data["context"]
                        break
                except json.JSONDecodeError:
                    continue

        return full_response

    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama service health and available models"""
        try:
            async with httpx.AsyncClient() as client:
                # Check service
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return {
                        "status": "error",
                        "message": "Ollama service not available",
                    }

                models = response.json()
                available_models = [model["name"] for model in models.get("models", [])]

                # Check our required models
                required_models = [config.name for config in self.MODELS.values()]
                missing_models = [
                    model for model in required_models if model not in available_models
                ]

                return {
                    "status": "healthy" if not missing_models else "partial",
                    "available_models": available_models,
                    "missing_models": missing_models,
                    "memory_usage": self._estimate_memory_usage(),
                    "context_cache_size": len(self.context_history),
                }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _estimate_memory_usage(self) -> int:
        """Estimate current memory usage in MB"""
        if self.current_model:
            return self.current_model.memory_usage_mb
        return 0

    def clear_context_cache(self):
        """Clear context cache to free memory"""
        self.context_history.clear()
        logger.info("Context cache cleared")

    def get_model_info(self, model_key: str) -> Optional[ModelConfig]:
        """Get information about a specific model"""
        return self.MODELS.get(model_key)
    
    def get_total_memory_usage(self) -> int:
        """Get total estimated memory usage of active models in MB"""
        total_memory = 0
        for model_key in self.active_models:
            model_config = self.MODELS.get(model_key)
            if model_config:
                total_memory += model_config.memory_usage_mb
        
        # Add context cache overhead (estimated)
        cache_overhead = len(self.context_history) * 50  # ~50MB per cached context
        return total_memory + cache_overhead
    
    def get_memory_budget_status(self, max_memory_mb: int = 7500) -> Dict[str, Any]:
        """
        Get memory budget status for M1 Mac optimization
        
        Args:
            max_memory_mb: Maximum allowed memory (default 7.5GB for 8GB Mac)
            
        Returns:
            Dictionary with memory status information
        """
        current_usage = self.get_total_memory_usage()
        available_memory = max_memory_mb - current_usage
        
        return {
            "total_usage_mb": current_usage,
            "max_budget_mb": max_memory_mb,
            "available_mb": available_memory,
            "usage_percentage": (current_usage / max_memory_mb) * 100,
            "active_models": len(self.active_models),
            "active_model_keys": list(self.active_models),
            "context_cache_size": len(self.context_history),
            "memory_efficient": current_usage < max_memory_mb,
            "can_load_large_models": available_memory > 4000  # Need ~4GB for largest models
        }
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage by clearing context cache if needed
        
        Returns:
            Dictionary with optimization actions taken
        """
        actions_taken = []
        memory_freed = 0
        
        # Clear old context history if memory is tight
        current_usage = self.get_total_memory_usage()
        if current_usage > 6000:  # If using more than 6GB
            contexts_cleared = len(self.context_history)
            memory_freed += contexts_cleared * 50  # Estimated 50MB per context
            self.context_history.clear()
            actions_taken.append(f"Cleared {contexts_cleared} context caches")
        
        # Limit context history size for future
        if len(self.context_history) > 3:
            oldest_contexts = list(self.context_history.keys())[:-3]
            for context_key in oldest_contexts:
                del self.context_history[context_key]
                memory_freed += 50
            actions_taken.append(f"Removed {len(oldest_contexts)} old contexts")
        
        return {
            "actions_taken": actions_taken,
            "estimated_memory_freed_mb": memory_freed,
            "new_usage_mb": self.get_total_memory_usage(),
            "optimization_successful": len(actions_taken) > 0
        }
    
    def get_available_models(self) -> Dict[str, ModelConfig]:
        """Get all available model configurations"""
        return self.MODELS.copy()
    
    def get_recommended_model_for_task(self, task_type: str) -> Optional[str]:
        """
        Get recommended model for a specific task type
        
        Args:
            task_type: Type of task (e.g., "market_analysis", "risk_assessment")
            
        Returns:
            Recommended model key or None
        """
        task_model_mapping = {
            "market_analysis": "analyst",
            "quantitative_analysis": "strategist", 
            "mathematical_analysis": "strategist",
            "technical_analysis": "analyst",
            "pattern_recognition": "analyst",
            "risk_management": "risk",
            "risk_assessment": "risk",
            "portfolio_analysis": "risk",
            "strategy_implementation": "strategist",
            "coding": "strategist",
            "algorithm_development": "strategist",
            "conversational": "chat"
        }
        
        return task_model_mapping.get(task_type.lower(), "analyst")
