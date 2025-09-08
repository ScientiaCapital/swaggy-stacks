"""
Memory-Efficient Model Serving Strategy for MLX Integration
Addresses memory constraints while providing optimal Chinese LLM support
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelSize(Enum):
    TINY = "tiny"      # <1GB
    SMALL = "small"    # 1-4GB  
    MEDIUM = "medium"  # 4-8GB
    LARGE = "large"    # 8-32GB
    XLARGE = "xlarge"  # 32GB+


class DeploymentMode(Enum):
    NATIVE_MLX = "native_mlx"          # Native macOS with MLX acceleration
    HYBRID = "hybrid"                  # MLX service + Docker containers
    CONTAINER_CPU = "container_cpu"    # Docker with CPU-only fallback
    API_ONLY = "api_only"              # External API calls only


@dataclass
class ModelConfig:
    """Configuration for individual model serving"""
    name: str
    size_category: ModelSize
    memory_requirement_gb: float
    specialization: str
    quantized_available: bool
    mlx_optimized: bool
    api_fallback: bool
    priority: int  # 1=highest, 5=lowest


class MemoryEfficientServingStrategy:
    """
    Design memory-efficient model serving that resolves MLX-Docker conflicts
    while maintaining Chinese LLM capabilities within hardware constraints
    """
    
    def __init__(self, available_memory_gb: float = 8.0, deployment_mode: DeploymentMode = DeploymentMode.HYBRID):
        self.available_memory_gb = available_memory_gb
        self.deployment_mode = deployment_mode
        self.models = self._initialize_model_configs()
        self.selected_models: List[ModelConfig] = []
        
        logger.info(f"🧠 Initializing serving strategy: {deployment_mode.value}, {available_memory_gb}GB")
    
    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """Define available models with memory requirements"""
        return {
            # Efficient Chinese LLMs for constrained environments
            "qwen2.5-coder-3b": ModelConfig(
                name="qwen2.5-coder-3b",
                size_category=ModelSize.SMALL,
                memory_requirement_gb=3.5,
                specialization="quantitative_analysis",
                quantized_available=True,
                mlx_optimized=True,
                api_fallback=True,
                priority=1
            ),
            "yi-6b-chat": ModelConfig(
                name="yi-6b-chat", 
                size_category=ModelSize.MEDIUM,
                memory_requirement_gb=6.8,
                specialization="technical_analysis",
                quantized_available=True,
                mlx_optimized=True,
                api_fallback=True,
                priority=2
            ),
            "glm-4-9b-chat": ModelConfig(
                name="glm-4-9b-chat",
                size_category=ModelSize.MEDIUM, 
                memory_requirement_gb=7.2,
                specialization="risk_management",
                quantized_available=True,
                mlx_optimized=True,
                api_fallback=True,
                priority=2
            ),
            "deepseek-coder-6.7b": ModelConfig(
                name="deepseek-coder-6.7b",
                size_category=ModelSize.MEDIUM,
                memory_requirement_gb=6.2,
                specialization="strategy_development", 
                quantized_available=True,
                mlx_optimized=True,
                api_fallback=False,  # Existing integration
                priority=1
            ),
            
            # Large models for API fallback only
            "yi-34b-chat": ModelConfig(
                name="yi-34b-chat",
                size_category=ModelSize.XLARGE,
                memory_requirement_gb=68.0,
                specialization="comprehensive_analysis",
                quantized_available=False,
                mlx_optimized=False,
                api_fallback=True,
                priority=4
            ),
            "qwen2.5-72b": ModelConfig(
                name="qwen2.5-72b", 
                size_category=ModelSize.XLARGE,
                memory_requirement_gb=144.0,
                specialization="deep_quantitative",
                quantized_available=False,
                mlx_optimized=False,
                api_fallback=True,
                priority=5
            ),
        }
    
    def select_optimal_models(self) -> List[ModelConfig]:
        """
        Select optimal model combination based on memory constraints and deployment mode
        """
        if self.deployment_mode == DeploymentMode.API_ONLY:
            # Use all models via API
            self.selected_models = list(self.models.values())
            logger.info("📡 API-only mode: All models available via external APIs")
            return self.selected_models
        
        # Memory-constrained model selection
        available_memory = self.available_memory_gb * 0.8  # 80% utilization safety margin
        selected = []
        total_memory = 0.0
        
        # Sort by priority and memory efficiency
        candidates = sorted(
            self.models.values(), 
            key=lambda m: (m.priority, m.memory_requirement_gb)
        )
        
        for model in candidates:
            if self.deployment_mode == DeploymentMode.CONTAINER_CPU and not model.api_fallback:
                continue  # Skip local models in container-only mode
                
            if total_memory + model.memory_requirement_gb <= available_memory:
                selected.append(model)
                total_memory += model.memory_requirement_gb
                logger.info(f"✅ Selected {model.name}: {model.memory_requirement_gb}GB ({model.specialization})")
            else:
                if model.api_fallback:
                    # Add as API fallback
                    selected.append(model)
                    logger.info(f"🔄 API fallback {model.name}: {model.specialization}")
                else:
                    logger.warning(f"❌ Skipped {model.name}: Insufficient memory ({model.memory_requirement_gb}GB)")
        
        self.selected_models = selected
        logger.info(f"📊 Total local memory usage: {total_memory:.1f}GB / {available_memory:.1f}GB")
        return selected
    
    def generate_deployment_config(self) -> Dict:
        """Generate deployment configuration based on selected models"""
        if not self.selected_models:
            self.select_optimal_models()
        
        config = {
            "deployment_mode": self.deployment_mode.value,
            "memory_allocation": {
                "total_available_gb": self.available_memory_gb,
                "safety_margin": 0.8,
                "reserved_system_gb": 1.5
            },
            "local_models": [],
            "api_models": [],
            "mlx_config": {},
            "fallback_strategy": {}
        }
        
        for model in self.selected_models:
            model_config = {
                "name": model.name,
                "specialization": model.specialization,
                "memory_gb": model.memory_requirement_gb,
                "priority": model.priority
            }
            
            if (self.deployment_mode in [DeploymentMode.NATIVE_MLX, DeploymentMode.HYBRID] and 
                model.memory_requirement_gb <= self.available_memory_gb):
                config["local_models"].append(model_config)
                
                if model.mlx_optimized:
                    config["mlx_config"][model.name] = {
                        "quantization": "4bit" if model.quantized_available else "none",
                        "cache_size_gb": min(2.0, model.memory_requirement_gb * 0.3),
                        "batch_size": 1 if model.memory_requirement_gb > 6 else 2
                    }
            else:
                config["api_models"].append(model_config)
        
        # Configure deployment-specific settings
        if self.deployment_mode == DeploymentMode.HYBRID:
            config["architecture"] = {
                "mlx_service_port": 8001,
                "docker_backend_port": 8000,
                "communication": "http_api",
                "load_balancing": "weighted_round_robin"
            }
        
        # Fallback strategy for high-priority models
        config["fallback_strategy"] = {
            "timeout_seconds": 10,
            "retry_attempts": 2,
            "fallback_order": [m.name for m in sorted(self.selected_models, key=lambda x: x.priority)]
        }
        
        return config
    
    def get_model_routing_strategy(self) -> Dict[str, str]:
        """Define which model handles which type of analysis"""
        routing = {}
        
        for model in self.selected_models:
            if "quantitative" in model.specialization.lower():
                routing.update({
                    "mathematical_analysis": model.name,
                    "statistical_modeling": model.name,
                    "risk_calculation": model.name
                })
            elif "technical" in model.specialization.lower():
                routing.update({
                    "pattern_recognition": model.name,
                    "chart_analysis": model.name, 
                    "technical_indicators": model.name
                })
            elif "risk" in model.specialization.lower():
                routing.update({
                    "risk_assessment": model.name,
                    "portfolio_analysis": model.name,
                    "drawdown_control": model.name
                })
            elif "strategy" in model.specialization.lower():
                routing.update({
                    "strategy_development": model.name,
                    "backtesting": model.name,
                    "optimization": model.name
                })
        
        return routing
    
    def estimate_inference_performance(self) -> Dict:
        """Estimate inference performance for selected models"""
        performance = {}
        
        for model in self.selected_models:
            # Performance estimates based on model size and deployment
            if model.mlx_optimized and self.deployment_mode != DeploymentMode.CONTAINER_CPU:
                # MLX acceleration estimates
                tokens_per_sec = {
                    ModelSize.SMALL: 150,
                    ModelSize.MEDIUM: 80, 
                    ModelSize.LARGE: 35,
                    ModelSize.XLARGE: 15
                }.get(model.size_category, 50)
                
                latency_ms = {
                    ModelSize.SMALL: 100,
                    ModelSize.MEDIUM: 200,
                    ModelSize.LARGE: 400, 
                    ModelSize.XLARGE: 800
                }.get(model.size_category, 300)
            else:
                # CPU fallback performance
                tokens_per_sec = {
                    ModelSize.SMALL: 25,
                    ModelSize.MEDIUM: 12,
                    ModelSize.LARGE: 5,
                    ModelSize.XLARGE: 2
                }.get(model.size_category, 10)
                
                latency_ms = {
                    ModelSize.SMALL: 500,
                    ModelSize.MEDIUM: 1200,
                    ModelSize.LARGE: 3000,
                    ModelSize.XLARGE: 8000
                }.get(model.size_category, 2000)
            
            performance[model.name] = {
                "tokens_per_second": tokens_per_sec,
                "latency_ms": latency_ms,
                "throughput_requests_per_minute": min(60, tokens_per_sec * 2),
                "concurrent_requests": 1 if model.size_category in [ModelSize.LARGE, ModelSize.XLARGE] else 2
            }
        
        return performance


def create_serving_strategy(memory_gb: float = 8.0, mode: str = "hybrid") -> Dict:
    """
    Factory function to create optimal serving strategy
    """
    deployment_mode = DeploymentMode(mode)
    strategy = MemoryEfficientServingStrategy(memory_gb, deployment_mode)
    
    strategy.select_optimal_models()
    config = strategy.generate_deployment_config()
    routing = strategy.get_model_routing_strategy()
    performance = strategy.estimate_inference_performance()
    
    return {
        "deployment_config": config,
        "model_routing": routing,
        "performance_estimates": performance,
        "selected_models": [m.name for m in strategy.selected_models],
        "total_local_models": len([m for m in strategy.selected_models if m.memory_requirement_gb <= memory_gb]),
        "total_api_models": len([m for m in strategy.selected_models if m.memory_requirement_gb > memory_gb])
    }


if __name__ == "__main__":
    # Example usage
    import json
    
    # Test different configurations
    configs = [
        ("8GB M1 MacBook", 8.0, "hybrid"),
        ("16GB M2 MacBook", 16.0, "native_mlx"), 
        ("Docker Only", 8.0, "container_cpu"),
        ("API Only", 4.0, "api_only")
    ]
    
    for name, memory, mode in configs:
        print(f"\n{'='*50}")
        print(f"Configuration: {name} ({memory}GB, {mode})")
        print(f"{'='*50}")
        
        strategy = create_serving_strategy(memory, mode)
        print(f"Selected models: {strategy['selected_models']}")
        print(f"Local models: {strategy['total_local_models']}")
        print(f"API models: {strategy['total_api_models']}")
        print(json.dumps(strategy['model_routing'], indent=2))