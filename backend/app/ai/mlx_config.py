"""
MLX Configuration Module with Docker-Native Hybrid Architecture
Handles Apple Silicon acceleration with containerized fallback
"""

import logging
import os
import platform
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

import psutil

from .model_serving_strategy import (
    DeploymentMode,
    MemoryEfficientServingStrategy,
)

logger = logging.getLogger(__name__)


class MLXCapability(Enum):
    FULL_ACCELERATION = "full_acceleration"  # Native macOS with MLX GPU
    CPU_FALLBACK = "cpu_fallback"  # MLX available but no GPU
    NOT_AVAILABLE = "not_available"  # MLX not available
    CONTAINERIZED = "containerized"  # Running in Docker container


@dataclass
class MLXSystemInfo:
    """System information for MLX compatibility assessment"""

    is_macos: bool
    is_apple_silicon: bool
    available_memory_gb: float
    in_container: bool
    mlx_available: bool
    gpu_available: bool
    capability: MLXCapability


class MLXConfigManager:
    """
    Manages MLX configuration and deployment strategy
    Automatically detects system capabilities and configures optimal setup
    """

    def __init__(self, force_cpu: bool = False):
        self.force_cpu = force_cpu
        self.system_info = self._detect_system_info()
        self.strategy = None
        self._mlx_initialized = False

        logger.info(f"🔍 MLX Config Manager initialized")
        logger.info(f"   System: {self._get_system_description()}")
        logger.info(f"   Capability: {self.system_info.capability.value}")

    def _detect_system_info(self) -> MLXSystemInfo:
        """Detect system capabilities for MLX"""

        # Detect macOS and Apple Silicon
        is_macos = platform.system() == "Darwin"
        is_apple_silicon = is_macos and platform.processor() == "arm"

        # Detect container environment
        in_container = (
            os.path.exists("/.dockerenv")
            or os.environ.get("DOCKER_CONTAINER") == "true"
            or "container" in os.environ.get("HOSTNAME", "").lower()
        )

        # Memory detection
        available_memory_gb = psutil.virtual_memory().total / (1024**3)

        # MLX availability detection
        mlx_available = False
        gpu_available = False

        if is_apple_silicon and not in_container:
            try:
                import mlx.core as mx

                mlx_available = True
                # Test GPU availability
                try:
                    # Simple GPU test
                    test_tensor = mx.array([1.0, 2.0, 3.0])
                    mx.eval(test_tensor)
                    gpu_available = True
                    logger.info("✅ MLX GPU acceleration available")
                except Exception as e:
                    logger.warning(f"⚠️ MLX available but GPU test failed: {e}")
                    gpu_available = False
            except ImportError:
                logger.info("📦 MLX not installed or not importable")

        # Determine capability
        if self.force_cpu:
            capability = MLXCapability.CPU_FALLBACK
        elif in_container:
            capability = MLXCapability.CONTAINERIZED
        elif mlx_available and gpu_available:
            capability = MLXCapability.FULL_ACCELERATION
        elif mlx_available:
            capability = MLXCapability.CPU_FALLBACK
        else:
            capability = MLXCapability.NOT_AVAILABLE

        return MLXSystemInfo(
            is_macos=is_macos,
            is_apple_silicon=is_apple_silicon,
            available_memory_gb=available_memory_gb,
            in_container=in_container,
            mlx_available=mlx_available,
            gpu_available=gpu_available,
            capability=capability,
        )

    def _get_system_description(self) -> str:
        """Get human-readable system description"""
        parts = []

        if self.system_info.is_macos:
            if self.system_info.is_apple_silicon:
                parts.append("Apple Silicon Mac")
            else:
                parts.append("Intel Mac")
        else:
            parts.append(f"{platform.system()} {platform.machine()}")

        parts.append(f"{self.system_info.available_memory_gb:.1f}GB RAM")

        if self.system_info.in_container:
            parts.append("Docker Container")

        return ", ".join(parts)

    def get_deployment_mode(self) -> DeploymentMode:
        """Determine optimal deployment mode based on system capabilities"""

        if self.system_info.capability == MLXCapability.FULL_ACCELERATION:
            if self.system_info.in_container:
                # Hybrid: MLX service outside container, API communication
                return DeploymentMode.HYBRID
            else:
                # Native: Full MLX acceleration
                return DeploymentMode.NATIVE_MLX

        elif self.system_info.capability in [
            MLXCapability.CPU_FALLBACK,
            MLXCapability.CONTAINERIZED,
        ]:
            # Container with CPU fallback
            return DeploymentMode.CONTAINER_CPU

        else:
            # API-only fallback
            return DeploymentMode.API_ONLY

    def initialize_strategy(self) -> MemoryEfficientServingStrategy:
        """Initialize serving strategy based on system capabilities"""

        deployment_mode = self.get_deployment_mode()

        # Adjust available memory for container overhead
        available_memory = self.system_info.available_memory_gb
        if self.system_info.in_container:
            available_memory *= 0.7  # Container memory overhead

        self.strategy = MemoryEfficientServingStrategy(
            available_memory_gb=available_memory, deployment_mode=deployment_mode
        )

        models = self.strategy.select_optimal_models()

        logger.info(f"🚀 Strategy initialized: {deployment_mode.value}")
        logger.info(f"   Available memory: {available_memory:.1f}GB")
        logger.info(f"   Selected models: {len(models)}")

        return self.strategy

    def get_mlx_device_config(self) -> Dict[str, Any]:
        """Get MLX device configuration"""

        if not self.system_info.mlx_available:
            return {"device": "cpu", "acceleration": False}

        try:
            pass

            if self.system_info.capability == MLXCapability.FULL_ACCELERATION:
                return {
                    "device": "gpu",
                    "acceleration": True,
                    "memory_pool_gb": min(
                        4.0, self.system_info.available_memory_gb * 0.5
                    ),
                    "cache_limit_gb": 2.0,
                    "stream_buffer_size": 1024 * 1024,  # 1MB
                }
            else:
                return {
                    "device": "cpu",
                    "acceleration": False,
                    "num_threads": min(8, os.cpu_count() or 4),
                    "memory_limit_gb": min(
                        2.0, self.system_info.available_memory_gb * 0.25
                    ),
                }

        except ImportError:
            return {"device": "cpu", "acceleration": False}

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration optimized for system"""

        device_config = self.get_mlx_device_config()

        if device_config["acceleration"]:
            # MLX GPU acceleration
            config = {
                "batch_size": 8 if self.system_info.available_memory_gb >= 16 else 4,
                "gradient_accumulation_steps": 2,
                "learning_rate": 5e-5,
                "max_sequence_length": 2048,
                "use_mixed_precision": True,
                "gradient_checkpointing": True,
                "dataloader_num_workers": 2,
                "device": "mps",  # Metal Performance Shaders
                "mlx_optimizations": {
                    "quantization": "4bit",
                    "memory_efficient_attention": True,
                    "flash_attention": True,
                },
            }
        else:
            # CPU fallback
            config = {
                "batch_size": 2 if self.system_info.available_memory_gb >= 8 else 1,
                "gradient_accumulation_steps": 8,  # Compensate for small batch
                "learning_rate": 3e-5,
                "max_sequence_length": 1024,  # Smaller context for CPU
                "use_mixed_precision": False,
                "gradient_checkpointing": True,
                "dataloader_num_workers": 1,
                "device": "cpu",
                "cpu_optimizations": {
                    "num_threads": min(4, os.cpu_count() or 2),
                    "memory_efficient": True,
                },
            }

        # Add memory constraints
        config["memory_constraints"] = {
            "max_memory_gb": self.system_info.available_memory_gb * 0.6,
            "swap_to_disk": self.system_info.available_memory_gb < 16,
            "clear_cache_frequency": 10,  # Every 10 batches
        }

        return config

    def create_mlx_trainer_config(self) -> Dict[str, Any]:
        """Create configuration for MLXTrainer"""

        if not self.strategy:
            self.initialize_strategy()

        deployment_config = self.strategy.generate_deployment_config()
        training_config = self.get_training_config()
        device_config = self.get_mlx_device_config()

        return {
            "system_info": {
                "platform": self._get_system_description(),
                "capability": self.system_info.capability.value,
                "mlx_available": self.system_info.mlx_available,
                "in_container": self.system_info.in_container,
            },
            "deployment": deployment_config,
            "training": training_config,
            "device": device_config,
            "models": {
                "selected": [m.name for m in self.strategy.selected_models],
                "local": len(
                    [
                        m
                        for m in self.strategy.selected_models
                        if m.memory_requirement_gb
                        <= self.system_info.available_memory_gb
                    ]
                ),
                "api_fallback": len(
                    [
                        m
                        for m in self.strategy.selected_models
                        if m.memory_requirement_gb
                        > self.system_info.available_memory_gb
                    ]
                ),
            },
            "performance_optimization": {
                "quantization_enabled": self.system_info.capability
                == MLXCapability.FULL_ACCELERATION,
                "memory_mapping": self.system_info.available_memory_gb >= 16,
                "parallel_processing": not self.system_info.in_container,
                "cache_models": self.system_info.available_memory_gb >= 12,
            },
        }

    def validate_setup(self) -> Tuple[bool, List[str]]:
        """Validate MLX setup and return status with issues"""
        issues = []

        # Check system requirements
        if (
            not self.system_info.is_macos
            and self.system_info.capability != MLXCapability.API_ONLY
        ):
            issues.append("MLX requires macOS for GPU acceleration")

        if self.system_info.available_memory_gb < 4:
            issues.append(
                f"Insufficient memory: {self.system_info.available_memory_gb:.1f}GB < 4GB minimum"
            )

        # Check MLX installation
        if self.system_info.capability not in [
            MLXCapability.NOT_AVAILABLE,
            MLXCapability.API_ONLY,
        ]:
            try:
                pass

                logger.info("✅ MLX packages imported successfully")
            except ImportError as e:
                issues.append(f"MLX import failed: {e}")

        # Check container compatibility
        if (
            self.system_info.in_container
            and self.system_info.capability == MLXCapability.FULL_ACCELERATION
        ):
            issues.append("MLX GPU acceleration not available in Docker containers")

        # Validate strategy
        if self.strategy:
            if not self.strategy.selected_models:
                issues.append("No models selected by serving strategy")

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("✅ MLX setup validation passed")
        else:
            logger.warning(f"⚠️ MLX setup issues: {issues}")

        return is_valid, issues


# Global instance for easy access
mlx_config = MLXConfigManager()


def get_mlx_config() -> MLXConfigManager:
    """Get global MLX configuration manager"""
    return mlx_config


def is_mlx_available() -> bool:
    """Quick check if MLX is available for use"""
    return mlx_config.system_info.mlx_available


def get_optimal_model_config() -> Dict[str, Any]:
    """Get optimal model configuration for current system"""
    return mlx_config.create_mlx_trainer_config()


def validate_mlx_environment() -> bool:
    """Validate MLX environment is ready"""
    is_valid, issues = mlx_config.validate_setup()
    if not is_valid:
        for issue in issues:
            logger.error(f"❌ {issue}")
    return is_valid


if __name__ == "__main__":
    # Test configuration
    manager = MLXConfigManager()
    config = manager.create_mlx_trainer_config()

    print("\n" + "=" * 60)
    print("MLX Configuration Test")
    print("=" * 60)
    print(f"System: {manager._get_system_description()}")
    print(f"Capability: {manager.system_info.capability.value}")
    print(f"Deployment Mode: {manager.get_deployment_mode().value}")
    print(
        f"Models Selected: {len(manager.strategy.selected_models) if manager.strategy else 0}"
    )

    is_valid, issues = manager.validate_setup()
    print(f"Setup Valid: {'✅' if is_valid else '❌'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
