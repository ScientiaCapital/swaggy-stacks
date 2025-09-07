"""
Base Tool Interface for Agent Tools
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ToolParameter:
    """Tool parameter definition"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class AgentTool(ABC):
    """Base class for all agent tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.category = "general"
        self.version = "1.0.0"
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """Get tool parameter definitions"""
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> ToolResult:
        """Validate tool parameters"""
        try:
            param_definitions = {p.name: p for p in self.get_parameters()}
            
            # Check required parameters
            for param_def in param_definitions.values():
                if param_def.required and param_def.name not in parameters:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Required parameter '{param_def.name}' is missing"
                    )
            
            # Check parameter types (basic validation)
            for param_name, param_value in parameters.items():
                if param_name in param_definitions:
                    param_def = param_definitions[param_name]
                    # Basic type checking could be expanded here
                    if param_def.type == "str" and not isinstance(param_value, str):
                        return ToolResult(
                            success=False,
                            data=None,
                            error=f"Parameter '{param_name}' must be a string"
                        )
                    elif param_def.type == "float" and not isinstance(param_value, (int, float)):
                        return ToolResult(
                            success=False,
                            data=None,
                            error=f"Parameter '{param_name}' must be a number"
                        )
                    elif param_def.type == "int" and not isinstance(param_value, int):
                        return ToolResult(
                            success=False,
                            data=None,
                            error=f"Parameter '{param_name}' must be an integer"
                        )
            
            return ToolResult(success=True, data=None)
            
        except Exception as e:
            logger.error(f"Parameter validation error for {self.name}: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Parameter validation failed: {str(e)}"
            )
    
    async def safe_execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Safely execute tool with parameter validation and error handling"""
        try:
            # Validate parameters first
            validation_result = self.validate_parameters(parameters)
            if not validation_result.success:
                return validation_result
            
            # Execute the tool
            result = await self.execute(parameters)
            
            logger.info(f"Tool {self.name} executed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool execution failed: {str(e)}"
            )
    
    def get_info(self) -> Dict[str, Any]:
        """Get tool information"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default
                }
                for p in self.get_parameters()
            ]
        }