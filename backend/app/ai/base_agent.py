"""
Base AI Agent - Common functionality for all trading agents

Eliminates duplicate patterns and provides shared functionality
"""

import json
import os
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

import structlog

from .ollama_client import OllamaClient

logger = structlog.get_logger()


class BaseAIAgent(ABC):
    """
    Abstract base class for all AI trading agents

    Provides common functionality:
    - Prompt loading and management
    - JSON response parsing with fallbacks
    - Error handling and logging
    - Agent state tracking
    """

    def __init__(self, ollama_client: OllamaClient, agent_type: str, prompt_filename: str):
        self.ollama_client = ollama_client
        self.agent_type = agent_type
        self.system_prompt = self._load_prompt(prompt_filename)

        # Agent state tracking
        self.last_execution = None
        self.execution_count = 0
        self.error_count = 0

        logger.info(f"Initialized {self.__class__.__name__}", agent_type=agent_type)

    def _load_prompt(self, filename: str) -> str:
        """Load system prompt from file with fallback"""
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), "prompts", filename)
            with open(prompt_path, "r") as f:
                content = f.read().strip()
                logger.debug(f"Loaded prompt from {filename}", length=len(content))
                return content
        except Exception as e:
            logger.error("Failed to load prompt", filename=filename, error=str(e))
            return self._get_default_prompt()

    @abstractmethod
    def _get_default_prompt(self) -> str:
        """Get default system prompt if file loading fails"""
        pass

    async def _generate_response(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """Generate AI response with error handling and tracking"""
        try:
            self.execution_count += 1
            self.last_execution = datetime.now()

            response = await self.ollama_client.generate_response(
                prompt,
                model_key=self.agent_type,
                system_prompt=self.system_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

            logger.debug("Generated AI response",
                        agent_type=self.agent_type,
                        prompt_length=len(prompt),
                        response_length=len(response))

            return response

        except Exception as e:
            self.error_count += 1
            logger.error("AI response generation failed",
                        agent_type=self.agent_type,
                        error=str(e))
            raise

    def _parse_json_response(self, response: str, default_values: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON response with robust fallback handling"""
        try:
            # Clean response - remove markdown code blocks if present
            clean_response = response.strip()

            # Handle markdown code blocks
            if clean_response.startswith("```"):
                lines = clean_response.split("\n")
                # Find JSON content between code block markers
                if len(lines) > 2:
                    clean_response = "\n".join(lines[1:-1])
                    # Handle specific language markers
                    if clean_response.startswith(("json", "JSON")):
                        clean_response = "\n".join(lines[2:-1])

            # Additional cleaning for common AI response patterns
            clean_response = clean_response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:].strip()
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3].strip()

            # Parse JSON
            parsed = json.loads(clean_response)

            # Validate that we got a dictionary
            if not isinstance(parsed, dict):
                logger.warning("Parsed JSON is not a dictionary", type=type(parsed).__name__)
                return default_values

            # Merge with defaults to ensure all required fields exist
            result = {**default_values, **parsed}

            logger.debug("Successfully parsed JSON response",
                        fields=list(result.keys()))

            return result

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON response",
                          error=str(e),
                          response_preview=response[:200])
            return default_values
        except Exception as e:
            logger.error("Unexpected error parsing JSON response",
                        error=str(e))
            return default_values

    def _validate_confidence(self, confidence: Any) -> float:
        """Validate and normalize confidence value"""
        try:
            conf = float(confidence)
            return max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            logger.warning("Invalid confidence value", value=confidence)
            return 0.0

    def _validate_choice(self, value: Any, valid_choices: list, default: str) -> str:
        """Validate value is in valid choices, return default if not"""
        if str(value).lower() in [choice.lower() for choice in valid_choices]:
            return str(value).lower()
        else:
            logger.warning("Invalid choice value",
                          value=value,
                          valid_choices=valid_choices)
            return default

    def _build_standard_prompt_template(
        self,
        symbol: str,
        data_sections: Dict[str, Any],
        instruction: str,
        json_schema: Dict[str, Any]
    ) -> str:
        """Build standardized prompt template"""
        sections = []

        # Header
        sections.append(f"Analyze {symbol} with the following data:")
        sections.append("")

        # Data sections
        for section_name, section_data in data_sections.items():
            sections.append(f"{section_name}:")
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    sections.append(f"- {key}: {value}")
            elif isinstance(section_data, list):
                for item in section_data:
                    sections.append(f"- {item}")
            else:
                sections.append(f"{section_data}")
            sections.append("")

        # Instruction
        sections.append(instruction)
        sections.append("")

        # JSON schema
        sections.append("Provide your response in this exact JSON format:")
        sections.append(json.dumps(json_schema, indent=2))

        return "\n".join(sections)

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent execution statistics"""
        return {
            'agent_type': self.agent_type,
            'execution_count': self.execution_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.execution_count, 1),
            'last_execution': self.last_execution.isoformat() if self.last_execution else None
        }

    @abstractmethod
    async def process(self, *args, **kwargs) -> Any:
        """Main processing method - must be implemented by subclasses"""
        pass