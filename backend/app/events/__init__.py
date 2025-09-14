"""
Event-driven architecture components for agent coordination
"""

from .agent_event_bus import AgentEvent, AgentEventBus, AgentEventType, agent_event_bus
from .multi_agent_coordinator import (
    ConflictResolution,
    ConsensusMethod,
    MultiAgentCoordinator,
    multi_agent_coordinator,
)

__all__ = [
    "AgentEventBus",
    "AgentEvent",
    "AgentEventType",
    "agent_event_bus",
    "MultiAgentCoordinator",
    "ConsensusMethod",
    "ConflictResolution",
    "multi_agent_coordinator",
]
