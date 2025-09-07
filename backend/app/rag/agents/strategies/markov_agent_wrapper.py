"""
Markov Trading Agent - Compatibility Wrapper
This file maintains backwards compatibility while using the consolidated strategy system
"""

from app.rag.agents.consolidated_strategy_agent import create_markov_agent, ConsolidatedStrategyAgent

# Create the Markov agent using the consolidated system
def MarkovTradingAgent(**kwargs):
    """
    Create a Markov trading agent using the consolidated strategy system
    Maintains backwards compatibility
    """
    return create_markov_agent(**kwargs)

# For direct class access
MarkovAgent = ConsolidatedStrategyAgent

# Export for backwards compatibility
__all__ = ['MarkovTradingAgent', 'MarkovAgent']