"""
Reinforcement Learning Module for Trading System

Provides comprehensive RL algorithms for autonomous trading:
- Deep Q-Learning (DQN, Double DQN, Dueling DQN)
- Policy Gradient Methods (REINFORCE, A2C, PPO)
- Value Functions and TD Learning
- Learning Orchestrator for coordinated training
"""

from .q_learning import (
    QLearningConfig,
    QLearningAgent,
    TradingQLearningAgent,
    DQN,
    PrioritizedReplayBuffer
)

from .policy_gradient import (
    PolicyGradientConfig,
    PolicyGradientAgent,
    TradingPolicyGradientAgent,
    PolicyNetwork,
    ValueNetwork,
    ActorCriticNetwork
)

from .value_functions import (
    ValueFunctionConfig,
    TemporalDifferenceLearner,
    ValueFunction,
    ActionValueFunction,
    EligibilityTraces
)

__all__ = [
    # Q-Learning
    'QLearningConfig',
    'QLearningAgent',
    'TradingQLearningAgent',
    'DQN',
    'PrioritizedReplayBuffer',

    # Policy Gradient
    'PolicyGradientConfig',
    'PolicyGradientAgent',
    'TradingPolicyGradientAgent',
    'PolicyNetwork',
    'ValueNetwork',
    'ActorCriticNetwork',

    # Value Functions
    'ValueFunctionConfig',
    'TemporalDifferenceLearner',
    'ValueFunction',
    'ActionValueFunction',
    'EligibilityTraces'
]