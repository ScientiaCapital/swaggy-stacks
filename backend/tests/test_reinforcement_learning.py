"""
Comprehensive tests for the Continuous Learning System

Tests reinforcement learning algorithms, memory management,
learning orchestration, and trading integration.
"""

import asyncio
import pytest
import numpy as np
import torch
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

# Import learning components
from app.ml.reinforcement.q_learning import (
    QLearningConfig,
    QLearningAgent,
    TradingQLearningAgent
)
from app.ml.reinforcement.policy_gradient import (
    PolicyGradientConfig,
    PolicyGradientAgent,
    TradingPolicyGradientAgent
)
from app.ml.reinforcement.value_functions import (
    ValueFunctionConfig,
    TemporalDifferenceLearner
)
from app.ml.reinforcement.enhanced_memory import (
    EnhancedAgentMemory,
    SARSATuple
)
from app.ml.reinforcement.learning_orchestrator import (
    LearningOrchestrator,
    LearningMode,
    AlgorithmType
)
from app.ml.reinforcement.trading_integration import (
    ContinuousLearningTradingSystem
)


class TestQLearning:
    """Test Q-Learning algorithms"""

    def test_dqn_initialization(self):
        """Test DQN agent initialization"""
        config = QLearningConfig(
            state_dim=10,
            action_dim=5,
            use_double_dqn=True,
            use_dueling_dqn=True
        )
        agent = QLearningAgent(config)

        assert agent.config.state_dim == 10
        assert agent.config.action_dim == 5
        assert agent.config.use_double_dqn == True
        assert agent.config.use_dueling_dqn == True

    def test_action_selection(self):
        """Test epsilon-greedy action selection"""
        config = QLearningConfig(
            state_dim=10,
            action_dim=5,
            epsilon_start=1.0,
            exploration_strategy="epsilon_greedy"
        )
        agent = QLearningAgent(config)

        # Test exploration (epsilon=1.0)
        state = np.random.randn(10)
        action = agent.select_action(state, training=True)
        assert 0 <= action < 5

        # Test exploitation (epsilon=0)
        agent.epsilon = 0
        action = agent.select_action(state, training=True)
        assert 0 <= action < 5

    def test_experience_storage(self):
        """Test experience replay buffer"""
        config = QLearningConfig(
            state_dim=10,
            action_dim=5,
            buffer_size=1000
        )
        agent = QLearningAgent(config)

        # Store experiences
        for _ in range(100):
            state = np.random.randn(10)
            action = np.random.randint(5)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = np.random.choice([True, False])

            agent.store_experience(state, action, reward, next_state, done)

        assert len(agent.replay_buffer) == 100

    def test_training_step(self):
        """Test Q-learning training step"""
        config = QLearningConfig(
            state_dim=10,
            action_dim=5,
            batch_size=32,
            min_buffer_size=50
        )
        agent = QLearningAgent(config)

        # Fill buffer
        for _ in range(60):
            state = np.random.randn(10)
            action = np.random.randint(5)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = False

            agent.store_experience(state, action, reward, next_state, done)

        # Train
        loss = agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)

    def test_trading_q_learning(self):
        """Test trading-specific Q-learning agent"""
        config = QLearningConfig(state_dim=128, action_dim=5)
        agent = TradingQLearningAgent(config)

        # Test state representation
        market_data = {
            'price': 100,
            'price_change_1h': 0.01,
            'price_change_24h': 0.02,
            'volume_ratio': 1.2,
            'volatility': 0.015,
            'rsi': 55,
            'macd_signal': 0.001,
            'bb_position': 0.5,
            'market_regime': 'bull'
        }

        state = agent.create_state_representation(market_data)
        assert state.shape == (128,)
        assert state.dtype == np.float32

        # Test trading decision
        decision = agent.make_trading_decision(market_data)
        assert 'action' in decision
        assert 'confidence' in decision
        assert decision['action'] in agent.ACTIONS.values()


class TestPolicyGradient:
    """Test Policy Gradient methods"""

    def test_ppo_initialization(self):
        """Test PPO agent initialization"""
        config = PolicyGradientConfig(
            state_dim=10,
            action_dim=5,
            algorithm="ppo",
            ppo_epochs=10,
            ppo_clip_ratio=0.2
        )
        agent = PolicyGradientAgent(config)

        assert agent.config.algorithm == "ppo"
        assert agent.config.ppo_epochs == 10
        assert hasattr(agent, 'actor_critic')
        assert hasattr(agent, 'old_actor_critic')

    def test_trajectory_storage(self):
        """Test trajectory buffer management"""
        config = PolicyGradientConfig(
            state_dim=10,
            action_dim=5,
            algorithm="a2c",
            trajectory_length=32
        )
        agent = PolicyGradientAgent(config)

        # Store transitions
        for _ in range(35):
            state = np.random.randn(10)
            action = np.random.randint(5)
            reward = np.random.randn()
            log_prob = np.random.randn()
            value = np.random.randn()
            done = False

            agent.store_transition(state, action, reward, log_prob, value, done)

        # Should have created one trajectory
        assert len(agent.trajectory_buffer) == 1

    def test_advantage_calculation(self):
        """Test GAE advantage calculation"""
        config = PolicyGradientConfig(
            state_dim=10,
            action_dim=5,
            use_gae=True,
            gae_lambda=0.95
        )
        agent = PolicyGradientAgent(config)

        # Create mock trajectory
        rewards = torch.FloatTensor([1, 0, -1, 1, 0])
        values = torch.FloatTensor([0.5, 0.3, 0.1, 0.6, 0.2])
        dones = torch.FloatTensor([0, 0, 0, 0, 1])

        from app.ml.reinforcement.policy_gradient import Trajectory
        trajectory = Trajectory(
            states=torch.randn(5, 10),
            actions=torch.randint(0, 5, (5,)),
            rewards=rewards,
            log_probs=torch.randn(5),
            values=values,
            dones=dones
        )

        returns, advantages = agent.compute_returns_and_advantages(trajectory)
        assert returns.shape == (5,)
        assert advantages.shape == (5,)


class TestValueFunctions:
    """Test Value Function and TD Learning"""

    def test_td_learner_initialization(self):
        """Test TD learner initialization"""
        config = ValueFunctionConfig(
            state_dim=10,
            method="td_lambda",
            use_eligibility_traces=True
        )
        learner = TemporalDifferenceLearner(config)

        assert learner.config.method == "td_lambda"
        assert learner.config.use_eligibility_traces == True
        assert hasattr(learner, 'eligibility_traces')

    def test_td0_update(self):
        """Test TD(0) update"""
        config = ValueFunctionConfig(
            state_dim=10,
            method="td0",
            use_neural_network=True
        )
        learner = TemporalDifferenceLearner(config)

        state = np.random.randn(10)
        reward = 1.0
        next_state = np.random.randn(10)
        done = False

        td_error = learner.td0_update(state, reward, next_state, done)
        assert isinstance(td_error, float)

    def test_eligibility_traces(self):
        """Test eligibility trace management"""
        config = ValueFunctionConfig(
            state_dim=10,
            method="td_lambda",
            use_eligibility_traces=True,
            lambda_param=0.9
        )
        learner = TemporalDifferenceLearner(config)

        # Update traces
        for i in range(5):
            state = np.random.randn(10)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = False

            td_error = learner.td_lambda_update(state, reward, next_state, done)
            assert isinstance(td_error, float)

        # Check traces exist
        assert len(learner.eligibility_traces.traces) > 0


class TestEnhancedMemory:
    """Test Enhanced Memory System"""

    def test_sarsa_storage(self):
        """Test SARSA tuple storage"""
        memory = EnhancedAgentMemory(
            max_experiences=1000,
            n_step=5,
            gamma=0.99
        )

        # Store SARSA tuples
        for i in range(10):
            state = np.random.randn(10)
            action = np.random.randint(5)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = i == 9

            experience_id = memory.store_sarsa(
                agent_id="test_agent",
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )

            assert experience_id is not None

        assert len(memory.sarsa_buffer) == 10

    def test_prioritized_sampling(self):
        """Test prioritized experience replay"""
        memory = EnhancedAgentMemory(
            max_experiences=1000,
            priority_alpha=0.6,
            priority_beta=0.4
        )

        # Fill buffer with varying rewards
        for i in range(100):
            state = np.random.randn(10)
            action = np.random.randint(5)
            reward = i / 10  # Increasing rewards
            next_state = np.random.randn(10)

            memory.store_sarsa(
                agent_id="test_agent",
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=False
            )

        # Sample with prioritization
        batch = memory.sample_batch(32, prioritized=True)
        assert len(batch) == 32

        # Check importance weights
        assert all('importance_weight' in t.info for t in batch)

    def test_episode_management(self):
        """Test episode buffer management"""
        memory = EnhancedAgentMemory()

        # Complete an episode
        total_reward = 0
        for i in range(20):
            state = np.random.randn(10)
            action = np.random.randint(5)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = i == 19

            total_reward += reward

            memory.store_sarsa(
                agent_id="test_agent",
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )

        # Check episode was finalized
        assert len(memory.episode_buffer) == 1
        episode = memory.episode_buffer[0]
        assert episode.episode_length == 20
        assert abs(episode.total_reward - total_reward) < 0.01

    def test_n_step_returns(self):
        """Test n-step return calculation"""
        memory = EnhancedAgentMemory(n_step=3, gamma=0.9)

        # Create sequence of experiences
        sarsa_tuples = []
        for i in range(5):
            sarsa_tuples.append(SARSATuple(
                state=np.random.randn(10),
                action=np.random.randint(5),
                reward=float(i),  # Rewards: 0, 1, 2, 3, 4
                next_state=np.random.randn(10),
                next_action=np.random.randint(5),
                done=False,
                info={},
                timestamp=datetime.now()
            ))

        # Calculate 3-step return
        n_step_return = memory.compute_n_step_return(sarsa_tuples, n=3)

        # Expected: 0 + 0.9*1 + 0.81*2 = 0 + 0.9 + 1.62 = 2.52
        expected = 0 + 0.9 * 1 + 0.81 * 2
        assert abs(n_step_return - expected) < 0.01


@pytest.mark.asyncio
class TestLearningOrchestrator:
    """Test Learning Orchestrator"""

    async def test_orchestrator_initialization(self):
        """Test orchestrator setup"""
        orchestrator = LearningOrchestrator(
            enable_q_learning=True,
            enable_policy_gradient=True,
            enable_td_learning=True
        )

        assert 'dqn' in orchestrator.agents
        assert 'ppo' in orchestrator.agents
        assert 'a2c' in orchestrator.agents
        assert orchestrator.td_learner is not None

    async def test_experience_processing(self):
        """Test experience processing through all systems"""
        orchestrator = LearningOrchestrator()

        state = np.random.randn(128)
        action = 2
        reward = 1.0
        next_state = np.random.randn(128)
        done = False

        await orchestrator.process_experience(
            agent_id="test",
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info={'log_prob': -0.5, 'value': 0.3}
        )

        # Check experience was stored
        assert orchestrator.step_count == 1
        assert len(orchestrator.memory.sarsa_buffer) == 1

    async def test_learning_mode_adaptation(self):
        """Test learning mode adaptation"""
        orchestrator = LearningOrchestrator()

        # Test with good performance
        episode_stats = {
            'success_rate': 0.85,
            'avg_reward': 10
        }
        await orchestrator._adapt_learning_mode(episode_stats)
        assert orchestrator.current_mode == LearningMode.EXPLOITATION

        # Test with poor performance
        episode_stats = {
            'success_rate': 0.2,
            'avg_reward': -5
        }
        await orchestrator._adapt_learning_mode(episode_stats)
        assert orchestrator.current_mode == LearningMode.EXPLORATION

    async def test_trading_decision(self):
        """Test trading decision making"""
        orchestrator = LearningOrchestrator()

        market_data = {
            'price': 100,
            'price_change_24h': 0.02,
            'volume_ratio': 1.1,
            'volatility': 0.01,
            'rsi': 60
        }

        decision = await orchestrator.make_trading_decision(market_data)

        assert 'action' in decision
        assert 'confidence' in decision
        assert decision['action'] in ['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY']

    async def test_checkpoint_save_load(self, tmp_path):
        """Test checkpoint persistence"""
        orchestrator = LearningOrchestrator(
            checkpoint_path=str(tmp_path)
        )

        # Process some experiences
        for _ in range(10):
            await orchestrator.process_experience(
                agent_id="test",
                state=np.random.randn(128),
                action=np.random.randint(5),
                reward=np.random.randn(),
                next_state=np.random.randn(128),
                done=False
            )

        # Save checkpoint
        await orchestrator.save_checkpoint()

        # Create new orchestrator and load
        new_orchestrator = LearningOrchestrator(
            checkpoint_path=str(tmp_path)
        )

        # Find and load checkpoint
        import os
        checkpoints = [f for f in os.listdir(tmp_path) if f.startswith('orchestrator_')]
        assert len(checkpoints) > 0

        timestamp = checkpoints[0].replace('orchestrator_', '').replace('.json', '')
        await new_orchestrator.load_checkpoint(timestamp)

        assert new_orchestrator.step_count == 10


@pytest.mark.asyncio
class TestTradingIntegration:
    """Test Trading System Integration"""

    async def test_system_initialization(self):
        """Test integrated system setup"""
        # Mock dependencies
        trading_manager = Mock()
        risk_manager = Mock()

        system = ContinuousLearningTradingSystem(
            trading_manager=trading_manager,
            risk_manager=risk_manager,
            enable_learning=True,
            enable_paper_trading=True
        )

        assert system.learning_orchestrator is not None
        assert system.enable_paper_trading == True

    async def test_symbol_learning_enablement(self):
        """Test enabling learning for symbols"""
        trading_manager = Mock()
        risk_manager = Mock()

        system = ContinuousLearningTradingSystem(
            trading_manager=trading_manager,
            risk_manager=risk_manager
        )

        await system.enable_learning_for_symbol(
            "AAPL",
            algorithm=AlgorithmType.DQN,
            config={'special_param': 123}
        )

        assert "AAPL" in system.learning_enabled_symbols
        assert system.symbol_configs["AAPL"]['algorithm'] == AlgorithmType.DQN

    async def test_market_data_processing(self):
        """Test market data processing for learning"""
        trading_manager = Mock()
        risk_manager = AsyncMock()
        risk_manager.check_trade_risk = AsyncMock(return_value={'approved': True})

        system = ContinuousLearningTradingSystem(
            trading_manager=trading_manager,
            risk_manager=risk_manager,
            enable_paper_trading=True
        )

        await system.enable_learning_for_symbol("AAPL")

        market_data = {
            'symbol': 'AAPL',
            'price': 150,
            'volume': 1000000,
            'rsi': 55,
            'volatility': 0.02
        }

        await system._process_market_data('AAPL', market_data)

        # Check market state was cached
        assert system.market_state_cache['AAPL'] == market_data

    async def test_paper_trading_execution(self):
        """Test paper trading execution"""
        trading_manager = Mock()
        risk_manager = Mock()

        system = ContinuousLearningTradingSystem(
            trading_manager=trading_manager,
            risk_manager=risk_manager,
            enable_paper_trading=True
        )

        order = await system._execute_paper_trade(
            symbol='AAPL',
            side='buy',
            quantity=100,
            price=150
        )

        assert order['paper_trade'] == True
        assert order['symbol'] == 'AAPL'
        assert order['side'] == 'buy'
        assert order['quantity'] == 100
        assert order['price'] == 150

    async def test_reward_calculation(self):
        """Test reward calculation for RL"""
        trading_manager = Mock()
        risk_manager = Mock()

        system = ContinuousLearningTradingSystem(
            trading_manager=trading_manager,
            risk_manager=risk_manager
        )

        position = {
            'has_position': True,
            'entry_price': 100,
            'quantity': 100,
            'entry_time': datetime.now()
        }

        market_data = {
            'price': 105,  # 5% profit
            'volatility': 0.02
        }

        reward = system._calculate_reward('AAPL', position, market_data)

        # Should be positive for profit
        assert reward > 0

    async def test_performance_metrics(self):
        """Test performance metrics calculation"""
        trading_manager = Mock()
        risk_manager = Mock()

        system = ContinuousLearningTradingSystem(
            trading_manager=trading_manager,
            risk_manager=risk_manager,
            enable_paper_trading=True
        )

        # Add some paper trades
        system.performance_tracker['AAPL'] = {
            'trades': [
                {'side': 'buy', 'price': 100, 'quantity': 100},
                {'side': 'sell', 'price': 105, 'quantity': 100}
            ],
            'total_pnl': 500
        }

        metrics = await system._calculate_performance_metrics()

        assert 'paper_trading' in metrics
        assert metrics['paper_trading']['total_pnl'] == 500


class TestPerformance:
    """Test performance and optimization"""

    def test_memory_efficiency(self):
        """Test memory usage is within limits"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large memory buffer
        memory = EnhancedAgentMemory(max_experiences=100000)

        # Fill with experiences
        for _ in range(10000):
            memory.store_sarsa(
                agent_id="test",
                state=np.random.randn(128),
                action=np.random.randint(5),
                reward=np.random.randn(),
                next_state=np.random.randn(128),
                done=False
            )

        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory

        # Should use less than 500MB for 10k experiences
        assert memory_increase < 500

    def test_training_speed(self):
        """Test training speed on M1 MacBook"""
        import time

        config = QLearningConfig(
            state_dim=128,
            action_dim=5,
            batch_size=64
        )
        agent = QLearningAgent(config)

        # Fill buffer
        for _ in range(1000):
            agent.store_experience(
                np.random.randn(128),
                np.random.randint(5),
                np.random.randn(),
                np.random.randn(128),
                False
            )

        # Measure training speed
        start_time = time.time()
        for _ in range(100):
            agent.train_step()
        elapsed = time.time() - start_time

        # Should complete 100 training steps in less than 5 seconds
        assert elapsed < 5.0

    def test_device_utilization(self):
        """Test PyTorch device utilization"""
        import torch

        # Check device availability
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            assert str(device) == "mps"

            # Test tensor operations on device
            tensor = torch.randn(1000, 1000, device=device)
            result = torch.matmul(tensor, tensor)
            assert result.device.type == "mps"
        else:
            # CPU fallback
            device = torch.device("cpu")
            assert str(device) == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])