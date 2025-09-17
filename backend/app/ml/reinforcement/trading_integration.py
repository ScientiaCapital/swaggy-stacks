"""
Trading System Integration for Continuous Learning

Integrates the reinforcement learning system with existing:
- Trading Manager
- Market Data Feeds
- WebSocket Events
- AI Agents
- Risk Management
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np

import structlog
from app.core.config import settings
from app.core.exceptions import TradingError

# Import learning components
from .learning_orchestrator import LearningOrchestrator, LearningMode, AlgorithmType

# Import existing trading components
from app.trading.trading_manager import TradingManager
from app.trading.risk_manager import RiskManager
from app.websockets.manager import WebSocketManager
from app.events.market_events import MarketEvent

logger = structlog.get_logger()


class ContinuousLearningTradingSystem:
    """
    Integration layer for continuous learning in the trading system.

    Seamlessly connects reinforcement learning with existing trading infrastructure
    while maintaining backward compatibility.
    """

    def __init__(
        self,
        trading_manager: TradingManager,
        risk_manager: RiskManager,
        websocket_manager: Optional[WebSocketManager] = None,
        enable_learning: bool = True,
        enable_paper_trading: bool = True,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize the continuous learning trading system.

        Args:
            trading_manager: Existing trading manager instance
            risk_manager: Risk management system
            websocket_manager: Optional WebSocket manager for real-time updates
            enable_learning: Enable continuous learning
            enable_paper_trading: Use paper trading for safe learning
            checkpoint_dir: Directory for model checkpoints
        """

        self.trading_manager = trading_manager
        self.risk_manager = risk_manager
        self.websocket_manager = websocket_manager
        self.enable_learning = enable_learning
        self.enable_paper_trading = enable_paper_trading

        # Initialize learning orchestrator if enabled
        if enable_learning:
            self.learning_orchestrator = LearningOrchestrator(
                enable_q_learning=True,
                enable_policy_gradient=True,
                enable_td_learning=True,
                enable_unsupervised=True,
                enable_evolution=True,
                memory_path=checkpoint_dir,
                checkpoint_path=checkpoint_dir
            )
        else:
            self.learning_orchestrator = None

        # State tracking
        self.current_positions = {}
        self.market_state_cache = {}
        self.learning_enabled_symbols = set()
        self.performance_tracker = {}

        # Event handlers
        self.event_handlers = {
            'market_update': self._handle_market_update,
            'order_filled': self._handle_order_filled,
            'position_closed': self._handle_position_closed,
            'risk_alert': self._handle_risk_alert
        }

        # Learning configuration per symbol
        self.symbol_configs = {}

        logger.info(
            "Continuous Learning Trading System initialized",
            learning_enabled=enable_learning,
            paper_trading=enable_paper_trading
        )

    async def start(self):
        """Start the continuous learning system"""

        logger.info("Starting continuous learning trading system")

        # Register WebSocket handlers if available
        if self.websocket_manager:
            await self._register_websocket_handlers()

        # Load checkpoint if exists
        if self.learning_orchestrator:
            await self._load_latest_checkpoint()

        # Start learning loop
        if self.enable_learning:
            asyncio.create_task(self._learning_loop())

        # Start performance monitoring
        asyncio.create_task(self._performance_monitoring_loop())

    async def _register_websocket_handlers(self):
        """Register handlers for WebSocket events"""

        if not self.websocket_manager:
            return

        # Register market data handler
        self.websocket_manager.register_handler(
            'market_data',
            self._handle_websocket_market_data
        )

        # Register trading event handler
        self.websocket_manager.register_handler(
            'trading_event',
            self._handle_websocket_trading_event
        )

    async def _handle_websocket_market_data(self, data: Dict[str, Any]):
        """Handle real-time market data from WebSocket"""

        symbol = data.get('symbol')
        if symbol and symbol in self.learning_enabled_symbols:
            await self._process_market_data(symbol, data)

    async def _handle_websocket_trading_event(self, event: Dict[str, Any]):
        """Handle trading events from WebSocket"""

        event_type = event.get('type')
        if event_type in self.event_handlers:
            await self.event_handlers[event_type](event)

    async def enable_learning_for_symbol(
        self,
        symbol: str,
        algorithm: Optional[AlgorithmType] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Enable continuous learning for a specific symbol.

        Args:
            symbol: Trading symbol
            algorithm: Preferred learning algorithm
            config: Symbol-specific configuration
        """

        self.learning_enabled_symbols.add(symbol)
        self.symbol_configs[symbol] = config or {}

        if algorithm:
            self.symbol_configs[symbol]['algorithm'] = algorithm

        logger.info(f"Learning enabled for {symbol}", config=config)

    async def _process_market_data(self, symbol: str, market_data: Dict[str, Any]):
        """Process market data for learning"""

        # Update market state cache
        self.market_state_cache[symbol] = market_data

        if not self.learning_orchestrator or not self.enable_learning:
            return

        # Convert market data to state representation
        state = await self._create_state_representation(symbol, market_data)

        # Get current position for symbol
        position = self.current_positions.get(symbol, {})

        # Make trading decision using learned policy
        decision = await self.learning_orchestrator.make_trading_decision(
            market_data,
            algorithm=self.symbol_configs.get(symbol, {}).get('algorithm')
        )

        # Apply risk management
        if decision['action'] != 'HOLD':
            decision = await self._apply_risk_management(symbol, decision)

        # Execute decision if confident enough
        if decision['confidence'] > 0.6:
            await self._execute_trading_decision(symbol, decision, market_data)

        # Store experience for learning
        if position.get('has_position'):
            reward = self._calculate_reward(symbol, position, market_data)
            await self._store_learning_experience(
                symbol, state, decision, reward, market_data
            )

    async def _create_state_representation(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> np.ndarray:
        """Create state representation for learning"""

        features = []

        # Market features
        features.extend([
            market_data.get('price', 0),
            market_data.get('volume', 0),
            market_data.get('bid', 0),
            market_data.get('ask', 0),
            market_data.get('spread', 0)
        ])

        # Technical indicators
        features.extend([
            market_data.get('rsi', 50) / 100,
            market_data.get('macd', 0),
            market_data.get('bb_upper', 0),
            market_data.get('bb_lower', 0),
            market_data.get('ema_20', 0)
        ])

        # Position information
        position = self.current_positions.get(symbol, {})
        features.extend([
            1 if position.get('has_position') else 0,
            position.get('quantity', 0),
            position.get('entry_price', 0),
            position.get('unrealized_pnl', 0),
            position.get('holding_time', 0)
        ])

        # Market microstructure
        features.extend([
            market_data.get('order_imbalance', 0),
            market_data.get('trade_intensity', 0),
            market_data.get('volatility', 0)
        ])

        # Pad to standard size
        state_dim = 128  # Match orchestrator configuration
        if len(features) < state_dim:
            features.extend([0] * (state_dim - len(features)))
        else:
            features = features[:state_dim]

        return np.array(features, dtype=np.float32)

    async def _apply_risk_management(
        self,
        symbol: str,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply risk management to trading decision"""

        # Check risk limits
        risk_check = await self.risk_manager.check_trade_risk(
            symbol=symbol,
            action=decision['action'],
            quantity=decision.get('quantity', 100)
        )

        if not risk_check['approved']:
            # Override to HOLD if risk check fails
            decision['action'] = 'HOLD'
            decision['risk_override'] = True
            decision['risk_reason'] = risk_check.get('reason', 'Risk limit exceeded')

        return decision

    async def _execute_trading_decision(
        self,
        symbol: str,
        decision: Dict[str, Any],
        market_data: Dict[str, Any]
    ):
        """Execute trading decision"""

        action = decision['action']

        # Map RL actions to trading actions
        action_map = {
            'STRONG_BUY': ('buy', 200),
            'BUY': ('buy', 100),
            'HOLD': (None, 0),
            'SELL': ('sell', 100),
            'STRONG_SELL': ('sell', 200)
        }

        trade_action, quantity = action_map.get(action, (None, 0))

        if trade_action and quantity > 0:
            # Use paper trading if enabled
            if self.enable_paper_trading:
                order = await self._execute_paper_trade(
                    symbol, trade_action, quantity, market_data['price']
                )
            else:
                # Execute real trade through trading manager
                order = await self.trading_manager.execute_trade(
                    symbol=symbol,
                    side=trade_action,
                    quantity=quantity,
                    order_type='market'
                )

            # Update position tracking
            await self._update_position(symbol, order)

            logger.info(
                f"Executed {trade_action} for {symbol}",
                quantity=quantity,
                confidence=decision['confidence'],
                paper_trading=self.enable_paper_trading
            )

    async def _execute_paper_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float
    ) -> Dict[str, Any]:
        """Execute paper trade for safe learning"""

        # Simulate order execution
        order = {
            'order_id': f"paper_{datetime.now().timestamp()}",
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'status': 'filled',
            'filled_at': datetime.now(),
            'paper_trade': True
        }

        # Track paper trading performance
        if symbol not in self.performance_tracker:
            self.performance_tracker[symbol] = {
                'trades': [],
                'total_pnl': 0,
                'win_rate': 0
            }

        self.performance_tracker[symbol]['trades'].append(order)

        return order

    async def _update_position(self, symbol: str, order: Dict[str, Any]):
        """Update position tracking"""

        if symbol not in self.current_positions:
            self.current_positions[symbol] = {
                'has_position': False,
                'quantity': 0,
                'entry_price': 0,
                'entry_time': None
            }

        position = self.current_positions[symbol]

        if order['side'] == 'buy':
            if not position['has_position']:
                position['has_position'] = True
                position['entry_price'] = order['price']
                position['entry_time'] = order.get('filled_at', datetime.now())
            position['quantity'] += order['quantity']

        elif order['side'] == 'sell':
            position['quantity'] -= order['quantity']
            if position['quantity'] <= 0:
                position['has_position'] = False
                position['quantity'] = 0

    def _calculate_reward(
        self,
        symbol: str,
        position: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate reward for reinforcement learning"""

        if not position.get('has_position'):
            return 0

        # Calculate unrealized P&L
        entry_price = position.get('entry_price', 0)
        current_price = market_data.get('price', entry_price)
        quantity = position.get('quantity', 0)

        pnl = (current_price - entry_price) * quantity
        pnl_percent = (pnl / (entry_price * quantity)) * 100 if entry_price > 0 else 0

        # Risk-adjusted reward
        volatility = market_data.get('volatility', 0.01)
        sharpe_ratio = pnl_percent / (volatility * 100) if volatility > 0 else pnl_percent

        # Time penalty for holding too long
        holding_time = (datetime.now() - position.get('entry_time', datetime.now())).seconds / 3600
        time_penalty = min(holding_time * 0.01, 0.1)

        reward = sharpe_ratio - time_penalty

        return np.clip(reward, -10, 10)

    async def _store_learning_experience(
        self,
        symbol: str,
        state: np.ndarray,
        decision: Dict[str, Any],
        reward: float,
        market_data: Dict[str, Any]
    ):
        """Store experience for learning"""

        if not self.learning_orchestrator:
            return

        # Get next state
        next_state = await self._create_state_representation(symbol, market_data)

        # Map decision to action index
        action_map = {
            'STRONG_SELL': 0,
            'SELL': 1,
            'HOLD': 2,
            'BUY': 3,
            'STRONG_BUY': 4
        }
        action = action_map.get(decision['action'], 2)

        # Check if episode is done (position closed)
        done = not self.current_positions.get(symbol, {}).get('has_position', False)

        # Store experience
        await self.learning_orchestrator.process_experience(
            agent_id=f"trading_{symbol}",
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info={
                'symbol': symbol,
                'confidence': decision.get('confidence', 0),
                'market_data': market_data
            }
        )

    async def _handle_market_update(self, event: Dict[str, Any]):
        """Handle market update event"""

        symbol = event.get('symbol')
        if symbol in self.learning_enabled_symbols:
            await self._process_market_data(symbol, event.get('data', {}))

    async def _handle_order_filled(self, event: Dict[str, Any]):
        """Handle order filled event"""

        symbol = event.get('symbol')
        order = event.get('order')

        if symbol and order:
            await self._update_position(symbol, order)

    async def _handle_position_closed(self, event: Dict[str, Any]):
        """Handle position closed event"""

        symbol = event.get('symbol')
        pnl = event.get('realized_pnl', 0)

        # Store final reward for episode
        if symbol in self.learning_enabled_symbols and self.learning_orchestrator:
            # Create terminal experience
            state = await self._create_state_representation(
                symbol,
                self.market_state_cache.get(symbol, {})
            )

            await self.learning_orchestrator.process_experience(
                agent_id=f"trading_{symbol}",
                state=state,
                action=2,  # HOLD
                reward=pnl / 100,  # Normalized P&L
                next_state=state,
                done=True,
                info={'symbol': symbol, 'final_pnl': pnl}
            )

        # Clear position
        if symbol in self.current_positions:
            del self.current_positions[symbol]

    async def _handle_risk_alert(self, event: Dict[str, Any]):
        """Handle risk alert event"""

        alert_type = event.get('type')
        severity = event.get('severity', 'low')

        if severity in ['high', 'critical'] and self.learning_orchestrator:
            # Switch to safe learning mode
            self.learning_orchestrator.current_mode = LearningMode.SAFE

            logger.warning(
                "Risk alert received, switching to safe mode",
                alert_type=alert_type,
                severity=severity
            )

    async def _learning_loop(self):
        """Main learning loop"""

        while self.enable_learning:
            try:
                # Periodic learning tasks
                await asyncio.sleep(60)  # Every minute

                # Update learning schedules based on performance
                await self._update_learning_schedules()

                # Save checkpoint periodically
                if self.learning_orchestrator.episode_count % 100 == 0:
                    await self.learning_orchestrator.save_checkpoint()

            except Exception as e:
                logger.error(f"Learning loop error: {e}")

    async def _performance_monitoring_loop(self):
        """Monitor and report performance"""

        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Calculate performance metrics
                metrics = await self._calculate_performance_metrics()

                # Log performance
                logger.info("Performance update", **metrics)

                # Send to WebSocket if available
                if self.websocket_manager:
                    await self.websocket_manager.broadcast({
                        'type': 'learning_performance',
                        'data': metrics
                    })

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")

    async def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""

        metrics = {
            'learning_enabled': self.enable_learning,
            'symbols_tracked': list(self.learning_enabled_symbols),
            'active_positions': len(self.current_positions)
        }

        # Add learning metrics if available
        if self.learning_orchestrator:
            metrics['learning_status'] = self.learning_orchestrator.get_status()

        # Add paper trading performance
        if self.enable_paper_trading and self.performance_tracker:
            total_pnl = 0
            total_trades = 0
            winning_trades = 0

            for symbol, tracker in self.performance_tracker.items():
                symbol_pnl = 0
                for i, trade in enumerate(tracker['trades']):
                    if trade['side'] == 'sell' and i > 0:
                        # Calculate P&L for sell trades
                        prev_trade = tracker['trades'][i-1]
                        if prev_trade['side'] == 'buy':
                            pnl = (trade['price'] - prev_trade['price']) * trade['quantity']
                            symbol_pnl += pnl
                            if pnl > 0:
                                winning_trades += 1
                            total_trades += 1

                total_pnl += symbol_pnl
                tracker['total_pnl'] = symbol_pnl

            metrics['paper_trading'] = {
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'win_rate': winning_trades / max(total_trades, 1),
                'performance_by_symbol': {
                    symbol: tracker['total_pnl']
                    for symbol, tracker in self.performance_tracker.items()
                }
            }

        return metrics

    async def _update_learning_schedules(self):
        """Update learning schedules based on performance"""

        if not self.learning_orchestrator:
            return

        # Get recent performance
        metrics = await self._calculate_performance_metrics()

        # Adjust learning based on win rate
        if 'paper_trading' in metrics:
            win_rate = metrics['paper_trading'].get('win_rate', 0.5)

            if win_rate > 0.6:
                # Good performance - reduce exploration
                self.learning_orchestrator.current_mode = LearningMode.EXPLOITATION
            elif win_rate < 0.4:
                # Poor performance - increase exploration
                self.learning_orchestrator.current_mode = LearningMode.EXPLORATION

    async def _load_latest_checkpoint(self):
        """Load the latest checkpoint if available"""

        if not self.learning_orchestrator or not self.learning_orchestrator.checkpoint_path:
            return

        checkpoint_path = Path(self.learning_orchestrator.checkpoint_path)
        if not checkpoint_path.exists():
            return

        # Find latest checkpoint
        checkpoints = list(checkpoint_path.glob("orchestrator_*.json"))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            timestamp = latest.stem.replace("orchestrator_", "")

            await self.learning_orchestrator.load_checkpoint(timestamp)
            logger.info(f"Loaded checkpoint from {timestamp}")

    async def stop(self):
        """Stop the continuous learning system"""

        self.enable_learning = False

        # Save final checkpoint
        if self.learning_orchestrator:
            await self.learning_orchestrator.save_checkpoint()

        logger.info("Continuous learning trading system stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get system status"""

        status = {
            'learning_enabled': self.enable_learning,
            'paper_trading': self.enable_paper_trading,
            'symbols': list(self.learning_enabled_symbols),
            'positions': self.current_positions,
            'performance': self.performance_tracker
        }

        if self.learning_orchestrator:
            status['learning'] = self.learning_orchestrator.get_status()

        return status