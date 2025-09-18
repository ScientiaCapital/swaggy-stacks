"""
Options Strategy Factory

This factory provides a centralized way to create and manage options trading strategies.
It implements the Factory pattern to enable easy strategy instantiation based on market
conditions, user preferences, or automated selection criteria.
"""

from enum import Enum
from typing import Dict, Type, Union, Optional, Any
import logging
from decimal import Decimal

from app.core.base_strategy import BaseStrategy, StrategyConfig
from app.core.market_data import MarketDataService

# Import all strategy classes
from .long_straddle_strategy import LongStraddleStrategy, LongStraddleConfig
from .iron_butterfly_strategy import IronButterflyStrategy, IronButterflyConfig
from .calendar_spread_strategy import CalendarSpreadStrategy, CalendarSpreadConfig
from .bull_call_spread_strategy import BullCallSpreadStrategy, BullCallSpreadConfig
from .bear_put_spread_strategy import BearPutSpreadStrategy, BearPutSpreadConfig
from .covered_call_strategy import CoveredCallStrategy, CoveredCallConfig
from .protective_put_strategy import ProtectivePutStrategy, ProtectivePutConfig

# Import existing strategies if available
try:
    from .zero_dte_strategy import ZeroDTEStrategy, ZeroDTEConfig
    from .wheel_strategy import WheelStrategy, WheelConfig
    from .iron_condor_strategy import IronCondorStrategy, IronCondorConfig
    from .gamma_scalping_strategy import GammaScalpingStrategy, GammaScalpingConfig
    EXISTING_STRATEGIES_AVAILABLE = True
except ImportError:
    EXISTING_STRATEGIES_AVAILABLE = False

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Enumeration of available options strategy types"""

    # Volatility strategies
    LONG_STRADDLE = "long_straddle"
    IRON_BUTTERFLY = "iron_butterfly"

    # Time decay strategies
    CALENDAR_SPREAD = "calendar_spread"

    # Directional strategies
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"

    # Income strategies
    COVERED_CALL = "covered_call"

    # Protection strategies
    PROTECTIVE_PUT = "protective_put"

    # Existing strategies (if available)
    ZERO_DTE = "zero_dte"
    WHEEL = "wheel"
    IRON_CONDOR = "iron_condor"
    GAMMA_SCALPING = "gamma_scalping"


class MarketRegime(Enum):
    """Market regime classification for strategy selection"""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    RANGE_BOUND = "range_bound"


class OptionsStrategyFactory:
    """
    Factory class for creating options trading strategies

    This factory enables:
    - Easy strategy instantiation by type
    - Market-condition-based strategy selection
    - Configuration management and defaults
    - Strategy recommendation based on market regime
    """

    def __init__(self):
        """Initialize the strategy factory"""

        # Strategy registry mapping
        self._strategy_registry: Dict[StrategyType, Dict[str, Type]] = {
            # New advanced strategies
            StrategyType.LONG_STRADDLE: {
                "strategy": LongStraddleStrategy,
                "config": LongStraddleConfig
            },
            StrategyType.IRON_BUTTERFLY: {
                "strategy": IronButterflyStrategy,
                "config": IronButterflyConfig
            },
            StrategyType.CALENDAR_SPREAD: {
                "strategy": CalendarSpreadStrategy,
                "config": CalendarSpreadConfig
            },
            StrategyType.BULL_CALL_SPREAD: {
                "strategy": BullCallSpreadStrategy,
                "config": BullCallSpreadConfig
            },
            StrategyType.BEAR_PUT_SPREAD: {
                "strategy": BearPutSpreadStrategy,
                "config": BearPutSpreadConfig
            },
            StrategyType.COVERED_CALL: {
                "strategy": CoveredCallStrategy,
                "config": CoveredCallConfig
            },
            StrategyType.PROTECTIVE_PUT: {
                "strategy": ProtectivePutStrategy,
                "config": ProtectivePutConfig
            },
        }

        # Add existing strategies if available
        if EXISTING_STRATEGIES_AVAILABLE:
            self._strategy_registry.update({
                StrategyType.ZERO_DTE: {
                    "strategy": ZeroDTEStrategy,
                    "config": ZeroDTEConfig
                },
                StrategyType.WHEEL: {
                    "strategy": WheelStrategy,
                    "config": WheelConfig
                },
                StrategyType.IRON_CONDOR: {
                    "strategy": IronCondorStrategy,
                    "config": IronCondorConfig
                },
                StrategyType.GAMMA_SCALPING: {
                    "strategy": GammaScalpingStrategy,
                    "config": GammaScalpingConfig
                },
            })

        # Market regime to strategy mapping
        self._regime_strategies = {
            MarketRegime.HIGH_VOLATILITY: [
                StrategyType.LONG_STRADDLE,
                StrategyType.PROTECTIVE_PUT,
                StrategyType.GAMMA_SCALPING
            ],
            MarketRegime.LOW_VOLATILITY: [
                StrategyType.IRON_BUTTERFLY,
                StrategyType.COVERED_CALL,
                StrategyType.IRON_CONDOR
            ],
            MarketRegime.BULLISH: [
                StrategyType.BULL_CALL_SPREAD,
                StrategyType.COVERED_CALL,
                StrategyType.CALENDAR_SPREAD
            ],
            MarketRegime.BEARISH: [
                StrategyType.BEAR_PUT_SPREAD,
                StrategyType.PROTECTIVE_PUT
            ],
            MarketRegime.NEUTRAL: [
                StrategyType.IRON_BUTTERFLY,
                StrategyType.IRON_CONDOR,
                StrategyType.CALENDAR_SPREAD
            ],
            MarketRegime.TRENDING: [
                StrategyType.BULL_CALL_SPREAD,
                StrategyType.BEAR_PUT_SPREAD,
                StrategyType.GAMMA_SCALPING
            ],
            MarketRegime.RANGE_BOUND: [
                StrategyType.IRON_BUTTERFLY,
                StrategyType.IRON_CONDOR,
                StrategyType.COVERED_CALL
            ]
        }

        logger.info(f"Initialized OptionsStrategyFactory with {len(self._strategy_registry)} strategies")

    def create_strategy(
        self,
        strategy_type: StrategyType,
        market_data: MarketDataService,
        config: Optional[StrategyConfig] = None,
        **config_kwargs
    ) -> BaseStrategy:
        """
        Create a strategy instance by type

        Args:
            strategy_type: The type of strategy to create
            market_data: Market data service instance
            config: Optional custom configuration
            **config_kwargs: Additional configuration parameters

        Returns:
            Configured strategy instance

        Raises:
            ValueError: If strategy type is not supported
        """

        if strategy_type not in self._strategy_registry:
            available_types = [st.value for st in self._strategy_registry.keys()]
            raise ValueError(f"Unsupported strategy type: {strategy_type.value}. "
                           f"Available types: {available_types}")

        strategy_info = self._strategy_registry[strategy_type]
        strategy_class = strategy_info["strategy"]
        config_class = strategy_info["config"]

        # Create configuration if not provided
        if config is None:
            config = config_class(**config_kwargs)
        elif config_kwargs:
            # Update existing config with additional kwargs
            for key, value in config_kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Create and return strategy instance
        strategy = strategy_class(config=config, market_data=market_data)

        logger.info(f"Created {strategy_type.value} strategy with config: {config}")
        return strategy

    def get_recommended_strategies(
        self,
        market_regime: MarketRegime,
        max_strategies: int = 3
    ) -> list[StrategyType]:
        """
        Get recommended strategies for a given market regime

        Args:
            market_regime: Current market regime
            max_strategies: Maximum number of strategies to recommend

        Returns:
            List of recommended strategy types
        """

        if market_regime not in self._regime_strategies:
            logger.warning(f"Unknown market regime: {market_regime}. Returning neutral strategies.")
            return self._regime_strategies[MarketRegime.NEUTRAL][:max_strategies]

        recommended = self._regime_strategies[market_regime][:max_strategies]
        logger.info(f"Recommended strategies for {market_regime.value}: {[s.value for s in recommended]}")

        return recommended

    def analyze_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """
        Analyze market conditions to determine current regime

        Args:
            market_data: Dictionary containing market indicators

        Returns:
            Detected market regime
        """

        # Extract key indicators
        vix = Decimal(str(market_data.get('vix', 20)))
        iv_rank = Decimal(str(market_data.get('iv_rank', 50)))
        trend_strength = Decimal(str(market_data.get('trend_strength', 0)))
        rsi = Decimal(str(market_data.get('rsi', 50)))
        bollinger_position = Decimal(str(market_data.get('bollinger_position', 0.5)))

        # Volatility analysis
        if vix >= 30 or iv_rank >= 70:
            return MarketRegime.HIGH_VOLATILITY
        elif vix <= 15 or iv_rank <= 30:
            return MarketRegime.LOW_VOLATILITY

        # Trend analysis
        if abs(trend_strength) >= 0.7:
            if trend_strength > 0:
                if rsi >= 60:
                    return MarketRegime.BULLISH
                else:
                    return MarketRegime.TRENDING
            else:
                if rsi <= 40:
                    return MarketRegime.BEARISH
                else:
                    return MarketRegime.TRENDING

        # Range-bound analysis
        if abs(trend_strength) <= 0.3 and 0.3 <= bollinger_position <= 0.7:
            return MarketRegime.RANGE_BOUND

        # Default to neutral
        return MarketRegime.NEUTRAL

    def create_portfolio_strategies(
        self,
        market_data: Dict[str, Any],
        risk_tolerance: str = "moderate",
        max_strategies: int = 3
    ) -> list[BaseStrategy]:
        """
        Create a diversified portfolio of strategies based on market conditions

        Args:
            market_data: Market data for analysis
            risk_tolerance: Risk tolerance level (conservative, moderate, aggressive)
            max_strategies: Maximum number of strategies to include

        Returns:
            List of configured strategy instances
        """

        market_data_service = MarketDataService()  # This would be injected in practice
        regime = self.analyze_market_regime(market_data)
        recommended_types = self.get_recommended_strategies(regime, max_strategies)

        strategies = []

        for strategy_type in recommended_types:
            try:
                # Adjust configuration based on risk tolerance
                config_kwargs = self._get_risk_adjusted_config(strategy_type, risk_tolerance)

                strategy = self.create_strategy(
                    strategy_type=strategy_type,
                    market_data=market_data_service,
                    **config_kwargs
                )

                strategies.append(strategy)

            except Exception as e:
                logger.error(f"Failed to create {strategy_type.value} strategy: {e}")
                continue

        logger.info(f"Created portfolio with {len(strategies)} strategies for {regime.value} regime")
        return strategies

    def _get_risk_adjusted_config(self, strategy_type: StrategyType, risk_tolerance: str) -> Dict[str, Any]:
        """Get configuration adjustments based on risk tolerance"""

        base_adjustments = {
            "conservative": {
                "max_position_size": Decimal("5000"),
                "profit_target": Decimal("0.25"),  # Take profits earlier
                "stop_loss": Decimal("1.50"),  # Tighter stops
            },
            "moderate": {
                "max_position_size": Decimal("10000"),
                "profit_target": Decimal("0.50"),  # Standard targets
                "stop_loss": Decimal("2.00"),  # Standard stops
            },
            "aggressive": {
                "max_position_size": Decimal("25000"),
                "profit_target": Decimal("0.75"),  # Let winners run
                "stop_loss": Decimal("3.00"),  # Wider stops
            }
        }

        return base_adjustments.get(risk_tolerance, base_adjustments["moderate"])

    def get_available_strategies(self) -> list[StrategyType]:
        """Get list of all available strategy types"""
        return list(self._strategy_registry.keys())

    def get_strategy_info(self, strategy_type: StrategyType) -> Dict[str, Any]:
        """Get information about a specific strategy type"""

        if strategy_type not in self._strategy_registry:
            return {}

        strategy_info = self._strategy_registry[strategy_type]

        # Get strategy description from docstring
        strategy_class = strategy_info["strategy"]
        description = strategy_class.__doc__.split('\n')[0] if strategy_class.__doc__ else "No description available"

        return {
            "type": strategy_type.value,
            "name": strategy_class.__name__,
            "description": description.strip(),
            "config_class": strategy_info["config"].__name__,
            "available": True
        }

    def get_all_strategies_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available strategies"""

        return {
            strategy_type.value: self.get_strategy_info(strategy_type)
            for strategy_type in self._strategy_registry.keys()
        }