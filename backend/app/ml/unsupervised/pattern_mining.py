"""
Market Basket Analysis and Pattern Mining for Trading Systems

This module implements efficient association rule mining optimized for:
- Asset correlation discovery
- Sector rotation patterns
- Cross-asset arbitrage opportunities
- Market regime transition signals
- Streaming Apriori algorithm for M1 MacBook memory constraints
"""

import asyncio
import json
import pickle
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import redis
import structlog
from scipy.stats import chi2_contingency, pearsonr

from .base import UnsupervisedBase

logger = structlog.get_logger()


@dataclass
class MarketBasketItem:
    """An item in market basket analysis (e.g., price movement, volume spike)"""
    symbol: str
    action: str  # 'buy_signal', 'sell_signal', 'volume_spike', 'volatility_high', etc.
    timeframe: str  # '1m', '5m', '1h', '1d'
    threshold: float  # Threshold value for the signal
    timestamp: datetime


@dataclass
class AssociationRule:
    """Association rule between market events"""
    antecedent: Set[str]  # If these events occur...
    consequent: Set[str]  # Then these events likely occur
    support: float  # How often the full rule appears
    confidence: float  # Reliability of the rule
    lift: float  # How much more likely consequent is given antecedent
    conviction: float  # How much more frequent antecedent would be without consequent
    leverage: float  # Difference between observed and expected frequency
    created_at: datetime
    observations: int  # Number of times this rule was observed
    success_rate: float  # Historical success rate in trading
    avg_time_delay: float  # Average time between antecedent and consequent (in minutes)


@dataclass
class MarketTransaction:
    """A market transaction (basket) containing multiple simultaneous events"""
    transaction_id: str
    timestamp: datetime
    items: List[MarketBasketItem]
    market_regime: Optional[str] = None  # 'bull', 'bear', 'sideways', 'volatile'
    volatility_regime: Optional[str] = None  # 'low', 'medium', 'high'


class StreamingApriori:
    """
    Memory-efficient streaming Apriori algorithm for M1 MacBook
    Processes market events in real-time without storing full transaction history
    """

    def __init__(self, min_support: float = 0.1, max_memory_mb: int = 512):
        self.min_support = min_support
        self.max_memory_mb = max_memory_mb

        # Counters for streaming processing
        self.item_counts = Counter()
        self.pair_counts = Counter()
        self.triplet_counts = Counter()
        self.total_transactions = 0

        # Frequent itemsets cache
        self.frequent_1_itemsets = set()
        self.frequent_2_itemsets = set()
        self.frequent_3_itemsets = set()

        # Memory management
        self.last_pruning = time.time()
        self.pruning_interval = 3600  # Prune every hour

    def add_transaction(self, items: List[str]):
        """Add a new transaction to the streaming analysis"""
        self.total_transactions += 1

        # Count individual items
        for item in items:
            self.item_counts[item] += 1

        # Count pairs
        for pair in combinations(items, 2):
            self.pair_counts[tuple(sorted(pair))] += 1

        # Count triplets (limited to save memory)
        if len(items) >= 3:
            for triplet in combinations(items, 3):
                self.triplet_counts[tuple(sorted(triplet))] += 1

        # Periodic pruning to manage memory
        if time.time() - self.last_pruning > self.pruning_interval:
            self._prune_infrequent_patterns()

    def _prune_infrequent_patterns(self):
        """Remove infrequent patterns to save memory"""
        min_count = max(1, int(self.total_transactions * self.min_support))

        # Prune infrequent items
        self.item_counts = Counter({
            item: count for item, count in self.item_counts.items()
            if count >= min_count
        })

        # Prune infrequent pairs
        self.pair_counts = Counter({
            pair: count for pair, count in self.pair_counts.items()
            if count >= min_count and all(item in self.item_counts for item in pair)
        })

        # Prune infrequent triplets
        self.triplet_counts = Counter({
            triplet: count for triplet, count in self.triplet_counts.items()
            if count >= min_count and all(item in self.item_counts for item in triplet)
        })

        self.last_pruning = time.time()

    def get_frequent_itemsets(self) -> Dict[int, List[Tuple]]:
        """Get current frequent itemsets"""
        min_count = max(1, int(self.total_transactions * self.min_support))

        frequent_itemsets = {
            1: [(item,) for item, count in self.item_counts.items() if count >= min_count],
            2: [pair for pair, count in self.pair_counts.items() if count >= min_count],
            3: [triplet for triplet, count in self.triplet_counts.items() if count >= min_count]
        }

        return frequent_itemsets

    def get_support(self, itemset: Tuple[str, ...]) -> float:
        """Get support for an itemset"""
        if len(itemset) == 1:
            return self.item_counts.get(itemset[0], 0) / max(1, self.total_transactions)
        elif len(itemset) == 2:
            return self.pair_counts.get(tuple(sorted(itemset)), 0) / max(1, self.total_transactions)
        elif len(itemset) == 3:
            return self.triplet_counts.get(tuple(sorted(itemset)), 0) / max(1, self.total_transactions)
        else:
            return 0.0


class PatternMining(UnsupervisedBase):
    """
    Advanced pattern mining system for trading strategies
    Discovers hidden correlations and association rules in market data
    """

    def __init__(self,
                 min_support: float = 0.05,  # 5% minimum support
                 min_confidence: float = 0.6,  # 60% minimum confidence
                 min_lift: float = 1.2,  # 20% lift improvement
                 max_rules: int = 10000,  # Maximum rules to store
                 redis_host: str = "localhost",
                 redis_port: int = 6379):

        super().__init__()

        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_rules = max_rules

        # Initialize streaming Apriori
        self.apriori = StreamingApriori(min_support=min_support)

        # Association rules storage
        self.association_rules = {}
        self.rule_performance = {}

        # Market transaction history (limited)
        self.recent_transactions = []
        self.max_transaction_history = 10000

        # Pattern success tracking
        self.pattern_predictions = {}  # Track rule predictions for success rate
        self.pattern_outcomes = {}     # Track actual outcomes

        # Connect to Redis for caching
        try:
            self.redis_client = redis.Redis(
                host=redis_host, port=redis_port,
                decode_responses=False, db=1  # Use different DB than pattern memory
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected for pattern mining cache")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
            self.redis_client = None

        # Performance statistics
        self.stats = {
            'transactions_processed': 0,
            'rules_discovered': 0,
            'successful_predictions': 0,
            'total_predictions': 0,
            'avg_processing_time_ms': 0
        }

        logger.info(f"🔍 PatternMining initialized with {min_support:.1%} min support")

    def _create_market_items(self, market_data: Dict[str, Any]) -> List[MarketBasketItem]:
        """
        Convert market data into market basket items
        """
        items = []
        timestamp = datetime.now()

        for symbol, data in market_data.items():
            if not isinstance(data, dict):
                continue

            # Price movement signals
            if 'price_change_pct' in data:
                change = data['price_change_pct']
                if change > 2.0:
                    items.append(MarketBasketItem(
                        symbol=symbol,
                        action='strong_bullish',
                        timeframe='1h',
                        threshold=2.0,
                        timestamp=timestamp
                    ))
                elif change > 0.5:
                    items.append(MarketBasketItem(
                        symbol=symbol,
                        action='bullish',
                        timeframe='1h',
                        threshold=0.5,
                        timestamp=timestamp
                    ))
                elif change < -2.0:
                    items.append(MarketBasketItem(
                        symbol=symbol,
                        action='strong_bearish',
                        timeframe='1h',
                        threshold=-2.0,
                        timestamp=timestamp
                    ))
                elif change < -0.5:
                    items.append(MarketBasketItem(
                        symbol=symbol,
                        action='bearish',
                        timeframe='1h',
                        threshold=-0.5,
                        timestamp=timestamp
                    ))

            # Volume signals
            if 'volume_ratio' in data:
                vol_ratio = data['volume_ratio']
                if vol_ratio > 2.0:
                    items.append(MarketBasketItem(
                        symbol=symbol,
                        action='volume_spike',
                        timeframe='1h',
                        threshold=2.0,
                        timestamp=timestamp
                    ))

            # Volatility signals
            if 'volatility' in data:
                volatility = data['volatility']
                if volatility > 0.03:  # 3% volatility
                    items.append(MarketBasketItem(
                        symbol=symbol,
                        action='high_volatility',
                        timeframe='1h',
                        threshold=0.03,
                        timestamp=timestamp
                    ))

            # Technical indicator signals
            if 'rsi' in data:
                rsi = data['rsi']
                if rsi > 70:
                    items.append(MarketBasketItem(
                        symbol=symbol,
                        action='rsi_overbought',
                        timeframe='1h',
                        threshold=70,
                        timestamp=timestamp
                    ))
                elif rsi < 30:
                    items.append(MarketBasketItem(
                        symbol=symbol,
                        action='rsi_oversold',
                        timeframe='1h',
                        threshold=30,
                        timestamp=timestamp
                    ))

        return items

    def process_market_data(self, market_data: Dict[str, Any], market_regime: Optional[str] = None):
        """
        Process real-time market data and update pattern mining
        """
        start_time = time.time()

        # Create market basket items
        items = self._create_market_items(market_data)

        if len(items) < 2:  # Need at least 2 items for association rules
            return

        # Create transaction
        transaction_id = f"tx_{int(time.time() * 1000)}"
        transaction = MarketTransaction(
            transaction_id=transaction_id,
            timestamp=datetime.now(),
            items=items,
            market_regime=market_regime
        )

        # Add to recent transactions (with memory limit)
        self.recent_transactions.append(transaction)
        if len(self.recent_transactions) > self.max_transaction_history:
            self.recent_transactions = self.recent_transactions[-self.max_transaction_history:]

        # Convert items to string format for Apriori
        item_strings = [f"{item.symbol}_{item.action}" for item in items]

        # Add to streaming Apriori
        self.apriori.add_transaction(item_strings)

        # Generate new association rules periodically
        if self.apriori.total_transactions % 100 == 0:  # Every 100 transactions
            self._update_association_rules()

        # Update performance stats
        processing_time = (time.time() - start_time) * 1000
        self.stats['transactions_processed'] += 1
        self.stats['avg_processing_time_ms'] = (
            0.9 * self.stats['avg_processing_time_ms'] + 0.1 * processing_time
        )

    def _update_association_rules(self):
        """Generate association rules from frequent itemsets"""
        frequent_itemsets = self.apriori.get_frequent_itemsets()

        new_rules = []

        # Generate rules from 2-itemsets
        for itemset in frequent_itemsets.get(2, []):
            for i in range(len(itemset)):
                antecedent = {itemset[i]}
                consequent = {itemset[1-i]}

                rule = self._create_association_rule(antecedent, consequent)
                if rule and rule.confidence >= self.min_confidence and rule.lift >= self.min_lift:
                    new_rules.append(rule)

        # Generate rules from 3-itemsets
        for itemset in frequent_itemsets.get(3, []):
            # Try different combinations of antecedent/consequent
            for i in range(len(itemset)):
                for j in range(i+1, len(itemset)):
                    antecedent = {itemset[i], itemset[j]}
                    consequent = {itemset[k] for k in range(len(itemset)) if k not in [i, j]}

                    rule = self._create_association_rule(antecedent, consequent)
                    if rule and rule.confidence >= self.min_confidence and rule.lift >= self.min_lift:
                        new_rules.append(rule)

        # Store new rules
        for rule in new_rules:
            rule_key = f"{frozenset(rule.antecedent)}->{frozenset(rule.consequent)}"
            self.association_rules[rule_key] = rule

        # Prune rules if too many
        if len(self.association_rules) > self.max_rules:
            self._prune_association_rules()

        self.stats['rules_discovered'] = len(self.association_rules)
        logger.debug(f"🔍 Updated association rules: {len(new_rules)} new, {len(self.association_rules)} total")

    def _create_association_rule(self, antecedent: Set[str], consequent: Set[str]) -> Optional[AssociationRule]:
        """Create an association rule with calculated metrics"""

        # Calculate support, confidence, and lift
        antecedent_tuple = tuple(sorted(antecedent))
        consequent_tuple = tuple(sorted(consequent))
        full_itemset = tuple(sorted(antecedent | consequent))

        support_full = self.apriori.get_support(full_itemset)
        support_antecedent = self.apriori.get_support(antecedent_tuple)
        support_consequent = self.apriori.get_support(consequent_tuple)

        if support_antecedent == 0 or support_full < self.min_support:
            return None

        confidence = support_full / support_antecedent
        lift = confidence / max(support_consequent, 0.001)  # Avoid division by zero

        # Calculate conviction
        if confidence == 1.0:
            conviction = float('inf')
        else:
            conviction = (1 - support_consequent) / max(1 - confidence, 0.001)

        # Calculate leverage
        leverage = support_full - (support_antecedent * support_consequent)

        return AssociationRule(
            antecedent=antecedent,
            consequent=consequent,
            support=support_full,
            confidence=confidence,
            lift=lift,
            conviction=conviction,
            leverage=leverage,
            created_at=datetime.now(),
            observations=int(support_full * self.apriori.total_transactions),
            success_rate=0.5,  # Will be updated based on actual performance
            avg_time_delay=30.0  # Default 30 minutes, will be calculated from observations
        )

    def _prune_association_rules(self):
        """Remove low-performing or old association rules"""
        # Sort by performance score (combination of confidence, lift, and success rate)
        rules_with_scores = []

        for rule_key, rule in self.association_rules.items():
            performance_score = (
                rule.confidence * 0.4 +
                min(rule.lift / 2.0, 1.0) * 0.3 +  # Cap lift contribution
                rule.success_rate * 0.3
            )
            rules_with_scores.append((rule_key, performance_score))

        # Keep top performing rules
        rules_with_scores.sort(key=lambda x: x[1], reverse=True)
        rules_to_keep = dict(rules_with_scores[:self.max_rules * 3 // 4])  # Keep top 75%

        # Update rules dictionary
        self.association_rules = {
            rule_key: rule for rule_key, rule in self.association_rules.items()
            if rule_key in rules_to_keep
        }

    def predict_market_events(self, current_events: List[str]) -> List[Dict[str, Any]]:
        """
        Predict likely future market events based on current events and association rules
        """
        predictions = []
        current_event_set = set(current_events)

        for rule_key, rule in self.association_rules.items():
            # Check if current events match the rule's antecedent
            if rule.antecedent.issubset(current_event_set):
                for consequent_event in rule.consequent:
                    if consequent_event not in current_event_set:  # Don't predict what already happened
                        predictions.append({
                            'predicted_event': consequent_event,
                            'confidence': rule.confidence,
                            'lift': rule.lift,
                            'support': rule.support,
                            'success_rate': rule.success_rate,
                            'avg_time_delay_minutes': rule.avg_time_delay,
                            'antecedent': list(rule.antecedent),
                            'rule_key': rule_key
                        })

        # Sort by confidence * success_rate * lift
        predictions.sort(
            key=lambda x: x['confidence'] * x['success_rate'] * min(x['lift'], 2.0),
            reverse=True
        )

        # Track predictions for success rate calculation
        for pred in predictions[:10]:  # Track top 10 predictions
            self.pattern_predictions[f"{pred['rule_key']}_{int(time.time())}"] = {
                'predicted_event': pred['predicted_event'],
                'timestamp': datetime.now(),
                'confidence': pred['confidence']
            }

        self.stats['total_predictions'] += len(predictions)

        return predictions[:20]  # Return top 20 predictions

    def get_asset_correlations(self, lookback_hours: int = 24) -> Dict[str, List[Dict]]:
        """
        Get asset correlations discovered through association rule mining
        """
        correlations = defaultdict(list)
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)

        for rule_key, rule in self.association_rules.items():
            if rule.created_at < cutoff_time:
                continue

            # Extract asset correlations from rules
            antecedent_assets = set()
            consequent_assets = set()

            for item in rule.antecedent:
                if '_' in item:
                    asset = item.split('_')[0]
                    antecedent_assets.add(asset)

            for item in rule.consequent:
                if '_' in item:
                    asset = item.split('_')[0]
                    consequent_assets.add(asset)

            # Create correlation entries
            for ant_asset in antecedent_assets:
                for cons_asset in consequent_assets:
                    if ant_asset != cons_asset:
                        correlations[ant_asset].append({
                            'correlated_asset': cons_asset,
                            'correlation_strength': rule.confidence * rule.lift,
                            'confidence': rule.confidence,
                            'lift': rule.lift,
                            'avg_delay_minutes': rule.avg_time_delay,
                            'rule_description': f"{list(rule.antecedent)} -> {list(rule.consequent)}"
                        })

        # Sort correlations by strength
        for asset in correlations:
            correlations[asset].sort(key=lambda x: x['correlation_strength'], reverse=True)
            correlations[asset] = correlations[asset][:10]  # Top 10 correlations per asset

        return dict(correlations)

    def get_sector_rotation_patterns(self) -> List[Dict[str, Any]]:
        """
        Identify sector rotation patterns from association rules
        """
        sector_patterns = []

        # Define sector mappings (simplified)
        sector_keywords = {
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'],
            'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C'],
            'healthcare': ['JNJ', 'PFE', 'ABBV', 'MRK', 'TMO'],
            'energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
            'consumer': ['WMT', 'PG', 'KO', 'PEP', 'MCD']
        }

        for rule_key, rule in self.association_rules.items():
            # Identify sectors in antecedent and consequent
            ante_sectors = set()
            cons_sectors = set()

            for item in rule.antecedent:
                symbol = item.split('_')[0] if '_' in item else item
                for sector, symbols in sector_keywords.items():
                    if symbol in symbols:
                        ante_sectors.add(sector)

            for item in rule.consequent:
                symbol = item.split('_')[0] if '_' in item else item
                for sector, symbols in sector_keywords.items():
                    if symbol in symbols:
                        cons_sectors.add(sector)

            # If we have different sectors, it's a rotation pattern
            if ante_sectors and cons_sectors and ante_sectors != cons_sectors:
                sector_patterns.append({
                    'from_sectors': list(ante_sectors),
                    'to_sectors': list(cons_sectors),
                    'confidence': rule.confidence,
                    'lift': rule.lift,
                    'support': rule.support,
                    'avg_delay_minutes': rule.avg_time_delay,
                    'observations': rule.observations,
                    'success_rate': rule.success_rate
                })

        # Sort by confidence * lift
        sector_patterns.sort(key=lambda x: x['confidence'] * x['lift'], reverse=True)

        return sector_patterns[:20]

    def update_pattern_success_rates(self, market_data: Dict[str, Any]):
        """
        Update success rates of patterns based on actual market outcomes
        """
        current_time = datetime.now()
        current_events = [f"{symbol}_{action}" for symbol, data in market_data.items()
                         for action in self._extract_actions_from_data(data)]

        # Check previous predictions
        expired_predictions = []

        for pred_key, prediction in self.pattern_predictions.items():
            time_elapsed = (current_time - prediction['timestamp']).total_seconds() / 60

            # Check if prediction time window has passed (default 60 minutes)
            if time_elapsed > 60:
                # Check if prediction came true
                if prediction['predicted_event'] in current_events:
                    self.stats['successful_predictions'] += 1

                    # Update rule success rate
                    rule_key = pred_key.split('_')[0]
                    if rule_key in self.association_rules:
                        rule = self.association_rules[rule_key]
                        rule.success_rate = 0.9 * rule.success_rate + 0.1 * 1.0  # Exponential moving average

                expired_predictions.append(pred_key)

        # Clean up expired predictions
        for pred_key in expired_predictions:
            del self.pattern_predictions[pred_key]

    def _extract_actions_from_data(self, data: Dict[str, Any]) -> List[str]:
        """Extract action signals from market data"""
        actions = []

        if 'price_change_pct' in data:
            change = data['price_change_pct']
            if change > 2.0:
                actions.append('strong_bullish')
            elif change > 0.5:
                actions.append('bullish')
            elif change < -2.0:
                actions.append('strong_bearish')
            elif change < -0.5:
                actions.append('bearish')

        if 'volume_ratio' in data and data['volume_ratio'] > 2.0:
            actions.append('volume_spike')

        if 'volatility' in data and data['volatility'] > 0.03:
            actions.append('high_volatility')

        return actions

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pattern mining performance statistics"""
        success_rate = (
            self.stats['successful_predictions'] / max(1, self.stats['total_predictions'])
        )

        return {
            **self.stats,
            'success_rate': success_rate,
            'total_rules': len(self.association_rules),
            'recent_transactions': len(self.recent_transactions),
            'frequent_items': len(self.apriori.frequent_1_itemsets),
            'redis_available': self.redis_client is not None
        }