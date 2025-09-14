"""
Trading Utilities
Common functions used across trading modules to eliminate redundancy
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


class OrderSide(Enum):
    """Order side enumeration"""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(Enum):
    """Time in force enumeration"""

    GTC = "gtc"  # Good till canceled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    DAY = "day"  # Day order


def validate_symbol(symbol: str) -> str:
    """
    Validate and clean trading symbol

    Args:
        symbol: Trading symbol to validate

    Returns:
        Cleaned symbol

    Raises:
        ValueError: If symbol is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")

    # Clean and uppercase
    clean_symbol = symbol.strip().upper()

    # Basic validation (alphanumeric with optional dots/hyphens)
    if not clean_symbol.replace("-", "").replace(".", "").isalnum():
        raise ValueError(f"Invalid symbol format: {symbol}")

    # Length check
    if len(clean_symbol) < 1 or len(clean_symbol) > 10:
        raise ValueError(f"Symbol length must be 1-10 characters: {symbol}")

    return clean_symbol


def validate_quantity(quantity: Union[int, float]) -> float:
    """
    Validate trading quantity

    Args:
        quantity: Quantity to validate

    Returns:
        Valid quantity as float

    Raises:
        ValueError: If quantity is invalid
    """
    try:
        qty = float(quantity)
    except (TypeError, ValueError):
        raise ValueError(f"Quantity must be numeric: {quantity}")

    if qty <= 0:
        raise ValueError(f"Quantity must be positive: {quantity}")

    if qty > 1000000:  # Reasonable upper limit
        raise ValueError(f"Quantity too large: {quantity}")

    return qty


def validate_price(price: Union[int, float]) -> float:
    """
    Validate price value

    Args:
        price: Price to validate

    Returns:
        Valid price as float

    Raises:
        ValueError: If price is invalid
    """
    try:
        p = float(price)
    except (TypeError, ValueError):
        raise ValueError(f"Price must be numeric: {price}")

    if p < 0:
        raise ValueError(f"Price cannot be negative: {price}")

    if p > 100000:  # Reasonable upper limit for most stocks
        raise ValueError(f"Price too high: {price}")

    return p


def calculate_position_value(quantity: float, price: float) -> float:
    """
    Calculate position value

    Args:
        quantity: Number of shares
        price: Price per share

    Returns:
        Total position value
    """
    return abs(quantity) * price


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change

    Args:
        old_value: Original value
        new_value: New value

    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0 if new_value == 0 else float("inf")

    return ((new_value - old_value) / old_value) * 100


def calculate_pnl(
    entry_price: float, current_price: float, quantity: float
) -> Dict[str, float]:
    """
    Calculate profit/loss for a position

    Args:
        entry_price: Entry price
        current_price: Current market price
        quantity: Position size (positive for long, negative for short)

    Returns:
        Dictionary with PnL information
    """
    if quantity == 0:
        return {"unrealized_pnl": 0.0, "unrealized_pnl_pct": 0.0, "position_value": 0.0}

    position_value = abs(quantity) * current_price

    if quantity > 0:  # Long position
        unrealized_pnl = quantity * (current_price - entry_price)
    else:  # Short position
        unrealized_pnl = abs(quantity) * (entry_price - current_price)

    unrealized_pnl_pct = (
        calculate_percentage_change(
            abs(quantity) * entry_price, abs(quantity) * current_price
        )
        if entry_price > 0
        else 0.0
    )

    # Adjust percentage for short positions
    if quantity < 0:
        unrealized_pnl_pct = -unrealized_pnl_pct

    return {
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "position_value": position_value,
    }


def calculate_risk_metrics(
    positions: List[Dict], account_value: float
) -> Dict[str, float]:
    """
    Calculate portfolio risk metrics

    Args:
        positions: List of position dictionaries
        account_value: Total account value

    Returns:
        Risk metrics dictionary
    """
    if not positions or account_value <= 0:
        return {
            "total_exposure": 0.0,
            "exposure_ratio": 0.0,
            "largest_position_pct": 0.0,
            "position_count": 0,
            "long_exposure": 0.0,
            "short_exposure": 0.0,
            "net_exposure": 0.0,
        }

    total_long_value = 0.0
    total_short_value = 0.0
    largest_position = 0.0

    for pos in positions:
        quantity = pos.get("quantity", 0)
        current_price = pos.get("current_price", pos.get("price", 0))
        position_value = abs(quantity) * current_price

        if quantity > 0:
            total_long_value += position_value
        else:
            total_short_value += position_value

        largest_position = max(largest_position, position_value)

    total_exposure = total_long_value + total_short_value
    net_exposure = total_long_value - total_short_value

    return {
        "total_exposure": total_exposure,
        "exposure_ratio": total_exposure / account_value,
        "largest_position_pct": (largest_position / account_value) * 100,
        "position_count": len(positions),
        "long_exposure": total_long_value,
        "short_exposure": total_short_value,
        "net_exposure": net_exposure,
    }


def round_to_tick_size(price: float, tick_size: float = 0.01) -> float:
    """
    Round price to valid tick size

    Args:
        price: Price to round
        tick_size: Minimum tick size (default 0.01 for most stocks)

    Returns:
        Rounded price
    """
    if tick_size <= 0:
        return price

    return round(price / tick_size) * tick_size


def calculate_stop_loss(
    entry_price: float,
    side: str,
    stop_pct: float = 0.02,
    atr_multiplier: Optional[float] = None,
    atr_value: Optional[float] = None,
) -> float:
    """
    Calculate stop loss price

    Args:
        entry_price: Entry price
        side: Order side ('buy' or 'sell')
        stop_pct: Stop loss percentage (default 2%)
        atr_multiplier: ATR multiplier for dynamic stop
        atr_value: ATR value for dynamic stop

    Returns:
        Stop loss price
    """
    if atr_multiplier and atr_value:
        # ATR-based stop loss
        if side.lower() == "buy":
            return entry_price - (atr_multiplier * atr_value)
        else:
            return entry_price + (atr_multiplier * atr_value)
    else:
        # Percentage-based stop loss
        if side.lower() == "buy":
            return entry_price * (1 - stop_pct)
        else:
            return entry_price * (1 + stop_pct)


def calculate_position_size(
    account_value: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss: float,
    max_position_pct: float = 0.1,
) -> Dict[str, float]:
    """
    Calculate optimal position size based on risk management

    Args:
        account_value: Total account value
        risk_per_trade: Risk per trade (e.g., 0.02 for 2%)
        entry_price: Entry price
        stop_loss: Stop loss price
        max_position_pct: Maximum position as percentage of account

    Returns:
        Position sizing information
    """
    if entry_price <= 0 or stop_loss <= 0 or account_value <= 0:
        return {
            "quantity": 0.0,
            "position_value": 0.0,
            "risk_amount": 0.0,
            "method": "error",
        }

    # Risk-based sizing
    risk_amount = account_value * risk_per_trade
    risk_per_share = abs(entry_price - stop_loss)

    if risk_per_share > 0:
        risk_based_quantity = risk_amount / risk_per_share
        risk_based_quantity * entry_price
    else:
        risk_based_quantity = 0

    # Maximum position sizing
    max_position_value = account_value * max_position_pct
    max_quantity = max_position_value / entry_price

    # Choose the smaller of the two
    final_quantity = min(risk_based_quantity, max_quantity)
    final_value = final_quantity * entry_price

    method = "risk_based" if final_quantity == risk_based_quantity else "position_limit"

    return {
        "quantity": final_quantity,
        "position_value": final_value,
        "risk_amount": final_quantity * risk_per_share,
        "method": method,
        "max_quantity": max_quantity,
        "risk_based_quantity": risk_based_quantity,
    }


def is_market_open(current_time: Optional[datetime] = None) -> bool:
    """
    Check if US stock market is open

    Args:
        current_time: Time to check (defaults to now)

    Returns:
        True if market is open
    """
    if current_time is None:
        current_time = datetime.now()

    # Convert to ET (approximate, doesn't handle DST perfectly)
    # In production, would use proper timezone handling

    # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
    if current_time.weekday() >= 5:  # Weekend
        return False

    # Simple time check (not handling holidays or DST properly)
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= current_time <= market_close


def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format currency amount

    Args:
        amount: Amount to format
        currency: Currency code

    Returns:
        Formatted currency string
    """
    if currency.upper() == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def calculate_sharpe_ratio(
    returns: List[float], risk_free_rate: float = 0.02, periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio

    Args:
        returns: List of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0

    returns_array = np.array(returns)
    excess_returns = returns_array - (risk_free_rate / periods_per_year)

    if np.std(excess_returns) == 0:
        return 0.0

    sharpe = (
        np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    )
    return sharpe


def generate_order_id() -> str:
    """
    Generate unique order ID

    Returns:
        Unique order ID string
    """
    from uuid import uuid4

    return f"order_{int(datetime.now().timestamp())}_{str(uuid4())[:8]}"


def parse_market_data(data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
    """
    Parse and standardize market data format

    Args:
        data: Market data in various formats

    Returns:
        Standardized DataFrame with OHLCV columns
    """
    if isinstance(data, pd.DataFrame):
        # Standardize column names
        column_mapping = {
            "Close": "close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }

        df = data.copy()
        df.columns = [column_mapping.get(col, col.lower()) for col in df.columns]

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close"]
        for col in required_cols:
            if col not in df.columns:
                if "close" in df.columns:
                    df[col] = df["close"]  # Use close as fallback
                else:
                    raise ValueError(
                        f"Required column '{col}' not found in market data"
                    )

        return df

    elif isinstance(data, dict):
        # Convert dict to DataFrame
        df = pd.DataFrame(data)
        return parse_market_data(df)  # Recursive call with DataFrame

    else:
        raise ValueError(f"Unsupported market data format: {type(data)}")


# Common error handlers
def handle_trading_error(func):
    """Decorator for handling common trading errors"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Trading operation failed in {func.__name__}", error=str(e))
            raise

    return wrapper


# Export all utility functions
__all__ = [
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "validate_symbol",
    "validate_quantity",
    "validate_price",
    "calculate_position_value",
    "calculate_percentage_change",
    "calculate_pnl",
    "calculate_risk_metrics",
    "round_to_tick_size",
    "calculate_stop_loss",
    "calculate_position_size",
    "is_market_open",
    "format_currency",
    "calculate_sharpe_ratio",
    "generate_order_id",
    "parse_market_data",
    "handle_trading_error",
]
