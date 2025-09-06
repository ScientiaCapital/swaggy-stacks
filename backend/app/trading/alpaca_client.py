"""
Alpaca API client for trading operations
"""

import asyncio
from typing import Dict, List, Optional, Any
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError
import structlog
from app.core.config import settings
from app.core.exceptions import TradingError, MarketDataError

logger = structlog.get_logger()


class AlpacaClient:
    """Alpaca API client for trading operations"""
    
    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        self.api_key = api_key or settings.ALPACA_API_KEY
        self.secret_key = secret_key or settings.ALPACA_SECRET_KEY
        self.paper = paper
        
        if not self.api_key or not self.secret_key:
            raise TradingError("Alpaca API credentials not provided")
        
        self.api = tradeapi.REST(
            self.api_key,
            self.secret_key,
            base_url=settings.ALPACA_BASE_URL,
            api_version='v2'
        )
        
        self.data_api = tradeapi.REST(
            self.api_key,
            self.secret_key,
            base_url=settings.ALPACA_DATA_URL,
            api_version='v2'
        )
        
        logger.info("Alpaca client initialized", paper=paper)
    
    async def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                "id": account.id,
                "account_number": account.account_number,
                "status": account.status,
                "currency": account.currency,
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "initial_margin": float(account.initial_margin),
                "maintenance_margin": float(account.maintenance_margin),
                "daytrade_count": account.daytrade_count,
                "pattern_day_trader": account.pattern_day_trader,
            }
        except APIError as e:
            logger.error("Failed to get account info", error=str(e))
            raise TradingError(f"Failed to get account info: {e}")
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            return [
                {
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "side": pos.side,
                    "market_value": float(pos.market_value),
                    "cost_basis": float(pos.cost_basis),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "current_price": float(pos.current_price),
                    "lastday_price": float(pos.lastday_price),
                    "change_today": float(pos.change_today),
                }
                for pos in positions
            ]
        except APIError as e:
            logger.error("Failed to get positions", error=str(e))
            raise TradingError(f"Failed to get positions: {e}")
    
    async def execute_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "gtc",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a trading order"""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id
            )
            
            logger.info(
                "Order submitted",
                symbol=symbol,
                qty=qty,
                side=side,
                order_type=order_type,
                order_id=order.id
            )
            
            return {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side,
                "type": order.type,
                "time_in_force": order.time_in_force,
                "status": order.status,
                "submitted_at": order.submitted_at,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "stop_price": float(order.stop_price) if order.stop_price else None,
            }
        except APIError as e:
            logger.error("Failed to execute order", error=str(e), symbol=symbol)
            raise TradingError(f"Failed to execute order: {e}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.api.cancel_order(order_id)
            logger.info("Order cancelled", order_id=order_id)
            return True
        except APIError as e:
            logger.error("Failed to cancel order", error=str(e), order_id=order_id)
            raise TradingError(f"Failed to cancel order: {e}")
    
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order details"""
        try:
            order = self.api.get_order(order_id)
            return {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side,
                "type": order.type,
                "time_in_force": order.time_in_force,
                "status": order.status,
                "submitted_at": order.submitted_at,
                "filled_at": order.filled_at,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "stop_price": float(order.stop_price) if order.stop_price else None,
            }
        except APIError as e:
            logger.error("Failed to get order", error=str(e), order_id=order_id)
            raise TradingError(f"Failed to get order: {e}")
    
    async def get_orders(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        after: Optional[str] = None,
        until: Optional[str] = None,
        direction: str = "desc"
    ) -> List[Dict[str, Any]]:
        """Get list of orders"""
        try:
            orders = self.api.list_orders(
                status=status,
                limit=limit,
                after=after,
                until=until,
                direction=direction
            )
            
            return [
                {
                    "id": order.id,
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "qty": float(order.qty),
                    "side": order.side,
                    "type": order.type,
                    "time_in_force": order.time_in_force,
                    "status": order.status,
                    "submitted_at": order.submitted_at,
                    "filled_at": order.filled_at,
                    "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                    "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                }
                for order in orders
            ]
        except APIError as e:
            logger.error("Failed to get orders", error=str(e))
            raise TradingError(f"Failed to get orders: {e}")
    
    async def get_market_data(
        self,
        symbols: List[str],
        timeframe: str = "1Min",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 1000
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical market data"""
        try:
            data = {}
            for symbol in symbols:
                bars = self.data_api.get_bars(
                    symbol,
                    timeframe,
                    start=start,
                    end=end,
                    limit=limit
                )
                
                data[symbol] = [
                    {
                        "timestamp": bar.t,
                        "open": float(bar.o),
                        "high": float(bar.h),
                        "low": float(bar.l),
                        "close": float(bar.c),
                        "volume": int(bar.v),
                        "trade_count": int(bar.n) if hasattr(bar, 'n') else None,
                        "vwap": float(bar.vw) if hasattr(bar, 'vw') else None,
                    }
                    for bar in bars
                ]
            
            return data
        except APIError as e:
            logger.error("Failed to get market data", error=str(e), symbols=symbols)
            raise MarketDataError(f"Failed to get market data: {e}")
    
    async def get_latest_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get latest quotes for symbols"""
        try:
            quotes = self.data_api.get_latest_quotes(symbols)
            return {
                symbol: {
                    "bid": float(quote.bid_price) if quote.bid_price else None,
                    "ask": float(quote.ask_price) if quote.ask_price else None,
                    "bid_size": int(quote.bid_size) if quote.bid_size else None,
                    "ask_size": int(quote.ask_size) if quote.ask_size else None,
                    "timestamp": quote.timestamp,
                }
                for symbol, quote in quotes.items()
            }
        except APIError as e:
            logger.error("Failed to get latest quotes", error=str(e), symbols=symbols)
            raise MarketDataError(f"Failed to get latest quotes: {e}")
    
    async def get_latest_trades(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get latest trades for symbols"""
        try:
            trades = self.data_api.get_latest_trades(symbols)
            return {
                symbol: {
                    "price": float(trade.price),
                    "size": int(trade.size),
                    "timestamp": trade.timestamp,
                    "conditions": trade.conditions,
                }
                for symbol, trade in trades.items()
            }
        except APIError as e:
            logger.error("Failed to get latest trades", error=str(e), symbols=symbols)
            raise MarketDataError(f"Failed to get latest trades: {e}")
