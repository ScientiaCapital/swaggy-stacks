#!/usr/bin/env python3
"""
Check actual trades executed in Alpaca account
"""

from alpaca.trading.client import TradingClient
from app.core.config import settings
from datetime import datetime, timedelta

def check_real_trades():
    """Check actual trades in Alpaca account"""

    print('🔍 CHECKING REAL ALPACA TRADES')
    print('=' * 50)

    try:
        # Connect to Alpaca
        client = TradingClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            paper=True
        )

        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import OrderStatus

        # Get all recent orders
        orders = client.get_orders()

        if orders:
            print(f'📊 Found {len(orders)} total orders')

            # Filter for today's orders
            today = datetime.now().date()
            today_orders = [order for order in orders if order.created_at.date() == today]

            print(f'📅 Today\'s orders: {len(today_orders)}')

            if today_orders:
                for order in today_orders:
                    print(f'   {order.created_at.time()}: {order.symbol} {order.side} ${order.notional or order.qty} - {order.status}')
            else:
                print('❌ No trades executed today')
                print('   Last few orders:')
                for order in orders[:5]:
                    print(f'   {order.created_at.date()}: {order.symbol} {order.side} ${order.notional or order.qty} - {order.status}')
        else:
            print('❌ No orders found')

        # Check account positions
        positions = client.get_all_positions()
        if positions:
            print(f'\n💰 Current positions: {len(positions)}')
            for pos in positions:
                print(f'   {pos.symbol}: {pos.qty} shares, Value: ${pos.market_value}')
        else:
            print('\n💰 No current positions')

        # Check account info
        account = client.get_account()
        print(f'\n📊 Account Status:')
        print(f'   Portfolio Value: ${account.portfolio_value}')
        print(f'   Buying Power: ${account.buying_power}')
        print(f'   Day Trading Buying Power: ${account.day_trading_buying_power}')

    except Exception as e:
        print(f'❌ Error checking trades: {e}')

if __name__ == "__main__":
    check_real_trades()