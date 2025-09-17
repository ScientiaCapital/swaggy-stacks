#!/usr/bin/env python3
"""Check Alpaca account for recent orders and positions"""

import sys
sys.path.append('/Users/tmkipper/repos/swaggy-stacks/backend')

import alpaca_trade_api as tradeapi
from app.core.config import settings
from datetime import datetime

# Initialize Alpaca client
api = tradeapi.REST(
    settings.ALPACA_API_KEY,
    settings.ALPACA_SECRET_KEY,
    base_url=settings.ALPACA_BASE_URL,
    api_version='v2'
)

print('🔍 CHECKING YOUR ALPACA ACCOUNT FOR RECENT ORDERS...')
print('=' * 60)

# Get account info
account = api.get_account()
print(f'💰 Account Status: {account.status}')
print(f'💰 Portfolio Value: ${float(account.portfolio_value):,.2f}')
print(f'💸 Buying Power: ${float(account.buying_power):,.2f}')
crypto_status = getattr(account, 'crypto_status', 'N/A')
print(f'🪙 Crypto Status: {crypto_status}')
print()

# Get all orders (open and closed)
print('📊 RECENT ORDERS:')
print('-' * 60)
orders = api.list_orders(status='all', limit=10)

if orders:
    for order in orders:
        print(f'Order ID: {order.id}')
        print(f'  Symbol: {order.symbol}')
        print(f'  Side: {order.side}')
        print(f'  Type: {order.type}')
        print(f'  Status: {order.status}')
        if hasattr(order, 'qty') and order.qty:
            print(f'  Quantity: {order.qty}')
        if hasattr(order, 'notional') and order.notional:
            print(f'  Notional: ${order.notional}')
        print(f'  Submitted: {order.submitted_at}')
        if hasattr(order, 'filled_at') and order.filled_at:
            print(f'  Filled: {order.filled_at}')
        if hasattr(order, 'filled_qty') and order.filled_qty:
            print(f'  Filled Qty: {order.filled_qty}')
        if hasattr(order, 'filled_avg_price') and order.filled_avg_price:
            print(f'  Avg Price: ${order.filled_avg_price}')
        print('-' * 60)
else:
    print('No orders found')

# Get current positions
print()
print('📈 CURRENT POSITIONS:')
print('-' * 60)
positions = api.list_positions()
if positions:
    for pos in positions:
        print(f'Symbol: {pos.symbol}')
        print(f'  Quantity: {pos.qty}')
        print(f'  Market Value: ${float(pos.market_value):,.2f}')
        print(f'  Avg Cost: ${float(pos.avg_entry_price):,.2f}')
        unrealized_pl = float(pos.unrealized_pl)
        print(f'  Unrealized P&L: ${unrealized_pl:,.2f}')
        print('-' * 60)
else:
    print('No open positions')

print()
print('✅ Check complete!')