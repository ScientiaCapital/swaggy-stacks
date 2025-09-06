#!/usr/bin/env python3
"""
Test script to check Alpaca paper account holdings, cash, and positions
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_account_details():
    """Test getting detailed account information"""
    try:
        import alpaca_trade_api as tradeapi
        
        # Get API credentials
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        # Initialize Alpaca client
        api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        
        print("🔍 Fetching account details...")
        
        # Get account information
        account = api.get_account()
        
        print("\n💰 ACCOUNT SUMMARY")
        print("=" * 50)
        print(f"Account Status: {account.status}")
        print(f"Pattern Day Trader: {account.pattern_day_trader}")
        print(f"Trading Blocked: {account.trading_blocked}")
        print(f"Transfers Blocked: {account.transfers_blocked}")
        
        print("\n💵 BALANCE INFORMATION")
        print("=" * 50)
        print(f"Portfolio Value:     ${float(account.portfolio_value):>12,.2f}")
        print(f"Cash Balance:        ${float(account.cash):>12,.2f}")
        print(f"Buying Power:        ${float(account.buying_power):>12,.2f}")
        print(f"Long Market Value:   ${float(account.long_market_value or 0):>12,.2f}")
        print(f"Short Market Value:  ${float(account.short_market_value or 0):>12,.2f}")
        print(f"Equity:             ${float(account.equity):>12,.2f}")
        print(f"Last Equity:        ${float(account.last_equity):>12,.2f}")
        
        # Calculate P&L
        initial_equity = float(account.last_equity)
        current_equity = float(account.equity)
        pnl = current_equity - initial_equity
        pnl_pct = (pnl / initial_equity * 100) if initial_equity > 0 else 0
        
        print(f"\n📊 PROFIT & LOSS")
        print("=" * 50)
        print(f"Unrealized P&L:      ${pnl:>12,.2f} ({pnl_pct:+.2f}%)")
        
        # Get current positions
        print(f"\n📈 CURRENT POSITIONS")
        print("=" * 50)
        
        positions = api.list_positions()
        
        if positions:
            print(f"{'Symbol':<8} {'Qty':<8} {'Side':<6} {'Market Value':<12} {'Unrealized P&L':<15} {'%':<8}")
            print("-" * 70)
            
            total_market_value = 0
            total_unrealized_pnl = 0
            
            for position in positions:
                qty = float(position.qty)
                market_value = float(position.market_value)
                unrealized_pnl = float(position.unrealized_pl)
                unrealized_pnl_pct = float(position.unrealized_plpc) * 100
                side = "LONG" if qty > 0 else "SHORT"
                
                total_market_value += abs(market_value)
                total_unrealized_pnl += unrealized_pnl
                
                print(f"{position.symbol:<8} {qty:<8.0f} {side:<6} ${market_value:<11,.2f} ${unrealized_pnl:<14,.2f} {unrealized_pnl_pct:+.2f}%")
            
            print("-" * 70)
            print(f"{'TOTAL':<8} {'':<8} {'':<6} ${total_market_value:<11,.2f} ${total_unrealized_pnl:<14,.2f}")
            
        else:
            print("No positions currently held")
        
        # Get recent orders
        print(f"\n📋 RECENT ORDERS (Last 10)")
        print("=" * 50)
        
        try:
            orders = api.list_orders(status='all', limit=10)
            
            if orders:
                print(f"{'Symbol':<8} {'Side':<4} {'Qty':<6} {'Type':<8} {'Status':<10} {'Submitted':<20}")
                print("-" * 70)
                
                for order in orders:
                    submitted_at = order.submitted_at.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"{order.symbol:<8} {order.side:<4} {float(order.qty):<6.0f} {order.order_type:<8} {order.status:<10} {submitted_at}")
            else:
                print("No recent orders found")
                
        except Exception as e:
            print(f"Could not fetch orders: {e}")
        
        # Account restrictions and day trading info
        print(f"\n⚠️  ACCOUNT RESTRICTIONS")
        print("=" * 50)
        print(f"Day Trading Buying Power: ${float(account.daytrading_buying_power):,.2f}")
        
        # Some attributes might not be available, handle gracefully
        try:
            print(f"Daytrade Count:          {account.daytrade_count}")
        except AttributeError:
            print("Daytrade Count:          Not available")
            
        print(f"Pattern Day Trader:      {account.pattern_day_trader}")
        print(f"Account Created:         {account.created_at}")
        print(f"Currency:               {account.currency}")
        print(f"Account Type:           Paper Trading Account")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Swaggy Stacks - Account Information Test")
    print("=" * 50)
    
    if test_account_details():
        print("\n✅ Account information retrieved successfully!")
    else:
        print("\n❌ Failed to get account information")
        sys.exit(1)