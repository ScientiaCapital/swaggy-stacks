"""
Subscription and Billing Management System
Handles subscription plans, payments, and usage tracking for SwaggyStacks API
"""

import stripe
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import uuid
from enum import Enum

# Initialize Stripe (replace with your actual keys)
stripe.api_key = "sk_test_your_stripe_secret_key_here"  # Use environment variable in production

class SubscriptionTier(Enum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class PaymentStatus(Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"

class SubscriptionManager:
    """
    Comprehensive subscription and billing management system
    """
    
    def __init__(self):
        self.subscription_plans = {
            SubscriptionTier.FREE: {
                "name": "Free",
                "price": 0,
                "monthly_quota": 100,
                "rate_limit": 10,  # requests per minute
                "features": [
                    "basic_stock_analysis",
                    "limited_signals",
                    "community_support"
                ],
                "stripe_price_id": None
            },
            SubscriptionTier.BASIC: {
                "name": "Basic",
                "price": 49,
                "monthly_quota": 1000,
                "rate_limit": 60,
                "features": [
                    "advanced_stock_analysis",
                    "portfolio_analysis",
                    "basic_signals",
                    "email_support"
                ],
                "stripe_price_id": "price_basic_monthly"
            },
            SubscriptionTier.PRO: {
                "name": "Professional",
                "price": 199,
                "monthly_quota": 10000,
                "rate_limit": 300,
                "features": [
                    "all_analysis_types",
                    "trading_signals",
                    "backtesting",
                    "api_access",
                    "priority_support"
                ],
                "stripe_price_id": "price_pro_monthly"
            },
            SubscriptionTier.ENTERPRISE: {
                "name": "Enterprise",
                "price": 999,
                "monthly_quota": 100000,
                "rate_limit": 1000,
                "features": [
                    "all_features",
                    "mcp_access",
                    "custom_models",
                    "dedicated_support",
                    "sla_guarantee",
                    "white_label_options"
                ],
                "stripe_price_id": "price_enterprise_monthly"
            }
        }
        
        # Usage-based pricing
        self.usage_pricing = {
            "analyze_stock_basic": 0.001,
            "analyze_stock_standard": 0.003,
            "analyze_stock_advanced": 0.005,
            "analyze_portfolio": 0.05,
            "generate_signals": 0.01,
            "backtest_strategy": 0.10,
            "market_regime_analysis": 0.002,
            "mcp_access": 0.02  # per minute
        }
        
        # Overage rates by tier
        self.overage_rates = {
            SubscriptionTier.FREE: 0.02,
            SubscriptionTier.BASIC: 0.015,
            SubscriptionTier.PRO: 0.01,
            SubscriptionTier.ENTERPRISE: 0.005
        }
    
    async def create_subscription(self, user_id: str, tier: SubscriptionTier, 
                                payment_method_id: str, customer_email: str) -> Dict[str, Any]:
        """
        Create a new subscription
        
        Args:
            user_id: User ID
            tier: Subscription tier
            payment_method_id: Stripe payment method ID
            customer_email: Customer email
            
        Returns:
            Subscription creation result
        """
        try:
            plan = self.subscription_plans[tier]
            
            if tier == SubscriptionTier.FREE:
                # Free tier doesn't require payment
                subscription = await self._create_free_subscription(user_id, customer_email)
                return {
                    "success": True,
                    "subscription_id": subscription["id"],
                    "tier": tier.value,
                    "status": "active",
                    "message": "Free subscription created successfully"
                }
            
            # Create Stripe customer
            customer = stripe.Customer.create(
                email=customer_email,
                payment_method=payment_method_id,
                invoice_settings={
                    'default_payment_method': payment_method_id,
                },
            )
            
            # Create subscription
            subscription = stripe.Subscription.create(
                customer=customer.id,
                items=[{
                    'price': plan["stripe_price_id"],
                }],
                payment_behavior='default_incomplete',
                payment_settings={'save_default_payment_method': 'on_subscription'},
                expand=['latest_invoice.payment_intent'],
            )
            
            # Store subscription in database
            subscription_data = {
                "id": subscription.id,
                "user_id": user_id,
                "tier": tier.value,
                "status": subscription.status,
                "stripe_customer_id": customer.id,
                "stripe_subscription_id": subscription.id,
                "created_at": datetime.now(),
                "current_period_start": datetime.fromtimestamp(subscription.current_period_start),
                "current_period_end": datetime.fromtimestamp(subscription.current_period_end),
                "monthly_quota": plan["monthly_quota"],
                "rate_limit": plan["rate_limit"],
                "requests_this_month": 0
            }
            
            await self._store_subscription(subscription_data)
            
            return {
                "success": True,
                "subscription_id": subscription.id,
                "tier": tier.value,
                "status": subscription.status,
                "client_secret": subscription.latest_invoice.payment_intent.client_secret,
                "message": "Subscription created successfully"
            }
            
        except stripe.error.StripeError as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Payment processing failed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Subscription creation failed"
            }
    
    async def upgrade_subscription(self, user_id: str, new_tier: SubscriptionTier) -> Dict[str, Any]:
        """
        Upgrade a user's subscription
        
        Args:
            user_id: User ID
            new_tier: New subscription tier
            
        Returns:
            Upgrade result
        """
        try:
            # Get current subscription
            current_subscription = await self._get_user_subscription(user_id)
            if not current_subscription:
                return {
                    "success": False,
                    "error": "No active subscription found",
                    "message": "Please create a subscription first"
                }
            
            current_tier = SubscriptionTier(current_subscription["tier"])
            current_plan = self.subscription_plans[current_tier]
            new_plan = self.subscription_plans[new_tier]
            
            # Calculate prorated upgrade cost
            upgrade_cost = self._calculate_prorated_upgrade_cost(
                current_subscription, current_plan, new_plan
            )
            
            if upgrade_cost > 0:
                # Process upgrade payment
                payment_success = await self._process_upgrade_payment(
                    current_subscription["stripe_customer_id"], 
                    upgrade_cost
                )
                
                if not payment_success:
                    return {
                        "success": False,
                        "error": "Upgrade payment failed",
                        "message": "Please check your payment method"
                    }
            
            # Update Stripe subscription
            stripe_subscription = stripe.Subscription.retrieve(
                current_subscription["stripe_subscription_id"]
            )
            
            stripe.Subscription.modify(
                stripe_subscription.id,
                items=[{
                    'id': stripe_subscription['items']['data'][0].id,
                    'price': new_plan["stripe_price_id"],
                }],
                proration_behavior='create_prorations',
            )
            
            # Update database
            await self._update_subscription_tier(
                user_id, 
                new_tier.value, 
                new_plan["monthly_quota"], 
                new_plan["rate_limit"]
            )
            
            return {
                "success": True,
                "old_tier": current_tier.value,
                "new_tier": new_tier.value,
                "upgrade_cost": upgrade_cost,
                "message": "Subscription upgraded successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Subscription upgrade failed"
            }
    
    async def cancel_subscription(self, user_id: str) -> Dict[str, Any]:
        """
        Cancel a user's subscription
        
        Args:
            user_id: User ID
            
        Returns:
            Cancellation result
        """
        try:
            subscription = await self._get_user_subscription(user_id)
            if not subscription:
                return {
                    "success": False,
                    "error": "No active subscription found",
                    "message": "No subscription to cancel"
                }
            
            # Cancel Stripe subscription
            stripe.Subscription.modify(
                subscription["stripe_subscription_id"],
                cancel_at_period_end=True
            )
            
            # Update database
            await self._cancel_subscription(user_id)
            
            return {
                "success": True,
                "message": "Subscription will be canceled at the end of the current period"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Subscription cancellation failed"
            }
    
    async def add_credits(self, user_id: str, amount: float, payment_method_id: str) -> Dict[str, Any]:
        """
        Add credits to user's account
        
        Args:
            user_id: User ID
            amount: Amount to add
            payment_method_id: Stripe payment method ID
            
        Returns:
            Credit addition result
        """
        try:
            # Get user's Stripe customer ID
            subscription = await self._get_user_subscription(user_id)
            if not subscription:
                return {
                    "success": False,
                    "error": "No subscription found",
                    "message": "Please create a subscription first"
                }
            
            # Create payment intent
            intent = stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Convert to cents
                currency='usd',
                customer=subscription["stripe_customer_id"],
                payment_method=payment_method_id,
                confirmation_method='manual',
                confirm=True,
            )
            
            if intent.status == 'succeeded':
                # Update user's credit balance
                await self._add_user_credits(user_id, amount)
                
                return {
                    "success": True,
                    "amount_added": amount,
                    "new_balance": await self._get_user_balance(user_id),
                    "message": "Credits added successfully"
                }
            else:
                return {
                    "success": False,
                    "error": "Payment not completed",
                    "message": "Please try again"
                }
                
        except stripe.error.StripeError as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Payment processing failed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Credit addition failed"
            }
    
    async def track_usage(self, user_id: str, endpoint: str, cost: float) -> Dict[str, Any]:
        """
        Track API usage and calculate billing
        
        Args:
            user_id: User ID
            endpoint: API endpoint used
            cost: Cost of the API call
            
        Returns:
            Usage tracking result
        """
        try:
            subscription = await self._get_user_subscription(user_id)
            if not subscription:
                return {
                    "success": False,
                    "error": "No active subscription",
                    "message": "Please create a subscription first"
                }
            
            tier = SubscriptionTier(subscription["tier"])
            monthly_quota = subscription["monthly_quota"]
            requests_this_month = subscription["requests_this_month"]
            
            # Check if user has exceeded monthly quota
            if requests_this_month >= monthly_quota:
                # Calculate overage cost
                overage_rate = self.overage_rates[tier]
                overage_cost = cost * overage_rate
                
                # Check if user has sufficient balance for overage
                user_balance = await self._get_user_balance(user_id)
                if user_balance < overage_cost:
                    return {
                        "success": False,
                        "error": "Insufficient balance for overage",
                        "message": "Please add credits to your account",
                        "required_balance": overage_cost,
                        "current_balance": user_balance
                    }
                
                # Deduct overage cost from balance
                await self._deduct_user_balance(user_id, overage_cost)
            
            # Update usage statistics
            await self._update_usage_stats(user_id, endpoint, cost)
            
            return {
                "success": True,
                "cost": cost,
                "is_overage": requests_this_month >= monthly_quota,
                "remaining_quota": max(0, monthly_quota - requests_this_month - 1),
                "message": "Usage tracked successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Usage tracking failed"
            }
    
    async def get_usage_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's usage summary
        
        Args:
            user_id: User ID
            
        Returns:
            Usage summary
        """
        try:
            subscription = await self._get_user_subscription(user_id)
            if not subscription:
                return {
                    "success": False,
                    "error": "No active subscription",
                    "message": "Please create a subscription first"
                }
            
            tier = SubscriptionTier(subscription["tier"])
            plan = self.subscription_plans[tier]
            
            # Get usage statistics
            usage_stats = await self._get_usage_statistics(user_id)
            
            return {
                "success": True,
                "subscription": {
                    "tier": tier.value,
                    "plan_name": plan["name"],
                    "monthly_quota": plan["monthly_quota"],
                    "rate_limit": plan["rate_limit"],
                    "features": plan["features"]
                },
                "usage": {
                    "requests_this_month": subscription["requests_this_month"],
                    "remaining_quota": max(0, plan["monthly_quota"] - subscription["requests_this_month"]),
                    "quota_usage_percent": (subscription["requests_this_month"] / plan["monthly_quota"]) * 100
                },
                "billing": {
                    "current_balance": await self._get_user_balance(user_id),
                    "overage_rate": self.overage_rates[tier],
                    "next_billing_date": subscription["current_period_end"]
                },
                "statistics": usage_stats
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to get usage summary"
            }
    
    def calculate_cost(self, endpoint: str, parameters: Dict[str, Any] = None) -> float:
        """
        Calculate the cost of an API call
        
        Args:
            endpoint: API endpoint
            parameters: Request parameters
            
        Returns:
            Cost in dollars
        """
        base_cost = self.usage_pricing.get(endpoint, 0.01)
        
        # Adjust cost based on parameters
        if endpoint == "analyze_stock":
            depth = parameters.get("depth", "standard") if parameters else "standard"
            if depth == "basic":
                return self.usage_pricing["analyze_stock_basic"]
            elif depth == "advanced":
                return self.usage_pricing["analyze_stock_advanced"]
            else:
                return self.usage_pricing["analyze_stock_standard"]
        
        return base_cost
    
    def estimate_monthly_revenue(self, user_counts: Dict[str, int], 
                               average_usage: Dict[str, float]) -> float:
        """
        Estimate monthly revenue based on user counts and usage patterns
        
        Args:
            user_counts: Number of users per tier
            average_usage: Average usage per tier
            
        Returns:
            Estimated monthly revenue
        """
        revenue = 0
        
        for tier_name, count in user_counts.items():
            tier = SubscriptionTier(tier_name)
            plan = self.subscription_plans[tier]
            
            # Subscription revenue
            revenue += count * plan["price"]
            
            # Overage revenue
            if tier_name in average_usage:
                included_credits = plan["monthly_quota"]
                average_credits = average_usage[tier_name]
                
                if average_credits > included_credits:
                    overage = average_credits - included_credits
                    revenue += count * overage * self.overage_rates[tier]
        
        return revenue
    
    # Private helper methods
    async def _create_free_subscription(self, user_id: str, email: str) -> Dict[str, Any]:
        """Create a free subscription"""
        subscription_id = str(uuid.uuid4())
        
        subscription_data = {
            "id": subscription_id,
            "user_id": user_id,
            "tier": SubscriptionTier.FREE.value,
            "status": "active",
            "created_at": datetime.now(),
            "monthly_quota": self.subscription_plans[SubscriptionTier.FREE]["monthly_quota"],
            "rate_limit": self.subscription_plans[SubscriptionTier.FREE]["rate_limit"],
            "requests_this_month": 0
        }
        
        await self._store_subscription(subscription_data)
        return subscription_data
    
    def _calculate_prorated_upgrade_cost(self, subscription: Dict[str, Any], 
                                       current_plan: Dict[str, Any], 
                                       new_plan: Dict[str, Any]) -> float:
        """Calculate prorated upgrade cost"""
        current_price = current_plan["price"]
        new_price = new_plan["price"]
        
        if new_price <= current_price:
            return 0  # Downgrade or same price
        
        # Calculate days remaining in current period
        current_period_end = subscription["current_period_end"]
        days_remaining = (current_period_end - datetime.now()).days
        
        if days_remaining <= 0:
            return new_price - current_price
        
        # Calculate prorated cost
        days_in_period = 30  # Assume 30-day billing cycle
        prorated_cost = (new_price - current_price) * (days_remaining / days_in_period)
        
        return prorated_cost
    
    async def _process_upgrade_payment(self, customer_id: str, amount: float) -> bool:
        """Process upgrade payment"""
        try:
            # Create payment intent for upgrade
            intent = stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Convert to cents
                currency='usd',
                customer=customer_id,
                confirmation_method='automatic',
            )
            
            return intent.status == 'succeeded'
        except:
            return False
    
    # Database methods (replace with actual database integration)
    async def _store_subscription(self, subscription_data: Dict[str, Any]):
        """Store subscription in database"""
        # Replace with actual database integration
        print(f"Storing subscription: {subscription_data['id']}")
    
    async def _get_user_subscription(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user's subscription from database"""
        # Replace with actual database integration
        # Mock implementation
        return None
    
    async def _update_subscription_tier(self, user_id: str, tier: str, 
                                      monthly_quota: int, rate_limit: int):
        """Update subscription tier"""
        # Replace with actual database integration
        print(f"Updating subscription tier for {user_id} to {tier}")
    
    async def _cancel_subscription(self, user_id: str):
        """Cancel subscription"""
        # Replace with actual database integration
        print(f"Canceling subscription for {user_id}")
    
    async def _add_user_credits(self, user_id: str, amount: float):
        """Add credits to user account"""
        # Replace with actual database integration
        print(f"Adding ${amount} credits to {user_id}")
    
    async def _get_user_balance(self, user_id: str) -> float:
        """Get user's credit balance"""
        # Replace with actual database integration
        return 0.0
    
    async def _deduct_user_balance(self, user_id: str, amount: float):
        """Deduct credits from user account"""
        # Replace with actual database integration
        print(f"Deducting ${amount} from {user_id}")
    
    async def _update_usage_stats(self, user_id: str, endpoint: str, cost: float):
        """Update usage statistics"""
        # Replace with actual database integration
        print(f"Updating usage stats for {user_id}: {endpoint} - ${cost}")
    
    async def _get_usage_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get usage statistics"""
        # Replace with actual database integration
        return {
            "total_requests": 0,
            "total_cost": 0.0,
            "endpoint_usage": {},
            "daily_usage": []
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_subscription_manager():
        """Test the subscription manager"""
        manager = SubscriptionManager()
        
        # Test cost calculation
        cost = manager.calculate_cost("analyze_stock", {"depth": "advanced"})
        print(f"Cost for advanced stock analysis: ${cost}")
        
        # Test revenue estimation
        user_counts = {
            "free": 1000,
            "basic": 500,
            "pro": 100,
            "enterprise": 10
        }
        
        average_usage = {
            "free": 50,
            "basic": 800,
            "pro": 8000,
            "enterprise": 80000
        }
        
        revenue = manager.estimate_monthly_revenue(user_counts, average_usage)
        print(f"Estimated monthly revenue: ${revenue:,.2f}")
        
        # Test subscription creation (mock)
        result = await manager.create_subscription(
            "user123", 
            SubscriptionTier.BASIC, 
            "pm_test_payment_method", 
            "test@example.com"
        )
        print(f"Subscription creation result: {result}")
    
    # Run test
    asyncio.run(test_subscription_manager())
