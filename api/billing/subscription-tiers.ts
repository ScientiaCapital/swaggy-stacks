/**
 * Subscription Tiers and Billing Configuration
 * Defines different pricing tiers for the Swaggy Stacks API
 */

export interface SubscriptionTier {
  name: string;
  monthlyPrice: number;
  includedCalls: Record<string, number>;
  rateLimit: string;
  support: string;
  features: string[];
  mcpAccess?: boolean;
}

export class SubscriptionBilling {
  private tiers: Record<string, SubscriptionTier> = {
    'developer': {
      monthlyPrice: 99,
      includedCalls: {
        'basic_analysis': 10000,
        'pattern_recognition': 1000,
        'multi_model_analysis': 100
      },
      rateLimit: '100 calls/minute',
      support: 'community',
      features: [
        'basic_analysis',
        'pattern_recognition',
        'email_support',
        'api_documentation'
      ]
    },
    
    'professional': {
      monthlyPrice: 499,
      includedCalls: {
        'basic_analysis': 100000,
        'pattern_recognition': 10000,
        'multi_model_analysis': 5000,
        'portfolio_analysis': 100
      },
      rateLimit: '1000 calls/minute',
      support: 'email',
      features: [
        'all_developer_features',
        'multi_model_analysis',
        'portfolio_analysis',
        'bulk_analysis',
        'webhook_alerts',
        'custom_models',
        'priority_support',
        'advanced_analytics'
      ]
    },
    
    'enterprise': {
      monthlyPrice: 4999,
      includedCalls: 'unlimited',
      rateLimit: 'custom',
      support: 'dedicated',
      features: [
        'all_professional_features',
        'kimi_k2_orchestration',
        'custom_deployment',
        'sla_guarantee',
        'dedicated_support',
        'custom_integrations',
        'white_label_options'
      ],
      mcpAccess: true
    }
  };

  /**
   * Validate subscription access for an endpoint
   */
  async validateSubscriptionAccess(client: APIClient, endpoint: string): Promise<boolean> {
    const tier = this.tiers[client.subscriptionTier];
    
    if (!tier) {
      throw new Error(`Invalid subscription tier: ${client.subscriptionTier}`);
    }
    
    // Check if endpoint is available in tier
    if (!this.endpointAvailableInTier(endpoint, client.subscriptionTier)) {
      throw new UpgradeRequiredError({
        currentTier: client.subscriptionTier,
        requiredTier: this.getRequiredTier(endpoint),
        upgradeUrl: 'https://swaggystack.com/api/upgrade'
      });
    }
    
    // Check usage limits
    const usage = await this.getMonthlyUsage(client.id, endpoint);
    const limit = tier.includedCalls[endpoint] || Infinity;
    
    if (usage >= limit) {
      return this.handleOverage(client, endpoint);
    }
    
    return true;
  }

  /**
   * Check if endpoint is available in tier
   */
  private endpointAvailableInTier(endpoint: string, tier: string): boolean {
    const tierConfig = this.tiers[tier];
    
    if (!tierConfig) return false;
    
    // Enterprise tier has access to everything
    if (tier === 'enterprise') return true;
    
    // Check if endpoint is in included calls
    return endpoint in tierConfig.includedCalls;
  }

  /**
   * Get required tier for an endpoint
   */
  private getRequiredTier(endpoint: string): string {
    // Define which tier is required for each endpoint
    const endpointTiers = {
      'basic_analysis': 'developer',
      'pattern_recognition': 'developer',
      'multi_model_analysis': 'professional',
      'portfolio_analysis': 'professional',
      'kimi_k2_orchestration': 'enterprise'
    };
    
    return endpointTiers[endpoint] || 'developer';
  }

  /**
   * Handle overage charges
   */
  private async handleOverage(client: APIClient, endpoint: string): Promise<boolean> {
    const tier = this.tiers[client.subscriptionTier];
    const usage = await this.getMonthlyUsage(client.id, endpoint);
    const limit = tier.includedCalls[endpoint];
    
    // Calculate overage
    const overage = usage - limit;
    
    // Overage pricing
    const overagePricing = {
      'basic_analysis': 0.0005,      // $0.0005 per call
      'pattern_recognition': 0.002,  // $0.002 per call
      'multi_model_analysis': 0.005, // $0.005 per call
      'portfolio_analysis': 0.05,    // $0.05 per call
      'kimi_k2_orchestration': 0.50  // $0.50 per call
    };
    
    const overageCost = overage * (overagePricing[endpoint] || 0.001);
    
    // Check if client has sufficient balance for overage
    if (client.balance >= overageCost) {
      // Deduct overage cost
      await this.deductBalance(client.id, overageCost);
      return true;
    } else {
      // Insufficient balance for overage
      throw new InsufficientCreditsError({
        required: overageCost,
        available: client.balance,
        overage: overage,
        topUpUrl: 'https://api.swaggystack.com/billing/topup'
      });
    }
  }

  /**
   * Get monthly usage for a client and endpoint
   */
  private async getMonthlyUsage(clientId: string, endpoint: string): Promise<number> {
    // This would query your database for actual usage
    // For now, return mock data
    return Math.floor(Math.random() * 1000);
  }

  /**
   * Deduct balance for overage
   */
  private async deductBalance(clientId: string, amount: number): Promise<void> {
    // This would update the client's balance in your database
    console.log(`Deducting ${amount} from client ${clientId} for overage`);
  }

  /**
   * Get tier information
   */
  getTier(tierName: string): SubscriptionTier | null {
    return this.tiers[tierName] || null;
  }

  /**
   * Get all tiers
   */
  getAllTiers(): Record<string, SubscriptionTier> {
    return this.tiers;
  }

  /**
   * Calculate upgrade cost
   */
  calculateUpgradeCost(currentTier: string, targetTier: string): number {
    const current = this.tiers[currentTier];
    const target = this.tiers[targetTier];
    
    if (!current || !target) {
      throw new Error('Invalid tier specified');
    }
    
    return target.monthlyPrice - current.monthlyPrice;
  }

  /**
   * Get tier comparison
   */
  getTierComparison(): any[] {
    return Object.entries(this.tiers).map(([name, tier]) => ({
      name,
      monthlyPrice: tier.monthlyPrice,
      features: tier.features,
      rateLimit: tier.rateLimit,
      support: tier.support,
      mcpAccess: tier.mcpAccess || false
    }));
  }
}

// Usage-based pricing for pay-per-call model
export class UsageBasedBilling {
  private computeCosts = {
    'qwen_token': 0.000001,        // $0.000001 per token
    'moonshot_pattern': 0.00001,    // $0.00001 per pattern analyzed
    'deepseek_calculation': 0.00005, // $0.00005 per calculation
    'kimi_k2_second': 0.10,         // $0.10 per compute second
    'mooncake_cache_gb_hour': 0.01  // $0.01 per GB-hour cached
  };

  /**
   * Calculate usage charge based on actual resources consumed
   */
  async calculateUsageCharge(session: APISession): Promise<UsageCharge> {
    let totalCharge = 0;
    
    // Calculate token usage
    totalCharge += session.tokensProcessed.qwen * this.computeCosts.qwen_token;
    
    // Calculate pattern analysis
    totalCharge += session.patternsAnalyzed * this.computeCosts.moonshot_pattern;
    
    // Calculate computational time
    totalCharge += session.computeSeconds.kimi_k2 * this.computeCosts.kimi_k2_second;
    
    // Calculate cache usage
    const cacheHours = session.cacheUsageGB * session.sessionDurationHours;
    totalCharge += cacheHours * this.computeCosts.mooncake_cache_gb_hour;
    
    // Apply volume discounts
    if (totalCharge > 1000) {
      totalCharge *= 0.9; // 10% discount for high usage
    }
    
    return {
      baseCharge: totalCharge,
      breakdown: this.generateCostBreakdown(session),
      invoice: this.generateInvoice(session, totalCharge)
    };
  }

  /**
   * Generate cost breakdown
   */
  private generateCostBreakdown(session: APISession): CostBreakdown {
    return {
      qwenTokens: {
        count: session.tokensProcessed.qwen,
        cost: session.tokensProcessed.qwen * this.computeCosts.qwen_token
      },
      moonshotPatterns: {
        count: session.patternsAnalyzed,
        cost: session.patternsAnalyzed * this.computeCosts.moonshot_pattern
      },
      deepseekCalculations: {
        count: session.calculationsPerformed,
        cost: session.calculationsPerformed * this.computeCosts.deepseek_calculation
      },
      kimiK2Seconds: {
        count: session.computeSeconds.kimi_k2,
        cost: session.computeSeconds.kimi_k2 * this.computeCosts.kimi_k2_second
      },
      cacheUsage: {
        gbHours: session.cacheUsageGB * session.sessionDurationHours,
        cost: session.cacheUsageGB * session.sessionDurationHours * this.computeCosts.mooncake_cache_gb_hour
      }
    };
  }

  /**
   * Generate invoice
   */
  private generateInvoice(session: APISession, totalCharge: number): Invoice {
    return {
      invoiceId: `INV-${Date.now()}`,
      clientId: session.clientId,
      date: new Date().toISOString(),
      totalAmount: totalCharge,
      breakdown: this.generateCostBreakdown(session),
      paymentStatus: 'pending'
    };
  }
}

// Types
interface APIClient {
  id: string;
  subscriptionTier: string;
  balance: number;
  ratePerSecond?: number;
}

interface APISession {
  clientId: string;
  tokensProcessed: {
    qwen: number;
    moonshot: number;
    deepseek: number;
  };
  patternsAnalyzed: number;
  calculationsPerformed: number;
  computeSeconds: {
    kimi_k2: number;
  };
  cacheUsageGB: number;
  sessionDurationHours: number;
}

interface UsageCharge {
  baseCharge: number;
  breakdown: CostBreakdown;
  invoice: Invoice;
}

interface CostBreakdown {
  qwenTokens: { count: number; cost: number };
  moonshotPatterns: { count: number; cost: number };
  deepseekCalculations: { count: number; cost: number };
  kimiK2Seconds: { count: number; cost: number };
  cacheUsage: { gbHours: number; cost: number };
}

interface Invoice {
  invoiceId: string;
  clientId: string;
  date: string;
  totalAmount: number;
  breakdown: CostBreakdown;
  paymentStatus: string;
}

class UpgradeRequiredError extends Error {
  constructor(public details: any) {
    super('Upgrade required for this endpoint');
  }
}

class InsufficientCreditsError extends Error {
  constructor(public details: any) {
    super('Insufficient credits for this operation');
  }
}
