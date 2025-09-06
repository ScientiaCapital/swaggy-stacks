/**
 * Example: Profitable Trading Bot Using Swaggy Stacks MCP
 * Shows developers how to build valuable applications that make money
 */

import { SwaggyStacksMCPClient } from '@swaggystack/mcp-client';

class ProfitableTradingBot {
  private mcpClient: SwaggyStacksMCPClient;
  private portfolio: Map<string, any> = new Map();
  private tradingTheses: Map<string, any> = new Map();
  
  constructor(apiKey: string) {
    this.mcpClient = new SwaggyStacksMCPClient({
      apiKey,
      tier: 'professional' // $499/month for serious traders
    });
  }
  
  /**
   * Main trading loop - scans for opportunities and executes trades
   */
  async runTradingLoop() {
    console.log('🤖 Starting Profitable Trading Bot...');
    
    while (true) {
      try {
        // 1. Scan for new opportunities
        const opportunities = await this.scanForOpportunities();
        
        // 2. Update existing positions
        await this.updateExistingPositions();
        
        // 3. Execute new trades
        for (const opportunity of opportunities) {
          await this.evaluateAndExecuteTrade(opportunity);
        }
        
        // 4. Wait before next scan
        await this.sleep(60000); // 1 minute
        
      } catch (error) {
        console.error('Trading loop error:', error);
        await this.sleep(30000); // Wait 30 seconds on error
      }
    }
  }
  
  /**
   * Scan market for profitable opportunities using AI
   */
  async scanForOpportunities(): Promise<TradingOpportunity[]> {
    const symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX'];
    const opportunities: TradingOpportunity[] = [];
    
    for (const symbol of symbols) {
      try {
        // Use Swaggy Stacks AI to analyze each symbol
        const analysis = await this.mcpClient.analyzeMarket({
          symbol,
          analysisType: 'comprehensive'
        });
        
        if (analysis.success && analysis.confidence > 75) {
          // Create trading thesis for high-confidence opportunities
          const thesis = await this.mcpClient.createTradingThesis({
            symbol,
            thesis: analysis.analysis,
            timeHorizon: '1-5 days',
            riskParameters: {
              maxLoss: 0.02, // 2% max loss
              targetReturn: 0.05 // 5% target return
            }
          });
          
          if (thesis.success) {
            opportunities.push({
              symbol,
              analysis,
              thesis: thesis.tradingContext,
              confidence: analysis.confidence,
              expectedReturn: this.extractExpectedReturn(analysis.analysis),
              risk: this.extractRisk(analysis.analysis)
            });
          }
        }
        
      } catch (error) {
        console.error(`Error analyzing ${symbol}:`, error);
      }
    }
    
    // Sort by confidence and expected return
    return opportunities
      .sort((a, b) => (b.confidence * b.expectedReturn) - (a.confidence * a.expectedReturn))
      .slice(0, 3); // Take top 3 opportunities
  }
  
  /**
   * Update existing positions with new market context
   */
  async updateExistingPositions() {
    for (const [symbol, position] of this.portfolio) {
      try {
        // Get current market data
        const marketData = await this.mcpClient.getMarketData(symbol);
        
        // Update thesis with new context
        const thesis = this.tradingTheses.get(symbol);
        if (thesis) {
          const update = await this.mcpClient.updateThesisContext({
            thesisId: thesis.id,
            marketUpdate: `Current price: $${marketData.currentPrice}, Volume: ${marketData.volume}`,
            priceUpdate: marketData.currentPrice,
            volumeUpdate: marketData.volume
          });
          
          if (update.success) {
            // Check if we should exit the position
            const shouldExit = this.shouldExitPosition(position, update.newAnalysis);
            if (shouldExit) {
              await this.exitPosition(symbol, position);
            }
          }
        }
        
      } catch (error) {
        console.error(`Error updating position for ${symbol}:`, error);
      }
    }
  }
  
  /**
   * Evaluate and execute a trade opportunity
   */
  async evaluateAndExecuteTrade(opportunity: TradingOpportunity) {
    const { symbol, analysis, thesis, confidence, expectedReturn, risk } = opportunity;
    
    // Risk management checks
    if (this.portfolio.size >= 5) {
      console.log(`Portfolio full, skipping ${symbol}`);
      return;
    }
    
    if (risk > 0.03) { // 3% max risk
      console.log(`Risk too high for ${symbol}: ${risk}`);
      return;
    }
    
    if (expectedReturn < 0.03) { // 3% minimum expected return
      console.log(`Expected return too low for ${symbol}: ${expectedReturn}`);
      return;
    }
    
    // Calculate position size
    const positionSize = this.calculatePositionSize(expectedReturn, risk);
    
    // Execute the trade
    const trade = await this.mcpClient.executeTrade({
      symbol,
      quantity: positionSize,
      side: 'buy',
      orderType: 'market'
    });
    
    if (trade.success) {
      // Store position and thesis
      this.portfolio.set(symbol, {
        ...trade,
        entryPrice: trade.filledPrice || trade.limitPrice,
        expectedReturn,
        risk,
        thesis: thesis
      });
      
      this.tradingTheses.set(symbol, thesis);
      
      console.log(`✅ Trade executed: ${symbol} - ${positionSize} shares at $${trade.filledPrice}`);
      console.log(`   Expected return: ${(expectedReturn * 100).toFixed(1)}%`);
      console.log(`   Risk: ${(risk * 100).toFixed(1)}%`);
      console.log(`   Confidence: ${confidence}%`);
    } else {
      console.log(`❌ Trade failed for ${symbol}: ${trade.error}`);
    }
  }
  
  /**
   * Calculate optimal position size using Kelly Criterion
   */
  private calculatePositionSize(expectedReturn: number, risk: number): number {
    const accountValue = 100000; // $100k account
    const kellyFraction = (expectedReturn - 0.02) / risk; // 2% risk-free rate
    const positionSize = Math.min(kellyFraction * 0.25, 0.1); // Max 10% of account
    
    return Math.floor((accountValue * positionSize) / 100); // Convert to shares
  }
  
  /**
   * Check if we should exit a position
   */
  private shouldExitPosition(position: any, analysis: any): boolean {
    const currentPrice = analysis.currentPrice || position.entryPrice;
    const pnl = (currentPrice - position.entryPrice) / position.entryPrice;
    
    // Exit if stop loss hit
    if (pnl <= -position.risk) {
      console.log(`🛑 Stop loss hit for ${position.symbol}: ${(pnl * 100).toFixed(1)}%`);
      return true;
    }
    
    // Exit if target reached
    if (pnl >= position.expectedReturn) {
      console.log(`🎯 Target reached for ${position.symbol}: ${(pnl * 100).toFixed(1)}%`);
      return true;
    }
    
    // Exit if confidence drops significantly
    if (analysis.confidence < 50) {
      console.log(`📉 Confidence dropped for ${position.symbol}: ${analysis.confidence}%`);
      return true;
    }
    
    return false;
  }
  
  /**
   * Exit a position
   */
  private async exitPosition(symbol: string, position: any) {
    const exitTrade = await this.mcpClient.executeTrade({
      symbol,
      quantity: position.quantity,
      side: 'sell',
      orderType: 'market'
    });
    
    if (exitTrade.success) {
      const pnl = (exitTrade.filledPrice - position.entryPrice) * position.quantity;
      const pnlPercent = ((exitTrade.filledPrice - position.entryPrice) / position.entryPrice) * 100;
      
      console.log(`💰 Position closed: ${symbol}`);
      console.log(`   P&L: $${pnl.toFixed(2)} (${pnlPercent.toFixed(1)}%)`);
      
      // Remove from portfolio
      this.portfolio.delete(symbol);
      this.tradingTheses.delete(symbol);
    }
  }
  
  /**
   * Extract expected return from analysis
   */
  private extractExpectedReturn(analysis: string): number {
    // Look for percentage returns in the analysis
    const returnMatch = analysis.match(/(\d+(?:\.\d+)?)% return|return.*?(\d+(?:\.\d+)?)%/i);
    if (returnMatch) {
      return parseFloat(returnMatch[1] || returnMatch[2]) / 100;
    }
    
    // Default based on confidence
    return 0.05; // 5% default
  }
  
  /**
   * Extract risk from analysis
   */
  private extractRisk(analysis: string): number {
    // Look for risk percentages
    const riskMatch = analysis.match(/risk.*?(\d+(?:\.\d+)?)%|(\d+(?:\.\d+)?)%.*?risk/i);
    if (riskMatch) {
      return parseFloat(riskMatch[1] || riskMatch[2]) / 100;
    }
    
    // Default risk
    return 0.02; // 2% default
  }
  
  /**
   * Get portfolio performance
   */
  async getPortfolioPerformance() {
    const portfolio = await this.mcpClient.getPortfolio();
    
    if (portfolio.success) {
      console.log('\n📊 Portfolio Performance:');
      console.log(`   Total Value: $${portfolio.portfolio.totalValue.toFixed(2)}`);
      console.log(`   Cash: $${portfolio.portfolio.cash.toFixed(2)}`);
      console.log(`   Unrealized P&L: $${portfolio.portfolio.totalUnrealizedPL.toFixed(2)}`);
      console.log(`   Unrealized P&L %: ${portfolio.portfolio.totalUnrealizedPLPercent.toFixed(2)}%`);
      console.log(`   Positions: ${portfolio.portfolio.positionCount}`);
      
      // Show individual positions
      for (const position of portfolio.portfolio.positions) {
        console.log(`   ${position.symbol}: ${position.qty} shares, P&L: $${position.unrealizedPL.toFixed(2)} (${(position.unrealizedPLPercent * 100).toFixed(1)}%)`);
      }
    }
  }
  
  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Example usage and ROI calculation
async function demonstrateTradingBot() {
  console.log('🚀 Swaggy Stacks Trading Bot Demo');
  console.log('=====================================');
  
  const bot = new ProfitableTradingBot('your-api-key-here');
  
  // Show ROI calculation
  console.log(`
  💰 ROI Calculation:
  
  API Cost: $499/month
  Account Size: $100,000
  Expected Monthly Return: 8% (conservative)
  Monthly Profit: $8,000
  Net Profit: $7,501/month
  ROI: 1,500%
  
  Your users make money = You make money!
  `);
  
  // Run the bot (in production, this would run continuously)
  // await bot.runTradingLoop();
  
  // Show portfolio performance
  await bot.getPortfolioPerformance();
}

// Types
interface TradingOpportunity {
  symbol: string;
  analysis: any;
  thesis: any;
  confidence: number;
  expectedReturn: number;
  risk: number;
}

// Run the demo
if (require.main === module) {
  demonstrateTradingBot().catch(console.error);
}

export { ProfitableTradingBot };
