/**
 * Alpaca Trading Tool for Swaggy Stacks MCP Server
 * Integrates with existing Alpaca paper trading setup
 */

const { AlpacaClient } = require('@alpacahq/alpaca-trade-api');
const config = require('../config/alpaca-config');
const { v4: uuidv4 } = require('uuid');

class AlpacaTradingTool {
  constructor() {
    this.client = new AlpacaClient({
      key: config.apiKey,
      secret: config.secretKey,
      paper: config.paperTrading,
      baseUrl: config.baseURL
    });
    
    this.tradeHistory = new Map(); // In-memory trade tracking
    this.performanceMetrics = new Map(); // Performance tracking
  }

  /**
   * Execute a paper trade through Alpaca
   * @param {string} symbol - Stock symbol
   * @param {number} qty - Quantity of shares
   * @param {string} side - 'buy' or 'sell'
   * @param {string} type - Order type (market, limit, stop)
   * @param {string} time_in_force - Time in force
   * @param {number} limit_price - Limit price (for limit orders)
   * @param {number} stop_price - Stop price (for stop orders)
   */
  async executeTrade({
    symbol,
    qty,
    side,
    type = 'market',
    time_in_force = 'day',
    limit_price = null,
    stop_price = null
  }) {
    try {
      // Validate trade parameters
      const validation = await this.validateTrade(symbol, qty, side);
      if (!validation.valid) {
        return {
          success: false,
          error: validation.error,
          tradeId: null
        };
      }

      // Create order object
      const orderParams = {
        symbol: symbol.toUpperCase(),
        qty: Math.abs(qty),
        side: side.toLowerCase(),
        type: type.toLowerCase(),
        time_in_force: time_in_force.toLowerCase()
      };

      // Add optional parameters
      if (limit_price) orderParams.limit_price = limit_price;
      if (stop_price) orderParams.stop_price = stop_price;

      // Execute the order
      const order = await this.client.createOrder(orderParams);
      
      // Generate trade ID for tracking
      const tradeId = uuidv4();
      
      // Store trade information
      this.tradeHistory.set(tradeId, {
        tradeId,
        orderId: order.id,
        symbol: symbol.toUpperCase(),
        qty,
        side,
        type,
        status: order.status,
        submittedAt: new Date(),
        orderParams
      });

      // Update performance metrics
      await this.updatePerformanceMetrics(symbol, side, qty);

      return {
        success: true,
        tradeId,
        orderId: order.id,
        status: order.status,
        symbol: symbol.toUpperCase(),
        qty,
        side,
        type,
        submittedAt: new Date().toISOString(),
        message: `Order placed successfully for ${qty} shares of ${symbol}`
      };

    } catch (error) {
      console.error('Trade execution error:', error);
      return {
        success: false,
        error: error.message,
        tradeId: null
      };
    }
  }

  /**
   * Get current portfolio information
   */
  async getPortfolio() {
    try {
      const account = await this.client.getAccount();
      const positions = await this.client.getPositions();
      
      // Calculate portfolio metrics
      const portfolioValue = parseFloat(account.portfolio_value);
      const cash = parseFloat(account.cash);
      const buyingPower = parseFloat(account.buying_power);
      
      // Process positions
      const processedPositions = positions.map(position => ({
        symbol: position.symbol,
        qty: parseFloat(position.qty),
        side: parseFloat(position.qty) > 0 ? 'long' : 'short',
        marketValue: parseFloat(position.market_value),
        costBasis: parseFloat(position.cost_basis),
        unrealizedPL: parseFloat(position.unrealized_pl),
        unrealizedPLPercent: parseFloat(position.unrealized_plpc),
        currentPrice: parseFloat(position.current_price),
        avgEntryPrice: parseFloat(position.avg_entry_price)
      }));

      // Calculate total unrealized P&L
      const totalUnrealizedPL = processedPositions.reduce((sum, pos) => sum + pos.unrealizedPL, 0);
      const totalUnrealizedPLPercent = portfolioValue > 0 ? (totalUnrealizedPL / portfolioValue) * 100 : 0;

      return {
        success: true,
        portfolio: {
          totalValue: portfolioValue,
          cash: cash,
          buyingPower: buyingPower,
          totalUnrealizedPL: totalUnrealizedPL,
          totalUnrealizedPLPercent: totalUnrealizedPLPercent,
          positions: processedPositions,
          positionCount: processedPositions.length
        },
        account: {
          status: account.status,
          currency: account.currency,
          dayTradingBuyingPower: parseFloat(account.daytrading_buying_power),
          patternDayTrader: account.pattern_day_trader,
          tradingBlocked: account.trading_blocked,
          transfersBlocked: account.transfers_blocked
        }
      };

    } catch (error) {
      console.error('Portfolio retrieval error:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get account information
   */
  async getAccount() {
    try {
      const account = await this.client.getAccount();
      
      return {
        success: true,
        account: {
          id: account.id,
          accountNumber: account.account_number,
          status: account.status,
          currency: account.currency,
          buyingPower: parseFloat(account.buying_power),
          cash: parseFloat(account.cash),
          portfolioValue: parseFloat(account.portfolio_value),
          equity: parseFloat(account.equity),
          lastEquity: parseFloat(account.last_equity),
          longMarketValue: parseFloat(account.long_market_value),
          shortMarketValue: parseFloat(account.short_market_value),
          initialMargin: parseFloat(account.initial_margin),
          maintenanceMargin: parseFloat(account.maintenance_margin),
          dayTradingBuyingPower: parseFloat(account.daytrading_buying_power),
          dayTradeCount: account.daytrade_count,
          patternDayTrader: account.pattern_day_trader,
          tradingBlocked: account.trading_blocked,
          transfersBlocked: account.transfers_blocked
        }
      };

    } catch (error) {
      console.error('Account retrieval error:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get market data for a symbol
   */
  async getMarketData(symbol, timeframe = '1Day', limit = 100) {
    try {
      const bars = await this.client.getBars(symbol, timeframe, { limit });
      
      const processedBars = bars.map(bar => ({
        timestamp: bar.t,
        open: parseFloat(bar.o),
        high: parseFloat(bar.h),
        low: parseFloat(bar.l),
        close: parseFloat(bar.c),
        volume: parseInt(bar.v),
        tradeCount: bar.n ? parseInt(bar.n) : null,
        vwap: bar.vw ? parseFloat(bar.vw) : null
      }));

      return {
        success: true,
        symbol: symbol.toUpperCase(),
        timeframe,
        data: processedBars,
        count: processedBars.length
      };

    } catch (error) {
      console.error('Market data retrieval error:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get order history
   */
  async getOrderHistory(status = null, limit = 50) {
    try {
      const orders = await this.client.getOrders({ status, limit });
      
      const processedOrders = orders.map(order => ({
        id: order.id,
        clientOrderId: order.client_order_id,
        symbol: order.symbol,
        qty: parseFloat(order.qty),
        side: order.side,
        type: order.type,
        timeInForce: order.time_in_force,
        status: order.status,
        submittedAt: order.submitted_at,
        filledAt: order.filled_at,
        filledQty: order.filled_qty ? parseFloat(order.filled_qty) : 0,
        filledAvgPrice: order.filled_avg_price ? parseFloat(order.filled_avg_price) : null,
        limitPrice: order.limit_price ? parseFloat(order.limit_price) : null,
        stopPrice: order.stop_price ? parseFloat(order.stop_price) : null
      }));

      return {
        success: true,
        orders: processedOrders,
        count: processedOrders.length
      };

    } catch (error) {
      console.error('Order history retrieval error:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Cancel an order
   */
  async cancelOrder(orderId) {
    try {
      await this.client.cancelOrder(orderId);
      
      return {
        success: true,
        orderId,
        message: 'Order cancelled successfully'
      };

    } catch (error) {
      console.error('Order cancellation error:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Validate trade parameters
   */
  async validateTrade(symbol, qty, side) {
    try {
      // Check if market is open (basic check)
      const now = new Date();
      const marketOpen = new Date(now);
      marketOpen.setHours(9, 30, 0, 0);
      const marketClose = new Date(now);
      marketClose.setHours(16, 0, 0, 0);
      
      if (now < marketOpen || now > marketClose) {
        return {
          valid: false,
          error: 'Market is currently closed'
        };
      }

      // Check quantity
      if (qty <= 0) {
        return {
          valid: false,
          error: 'Quantity must be positive'
        };
      }

      // Check symbol format
      if (!symbol || typeof symbol !== 'string') {
        return {
          valid: false,
          error: 'Valid symbol required'
        };
      }

      // Get account info to check buying power
      const account = await this.client.getAccount();
      const buyingPower = parseFloat(account.buying_power);
      
      if (side.toLowerCase() === 'buy') {
        // For buy orders, check if we have enough buying power
        // This is a simplified check - in reality, you'd need current price
        const estimatedCost = qty * 100; // Rough estimate
        if (estimatedCost > buyingPower) {
          return {
            valid: false,
            error: 'Insufficient buying power'
          };
        }
      }

      return { valid: true };

    } catch (error) {
      return {
        valid: false,
        error: `Validation error: ${error.message}`
      };
    }
  }

  /**
   * Update performance metrics
   */
  async updatePerformanceMetrics(symbol, side, qty) {
    const key = `${symbol}_${side}`;
    const current = this.performanceMetrics.get(key) || {
      totalTrades: 0,
      totalVolume: 0,
      lastTrade: null
    };

    current.totalTrades += 1;
    current.totalVolume += qty;
    current.lastTrade = new Date();

    this.performanceMetrics.set(key, current);
  }

  /**
   * Get performance metrics
   */
  getPerformanceMetrics() {
    const metrics = {};
    for (const [key, value] of this.performanceMetrics) {
      metrics[key] = value;
    }
    return metrics;
  }

  /**
   * Get trade history
   */
  getTradeHistory() {
    const history = [];
    for (const [tradeId, trade] of this.tradeHistory) {
      history.push(trade);
    }
    return history.sort((a, b) => new Date(b.submittedAt) - new Date(a.submittedAt));
  }
}

module.exports = AlpacaTradingTool;
