/**
 * Alpaca API Configuration for Swaggy Stacks MCP Server
 * Integrates with existing Alpaca paper trading setup
 */

module.exports = {
  // Use existing Alpaca credentials from environment
  baseURL: process.env.ALPACA_BASE_URL || 'https://paper-api.alpaca.markets',
  apiKey: process.env.ALPACA_API_KEY,
  secretKey: process.env.ALPACA_SECRET_KEY,
  dataURL: process.env.ALPACA_DATA_URL || 'https://data.alpaca.markets',
  
  // Paper trading configuration
  paperTrading: true,
  
  // Rate limiting for API calls
  rateLimits: {
    orders: 200, // per minute
    marketData: 1000, // per minute
    account: 100 // per minute
  },
  
  // Trading parameters
  trading: {
    defaultOrderType: 'market',
    defaultTimeInForce: 'day',
    maxPositionSize: 10000, // dollars
    maxDailyLoss: 500, // dollars
    supportedAssets: ['stocks', 'etfs'],
    marketHours: {
      start: '09:30',
      end: '16:00',
      timezone: 'America/New_York'
    }
  },
  
  // Integration with Swaggy Stacks backend
  swaggyStacks: {
    backendURL: process.env.SWAGGY_STACKS_BACKEND_URL || 'http://localhost:8000',
    apiKey: process.env.SWAGGY_STACKS_API_KEY,
    webhookURL: process.env.SWAGGY_STACKS_WEBHOOK_URL
  }
};
