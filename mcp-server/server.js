/**
 * Swaggy Stacks MCP Server
 * Chinese LLM + Alpaca Paper Trading Integration
 * Provides persistent trading intelligence with full context
 */

const { Server } = require('@modelcontextprotocol/server');
const AlpacaTradingTool = require('./tools/alpaca-trading');
const DeepSeekAnalyst = require('./tools/deepseek-analyst');
require('dotenv').config();

class SwaggyStacksMCPServer {
  constructor() {
    this.server = new Server(
      { 
        name: 'swaggy-stacks-trading-mcp', 
        version: '1.0.0',
        description: 'Advanced trading intelligence with Chinese LLMs and Alpaca integration'
      },
      { 
        capabilities: { 
          tools: {},
          resources: {},
          prompts: {}
        } 
      }
    );
    
    // Initialize tools
    this.alpaca = new AlpacaTradingTool();
    this.deepseek = new DeepSeekAnalyst();
    
    // Context storage for persistent trading theses
    this.contexts = new Map();
    this.sessions = new Map();
    
    this.setupHandlers();
  }
  
  setupHandlers() {
    // List available tools
    this.server.setRequestHandler('tools/list', async () => ({
      tools: [
        {
          name: 'execute_trade',
          description: 'Execute a paper trade through Alpaca with risk validation',
          inputSchema: {
            type: 'object',
            properties: {
              symbol: { 
                type: 'string', 
                description: 'Stock symbol (e.g., AAPL, TSLA)' 
              },
              quantity: { 
                type: 'number', 
                description: 'Number of shares to trade' 
              },
              side: { 
                type: 'string', 
                enum: ['buy', 'sell'], 
                description: 'Trade direction' 
              },
              order_type: {
                type: 'string',
                enum: ['market', 'limit', 'stop'],
                description: 'Order type',
                default: 'market'
              },
              limit_price: {
                type: 'number',
                description: 'Limit price (for limit orders)'
              },
              stop_price: {
                type: 'number',
                description: 'Stop price (for stop orders)'
              }
            },
            required: ['symbol', 'quantity', 'side']
          }
        },
        {
          name: 'analyze_market',
          description: 'Analyze market conditions using DeepSeek mathematical analysis',
          inputSchema: {
            type: 'object',
            properties: {
              symbol: { 
                type: 'string', 
                description: 'Stock symbol to analyze' 
              },
              analysis_type: {
                type: 'string',
                enum: ['comprehensive', 'mathematical', 'risk', 'options'],
                description: 'Type of analysis to perform',
                default: 'comprehensive'
              },
              market_data: {
                type: 'object',
                description: 'Market data context (optional)'
              }
            },
            required: ['symbol']
          }
        },
        {
          name: 'create_trading_thesis',
          description: 'Create a persistent trading thesis with full context tracking',
          inputSchema: {
            type: 'object',
            properties: {
              symbol: { 
                type: 'string', 
                description: 'Stock symbol' 
              },
              thesis: { 
                type: 'string', 
                description: 'Trading thesis and reasoning' 
              },
              time_horizon: { 
                type: 'string', 
                description: 'Expected holding period' 
              },
              risk_parameters: { 
                type: 'object', 
                description: 'Risk management parameters' 
              },
              entry_price: {
                type: 'number',
                description: 'Planned entry price'
              },
              stop_loss: {
                type: 'number',
                description: 'Stop loss price'
              },
              take_profit: {
                type: 'number',
                description: 'Take profit price'
              }
            },
            required: ['symbol', 'thesis', 'time_horizon']
          }
        },
        {
          name: 'update_thesis_context',
          description: 'Update existing trading thesis with new market information',
          inputSchema: {
            type: 'object',
            properties: {
              thesis_id: { 
                type: 'string', 
                description: 'ID of the trading thesis to update' 
              },
              market_update: { 
                type: 'string', 
                description: 'New market information or analysis' 
              },
              price_update: {
                type: 'number',
                description: 'Current market price'
              },
              volume_update: {
                type: 'number',
                description: 'Current trading volume'
              }
            },
            required: ['thesis_id', 'market_update']
          }
        },
        {
          name: 'get_portfolio',
          description: 'Get current portfolio information and performance',
          inputSchema: {
            type: 'object',
            properties: {
              include_positions: {
                type: 'boolean',
                description: 'Include detailed position information',
                default: true
              },
              include_performance: {
                type: 'boolean',
                description: 'Include performance metrics',
                default: true
              }
            }
          }
        },
        {
          name: 'analyze_portfolio_risk',
          description: 'Perform comprehensive risk analysis on the entire portfolio',
          inputSchema: {
            type: 'object',
            properties: {
              risk_parameters: {
                type: 'object',
                description: 'Risk analysis parameters',
                properties: {
                  max_drawdown: { type: 'number', default: 15 },
                  var_confidence: { type: 'number', default: 95 },
                  risk_tolerance: { type: 'string', default: 'moderate' }
                }
              }
            }
          }
        },
        {
          name: 'calculate_options_greeks',
          description: 'Calculate options pricing and Greeks using DeepSeek',
          inputSchema: {
            type: 'object',
            properties: {
              symbol: { type: 'string', description: 'Underlying stock symbol' },
              strike: { type: 'number', description: 'Strike price' },
              current_price: { type: 'number', description: 'Current stock price' },
              days_to_expiration: { type: 'number', description: 'Days until expiration' },
              option_type: { type: 'string', enum: ['call', 'put'], description: 'Option type' },
              volatility: { type: 'number', description: 'Implied volatility' },
              risk_free_rate: { type: 'number', description: 'Risk-free rate', default: 0.05 }
            },
            required: ['symbol', 'strike', 'current_price', 'days_to_expiration', 'option_type']
          }
        }
      ]
    }));

    // Handle tool calls
    this.server.setRequestHandler('tools/call', async (request) => {
      const { name, arguments: args } = request.params;
      
      try {
        let result;
        
        switch (name) {
          case 'execute_trade':
            result = await this.alpaca.executeTrade(args);
            break;
            
          case 'analyze_market':
            // Get market data if not provided
            const marketData = args.market_data || await this.getMarketData(args.symbol);
            result = await this.deepseek.analyzeMarket(args.symbol, marketData, args.analysis_type);
            break;
            
          case 'create_trading_thesis':
            result = await this.createTradingThesis(args);
            break;
            
          case 'update_thesis_context':
            result = await this.updateThesisContext(args);
            break;
            
          case 'get_portfolio':
            result = await this.alpaca.getPortfolio();
            break;
            
          case 'analyze_portfolio_risk':
            const portfolio = await this.alpaca.getPortfolio();
            result = await this.deepseek.analyzeRisk(portfolio.portfolio, args.risk_parameters);
            break;
            
          case 'calculate_options_greeks':
            result = await this.deepseek.calculateOptionsGreeks(args);
            break;
            
          default:
            throw new Error(`Unknown tool: ${name}`);
        }
        
        return { 
          content: [{ 
            type: 'text', 
            text: JSON.stringify(result, null, 2) 
          }] 
        };
        
      } catch (error) {
        console.error(`Tool execution error for ${name}:`, error);
        return { 
          content: [{ 
            type: 'text', 
            text: JSON.stringify({ 
              success: false, 
              error: error.message 
            }, null, 2) 
          }] 
        };
      }
    });

    // List available resources
    this.server.setRequestHandler('resources/list', async () => ({
      resources: [
        {
          uri: 'swaggy-stacks://trading-theses',
          name: 'Trading Theses',
          description: 'Persistent trading theses with full context',
          mimeType: 'application/json'
        },
        {
          uri: 'swaggy-stacks://portfolio-performance',
          name: 'Portfolio Performance',
          description: 'Real-time portfolio performance metrics',
          mimeType: 'application/json'
        },
        {
          uri: 'swaggy-stacks://market-data',
          name: 'Market Data',
          description: 'Current market data and analysis',
          mimeType: 'application/json'
        }
      ]
    }));

    // Handle resource requests
    this.server.setRequestHandler('resources/read', async (request) => {
      const { uri } = request.params;
      
      try {
        let content;
        
        switch (uri) {
          case 'swaggy-stacks://trading-theses':
            content = Array.from(this.contexts.values());
            break;
            
          case 'swaggy-stacks://portfolio-performance':
            const portfolio = await this.alpaca.getPortfolio();
            content = portfolio;
            break;
            
          case 'swaggy-stacks://market-data':
            content = await this.getMarketData('AAPL'); // Default to AAPL
            break;
            
          default:
            throw new Error(`Unknown resource: ${uri}`);
        }
        
        return {
          contents: [{
            uri,
            mimeType: 'application/json',
            text: JSON.stringify(content, null, 2)
          }]
        };
        
      } catch (error) {
        console.error(`Resource read error for ${uri}:`, error);
        throw error;
      }
    });
  }

  /**
   * Create a persistent trading thesis
   */
  async createTradingThesis(args) {
    const thesisId = this.generateThesisId();
    
    const tradingContext = {
      id: thesisId,
      symbol: args.symbol.toUpperCase(),
      originalThesis: args.thesis,
      timeHorizon: args.time_horizon,
      riskParameters: args.risk_parameters || {},
      entryPrice: args.entry_price,
      stopLoss: args.stop_loss,
      takeProfit: args.take_profit,
      createdAt: new Date().toISOString(),
      status: 'active',
      updates: [],
      analysis: []
    };

    // Store the context
    this.contexts.set(thesisId, tradingContext);

    // Perform initial analysis
    const marketData = await this.getMarketData(args.symbol);
    const analysis = await this.deepseek.analyzeMarket(args.symbol, marketData, 'comprehensive');
    
    if (analysis.success) {
      tradingContext.analysis.push({
        timestamp: new Date().toISOString(),
        type: 'initial_analysis',
        analysis: analysis.analysis,
        confidence: analysis.confidence,
        recommendations: analysis.recommendations
      });
    }

    return {
      success: true,
      thesisId,
      tradingContext,
      initialAnalysis: analysis,
      message: 'Trading thesis created successfully with initial analysis'
    };
  }

  /**
   * Update trading thesis with new context
   */
  async updateThesisContext(args) {
    const tradingContext = this.contexts.get(args.thesis_id);
    
    if (!tradingContext) {
      return {
        success: false,
        error: 'Trading thesis not found'
      };
    }

    // Add market update
    const update = {
      timestamp: new Date().toISOString(),
      marketUpdate: args.market_update,
      priceUpdate: args.price_update,
      volumeUpdate: args.volume_update
    };
    
    tradingContext.updates.push(update);

    // Re-analyze with full context
    const marketData = await this.getMarketData(tradingContext.symbol);
    const analysis = await this.deepseek.analyzeMarket(
      tradingContext.symbol, 
      marketData, 
      'comprehensive'
    );

    if (analysis.success) {
      tradingContext.analysis.push({
        timestamp: new Date().toISOString(),
        type: 'context_update',
        update: update,
        analysis: analysis.analysis,
        confidence: analysis.confidence,
        recommendations: analysis.recommendations
      });
    }

    // Update thesis status based on analysis
    const latestAnalysis = tradingContext.analysis[tradingContext.analysis.length - 1];
    if (latestAnalysis && latestAnalysis.recommendations.includes('SELL')) {
      tradingContext.status = 'review_required';
    }

    return {
      success: true,
      thesisId: args.thesis_id,
      updatedContext: tradingContext,
      newAnalysis: analysis,
      statusChange: tradingContext.status,
      message: 'Trading thesis updated with new market context'
    };
  }

  /**
   * Get market data for a symbol
   */
  async getMarketData(symbol) {
    try {
      const result = await this.alpaca.getMarketData(symbol, '1Day', 30);
      
      if (result.success && result.data.length > 0) {
        const latest = result.data[0];
        const previous = result.data[1] || latest;
        
        return {
          currentPrice: latest.close,
          previousPrice: previous.close,
          volume: latest.volume,
          high: latest.high,
          low: latest.low,
          open: latest.open,
          change: latest.close - previous.close,
          changePercent: ((latest.close - previous.close) / previous.close) * 100,
          vwap: latest.vwap,
          timestamp: latest.timestamp
        };
      }
      
      // Fallback data
      return {
        currentPrice: 100,
        volume: 1000000,
        change: 0,
        changePercent: 0
      };
      
    } catch (error) {
      console.error('Market data error:', error);
      return {
        currentPrice: 100,
        volume: 1000000,
        change: 0,
        changePercent: 0
      };
    }
  }

  /**
   * Generate unique thesis ID
   */
  generateThesisId() {
    return `thesis_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Start the MCP server
   */
  start() {
    this.server.start();
    console.log('🚀 Swaggy Stacks MCP Server started');
    console.log('📊 Available tools: execute_trade, analyze_market, create_trading_thesis, update_thesis_context, get_portfolio, analyze_portfolio_risk, calculate_options_greeks');
    console.log('🔗 Resources: trading-theses, portfolio-performance, market-data');
    console.log('💡 Connect with: npx @modelcontextprotocol/cli connect stdio node server.js');
  }
}

// Start the server
const mcpServer = new SwaggyStacksMCPServer();
mcpServer.start();
