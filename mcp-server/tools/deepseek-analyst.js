/**
 * DeepSeek Analyst Tool for Mathematical and Risk Analysis
 * Specializes in quantitative analysis, options pricing, and risk calculations
 */

const axios = require('axios');
const config = require('../config/llm-config').deepseek;

class DeepSeekAnalyst {
  constructor() {
    this.apiKey = config.apiKey;
    this.endpoint = config.endpoint;
    this.model = config.model;
    this.rateLimiter = new Map(); // Simple rate limiting
  }

  /**
   * Analyze market conditions with mathematical precision
   * @param {string} symbol - Stock symbol
   * @param {object} marketData - Market data context
   * @param {string} analysisType - Type of analysis to perform
   */
  async analyzeMarket(symbol, marketData, analysisType = 'comprehensive') {
    try {
      await this.checkRateLimit();

      const prompt = this.buildAnalysisPrompt(symbol, marketData, analysisType);
      
      const response = await axios.post(this.endpoint, {
        model: this.model,
        messages: [
          {
            role: "system",
            content: "You are a quantitative financial analyst specializing in mathematical analysis, risk calculations, and statistical modeling. Provide precise, data-driven insights with specific numerical recommendations."
          },
          {
            role: "user",
            content: prompt
          }
        ],
        temperature: config.temperature,
        max_tokens: config.maxTokens
      }, {
        headers: { 
          Authorization: `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        },
        timeout: config.global.timeout
      });

      const analysis = response.data.choices[0].message.content;
      
      // Track usage for billing
      await this.trackUsage(prompt.length, analysis.length);

      return {
        success: true,
        symbol: symbol.toUpperCase(),
        analysisType,
        analysis: analysis,
        model: 'deepseek',
        timestamp: new Date().toISOString(),
        confidence: this.extractConfidence(analysis),
        recommendations: this.extractRecommendations(analysis)
      };

    } catch (error) {
      console.error('DeepSeek analysis error:', error);
      return {
        success: false,
        error: error.message,
        symbol: symbol.toUpperCase()
      };
    }
  }

  /**
   * Calculate options Greeks and pricing
   * @param {object} optionsData - Options contract data
   */
  async calculateOptionsGreeks(optionsData) {
    const prompt = `
    Calculate the Black-Scholes options pricing and Greeks for the following option:
    
    Symbol: ${optionsData.symbol}
    Strike Price: $${optionsData.strike}
    Current Stock Price: $${optionsData.currentPrice}
    Time to Expiration: ${optionsData.daysToExpiration} days
    Risk-Free Rate: ${optionsData.riskFreeRate || 0.05}%
    Volatility: ${optionsData.volatility || 0.25}
    Option Type: ${optionsData.type} (call/put)
    
    Please provide:
    1. Theoretical option price
    2. Delta
    3. Gamma
    4. Theta
    5. Vega
    6. Rho
    7. Implied volatility analysis
    8. Risk assessment
    9. Trading recommendations based on Greeks
    `;

    try {
      await this.checkRateLimit();

      const response = await axios.post(this.endpoint, {
        model: this.model,
        messages: [
          {
            role: "system",
            content: "You are an options pricing expert. Calculate precise Black-Scholes values and provide detailed Greek analysis with trading implications."
          },
          {
            role: "user",
            content: prompt
          }
        ],
        temperature: 0.1, // Low temperature for mathematical precision
        max_tokens: config.maxTokens
      }, {
        headers: { 
          Authorization: `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      const analysis = response.data.choices[0].message.content;
      
      await this.trackUsage(prompt.length, analysis.length);

      return {
        success: true,
        optionsData,
        analysis: analysis,
        model: 'deepseek',
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      console.error('Options Greeks calculation error:', error);
      return {
        success: false,
        error: error.message,
        optionsData
      };
    }
  }

  /**
   * Perform risk analysis and portfolio optimization
   * @param {object} portfolio - Portfolio data
   * @param {object} riskParams - Risk parameters
   */
  async analyzeRisk(portfolio, riskParams = {}) {
    const prompt = `
    Perform comprehensive risk analysis for the following portfolio:
    
    Portfolio Holdings:
    ${JSON.stringify(portfolio.positions, null, 2)}
    
    Portfolio Value: $${portfolio.totalValue}
    Cash: $${portfolio.cash}
    
    Risk Parameters:
    - Maximum Drawdown: ${riskParams.maxDrawdown || 15}%
    - Value at Risk (VaR): ${riskParams.varConfidence || 95}% confidence
    - Risk Tolerance: ${riskParams.riskTolerance || 'moderate'}
    
    Please analyze:
    1. Portfolio volatility and correlation analysis
    2. Value at Risk (VaR) calculations
    3. Expected Shortfall (CVaR)
    4. Maximum Drawdown analysis
    5. Sharpe ratio and risk-adjusted returns
    6. Concentration risk assessment
    7. Sector and geographic diversification
    8. Stress testing scenarios
    9. Risk mitigation recommendations
    10. Optimal position sizing suggestions
    `;

    try {
      await this.checkRateLimit();

      const response = await axios.post(this.endpoint, {
        model: this.model,
        messages: [
          {
            role: "system",
            content: "You are a quantitative risk analyst. Provide detailed risk metrics, statistical analysis, and actionable risk management recommendations."
          },
          {
            role: "user",
            content: prompt
          }
        ],
        temperature: 0.2,
        max_tokens: config.maxTokens
      }, {
        headers: { 
          Authorization: `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      const analysis = response.data.choices[0].message.content;
      
      await this.trackUsage(prompt.length, analysis.length);

      return {
        success: true,
        portfolio: portfolio,
        riskParams: riskParams,
        analysis: analysis,
        model: 'deepseek',
        timestamp: new Date().toISOString(),
        riskMetrics: this.extractRiskMetrics(analysis)
      };

    } catch (error) {
      console.error('Risk analysis error:', error);
      return {
        success: false,
        error: error.message,
        portfolio: portfolio
      };
    }
  }

  /**
   * Build analysis prompt based on type
   */
  buildAnalysisPrompt(symbol, marketData, analysisType) {
    const basePrompt = `
    Analyze ${symbol} with the following market data:
    
    Current Price: $${marketData.currentPrice}
    Volume: ${marketData.volume}
    Market Cap: $${marketData.marketCap}
    P/E Ratio: ${marketData.peRatio}
    Beta: ${marketData.beta}
    Recent Performance: ${marketData.recentPerformance}%
    
    Technical Indicators:
    - RSI: ${marketData.rsi}
    - MACD: ${marketData.macd}
    - Moving Averages: ${JSON.stringify(marketData.movingAverages)}
    - Support/Resistance: ${JSON.stringify(marketData.supportResistance)}
    `;

    switch (analysisType) {
      case 'mathematical':
        return basePrompt + `
        
        Please provide:
        1. Statistical analysis of price movements
        2. Probability calculations for price targets
        3. Volatility analysis and forecasting
        4. Mathematical models for price prediction
        5. Risk-reward ratio calculations
        6. Quantitative trading signals
        `;

      case 'risk':
        return basePrompt + `
        
        Please analyze:
        1. Downside risk assessment
        2. Value at Risk (VaR) calculations
        3. Maximum drawdown potential
        4. Correlation with market indices
        5. Stress testing scenarios
        6. Risk mitigation strategies
        `;

      case 'options':
        return basePrompt + `
        
        Please provide:
        1. Options pricing analysis
        2. Implied volatility assessment
        3. Greeks calculations
        4. Options strategies recommendations
        5. Risk-reward analysis for options
        6. Time decay considerations
        `;

      default: // comprehensive
        return basePrompt + `
        
        Please provide comprehensive analysis including:
        1. Mathematical price modeling
        2. Statistical significance of patterns
        3. Risk assessment and metrics
        4. Probability-based price targets
        5. Quantitative trading recommendations
        6. Confidence intervals and uncertainty measures
        `;
    }
  }

  /**
   * Extract confidence level from analysis
   */
  extractConfidence(analysis) {
    // Look for confidence indicators in the text
    const confidenceMatch = analysis.match(/(\d+)% confidence|confidence: (\d+)%|(\d+)% certain/i);
    if (confidenceMatch) {
      return parseInt(confidenceMatch[1] || confidenceMatch[2] || confidenceMatch[3]);
    }
    
    // Default confidence based on analysis length and detail
    const wordCount = analysis.split(' ').length;
    if (wordCount > 500) return 85;
    if (wordCount > 300) return 75;
    return 65;
  }

  /**
   * Extract trading recommendations
   */
  extractRecommendations(analysis) {
    const recommendations = [];
    
    // Look for buy/sell/hold recommendations
    if (analysis.toLowerCase().includes('buy') || analysis.toLowerCase().includes('bullish')) {
      recommendations.push('BUY');
    }
    if (analysis.toLowerCase().includes('sell') || analysis.toLowerCase().includes('bearish')) {
      recommendations.push('SELL');
    }
    if (analysis.toLowerCase().includes('hold') || analysis.toLowerCase().includes('neutral')) {
      recommendations.push('HOLD');
    }
    
    return recommendations.length > 0 ? recommendations : ['HOLD'];
  }

  /**
   * Extract risk metrics from analysis
   */
  extractRiskMetrics(analysis) {
    const metrics = {};
    
    // Extract VaR
    const varMatch = analysis.match(/VaR[:\s]*(\d+(?:\.\d+)?)%/i);
    if (varMatch) {
      metrics.var = parseFloat(varMatch[1]);
    }
    
    // Extract volatility
    const volMatch = analysis.match(/volatility[:\s]*(\d+(?:\.\d+)?)%/i);
    if (volMatch) {
      metrics.volatility = parseFloat(volMatch[1]);
    }
    
    // Extract Sharpe ratio
    const sharpeMatch = analysis.match(/sharpe[:\s]*(\d+(?:\.\d+)?)/i);
    if (sharpeMatch) {
      metrics.sharpeRatio = parseFloat(sharpeMatch[1]);
    }
    
    return metrics;
  }

  /**
   * Check rate limiting
   */
  async checkRateLimit() {
    const now = Date.now();
    const minute = Math.floor(now / 60000);
    
    if (!this.rateLimiter.has(minute)) {
      this.rateLimiter.set(minute, 0);
    }
    
    const currentCount = this.rateLimiter.get(minute);
    if (currentCount >= config.rateLimit) {
      throw new Error('Rate limit exceeded for DeepSeek API');
    }
    
    this.rateLimiter.set(minute, currentCount + 1);
    
    // Clean up old entries
    for (const [key] of this.rateLimiter) {
      if (key < minute - 1) {
        this.rateLimiter.delete(key);
      }
    }
  }

  /**
   * Track usage for billing
   */
  async trackUsage(inputTokens, outputTokens) {
    const totalTokens = inputTokens + outputTokens;
    const cost = (totalTokens / 1000000) * config.pricePerMillion;
    
    console.log(`DeepSeek usage: ${totalTokens} tokens, estimated cost: $${cost.toFixed(6)}`);
    
    // Here you would typically send this to your billing system
    // await this.billingService.recordUsage('deepseek', totalTokens, cost);
  }
}

module.exports = DeepSeekAnalyst;
