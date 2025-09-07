/**
 * Swaggy Stacks API Endpoints
 * Revenue-generating API with tiered access and billing
 */

import express from 'express';
import { Request, Response } from 'express';

export class SwaggyStacksAPI {
  private app: express.Application;
  private pricing = {
    'basic_analysis': 0.001,      // $0.001 per call
    'pattern_recognition': 0.005,  // $0.005 per call
    'multi_model_analysis': 0.01,  // $0.01 per call
    'portfolio_analysis': 0.10,    // $0.10 per call
    'kimi_k2_orchestration': 1.00  // $1.00 per call
  };

  constructor() {
    this.app = express();
    this.setupMiddleware();
    this.setupRoutes();
  }

  private setupMiddleware() {
    // API key validation
    this.app.use(async (req: Request, res: Response, next) => {
      const apiKey = req.headers['x-api-key'] as string;
      
      if (!apiKey) {
        return res.status(401).json({ error: 'API key required' });
      }
      
      // Validate against your Neon database
      const client = await this.validateAPIKey(apiKey);
      
      if (!client) {
        return res.status(401).json({ error: 'Invalid API key' });
      }
      
      // Attach client to request for billing
      (req as any).client = client;
      next();
    });
    
    // Rate limiting based on client tier
    this.app.use(this.createRateLimiter());
    
    // Usage tracking
    this.app.use(this.trackUsage());
  }

  private setupRoutes() {
    // Version 1 API routes
    const v1Router = express.Router();
    
    // Basic endpoints (Tier 1)
    v1Router.get('/analyze/:symbol', this.handleBasicAnalysis);
    v1Router.post('/patterns/detect', this.handlePatternDetection);
    
    // Advanced endpoints (Tier 2)
    v1Router.post('/portfolio/analyze', this.requireTier(2), this.handlePortfolioAnalysis);
    v1Router.post('/intelligence/multi-model', this.requireTier(2), this.handleMultiModel);
    
    // Enterprise endpoints (Tier 3)
    v1Router.post('/orchestrate', this.requireTier(3), this.handleKimiOrchestration);
    
    this.app.use('/api/v1', v1Router);
  }

  /**
   * Basic Stock Analysis (Tier 1)
   */
  private handleBasicAnalysis = async (req: Request, res: Response) => {
    try {
      const { symbol } = req.params;
      const client = (req as any).client;
      
      // Check if client has sufficient credits
      const cost = this.pricing['basic_analysis'];
      
      if (client.balance < cost) {
        return res.status(402).json({
          error: 'Insufficient credits',
          required: cost,
          available: client.balance,
          topUpUrl: 'https://api.swaggystack.com/billing/topup'
        });
      }
      
      // Use Qwen for basic analysis - fast and cost-effective
      const analysis = await this.qwenModel.analyze({
        symbol: symbol,
        timeframe: req.query.timeframe || '1d',
        includeTA: req.query.technicalAnalysis === 'true'
      });
      
      // Track usage for billing
      await this.recordUsage(client.id, 'basic_analysis', 1);
      
      res.json({
        symbol: symbol,
        analysis: analysis,
        confidence: analysis.confidence,
        timestamp: new Date().toISOString(),
        usage: {
          creditsUsed: cost,
          remainingCredits: client.balance - cost,
          endpoint: 'basic_analysis'
        }
      });
      
    } catch (error) {
      console.error('Basic analysis error:', error);
      res.status(500).json({ error: 'Analysis failed' });
    }
  };

  /**
   * Pattern Recognition (Tier 1)
   */
  private handlePatternDetection = async (req: Request, res: Response) => {
    try {
      const { symbol, patterns } = req.body;
      const client = (req as any).client;
      
      const cost = this.pricing['pattern_recognition'];
      
      if (client.balance < cost) {
        return res.status(402).json({
          error: 'Insufficient credits',
          required: cost,
          available: client.balance
        });
      }
      
      // Use Moonshot for pattern recognition
      const patternAnalysis = await this.moonshotModel.detectPatterns({
        symbol: symbol,
        patterns: patterns || ['head_and_shoulders', 'double_top', 'triangle', 'flag'],
        timeframe: req.body.timeframe || '1d'
      });
      
      await this.recordUsage(client.id, 'pattern_recognition', 1);
      
      res.json({
        symbol: symbol,
        patterns: patternAnalysis,
        confidence: patternAnalysis.confidence,
        timestamp: new Date().toISOString(),
        usage: {
          creditsUsed: cost,
          remainingCredits: client.balance - cost,
          endpoint: 'pattern_recognition'
        }
      });
      
    } catch (error) {
      console.error('Pattern detection error:', error);
      res.status(500).json({ error: 'Pattern detection failed' });
    }
  };

  /**
   * Portfolio Analysis (Tier 2)
   */
  private handlePortfolioAnalysis = async (req: Request, res: Response) => {
    try {
      const { portfolio, analysisType } = req.body;
      const client = (req as any).client;
      
      const cost = this.pricing['portfolio_analysis'];
      
      if (client.balance < cost) {
        return res.status(402).json({
          error: 'Insufficient credits',
          required: cost,
          available: client.balance
        });
      }
      
      // Use multiple models for portfolio analysis
      const analysis = await this.orchestrateModels({
        portfolio: portfolio,
        analysisType: analysisType || 'comprehensive',
        models: ['qwen', 'moonshot', 'deepseek'], // Not all models
        depth: 'medium',
        maxProcessingTime: 2000 // 2 second limit
      });
      
      await this.recordUsage(client.id, 'portfolio_analysis', 1);
      
      res.json({
        portfolio: portfolio,
        analysis: analysis,
        timestamp: new Date().toISOString(),
        usage: {
          creditsUsed: cost,
          remainingCredits: client.balance - cost,
          endpoint: 'portfolio_analysis'
        }
      });
      
    } catch (error) {
      console.error('Portfolio analysis error:', error);
      res.status(500).json({ error: 'Portfolio analysis failed' });
    }
  };

  /**
   * Multi-Model Analysis (Tier 2)
   */
  private handleMultiModel = async (req: Request, res: Response) => {
    try {
      const { symbol, models, analysisType } = req.body;
      const client = (req as any).client;
      
      const cost = this.pricing['multi_model_analysis'];
      
      if (client.balance < cost) {
        return res.status(402).json({
          error: 'Insufficient credits',
          required: cost,
          available: client.balance
        });
      }
      
      // Use specified models for analysis
      const analysis = await this.orchestrateModels({
        symbol: symbol,
        models: models || ['qwen', 'moonshot', 'deepseek'],
        analysisType: analysisType || 'comprehensive',
        depth: 'medium'
      });
      
      await this.recordUsage(client.id, 'multi_model_analysis', 1);
      
      res.json({
        symbol: symbol,
        models: models,
        analysis: analysis,
        timestamp: new Date().toISOString(),
        usage: {
          creditsUsed: cost,
          remainingCredits: client.balance - cost,
          endpoint: 'multi_model_analysis'
        }
      });
      
    } catch (error) {
      console.error('Multi-model analysis error:', error);
      res.status(500).json({ error: 'Multi-model analysis failed' });
    }
  };

  /**
   * Kimi K2 Orchestration (Tier 3 - Enterprise)
   */
  private handleKimiOrchestration = async (req: Request, res: Response) => {
    try {
      const { query, context, models, analysisDepth } = req.body;
      const client = (req as any).client;
      
      const cost = this.pricing['kimi_k2_orchestration'];
      
      if (client.balance < cost) {
        return res.status(402).json({
          error: 'Insufficient credits',
          required: cost,
          available: client.balance
        });
      }
      
      // Use full Kimi K2 orchestration with all models
      const intelligence = await this.kimiK2Orchestrator.analyzeWithFullContext({
        query: query,
        context: context,
        models: models || 'all', // All seven models available
        analysisDepth: analysisDepth || 'maximum',
        includeConfidenceIntervals: true,
        includeAlternativeScenarios: true
      });
      
      // Enterprise clients pay per compute second
      const computeTime = intelligence.processingTime;
      await this.recordUsage(client.id, 'kimi_k2_orchestration', computeTime);
      
      res.json({
        query: query,
        intelligence: intelligence,
        computeSecondsUsed: computeTime,
        estimatedCost: computeTime * client.ratePerSecond,
        timestamp: new Date().toISOString(),
        usage: {
          creditsUsed: cost,
          remainingCredits: client.balance - cost,
          endpoint: 'kimi_k2_orchestration'
        }
      });
      
    } catch (error) {
      console.error('Kimi orchestration error:', error);
      res.status(500).json({ error: 'Kimi orchestration failed' });
    }
  };

  /**
   * Validate API key
   */
  private async validateAPIKey(apiKey: string) {
    // This would query your Neon database
    // For now, return a mock client
    return {
      id: 'client_123',
      apiKey: apiKey,
      balance: 100.0,
      tier: 'professional',
      ratePerSecond: 0.10
    };
  }

  /**
   * Create rate limiter based on client tier
   */
  private createRateLimiter() {
    return (req: Request, res: Response, next: any) => {
      const client = (req as any).client;
      
      // Different rate limits for different tiers
      const rateLimits = {
        'developer': 100, // per minute
        'professional': 1000, // per minute
        'enterprise': 10000 // per minute
      };
      
      const limit = rateLimits[client.tier] || 100;
      
      // Implement rate limiting logic here
      // This is a simplified version
      next();
    };
  }

  /**
   * Track API usage
   */
  private trackUsage() {
    return (req: Request, res: Response, next: any) => {
      const client = (req as any).client;
      
      // Track usage for billing
      console.log(`API call from client ${client.id} to ${req.path}`);
      
      next();
    };
  }

  /**
   * Require specific tier
   */
  private requireTier(requiredTier: number) {
    return (req: Request, res: Response, next: any) => {
      const client = (req as any).client;
      
      const tierLevels = {
        'developer': 1,
        'professional': 2,
        'enterprise': 3
      };
      
      const clientTier = tierLevels[client.tier] || 1;
      
      if (clientTier < requiredTier) {
        return res.status(403).json({
          error: 'Upgrade required',
          currentTier: client.tier,
          requiredTier: requiredTier,
          upgradeUrl: 'https://swaggystack.com/api/upgrade'
        });
      }
      
      next();
    };
  }

  /**
   * Record usage for billing
   */
  private async recordUsage(clientId: string, endpoint: string, quantity: number) {
    // This would record usage in your database
    console.log(`Recording usage: ${clientId} - ${endpoint} - ${quantity}`);
  }

  /**
   * Orchestrate multiple models
   */
  private async orchestrateModels(params: any) {
    // This would coordinate multiple AI models
    // For now, return mock analysis
    return {
      models: params.models,
      analysis: 'Mock analysis from multiple models',
      confidence: 0.85,
      processingTime: 1.5
    };
  }

  /**
   * Get the Express app
   */
  getApp() {
    return this.app;
  }
}
