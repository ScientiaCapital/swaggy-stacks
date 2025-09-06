/**
 * Chinese LLM Configuration for Swaggy Stacks MCP Server
 * Supports multiple Chinese LLM providers for diverse analysis
 */

module.exports = {
  // DeepSeek - Excellent for mathematical analysis and risk calculations
  deepseek: {
    apiKey: process.env.DEEPSEEK_API_KEY,
    endpoint: 'https://api.deepseek.com/v1/chat/completions',
    model: 'deepseek-chat',
    pricePerMillion: 0.27, // USD per million tokens
    maxTokens: 4000,
    temperature: 0.3,
    specializations: [
      'mathematical-analysis',
      'risk-calculations',
      'options-greeks',
      'probability-models',
      'statistical-analysis'
    ],
    rateLimit: 100 // requests per minute
  },
  
  // MiniMax - Great for document processing and pattern recognition
  minimax: {
    apiKey: process.env.MINIMAX_API_KEY,
    endpoint: 'https://api.minimax.chat/v1/chat/completions',
    model: 'minimax-01',
    pricePerMillion: 0.40,
    maxTokens: 4000,
    temperature: 0.2,
    specializations: [
      'document-analysis',
      'pattern-recognition',
      'news-sentiment',
      'earnings-analysis',
      'regulatory-documents'
    ],
    rateLimit: 80
  },
  
  // Qwen (Alibaba) - Strong for general financial analysis
  qwen: {
    apiKey: process.env.QWEN_API_KEY,
    endpoint: 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation',
    model: 'qwen-turbo',
    pricePerMillion: 0.35,
    maxTokens: 6000,
    temperature: 0.4,
    specializations: [
      'market-analysis',
      'sector-analysis',
      'company-fundamentals',
      'economic-indicators',
      'trend-analysis'
    ],
    rateLimit: 120
  },
  
  // Yi (01.AI) - Good for technical analysis and chart patterns
  yi: {
    apiKey: process.env.YI_API_KEY,
    endpoint: 'https://api.lingyiwanwu.com/v1/chat/completions',
    model: 'yi-34b-chat',
    pricePerMillion: 0.30,
    maxTokens: 4000,
    temperature: 0.3,
    specializations: [
      'technical-analysis',
      'chart-patterns',
      'support-resistance',
      'momentum-indicators',
      'volume-analysis'
    ],
    rateLimit: 90
  },
  
  // ChatGLM (Zhipu AI) - Excellent for Chinese market analysis
  chatglm: {
    apiKey: process.env.CHATGLM_API_KEY,
    endpoint: 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
    model: 'glm-4',
    pricePerMillion: 0.45,
    maxTokens: 8000,
    temperature: 0.3,
    specializations: [
      'chinese-markets',
      'asian-economics',
      'cross-market-analysis',
      'currency-analysis',
      'commodity-analysis'
    ],
    rateLimit: 60
  },
  
  // Moonshot - Great for long-context analysis
  moonshot: {
    apiKey: process.env.MOONSHOT_API_KEY,
    endpoint: 'https://api.moonshot.cn/v1/chat/completions',
    model: 'moonshot-v1-8k',
    pricePerMillion: 0.50,
    maxTokens: 8000,
    temperature: 0.2,
    specializations: [
      'long-context-analysis',
      'historical-patterns',
      'market-cycles',
      'anomaly-detection',
      'correlation-analysis'
    ],
    rateLimit: 50
  },
  
  // InternLM2 - Good for research and comprehensive analysis
  internlm2: {
    apiKey: process.env.INTERNLM2_API_KEY,
    endpoint: 'https://api.internlm.ai/v1/chat/completions',
    model: 'internlm2-chat-7b',
    pricePerMillion: 0.25,
    maxTokens: 4000,
    temperature: 0.4,
    specializations: [
      'research-analysis',
      'comprehensive-reports',
      'due-diligence',
      'market-research',
      'competitive-analysis'
    ],
    rateLimit: 100
  },
  
  // Global configuration
  global: {
    timeout: 30000, // 30 seconds
    retryAttempts: 3,
    retryDelay: 1000, // 1 second
    fallbackModel: 'qwen', // Use Qwen as fallback
    costTracking: true,
    usageLogging: true
  },
  
  // Model selection strategy
  selectionStrategy: {
    // Route requests based on analysis type
    routing: {
      'mathematical': 'deepseek',
      'pattern': 'moonshot',
      'document': 'minimax',
      'technical': 'yi',
      'fundamental': 'qwen',
      'chinese-market': 'chatglm',
      'research': 'internlm2',
      'general': 'qwen'
    },
    
    // Load balancing for high-volume requests
    loadBalancing: {
      enabled: true,
      strategy: 'round-robin',
      healthCheck: true
    }
  }
};
