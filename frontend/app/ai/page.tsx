"use client";

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Header } from '@/components/layout/header';
import { Sidebar } from '@/components/layout/sidebar';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface MarketAnalysis {
  symbol: string;
  sentiment: string;
  confidence: number;
  key_factors: string[];
  recommendations: string[];
  risk_level: string;
  reasoning: string;
}

export default function AITradingPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: 'Hello! I\'m your AI trading assistant. I can help you with market analysis, risk assessment, and trading insights. Try asking me to analyze a stock symbol like "AAPL" or "TSLA".',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedTool, setSelectedTool] = useState<'chat' | 'analysis' | 'risk' | 'signal' | 'comprehensive'>('chat');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      let response;
      
      if (selectedTool === 'chat') {
        // Regular chat
        response = await fetch('/api/v1/ai/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: inputMessage })
        });
      } else if (selectedTool === 'analysis') {
        // Check if message contains a stock symbol
        const symbol = extractSymbol(inputMessage);
        if (!symbol) {
          throw new Error('Please provide a stock symbol for analysis (e.g., "analyze AAPL")');
        }
        
        response = await fetch('/api/v1/ai/analyze/market', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            symbol: symbol.toUpperCase(),
            context: inputMessage 
          })
        });
      } else if (selectedTool === 'comprehensive') {
        const symbol = extractSymbol(inputMessage);
        if (!symbol) {
          throw new Error('Please provide a stock symbol for comprehensive analysis');
        }
        
        response = await fetch('/api/v1/ai/analyze/comprehensive', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            symbol: symbol.toUpperCase(),
            account_value: 100000,
            current_positions: []
          })
        });
      } else {
        // Default to chat for other tools (not implemented in this simple version)
        response = await fetch('/api/v1/ai/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: inputMessage })
        });
      }

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Request failed');
      }

      const data = await response.json();
      
      // Format response based on tool
      let assistantResponse: string;
      
      if (selectedTool === 'analysis' && data.sentiment) {
        // Format market analysis
        assistantResponse = formatMarketAnalysis(data as MarketAnalysis);
      } else if (selectedTool === 'comprehensive' && data.market_analysis) {
        // Format comprehensive analysis
        assistantResponse = formatComprehensiveAnalysis(data);
      } else if (data.response) {
        // Regular chat response
        assistantResponse = data.response;
      } else {
        assistantResponse = JSON.stringify(data, null, 2);
      }

      const aiMessage: ChatMessage = {
        role: 'assistant',
        content: assistantResponse,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessage]);

    } catch (error) {
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Something went wrong'}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const extractSymbol = (text: string): string | null => {
    // Simple symbol extraction - look for 1-5 uppercase letters
    const match = text.match(/\b[A-Z]{1,5}\b/);
    return match ? match[0] : null;
  };

  const formatMarketAnalysis = (analysis: MarketAnalysis): string => {
    return `📊 **Market Analysis for ${analysis.symbol}**

**Sentiment:** ${analysis.sentiment.toUpperCase()} 
**Confidence:** ${(analysis.confidence * 100).toFixed(1)}%
**Risk Level:** ${analysis.risk_level.toUpperCase()}

**Key Factors:**
${analysis.key_factors.map(factor => `• ${factor}`).join('\n')}

**Recommendations:**
${analysis.recommendations.map(rec => `• ${rec}`).join('\n')}

**Analysis:**
${analysis.reasoning}`;
  };

  const formatComprehensiveAnalysis = (data: any): string => {
    const market = data.market_analysis;
    const risk = data.risk_assessment;
    const strategy = data.strategy_signal;
    
    return `🧠 **Comprehensive AI Analysis for ${data.symbol}**

**Final Recommendation:** ${data.final_recommendation}

**Market Sentiment:** ${market.sentiment} (${(market.confidence * 100).toFixed(1)}% confidence)
**Risk Level:** ${risk.risk_level}
**Strategy Action:** ${strategy.action} (${(strategy.confidence * 100).toFixed(1)}% confidence)

**Key Insights:**
${market.key_factors.map((factor: string) => `• ${factor}`).join('\n')}

**Risk Factors:**
${risk.key_risk_factors.map((factor: string) => `• ${factor}`).join('\n')}

**Trading Reasoning:**
${strategy.reasoning}`;
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="flex">
        <Sidebar />
        <main className="flex-1 p-6">
          <div className="max-w-4xl">
            <h1 className="text-3xl font-bold mb-6">🤖 AI Trading Assistant</h1>
      
      {/* Tool Selection */}
      <div className="mb-4 p-4 bg-gray-100 rounded-lg">
        <label className="block text-sm font-medium mb-2">Analysis Mode:</label>
        <div className="flex flex-wrap gap-2">
          {[
            { value: 'chat', label: '💬 Chat', desc: 'General conversation' },
            { value: 'analysis', label: '📊 Market Analysis', desc: 'Analyze stock sentiment' },
            { value: 'comprehensive', label: '🧠 Comprehensive', desc: 'Full AI analysis' },
            { value: 'risk', label: '⚠️ Risk Assessment', desc: 'Risk evaluation' },
            { value: 'signal', label: '📈 Trading Signal', desc: 'Generate signals' }
          ].map(tool => (
            <button
              key={tool.value}
              onClick={() => setSelectedTool(tool.value as any)}
              className={`px-3 py-2 rounded text-sm ${
                selectedTool === tool.value 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              } border transition-colors`}
              title={tool.desc}
            >
              {tool.label}
            </button>
          ))}
        </div>
      </div>

      {/* Chat Messages */}
      <div className="bg-white rounded-lg shadow-md h-96 overflow-y-auto p-4 mb-4 border">
        {messages.map((message, index) => (
          <div key={index} className={`mb-4 ${message.role === 'user' ? 'text-right' : 'text-left'}`}>
            <div className={`inline-block max-w-3xl p-3 rounded-lg ${
              message.role === 'user' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-100 text-gray-800'
            }`}>
              <div className="whitespace-pre-wrap">{message.content}</div>
              <div className={`text-xs mt-1 opacity-70`}>
                {message.timestamp.toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="text-left">
            <div className="inline-block bg-gray-100 text-gray-800 p-3 rounded-lg">
              <div className="flex items-center space-x-2">
                <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
                <span>AI is thinking...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="flex gap-2">
        <textarea
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={
            selectedTool === 'analysis' || selectedTool === 'comprehensive'
              ? "Ask me to analyze a stock (e.g., 'analyze AAPL' or 'what do you think about TSLA?')"
              : "Type your message here... (Press Enter to send, Shift+Enter for new line)"
          }
          className="flex-1 p-3 border rounded-lg resize-none h-20"
          disabled={isLoading}
        />
        <Button 
          onClick={handleSendMessage}
          disabled={isLoading || !inputMessage.trim()}
          className="px-6 h-20"
        >
          Send
        </Button>
      </div>

      {/* Health Status */}
      <div className="mt-4 text-sm text-gray-600">
        <p>💡 <strong>Tips:</strong></p>
        <ul className="list-disc list-inside space-y-1 mt-2">
          <li>Use <strong>Market Analysis</strong> mode to get AI sentiment analysis for stocks</li>
          <li>Use <strong>Comprehensive</strong> mode for full AI analysis with all agents</li>
          <li>Ask questions like "analyze AAPL", "what's the risk of TSLA?", or "should I buy NVDA?"</li>
          <li>The AI uses real market data and multiple specialized trading agents</li>
        </ul>
      </div>
          </div>
        </main>
      </div>
    </div>
  );
}