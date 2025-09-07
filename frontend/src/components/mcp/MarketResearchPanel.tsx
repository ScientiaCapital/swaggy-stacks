'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { cn, formatDate, formatPercentage } from '@/lib/utils'
import {
  TrendingUp,
  TrendingDown,
  Search,
  RefreshCw,
  ExternalLink,
  Calendar,
  BarChart3,
  Newspaper,
  MessageSquare,
  AlertCircle,
  CheckCircle,
  XCircle
} from 'lucide-react'

interface MarketSentiment {
  overall_sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
  confidence_score: number
  key_factors: string[]
  news_sentiment: number
  social_sentiment: number
}

interface NewsArticle {
  title: string
  url: string
  published_date: string
  source: string
  snippet: string
  sentiment?: number
}

interface MarketResearchResult {
  symbol: string
  timestamp: string
  market_sentiment: MarketSentiment
  news_articles: NewsArticle[]
  analysis_summary: string
  confidence_score: number
  trading_recommendation: {
    action: 'buy' | 'sell' | 'hold'
    confidence: number
    reasoning: string
  }
}

const sentimentColors = {
  BULLISH: 'text-green-600 bg-green-50 dark:text-green-400 dark:bg-green-950',
  BEARISH: 'text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-950',
  NEUTRAL: 'text-yellow-600 bg-yellow-50 dark:text-yellow-400 dark:bg-yellow-950',
}

const sentimentIcons = {
  BULLISH: TrendingUp,
  BEARISH: TrendingDown,
  NEUTRAL: BarChart3,
}

const actionColors = {
  buy: 'text-green-600 bg-green-50 dark:text-green-400 dark:bg-green-950',
  sell: 'text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-950',  
  hold: 'text-yellow-600 bg-yellow-50 dark:text-yellow-400 dark:bg-yellow-950',
}

export function MarketResearchPanel() {
  const [researchData, setResearchData] = useState<MarketResearchResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL')
  const [symbols] = useState(['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META'])

  const fetchMarketResearch = async (symbol: string) => {
    try {
      setLoading(true)
      setError(null)

      const response = await fetch(`/api/v1/market-research/analyze/${symbol}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lookback_days: 7,
          include_sentiment: true,
          include_news: true
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()
      setResearchData(data)
    } catch (err) {
      console.error('Failed to fetch market research:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch market research')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchMarketResearch(selectedSymbol)
  }, [selectedSymbol])

  const getSentimentIcon = (sentiment: string) => {
    const IconComponent = sentimentIcons[sentiment as keyof typeof sentimentIcons] || BarChart3
    return <IconComponent className="h-5 w-5" />
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600 dark:text-green-400'
    if (confidence >= 0.6) return 'text-yellow-600 dark:text-yellow-400'
    return 'text-red-600 dark:text-red-400'
  }

  const getSentimentScore = (score: number) => {
    if (score > 0.1) return 'Positive'
    if (score < -0.1) return 'Negative'
    return 'Neutral'
  }

  if (loading && !researchData) {
    return (
      <div className="space-y-6 p-6 bg-card rounded-lg border">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-card-foreground">Market Research</h2>
          <RefreshCw className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
        <div className="space-y-4">
          <div className="animate-pulse h-32 bg-muted rounded-lg"></div>
          <div className="animate-pulse h-48 bg-muted rounded-lg"></div>
          <div className="animate-pulse h-24 bg-muted rounded-lg"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 p-6 bg-card rounded-lg border">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Search className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-card-foreground">Market Research</h2>
        </div>
        <div className="flex items-center space-x-3">
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="px-3 py-2 border border-border rounded-md bg-background text-foreground"
          >
            {symbols.map((symbol) => (
              <option key={symbol} value={symbol}>
                {symbol}
              </option>
            ))}
          </select>
          <Button
            onClick={() => fetchMarketResearch(selectedSymbol)}
            variant="outline"
            size="sm"
            disabled={loading}
          >
            <RefreshCw className={cn('h-4 w-4 mr-2', loading && 'animate-spin')} />
            Research
          </Button>
        </div>
      </div>

      {error && (
        <div className="flex items-center space-x-3 p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
          <XCircle className="h-5 w-5 text-destructive" />
          <div>
            <p className="font-medium text-destructive">Research Error</p>
            <p className="text-sm text-destructive/80">{error}</p>
          </div>
        </div>
      )}

      {researchData && (
        <>
          {/* Market Sentiment Overview */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-card-foreground">Market Sentiment</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Overall Sentiment */}
              <div className={cn('p-4 rounded-lg border', sentimentColors[researchData.market_sentiment.overall_sentiment])}>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium">Overall Sentiment</h4>
                  {getSentimentIcon(researchData.market_sentiment.overall_sentiment)}
                </div>
                <div className="space-y-1">
                  <p className="text-lg font-semibold capitalize">
                    {researchData.market_sentiment.overall_sentiment.toLowerCase()}
                  </p>
                  <p className="text-sm">
                    Confidence: {formatPercentage(researchData.market_sentiment.confidence_score / 100)}
                  </p>
                </div>
              </div>

              {/* News Sentiment */}
              <div className="p-4 rounded-lg border bg-muted/30">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium flex items-center space-x-2">
                    <Newspaper className="h-4 w-4" />
                    <span>News Sentiment</span>
                  </h4>
                </div>
                <div className="space-y-1">
                  <p className="text-lg font-semibold">
                    {getSentimentScore(researchData.market_sentiment.news_sentiment)}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Score: {researchData.market_sentiment.news_sentiment.toFixed(2)}
                  </p>
                </div>
              </div>

              {/* Social Sentiment */}
              <div className="p-4 rounded-lg border bg-muted/30">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium flex items-center space-x-2">
                    <MessageSquare className="h-4 w-4" />
                    <span>Social Sentiment</span>
                  </h4>
                </div>
                <div className="space-y-1">
                  <p className="text-lg font-semibold">
                    {getSentimentScore(researchData.market_sentiment.social_sentiment)}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Score: {researchData.market_sentiment.social_sentiment.toFixed(2)}
                  </p>
                </div>
              </div>
            </div>

            {/* Key Factors */}
            {researchData.market_sentiment.key_factors.length > 0 && (
              <div>
                <h4 className="font-medium mb-2">Key Factors</h4>
                <div className="space-y-1">
                  {researchData.market_sentiment.key_factors.map((factor, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
                      <span className="text-sm">{factor}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Trading Recommendation */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-card-foreground">Trading Recommendation</h3>
            <div className={cn('p-4 rounded-lg border', actionColors[researchData.trading_recommendation.action])}>
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-medium text-lg capitalize">
                  {researchData.trading_recommendation.action}
                </h4>
                <div className={cn('px-2 py-1 rounded text-sm font-medium', getConfidenceColor(researchData.trading_recommendation.confidence))}>
                  {formatPercentage(researchData.trading_recommendation.confidence)} confidence
                </div>
              </div>
              <p className="text-sm mt-2">{researchData.trading_recommendation.reasoning}</p>
            </div>
          </div>

          {/* Analysis Summary */}
          {researchData.analysis_summary && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-card-foreground">Analysis Summary</h3>
              <div className="p-4 bg-muted/30 rounded-lg border">
                <p className="text-sm leading-relaxed">{researchData.analysis_summary}</p>
              </div>
            </div>
          )}

          {/* Recent News Articles */}
          {researchData.news_articles.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-card-foreground">Recent News</h3>
              <div className="space-y-3">
                {researchData.news_articles.slice(0, 5).map((article, index) => (
                  <div key={index} className="p-4 border rounded-lg bg-muted/20">
                    <div className="flex items-start justify-between space-x-4">
                      <div className="flex-1">
                        <h4 className="font-medium text-sm mb-1">{article.title}</h4>
                        <p className="text-sm text-muted-foreground mb-2 line-clamp-2">
                          {article.snippet}
                        </p>
                        <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                          <span className="flex items-center space-x-1">
                            <Calendar className="h-3 w-3" />
                            <span>{formatDate(article.published_date)}</span>
                          </span>
                          <span>{article.source}</span>
                          {article.sentiment !== undefined && (
                            <span className={cn(
                              'px-2 py-1 rounded',
                              article.sentiment > 0.1 
                                ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                                : article.sentiment < -0.1 
                                ? 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                                : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
                            )}>
                              {getSentimentScore(article.sentiment)}
                            </span>
                          )}
                        </div>
                      </div>
                      <a 
                        href={article.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-muted-foreground hover:text-foreground transition-colors"
                      >
                        <ExternalLink className="h-4 w-4" />
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Metadata */}
          <div className="flex justify-between items-center pt-4 border-t text-sm text-muted-foreground">
            <span>Symbol: {researchData.symbol}</span>
            <span>Last updated: {formatDate(researchData.timestamp)}</span>
          </div>
        </>
      )}
    </div>
  )
}

export default MarketResearchPanel