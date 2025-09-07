'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { cn, formatDate } from '@/lib/utils'
import {
  Brain,
  MessageSquare,
  Lightbulb,
  ChevronDown,
  ChevronRight,
  RefreshCw,
  Zap,
  Target,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Clock,
  ArrowRight
} from 'lucide-react'

interface ThoughtStep {
  stage: string
  thought: string
  thought_number: number
  next_thought_needed: boolean
  assumptions_challenged?: string[]
  axioms_used?: string[]
  tags?: string[]
}

interface AIInsight {
  id: string
  title: string
  query: string
  timestamp: string
  status: 'processing' | 'completed' | 'error'
  thought_process: ThoughtStep[]
  final_answer?: string
  confidence_score?: number
  key_insights?: string[]
  recommendations?: string[]
  context?: string
}

interface AIInsightsData {
  insights: AIInsight[]
  active_processes: number
  completed_today: number
  average_processing_time: number
}

const stageColors = {
  'Problem Definition': 'bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:border-blue-800',
  'Information Gathering': 'bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-950 dark:text-purple-300 dark:border-purple-800',
  'Research': 'bg-green-50 text-green-700 border-green-200 dark:bg-green-950 dark:text-green-300 dark:border-green-800',
  'Analysis': 'bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-950 dark:text-orange-300 dark:border-orange-800',
  'Synthesis': 'bg-indigo-50 text-indigo-700 border-indigo-200 dark:bg-indigo-950 dark:text-indigo-300 dark:border-indigo-800',
  'Conclusion': 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950 dark:text-emerald-300 dark:border-emerald-800',
  'Critical Questioning': 'bg-red-50 text-red-700 border-red-200 dark:bg-red-950 dark:text-red-300 dark:border-red-800',
  'Planning': 'bg-yellow-50 text-yellow-700 border-yellow-200 dark:bg-yellow-950 dark:text-yellow-300 dark:border-yellow-800',
}

const stageIcons = {
  'Problem Definition': Target,
  'Information Gathering': MessageSquare,
  'Research': MessageSquare,
  'Analysis': Brain,
  'Synthesis': Lightbulb,
  'Conclusion': CheckCircle,
  'Critical Questioning': AlertCircle,
  'Planning': TrendingUp,
}

export function AIInsightsWidget() {
  const [insightsData, setInsightsData] = useState<AIInsightsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expandedInsights, setExpandedInsights] = useState<Set<string>>(new Set())
  const [selectedInsight, setSelectedInsight] = useState<AIInsight | null>(null)

  const fetchInsights = async () => {
    try {
      setLoading(true)
      setError(null)

      const response = await fetch('/api/v1/ai-insights/recent', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()
      setInsightsData(data)
    } catch (err) {
      console.error('Failed to fetch AI insights:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch AI insights')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchInsights()
    
    // Set up periodic refresh every 60 seconds
    const interval = setInterval(fetchInsights, 60000)
    
    return () => clearInterval(interval)
  }, [])

  const toggleInsightExpansion = (insightId: string) => {
    const newExpanded = new Set(expandedInsights)
    if (newExpanded.has(insightId)) {
      newExpanded.delete(insightId)
    } else {
      newExpanded.add(insightId)
    }
    setExpandedInsights(newExpanded)
  }

  const getStageIcon = (stage: string) => {
    const IconComponent = stageIcons[stage as keyof typeof stageIcons] || MessageSquare
    return <IconComponent className="h-4 w-4" />
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-600" />
      case 'processing':
        return <Clock className="h-4 w-4 text-yellow-600 animate-pulse" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-600" />
      default:
        return <Brain className="h-4 w-4 text-muted-foreground" />
    }
  }

  if (loading && !insightsData) {
    return (
      <div className="space-y-6 p-6 bg-card rounded-lg border">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-card-foreground">AI Insights</h2>
          <RefreshCw className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="animate-pulse h-24 bg-muted rounded-lg"></div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 p-6 bg-card rounded-lg border">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Brain className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-card-foreground">AI Insights</h2>
        </div>
        <Button 
          onClick={fetchInsights} 
          variant="outline" 
          size="sm"
          disabled={loading}
        >
          <RefreshCw className={cn('h-4 w-4 mr-2', loading && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      {error && (
        <div className="flex items-center space-x-3 p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
          <AlertCircle className="h-5 w-5 text-destructive" />
          <div>
            <p className="font-medium text-destructive">Failed to load insights</p>
            <p className="text-sm text-destructive/80">{error}</p>
          </div>
        </div>
      )}

      {insightsData && (
        <>
          {/* Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-muted/30 rounded-lg border">
              <div className="flex items-center space-x-2 mb-2">
                <Zap className="h-4 w-4 text-yellow-600" />
                <h3 className="font-medium">Active Processes</h3>
              </div>
              <p className="text-2xl font-bold">{insightsData.active_processes}</p>
            </div>
            <div className="p-4 bg-muted/30 rounded-lg border">
              <div className="flex items-center space-x-2 mb-2">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <h3 className="font-medium">Completed Today</h3>
              </div>
              <p className="text-2xl font-bold">{insightsData.completed_today}</p>
            </div>
            <div className="p-4 bg-muted/30 rounded-lg border">
              <div className="flex items-center space-x-2 mb-2">
                <Clock className="h-4 w-4 text-blue-600" />
                <h3 className="font-medium">Avg. Processing</h3>
              </div>
              <p className="text-2xl font-bold">{insightsData.average_processing_time}s</p>
            </div>
          </div>

          {/* Recent Insights */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-card-foreground">Recent Insights</h3>
            {insightsData.insights.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No recent insights available</p>
              </div>
            ) : (
              <div className="space-y-4">
                {insightsData.insights.map((insight) => (
                  <div key={insight.id} className="border rounded-lg overflow-hidden">
                    {/* Insight Header */}
                    <div 
                      className="p-4 cursor-pointer hover:bg-muted/50 transition-colors"
                      onClick={() => toggleInsightExpansion(insight.id)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3 flex-1">
                          {getStatusIcon(insight.status)}
                          <div className="flex-1 min-w-0">
                            <h4 className="font-medium truncate">{insight.title}</h4>
                            <p className="text-sm text-muted-foreground truncate">
                              {insight.query}
                            </p>
                          </div>
                          {insight.confidence_score && (
                            <div className="text-sm font-medium px-2 py-1 bg-muted rounded">
                              {Math.round(insight.confidence_score * 100)}% confidence
                            </div>
                          )}
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm text-muted-foreground">
                            {formatDate(insight.timestamp)}
                          </span>
                          {expandedInsights.has(insight.id) ? (
                            <ChevronDown className="h-4 w-4" />
                          ) : (
                            <ChevronRight className="h-4 w-4" />
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Expanded Content */}
                    {expandedInsights.has(insight.id) && (
                      <div className="border-t bg-muted/20">
                        {/* Final Answer */}
                        {insight.final_answer && (
                          <div className="p-4 border-b">
                            <h5 className="font-medium mb-2 flex items-center space-x-2">
                              <Lightbulb className="h-4 w-4 text-yellow-600" />
                              <span>Final Answer</span>
                            </h5>
                            <p className="text-sm leading-relaxed">{insight.final_answer}</p>
                          </div>
                        )}

                        {/* Key Insights */}
                        {insight.key_insights && insight.key_insights.length > 0 && (
                          <div className="p-4 border-b">
                            <h5 className="font-medium mb-2">Key Insights</h5>
                            <ul className="space-y-1">
                              {insight.key_insights.map((keyInsight, index) => (
                                <li key={index} className="text-sm flex items-start space-x-2">
                                  <ArrowRight className="h-3 w-3 mt-0.5 text-muted-foreground" />
                                  <span>{keyInsight}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Recommendations */}
                        {insight.recommendations && insight.recommendations.length > 0 && (
                          <div className="p-4 border-b">
                            <h5 className="font-medium mb-2">Recommendations</h5>
                            <ul className="space-y-1">
                              {insight.recommendations.map((recommendation, index) => (
                                <li key={index} className="text-sm flex items-start space-x-2">
                                  <Target className="h-3 w-3 mt-0.5 text-muted-foreground" />
                                  <span>{recommendation}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Thought Process */}
                        {insight.thought_process && insight.thought_process.length > 0 && (
                          <div className="p-4">
                            <h5 className="font-medium mb-3">Thought Process</h5>
                            <div className="space-y-3">
                              {insight.thought_process.map((step, index) => (
                                <div key={index} className="flex space-x-3">
                                  <div className="flex-shrink-0">
                                    <div className={cn(
                                      'inline-flex items-center px-2 py-1 rounded text-xs font-medium border',
                                      stageColors[step.stage as keyof typeof stageColors] || 'bg-muted text-muted-foreground'
                                    )}>
                                      {getStageIcon(step.stage)}
                                      <span className="ml-1">{step.stage}</span>
                                    </div>
                                  </div>
                                  <div className="flex-1 min-w-0">
                                    <p className="text-sm leading-relaxed">{step.thought}</p>
                                    {step.assumptions_challenged && step.assumptions_challenged.length > 0 && (
                                      <div className="mt-2">
                                        <p className="text-xs font-medium text-muted-foreground mb-1">
                                          Assumptions Challenged:
                                        </p>
                                        <ul className="text-xs space-y-1">
                                          {step.assumptions_challenged.map((assumption, i) => (
                                            <li key={i} className="ml-2">• {assumption}</li>
                                          ))}
                                        </ul>
                                      </div>
                                    )}
                                    {step.tags && step.tags.length > 0 && (
                                      <div className="flex flex-wrap gap-1 mt-2">
                                        {step.tags.map((tag, i) => (
                                          <span 
                                            key={i} 
                                            className="inline-block px-2 py-1 text-xs bg-muted rounded"
                                          >
                                            {tag}
                                          </span>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}

export default AIInsightsWidget