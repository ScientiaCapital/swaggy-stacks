import React from 'react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { 
  Activity,
  BarChart3,
  Brain,
  Search,
  TrendingUp,
  Zap
} from 'lucide-react'

export function Dashboard() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-foreground">Swaggy Stacks</h1>
        <p className="text-lg text-muted-foreground mt-2">
          Advanced Markov Trading System with MCP Integration
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="p-6 bg-card rounded-lg border">
          <div className="flex items-center space-x-2 mb-2">
            <Activity className="h-5 w-5 text-green-600" />
            <h3 className="font-semibold">System Status</h3>
          </div>
          <p className="text-2xl font-bold text-green-600">Healthy</p>
          <p className="text-sm text-muted-foreground">All systems operational</p>
        </div>

        <div className="p-6 bg-card rounded-lg border">
          <div className="flex items-center space-x-2 mb-2">
            <TrendingUp className="h-5 w-5 text-blue-600" />
            <h3 className="font-semibold">Active Trades</h3>
          </div>
          <p className="text-2xl font-bold">12</p>
          <p className="text-sm text-muted-foreground">Positions open</p>
        </div>

        <div className="p-6 bg-card rounded-lg border">
          <div className="flex items-center space-x-2 mb-2">
            <BarChart3 className="h-5 w-5 text-purple-600" />
            <h3 className="font-semibold">Portfolio Value</h3>
          </div>
          <p className="text-2xl font-bold">$125,432</p>
          <p className="text-sm text-green-600">+2.3% today</p>
        </div>

        <div className="p-6 bg-card rounded-lg border">
          <div className="flex items-center space-x-2 mb-2">
            <Zap className="h-5 w-5 text-yellow-600" />
            <h3 className="font-semibold">AI Insights</h3>
          </div>
          <p className="text-2xl font-bold">5</p>
          <p className="text-sm text-muted-foreground">New recommendations</p>
        </div>
      </div>

      {/* MCP Integration Section */}
      <div className="bg-card rounded-lg border p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold">MCP Integration Dashboard</h2>
            <p className="text-muted-foreground">
              Access advanced MCP-powered features for market research and AI insights
            </p>
          </div>
          <Link href="/mcp">
            <Button>
              <Brain className="h-4 w-4 mr-2" />
              Open MCP Dashboard
            </Button>
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-muted/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <Activity className="h-4 w-4 text-green-600" />
              <h3 className="font-medium">System Health</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Monitor MCP server status and service health in real-time
            </p>
          </div>

          <div className="p-4 bg-muted/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <Search className="h-4 w-4 text-blue-600" />
              <h3 className="font-medium">Market Research</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              AI-powered market sentiment analysis and news aggregation
            </p>
          </div>

          <div className="p-4 bg-muted/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <Brain className="h-4 w-4 text-purple-600" />
              <h3 className="font-medium">AI Insights</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Sequential thinking processes and intelligent trading recommendations
            </p>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-card rounded-lg border p-6">
        <h2 className="text-xl font-semibold mb-4">Recent Activity</h2>
        <div className="space-y-3">
          {[
            { time: '10:30 AM', event: 'BUY order executed for AAPL (100 shares)', type: 'trade' },
            { time: '10:15 AM', event: 'Market research completed for TSLA', type: 'research' },
            { time: '09:45 AM', event: 'AI insight generated: Strong bullish sentiment detected', type: 'insight' },
            { time: '09:30 AM', event: 'Daily health check completed - All systems healthy', type: 'system' },
          ].map((activity, index) => (
            <div key={index} className="flex items-center space-x-3 p-3 bg-muted/20 rounded">
              <div className="w-2 h-2 bg-primary rounded-full"></div>
              <div className="flex-1">
                <p className="text-sm">{activity.event}</p>
                <p className="text-xs text-muted-foreground">{activity.time}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}