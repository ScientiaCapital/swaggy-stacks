'use client'

import React, { useState } from 'react'
import { 
  MCPHealthDashboard, 
  MarketResearchPanel, 
  AIInsightsWidget 
} from '@/components/mcp'
import { Button } from '@/components/ui/button'
import {
  Activity,
  Search,
  Brain,
  LayoutDashboard,
  Maximize2,
  Minimize2
} from 'lucide-react'

type DashboardView = 'overview' | 'health' | 'research' | 'insights'

export default function MCPDashboard() {
  const [activeView, setActiveView] = useState<DashboardView>('overview')
  const [isFullscreen, setIsFullscreen] = useState(false)

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen)
  }

  const NavigationButton = ({ 
    view, 
    icon: Icon, 
    label 
  }: { 
    view: DashboardView
    icon: React.ComponentType<any>
    label: string 
  }) => (
    <Button
      variant={activeView === view ? 'default' : 'outline'}
      size="sm"
      onClick={() => setActiveView(view)}
      className="flex items-center space-x-2"
    >
      <Icon className="h-4 w-4" />
      <span>{label}</span>
    </Button>
  )

  const renderContent = () => {
    switch (activeView) {
      case 'health':
        return <MCPHealthDashboard />
      case 'research':
        return <MarketResearchPanel />
      case 'insights':
        return <AIInsightsWidget />
      case 'overview':
      default:
        return (
          <div className="space-y-8">
            {/* Overview Grid */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
              <div className="space-y-6">
                <MCPHealthDashboard />
                <AIInsightsWidget />
              </div>
              <div>
                <MarketResearchPanel />
              </div>
            </div>
          </div>
        )
    }
  }

  return (
    <div className={`min-h-screen bg-background ${isFullscreen ? 'fixed inset-0 z-50 overflow-auto' : ''}`}>
      <div className="container mx-auto py-8 space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold text-foreground">MCP Integration Dashboard</h1>
            <p className="text-lg text-muted-foreground mt-2">
              Monitor MCP services, market research, and AI-driven insights for the Swaggy Stacks trading system
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={toggleFullscreen}
            >
              {isFullscreen ? (
                <Minimize2 className="h-4 w-4" />
              ) : (
                <Maximize2 className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>

        {/* Navigation */}
        <div className="flex items-center space-x-4 border-b pb-4">
          <NavigationButton view="overview" icon={LayoutDashboard} label="Overview" />
          <NavigationButton view="health" icon={Activity} label="System Health" />
          <NavigationButton view="research" icon={Search} label="Market Research" />
          <NavigationButton view="insights" icon={Brain} label="AI Insights" />
        </div>

        {/* Content */}
        <div className="pb-8">
          {renderContent()}
        </div>

        {/* Footer */}
        <div className="border-t pt-6">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <div className="flex items-center space-x-4">
              <span>🤖 Powered by MCP Integration</span>
              <span>•</span>
              <span>Real-time market analysis</span>
              <span>•</span>
              <span>AI-driven insights</span>
            </div>
            <div className="flex items-center space-x-2">
              <span>Swaggy Stacks Trading System</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}