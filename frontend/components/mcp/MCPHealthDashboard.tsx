'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { cn, formatDate } from '@/lib/utils'
import { 
  Activity, 
  CheckCircle, 
  XCircle, 
  AlertTriangle, 
  RefreshCw, 
  Server,
  Github,
  Database,
  Search,
  Brain,
  Code
} from 'lucide-react'

interface MCPServerStatus {
  name: string
  type: 'github' | 'memory' | 'serena' | 'tavily' | 'sequential_thinking'
  status: 'healthy' | 'degraded' | 'critical' | 'unknown'
  lastCheck: string
  responseTime?: number
  errorCount?: number
  uptime?: string
}

interface MCPServiceStatus {
  name: string
  status: 'healthy' | 'degraded' | 'critical' | 'unknown'
  lastCheck: string
  checks: Record<string, string>
}

interface MCPHealthData {
  overall_status: 'healthy' | 'degraded' | 'critical'
  timestamp: string
  services: Record<string, MCPServiceStatus>
  mcp_servers: Record<string, MCPServerStatus>
  issues: string[]
}

const serverIcons = {
  github: Github,
  memory: Database,
  serena: Code,
  tavily: Search,
  sequential_thinking: Brain,
}

const statusColors = {
  healthy: 'text-green-600 bg-green-50 dark:text-green-400 dark:bg-green-950',
  degraded: 'text-yellow-600 bg-yellow-50 dark:text-yellow-400 dark:bg-yellow-950',
  critical: 'text-red-600 bg-red-50 dark:text-red-400 dark:bg-red-950',
  unknown: 'text-gray-600 bg-gray-50 dark:text-gray-400 dark:bg-gray-950',
}

const statusIcons = {
  healthy: CheckCircle,
  degraded: AlertTriangle,
  critical: XCircle,
  unknown: Server,
}

export function MCPHealthDashboard() {
  const [healthData, setHealthData] = useState<MCPHealthData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastRefresh, setLastRefresh] = useState<string>('')

  const fetchHealthData = async () => {
    try {
      setLoading(true)
      setError(null)

      const response = await fetch('/api/v1/health/mcp', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()
      setHealthData(data)
      setLastRefresh(new Date().toISOString())
    } catch (err) {
      console.error('Failed to fetch MCP health data:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch health data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHealthData()
    
    // Set up periodic refresh every 30 seconds
    const interval = setInterval(fetchHealthData, 30000)
    
    return () => clearInterval(interval)
  }, [])

  const getOverallStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 dark:text-green-400'
      case 'degraded':
        return 'text-yellow-600 dark:text-yellow-400'
      case 'critical':
        return 'text-red-600 dark:text-red-400'
      default:
        return 'text-gray-600 dark:text-gray-400'
    }
  }

  const OverallStatusIcon = ({ status }: { status: string }) => {
    const IconComponent = statusIcons[status as keyof typeof statusIcons] || Server
    return <IconComponent className={cn('h-6 w-6', getOverallStatusColor(status))} />
  }

  if (loading && !healthData) {
    return (
      <div className="space-y-6 p-6 bg-card rounded-lg border">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-card-foreground">MCP System Health</h2>
          <RefreshCw className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="animate-pulse">
              <div className="h-24 bg-muted rounded-lg"></div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (error && !healthData) {
    return (
      <div className="space-y-6 p-6 bg-card rounded-lg border">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-card-foreground">MCP System Health</h2>
          <Button onClick={fetchHealthData} variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </div>
        <div className="flex items-center space-x-3 p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
          <XCircle className="h-5 w-5 text-destructive" />
          <div>
            <p className="font-medium text-destructive">Failed to load health data</p>
            <p className="text-sm text-destructive/80">{error}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 p-6 bg-card rounded-lg border">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Activity className="h-6 w-6 text-primary" />
          <h2 className="text-2xl font-bold text-card-foreground">MCP System Health</h2>
        </div>
        <div className="flex items-center space-x-3">
          <div className="text-sm text-muted-foreground">
            Last updated: {lastRefresh ? formatDate(lastRefresh) : 'Never'}
          </div>
          <Button 
            onClick={fetchHealthData} 
            variant="outline" 
            size="sm"
            disabled={loading}
          >
            <RefreshCw className={cn('h-4 w-4 mr-2', loading && 'animate-spin')} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Overall Status */}
      {healthData && (
        <div className="flex items-center space-x-4 p-4 bg-muted/50 rounded-lg">
          <OverallStatusIcon status={healthData.overall_status} />
          <div>
            <h3 className="font-semibold text-lg capitalize">
              System Status: {healthData.overall_status}
            </h3>
            <p className="text-sm text-muted-foreground">
              Last check: {formatDate(healthData.timestamp)}
            </p>
          </div>
        </div>
      )}

      {/* MCP Servers Grid */}
      {healthData && (
        <>
          <div>
            <h3 className="text-lg font-semibold mb-4 text-card-foreground">MCP Servers</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(healthData.mcp_servers).map(([serverName, serverData]) => {
                const IconComponent = serverIcons[serverData.type] || Server
                const StatusIcon = statusIcons[serverData.status] || Server

                return (
                  <div
                    key={serverName}
                    className={cn(
                      'p-4 rounded-lg border transition-colors',
                      statusColors[serverData.status]
                    )}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <IconComponent className="h-5 w-5" />
                        <h4 className="font-medium capitalize">{serverName}</h4>
                      </div>
                      <StatusIcon className="h-4 w-4" />
                    </div>
                    <div className="space-y-1 text-sm">
                      <p className="capitalize">Status: {serverData.status}</p>
                      {serverData.responseTime && (
                        <p>Response: {serverData.responseTime}ms</p>
                      )}
                      {serverData.errorCount !== undefined && (
                        <p>Errors: {serverData.errorCount}</p>
                      )}
                      <p className="text-xs">
                        Updated: {formatDate(serverData.lastCheck)}
                      </p>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Services Status */}
          <div>
            <h3 className="text-lg font-semibold mb-4 text-card-foreground">Services</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(healthData.services).map(([serviceName, serviceData]) => {
                const StatusIcon = statusIcons[serviceData.status] || Server

                return (
                  <div
                    key={serviceName}
                    className={cn(
                      'p-4 rounded-lg border transition-colors',
                      statusColors[serviceData.status]
                    )}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium capitalize">
                        {serviceName.replace('_', ' ')}
                      </h4>
                      <StatusIcon className="h-4 w-4" />
                    </div>
                    <div className="space-y-1 text-sm">
                      <p className="capitalize">Status: {serviceData.status}</p>
                      <p className="text-xs">
                        Updated: {formatDate(serviceData.lastCheck)}
                      </p>
                      {serviceData.checks && Object.keys(serviceData.checks).length > 0 && (
                        <div className="mt-2">
                          <p className="text-xs font-medium mb-1">Health Checks:</p>
                          {Object.entries(serviceData.checks).map(([check, status]) => (
                            <p key={check} className="text-xs ml-2">
                              • {check}: {status}
                            </p>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Issues */}
          {healthData.issues.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold mb-4 text-card-foreground">Issues</h3>
              <div className="space-y-2">
                {healthData.issues.map((issue, index) => (
                  <div key={index} className="flex items-center space-x-2 p-3 bg-yellow-50 dark:bg-yellow-950 rounded-lg">
                    <AlertTriangle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
                    <p className="text-sm text-yellow-800 dark:text-yellow-200">{issue}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default MCPHealthDashboard