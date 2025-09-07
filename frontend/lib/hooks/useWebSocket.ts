'use client'

import { useEffect, useRef, useState } from 'react'
import { io, Socket } from 'socket.io-client'

interface WebSocketOptions {
  url?: string
  autoReconnect?: boolean
  reconnectInterval?: number
  maxReconnectAttempts?: number
}

interface WebSocketState {
  socket: Socket | null
  connected: boolean
  error: string | null
  reconnectAttempts: number
}

export function useWebSocket(options: WebSocketOptions = {}) {
  const {
    url = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000',
    autoReconnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5
  } = options

  const [state, setState] = useState<WebSocketState>({
    socket: null,
    connected: false,
    error: null,
    reconnectAttempts: 0
  })

  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const shouldReconnect = useRef(true)

  const connect = () => {
    if (state.socket?.connected) {
      return state.socket
    }

    try {
      const socket = io(url, {
        transports: ['websocket', 'polling'],
        timeout: 5000,
        autoConnect: true
      })

      socket.on('connect', () => {
        setState(prev => ({
          ...prev,
          socket,
          connected: true,
          error: null,
          reconnectAttempts: 0
        }))
        console.log('WebSocket connected')
      })

      socket.on('disconnect', (reason) => {
        setState(prev => ({
          ...prev,
          connected: false,
          error: `Disconnected: ${reason}`
        }))
        console.log('WebSocket disconnected:', reason)

        // Auto reconnect if enabled and not a manual disconnect
        if (autoReconnect && shouldReconnect.current && reason !== 'io client disconnect') {
          handleReconnect()
        }
      })

      socket.on('connect_error', (error) => {
        setState(prev => ({
          ...prev,
          connected: false,
          error: `Connection error: ${error.message}`
        }))
        console.error('WebSocket connection error:', error)

        if (autoReconnect && shouldReconnect.current) {
          handleReconnect()
        }
      })

      setState(prev => ({ ...prev, socket }))
      return socket

    } catch (error) {
      setState(prev => ({
        ...prev,
        error: `Failed to create socket: ${error instanceof Error ? error.message : 'Unknown error'}`
      }))
      return null
    }
  }

  const handleReconnect = () => {
    if (state.reconnectAttempts >= maxReconnectAttempts) {
      setState(prev => ({
        ...prev,
        error: 'Max reconnection attempts reached'
      }))
      return
    }

    setState(prev => ({
      ...prev,
      reconnectAttempts: prev.reconnectAttempts + 1
    }))

    reconnectTimeoutRef.current = setTimeout(() => {
      if (shouldReconnect.current) {
        console.log(`Attempting to reconnect... (${state.reconnectAttempts + 1}/${maxReconnectAttempts})`)
        connect()
      }
    }, reconnectInterval)
  }

  const disconnect = () => {
    shouldReconnect.current = false
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }

    if (state.socket) {
      state.socket.disconnect()
      setState(prev => ({
        ...prev,
        socket: null,
        connected: false,
        error: null,
        reconnectAttempts: 0
      }))
    }
  }

  const emit = (event: string, data?: any) => {
    if (state.socket && state.connected) {
      state.socket.emit(event, data)
      return true
    }
    return false
  }

  const on = (event: string, callback: (data: any) => void) => {
    if (state.socket) {
      state.socket.on(event, callback)
      return () => state.socket?.off(event, callback)
    }
    return () => {}
  }

  const off = (event: string, callback?: (data: any) => void) => {
    if (state.socket) {
      if (callback) {
        state.socket.off(event, callback)
      } else {
        state.socket.off(event)
      }
    }
  }

  // Initialize connection on mount
  useEffect(() => {
    connect()

    return () => {
      shouldReconnect.current = false
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [])

  return {
    socket: state.socket,
    connected: state.connected,
    error: state.error,
    reconnectAttempts: state.reconnectAttempts,
    connect,
    disconnect,
    emit,
    on,
    off
  }
}

// Hook for MCP-specific WebSocket events
export function useMCPWebSocket() {
  const webSocket = useWebSocket()
  const [mcpHealthUpdates, setMcpHealthUpdates] = useState<any>(null)
  const [marketResearchUpdates, setMarketResearchUpdates] = useState<any>(null)
  const [aiInsightUpdates, setAiInsightUpdates] = useState<any>(null)

  useEffect(() => {
    if (!webSocket.connected) return

    // Subscribe to MCP health updates
    const unsubscribeHealth = webSocket.on('mcp_health_update', (data) => {
      setMcpHealthUpdates(data)
    })

    // Subscribe to market research updates
    const unsubscribeResearch = webSocket.on('market_research_update', (data) => {
      setMarketResearchUpdates(data)
    })

    // Subscribe to AI insight updates
    const unsubscribeInsights = webSocket.on('ai_insight_update', (data) => {
      setAiInsightUpdates(data)
    })

    return () => {
      unsubscribeHealth()
      unsubscribeResearch()
      unsubscribeInsights()
    }
  }, [webSocket.connected])

  const subscribeToMCPHealth = () => {
    return webSocket.emit('subscribe', { channel: 'mcp_health' })
  }

  const subscribeToMarketResearch = () => {
    return webSocket.emit('subscribe', { channel: 'market_research' })
  }

  const subscribeToAIInsights = () => {
    return webSocket.emit('subscribe', { channel: 'ai_insights' })
  }

  const requestHealthUpdate = () => {
    return webSocket.emit('request_health_update')
  }

  const requestMarketResearch = (symbol: string) => {
    return webSocket.emit('request_market_research', { symbol })
  }

  return {
    ...webSocket,
    mcpHealthUpdates,
    marketResearchUpdates,
    aiInsightUpdates,
    subscribeToMCPHealth,
    subscribeToMarketResearch,
    subscribeToAIInsights,
    requestHealthUpdate,
    requestMarketResearch
  }
}