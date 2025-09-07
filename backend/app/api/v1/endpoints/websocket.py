"""
WebSocket endpoints for real-time trading dashboard
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import HTMLResponse
import structlog

from app.websockets.trading_socket import dashboard_websocket
from app.core.deps import get_current_user

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.websocket("/ws/trading-dashboard")
async def websocket_trading_dashboard(websocket: WebSocket):
    """
    WebSocket endpoint for real-time trading dashboard.
    
    Provides:
    - Real-time market data updates
    - Trading signals and recommendations  
    - Portfolio performance updates
    - System health monitoring
    
    Message format:
    {
        "action": "subscribe|unsubscribe|add_symbol|remove_symbol",
        "data_type": "market_data|trading_signals|portfolio|system_health",
        "symbols": ["AAPL", "GOOGL", ...] // for market_data and trading_signals
    }
    """
    # Initialize WebSocket service if needed
    if not dashboard_websocket.trading_agent:
        await dashboard_websocket.initialize()
    
    await dashboard_websocket.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            await dashboard_websocket.handle_message(websocket, message)
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
    finally:
        await dashboard_websocket.disconnect(websocket)


@router.get("/ws/demo")
async def websocket_demo_page():
    """Demo page for WebSocket trading dashboard"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Dashboard WebSocket Demo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; }
        .section { margin-bottom: 20px; padding: 15px; border: 1px solid #333; border-radius: 5px; }
        .controls { display: flex; gap: 10px; margin-bottom: 15px; }
        button { padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; }
        button:hover { background: #0056b3; }
        button:disabled { background: #666; cursor: not-allowed; }
        input, select { padding: 8px; margin: 5px; border: 1px solid #555; background: #333; color: #fff; border-radius: 3px; }
        .status { padding: 10px; border-radius: 5px; font-weight: bold; }
        .status.connected { background: #155724; border: 1px solid #28a745; }
        .status.disconnected { background: #721c24; border: 1px solid #dc3545; }
        .data-display { height: 300px; overflow-y: auto; background: #2a2a2a; padding: 10px; border: 1px solid #555; border-radius: 3px; font-family: monospace; font-size: 12px; }
        .market-data, .trading-signal, .portfolio, .system-health { margin: 5px 0; padding: 8px; border-left: 4px solid; }
        .market-data { border-left-color: #007bff; background: rgba(0,123,255,0.1); }
        .trading-signal { border-left-color: #28a745; background: rgba(40,167,69,0.1); }
        .portfolio { border-left-color: #ffc107; background: rgba(255,193,7,0.1); }
        .system-health { border-left-color: #6f42c1; background: rgba(111,66,193,0.1); }
        .timestamp { color: #888; font-size: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Swaggy Stacks Trading Dashboard</h1>
        
        <div class="section">
            <h3>Connection Status</h3>
            <div id="status" class="status disconnected">Disconnected</div>
            <div class="controls">
                <button id="connectBtn" onclick="connect()">Connect</button>
                <button id="disconnectBtn" onclick="disconnect()" disabled>Disconnect</button>
            </div>
        </div>

        <div class="section">
            <h3>Subscriptions</h3>
            <div class="controls">
                <select id="dataType">
                    <option value="market_data">Market Data</option>
                    <option value="trading_signals">Trading Signals</option>
                    <option value="portfolio">Portfolio</option>
                    <option value="system_health">System Health</option>
                </select>
                <input id="symbols" placeholder="Symbols (comma-separated, e.g., AAPL,GOOGL)" />
                <button onclick="subscribe()">Subscribe</button>
                <button onclick="unsubscribe()">Unsubscribe</button>
            </div>
        </div>

        <div class="section">
            <h3>Symbol Management</h3>
            <div class="controls">
                <input id="newSymbol" placeholder="Symbol (e.g., AAPL)" />
                <button onclick="addSymbol()">Add Symbol</button>
                <button onclick="removeSymbol()">Remove Symbol</button>
            </div>
        </div>

        <div class="section">
            <h3>Live Data Feed</h3>
            <button onclick="clearData()">Clear</button>
            <div id="dataDisplay" class="data-display"></div>
        </div>
    </div>

    <script>
        let ws = null;
        let messageCount = 0;

        function connect() {
            if (ws) return;
            
            const wsUrl = `ws://localhost:8000/api/v1/ws/trading-dashboard`;
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').className = 'status connected';
                document.getElementById('connectBtn').disabled = true;
                document.getElementById('disconnectBtn').disabled = false;
                log('🟢 WebSocket connected');
                
                // Auto-subscribe to some data
                setTimeout(() => {
                    sendMessage({
                        action: 'subscribe',
                        data_type: 'market_data',
                        symbols: ['AAPL', 'GOOGL', 'MSFT']
                    });
                    sendMessage({
                        action: 'subscribe',
                        data_type: 'trading_signals',
                        symbols: ['AAPL', 'GOOGL']
                    });
                    sendMessage({
                        action: 'subscribe',
                        data_type: 'portfolio'
                    });
                    sendMessage({
                        action: 'subscribe',
                        data_type: 'system_health'
                    });
                }, 500);
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                displayMessage(data);
            };
            
            ws.onclose = function() {
                document.getElementById('status').textContent = 'Disconnected';
                document.getElementById('status').className = 'status disconnected';
                document.getElementById('connectBtn').disabled = false;
                document.getElementById('disconnectBtn').disabled = true;
                log('🔴 WebSocket disconnected');
                ws = null;
            };
            
            ws.onerror = function(error) {
                log('❌ WebSocket error: ' + error);
            };
        }

        function disconnect() {
            if (ws) {
                ws.close();
            }
        }

        function sendMessage(message) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify(message));
                log('📤 Sent: ' + JSON.stringify(message));
            } else {
                log('❌ WebSocket not connected');
            }
        }

        function subscribe() {
            const dataType = document.getElementById('dataType').value;
            const symbolsInput = document.getElementById('symbols').value;
            const symbols = symbolsInput ? symbolsInput.split(',').map(s => s.trim()) : [];
            
            sendMessage({
                action: 'subscribe',
                data_type: dataType,
                symbols: symbols
            });
        }

        function unsubscribe() {
            const dataType = document.getElementById('dataType').value;
            const symbolsInput = document.getElementById('symbols').value;
            const symbols = symbolsInput ? symbolsInput.split(',').map(s => s.trim()) : [];
            
            sendMessage({
                action: 'unsubscribe',
                data_type: dataType,
                symbols: symbols
            });
        }

        function addSymbol() {
            const symbol = document.getElementById('newSymbol').value.trim().toUpperCase();
            if (symbol) {
                sendMessage({
                    action: 'add_symbol',
                    symbol: symbol
                });
                document.getElementById('newSymbol').value = '';
            }
        }

        function removeSymbol() {
            const symbol = document.getElementById('newSymbol').value.trim().toUpperCase();
            if (symbol) {
                sendMessage({
                    action: 'remove_symbol',
                    symbol: symbol
                });
                document.getElementById('newSymbol').value = '';
            }
        }

        function displayMessage(data) {
            messageCount++;
            const display = document.getElementById('dataDisplay');
            
            let className = '';
            let emoji = '';
            
            switch(data.type) {
                case 'market_update':
                    className = 'market-data';
                    emoji = '📈';
                    break;
                case 'trading_signal':
                    className = 'trading-signal';
                    emoji = '🎯';
                    break;
                case 'portfolio_update':
                    className = 'portfolio';
                    emoji = '💰';
                    break;
                case 'system_health':
                    className = 'system-health';
                    emoji = '⚡';
                    break;
                default:
                    className = '';
                    emoji = '📊';
            }
            
            const messageDiv = document.createElement('div');
            messageDiv.className = className;
            messageDiv.innerHTML = `
                <div>
                    <strong>${emoji} ${data.type.toUpperCase()}</strong>
                    <span class="timestamp">${new Date().toLocaleTimeString()}</span>
                </div>
                <div>${JSON.stringify(data.data, null, 2)}</div>
            `;
            
            display.appendChild(messageDiv);
            display.scrollTop = display.scrollHeight;
            
            // Limit messages to prevent memory issues
            if (display.children.length > 100) {
                display.removeChild(display.firstChild);
            }
        }

        function log(message) {
            console.log(message);
            displayMessage({
                type: 'system_log',
                data: { message: message }
            });
        }

        function clearData() {
            document.getElementById('dataDisplay').innerHTML = '';
            messageCount = 0;
        }

        // Auto-connect on page load
        window.onload = function() {
            setTimeout(connect, 500);
        };
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@router.websocket("/ws/health")
async def websocket_health_check(websocket: WebSocket):
    """Simple WebSocket health check endpoint"""
    await websocket.accept()
    
    try:
        await websocket.send_text('{"status": "healthy", "service": "websocket", "timestamp": "' + 
                                 str(websocket) + '"}')
        await websocket.receive_text()  # Wait for acknowledgment
        await websocket.send_text('{"message": "WebSocket connection successful"}')
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket health check error", error=str(e))
    finally:
        await websocket.close()