"""
Standalone Prometheus metrics exporter on port 9090 (optional).

This lightweight HTTP proxy provides Prometheus-standard port 9090 access to
existing metrics endpoint. Improves ecosystem compatibility while maintaining
primary /api/v1/monitoring/metrics endpoint.

Enable via: ENABLE_METRICS_PORT_9090=true in .env
"""

import asyncio
import logging
from typing import Optional

import httpx
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SwaggyStacks Metrics Exporter",
    description="Lightweight proxy for Prometheus metrics on standard port 9090",
    version="1.0.0",
)


@app.get("/metrics", response_class=PlainTextResponse)
async def proxy_metrics() -> Response:
    """
    Proxy metrics from backend /api/v1/monitoring/metrics endpoint.
    
    Returns:
        Prometheus-formatted metrics text
    
    Raises:
        HTTPException: If backend is unavailable
    """
    backend_url = "http://backend:8000/api/v1/monitoring/metrics"
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(backend_url)
            response.raise_for_status()
            
            return PlainTextResponse(
                content=response.text,
                media_type="text/plain; version=0.0.4; charset=utf-8",
                headers={
                    "Content-Type": "text/plain; version=0.0.4; charset=utf-8",
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                },
            )
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch metrics from backend: {e}")
        return PlainTextResponse(
            content=f"# Error: Backend metrics unavailable - {str(e)}",
            status_code=503,
            media_type="text/plain",
        )


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "metrics-exporter", "port": 9090}


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting SwaggyStacks Metrics Exporter on port 9090")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9090,
        log_level="info",
    )
