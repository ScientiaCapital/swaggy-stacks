FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies including supervisor
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    supervisor \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies - use core requirements
COPY backend/requirements_core.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir supervisor alpaca-py runpod

# Copy backend application code
COPY backend/ /app/backend/

# Copy trading agents script
COPY live_trading_agents.py /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /var/log/supervisor

# Create supervisor configuration
RUN echo "[supervisord]" > /etc/supervisor/conf.d/supervisord.conf && \
    echo "nodaemon=true" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "[program:api]" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "command=python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "directory=/app" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "autostart=true" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "autorestart=true" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stderr_logfile=/var/log/supervisor/api.err.log" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stdout_logfile=/var/log/supervisor/api.out.log" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "[program:trading-agents]" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "command=python /app/live_trading_agents.py" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "directory=/app" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "autostart=true" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "autorestart=true" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stderr_logfile=/var/log/supervisor/agents.err.log" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stdout_logfile=/var/log/supervisor/agents.out.log" >> /etc/supervisor/conf.d/supervisord.conf

# Health check for the API
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Create startup script
RUN echo '#!/bin/bash' > /app/startup.sh && \
    echo 'echo "Starting Swaggy Stacks Trading System on RunPod..."' >> /app/startup.sh && \
    echo 'echo "API will be available at http://localhost:8000"' >> /app/startup.sh && \
    echo 'echo "Logs available at /var/log/supervisor/"' >> /app/startup.sh && \
    echo '/usr/bin/supervisord' >> /app/startup.sh && \
    chmod +x /app/startup.sh

# RunPod expects this specific entrypoint
ENTRYPOINT ["/app/startup.sh"]