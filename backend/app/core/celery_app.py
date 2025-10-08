"""
Celery configuration for distributed task processing in trading system
"""

import os
from celery import Celery
from kombu import Queue
from app.core.config import settings

# Create Celery instance
celery_app = Celery("swaggy_stacks")

# Configure Celery with Redis as broker and backend
celery_app.conf.update(
    # Broker settings
    broker_url=settings.REDIS_URL,
    result_backend=settings.REDIS_URL,

    # Task settings
    task_serializer='pickle',
    accept_content=['pickle', 'json'],
    result_serializer='pickle',
    timezone='UTC',
    enable_utc=True,

    # Performance optimizations
    task_routes={
        'app.tasks.market_data.*': {'queue': 'market_data'},
        'app.tasks.analysis.*': {'queue': 'analysis'},
        'app.tasks.trading.*': {'queue': 'trading'},
        'app.tasks.notifications.*': {'queue': 'notifications'},
        'app.tasks.monitoring.*': {'queue': 'monitoring'},
    },

    # Queue configurations
    task_default_queue='default',
    task_queues=(
        Queue('default', routing_key='default'),
        Queue('market_data', routing_key='market_data'),
        Queue('analysis', routing_key='analysis'),
        Queue('trading', routing_key='trading'),
        Queue('notifications', routing_key='notifications'),
        Queue('monitoring', routing_key='monitoring'),
    ),

    # Worker settings
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=100,
    worker_disable_rate_limits=False,
    worker_send_task_events=True,

    # Result backend settings
    result_expires=3600,  # 1 hour
    result_cache_max=10000,

    # Task execution settings
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,  # 10 minutes
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Beat scheduler settings (for periodic tasks)
    beat_schedule={
        'update-market-data': {
            'task': 'app.tasks.market_data.update_all_symbols',
            'schedule': 60.0,  # Every minute during market hours
        },
        'cleanup-old-data': {
            'task': 'app.tasks.maintenance.cleanup_old_data',
            'schedule': 3600.0,  # Every hour
        },
        'health-check': {
            'task': 'app.tasks.monitoring.system_health_check',
            'schedule': 300.0,  # Every 5 minutes
        },
        'calculate-portfolio-metrics': {
            'task': 'app.tasks.analysis.calculate_portfolio_metrics',
            'schedule': 300.0,  # Every 5 minutes
        },
        'update-database-metrics': {
            'task': 'app.tasks.monitoring.update_database_metrics',
            'schedule': 30.0,  # Every 30 seconds for real-time pool monitoring
        },
    },

    # Security
    worker_hijack_root_logger=False,
    worker_log_color=False,

    # Redis connection settings
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,

    # Visibility timeout
    visibility_timeout=3600,
)

# Auto-discover tasks
celery_app.autodiscover_tasks([
    'app.tasks.market_data',
    'app.tasks.analysis',
    'app.tasks.trading',
    'app.tasks.notifications',
    'app.tasks.monitoring',
    'app.tasks.maintenance',
])

@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery configuration"""
    print(f'Request: {self.request!r}')
    return {'status': 'success', 'worker': self.request.hostname}

# Export for use in other modules
__all__ = ['celery_app']