"""
System monitoring and health check tasks
"""

import psutil
import logging
from datetime import datetime
from typing import Dict, Any

from celery import current_task
from app.core.celery_app import celery_app
from app.core.cache import get_market_cache, get_embedding_cache
from app.core.database import get_redis, engine

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='app.tasks.monitoring.system_health_check')
def system_health_check(self) -> Dict[str, Any]:
    """Comprehensive system health check"""
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'starting', 'progress': 0}
        )

        health_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'components': {}
        }

        # Check system resources
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'system_resources', 'progress': 20}
        )

        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        health_data['components']['system'] = {
            'status': 'healthy',
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'disk_percent': disk.percent,
            'disk_free_gb': round(disk.free / (1024**3), 2)
        }

        # Check memory thresholds
        if memory.percent > 90 or cpu_percent > 90 or disk.percent > 90:
            health_data['components']['system']['status'] = 'critical'
            health_data['overall_status'] = 'critical'
        elif memory.percent > 80 or cpu_percent > 80 or disk.percent > 80:
            health_data['components']['system']['status'] = 'warning'
            if health_data['overall_status'] == 'healthy':
                health_data['overall_status'] = 'warning'

        # Check database connection
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'database', 'progress': 40}
        )

        try:
            # Test database connection
            with engine.connect() as conn:
                result = conn.execute("SELECT 1").fetchone()
                db_status = 'healthy' if result else 'unhealthy'

            # Get connection pool info
            pool_info = {
                'pool_size': engine.pool.size(),
                'checked_in': engine.pool.checkedin(),
                'checked_out': engine.pool.checkedout(),
                'overflow': engine.pool.overflow(),
                'invalid': engine.pool.invalid(),
            }

            health_data['components']['database'] = {
                'status': db_status,
                'pool_info': pool_info
            }

            if db_status == 'unhealthy':
                health_data['overall_status'] = 'critical'

        except Exception as e:
            health_data['components']['database'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_data['overall_status'] = 'critical'

        # Check Redis connection
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'redis', 'progress': 60}
        )

        try:
            redis_client = get_redis()
            redis_info = redis_client.info()
            redis_status = 'healthy'

            health_data['components']['redis'] = {
                'status': redis_status,
                'connected_clients': redis_info.get('connected_clients', 0),
                'used_memory_human': redis_info.get('used_memory_human', 'unknown'),
                'keyspace_hits': redis_info.get('keyspace_hits', 0),
                'keyspace_misses': redis_info.get('keyspace_misses', 0),
            }

        except Exception as e:
            health_data['components']['redis'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_data['overall_status'] = 'critical'

        # Check cache health
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'cache', 'progress': 80}
        )

        try:
            market_cache = get_market_cache()
            embedding_cache = get_embedding_cache()

            # For synchronous operation, we'll create a simple test
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            market_health = loop.run_until_complete(market_cache.health_check())
            embedding_health = loop.run_until_complete(embedding_cache.health_check())

            loop.close()

            health_data['components']['cache'] = {
                'market_cache': market_health,
                'embedding_cache': embedding_health
            }

        except Exception as e:
            health_data['components']['cache'] = {
                'status': 'unhealthy',
                'error': str(e)
            }

        # Check process info
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'processes', 'progress': 90}
        )

        process = psutil.Process()
        health_data['components']['process'] = {
            'status': 'healthy',
            'pid': process.pid,
            'memory_percent': process.memory_percent(),
            'cpu_percent': process.cpu_percent(),
            'num_threads': process.num_threads(),
            'create_time': datetime.fromtimestamp(process.create_time()).isoformat(),
        }

        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'completed', 'progress': 100}
        )

        return health_data

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'critical',
            'error': str(e)
        }

@celery_app.task(bind=True, name='app.tasks.monitoring.memory_optimization')
def memory_optimization(self) -> Dict[str, Any]:
    """Perform memory optimization and cleanup"""
    try:
        initial_memory = psutil.virtual_memory()

        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'cache_cleanup', 'progress': 20}
        )

        # Clear old cache entries
        market_cache = get_market_cache()
        embedding_cache = get_embedding_cache()

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Clear expired cache entries
        initial_market_size = len(market_cache.l1_cache) if hasattr(market_cache, 'l1_cache') else 0
        initial_embedding_size = len(embedding_cache.l1_cache) if hasattr(embedding_cache, 'l1_cache') else 0

        loop.close()

        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'garbage_collection', 'progress': 60}
        )

        # Force garbage collection
        import gc
        collected = gc.collect()

        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'database_cleanup', 'progress': 80}
        )

        # Database connection pool cleanup
        engine.dispose()

        final_memory = psutil.virtual_memory()

        memory_freed = initial_memory.used - final_memory.used

        return {
            'status': 'completed',
            'timestamp': datetime.utcnow().isoformat(),
            'initial_memory_mb': round(initial_memory.used / (1024**2), 2),
            'final_memory_mb': round(final_memory.used / (1024**2), 2),
            'memory_freed_mb': round(memory_freed / (1024**2), 2),
            'garbage_collected': collected,
            'cache_info': {
                'initial_market_cache_size': initial_market_size,
                'initial_embedding_cache_size': initial_embedding_size,
            }
        }

    except Exception as e:
        logger.error(f"Memory optimization failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }

@celery_app.task(bind=True, name='app.tasks.monitoring.performance_metrics')
def collect_performance_metrics(self) -> Dict[str, Any]:
    """Collect detailed performance metrics"""
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'collecting', 'progress': 0}
        )

        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'system': {},
            'database': {},
            'cache': {},
            'application': {}
        }

        # System metrics
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'system_metrics', 'progress': 25}
        )

        cpu_times = psutil.cpu_times()
        memory = psutil.virtual_memory()
        network = psutil.net_io_counters()
        disk_io = psutil.disk_io_counters()

        metrics['system'] = {
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'times': {
                    'user': cpu_times.user,
                    'system': cpu_times.system,
                    'idle': cpu_times.idle
                }
            },
            'memory': {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'percent': memory.percent
            },
            'network': {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            },
            'disk': {
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0,
                'read_count': disk_io.read_count if disk_io else 0,
                'write_count': disk_io.write_count if disk_io else 0
            }
        }

        # Database metrics
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'database_metrics', 'progress': 50}
        )

        try:
            pool = engine.pool
            metrics['database'] = {
                'pool_size': pool.size(),
                'checked_in': pool.checkedin(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
                'invalid': pool.invalid(),
                'total_connections': pool.size() + pool.overflow()
            }
        except Exception as e:
            metrics['database'] = {'error': str(e)}

        # Cache metrics
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'cache_metrics', 'progress': 75}
        )

        try:
            redis_client = get_redis()
            redis_info = redis_client.info()

            metrics['cache'] = {
                'redis': {
                    'used_memory': redis_info.get('used_memory', 0),
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'total_commands_processed': redis_info.get('total_commands_processed', 0),
                    'keyspace_hits': redis_info.get('keyspace_hits', 0),
                    'keyspace_misses': redis_info.get('keyspace_misses', 0),
                    'expired_keys': redis_info.get('expired_keys', 0),
                }
            }

            # Calculate hit rate
            hits = redis_info.get('keyspace_hits', 0)
            misses = redis_info.get('keyspace_misses', 0)
            total_requests = hits + misses
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0

            metrics['cache']['redis']['hit_rate_percent'] = round(hit_rate, 2)

        except Exception as e:
            metrics['cache'] = {'error': str(e)}

        # Application metrics
        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'application_metrics', 'progress': 90}
        )

        process = psutil.Process()
        metrics['application'] = {
            'pid': process.pid,
            'memory_percent': process.memory_percent(),
            'memory_rss_mb': round(process.memory_info().rss / (1024**2), 2),
            'memory_vms_mb': round(process.memory_info().vms / (1024**2), 2),
            'cpu_percent': process.cpu_percent(),
            'num_threads': process.num_threads(),
            'num_fds': process.num_fds() if hasattr(process, 'num_fds') else None,
            'create_time': datetime.fromtimestamp(process.create_time()).isoformat(),
        }

        current_task.update_state(
            state='PROGRESS',
            meta={'stage': 'completed', 'progress': 100}
        )

        return metrics

    except Exception as e:
        logger.error(f"Performance metrics collection failed: {e}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }


@celery_app.task(bind=True, name='app.tasks.monitoring.update_database_metrics')
def update_database_metrics(self) -> Dict[str, Any]:
    """Update database connection pool metrics for Prometheus monitoring (Task 1.5)"""
    try:
        from app.core.database import update_connection_pool_metrics
        
        # Update the metrics
        update_connection_pool_metrics()
        
        return {
            'status': 'success',
            'timestamp': datetime.utcnow().isoformat(),
            'message': 'Database connection pool metrics updated'
        }
    except Exception as e:
        logger.error(f"Failed to update database metrics: {e}")
        return {
            'status': 'error',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }
