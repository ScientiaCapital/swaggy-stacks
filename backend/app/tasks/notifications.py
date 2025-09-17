"""Notification background tasks"""
from datetime import datetime
from typing import Dict, Any
from app.core.celery_app import celery_app

@celery_app.task(name='app.tasks.notifications.send_alert')
def send_alert(message: str, alert_type: str = 'info') -> Dict[str, Any]:
    """Send alert notification"""
    return {
        'message': message,
        'alert_type': alert_type,
        'status': 'sent',
        'sent_at': datetime.utcnow().isoformat()
    }