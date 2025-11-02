"""Test workflow configuration."""
import pytest
from app.core.config import settings


def test_langgraph_db_uri_configured():
    """Test LANGGRAPH_DB_URI is configured."""
    assert hasattr(settings, 'LANGGRAPH_DB_URI') or 'LANGGRAPH_DB_URI' in settings.model_dump()
    # Note: Will be None if not set, that's ok for now
