"""
Base models for the Swaggy Stacks application
"""

from sqlalchemy.ext.declarative import declarative_base

# Create base model class
BaseModel = declarative_base()

# This will be the base for all database models
__all__ = ["BaseModel"]
