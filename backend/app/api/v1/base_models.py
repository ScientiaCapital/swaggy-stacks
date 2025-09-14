"""
Base Model Classes
Common patterns and validations for API models
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator


class BaseSymbolModel(BaseModel):
    """Base model for requests involving stock symbols"""

    symbol: str = Field(..., description="Stock symbol")

    @validator("symbol")
    def symbol_must_be_uppercase(cls, v):
        return v.strip().upper() if isinstance(v, str) else v


class BaseTimestampedModel(BaseModel):
    """Base model with automatic timestamp"""

    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")


class BaseResponseModel(BaseTimestampedModel):
    """Base response model with timestamp"""


class BaseMetricsModel(BaseModel):
    """Base model for metrics and analysis results"""

    confidence: float = Field(..., ge=0, le=1, description="Analysis confidence")
    reasoning: str = Field(..., description="Analysis reasoning")


class BasePaginationModel(BaseModel):
    """Base pagination parameters"""

    page: int = Field(1, ge=1, description="Page number")
    per_page: int = Field(20, ge=1, le=100, description="Items per page")


class BaseFilterModel(BaseModel):
    """Base filter parameters"""

    symbol: Optional[str] = Field(None, description="Filter by symbol")
    strategy: Optional[str] = Field(None, description="Filter by strategy")
    status: Optional[str] = Field(None, description="Filter by status")

    @validator("symbol")
    def symbol_must_be_uppercase(cls, v):
        return v.strip().upper() if isinstance(v, str) and v else v


class BaseTimeRangeModel(BaseModel):
    """Base time range parameters"""

    start_date: Optional[datetime] = Field(None, description="Start date")
    end_date: Optional[datetime] = Field(None, description="End date")

    @validator("end_date")
    def end_date_must_be_after_start_date(cls, v, values):
        if v and "start_date" in values and values["start_date"]:
            if v <= values["start_date"]:
                raise ValueError("end_date must be after start_date")
        return v


class BaseErrorModel(BaseTimestampedModel):
    """Base error response model"""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    code: Optional[str] = Field(None, description="Error code")
