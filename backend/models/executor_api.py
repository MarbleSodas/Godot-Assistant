"""
API Models for Executor Agent.

Pydantic models for the executor API.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from ..agents.execution_models import ExecutionPlan, ExecutionStep, ToolCall


class ExecutePlanRequest(BaseModel):
    """Request to execute a plan."""
    plan: ExecutionPlan = Field(..., description="Structured execution plan")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context")


class StreamEventModel(BaseModel):
    """Stream event model."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ExecutionStatus(BaseModel):
    """Execution status response."""
    execution_id: str
    status: str
    current_step: Optional[str] = None
    completed_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)
    started_at: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }