"""
API Models for Godot Assistant Executor Agent.

This module defines Pydantic models for request/response validation
in the executor API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class ExecutePlanRequest(BaseModel):
    """Request model for executing a plan."""
    plan_text: str = Field(..., description="Plan text from planning agent")
    execution_context: Optional[Dict[str, Any]] = Field(None, description="Additional execution context")
    validate_before_execution: bool = Field(True, description="Whether to validate before execution")
    dry_run: bool = Field(False, description="Whether to perform a dry run without executing")
    stream_progress: bool = Field(True, description="Whether to stream execution progress")


class ValidatePlanRequest(BaseModel):
    """Request model for validating a plan."""
    plan_text: str = Field(..., description="Plan text to validate")
    execution_context: Optional[Dict[str, Any]] = Field(None, description="Execution context for validation")


class ExecuteStepRequest(BaseModel):
    """Request model for executing a single step."""
    step_id: str = Field(..., description="ID of the step to execute")
    execution_context: Optional[Dict[str, Any]] = Field(None, description="Additional execution context")
    validate_before_execution: bool = Field(True, description="Whether to validate before execution")


class RollbackRequest(BaseModel):
    """Request model for rolling back execution."""
    execution_id: str = Field(..., description="ID of the execution to rollback")
    step_id: Optional[str] = Field(None, description="Specific step to rollback (optional)")


class CancelExecutionRequest(BaseModel):
    """Request model for canceling an execution."""
    execution_id: str = Field(..., description="ID of the execution to cancel")
    reason: Optional[str] = Field(None, description="Reason for cancellation")


class StreamEventModel(BaseModel):
    """Model for stream events sent to clients."""
    event_type: str = Field(..., description="Type of event")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    timestamp: datetime = Field(..., description="Event timestamp")
    execution_id: str = Field(..., description="ID of the execution")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ValidationResult(BaseModel):
    """Model for validation results."""
    is_valid: bool = Field(..., description="Whether the plan is valid")
    can_execute: bool = Field(..., description="Whether the plan can be executed")
    issues: List[Dict[str, Any]] = Field(..., description="List of validation issues")
    execution_time_estimate: int = Field(..., description="Estimated execution time in seconds")
    risk_assessment: str = Field(..., description="Overall risk assessment")
    recommendations: List[str] = Field(..., description="Recommendations for improvement")


class ExecutionStatusResponse(BaseModel):
    """Response model for execution status."""
    execution_id: str = Field(..., description="ID of the execution")
    plan_id: str = Field(..., description="ID of the plan being executed")
    overall_status: str = Field(..., description="Overall execution status")
    current_step_id: Optional[str] = Field(None, description="ID of currently executing step")
    completed_steps: List[str] = Field(..., description="List of completed step IDs")
    failed_steps: List[str] = Field(..., description="List of failed step IDs")
    started_at: datetime = Field(..., description="When execution started")
    updated_at: datetime = Field(..., description="When execution was last updated")
    rollback_available: bool = Field(..., description="Whether rollback is available")
    progress_percentage: float = Field(..., description="Progress percentage (0-100)")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StepResult(BaseModel):
    """Model for individual step execution results."""
    step_id: str = Field(..., description="ID of the step")
    status: str = Field(..., description="Execution status")
    started_at: datetime = Field(..., description="When step execution started")
    completed_at: Optional[datetime] = Field(None, description="When step execution completed")
    execution_time: Optional[float] = Field(None, description="Time taken to execute in seconds")
    output: Optional[str] = Field(None, description="Output from the execution")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    tool_results: List[Dict[str, Any]] = Field(default_factory=list, description="Results from tool calls")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ExecutionHistoryResponse(BaseModel):
    """Response model for execution history."""
    active_executions: List[ExecutionStatusResponse] = Field(..., description="List of active executions")
    total_active: int = Field(..., description="Total number of active executions")
    max_concurrent: int = Field(10, description="Maximum concurrent executions")


class PlanParseResult(BaseModel):
    """Model for plan parsing results."""
    success: bool = Field(..., description="Whether parsing was successful")
    plan_id: Optional[str] = Field(None, description="ID of the parsed plan")
    title: Optional[str] = Field(None, description="Title of the plan")
    step_count: Optional[int] = Field(None, description="Number of steps in the plan")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in seconds")
    error_message: Optional[str] = Field(None, description="Error message if parsing failed")


class DryRunResult(BaseModel):
    """Model for dry run results."""
    can_execute: bool = Field(..., description="Whether the step can be executed")
    issues: List[Dict[str, Any]] = Field(..., description="List of issues found")
    estimated_duration: int = Field(..., description="Estimated duration in seconds")
    resource_requirements: List[Dict[str, Any]] = Field(..., description="Required resources")


class ToolInfo(BaseModel):
    """Model for tool information."""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    category: str = Field(..., description="Tool category")
    required_parameters: List[str] = Field(..., description="Required parameters")
    optional_parameters: List[str] = Field(..., description="Optional parameters")
    risk_level: str = Field(..., description="Risk level of the tool")


class AvailableToolsResponse(BaseModel):
    """Response model for available tools."""
    tools: List[ToolInfo] = Field(..., description="List of available tools")
    categories: List[str] = Field(..., description="Available tool categories")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SuccessResponse(BaseModel):
    """Standard success response model."""
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional response data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchOperationRequest(BaseModel):
    """Request model for batch operations."""
    operations: List[Dict[str, Any]] = Field(..., description="List of operations to perform")
    continue_on_error: bool = Field(False, description="Whether to continue on individual failures")
    validate_all_first: bool = Field(True, description="Whether to validate all operations before executing")


class BatchOperationResult(BaseModel):
    """Result model for batch operations."""
    total_operations: int = Field(..., description="Total number of operations")
    successful_operations: int = Field(..., description="Number of successful operations")
    failed_operations: int = Field(..., description="Number of failed operations")
    results: List[Dict[str, Any]] = Field(..., description="Individual operation results")
    execution_time: float = Field(..., description="Total execution time in seconds")


class ExecutionStatistics(BaseModel):
    """Model for execution statistics."""
    total_executions: int = Field(..., description="Total number of executions")
    successful_executions: int = Field(..., description="Number of successful executions")
    failed_executions: int = Field(..., description="Number of failed executions")
    average_execution_time: float = Field(..., description="Average execution time in seconds")
    most_used_tools: List[Dict[str, Any]] = Field(..., description="Most frequently used tools")
    common_errors: List[Dict[str, Any]] = Field(..., description="Most common errors")


class PlanTemplate(BaseModel):
    """Model for plan templates."""
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    category: str = Field(..., description="Template category")
    template_text: str = Field(..., description="Template plan text")
    parameters: List[Dict[str, Any]] = Field(..., description="Template parameters")
    usage_count: int = Field(0, description="Number of times template has been used")


class PlanTemplateResponse(BaseModel):
    """Response model for plan templates."""
    templates: List[PlanTemplate] = Field(..., description="List of available templates")
    categories: List[str] = Field(..., description="Available template categories")


class ExecutionContextInfo(BaseModel):
    """Model for execution context information."""
    godot_connected: bool = Field(..., description="Whether Godot is connected")
    project_path: Optional[str] = Field(None, description="Path to the Godot project")
    current_scene: Optional[str] = Field(None, description="Currently open scene")
    available_tools: List[str] = Field(..., description="List of available tools")
    system_info: Dict[str, Any] = Field(..., description="System information")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Overall health status")
    components: Dict[str, str] = Field(..., description="Status of individual components")
    uptime: float = Field(..., description="Service uptime in seconds")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Request/Response for WebSocket connections
class WebSocketMessage(BaseModel):
    """Base model for WebSocket messages."""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WebSocketCommand(WebSocketMessage):
    """WebSocket command message."""
    command: str = Field(..., description="Command to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Command parameters")


class WebSocketResponse(WebSocketMessage):
    """WebSocket response message."""
    success: bool = Field(..., description="Whether command was successful")
    result: Optional[Dict[str, Any]] = Field(None, description="Command result")
    error: Optional[str] = Field(None, description="Error message if failed")


# Custom validators
class ExecutePlanRequestValidator(ExecutePlanRequest):
    """Validator with additional checks for ExecutePlanRequest."""

    @validator('plan_text')
    def validate_plan_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Plan text cannot be empty')
        if len(v) > 100000:  # 100KB limit
            raise ValueError('Plan text is too large (max 100KB)')
        return v

    @validator('execution_context')
    def validate_execution_context(cls, v):
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError('Execution context must be a dictionary')
            if len(v) > 100:  # Limit number of context items
                raise ValueError('Execution context has too many items')
        return v


class ValidatePlanRequestValidator(ValidatePlanRequest):
    """Validator with additional checks for ValidatePlanRequest."""

    @validator('plan_text')
    def validate_plan_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Plan text cannot be empty')
        return v