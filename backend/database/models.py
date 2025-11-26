"""
SQLAlchemy ORM models for metrics tracking.

Defines database models for message, session, and project-level metrics.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class MessageMetrics(Base):
    """
    Message-level metrics model.

    Stores token usage and cost information for individual API calls.
    Enhanced with global sequence tracking and partial operation support.
    """
    __tablename__ = "message_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String, unique=True, nullable=False, index=True)
    session_id = Column(String, ForeignKey("session_metrics.session_id", ondelete="CASCADE"), index=True)
    project_id = Column(String, ForeignKey("project_metrics.project_id", ondelete="CASCADE"), index=True)

    # Global sequence tracking for proper message ordering across agents
    global_sequence = Column(Integer, nullable=True, index=True)  # New field for cross-agent ordering
    agent_sequence = Column(Integer, nullable=True)  # Agent-specific sequence number

    # Agent information for multi-agent workflows
    agent_type = Column(String, nullable=True, index=True)  # "planning", "execution", "unknown"
    agent_id = Column(String, nullable=True, index=True)  # Unique agent identifier
    workflow_session_id = Column(String, nullable=True, index=True)  # For correlating planning+execution phases

    # Model information
    model_id = Column(String, nullable=False, index=True)

    # Token counts
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)

    # Cost information (in USD)
    estimated_cost = Column(Float, nullable=False, default=0.0)
    actual_cost = Column(Float, nullable=True)

    # OpenRouter generation ID
    generation_id = Column(String, index=True)

    # Timing information
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    response_time_ms = Column(Integer)

    # Partial operation support
    is_complete = Column(Boolean, nullable=False, default=True, index=True)  # Whether the operation completed
    is_partial = Column(Boolean, nullable=False, default=False, index=True)  # Whether this is partial data
    recovery_session_id = Column(String, nullable=True, index=True)  # ID for recovery operations
    checkpoint_timestamp = Column(DateTime, nullable=True)  # When partial data was checkpointed

    # Additional metadata
    stop_reason = Column(String)
    tool_calls_count = Column(Integer, default=0)
    tool_errors_count = Column(Integer, default=0)

    # Enhanced metadata for debugging and recovery
    error_details = Column(Text, nullable=True)  # Error information if operation failed
    completion_status = Column(String, nullable=True)  # Status of operation completion

    # Relationships
    session = relationship("SessionMetrics", back_populates="messages")
    project = relationship("ProjectMetrics", back_populates="messages")

    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "global_sequence": self.global_sequence,
            "agent_sequence": self.agent_sequence,
            "agent_type": self.agent_type,
            "agent_id": self.agent_id,
            "workflow_session_id": self.workflow_session_id,
            "model_id": self.model_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
            "actual_cost": self.actual_cost,
            "generation_id": self.generation_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "response_time_ms": self.response_time_ms,
            "is_complete": self.is_complete,
            "is_partial": self.is_partial,
            "recovery_session_id": self.recovery_session_id,
            "checkpoint_timestamp": self.checkpoint_timestamp.isoformat() if self.checkpoint_timestamp else None,
            "stop_reason": self.stop_reason,
            "tool_calls_count": self.tool_calls_count,
            "tool_errors_count": self.tool_errors_count,
            "error_details": self.error_details,
            "completion_status": self.completion_status,
        }


class SessionMetrics(Base):
    """
    Session-level metrics model.

    Stores aggregated metrics for a conversation session.
    Enhanced with partial operation tracking and global sequence management.
    """
    __tablename__ = "session_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, unique=True, nullable=False, index=True)
    project_id = Column(String, ForeignKey("project_metrics.project_id", ondelete="CASCADE"), index=True)

    # Global sequence management
    max_global_sequence = Column(Integer, nullable=True)  # Highest global sequence in this session
    agent_transitions = Column(Text, nullable=True)  # JSON data about agent transitions

    # Aggregated token counts
    total_prompt_tokens = Column(Integer, nullable=False, default=0)
    total_completion_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)

    # Aggregated costs
    total_estimated_cost = Column(Float, nullable=False, default=0.0)
    total_actual_cost = Column(Float)

    # Partial operation tracking
    partial_operations_count = Column(Integer, nullable=False, default=0)  # Count of partial operations
    recovered_operations_count = Column(Integer, nullable=False, default=0)  # Count of recovered operations
    is_fully_complete = Column(Boolean, nullable=False, default=True, index=True)  # Whether all operations completed
    last_partial_timestamp = Column(DateTime, nullable=True)  # When last partial operation occurred

    # Error tracking
    total_tool_errors = Column(Integer, nullable=False, default=0)
    total_cancellations = Column(Integer, nullable=False, default=0)  # Number of cancelled operations

    # Session information
    message_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)

    # Session metadata (JSON string)
    models_used = Column(Text)
    agent_metadata = Column(Text, nullable=True)  # JSON data about agents used in session

    # Recovery and debugging information
    recovery_session_ids = Column(Text, nullable=True)  # JSON array of recovery session IDs
    session_health = Column(String, nullable=True, default="healthy")  # Overall health status

    # Relationships
    messages = relationship("MessageMetrics", back_populates="session", cascade="all, delete-orphan")
    project = relationship("ProjectMetrics", back_populates="sessions")

    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "max_global_sequence": self.max_global_sequence,
            "agent_transitions": self.agent_transitions,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_estimated_cost": self.total_estimated_cost,
            "total_actual_cost": self.total_actual_cost,
            "partial_operations_count": self.partial_operations_count,
            "recovered_operations_count": self.recovered_operations_count,
            "is_fully_complete": self.is_fully_complete,
            "last_partial_timestamp": self.last_partial_timestamp.isoformat() if self.last_partial_timestamp else None,
            "total_tool_errors": self.total_tool_errors,
            "total_cancellations": self.total_cancellations,
            "message_count": self.message_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "models_used": self.models_used,
            "agent_metadata": self.agent_metadata,
            "recovery_session_ids": self.recovery_session_ids,
            "session_health": self.session_health,
        }


class ProjectMetrics(Base):
    """
    Project-level metrics model.
    
    Stores aggregated metrics for an entire project.
    """
    __tablename__ = "project_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String, unique=True, nullable=False, index=True)
    
    # Aggregated token counts
    total_prompt_tokens = Column(Integer, nullable=False, default=0)
    total_completion_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    
    # Aggregated costs
    total_estimated_cost = Column(Float, nullable=False, default=0.0)
    total_actual_cost = Column(Float)
    
    # Project information
    session_count = Column(Integer, nullable=False, default=0)
    message_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    
    # Project metadata
    name = Column(String)
    description = Column(Text)
    
    # Relationships
    sessions = relationship("SessionMetrics", back_populates="project", cascade="all, delete-orphan")
    messages = relationship("MessageMetrics", back_populates="project", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_estimated_cost": self.total_estimated_cost,
            "total_actual_cost": self.total_actual_cost,
            "session_count": self.session_count,
            "message_count": self.message_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "name": self.name,
            "description": self.description,
        }
