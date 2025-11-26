"""
Database manager for metrics tracking.

Provides database initialization, CRUD operations, and aggregation queries
for metrics data.
"""

import os
import logging
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, select, func, and_
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from .models import Base, MessageMetrics, SessionMetrics, ProjectMetrics
from .retry_utils import critical_retry_operation, default_retry_operation, retry_metrics

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections and operations for metrics tracking.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file. Defaults to .godoty_metrics.db
        """
        if db_path is None:
            db_path = os.path.join(os.getcwd(), ".godoty_metrics.db")

        self.db_path = db_path
        self.db_url = f"sqlite+aiosqlite:///{db_path}"

        # Create async engine
        self.engine = create_async_engine(
            self.db_url,
            echo=False,
            connect_args={"check_same_thread": False}
        )

        # Create session maker
        self.async_session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        logger.info(f"DatabaseManager initialized with db_path: {db_path}")

    async def initialize(self):
        """Initialize database tables."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def close(self):
        """Close database connections."""
        await self.engine.dispose()
        logger.info("Database connections closed")

    # Message Metrics CRUD Operations

    @critical_retry_operation
    async def create_message_metrics(
        self,
        message_id: str,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated_cost: float,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        generation_id: Optional[str] = None,
        response_time_ms: Optional[int] = None,
        stop_reason: Optional[str] = None,
        tool_calls_count: int = 0,
        tool_errors_count: int = 0,
        actual_cost: Optional[float] = None,
        # New fields for enhanced tracking
        global_sequence: Optional[int] = None,
        agent_sequence: Optional[int] = None,
        agent_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        workflow_session_id: Optional[str] = None,
        is_complete: bool = True,
        is_partial: bool = False,
        recovery_session_id: Optional[str] = None,
        checkpoint_timestamp: Optional[datetime] = None,
        error_details: Optional[str] = None,
        completion_status: Optional[str] = None
    ) -> MessageMetrics:
        """
        Create a new message metrics record.

        Args:
            message_id: Unique message identifier
            model_id: Model used for the message
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens (prompt + completion)
            estimated_cost: Estimated cost in USD
            session_id: Optional session ID
            project_id: Optional project ID
            generation_id: Optional OpenRouter generation ID
            response_time_ms: Optional response time in milliseconds
            stop_reason: Optional stop reason
            tool_calls_count: Number of tool calls made
            tool_errors_count: Number of tool errors
            actual_cost: Optional actual cost from generation endpoint

        Returns:
            Created MessageMetrics instance
        """
        async with self.async_session_maker() as session:
            try:
                metrics = MessageMetrics(
                    message_id=message_id,
                    session_id=session_id,
                    project_id=project_id,
                    model_id=model_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost=estimated_cost,
                    actual_cost=actual_cost,
                    generation_id=generation_id,
                    response_time_ms=response_time_ms,
                    stop_reason=stop_reason,
                    tool_calls_count=tool_calls_count,
                    tool_errors_count=tool_errors_count,
                    # Enhanced tracking fields
                    global_sequence=global_sequence,
                    agent_sequence=agent_sequence,
                    agent_type=agent_type,
                    agent_id=agent_id,
                    workflow_session_id=workflow_session_id,
                    is_complete=is_complete,
                    is_partial=is_partial,
                    recovery_session_id=recovery_session_id,
                    checkpoint_timestamp=checkpoint_timestamp,
                    error_details=error_details,
                    completion_status=completion_status
                )

                session.add(metrics)
                await session.commit()
                await session.refresh(metrics)

                logger.info(f"Created message metrics: {message_id} (global_sequence={global_sequence}, is_partial={is_partial})")
                return metrics
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to create message metrics: {e}")
                raise

    async def get_message_metrics(self, message_id: str) -> Optional[MessageMetrics]:
        """Get message metrics by ID."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(MessageMetrics).where(MessageMetrics.message_id == message_id)
            )
            return result.scalar_one_or_none()

    async def update_message_actual_cost(self, message_id: str, actual_cost: float):
        """Update actual cost for a message after querying generation endpoint."""
        async with self.async_session_maker() as session:
            try:
                result = await session.execute(
                    select(MessageMetrics).where(MessageMetrics.message_id == message_id)
                )
                metrics = result.scalar_one_or_none()

                if metrics:
                    metrics.actual_cost = actual_cost
                    await session.commit()
                    logger.info(f"Updated actual cost for message {message_id}: ${actual_cost}")
                else:
                    logger.warning(f"Message metrics not found: {message_id}")
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to update actual cost: {e}")
                raise

    # Session Metrics Operations

    async def get_or_create_session_metrics(
        self,
        session_id: str,
        project_id: Optional[str] = None
    ) -> SessionMetrics:
        """Get existing session metrics or create new one."""
        async with self.async_session_maker() as session:
            try:
                result = await session.execute(
                    select(SessionMetrics).where(SessionMetrics.session_id == session_id)
                )
                metrics = result.scalar_one_or_none()

                if metrics:
                    return metrics

                # Create new session metrics
                metrics = SessionMetrics(
                    session_id=session_id,
                    project_id=project_id
                )
                session.add(metrics)
                await session.commit()
                await session.refresh(metrics)

                logger.info(f"Created session metrics: {session_id}")
                return metrics
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to get/create session metrics: {e}")
                raise

    @critical_retry_operation
    async def update_session_metrics(
        self,
        session_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated_cost: float,
        model_id: str,
        actual_cost: Optional[float] = None,
        tool_errors_count: int = 0
    ):
        """Update session metrics by adding message metrics."""
        async with self.async_session_maker() as session:
            try:
                result = await session.execute(
                    select(SessionMetrics).where(SessionMetrics.session_id == session_id)
                )
                metrics = result.scalar_one_or_none()

                if not metrics:
                    logger.warning(f"Session metrics not found: {session_id}")
                    return

                # Update aggregated values
                metrics.total_prompt_tokens += prompt_tokens
                metrics.total_completion_tokens += completion_tokens
                metrics.total_tokens += total_tokens
                metrics.total_estimated_cost += estimated_cost
                if actual_cost:
                    metrics.total_actual_cost = (metrics.total_actual_cost or 0) + actual_cost
                metrics.total_tool_errors += tool_errors_count
                metrics.message_count += 1
                metrics.updated_at = datetime.utcnow()

                # Update models used
                models_used = json.loads(metrics.models_used) if metrics.models_used else []
                if model_id not in models_used:
                    models_used.append(model_id)
                metrics.models_used = json.dumps(models_used)

                await session.commit()
                logger.info(f"Updated session metrics: {session_id}")
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to update session metrics: {e}")
                raise

    async def get_session_metrics(self, session_id: str) -> Optional[SessionMetrics]:
        """Get session metrics by ID."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(SessionMetrics).where(SessionMetrics.session_id == session_id)
            )
            return result.scalar_one_or_none()

    async def get_session_messages(self, session_id: str) -> List[MessageMetrics]:
        """Get all message metrics for a session."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(MessageMetrics)
                .where(MessageMetrics.session_id == session_id)
                .order_by(MessageMetrics.created_at)
            )
            return list(result.scalars().all())

    # Project Metrics Operations

    async def get_or_create_project_metrics(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> ProjectMetrics:
        """Get existing project metrics or create new one."""
        async with self.async_session_maker() as session:
            try:
                result = await session.execute(
                    select(ProjectMetrics).where(ProjectMetrics.project_id == project_id)
                )
                metrics = result.scalar_one_or_none()

                if metrics:
                    return metrics

                # Create new project metrics
                metrics = ProjectMetrics(
                    project_id=project_id,
                    name=name,
                    description=description
                )
                session.add(metrics)
                await session.commit()
                await session.refresh(metrics)

                logger.info(f"Created project metrics: {project_id}")
                return metrics
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to get/create project metrics: {e}")
                raise

    async def update_project_metrics(
        self,
        project_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated_cost: float,
        actual_cost: Optional[float] = None,
        is_new_session: bool = False,
        is_new_message: bool = True
    ):
        """Update project metrics."""
        async with self.async_session_maker() as session:
            try:
                result = await session.execute(
                    select(ProjectMetrics).where(ProjectMetrics.project_id == project_id)
                )
                metrics = result.scalar_one_or_none()

                if not metrics:
                    logger.warning(f"Project metrics not found: {project_id}")
                    return

                # Update aggregated values
                metrics.total_prompt_tokens += prompt_tokens
                metrics.total_completion_tokens += completion_tokens
                metrics.total_tokens += total_tokens
                metrics.total_estimated_cost += estimated_cost
                if actual_cost:
                    metrics.total_actual_cost = (metrics.total_actual_cost or 0) + actual_cost

                if is_new_session:
                    metrics.session_count += 1
                if is_new_message:
                    metrics.message_count += 1

                metrics.updated_at = datetime.utcnow()

                await session.commit()
                logger.info(f"Updated project metrics: {project_id}")
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to update project metrics: {e}")
                raise

    async def get_project_metrics(self, project_id: str) -> Optional[ProjectMetrics]:
        """Get project metrics by ID."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(ProjectMetrics).where(ProjectMetrics.project_id == project_id)
            )
            return result.scalar_one_or_none()

    async def get_project_sessions(self, project_id: str) -> List[SessionMetrics]:
        """Get all session metrics for a project."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(SessionMetrics)
                .where(SessionMetrics.project_id == project_id)
                .order_by(SessionMetrics.created_at)
            )
            return list(result.scalars().all())

    async def list_all_projects(self) -> List[ProjectMetrics]:
        """List all projects."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(ProjectMetrics).order_by(ProjectMetrics.created_at.desc())
            )
            return list(result.scalars().all())

    # Enhanced operations for partial metrics and global sequences

    async def get_session_messages_with_sequence(
        self,
        session_id: str,
        include_partial: bool = False,
        order_by_global_sequence: bool = True
    ) -> List[MessageMetrics]:
        """
        Get all messages for a session with sequence ordering.

        Args:
            session_id: The session ID
            include_partial: Whether to include partial metrics
            order_by_global_sequence: Whether to order by global sequence

        Returns:
            List of MessageMetrics ordered by global sequence
        """
        async with self.async_session_maker() as session:
            query = select(MessageMetrics).where(MessageMetrics.session_id == session_id)

            if not include_partial:
                query = query.where(MessageMetrics.is_partial == False)

            if order_by_global_sequence:
                query = query.order_by(MessageMetrics.global_sequence.asc(), MessageMetrics.created_at.asc())
            else:
                query = query.order_by(MessageMetrics.created_at.asc())

            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_max_global_sequence(self, session_id: str) -> Optional[int]:
        """
        Get the maximum global sequence number for a session.

        Args:
            session_id: The session ID

        Returns:
            Maximum global sequence number, or None if no messages
        """
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(func.max(MessageMetrics.global_sequence))
                .where(MessageMetrics.session_id == session_id)
            )
            return result.scalar()

    async def update_session_global_sequence(self, session_id: str) -> None:
        """
        Update the max global sequence in SessionMetrics.

        Args:
            session_id: The session ID
        """
        max_seq = await self.get_max_global_sequence(session_id)
        if max_seq is not None:
            async with self.async_session_maker() as session:
                try:
                    result = await session.execute(
                        select(SessionMetrics).where(SessionMetrics.session_id == session_id)
                    )
                    session_metrics = result.scalar_one_or_none()

                    if session_metrics and session_metrics.max_global_sequence != max_seq:
                        session_metrics.max_global_sequence = max_seq
                        session_metrics.updated_at = datetime.utcnow()
                        await session.commit()
                        logger.debug(f"Updated max global sequence for session {session_id}: {max_seq}")
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Failed to update session global sequence: {e}")

    @critical_retry_operation
    async def create_partial_message_metrics(
        self,
        message_id: str,
        model_id: str,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        partial_tokens: int = 0,
        partial_cost: float = 0.0,
        agent_type: Optional[str] = None,
        global_sequence: Optional[int] = None,
        error_details: Optional[str] = None,
        completion_status: Optional[str] = "partial"
    ) -> MessageMetrics:
        """
        Create partial message metrics for cancelled or interrupted operations.

        Args:
            message_id: Unique message identifier
            model_id: Model used for the message
            session_id: Optional session ID
            project_id: Optional project ID
            partial_tokens: Partial token count
            partial_cost: Partial cost estimate
            agent_type: Optional agent type
            global_sequence: Optional global sequence number
            error_details: Error information if operation failed
            completion_status: Status of completion

        Returns:
            Created MessageMetrics instance with partial data
        """
        return await self.create_message_metrics(
            message_id=message_id,
            model_id=model_id,
            prompt_tokens=0,  # Will be updated if available
            completion_tokens=partial_tokens,
            total_tokens=partial_tokens,
            estimated_cost=partial_cost,
            session_id=session_id,
            project_id=project_id,
            global_sequence=global_sequence,
            agent_type=agent_type,
            is_complete=False,
            is_partial=True,
            error_details=error_details,
            completion_status=completion_status
        )

    async def get_partial_metrics_for_session(
        self,
        session_id: str
    ) -> List[MessageMetrics]:
        """
        Get all partial metrics for a session.

        Args:
            session_id: The session ID

        Returns:
            List of partial MessageMetrics
        """
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(MessageMetrics)
                .where(
                    and_(
                        MessageMetrics.session_id == session_id,
                        MessageMetrics.is_partial == True
                    )
                )
                .order_by(MessageMetrics.created_at.desc())
            )
            return list(result.scalars().all())

    async def mark_metrics_as_recovered(
        self,
        message_id: str,
        recovery_session_id: Optional[str] = None
    ) -> bool:
        """
        Mark partial metrics as recovered.

        Args:
            message_id: The message ID to mark as recovered
            recovery_session_id: Optional recovery session ID

        Returns:
            True if successful, False otherwise
        """
        async with self.async_session_maker() as session:
            try:
                result = await session.execute(
                    select(MessageMetrics).where(MessageMetrics.message_id == message_id)
                )
                metrics = result.scalar_one_or_none()

                if metrics:
                    metrics.is_complete = False  # Still not complete, but recovered
                    metrics.recovery_session_id = recovery_session_id
                    metrics.completion_status = "recovered"
                    metrics.updated_at = datetime.utcnow()

                    await session.commit()
                    logger.info(f"Marked metrics as recovered: {message_id}")
                    return True
                return False
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to mark metrics as recovered: {e}")
                return False

    async def cleanup_partial_metrics(self, session_id: str, older_than_hours: int = 24) -> int:
        """
        Clean up old partial metrics for a session.

        Args:
            session_id: The session ID
            older_than_hours: Clean up partial metrics older than this many hours

        Returns:
            Number of cleaned up metrics
        """
        cutoff_time = datetime.utcnow().replace(microsecond=0) - timedelta(hours=older_than_hours)

        async with self.async_session_maker() as session:
            try:
                result = await session.execute(
                    select(MessageMetrics)
                    .where(
                        and_(
                            MessageMetrics.session_id == session_id,
                            MessageMetrics.is_partial == True,
                            MessageMetrics.created_at < cutoff_time
                        )
                    )
                )
                partial_metrics = list(result.scalars().all())

                # Delete the old partial metrics
                for metrics in partial_metrics:
                    await session.delete(metrics)

                await session.commit()
                cleaned_count = len(partial_metrics)
                logger.info(f"Cleaned up {cleaned_count} old partial metrics for session {session_id}")
                return cleaned_count
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to cleanup partial metrics: {e}")
                return 0

    async def get_session_health_report(self, session_id: str) -> Dict[str, Any]:
        """
        Generate a health report for a session.

        Args:
            session_id: The session ID

        Returns:
            Health report with statistics and issues
        """
        try:
            # Get session metrics
            async with self.async_session_maker() as session:
                result = await session.execute(
                    select(SessionMetrics).where(SessionMetrics.session_id == session_id)
                )
                session_metrics = result.scalar_one_or_none()

            # Get message counts
            total_messages = await self.get_session_messages_with_sequence(session_id, include_partial=True)
            partial_messages = await self.get_partial_metrics_for_session(session_id)
            completed_messages = [m for m in total_messages if not m.is_partial]

            # Calculate statistics
            health_report = {
                "session_id": session_id,
                "total_messages": len(total_messages),
                "completed_messages": len(completed_messages),
                "partial_messages": len(partial_messages),
                "completion_rate": len(completed_messages) / len(total_messages) if total_messages else 0,
                "has_sequence_numbers": any(m.global_sequence is not None for m in total_messages),
                "max_global_sequence": await self.get_max_global_sequence(session_id),
                "issues": [],
                "recommendations": []
            }

            # Identify issues
            if len(partial_messages) > len(completed_messages):
                health_report["issues"].append("High number of partial operations")
                health_report["recommendations"].append("Consider recovering partial metrics")

            if health_report["completion_rate"] < 0.8:
                health_report["issues"].append("Low completion rate")
                health_report["recommendations"].append("Investigate cancellation issues")

            if not health_report["has_sequence_numbers"]:
                health_report["issues"].append("No sequence tracking")
                health_report["recommendations"].append("Enable global sequence tracking")

            if session_metrics:
                health_report["session_health"] = session_metrics.session_health
                if session_metrics.total_cancellations > 5:
                    health_report["issues"].append("High cancellation count")

            return health_report

        except Exception as e:
            logger.error(f"Failed to generate health report for session {session_id}: {e}")
            return {
                "session_id": session_id,
                "error": str(e),
                "healthy": False
            }

    @default_retry_operation
    def increment_session_cancellations(self, session_id: str) -> bool:
        """
        Increment the cancellation count for a session.

        Args:
            session_id: The session ID

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_session() as session:
                session_metrics = session.query(SessionMetrics).filter_by(session_id=session_id).first()
                if session_metrics:
                    session_metrics.total_cancellations += 1
                    session.commit()
                    logger.debug(f"Incremented cancellation count for session {session_id}")
                    return True
                else:
                    logger.warning(f"Session metrics not found for cancellation increment: {session_id}")
                    return False

        except Exception as e:
            logger.error(f"Failed to increment session cancellations for {session_id}: {e}")
            return False

    @default_retry_operation
    def increment_recovered_operations(self, session_id: str) -> bool:
        """
        Increment the recovered operations count for a session.

        Args:
            session_id: The session ID

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_session() as session:
                session_metrics = session.query(SessionMetrics).filter_by(session_id=session_id).first()
                if session_metrics:
                    session_metrics.recovered_operations_count += 1
                    session.commit()
                    logger.debug(f"Incremented recovered operations count for session {session_id}")
                    return True
                else:
                    logger.warning(f"Session metrics not found for recovery increment: {session_id}")
                    return False

        except Exception as e:
            logger.error(f"Failed to increment recovered operations for {session_id}: {e}")
            return False

    @default_retry_operation
    def update_session_health(self, session_id: str, health_status: str) -> bool:
        """
        Update the health status for a session.

        Args:
            session_id: The session ID
            health_status: The health status to set

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_session() as session:
                session_metrics = session.query(SessionMetrics).filter_by(session_id=session_id).first()
                if session_metrics:
                    session_metrics.session_health = health_status
                    session.commit()
                    logger.debug(f"Updated health status for session {session_id}: {health_status}")
                    return True
                else:
                    logger.warning(f"Session metrics not found for health update: {session_id}")
                    return False

        except Exception as e:
            logger.error(f"Failed to update session health for {session_id}: {e}")
            return False

    def get_partial_metrics_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all partial metrics for a session.

        Args:
            session_id: The session ID

        Returns:
            List of partial metrics dictionaries
        """
        try:
            with self.get_session() as session:
                partial_metrics = session.query(MessageMetrics).filter(
                    MessageMetrics.session_id == session_id,
                    MessageMetrics.is_complete == False,
                    MessageMetrics.is_partial == True
                ).all()

                return [metric.to_dict() for metric in partial_metrics]

        except Exception as e:
            logger.error(f"Failed to get partial metrics for session {session_id}: {e}")
            return []

    def mark_metrics_as_recovered(self, session_id: str, message_id: str) -> bool:
        """
        Mark metrics as recovered.

        Args:
            session_id: The session ID
            message_id: The message ID to mark as recovered

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_session() as session:
                message_metric = session.query(MessageMetrics).filter(
                    MessageMetrics.session_id == session_id,
                    MessageMetrics.message_id == message_id
                ).first()

                if message_metric:
                    message_metric.completion_status = "recovered"
                    session.commit()
                    logger.debug(f"Marked metrics as recovered: session {session_id}, message {message_id}")
                    return True
                else:
                    logger.warning(f"Message metric not found for recovery marking: session {session_id}, message {message_id}")
                    return False

        except Exception as e:
            logger.error(f"Failed to mark metrics as recovered: session {session_id}, message {message_id}: {e}")
            return False

    def cleanup_partial_metrics(self, max_age_hours: int = 24) -> Dict[str, int]:
        """
        Clean up old partial metrics.

        Args:
            max_age_hours: Maximum age in hours for partial metrics

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            from datetime import timedelta
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

            with self.get_session() as session:
                # Delete old partial metrics
                deleted = session.query(MessageMetrics).filter(
                    MessageMetrics.is_partial == True,
                    MessageMetrics.is_complete == False,
                    MessageMetrics.created_at < cutoff_time
                ).delete()

                session.commit()
                logger.info(f"Cleaned up {deleted} old partial metrics")

                return {
                    "deleted_metrics": deleted,
                    "max_age_hours": max_age_hours
                }

        except Exception as e:
            logger.error(f"Failed to cleanup partial metrics: {e}")
            return {"deleted_metrics": 0, "error": str(e)}


# Global instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
