"""
Streaming Metrics Tracker for partial operations.

This module provides enhanced metrics collection with partial data support
and graceful cancellation handling for streaming operations.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PartialMetrics:
    """Container for partial metrics during streaming."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    model_name: str = ""
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    message_count: int = 0
    is_complete: bool = False


@dataclass
class CheckpointData:
    """Container for checkpoint data."""
    session_id: str
    timestamp: float
    partial_metrics: PartialMetrics
    buffer_data: Dict[str, Any] = field(default_factory=dict)


class StreamingMetricsTracker:
    """
    Enhanced metrics collection with partial data support and graceful cancellation.

    This tracker accumulates metrics during streaming operations and can
    capture partial metrics when operations are cancelled or interrupted.
    """

    def __init__(self):
        """Initialize the StreamingMetricsTracker."""
        self.active_sessions: Dict[str, PartialMetrics] = {}
        self.session_checkpoints: Dict[str, List[CheckpointData]] = {}
        self.cancellation_handlers: Dict[str, callable] = {}
        self.auto_checkpoint_interval = 30  # seconds
        self.max_checkpoints_per_session = 10

    def start_session_tracking(self, session_id: str, model_name: str = "") -> None:
        """
        Start tracking metrics for a new streaming session.

        Args:
            session_id: The session ID to track
            model_name: The model being used for this session
        """
        if session_id in self.active_sessions:
            logger.warning(f"Session {session_id} is already being tracked")

        self.active_sessions[session_id] = PartialMetrics(
            model_name=model_name,
            start_time=time.time()
        )

        # Initialize checkpoint list for this session
        if session_id not in self.session_checkpoints:
            self.session_checkpoints[session_id] = []

        logger.info(f"Started metrics tracking for session {session_id}")

    def accumulate_partial_usage(self, event: Dict[str, Any], session_id: str) -> None:
        """
        Accumulate partial usage information from streaming events.

        Args:
            event: The streaming event (e.g., contentBlockDelta)
            session_id: The session ID
        """
        if session_id not in self.active_sessions:
            logger.warning(f"No active tracking for session {session_id}, initializing")
            self.start_session_tracking(session_id)

        metrics = self.active_sessions[session_id]

        try:
            # Extract usage information from different event types
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]
                # Some APIs include token usage in delta events
                if "usage" in delta:
                    usage = delta["usage"]
                    metrics.input_tokens += usage.get("input_tokens", 0)
                    metrics.output_tokens += usage.get("output_tokens", 0)
                    metrics.total_tokens += usage.get("total_tokens", 0)
                    metrics.cost += usage.get("cost", 0.0)

                # Track content blocks for rough estimation
                if "delta" in delta:
                    delta_content = delta["delta"]
                    if isinstance(delta_content, dict) and "text" in delta_content:
                        # Rough estimation: ~4 characters per token
                        text_length = len(delta_content["text"])
                        estimated_tokens = max(1, text_length // 4)
                        metrics.output_tokens += estimated_tokens
                        metrics.total_tokens += estimated_tokens

            elif "toolUse" in event:
                # Track tool usage
                metrics.message_count += 1

            elif "messageStart" in event:
                # Message started
                start_info = event["messageStart"]
                if "role" in start_info and start_info["role"] == "assistant":
                    metrics.message_count += 1
                    if "model" in start_info:
                        metrics.model_name = start_info["model"]

            elif "contentBlockStart" in event:
                # Content block started
                metrics.message_count += 1

            # Update last activity timestamp
            metrics.last_update = time.time()

        except Exception as e:
            logger.error(f"Error accumulating partial usage for session {session_id}: {e}")

    def finalize_metrics(self, session_id: str, final_event: Optional[Dict[str, Any]] = None) -> PartialMetrics:
        """
        Finalize metrics for a session, typically when streaming completes.

        Args:
            session_id: The session ID to finalize
            final_event: Optional final event with complete usage data

        Returns:
            Complete PartialMetrics object
        """
        if session_id not in self.active_sessions:
            logger.warning(f"No active tracking for session {session_id}")
            return PartialMetrics()

        metrics = self.active_sessions[session_id]

        try:
            # Extract final usage information from messageStop event
            if final_event and "messageStop" in final_event:
                stop_info = final_event["messageStop"]
                if "usage" in stop_info:
                    usage = stop_info["usage"]
                    # Override with final accurate metrics
                    metrics.input_tokens = usage.get("input_tokens", metrics.input_tokens)
                    metrics.output_tokens = usage.get("output_tokens", metrics.output_tokens)
                    metrics.total_tokens = usage.get("total_tokens", metrics.total_tokens)

                    # Calculate cost if not provided
                    if "cost" in usage:
                        metrics.cost = usage["cost"]
                    else:
                        # Rough cost calculation based on tokens
                        metrics.cost = self._calculate_cost(metrics.input_tokens, metrics.output_tokens, metrics.model_name)

            metrics.is_complete = True
            metrics.last_update = time.time()

            logger.info(f"Finalized metrics for session {session_id}: {metrics.total_tokens} tokens, ${metrics.cost:.6f}")

        except Exception as e:
            logger.error(f"Error finalizing metrics for session {session_id}: {e}")

        return metrics

    def capture_partial_metrics_on_cancellation(self, session_id: str) -> Optional[PartialMetrics]:
        """
        Capture partial metrics when a session is cancelled.

        Args:
            session_id: The session ID being cancelled

        Returns:
            PartialMetrics with current state, or None if no active session
        """
        if session_id not in self.active_sessions:
            logger.warning(f"No active tracking for session {session_id} during cancellation")
            return None

        metrics = self.active_sessions[session_id]

        # Mark as incomplete due to cancellation
        metrics.is_complete = False
        metrics.last_update = time.time()

        # Calculate estimated cost based on partial usage
        if metrics.cost == 0.0 and metrics.total_tokens > 0:
            metrics.cost = self._calculate_cost(metrics.input_tokens, metrics.output_tokens, metrics.model_name)

        logger.info(f"Captured partial metrics for cancelled session {session_id}: {metrics.total_tokens} tokens, ${metrics.cost:.6f}")

        # Save as final checkpoint before cleanup
        self._create_checkpoint(session_id, metrics)

        return metrics

    def create_checkpoint(self, session_id: str) -> CheckpointData:
        """
        Create a checkpoint of current metrics state.

        Args:
            session_id: The session ID

        Returns:
            CheckpointData with current state
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"No active tracking for session {session_id}")

        metrics = self.active_sessions[session_id]
        return self._create_checkpoint(session_id, metrics)

    def _create_checkpoint(self, session_id: str, metrics: PartialMetrics) -> CheckpointData:
        """Internal method to create a checkpoint."""
        checkpoint = CheckpointData(
            session_id=session_id,
            timestamp=time.time(),
            partial_metrics=metrics
        )

        checkpoints = self.session_checkpoints.get(session_id, [])
        checkpoints.append(checkpoint)

        # Limit number of checkpoints per session
        if len(checkpoints) > self.max_checkpoints_per_session:
            checkpoints.pop(0)  # Remove oldest checkpoint

        self.session_checkpoints[session_id] = checkpoints

        return checkpoint

    def restore_from_checkpoint(self, session_id: str, checkpoint_index: int = -1) -> Optional[PartialMetrics]:
        """
        Restore metrics from a checkpoint.

        Args:
            session_id: The session ID
            checkpoint_index: Index of checkpoint to restore (-1 for latest)

        Returns:
            Restored PartialMetrics, or None if no checkpoint found
        """
        checkpoints = self.session_checkpoints.get(session_id, [])
        if not checkpoints:
            return None

        try:
            checkpoint = checkpoints[checkpoint_index]
            self.active_sessions[session_id] = checkpoint.partial_metrics

            logger.info(f"Restored metrics from checkpoint for session {session_id}")
            return checkpoint.partial_metrics

        except (IndexError, KeyError) as e:
            logger.error(f"Error restoring from checkpoint for session {session_id}: {e}")
            return None

    def get_latest_metrics(self, session_id: str) -> Optional[PartialMetrics]:
        """
        Get the latest metrics for a session without modifying state.

        Args:
            session_id: The session ID

        Returns:
            Current PartialMetrics, or None if no active session
        """
        return self.active_sessions.get(session_id)

    def end_session_tracking(self, session_id: str) -> Optional[PartialMetrics]:
        """
        End tracking for a session and return final metrics.

        Args:
            session_id: The session ID to end

        Returns:
            Final PartialMetrics, or None if no active session
        """
        metrics = self.active_sessions.pop(session_id, None)

        if metrics:
            # Create final checkpoint
            self._create_checkpoint(session_id, metrics)
            logger.info(f"Ended metrics tracking for session {session_id}")

        return metrics

    def register_cancellation_handler(self, session_id: str, handler: callable) -> None:
        """
        Register a cancellation handler for a session.

        Args:
            session_id: The session ID
            handler: Function to call when cancellation occurs
        """
        self.cancellation_handlers[session_id] = handler

    def handle_cancellation(self, session_id: str) -> Optional[PartialMetrics]:
        """
        Handle session cancellation, capturing partial metrics and calling handlers.

        Args:
            session_id: The session ID being cancelled

        Returns:
            Captured partial metrics
        """
        # Capture partial metrics first
        partial_metrics = self.capture_partial_metrics_on_cancellation(session_id)

        # Call registered cancellation handler
        handler = self.cancellation_handlers.get(session_id)
        if handler:
            try:
                handler(session_id, partial_metrics)
            except Exception as e:
                logger.error(f"Error in cancellation handler for session {session_id}: {e}")

        # Clean up cancellation handler
        self.cancellation_handlers.pop(session_id, None)

        return partial_metrics

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float:
        """
        Calculate estimated cost based on token usage and model.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_name: Model name for pricing

        Returns:
            Estimated cost in USD
        """
        # Basic cost calculation - these rates should be configurable
        # Using approximate OpenRouter pricing as default
        input_cost_per_1k = 0.0005  # $0.0005 per 1k input tokens
        output_cost_per_1k = 0.0015  # $0.0015 per 1k output tokens

        # Adjust pricing based on model (simplified)
        if "claude-3" in model_name.lower():
            input_cost_per_1k = 0.003
            output_cost_per_1k = 0.015
        elif "gpt-4" in model_name.lower():
            input_cost_per_1k = 0.01
            output_cost_per_1k = 0.03
        elif "gpt-3.5" in model_name.lower():
            input_cost_per_1k = 0.0005
            output_cost_per_1k = 0.0015

        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k

        return input_cost + output_cost

    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a session.

        Args:
            session_id: The session ID

        Returns:
            Dictionary with session statistics
        """
        metrics = self.active_sessions.get(session_id)
        checkpoints = self.session_checkpoints.get(session_id, [])

        stats = {
            "session_id": session_id,
            "is_active": session_id in self.active_sessions,
            "is_complete": metrics.is_complete if metrics else False,
            "checkpoints_count": len(checkpoints),
            "has_cancellation_handler": session_id in self.cancellation_handlers
        }

        if metrics:
            stats.update({
                "total_tokens": metrics.total_tokens,
                "input_tokens": metrics.input_tokens,
                "output_tokens": metrics.output_tokens,
                "cost": metrics.cost,
                "model_name": metrics.model_name,
                "message_count": metrics.message_count,
                "start_time": metrics.start_time,
                "last_update": metrics.last_update,
                "duration_seconds": metrics.last_update - metrics.start_time
            })

        return stats

    def cleanup_old_sessions(self, max_age_seconds: int = 3600) -> None:
        """
        Clean up old inactive sessions to prevent memory leaks.

        Args:
            max_age_seconds: Maximum age for inactive sessions
        """
        current_time = time.time()
        sessions_to_remove = []

        for session_id, metrics in self.active_sessions.items():
            age = current_time - metrics.last_update
            if age > max_age_seconds:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            self.end_session_tracking(session_id)
            logger.info(f"Cleaned up old inactive session: {session_id}")

        # Also clean up old checkpoints for inactive sessions
        for session_id in list(self.session_checkpoints.keys()):
            if session_id not in self.active_sessions:
                checkpoints = self.session_checkpoints[session_id]
                # Keep only recent checkpoints
                recent_checkpoints = [
                    cp for cp in checkpoints
                    if current_time - cp.timestamp < max_age_seconds
                ]
                if not recent_checkpoints:
                    # Remove entire checkpoint list if all are old
                    del self.session_checkpoints[session_id]
                else:
                    self.session_checkpoints[session_id] = recent_checkpoints[-5:]  # Keep only last 5

    def get_all_active_sessions(self) -> List[str]:
        """Get list of all currently active session IDs."""
        return list(self.active_sessions.keys())

    def force_save_metrics(self, session_id: str, target_db=None) -> bool:
        """
        Force save current metrics to database, useful for interrupted operations.

        Args:
            session_id: The session ID
            target_db: Database instance to save to

        Returns:
            True if save was successful, False otherwise
        """
        if session_id not in self.active_sessions:
            return False

        metrics = self.active_sessions[session_id]

        try:
            # This would integrate with ProjectDB or other database
            # Implementation would depend on the specific database interface
            if target_db:
                # Convert PartialMetrics to database format
                metrics_data = {
                    "session_id": session_id,
                    "input_tokens": metrics.input_tokens,
                    "output_tokens": metrics.output_tokens,
                    "total_tokens": metrics.total_tokens,
                    "cost": metrics.cost,
                    "model_name": metrics.model_name,
                    "is_complete": metrics.is_complete,
                    "timestamp": metrics.last_update
                }

                # Save to database (implementation would vary)
                # target_db.save_partial_metrics(metrics_data)
                logger.info(f"Force saved partial metrics for session {session_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to force save metrics for session {session_id}: {e}")

        return False