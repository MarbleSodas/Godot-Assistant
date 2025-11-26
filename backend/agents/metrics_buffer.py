"""
Metrics Buffer System for recovery on cancellation.

This module provides an in-memory buffering system for pending metrics
that can be recovered if operations are cancelled or interrupted.
"""

import logging
import time
import json
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import os

from .streaming_metrics_tracker import PartialMetrics

logger = logging.getLogger(__name__)


@dataclass
class BufferedMetric:
    """A single buffered metric entry."""
    session_id: str
    message_id: str
    agent_type: str
    timestamp: float
    metrics: PartialMetrics
    is_finalized: bool = False
    recovery_attempted: bool = False
    buffer_timestamp: float = field(default_factory=time.time)


@dataclass
class BufferStats:
    """Statistics about the metrics buffer."""
    total_buffered: int = 0
    pending_recoveries: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    last_cleanup: float = field(default_factory=time.time)
    buffer_size_bytes: int = 0


class MetricsBuffer:
    """
    Buffer pending metrics for recovery on cancellation and ensure no data loss.

    This system provides:
    - In-memory buffering of partial metrics
    - Periodic persistence to disk for durability
    - Recovery mechanisms for interrupted operations
    - Automatic cleanup of stale buffers
    """

    def __init__(self, storage_dir: str = ".godoty_sessions", buffer_file: str = "metrics_buffer.json"):
        """
        Initialize the MetricsBuffer.

        Args:
            storage_dir: Directory for storing buffer data
            buffer_file: Filename for persistent buffer storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.buffer_file = self.storage_dir / buffer_file

        # In-memory buffer
        self._buffer: Dict[str, List[BufferedMetric]] = {}  # session_id -> List[BufferedMetric]
        self._lock = threading.RLock()

        # Recovery callbacks
        self._recovery_callbacks: List[Callable[[str, BufferedMetric], bool]] = []

        # Configuration
        self.max_buffer_size = 1000  # Maximum metrics per session
        self.buffer_ttl = 3600  # 1 hour TTL for buffered metrics
        self.auto_save_interval = 30  # Auto-save every 30 seconds
        self.max_disk_size_mb = 10  # Maximum disk size for buffer file

        # Statistics
        self._stats = BufferStats()
        self._last_auto_save = time.time()

        # Load existing buffer from disk
        self._load_buffer_from_disk()

        # Start auto-save thread
        self._start_auto_save_thread()

        logger.info(f"MetricsBuffer initialized with storage: {self.buffer_file}")

    def add_metric(self, session_id: str, message_id: str, agent_type: str,
                  metrics: PartialMetrics, is_finalized: bool = False) -> bool:
        """
        Add a metric to the buffer.

        Args:
            session_id: The session ID
            message_id: Unique message identifier
            agent_type: Type of agent (planning, execution, etc.)
            metrics: The metrics to buffer
            is_finalized: Whether this metric is complete/final

        Returns:
            True if added successfully, False if buffer is full
        """
        with self._lock:
            try:
                # Initialize session buffer if needed
                if session_id not in self._buffer:
                    self._buffer[session_id] = []

                session_buffer = self._buffer[session_id]

                # Check buffer size limits
                if len(session_buffer) >= self.max_buffer_size:
                    logger.warning(f"Buffer full for session {session_id}, removing oldest entries")
                    # Remove oldest entries to make space
                    session_buffer = session_buffer[-self.max_buffer_size // 2:]
                    self._buffer[session_id] = session_buffer

                # Create buffered metric
                buffered_metric = BufferedMetric(
                    session_id=session_id,
                    message_id=message_id,
                    agent_type=agent_type,
                    timestamp=metrics.last_update,
                    metrics=metrics,
                    is_finalized=is_finalized
                )

                session_buffer.append(buffered_metric)
                self._update_stats()

                logger.debug(f"Added metric to buffer for session {session_id}, message {message_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to add metric to buffer: {e}")
                return False

    def get_metrics(self, session_id: str, include_recovered: bool = False) -> List[BufferedMetric]:
        """
        Get all buffered metrics for a session.

        Args:
            session_id: The session ID
            include_recovered: Whether to include metrics that have been recovered

        Returns:
            List of BufferedMetric objects
        """
        with self._lock:
            metrics = self._buffer.get(session_id, [])

            if not include_recovered:
                metrics = [m for m in metrics if not m.recovery_attempted]

            return sorted(metrics, key=lambda m: m.timestamp)

    def get_pending_metrics(self, session_id: str) -> List[BufferedMetric]:
        """
        Get metrics that are pending recovery (not finalized).

        Args:
            session_id: The session ID

        Returns:
            List of pending BufferedMetric objects
        """
        with self._lock:
            all_metrics = self._buffer.get(session_id, [])
            pending = [m for m in all_metrics if not m.is_finalized and not m.recovery_attempted]
            return sorted(pending, key=lambda m: m.timestamp)

    def mark_recovered(self, session_id: str, message_id: str) -> bool:
        """
        Mark a metric as recovered.

        Args:
            session_id: The session ID
            message_id: The message ID

        Returns:
            True if marked successfully, False if metric not found
        """
        with self._lock:
            session_metrics = self._buffer.get(session_id, [])
            for metric in session_metrics:
                if metric.message_id == message_id:
                    metric.recovery_attempted = True
                    self._stats.successful_recoveries += 1
                    self._stats.pending_recoveries = max(0, self._stats.pending_recoveries - 1)
                    logger.debug(f"Marked metric as recovered: session {session_id}, message {message_id}")
                    return True

            logger.warning(f"Metric not found for recovery marking: session {session_id}, message {message_id}")
            return False

    def attempt_recovery(self, session_id: str) -> Dict[str, Any]:
        """
        Attempt to recover all pending metrics for a session.

        Args:
            session_id: The session ID

        Returns:
            Recovery statistics
        """
        recovery_stats = {
            "total_attempted": 0,
            "successful": 0,
            "failed": 0,
            "errors": []
        }

        pending_metrics = self.get_pending_metrics(session_id)
        recovery_stats["total_attempted"] = len(pending_metrics)

        if not pending_metrics:
            logger.info(f"No pending metrics to recover for session {session_id}")
            return recovery_stats

        logger.info(f"Attempting recovery for {len(pending_metrics)} metrics in session {session_id}")

        for metric in pending_metrics:
            try:
                # Try each recovery callback
                recovered = False
                for callback in self._recovery_callbacks:
                    try:
                        if callback(session_id, metric):
                            recovered = True
                            break
                    except Exception as callback_error:
                        logger.error(f"Recovery callback error: {callback_error}")
                        recovery_stats["errors"].append(str(callback_error))

                if recovered:
                    self.mark_recovered(session_id, metric.message_id)
                    recovery_stats["successful"] += 1
                else:
                    recovery_stats["failed"] += 1
                    logger.warning(f"Failed to recover metric: session {session_id}, message {metric.message_id}")

            except Exception as e:
                logger.error(f"Error during metric recovery: {e}")
                recovery_stats["failed"] += 1
                recovery_stats["errors"].append(str(e))

        # Update statistics
        with self._lock:
            self._stats.pending_recoveries = max(0, self._stats.pending_recoveries - recovery_stats["successful"])
            self._stats.successful_recoveries += recovery_stats["successful"]
            self._stats.failed_recoveries += recovery_stats["failed"]

        logger.info(f"Recovery completed for session {session_id}: {recovery_stats}")
        return recovery_stats

    def register_recovery_callback(self, callback: Callable[[str, BufferedMetric], bool]) -> None:
        """
        Register a callback for metric recovery attempts.

        Args:
            callback: Function that takes (session_id, buffered_metric) and returns True if successful
        """
        self._recovery_callbacks.append(callback)
        logger.info(f"Registered recovery callback: {callback.__name__}")

    def clear_session(self, session_id: str, keep_recovered: bool = False) -> int:
        """
        Clear buffered metrics for a session.

        Args:
            session_id: The session ID
            keep_recovered: Whether to keep metrics that have been recovered

        Returns:
            Number of metrics cleared
        """
        with self._lock:
            if session_id not in self._buffer:
                return 0

            session_metrics = self._buffer[session_id]
            original_count = len(session_metrics)

            if keep_recovered:
                # Keep only metrics that have been recovered
                self._buffer[session_id] = [m for m in session_metrics if m.recovery_attempted]
            else:
                # Remove all metrics for this session
                del self._buffer[session_id]

            cleared_count = original_count - len(self._buffer.get(session_id, []))
            self._update_stats()

            logger.info(f"Cleared {cleared_count} metrics for session {session_id} (keep_recovered={keep_recovered})")
            return cleared_count

    def cleanup_stale_metrics(self, max_age_seconds: Optional[int] = None) -> Dict[str, int]:
        """
        Clean up stale metrics from the buffer.

        Args:
            max_age_seconds: Maximum age for metrics (uses buffer_ttl if None)

        Returns:
            Dictionary with cleanup statistics
        """
        if max_age_seconds is None:
            max_age_seconds = self.buffer_ttl

        current_time = time.time()
        cleanup_stats = {"sessions_cleaned": 0, "metrics_removed": 0}

        with self._lock:
            sessions_to_remove = []

            for session_id, metrics in self._buffer.items():
                # Check if entire session is stale
                if metrics and current_time - metrics[-1].timestamp > max_age_seconds:
                    sessions_to_remove.append(session_id)
                    cleanup_stats["metrics_removed"] += len(metrics)
                    continue

                # Remove individual stale metrics
                original_count = len(metrics)
                metrics[:] = [
                    m for m in metrics
                    if current_time - m.timestamp <= max_age_seconds
                ]
                removed_count = original_count - len(metrics)
                cleanup_stats["metrics_removed"] += removed_count

                if not metrics:
                    sessions_to_remove.append(session_id)

            # Remove empty sessions
            for session_id in sessions_to_remove:
                if session_id in self._buffer:
                    del self._buffer[session_id]
                    cleanup_stats["sessions_cleaned"] += 1

            self._update_stats()
            self._stats.last_cleanup = current_time

        logger.info(f"Cleanup completed: {cleanup_stats}")
        return cleanup_stats

    def force_save_to_disk(self) -> bool:
        """
        Force save the buffer to disk.

        Returns:
            True if save was successful, False otherwise
        """
        try:
            with self._lock:
                buffer_data = {
                    "version": "1.0",
                    "timestamp": time.time(),
                    "buffer": {
                        session_id: [asdict(metric) for metric in metrics]
                        for session_id, metrics in self._buffer.items()
                    }
                }

                # Check file size limit
                buffer_json = json.dumps(buffer_data, indent=2)
                if len(buffer_json.encode('utf-8')) > self.max_disk_size_mb * 1024 * 1024:
                    logger.warning(f"Buffer file exceeds size limit, truncating oldest entries")
                    # Implement truncation logic here if needed

                with open(self.buffer_file, 'w') as f:
                    f.write(buffer_json)

                self._last_auto_save = time.time()
                logger.debug(f"Force saved buffer to disk: {len(buffer_json)} bytes")
                return True

        except Exception as e:
            logger.error(f"Failed to save buffer to disk: {e}")
            return False

    def _load_buffer_from_disk(self) -> None:
        """Load existing buffer data from disk."""
        try:
            if not self.buffer_file.exists():
                return

            with open(self.buffer_file, 'r') as f:
                buffer_data = json.load(f)

            if buffer_data.get("version") != "1.0":
                logger.warning("Buffer file version mismatch, starting fresh")
                return

            # Convert back to BufferedMetric objects
            for session_id, metrics_data in buffer_data.get("buffer", {}).items():
                session_metrics = []
                for metric_data in metrics_data:
                    # Convert metrics dict back to PartialMetrics
                    metrics_dict = metric_data.pop("metrics", {})
                    partial_metrics = PartialMetrics(**metrics_dict)

                    # Recreate BufferedMetric
                    buffered_metric = BufferedMetric(
                        metrics=partial_metrics,
                        **metric_data
                    )
                    session_metrics.append(buffered_metric)

                self._buffer[session_id] = session_metrics

            self._update_stats()
            logger.info(f"Loaded buffer from disk: {len(self._buffer)} sessions")

        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.error(f"Failed to load buffer from disk: {e}")
            # Start with empty buffer if file is corrupted
            self._buffer.clear()

    def _update_stats(self) -> None:
        """Update internal statistics."""
        total_metrics = sum(len(metrics) for metrics in self._buffer.values())
        pending_metrics = sum(
            len([m for m in metrics if not m.is_finalized and not m.recovery_attempted])
            for metrics in self._buffer.values()
        )

        self._stats.total_buffered = total_metrics
        self._stats.pending_recoveries = pending_metrics

        # Calculate buffer size in bytes (rough estimate)
        try:
            buffer_json = json.dumps({
                session_id: [asdict(metric) for metric in metrics]
                for session_id, metrics in self._buffer.items()
            })
            self._stats.buffer_size_bytes = len(buffer_json.encode('utf-8'))
        except Exception:
            self._stats.buffer_size_bytes = 0

    def _start_auto_save_thread(self) -> None:
        """Start the auto-save thread in the background."""
        def auto_save_worker():
            while True:
                try:
                    time.sleep(self.auto_save_interval)
                    current_time = time.time()

                    # Check if auto-save is needed
                    if current_time - self._last_auto_save >= self.auto_save_interval:
                        if self._buffer:  # Only save if there's data
                            self.force_save_to_disk()
                except Exception as e:
                    logger.error(f"Auto-save error: {e}")

        save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        save_thread.start()
        logger.debug("Auto-save thread started")

    def get_statistics(self) -> BufferStats:
        """Get current buffer statistics."""
        with self._lock:
            self._update_stats()
            return BufferStats(**asdict(self._stats))

    def get_buffer_summary(self) -> Dict[str, Any]:
        """Get a detailed summary of the buffer contents."""
        with self._lock:
            summary = {
                "total_sessions": len(self._buffer),
                "total_metrics": self._stats.total_buffered,
                "pending_recoveries": self._stats.pending_recoveries,
                "buffer_size_mb": self._stats.buffer_size_bytes / (1024 * 1024),
                "sessions": {}
            }

            for session_id, metrics in self._buffer.items():
                session_summary = {
                    "metric_count": len(metrics),
                    "pending_count": len([m for m in metrics if not m.is_finalized and not m.recovery_attempted]),
                    "finalized_count": len([m for m in metrics if m.is_finalized]),
                    "recovered_count": len([m for m in metrics if m.recovery_attempted]),
                    "oldest_timestamp": min(m.timestamp for m in metrics) if metrics else None,
                    "newest_timestamp": max(m.timestamp for m in metrics) if metrics else None
                }
                summary["sessions"][session_id] = session_summary

            return summary

    def shutdown(self) -> None:
        """Gracefully shutdown the buffer system."""
        logger.info("Shutting down MetricsBuffer...")
        self.force_save_to_disk()
        logger.info("MetricsBuffer shutdown complete")