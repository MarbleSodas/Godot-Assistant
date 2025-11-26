"""
Database Retry Utility

Provides retry logic with exponential backoff for database operations
to improve reliability and handle transient failures.
"""

import logging
import time
import functools
import random
from typing import Callable, Any, Optional, Type, Tuple

logger = logging.getLogger(__name__)


class DatabaseRetryError(Exception):
    """Custom exception for database retry failures."""
    pass


def retry_database_operation(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: Optional[Tuple[Type[Exception], ...]] = None
):
    """
    Decorator for retrying database operations with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to prevent thundering herd
        retry_on: Tuple of exception types to retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    if retry_on and not isinstance(e, retry_on):
                        # Not a retryable exception, re-raise immediately
                        raise

                    # Don't retry on the last attempt
                    if attempt == max_retries:
                        logger.error(f"Database operation {func.__name__} failed after {max_retries} retries: {e}")
                        raise DatabaseRetryError(f"Operation failed after {max_retries} retries") from e

                    # Calculate delay with exponential backoff
                    delay = min(initial_delay * (exponential_base ** attempt), max_delay)

                    # Add jitter if enabled
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)  # 0.5x to 1.0x the delay

                    logger.warning(
                        f"Database operation {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )

                    time.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise DatabaseRetryError("Unexpected error in retry logic") from last_exception

        return wrapper
    return decorator


# Default retry configuration for common database operations
DEFAULT_RETRY_CONFIG = {
    "max_retries": 3,
    "initial_delay": 0.5,
    "max_delay": 30.0,
    "exponential_base": 2.0,
    "jitter": True,
    "retry_on": (
        # Common transient database errors
        ConnectionError,
        TimeoutError,
        # SQLAlchemy specific errors would go here if imported
    )
}

# Aggressive retry configuration for critical operations
CRITICAL_RETRY_CONFIG = {
    "max_retries": 5,
    "initial_delay": 1.0,
    "max_delay": 60.0,
    "exponential_base": 2.0,
    "jitter": True,
    "retry_on": (
        ConnectionError,
        TimeoutError,
    )
}


# Convenience decorators
@retry_database_operation(**DEFAULT_RETRY_CONFIG)
def default_retry_operation(func: Callable) -> Callable:
    """Default retry decorator for most database operations."""
    return func


@retry_database_operation(**CRITICAL_RETRY_CONFIG)
def critical_retry_operation(func: Callable) -> Callable:
    """Aggressive retry decorator for critical database operations."""
    return func


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable.

    Args:
        error: The exception to check

    Returns:
        True if the error is likely transient and retryable
    """
    # Connection errors
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True

    # Check for specific error messages
    error_str = str(error).lower()
    retryable_patterns = [
        "connection",
        "timeout",
        "deadline",
        "service unavailable",
        "temporarily unavailable",
        "database is locked",
        "connection reset",
        "network",
        "dns",
    ]

    return any(pattern in error_str for pattern in retryable_patterns)


class RetryMetrics:
    """Track retry metrics for monitoring and debugging."""

    def __init__(self):
        self.attempts = {}
        self.failures = {}
        self.successes = {}

    def record_attempt(self, operation_name: str):
        """Record an attempt for an operation."""
        self.attempts[operation_name] = self.attempts.get(operation_name, 0) + 1

    def record_failure(self, operation_name: str, error: Exception):
        """Record a failure for an operation."""
        if operation_name not in self.failures:
            self.failures[operation_name] = []
        self.failures[operation_name].append(str(error))

    def record_success(self, operation_name: str):
        """Record a success for an operation."""
        self.successes[operation_name] = self.successes.get(operation_name, 0) + 1

    def get_summary(self) -> dict:
        """Get a summary of retry metrics."""
        return {
            "operations": list(self.attempts.keys()),
            "total_attempts": sum(self.attempts.values()),
            "total_failures": sum(len(failures) for failures in self.failures.values()),
            "total_successes": sum(self.successes.values()),
            "operation_details": {
                op: {
                    "attempts": self.attempts.get(op, 0),
                    "failures": len(self.failures.get(op, [])),
                    "successes": self.successes.get(op, 0),
                    "success_rate": (
                        self.successes.get(op, 0) / max(1, self.attempts.get(op, 0))
                    )
                }
                for op in self.attempts.keys()
            }
        }


# Global metrics instance
retry_metrics = RetryMetrics()