"""
Global Sequence Manager for Multi-Agent Sessions.

This module provides sequence tracking across multi-agent sessions to ensure
proper chronological ordering of messages from planning and execution agents.
"""

import logging
import os
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class GlobalSequenceManager:
    """
    Manages global sequence numbers across multi-agent sessions to ensure
    proper chronological ordering regardless of agent transitions.

    This class addresses the core issue where messages are stored per-agent
    but need to be displayed in proper conversational order across agents.
    """

    def __init__(self, storage_dir: str = ".godoty_sessions"):
        """
        Initialize the GlobalSequenceManager.

        Args:
            storage_dir: Base directory for session storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # In-memory cache for active sessions
        self._sequence_cache: Dict[str, Dict[str, Any]] = {}

        # Lock to prevent sequence conflicts
        self._sequence_locks: Dict[str, bool] = {}

    def _get_session_sequence_file(self, session_id: str) -> Path:
        """Get the path to the sequence metadata file for a session."""
        session_dir = self.storage_dir / f"session_{session_id}"
        return session_dir / "global_sequence.json"

    def _load_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """
        Load sequence metadata for a session from disk.

        Args:
            session_id: The session ID

        Returns:
            Dictionary containing sequence metadata
        """
        sequence_file = self._get_session_sequence_file(session_id)

        if not sequence_file.exists():
            # Initialize new session metadata
            metadata = {
                "session_id": session_id,
                "global_sequence_counter": 0,
                "agent_sequences": {},  # agent_id -> sequence info
                "agent_transitions": [],  # List of agent transition events
                "created_at": time.time(),
                "last_updated": time.time(),
                "version": "1.0"
            }

            # Ensure session directory exists
            sequence_file.parent.mkdir(exist_ok=True)

            # Save initial metadata
            with open(sequence_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            return metadata

        try:
            with open(sequence_file, 'r') as f:
                metadata = json.load(f)

            # Validate metadata structure
            if not all(key in metadata for key in ["session_id", "global_sequence_counter", "agent_sequences"]):
                logger.warning(f"Invalid sequence metadata for session {session_id}, initializing fresh")
                return self._load_session_metadata(session_id)  # Re-initialize

            return metadata

        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load sequence metadata for session {session_id}: {e}")
            # Initialize fresh metadata if file is corrupted
            return self._load_session_metadata(session_id)

    def _save_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> None:
        """
        Save sequence metadata for a session to disk.

        Args:
            session_id: The session ID
            metadata: The sequence metadata to save
        """
        sequence_file = self._get_session_sequence_file(session_id)

        try:
            metadata["last_updated"] = time.time()
            with open(sequence_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        except IOError as e:
            logger.error(f"Failed to save sequence metadata for session {session_id}: {e}")

    def get_next_sequence(self, session_id: str, agent_id: str, agent_type: str = "unknown") -> int:
        """
        Get the next global sequence number for a message.

        Args:
            session_id: The session ID
            agent_id: The agent ID generating the message
            agent_type: The type of agent (planning, execution, etc.)

        Returns:
            Next global sequence number
        """
        # Load or create session metadata
        if session_id not in self._sequence_cache:
            self._sequence_cache[session_id] = self._load_session_metadata(session_id)

        metadata = self._sequence_cache[session_id]

        # Increment global sequence counter
        metadata["global_sequence_counter"] += 1
        next_sequence = metadata["global_sequence_counter"]

        # Update agent sequence info
        if agent_id not in metadata["agent_sequences"]:
            metadata["agent_sequences"][agent_id] = {
                "agent_type": agent_type,
                "first_sequence": next_sequence,
                "last_sequence": next_sequence,
                "message_count": 1,
                "agent_transitions": []
            }
        else:
            agent_info = metadata["agent_sequences"][agent_id]
            agent_info["last_sequence"] = next_sequence
            agent_info["message_count"] += 1

        # Track agent transitions (when a different agent starts/continues)
        if len(metadata["agent_transitions"]) == 0:
            # First message in session
            metadata["agent_transitions"].append({
                "sequence": next_sequence,
                "agent_id": agent_id,
                "agent_type": agent_type,
                "transition_type": "session_start",
                "timestamp": time.time()
            })
        else:
            last_transition = metadata["agent_transitions"][-1]
            if last_transition["agent_id"] != agent_id:
                # Agent transition detected
                metadata["agent_transitions"].append({
                    "sequence": next_sequence,
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "transition_type": "agent_switch",
                    "timestamp": time.time()
                })

        # Save updated metadata
        self._save_session_metadata(session_id, metadata)

        return next_sequence

    def add_message_sequence(self, session_id: str, message_id: str, agent_id: str,
                           global_sequence: int, message_data: Dict[str, Any]) -> None:
        """
        Add sequence information for a specific message.

        Args:
            session_id: The session ID
            message_id: Unique identifier for the message
            agent_id: The agent ID that generated the message
            global_sequence: The global sequence number
            message_data: The message content and metadata
        """
        # Load session metadata
        if session_id not in self._sequence_cache:
            self._sequence_cache[session_id] = self._load_session_metadata(session_id)

        metadata = self._sequence_cache[session_id]

        # Add message to sequence tracking
        if "messages" not in metadata:
            metadata["messages"] = []

        message_entry = {
            "message_id": message_id,
            "agent_id": agent_id,
            "global_sequence": global_sequence,
            "timestamp": message_data.get("timestamp", time.time()),
            "role": message_data.get("role", "assistant"),
            "content_type": message_data.get("type", "message"),
            "stored_at": time.time()
        }

        # Insert in correct order to maintain sequence
        metadata["messages"].append(message_entry)

        # Sort messages by global sequence to maintain order
        metadata["messages"].sort(key=lambda m: m["global_sequence"])

        # Save updated metadata
        self._save_session_metadata(session_id, metadata)

    def get_ordered_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages for a session in proper chronological order.

        Args:
            session_id: The session ID

        Returns:
            List of messages ordered by global sequence number
        """
        # Load session metadata
        if session_id not in self._sequence_cache:
            self._sequence_cache[session_id] = self._load_session_metadata(session_id)

        metadata = self._sequence_cache[session_id]

        # Check if we have sequence information
        if "messages" not in metadata or not metadata["messages"]:
            logger.info(f"No sequence information available for session {session_id}")
            return []

        # Return messages ordered by global sequence
        ordered_messages = sorted(metadata["messages"], key=lambda m: m["global_sequence"])
        return ordered_messages

    def reconstruct_session_order(self, session_id: str, agent_messages: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Reconstruct message order for sessions that don't have global sequence tracking.

        This is used for backward compatibility with existing sessions.

        Args:
            session_id: The session ID
            agent_messages: Dictionary mapping agent_id -> list of messages

        Returns:
            List of messages in reconstructed chronological order
        """
        try:
            # Load session metadata to see if we have transitions
            metadata = self._load_session_metadata(session_id)

            # If we have global sequence info, use it
            if "messages" in metadata and metadata["messages"]:
                return self.get_ordered_messages(session_id)

            # Otherwise, reconstruct from timestamps with agent transition awareness
            all_messages = []
            for agent_id, messages in agent_messages.items():
                for message in messages:
                    # Add sequence reconstruction metadata
                    message["_reconstructed_agent_id"] = agent_id
                    message["_reconstructed_timestamp"] = message.get("timestamp", message.get("created_at", time.time()))
                    all_messages.append(message)

            # Sort by timestamp, then by agent transition order if available
            if metadata.get("agent_transitions"):
                # Use agent transitions to better order messages
                transition_order = {t["agent_id"]: i for i, t in enumerate(metadata["agent_transitions"])}

                def sort_key(msg):
                    timestamp = msg.get("_reconstructed_timestamp", 0)
                    agent_id = msg.get("_reconstructed_agent_id", "")
                    transition_priority = transition_order.get(agent_id, float('inf'))
                    return (timestamp, transition_priority)

                all_messages.sort(key=sort_key)
            else:
                # Fallback to timestamp-only sorting
                all_messages.sort(key=lambda m: m.get("_reconstructed_timestamp", 0))

            # Clean up reconstruction metadata
            for message in all_messages:
                message.pop("_reconstructed_agent_id", None)
                message.pop("_reconstructed_timestamp", None)

            return all_messages

        except Exception as e:
            logger.error(f"Failed to reconstruct session order for {session_id}: {e}")
            # Final fallback - return all messages sorted by timestamp
            all_messages = []
            for messages in agent_messages.values():
                all_messages.extend(messages)
            all_messages.sort(key=lambda m: m.get("timestamp", m.get("created_at", 0)))
            return all_messages

    def get_agent_transitions(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get the agent transition timeline for a session.

        Args:
            session_id: The session ID

        Returns:
            List of agent transition events
        """
        metadata = self._load_session_metadata(session_id)
        return metadata.get("agent_transitions", [])

    def clear_cache(self, session_id: Optional[str] = None) -> None:
        """
        Clear the in-memory cache for one or all sessions.

        Args:
            session_id: Specific session to clear, or None to clear all
        """
        if session_id:
            self._sequence_cache.pop(session_id, None)
        else:
            self._sequence_cache.clear()

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics about sequence usage for a session.

        Args:
            session_id: The session ID

        Returns:
            Dictionary with session statistics
        """
        metadata = self._load_session_metadata(session_id)

        stats = {
            "session_id": session_id,
            "total_messages": len(metadata.get("messages", [])),
            "global_sequence_counter": metadata.get("global_sequence_counter", 0),
            "agent_count": len(metadata.get("agent_sequences", {})),
            "agent_transitions": len(metadata.get("agent_transitions", [])),
            "created_at": metadata.get("created_at"),
            "last_updated": metadata.get("last_updated"),
            "version": metadata.get("version", "1.0")
        }

        # Add per-agent stats
        agent_stats = {}
        for agent_id, agent_info in metadata.get("agent_sequences", {}).items():
            agent_stats[agent_id] = {
                "agent_type": agent_info.get("agent_type", "unknown"),
                "message_count": agent_info.get("message_count", 0),
                "first_sequence": agent_info.get("first_sequence", 0),
                "last_sequence": agent_info.get("last_sequence", 0)
            }

        stats["agents"] = agent_stats
        return stats

    def validate_session_integrity(self, session_id: str) -> Dict[str, Any]:
        """
        Validate the integrity of sequence tracking for a session.

        Args:
            session_id: The session ID

        Returns:
            Dictionary with validation results
        """
        try:
            metadata = self._load_session_metadata(session_id)
            messages = metadata.get("messages", [])

            validation_result = {
                "session_id": session_id,
                "is_valid": True,
                "issues": [],
                "warnings": []
            }

            # Check for sequence gaps
            if messages:
                sequences = [m["global_sequence"] for m in messages]
                expected_sequences = list(range(1, max(sequences) + 1))
                missing_sequences = set(expected_sequences) - set(sequences)

                if missing_sequences:
                    validation_result["issues"].append(f"Missing sequence numbers: {sorted(missing_sequences)}")
                    validation_result["is_valid"] = False

                # Check for duplicates
                if len(sequences) != len(set(sequences)):
                    duplicates = [seq for seq in set(sequences) if sequences.count(seq) > 1]
                    validation_result["issues"].append(f"Duplicate sequence numbers: {duplicates}")
                    validation_result["is_valid"] = False

            # Check agent transition consistency
            transitions = metadata.get("agent_transitions", [])
            if transitions and len(transitions) > 1:
                # First transition should be session_start
                if transitions[0].get("transition_type") != "session_start":
                    validation_result["warnings"].append("First transition is not session_start")

            return validation_result

        except Exception as e:
            return {
                "session_id": session_id,
                "is_valid": False,
                "issues": [f"Validation failed with error: {e}"],
                "warnings": []
            }