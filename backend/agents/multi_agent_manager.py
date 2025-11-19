"""
Multi-Agent Manager for Godot Assistant.

This module manages multi-agent sessions and orchestration using Strands Agents.
It handles:
- Session creation and persistence
- Multi-agent graph execution
- Message processing
"""

import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

from strands import Agent
from strands.multiagent.graph import Graph, GraphBuilder
from strands.session.file_session_manager import FileSessionManager

from .planning_agent import get_planning_agent
from .executor_agent import get_executor_agent
from .config import AgentConfig

logger = logging.getLogger(__name__)


class MultiAgentManager:
    """
    Manages multi-agent sessions and execution.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the multi-agent manager.

        Args:
            storage_dir: Directory to store session files. Defaults to .godoty_sessions
        """
        self.storage_dir = storage_dir or os.path.join(os.getcwd(), ".godoty_sessions")
        os.makedirs(self.storage_dir, exist_ok=True)
        self._active_graphs: Dict[str, Graph] = {}
        logger.info(f"MultiAgentManager initialized with storage: {self.storage_dir}")

    def create_session(self, session_id: str) -> str:
        """
        Create a new multi-agent session.

        Args:
            session_id: Unique session identifier

        Returns:
            Session ID
        """
        if session_id in self._active_graphs:
            logger.info(f"Session {session_id} already active")
            return session_id

        try:
            # Get agents
            planning_agent = get_planning_agent()
            
            # Create session manager
            session_manager = FileSessionManager(
                session_id=session_id,
                storage_dir=self.storage_dir
            )

            # Create graph using GraphBuilder
            builder = GraphBuilder()
            builder.add_node(planning_agent.agent, "planner")
            builder.set_entry_point("planner")
            builder.set_session_manager(session_manager)
            
            graph = builder.build()
            
            self._active_graphs[session_id] = graph
            logger.info(f"Created session {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            raise

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session details.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session details or None if not found
        """
        # Check if session file exists even if not in memory
        session_path = os.path.join(self.storage_dir, f"{session_id}.json")
        if os.path.exists(session_path):
            return {
                "session_id": session_id,
                "path": session_path,
                "active": session_id in self._active_graphs
            }
        return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available sessions.

        Returns:
            List of session details
        """
        sessions = []
        if os.path.exists(self.storage_dir):
            for filename in os.listdir(self.storage_dir):
                if filename.endswith(".json"):
                    session_id = filename[:-5]
                    sessions.append(self.get_session(session_id))
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session ID

        Returns:
            True if deleted, False otherwise
        """
        # Remove from active graphs
        if session_id in self._active_graphs:
            del self._active_graphs[session_id]

        # Delete file
        session_path = os.path.join(self.storage_dir, f"{session_id}.json")
        if os.path.exists(session_path):
            try:
                os.remove(session_path)
                logger.info(f"Deleted session {session_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete session file {session_path}: {e}")
                return False
        return False

    async def process_message(self, session_id: str, message: str) -> Any:
        """
        Process a message in a session.

        Args:
            session_id: Session ID
            message: User message

        Returns:
            Agent response
        """
        # Ensure session exists/is loaded
        if session_id not in self._active_graphs:
            if self.get_session(session_id):
                self.create_session(session_id)  # Re-load
            else:
                raise ValueError(f"Session {session_id} not found")

        graph = self._active_graphs[session_id]
        
        # Execute graph
        # Note: Graph execution might be synchronous or async depending on Strands implementation.
        # The docs say `result = graph("...")`.
        # If it's async, we await it. If not, we run it.
        # Based on PlanningAgent, `agent(prompt)` is synchronous but `agent.plan_async` uses model directly.
        # Strands Graph likely supports `__call__`.
        
        try:
            # We need to verify if Graph supports async or if we need to run it in a thread.
            # For now, assuming synchronous call wrapped in async for API consistency,
            # or if Strands supports async, we'd use that.
            # Given `PlanningAgent.plan` calls `self.agent(prompt)`, we'll assume sync for now.
            
            # However, `PlanningAgent.plan_async` bypasses Strands to use OpenRouter directly.
            # If we use Graph, we are using Strands.
            # We might need to ensure the underlying model supports what Strands expects.
            
            logger.info(f"Processing message in session {session_id}")
            result = graph(message)
            
            # Extract response from planner node
            if result.results and "planner" in result.results:
                node_result = result.results["planner"]
                # node_result.result is AgentResult, str(AgentResult) gives the message
                return str(node_result.result)
                
            return str(result)

        except Exception as e:
            logger.error(f"Error processing message in session {session_id}: {e}")
            raise


# Global instance
_multi_agent_manager = None


def get_multi_agent_manager() -> MultiAgentManager:
    """Get the global multi-agent manager instance."""
    global _multi_agent_manager
    if _multi_agent_manager is None:
        _multi_agent_manager = MultiAgentManager()
    return _multi_agent_manager
