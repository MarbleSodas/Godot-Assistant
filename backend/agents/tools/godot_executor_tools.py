"""
Godot Executor Tools for Agent Automation.

This module provides specialized tools for executor agents to perform
actions and make changes in Godot projects. These tools focus on
execution, modification, and automation capabilities.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

from strands import tool
from .godot_bridge import get_godot_bridge, ensure_godot_connection, CommandResponse
from .godot_debug_tools import NodeInfo

logger = logging.getLogger(__name__)


@dataclass
class CreationResult:
    """Result of a node/scene creation operation."""
    success: bool
    created_path: Optional[str] = None
    created_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ModificationResult:
    """Result of a modification operation."""
    success: bool
    modified_path: Optional[str] = None
    old_value: Any = None
    new_value: Any = None
    error: Optional[str] = None


class GodotExecutorTools:
    """
    Collection of execution and automation tools for Godot projects.

    These tools are designed for executor agents to perform actions
    and make modifications to Godot projects.
    """

    def __init__(self):
        """Initialize executor tools with Godot bridge connection."""
        self.bridge = get_godot_bridge()
        self._operation_history: List[Dict[str, Any]] = []

    async def ensure_connection(self) -> bool:
        """Ensure connection to Godot is active."""
        return await ensure_godot_connection()

    def _record_operation(self, operation_type: str, target: str, result: bool, details: Dict[str, Any] = None):
        """Record an operation in the history for undo/redo tracking."""
        operation = {
            "type": operation_type,
            "target": target,
            "result": result,
            "timestamp": asyncio.get_event_loop().time(),
            "details": details or {}
        }
        self._operation_history.append(operation)

        # Keep history manageable
        if len(self._operation_history) > 100:
            self._operation_history = self._operation_history[-50:]

    # Node Creation and Management
    async def create_node(
        self,
        node_type: str,
        parent_path: str,
        node_name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> CreationResult:
        """
        Create a new node in the scene tree.

        Args:
            node_type: Type of node to create (e.g., "Node2D", "Sprite2D")
            parent_path: Path to parent node
            node_name: Name for the new node (optional)
            properties: Initial properties to set (optional)

        Returns:
            CreationResult with operation outcome
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            params = {
                "node_type": node_type,
                "parent_path": parent_path
            }
            if node_name:
                params["node_name"] = node_name
            if properties:
                params["properties"] = properties

            response = await self.bridge.send_command("create_node", **params)

            if response.success:
                created_path = response.data.get("path")
                self._record_operation("create_node", created_path, True, params)
                logger.info(f"Created node {node_type} at {created_path}")

                return CreationResult(
                    success=True,
                    created_path=created_path,
                    created_id=response.data.get("id")
                )
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("create_node", parent_path, False, {"error": error_msg})
                logger.error(f"Failed to create node: {error_msg}")

                return CreationResult(success=False, error=error_msg)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error creating node: {error_msg}")
            self._record_operation("create_node", parent_path, False, {"error": error_msg})
            return CreationResult(success=False, error=error_msg)

    async def delete_node(self, node_path: str) -> bool:
        """
        Delete a node from the scene tree.

        Args:
            node_path: Path to the node to delete

        Returns:
            True if successful, False otherwise
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command("delete_node", node_path=node_path)

            if response.success:
                self._record_operation("delete_node", node_path, True)
                logger.info(f"Deleted node at {node_path}")
                return True
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("delete_node", node_path, False, {"error": error_msg})
                logger.error(f"Failed to delete node: {error_msg}")
                return False

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error deleting node: {error_msg}")
            self._record_operation("delete_node", node_path, False, {"error": error_msg})
            return False

    async def modify_node_property(
        self,
        node_path: str,
        property_name: str,
        new_value: Any
    ) -> ModificationResult:
        """
        Modify a property of a node.

        Args:
            node_path: Path to the target node
            property_name: Name of the property to modify
            new_value: New value for the property

        Returns:
            ModificationResult with operation outcome
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command(
                "modify_node_property",
                node_path=node_path,
                property_name=property_name,
                new_value=new_value
            )

            if response.success:
                old_value = response.data.get("old_value")
                self._record_operation("modify_property", node_path, True, {
                    "property": property_name,
                    "old_value": old_value,
                    "new_value": new_value
                })
                logger.info(f"Modified {property_name} on {node_path} from {old_value} to {new_value}")

                return ModificationResult(
                    success=True,
                    modified_path=node_path,
                    old_value=old_value,
                    new_value=new_value
                )
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("modify_property", node_path, False, {"error": error_msg})
                logger.error(f"Failed to modify property: {error_msg}")

                return ModificationResult(success=False, error=error_msg)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error modifying property: {error_msg}")
            self._record_operation("modify_property", node_path, False, {"error": error_msg})
            return ModificationResult(success=False, error=error_msg)

    async def reparent_node(
        self,
        node_path: str,
        new_parent_path: str,
        new_position: Optional[int] = None
    ) -> bool:
        """
        Move a node to a new parent.

        Args:
            node_path: Path to the node to move
            new_parent_path: Path to the new parent
            new_position: Position in new parent's children (optional)

        Returns:
            True if successful, False otherwise
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            params = {
                "node_path": node_path,
                "new_parent_path": new_parent_path
            }
            if new_position is not None:
                params["position"] = new_position

            response = await self.bridge.send_command("reparent_node", **params)

            if response.success:
                self._record_operation("reparent_node", node_path, True, params)
                logger.info(f"Reparented {node_path} to {new_parent_path}")
                return True
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("reparent_node", node_path, False, {"error": error_msg})
                logger.error(f"Failed to reparent node: {error_msg}")
                return False

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error reparenting node: {error_msg}")
            self._record_operation("reparent_node", node_path, False, {"error": error_msg})
            return False

    # Scene Management
    async def create_new_scene(
        self,
        scene_name: str,
        root_node_type: str = "Node",
        save_path: Optional[str] = None
    ) -> CreationResult:
        """
        Create a new scene.

        Args:
            scene_name: Name for the new scene
            root_node_type: Type of root node (default: "Node")
            save_path: Path to save the scene (optional)

        Returns:
            CreationResult with operation outcome
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            params = {
                "scene_name": scene_name,
                "root_node_type": root_node_type
            }
            if save_path:
                params["save_path"] = save_path

            response = await self.bridge.send_command("create_scene", **params)

            if response.success:
                scene_path = response.data.get("scene_path")
                self._record_operation("create_scene", scene_name, True, params)
                logger.info(f"Created new scene: {scene_path}")

                return CreationResult(
                    success=True,
                    created_path=scene_path
                )
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("create_scene", scene_name, False, {"error": error_msg})
                logger.error(f"Failed to create scene: {error_msg}")

                return CreationResult(success=False, error=error_msg)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error creating scene: {error_msg}")
            self._record_operation("create_scene", scene_name, False, {"error": error_msg})
            return CreationResult(success=False, error=error_msg)

    async def open_scene(self, scene_path: str) -> bool:
        """
        Open a scene in the editor.

        Args:
            scene_path: Path to the .tscn file

        Returns:
            True if successful, False otherwise
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command("open_scene", scene_path=scene_path)

            if response.success:
                self._record_operation("open_scene", scene_path, True)
                logger.info(f"Opened scene: {scene_path}")
                return True
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("open_scene", scene_path, False, {"error": error_msg})
                logger.error(f"Failed to open scene: {error_msg}")
                return False

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error opening scene: {error_msg}")
            self._record_operation("open_scene", scene_path, False, {"error": error_msg})
            return False

    async def save_current_scene(self) -> bool:
        """
        Save the currently active scene.

        Returns:
            True if successful, False otherwise
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command("save_current_scene")

            if response.success:
                self._record_operation("save_scene", "current", True)
                logger.info("Saved current scene")
                return True
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("save_scene", "current", False, {"error": error_msg})
                logger.error(f"Failed to save scene: {error_msg}")
                return False

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error saving scene: {error_msg}")
            self._record_operation("save_scene", "current", False, {"error": error_msg})
            return False

    # Selection and Focus
    async def select_nodes(self, node_paths: List[str]) -> bool:
        """
        Select nodes in the editor.

        Args:
            node_paths: List of node paths to select

        Returns:
            True if successful, False otherwise
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command("select_nodes", node_paths=node_paths)

            if response.success:
                self._record_operation("select_nodes", str(node_paths), True)
                logger.info(f"Selected nodes: {node_paths}")
                return True
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("select_nodes", str(node_paths), False, {"error": error_msg})
                logger.error(f"Failed to select nodes: {error_msg}")
                return False

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error selecting nodes: {error_msg}")
            self._record_operation("select_nodes", str(node_paths), False, {"error": error_msg})
            return False

    async def focus_node(self, node_path: str) -> bool:
        """
        Focus on a specific node in the editor.

        Args:
            node_path: Path to the node to focus

        Returns:
            True if successful, False otherwise
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command("focus_node", node_path=node_path)

            if response.success:
                self._record_operation("focus_node", node_path, True)
                logger.info(f"Focused on node: {node_path}")
                return True
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("focus_node", node_path, False, {"error": error_msg})
                logger.error(f"Failed to focus node: {error_msg}")
                return False

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error focusing node: {error_msg}")
            self._record_operation("focus_node", node_path, False, {"error": error_msg})
            return False

    # Project Operations
    async def play_scene(self) -> bool:
        """
        Start playing the current scene.

        Returns:
            True if successful, False otherwise
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command("play_scene")

            if response.success:
                self._record_operation("play_scene", "current", True)
                logger.info("Started playing scene")
                return True
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("play_scene", "current", False, {"error": error_msg})
                logger.error(f"Failed to play scene: {error_msg}")
                return False

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error playing scene: {error_msg}")
            self._record_operation("play_scene", "current", False, {"error": error_msg})
            return False

    async def stop_playing(self) -> bool:
        """
        Stop playing the current scene.

        Returns:
            True if successful, False otherwise
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command("stop_playing")

            if response.success:
                self._record_operation("stop_playing", "current", True)
                logger.info("Stopped playing scene")
                return True
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("stop_playing", "current", False, {"error": error_msg})
                logger.error(f"Failed to stop playing: {error_msg}")
                return False

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error stopping playing: {error_msg}")
            self._record_operation("stop_playing", "current", False, {"error": error_msg})
            return False

    # Batch Operations
    async def create_node_batch(
        self,
        creations: List[Dict[str, Any]]
    ) -> List[CreationResult]:
        """
        Create multiple nodes in a batch operation.

        Args:
            creations: List of creation specifications

        Returns:
            List of CreationResult objects
        """
        results = []

        for creation in creations:
            result = await self.create_node(**creation)
            results.append(result)

            # Small delay between operations to prevent overwhelming Godot
            await asyncio.sleep(0.1)

        return results

    async def modify_properties_batch(
        self,
        modifications: List[Dict[str, Any]]
    ) -> List[ModificationResult]:
        """
        Modify multiple properties in a batch operation.

        Args:
            modifications: List of modification specifications

        Returns:
            List of ModificationResult objects
        """
        results = []

        for modification in modifications:
            result = await self.modify_node_property(**modification)
            results.append(result)

            # Small delay between operations
            await asyncio.sleep(0.1)

        return results

    # Advanced Creation and Deletion Operations
    async def duplicate_node(self, node_path: str, new_name: Optional[str] = None) -> CreationResult:
        """
        Duplicate an existing node in the scene tree.

        Args:
            node_path: Path to the node to duplicate
            new_name: Name for the duplicated node (optional)

        Returns:
            CreationResult with operation outcome
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            params = {"node_path": node_path}
            if new_name:
                params["new_name"] = new_name

            response = await self.bridge.send_command("duplicate_node", **params)

            if response.success:
                duplicated_path = response.data.get("path")
                self._record_operation("duplicate_node", node_path, True, params)
                logger.info(f"Duplicated node {node_path} to {duplicated_path}")

                return CreationResult(
                    success=True,
                    created_path=duplicated_path,
                    created_id=response.data.get("id")
                )
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("duplicate_node", node_path, False, {"error": error_msg})
                logger.error(f"Failed to duplicate node: {error_msg}")

                return CreationResult(success=False, error=error_msg)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error duplicating node: {error_msg}")
            self._record_operation("duplicate_node", node_path, False, {"error": error_msg})
            return CreationResult(success=False, error=error_msg)

    async def create_resource(
        self,
        resource_type: str,
        resource_path: str,
        initial_data: Optional[Dict[str, Any]] = None
    ) -> CreationResult:
        """
        Create a new Godot resource file.

        Args:
            resource_type: Type of resource to create
            resource_path: Path where to save the resource
            initial_data: Initial data for the resource (optional)

        Returns:
            CreationResult with operation outcome
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            params = {
                "resource_type": resource_type,
                "resource_path": resource_path
            }
            if initial_data:
                params["initial_data"] = initial_data

            response = await self.bridge.send_command("create_resource", **params)

            if response.success:
                created_path = response.data.get("path")
                self._record_operation("create_resource", resource_path, True, params)
                logger.info(f"Created resource {resource_type} at {created_path}")

                return CreationResult(success=True, created_path=created_path)
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("create_resource", resource_path, False, {"error": error_msg})
                logger.error(f"Failed to create resource: {error_msg}")

                return CreationResult(success=False, error=error_msg)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error creating resource: {error_msg}")
            self._record_operation("create_resource", resource_path, False, {"error": error_msg})
            return CreationResult(success=False, error=error_msg)

    async def delete_resource(self, resource_path: str) -> bool:
        """
        Delete a Godot resource file.

        Args:
            resource_path: Path to the resource file to delete

        Returns:
            True if successful, False otherwise
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command("delete_resource", resource_path=resource_path)

            if response.success:
                self._record_operation("delete_resource", resource_path, True)
                logger.info(f"Deleted resource: {resource_path}")
                return True
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("delete_resource", resource_path, False, {"error": error_msg})
                logger.error(f"Failed to delete resource: {error_msg}")
                return False

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error deleting resource: {error_msg}")
            self._record_operation("delete_resource", resource_path, False, {"error": error_msg})
            return False

    async def attach_script_to_node(self, node_path: str, script_path: str) -> bool:
        """
        Attach a GDScript to a node.

        Args:
            node_path: Path to the target node
            script_path: Path to the GDScript file

        Returns:
            True if successful, False otherwise
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command(
                "attach_script_to_node",
                node_path=node_path,
                script_path=script_path
            )

            if response.success:
                self._record_operation("attach_script", node_path, True, {"script_path": script_path})
                logger.info(f"Attached script {script_path} to {node_path}")
                return True
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("attach_script", node_path, False, {"error": error_msg})
                logger.error(f"Failed to attach script: {error_msg}")
                return False

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error attaching script: {error_msg}")
            self._record_operation("attach_script", node_path, False, {"error": error_msg})
            return False

    async def create_and_attach_script(
        self,
        node_path: str,
        script_content: str,
        script_name: Optional[str] = None
    ) -> CreationResult:
        """
        Create a new GDScript and attach it to a node.

        Args:
            node_path: Path to the target node
            script_content: Content of the GDScript
            script_name: Name for the script file (optional)

        Returns:
            CreationResult with operation outcome
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            params = {
                "node_path": node_path,
                "script_content": script_content
            }
            if script_name:
                params["script_name"] = script_name

            response = await self.bridge.send_command("create_and_attach_script", **params)

            if response.success:
                script_path = response.data.get("script_path")
                self._record_operation("create_and_attach_script", node_path, True, params)
                logger.info(f"Created and attached script {script_path} to {node_path}")

                return CreationResult(success=True, created_path=script_path)
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("create_and_attach_script", node_path, False, {"error": error_msg})
                logger.error(f"Failed to create and attach script: {error_msg}")

                return CreationResult(success=False, error=error_msg)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error creating and attaching script: {error_msg}")
            self._record_operation("create_and_attach_script", node_path, False, {"error": error_msg})
            return CreationResult(success=False, error=error_msg)

    async def create_node_with_script(
        self,
        node_type: str,
        parent_path: str,
        script_content: str,
        node_name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> CreationResult:
        """
        Create a node with an attached script in one operation.

        Args:
            node_type: Type of node to create
            parent_path: Path to parent node
            script_content: Content of the script to attach
            node_name: Name for the new node (optional)
            properties: Initial properties for the node (optional)

        Returns:
            CreationResult with operation outcome
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            params = {
                "node_type": node_type,
                "parent_path": parent_path,
                "script_content": script_content
            }
            if node_name:
                params["node_name"] = node_name
            if properties:
                params["properties"] = properties

            response = await self.bridge.send_command("create_node_with_script", **params)

            if response.success:
                created_path = response.data.get("path")
                script_path = response.data.get("script_path")
                self._record_operation("create_node_with_script", parent_path, True, params)
                logger.info(f"Created node {node_type} with script at {created_path}")

                return CreationResult(
                    success=True,
                    created_path=created_path,
                    created_id=response.data.get("id")
                )
            else:
                error_msg = response.error or "Unknown error"
                self._record_operation("create_node_with_script", parent_path, False, {"error": error_msg})
                logger.error(f"Failed to create node with script: {error_msg}")

                return CreationResult(success=False, error=error_msg)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error creating node with script: {error_msg}")
            self._record_operation("create_node_with_script", parent_path, False, {"error": error_msg})
            return CreationResult(success=False, error=error_msg)

    async def delete_node_batch(self, node_paths: List[str]) -> List[bool]:
        """
        Delete multiple nodes in a batch operation.

        Args:
            node_paths: List of node paths to delete

        Returns:
            List of bool indicating success for each deletion
        """
        results = []

        for node_path in node_paths:
            result = await self.delete_node(node_path)
            results.append(result)
            await asyncio.sleep(0.1)  # Small delay between operations

        return results

    # Utility Methods
    async def get_operation_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of performed operations.

        Returns:
            List of operation records
        """
        return self._operation_history.copy()

    async def clear_operation_history(self):
        """Clear the operation history."""
        self._operation_history.clear()

    async def undo_last_operation(self) -> bool:
        """
        Attempt to undo the last operation using Godot's undo system.

        Returns:
            True if undo was successful, False otherwise
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command("undo")

            if response.success:
                logger.info("Successfully undid last operation")
                return True
            else:
                logger.error(f"Failed to undo: {response.error}")
                return False

        except Exception as e:
            logger.error(f"Error during undo: {e}")
            return False

    async def redo_last_operation(self) -> bool:
        """
        Attempt to redo the last undone operation using Godot's redo system.

        Returns:
            True if redo was successful, False otherwise
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command("redo")

            if response.success:
                logger.info("Successfully redid last operation")
                return True
            else:
                logger.error(f"Failed to redo: {response.error}")
                return False

        except Exception as e:
            logger.error(f"Error during redo: {e}")
            return False


# Convenience functions for direct tool access
@tool
async def create_node(node_type: str, parent_path: str, **kwargs) -> CreationResult:
    """Create a new node in the Godot scene tree.

    Args:
        node_type: Type of node to create (e.g., 'Node2D', 'Sprite2D', 'Camera2D')
        parent_path: Path to the parent node where the new node will be added
        **kwargs: Additional node properties to set

    Returns:
        CreationResult containing success status and created node information
    """
    tools = GodotExecutorTools()
    return await tools.create_node(node_type, parent_path, **kwargs)


@tool
async def modify_node_property(node_path: str, property_name: str, new_value: Any) -> ModificationResult:
    """Modify a property of a node in the Godot scene tree.

    Args:
        node_path: Path to the node whose property will be modified
        property_name: Name of the property to modify
        new_value: New value for the property

    Returns:
        ModificationResult containing success status and modification details
    """
    tools = GodotExecutorTools()
    return await tools.modify_node_property(node_path, property_name, new_value)


@tool
async def create_scene(scene_name: str, **kwargs) -> CreationResult:
    """Create a new scene in the Godot project.

    Args:
        scene_name: Name for the new scene
        **kwargs: Additional scene configuration options

    Returns:
        CreationResult containing success status and created scene information
    """
    tools = GodotExecutorTools()
    return await tools.create_new_scene(scene_name, **kwargs)


@tool
async def open_scene(scene_path: str) -> bool:
    """Open a scene in the Godot editor.

    Args:
        scene_path: Path to the scene file to open

    Returns:
        bool: True if scene was opened successfully, False otherwise
    """
    tools = GodotExecutorTools()
    return await tools.open_scene(scene_path)


@tool
async def select_nodes(node_paths: List[str]) -> bool:
    """Select nodes in the Godot editor.

    Args:
        node_paths: List of node paths to select

    Returns:
        bool: True if nodes were selected successfully, False otherwise
    """
    tools = GodotExecutorTools()
    return await tools.select_nodes(node_paths)


@tool
async def play_scene() -> bool:
    """Start playing the current scene in Godot.

    Returns:
        bool: True if scene started playing successfully, False otherwise
    """
    tools = GodotExecutorTools()
    return await tools.play_scene()


@tool
async def stop_playing() -> bool:
    """Stop playing the current scene in Godot.

    Returns:
        bool: True if scene stopped playing successfully, False otherwise
    """
    tools = GodotExecutorTools()
    return await tools.stop_playing()


@tool
async def delete_node(node_path: str) -> bool:
    """Delete a node from the Godot scene tree.

    Args:
        node_path: Path to the node to delete (e.g., "Root/Player/Sprite")

    Returns:
        bool: True if node was deleted successfully, False otherwise
    """
    tools = GodotExecutorTools()
    return await tools.delete_node(node_path)


@tool
async def reparent_node(node_path: str, new_parent_path: str, **kwargs) -> bool:
    """Move a node to a new parent in the scene tree.

    Args:
        node_path: Path to the node to move
        new_parent_path: Path to the new parent node
        **kwargs: Additional options like position index

    Returns:
        bool: True if node was reparented successfully, False otherwise
    """
    tools = GodotExecutorTools()
    return await tools.reparent_node(node_path, new_parent_path, **kwargs)


@tool
async def duplicate_node(node_path: str, new_name: Optional[str] = None) -> CreationResult:
    """Duplicate an existing node in the scene tree.

    Args:
        node_path: Path to the node to duplicate
        new_name: Name for the duplicated node (optional)

    Returns:
        CreationResult with operation outcome
    """
    tools = GodotExecutorTools()
    return await tools.duplicate_node(node_path, new_name)


@tool
async def create_resource(
    resource_type: str,
    resource_path: str,
    initial_data: Optional[Dict[str, Any]] = None
) -> CreationResult:
    """Create a new Godot resource file.

    Args:
        resource_type: Type of resource to create (e.g., "ShaderMaterial", "GDScript")
        resource_path: Path where to save the resource
        initial_data: Initial data for the resource (optional)

    Returns:
        CreationResult with operation outcome
    """
    tools = GodotExecutorTools()
    return await tools.create_resource(resource_type, resource_path, initial_data)


@tool
async def delete_resource(resource_path: str) -> bool:
    """Delete a Godot resource file.

    Args:
        resource_path: Path to the resource file to delete

    Returns:
        bool: True if resource was deleted successfully, False otherwise
    """
    tools = GodotExecutorTools()
    return await tools.delete_resource(resource_path)


@tool
async def attach_script_to_node(node_path: str, script_path: str) -> bool:
    """Attach a GDScript to a node.

    Args:
        node_path: Path to the target node
        script_path: Path to the GDScript file

    Returns:
        bool: True if script was attached successfully, False otherwise
    """
    tools = GodotExecutorTools()
    return await tools.attach_script_to_node(node_path, script_path)


@tool
async def create_and_attach_script(node_path: str, script_content: str, script_name: Optional[str] = None) -> CreationResult:
    """Create a new GDScript and attach it to a node.

    Args:
        node_path: Path to the target node
        script_content: Content of the GDScript
        script_name: Name for the script file (optional)

    Returns:
        CreationResult with operation outcome
    """
    tools = GodotExecutorTools()
    return await tools.create_and_attach_script(node_path, script_content, script_name)


@tool
async def create_node_with_script(
    node_type: str,
    parent_path: str,
    script_content: str,
    node_name: Optional[str] = None,
    properties: Optional[Dict[str, Any]] = None
) -> CreationResult:
    """Create a node with an attached script in one operation.

    Args:
        node_type: Type of node to create
        parent_path: Path to parent node
        script_content: Content of the script to attach
        node_name: Name for the new node (optional)
        properties: Initial properties for the node (optional)

    Returns:
        CreationResult with operation outcome
    """
    tools = GodotExecutorTools()
    return await tools.create_node_with_script(node_type, parent_path, script_content, node_name, properties)


@tool
async def batch_create_nodes(creations: List[Dict[str, Any]]) -> List[CreationResult]:
    """Create multiple nodes in a batch operation.

    Args:
        creations: List of node creation specifications

    Returns:
        List of CreationResult objects
    """
    tools = GodotExecutorTools()
    return await tools.create_node_batch(creations)


@tool
async def batch_delete_nodes(node_paths: List[str]) -> List[bool]:
    """Delete multiple nodes in a batch operation.

    Args:
        node_paths: List of node paths to delete

    Returns:
        List of bool indicating success for each deletion
    """
    tools = GodotExecutorTools()
    return await tools.delete_node_batch(node_paths)


@tool
async def batch_modify_properties(modifications: List[Dict[str, Any]]) -> List[ModificationResult]:
    """Modify multiple properties in a batch operation.

    Args:
        modifications: List of property modification specifications

    Returns:
        List of ModificationResult objects
    """
    tools = GodotExecutorTools()
    return await tools.modify_properties_batch(modifications)