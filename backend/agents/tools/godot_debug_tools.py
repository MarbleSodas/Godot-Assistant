"""
Godot Debug Tools for Planning Agents.

This module provides specialized tools for planning agents to analyze,
inspect, and understand Godot projects. These tools focus on gathering
information and context rather than making changes.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from strands import tool
from .godot_bridge import get_godot_bridge, ensure_godot_connection, CommandResponse

logger = logging.getLogger(__name__)


@dataclass
class SceneInfo:
    """Information about a Godot scene."""
    name: str
    path: str
    root_node_type: str
    node_count: int
    has_script: bool
    script_path: Optional[str] = None


@dataclass
class NodeInfo:
    """Information about a specific node in the scene tree."""
    name: str
    type: str
    path: str
    parent: Optional[str]
    children: List[str]
    properties: Dict[str, Any]
    groups: List[str]
    has_script: bool
    script_path: Optional[str] = None


@dataclass
class VisualSnapshot:
    """Visual context information from Godot viewport."""
    screenshot_path: Optional[str]
    viewport_size: Tuple[int, int]
    camera_info: Dict[str, Any]
    selected_nodes: List[str]
    scene_tree_state: Dict[str, Any]


class GodotDebugTools:
    """
    Collection of debug and analysis tools for Godot projects.

    These tools are designed for planning agents to gather information
    and context about Godot projects without making modifications.
    """

    def __init__(self):
        """Initialize debug tools with Godot bridge connection."""
        self.bridge = get_godot_bridge()

    async def ensure_connection(self) -> bool:
        """Ensure connection to Godot is active."""
        return await ensure_godot_connection()

    async def get_project_overview(self) -> Dict[str, Any]:
        """
        Get comprehensive overview of the current Godot project.

        Returns:
            Dictionary containing project information, settings, and statistics
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            # Get basic project info
            project_info = await self.bridge.get_project_info()
            if not project_info:
                raise RuntimeError("Unable to retrieve project information")

            # Get current scene info
            current_scene_response = await self.bridge.send_command("get_current_scene_info")
            current_scene = current_scene_response.data if current_scene_response.success else None

            # Get project statistics
            stats_response = await self.bridge.send_command("get_project_statistics")
            stats = stats_response.data if stats_response.success else {}

            # Get editor state
            editor_state_response = await self.bridge.send_command("get_editor_state")
            editor_state = editor_state_response.data if editor_state_response.success else {}

            return {
                "project_info": {
                    "path": project_info.project_path,
                    "name": project_info.project_name,
                    "godot_version": project_info.godot_version,
                    "plugin_version": project_info.plugin_version
                },
                "current_scene": current_scene,
                "statistics": stats,
                "editor_state": editor_state,
                "timestamp": asyncio.get_event_loop().time()
            }

        except Exception as e:
            logger.error(f"Error getting project overview: {e}")
            raise

    async def get_scene_tree_analysis(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Analyze the current scene tree structure.

        Args:
            detailed: Include detailed node information

        Returns:
            Scene tree analysis with node hierarchy and properties
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            command = "get_scene_tree_detailed" if detailed else "get_scene_tree_simple"
            response = await self.bridge.send_command(command)

            if not response.success:
                raise RuntimeError(f"Failed to get scene tree: {response.error}")

            scene_tree = response.data

            # Analyze scene structure
            analysis = {
                "scene_tree": scene_tree,
                "analysis": self._analyze_scene_structure(scene_tree),
                "recommendations": self._generate_scene_recommendations(scene_tree)
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing scene tree: {e}")
            raise

    async def get_node_details(self, node_path: str) -> Optional[NodeInfo]:
        """
        Get detailed information about a specific node.

        Args:
            node_path: Path to the node in scene tree

        Returns:
            NodeInfo with detailed node information, or None if not found
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command(
                "get_node_info",
                node_path=node_path
            )

            if not response.success:
                logger.warning(f"Failed to get node info for {node_path}: {response.error}")
                return None

            node_data = response.data
            return NodeInfo(
                name=node_data.get("name", ""),
                type=node_data.get("type", ""),
                path=node_data.get("path", ""),
                parent=node_data.get("parent"),
                children=node_data.get("children", []),
                properties=node_data.get("properties", {}),
                groups=node_data.get("groups", []),
                has_script=node_data.get("has_script", False),
                script_path=node_data.get("script_path")
            )

        except Exception as e:
            logger.error(f"Error getting node details for {node_path}: {e}")
            return None

    async def search_nodes(
        self,
        search_type: str = "type",
        query: str = "",
        scene_root: Optional[str] = None
    ) -> List[NodeInfo]:
        """
        Search for nodes in the scene tree.

        Args:
            search_type: Type of search ("type", "name", "group", "script")
            query: Search query
            scene_root: Root node to search within (optional)

        Returns:
            List of matching nodes
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            command_map = {
                "type": "search_nodes_by_type",
                "name": "search_nodes_by_name",
                "group": "search_nodes_by_group",
                "script": "search_nodes_by_script"
            }

            if search_type not in command_map:
                raise ValueError(f"Invalid search type: {search_type}")

            command = command_map[search_type]
            params = {"query": query}
            if scene_root:
                params["scene_root"] = scene_root

            response = await self.bridge.send_command(command, **params)

            if not response.success:
                raise RuntimeError(f"Search failed: {response.error}")

            nodes_data = response.data or []
            return [
                NodeInfo(
                    name=node.get("name", ""),
                    type=node.get("type", ""),
                    path=node.get("path", ""),
                    parent=node.get("parent"),
                    children=node.get("children", []),
                    properties=node.get("properties", {}),
                    groups=node.get("groups", []),
                    has_script=node.get("has_script", False),
                    script_path=node.get("script_path")
                )
                for node in nodes_data
            ]

        except Exception as e:
            logger.error(f"Error searching nodes: {e}")
            return []

    async def capture_visual_context(self, include_3d: bool = True) -> VisualSnapshot:
        """
        Capture visual context from current Godot viewport.

        Args:
            include_3d: Include 3D viewport information

        Returns:
            VisualSnapshot with screenshot and viewport information
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            # Capture screenshot
            screenshot_response = await self.bridge.send_command(
                "capture_viewport_screenshot",
                include_3d=include_3d
            )

            # Get viewport information
            viewport_response = await self.bridge.send_command("get_viewport_info")

            # Get selected nodes
            selection_response = await self.bridge.send_command("get_selected_nodes")

            screenshot_path = screenshot_response.data if screenshot_response.success else None
            viewport_info = viewport_response.data if viewport_response.success else {}
            selected_nodes = selection_response.data if selection_response.success else []

            return VisualSnapshot(
                screenshot_path=screenshot_path,
                viewport_size=(
                    viewport_info.get("width", 0),
                    viewport_info.get("height", 0)
                ),
                camera_info=viewport_info.get("camera", {}),
                selected_nodes=selected_nodes,
                scene_tree_state=viewport_info.get("scene_tree", {})
            )

        except Exception as e:
            logger.error(f"Error capturing visual context: {e}")
            raise

    async def get_debug_output(self, lines: int = 100, severity_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Get recent debug output from Godot editor with enhanced parsing.

        Args:
            lines: Number of recent lines to retrieve
            severity_filter: Optional filter by severity level ('error', 'warning', 'info')

        Returns:
            Dict containing parsed debug messages with metadata
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command(
                "get_debug_output",
                lines=lines
            )

            if not response.success:
                raise RuntimeError(f"Failed to get debug output: {response.error}")

            raw_lines = response.data or []

            # Parse debug messages for better agent understanding
            parsed_messages = []
            error_count = 0
            warning_count = 0
            info_count = 0

            for line in raw_lines:
                # Try to extract severity and message
                severity = "info"  # default
                message = line

                if "ERROR:" in line or "[ERROR]" in line:
                    severity = "error"
                    error_count += 1
                elif "WARNING:" in line or "[WARNING]" in line or "[WARN]" in line:
                    severity = "warning"
                    warning_count += 1
                else:
                    info_count += 1

                parsed_messages.append({
                    "severity": severity,
                    "message": message,
                    "raw": line
                })

            # Apply filter if specified
            if severity_filter:
                parsed_messages = [msg for msg in parsed_messages if msg["severity"] == severity_filter]

            return {
                "messages": parsed_messages,
                "summary": {
                    "total_messages": len(parsed_messages),
                    "error_count": error_count,
                    "warning_count": warning_count,
                    "info_count": info_count,
                    "has_errors": error_count > 0,
                    "has_warnings": warning_count > 0
                },
                "raw_lines": raw_lines
            }

        except Exception as e:
            logger.error(f"Error getting debug output: {e}")
            return {
                "messages": [],
                "summary": {
                    "total_messages": 0,
                    "error_count": 0,
                    "warning_count": 0,
                    "info_count": 0,
                    "has_errors": False,
                    "has_warnings": False
                },
                "raw_lines": []
            }

    async def capture_editor_viewport(self, include_3d: bool = True, include_2d: bool = True) -> Dict[str, Any]:
        """
        Capture focused editor viewport screenshot with metadata.

        Args:
            include_3d: Whether to capture 3D viewport information
            include_2d: Whether to capture 2D viewport information

        Returns:
            Dict containing screenshot path, viewport info, and visual context
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command(
                "capture_visual_context",
                include_3d=include_3d,
                include_2d=include_2d,
                focused_viewport=True
            )

            if not response.success:
                raise RuntimeError(f"Failed to capture editor viewport: {response.error}")

            return {
                "screenshot_path": response.data.get("screenshot_path"),
                "viewport_info": response.data.get("viewport_info", {}),
                "editor_state": response.data.get("editor_state", {}),
                "timestamp": response.data.get("timestamp"),
                "capture_type": "editor_viewport"
            }

        except Exception as e:
            logger.error(f"Error capturing editor viewport: {e}")
            raise

    async def capture_game_viewport(self, wait_frames: int = 3) -> Dict[str, Any]:
        """
        Capture in-game viewport screenshot with timing control.

        Args:
            wait_frames: Number of frames to wait before capturing

        Returns:
            Dict containing screenshot path and game state information
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command(
                "capture_game_screenshot",
                wait_frames=wait_frames
            )

            if not response.success:
                raise RuntimeError(f"Failed to capture game viewport: {response.error}")

            return {
                "screenshot_path": response.data.get("screenshot_path"),
                "game_state": response.data.get("game_state", {}),
                "timestamp": response.data.get("timestamp"),
                "capture_type": "game_viewport",
                "frames_waited": wait_frames
            }

        except Exception as e:
            logger.error(f"Error capturing game viewport: {e}")
            raise

    async def get_visual_debug_info(self) -> Dict[str, Any]:
        """
        Get visual debugging overlays and information.

        Returns:
            Dict containing debug overlays, collision shapes, navigation data, etc.
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command(
                "get_debug_info",
                include_overlays=True,
                include_collision_shapes=True,
                include_navigation=True
            )

            if not response.success:
                raise RuntimeError(f"Failed to get visual debug info: {response.error}")

            return {
                "debug_overlays": response.data.get("debug_overlays", {}),
                "collision_shapes": response.data.get("collision_shapes", {}),
                "navigation_data": response.data.get("navigation_data", {}),
                "performance_overlay": response.data.get("performance_overlay", {}),
                "debug_settings": response.data.get("debug_settings", {})
            }

        except Exception as e:
            logger.error(f"Error getting visual debug info: {e}")
            raise

    async def get_debug_logs(self, severity_filter: Optional[str] = None, time_range: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
        """
        Get filtered debug logs with advanced parsing and analytics.

        Args:
            severity_filter: Filter by severity ('error', 'warning', 'info', 'debug')
            time_range: Time range filter ('recent', 'last_minute', 'last_hour')
            limit: Maximum number of log entries to return

        Returns:
            Dict containing filtered logs, analytics, and metadata
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            # Start debug capture if not already active
            await self.bridge.send_command("start_debug_capture")

            # Get debug output
            response = await self.bridge.send_command("get_debug_output", lines=limit * 2)  # Get more to filter

            if not response.success:
                raise RuntimeError(f"Failed to get debug logs: {response.error}")

            raw_lines = response.data or []

            # Parse and filter logs
            parsed_logs = []
            analytics = {
                "error_count": 0,
                "warning_count": 0,
                "info_count": 0,
                "debug_count": 0,
                "total_entries": 0,
                "time_span": None,
                "common_errors": {},
                "log_frequency": {}
            }

            import re
            from datetime import datetime, timedelta

            current_time = datetime.now()
            time_threshold = None

            if time_range:
                if time_range == "last_minute":
                    time_threshold = current_time - timedelta(minutes=1)
                elif time_range == "last_hour":
                    time_threshold = current_time - timedelta(hours=1)

            for line in raw_lines:
                # Parse log entry
                severity = "info"
                timestamp = current_time
                message = line

                # Try to extract timestamp and severity
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if timestamp_match:
                    try:
                        timestamp = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                    except:
                        pass

                if "ERROR:" in line or "[ERROR]" in line:
                    severity = "error"
                elif "WARNING:" in line or "[WARNING]" in line or "[WARN]" in line:
                    severity = "warning"
                elif "DEBUG:" in line or "[DEBUG]" in line:
                    severity = "debug"
                else:
                    severity = "info"

                # Apply filters
                if severity_filter and severity != severity_filter:
                    continue

                if time_threshold and timestamp < time_threshold:
                    continue

                parsed_logs.append({
                    "timestamp": timestamp.isoformat(),
                    "severity": severity,
                    "message": message,
                    "raw": line
                })

                # Update analytics
                analytics[f"{severity}_count"] += 1
                analytics["total_entries"] += 1

                # Track common errors
                if severity == "error":
                    error_key = message[:50]  # First 50 chars as key
                    analytics["common_errors"][error_key] = analytics["common_errors"].get(error_key, 0) + 1

            # Sort by timestamp and apply limit
            parsed_logs.sort(key=lambda x: x["timestamp"], reverse=True)
            parsed_logs = parsed_logs[:limit]

            # Calculate time span
            if parsed_logs:
                earliest = datetime.fromisoformat(parsed_logs[-1]["timestamp"])
                latest = datetime.fromisoformat(parsed_logs[0]["timestamp"])
                analytics["time_span"] = str(latest - earliest)

            return {
                "logs": parsed_logs,
                "analytics": analytics,
                "filters_applied": {
                    "severity_filter": severity_filter,
                    "time_range": time_range,
                    "limit": limit
                }
            }

        except Exception as e:
            logger.error(f"Error getting debug logs: {e}")
            raise

    async def search_debug_logs(self, pattern: str, case_sensitive: bool = False, regex: bool = False) -> Dict[str, Any]:
        """
        Search debug logs for specific patterns.

        Args:
            pattern: Search pattern to match in log messages
            case_sensitive: Whether the search should be case sensitive
            regex: Whether to use regex pattern matching

        Returns:
            Dict containing matching log entries and search metadata
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            # Get debug logs
            response = await self.bridge.send_command("get_debug_output", lines=500)

            if not response.success:
                raise RuntimeError(f"Failed to search debug logs: {response.error}")

            raw_lines = response.data or []

            # Prepare search pattern
            import re
            if regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                search_pattern = re.compile(pattern, flags)
            else:
                if case_sensitive:
                    search_pattern = pattern
                else:
                    search_pattern = pattern.lower()

            matches = []
            total_searched = len(raw_lines)

            for i, line in enumerate(raw_lines):
                search_text = line if case_sensitive else line.lower()

                if regex:
                    if search_pattern.search(line):
                        matches.append({
                            "line_number": i + 1,
                            "message": line,
                            "match_groups": search_pattern.findall(line)
                        })
                else:
                    if search_pattern in search_text:
                        matches.append({
                            "line_number": i + 1,
                            "message": line
                        })

            return {
                "matches": matches,
                "search_metadata": {
                    "pattern": pattern,
                    "case_sensitive": case_sensitive,
                    "regex": regex,
                    "total_searched": total_searched,
                    "match_count": len(matches),
                    "match_percentage": round((len(matches) / total_searched) * 100, 2) if total_searched > 0 else 0
                }
            }

        except Exception as e:
            logger.error(f"Error searching debug logs: {e}")
            raise

    async def monitor_debug_output(self, duration: int = 10, severity_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Monitor debug output in real-time for a specified duration.

        Args:
            duration: Duration in seconds to monitor output
            severity_filter: Optional filter by severity level

        Returns:
            Dict containing captured debug output during monitoring period
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            import asyncio
            from datetime import datetime

            # Clear existing debug output
            await self.bridge.send_command("clear_debug_output")

            # Start debug capture
            await self.bridge.send_command("start_debug_capture")

            start_time = datetime.now()
            captured_messages = []

            # Monitor for specified duration
            while (datetime.now() - start_time).total_seconds() < duration:
                await asyncio.sleep(1)  # Check every second

                # Get current debug output
                response = await self.bridge.send_command("get_debug_output", lines=50)

                if response.success and response.data:
                    new_messages = [msg for msg in response.data if msg not in [c["message"] for c in captured_messages]]

                    for msg in new_messages:
                        # Apply severity filter if specified
                        if severity_filter:
                            if severity_filter == "error" and ("ERROR:" in msg or "[ERROR]" in msg):
                                captured_messages.append({"timestamp": datetime.now().isoformat(), "message": msg, "severity": "error"})
                            elif severity_filter == "warning" and ("WARNING:" in msg or "[WARNING]" in msg):
                                captured_messages.append({"timestamp": datetime.now().isoformat(), "message": msg, "severity": "warning"})
                        else:
                            # Determine severity
                            severity = "info"
                            if "ERROR:" in msg or "[ERROR]" in msg:
                                severity = "error"
                            elif "WARNING:" in msg or "[WARNING]" in msg:
                                severity = "warning"

                            captured_messages.append({"timestamp": datetime.now().isoformat(), "message": msg, "severity": severity})

            # Stop debug capture
            await self.bridge.send_command("stop_debug_capture")

            return {
                "monitoring_metadata": {
                    "duration": duration,
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "severity_filter": severity_filter,
                    "total_messages": len(captured_messages)
                },
                "messages": captured_messages,
                "summary": {
                    "error_count": len([m for m in captured_messages if m["severity"] == "error"]),
                    "warning_count": len([m for m in captured_messages if m["severity"] == "warning"]),
                    "info_count": len([m for m in captured_messages if m["severity"] == "info"]),
                    "message_rate": round(len(captured_messages) / duration, 2)
                }
            }

        except Exception as e:
            logger.error(f"Error monitoring debug output: {e}")
            raise

    async def analyze_project_structure(self) -> Dict[str, Any]:
        """
        Analyze overall project structure and organization.

        Returns:
            Project structure analysis with recommendations
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            # Get project file list
            files_response = await self.bridge.send_command("get_project_files")
            project_files = files_response.data if files_response.success else []

            # Get scene analysis
            scenes_response = await self.bridge.send_command("analyze_all_scenes")
            scenes_analysis = scenes_response.data if scenes_response.success else {}

            # Get script analysis
            scripts_response = await self.bridge.send_command("analyze_scripts")
            scripts_analysis = scripts_response.data if scripts_response.success else {}

            # Get resource analysis
            resources_response = await self.bridge.send_command("analyze_resources")
            resources_analysis = resources_response.data if resources_response.success else {}

            analysis = {
                "project_files": project_files,
                "scenes": scenes_analysis,
                "scripts": scripts_analysis,
                "resources": resources_analysis,
                "recommendations": self._generate_structure_recommendations(
                    project_files, scenes_analysis, scripts_analysis, resources_analysis
                )
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing project structure: {e}")
            raise

    async def inspect_scene_file(self, scene_path: str) -> Dict[str, Any]:
        """
        Inspect a scene file without loading it.

        Args:
            scene_path: Path to the .tscn file

        Returns:
            Scene file analysis
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command(
                "inspect_scene_file",
                scene_path=scene_path
            )

            if not response.success:
                raise RuntimeError(f"Failed to inspect scene file: {response.error}")

            return response.data

        except Exception as e:
            logger.error(f"Error inspecting scene file {scene_path}: {e}")
            raise

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics from Godot.

        Returns:
            Performance metrics including FPS, memory usage, etc.
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            response = await self.bridge.send_command("get_performance_metrics")

            if not response.success:
                raise RuntimeError(f"Failed to get performance metrics: {response.error}")

            return response.data

        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

    def _analyze_scene_structure(self, scene_tree: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scene tree structure and identify patterns."""
        analysis = {
            "total_nodes": 0,
            "node_types": {},
            "depth": 0,
            "complexity_score": 0,
            "issues": []
        }

        def analyze_node(node: Dict[str, Any], depth: int = 0):
            analysis["total_nodes"] += 1
            analysis["depth"] = max(analysis["depth"], depth)

            node_type = node.get("type", "Unknown")
            analysis["node_types"][node_type] = analysis["node_types"].get(node_type, 0) + 1

            # Check for potential issues
            children = node.get("children", [])
            if len(children) > 20:
                analysis["issues"].append(f"Node '{node.get('name', '')}' has too many children ({len(children)})")

            for child in children:
                analyze_node(child, depth + 1)

        if scene_tree:
            analyze_node(scene_tree)

        # Calculate complexity score
        analysis["complexity_score"] = (
            analysis["total_nodes"] * 0.1 +
            analysis["depth"] * 2 +
            len(analysis["node_types"]) * 0.5
        )

        return analysis

    def _generate_scene_recommendations(self, scene_tree: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on scene analysis."""
        recommendations = []

        analysis = self._analyze_scene_structure(scene_tree)

        if analysis["total_nodes"] > 100:
            recommendations.append("Consider splitting large scenes into smaller sub-scenes")

        if analysis["depth"] > 10:
            recommendations.append("Scene hierarchy is very deep, consider flattening some levels")

        if "Node2D" in analysis["node_types"] and analysis["node_types"]["Node2D"] > 50:
            recommendations.append("Many Node2D nodes found, consider grouping them under Container nodes")

        if len(analysis["issues"]) > 0:
            recommendations.append(f"Found {len(analysis['issues'])} potential issues to address")

        return recommendations

    def _generate_structure_recommendations(
        self,
        files: List[str],
        scenes: Dict[str, Any],
        scripts: Dict[str, Any],
        resources: Dict[str, Any]
    ) -> List[str]:
        """Generate project structure recommendations."""
        recommendations = []

        # File organization
        scene_files = [f for f in files if f.endswith('.tscn')]
        script_files = [f for f in files if f.endswith(('.gd', '.cs'))]

        if len(scene_files) > 20 and not any('scenes/' in f for f in scene_files):
            recommendations.append("Consider organizing scenes into a 'scenes/' subfolder")

        if len(script_files) > 20 and not any('scripts/' in f for f in script_files):
            recommendations.append("Consider organizing scripts into a 'scripts/' subfolder")

        # Scene complexity
        if scenes.get("average_node_count", 0) > 50:
            recommendations.append("Average scene complexity is high, consider scene optimization")

        # Script analysis
        if scripts.get("total_lines", 0) > 10000:
            recommendations.append("Large codebase detected, consider code organization and refactoring")

        return recommendations

    async def analyze_node_performance(self, node_path: str) -> Dict[str, Any]:
        """
        Analyze performance metrics for a specific node.

        Args:
            node_path: Path to the node to analyze

        Returns:
            Dict containing node performance data and optimization suggestions
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            # Get node details and performance info
            response = await self.bridge.send_command(
                "get_node_details",
                node_path=node_path
            )

            if not response.success:
                raise RuntimeError(f"Failed to analyze node performance: {response.error}")

            data = response.data or {}

            # Get current performance metrics for comparison
            perf_response = await self.bridge.send_command("get_performance_metrics")
            current_metrics = perf_response.data if perf_response.success else {}

            # Analyze and provide recommendations
            recommendations = []
            node_type = data.get("type", "")
            child_count = len(data.get("children", []))
            has_script = data.get("has_script", False)
            script_path = data.get("script_path", "")

            # Performance analysis
            if child_count > 50:
                recommendations.append(f"Node has {child_count} children - consider organizing into smaller groups")

            if has_script and script_path:
                recommendations.append("Node has attached script - check for expensive operations in _process()")

            if "3D" in node_type and child_count > 20:
                recommendations.append("3D node with many children - consider using culling or lod")

            if "MeshInstance" in node_type:
                recommendations.append("Check mesh complexity and consider LOD for distant views")

            return {
                "node_info": {
                    "path": node_path,
                    "type": node_type,
                    "child_count": child_count,
                    "has_script": has_script,
                    "script_path": script_path
                },
                "performance_metrics": data.get("properties", {}),
                "current_system_metrics": current_metrics,
                "recommendations": recommendations,
                "analysis_timestamp": data.get("timestamp")
            }

        except Exception as e:
            logger.error(f"Error analyzing node performance: {e}")
            raise

    async def get_scene_debug_overlays(self, scene_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get scene debug overlays and visualization information.

        Args:
            scene_path: Optional path to specific scene, defaults to current scene

        Returns:
            Dict containing debug overlays, collision shapes, navigation meshes, etc.
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            # Get current scene debug information
            response = await self.bridge.send_command("capture_visual_context", include_debug_info=True)

            if not response.success:
                raise RuntimeError(f"Failed to get scene debug overlays: {response.error}")

            data = response.data or {}

            # Organize debug information
            overlays = {
                "collision_shapes": data.get("collision_shapes", []),
                "navigation_meshes": data.get("navigation_meshes", []),
                "visibility_rects": data.get("visibility_rects", []),
                "lighting_info": data.get("lighting_info", {}),
                "audio_zones": data.get("audio_zones", []),
                "debug_draws": data.get("debug_draws", [])
            }

            return {
                "scene_path": scene_path or "current_scene",
                "overlays": overlays,
                "debug_settings": data.get("debug_settings", {}),
                "visualization_info": data.get("viewport_info", {}),
                "capture_success": True
            }

        except Exception as e:
            logger.error(f"Error getting scene debug overlays: {e}")
            raise

    async def compare_scenes(self, scene_path_a: str, scene_path_b: str) -> Dict[str, Any]:
        """
        Compare two scenes and highlight differences.

        Args:
            scene_path_a: Path to the first scene
            scene_path_b: Path to the second scene

        Returns:
            Dict containing scene comparison results and differences
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            # Analyze both scenes
            response_a = await self.bridge.send_command("inspect_scene_file", scene_path=scene_path_a)
            response_b = await self.bridge.send_command("inspect_scene_file", scene_path=scene_path_b)

            if not response_a.success or not response_b.success:
                raise RuntimeError(f"Failed to inspect scenes for comparison")

            scene_a_data = response_a.data
            scene_b_data = response_b.data

            # Compare scenes
            differences = {
                "structure_changes": [],
                "node_differences": [],
                "property_differences": [],
                "script_differences": []
            }

            # Compare node counts
            nodes_a = scene_a_data.get("nodes", [])
            nodes_b = scene_b_data.get("nodes", [])

            if len(nodes_a) != len(nodes_b):
                differences["structure_changes"].append(f"Node count differs: {len(nodes_a)} vs {len(nodes_b)}")

            # Compare root nodes
            root_a = scene_a_data.get("root_node", {})
            root_b = scene_b_data.get("root_node", {})

            if root_a.get("type") != root_b.get("type"):
                differences["structure_changes"].append(f"Root node type differs: {root_a.get('type')} vs {root_b.get('type')}")

            # Find nodes that exist in one but not the other
            nodes_a_dict = {node["path"]: node for node in nodes_a}
            nodes_b_dict = {node["path"]: node for node in nodes_b}

            for path in nodes_a_dict:
                if path not in nodes_b_dict:
                    differences["node_differences"].append({
                        "path": path,
                        "type": "removed",
                        "node_info": nodes_a_dict[path]
                    })

            for path in nodes_b_dict:
                if path not in nodes_a_dict:
                    differences["node_differences"].append({
                        "path": path,
                        "type": "added",
                        "node_info": nodes_b_dict[path]
                    })

            # Compare common nodes
            for path in nodes_a_dict:
                if path in nodes_b_dict:
                    node_a = nodes_a_dict[path]
                    node_b = nodes_b_dict[path]

                    if node_a.get("type") != node_b.get("type"):
                        differences["node_differences"].append({
                            "path": path,
                            "type": "type_changed",
                            "old_type": node_a.get("type"),
                            "new_type": node_b.get("type")
                        })

                    # Compare scripts
                    script_a = node_a.get("script_path")
                    script_b = node_b.get("script_path")

                    if script_a != script_b:
                        differences["script_differences"].append({
                            "path": path,
                            "old_script": script_a,
                            "new_script": script_b
                        })

            return {
                "comparison_metadata": {
                    "scene_a": scene_path_a,
                    "scene_b": scene_path_b,
                    "comparison_timestamp": response_a.data.get("timestamp"),
                    "total_differences": len(sum(differences.values(), []))
                },
                "differences": differences,
                "similarities": {
                    "common_node_types": self._find_common_node_types(nodes_a, nodes_b),
                    "structure_similarity": self._calculate_structure_similarity(nodes_a, nodes_b)
                }
            }

        except Exception as e:
            logger.error(f"Error comparing scenes: {e}")
            raise

    def _find_common_node_types(self, nodes_a: List[Dict], nodes_b: List[Dict]) -> List[str]:
        """Find common node types between two scenes."""
        types_a = {node.get("type") for node in nodes_a}
        types_b = {node.get("type") for node in nodes_b}
        return list(types_a.intersection(types_b))

    def _calculate_structure_similarity(self, nodes_a: List[Dict], nodes_b: List[Dict]) -> float:
        """Calculate structural similarity percentage between two scenes."""
        paths_a = {node.get("path") for node in nodes_a}
        paths_b = {node.get("path") for node in nodes_b}

        if not paths_a and not paths_b:
            return 100.0

        common_paths = paths_a.intersection(paths_b)
        total_paths = paths_a.union(paths_b)

        return round((len(common_paths) / len(total_paths)) * 100, 2) if total_paths else 0.0

    async def get_debugger_state(self) -> Dict[str, Any]:
        """
        Get current debugger state and information.

        Returns:
            Dict containing debugger state, breakpoints, and debugging context
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            # Get debugger state
            response = await self.bridge.send_command("get_debugger_state")

            if not response.success:
                raise RuntimeError(f"Failed to get debugger state: {response.error}")

            data = response.data or {}

            return {
                "debugger_active": data.get("debugger_active", False),
                "breakpoints": data.get("breakpoints", []),
                "current_scene": data.get("current_scene", ""),
                "debug_mode": data.get("debug_mode", "none"),
                "error_state": data.get("error_state", {}),
                "step_mode": data.get("step_mode", ""),
                "last_error": data.get("last_error", ""),
                "debug_context": data.get("debug_context", {})
            }

        except Exception as e:
            logger.error(f"Error getting debugger state: {e}")
            raise

    async def access_debug_variables(self, variable_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Access debug variables and watches from the current debugging context.

        Args:
            variable_filter: Optional filter to limit variables by name or type

        Returns:
            Dict containing debug variables, values, and metadata
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            # Get debug variables
            response = await self.bridge.send_command(
                "get_debug_variables",
                filter=variable_filter
            )

            if not response.success:
                raise RuntimeError(f"Failed to access debug variables: {response.error}")

            data = response.data or {}

            # Organize variables by scope and type
            variables = data.get("variables", {})
            organized_vars = {
                "local_variables": [],
                "global_variables": [],
                "member_variables": [],
                "constants": [],
                "functions": []
            }

            for var_name, var_info in variables.items():
                var_entry = {
                    "name": var_name,
                    "type": var_info.get("type", "unknown"),
                    "value": var_info.get("value", ""),
                    "scope": var_info.get("scope", "unknown"),
                    "is_mutable": var_info.get("is_mutable", True)
                }

                scope = var_info.get("scope", "")
                if scope == "local":
                    organized_vars["local_variables"].append(var_entry)
                elif scope == "global":
                    organized_vars["global_variables"].append(var_entry)
                elif scope == "member":
                    organized_vars["member_variables"].append(var_entry)
                elif var_info.get("is_constant", False):
                    organized_vars["constants"].append(var_entry)
                elif var_info.get("type") in ["function", "callable"]:
                    organized_vars["functions"].append(var_entry)

            return {
                "variables": organized_vars,
                "metadata": {
                    "filter_applied": variable_filter,
                    "total_variables": len(variables),
                    "scopes_available": list(set([v.get("scope", "unknown") for v in variables.values()])),
                    "types_available": list(set([v.get("type", "unknown") for v in variables.values()]))
                }
            }

        except Exception as e:
            logger.error(f"Error accessing debug variables: {e}")
            raise

    async def get_call_stack_info(self, max_depth: int = 10) -> Dict[str, Any]:
        """
        Get current call stack information.

        Args:
            max_depth: Maximum depth of call stack to retrieve

        Returns:
            Dict containing call stack frames and debugging information
        """
        if not await self.ensure_connection():
            raise ConnectionError("Failed to connect to Godot plugin")

        try:
            # Get call stack
            response = await self.bridge.send_command(
                "get_call_stack",
                max_depth=max_depth
            )

            if not response.success:
                raise RuntimeError(f"Failed to get call stack info: {response.error}")

            data = response.data or {}

            # Process call stack frames
            frames = data.get("frames", [])
            processed_frames = []

            for i, frame in enumerate(frames[:max_depth]):
                processed_frame = {
                    "level": i,
                    "function": frame.get("function", ""),
                    "script": frame.get("script", ""),
                    "line": frame.get("line", 0),
                    "file": frame.get("file", ""),
                    "arguments": frame.get("arguments", []),
                    "locals": frame.get("locals", {}),
                    "node_path": frame.get("node_path", "")
                }
                processed_frames.append(processed_frame)

            return {
                "frames": processed_frames,
                "metadata": {
                    "total_frames": len(frames),
                    "max_depth": max_depth,
                    "truncated": len(frames) > max_depth,
                    "current_function": processed_frames[0]["function"] if processed_frames else "",
                    "current_script": processed_frames[0]["script"] if processed_frames else ""
                }
            }

        except Exception as e:
            logger.error(f"Error getting call stack info: {e}")
            raise


# Convenience functions for direct tool access
@tool
async def get_project_overview() -> Dict[str, Any]:
    """Get comprehensive project overview from Godot.

    Returns:
        Dict containing project structure, scenes, resources, and metadata
    """
    tools = GodotDebugTools()
    return await tools.get_project_overview()


@tool
async def analyze_scene_tree(detailed: bool = False) -> Dict[str, Any]:
    """Analyze the current scene tree structure in Godot.

    Args:
        detailed: Whether to include detailed node properties

    Returns:
        Dict containing scene tree hierarchy and node information
    """
    tools = GodotDebugTools()
    return await tools.get_scene_tree_analysis(detailed)


@tool
async def capture_visual_context(include_3d: bool = True) -> VisualSnapshot:
    """Capture visual context from the Godot viewport.

    Args:
        include_3d: Whether to capture 3D viewport information

    Returns:
        VisualSnapshot containing screenshot path and viewport information
    """
    tools = GodotDebugTools()
    return await tools.capture_visual_context(include_3d)


@tool
async def search_nodes(search_type: str, query: str, scene_root: Optional[str] = None) -> List[NodeInfo]:
    """Search for nodes in the Godot scene tree.

    Args:
        search_type: Type of search ('name', 'type', 'group', 'script')
        query: Search query string
        scene_root: Optional scene root path to limit search scope

    Returns:
        List of NodeInfo objects matching the search criteria
    """
    tools = GodotDebugTools()
    return await tools.search_nodes(search_type, query, scene_root)


@tool
async def get_debug_output(lines: int = 100, severity_filter: Optional[str] = None) -> Dict[str, Any]:
    """Get recent debug output from Godot with enhanced parsing.

    Args:
        lines: Number of recent debug lines to retrieve
        severity_filter: Optional filter by severity level ('error', 'warning', 'info')

    Returns:
        Dict containing parsed debug messages, severity counts, and metadata
    """
    tools = GodotDebugTools()
    return await tools.get_debug_output(lines, severity_filter)


@tool
async def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics from Godot.

    Returns:
        Dict containing FPS, memory usage, frame time, and other performance data
    """
    tools = GodotDebugTools()
    return await tools.get_performance_metrics()


@tool
async def inspect_scene_file(scene_path: str) -> Dict[str, Any]:
    """Inspect a scene file without loading it.

    Args:
        scene_path: Path to the scene file to inspect

    Returns:
        Dict containing scene structure and metadata
    """
    tools = GodotDebugTools()
    return await tools.inspect_scene_file(scene_path)


# New visual input tools for enhanced debugging
@tool
async def capture_editor_viewport(include_3d: bool = True, include_2d: bool = True) -> Dict[str, Any]:
    """Capture focused editor viewport screenshot with metadata.

    Args:
        include_3d: Whether to capture 3D viewport information
        include_2d: Whether to capture 2D viewport information

    Returns:
        Dict containing screenshot path, viewport info, and visual context
    """
    tools = GodotDebugTools()
    return await tools.capture_editor_viewport(include_3d, include_2d)


@tool
async def capture_game_viewport(wait_frames: int = 3) -> Dict[str, Any]:
    """Capture in-game viewport screenshot with timing control.

    Args:
        wait_frames: Number of frames to wait before capturing

    Returns:
        Dict containing screenshot path and game state information
    """
    tools = GodotDebugTools()
    return await tools.capture_game_viewport(wait_frames)


@tool
async def get_visual_debug_info() -> Dict[str, Any]:
    """Get visual debugging overlays and information.

    Returns:
        Dict containing debug overlays, collision shapes, navigation data, etc.
    """
    tools = GodotDebugTools()
    return await tools.get_visual_debug_info()


# Enhanced log access tools
@tool
async def get_debug_logs(severity_filter: Optional[str] = None, time_range: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
    """Get filtered debug logs with advanced parsing.

    Args:
        severity_filter: Filter by severity ('error', 'warning', 'info', 'debug')
        time_range: Time range filter ('recent', 'last_minute', 'last_hour')
        limit: Maximum number of log entries to return

    Returns:
        Dict containing filtered logs, analytics, and metadata
    """
    tools = GodotDebugTools()
    return await tools.get_debug_logs(severity_filter, time_range, limit)


@tool
async def search_debug_logs(pattern: str, case_sensitive: bool = False, regex: bool = False) -> Dict[str, Any]:
    """Search debug logs for specific patterns.

    Args:
        pattern: Search pattern to match in log messages
        case_sensitive: Whether the search should be case sensitive
        regex: Whether to use regex pattern matching

    Returns:
        Dict containing matching log entries and search metadata
    """
    tools = GodotDebugTools()
    return await tools.search_debug_logs(pattern, case_sensitive, regex)


@tool
async def monitor_debug_output(duration: int = 10, severity_filter: Optional[str] = None) -> Dict[str, Any]:
    """Monitor debug output in real-time for a specified duration.

    Args:
        duration: Duration in seconds to monitor output
        severity_filter: Optional filter by severity level

    Returns:
        Dict containing captured debug output during monitoring period
    """
    tools = GodotDebugTools()
    return await tools.monitor_debug_output(duration, severity_filter)


# Advanced scene analysis tools
@tool
async def analyze_node_performance(node_path: str) -> Dict[str, Any]:
    """Analyze performance metrics for a specific node.

    Args:
        node_path: Path to the node to analyze

    Returns:
        Dict containing node performance data and optimization suggestions
    """
    tools = GodotDebugTools()
    return await tools.analyze_node_performance(node_path)


@tool
async def get_scene_debug_overlays(scene_path: Optional[str] = None) -> Dict[str, Any]:
    """Get scene debug overlays and visualization information.

    Args:
        scene_path: Optional path to specific scene, defaults to current scene

    Returns:
        Dict containing debug overlays, collision shapes, navigation meshes, etc.
    """
    tools = GodotDebugTools()
    return await tools.get_scene_debug_overlays(scene_path)


@tool
async def compare_scenes(scene_path_a: str, scene_path_b: str) -> Dict[str, Any]:
    """Compare two scenes and highlight differences.

    Args:
        scene_path_a: Path to the first scene
        scene_path_b: Path to the second scene

    Returns:
        Dict containing scene comparison results and differences
    """
    tools = GodotDebugTools()
    return await tools.compare_scenes(scene_path_a, scene_path_b)


# Runtime debugging tools
@tool
async def get_debugger_state() -> Dict[str, Any]:
    """Get current debugger state and information.

    Returns:
        Dict containing debugger state, breakpoints, and debugging context
    """
    tools = GodotDebugTools()
    return await tools.get_debugger_state()


@tool
async def access_debug_variables(variable_filter: Optional[str] = None) -> Dict[str, Any]:
    """Access debug variables and watches from the current debugging context.

    Args:
        variable_filter: Optional filter to limit variables by name or type

    Returns:
        Dict containing debug variables, values, and metadata
    """
    tools = GodotDebugTools()
    return await tools.access_debug_variables(variable_filter)


@tool
async def get_call_stack_info(max_depth: int = 10) -> Dict[str, Any]:
    """Get current call stack information.

    Args:
        max_depth: Maximum depth of call stack to retrieve

    Returns:
        Dict containing call stack frames and debugging information
    """
    tools = GodotDebugTools()
    return await tools.get_call_stack_info(max_depth)