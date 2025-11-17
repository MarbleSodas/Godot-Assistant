"""
Planning Agent Implementation

This module provides the main planning agent that uses OpenRouter and Strands
to generate execution plans for other agents.
"""

import logging
from typing import Optional, AsyncIterable, Dict, Any
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager

from .models import OpenRouterModel
from .config import AgentConfig
from .tools import (
    read_file,
    list_files,
    search_codebase,
    search_documentation,
    fetch_webpage,
    get_godot_api_reference
)
from .tools.mcp_tools import MCPToolManager

logger = logging.getLogger(__name__)


class PlanningAgent:
    """
    Planning agent that creates execution plans for other agents.

    This agent uses OpenRouter models via Strands Agents framework to:
    - Analyze user requests
    - Break down complex tasks into steps
    - Identify dependencies and resources
    - Provide actionable execution plans
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        enable_mcp: Optional[bool] = None,
        **kwargs
    ):
        """
        Initialize the planning agent.

        Args:
            api_key: OpenRouter API key (defaults to config)
            model_id: Model ID to use (defaults to config)
            enable_mcp: Enable MCP tools (defaults to config)
            **kwargs: Additional configuration options
        """
        # Get configuration
        config = AgentConfig.get_openrouter_config()
        model_config = AgentConfig.get_model_config()

        # Override with provided values
        api_key_value = api_key if api_key else config["api_key"]
        if model_id:
            model_config["model_id"] = model_id
        else:
            model_config["model_id"] = config["model_id"]

        # Add app name and URL to model config
        model_config["app_name"] = config.get("app_name", "Godot-Assistant")
        model_config["app_url"] = config.get("app_url", "http://localhost:8000")

        # Merge additional config
        model_config.update(kwargs)

        # Initialize OpenRouter model with new signature
        try:
            self.model = OpenRouterModel(api_key_value, **model_config)
            logger.info(f"Initialized OpenRouter model: {model_config.get('model_id')}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            # Try fallback model
            if model_config.get('model_id') != AgentConfig.FALLBACK_MODEL:
                logger.info(f"Attempting fallback model: {AgentConfig.FALLBACK_MODEL}")
                model_config['model_id'] = AgentConfig.FALLBACK_MODEL
                self.model = OpenRouterModel(api_key_value, **model_config)
            else:
                raise

        # Define base tools for the agent
        self.tools = [
            read_file,
            list_files,
            search_codebase,
            search_documentation,
            fetch_webpage,
            get_godot_api_reference
        ]

        # Track MCP manager for cleanup
        self.mcp_manager: Optional[MCPToolManager] = None

        # Initialize MCP tools if enabled
        mcp_enabled = enable_mcp if enable_mcp is not None else AgentConfig.is_mcp_enabled()
        if mcp_enabled:
            self._initialize_mcp_tools_sync()

        # Initialize conversation manager for context handling
        self.conversation_manager = SlidingWindowConversationManager(
            window_size=20,  # Keep last 20 messages for context
            should_truncate_results=True
        )

        # Create Strands agent
        self.agent = Agent(
            model=self.model,
            tools=self.tools,
            system_prompt=AgentConfig.PLANNING_AGENT_SYSTEM_PROMPT,
            conversation_manager=self.conversation_manager
        )

        logger.info("Planning agent initialized successfully")

    def _initialize_mcp_tools_sync(self):
        """
        Initialize MCP tools synchronously during agent construction.

        This is a workaround since __init__ can't be async. MCP tools are
        initialized lazily on first use.
        """
        try:
            logger.info("MCP tools will be initialized on first agent invocation")
            self.mcp_manager = MCPToolManager.get_instance()
            # Note: Actual initialization happens in _ensure_mcp_initialized
        except Exception as e:
            error_msg = f"Failed to prepare MCP tool manager: {e}"
            if AgentConfig.MCP_FAIL_SILENTLY:
                logger.warning(error_msg)
                logger.warning("Continuing without MCP tools")
            else:
                logger.error(error_msg)
                raise

    async def _ensure_mcp_initialized(self):
        """
        Ensure MCP tools are initialized before use.

        This is called before agent invocations to lazily initialize MCP.
        """
        if self.mcp_manager and not self.mcp_manager.is_connected():
            try:
                logger.info("Initializing MCP tools...")
                servers_config = AgentConfig.get_mcp_servers_config()
                success = await self.mcp_manager.initialize(
                    servers=servers_config,
                    fail_silently=AgentConfig.MCP_FAIL_SILENTLY
                )

                if success:
                    # Add MCP tools to the tools list
                    mcp_tools = self.mcp_manager.get_all_tools()
                    logger.info(f"Adding {len(mcp_tools)} MCP tools to agent")
                    self.tools.extend(mcp_tools)

                    # Recreate agent with updated tools
                    self.agent = Agent(
                        model=self.model,
                        tools=self.tools,
                        system_prompt=AgentConfig.PLANNING_AGENT_SYSTEM_PROMPT,
                        conversation_manager=self.conversation_manager
                    )

                    connected_servers = self.mcp_manager.get_connected_servers()
                    logger.info(f"MCP tools initialized successfully: {', '.join(connected_servers)}")
                else:
                    logger.warning("No MCP servers connected")

            except Exception as e:
                error_msg = f"Failed to initialize MCP tools: {e}"
                if AgentConfig.MCP_FAIL_SILENTLY:
                    logger.warning(error_msg)
                    logger.warning("Continuing without MCP tools")
                else:
                    logger.error(error_msg)
                    raise

    def plan(self, prompt: str) -> str:
        """
        Generate a plan synchronously.

        Args:
            prompt: User's request for planning

        Returns:
            Generated plan as a string
        """
        try:
            result = self.agent(prompt)

            # Extract text from result
            if hasattr(result, 'message'):
                content = result.message.get('content', [])
                if content and isinstance(content, list):
                    text_parts = [
                        block.get('text', '')
                        for block in content
                        if 'text' in block
                    ]
                    return '\n'.join(text_parts)

            return str(result)

        except Exception as e:
            logger.error(f"Error generating plan: {e}")
            raise

    async def plan_async(self, prompt: str) -> str:
        """
        Generate a plan asynchronously.

        Args:
            prompt: User's request for planning

        Returns:
            Generated plan as a string
        """
        try:
            # Ensure MCP tools are initialized
            await self._ensure_mcp_initialized()

            result = await self.agent.invoke_async(prompt)

            # Extract text from result
            if hasattr(result, 'message'):
                content = result.message.get('content', [])
                if content and isinstance(content, list):
                    text_parts = [
                        block.get('text', '')
                        for block in content
                        if 'text' in block
                    ]
                    return '\n'.join(text_parts)

            return str(result)

        except Exception as e:
            logger.error(f"Error generating plan: {e}")
            raise

    async def plan_stream(self, prompt: str) -> AsyncIterable[Dict[str, Any]]:
        """
        Generate a plan with streaming responses.

        Args:
            prompt: User's request for planning

        Yields:
            Dictionary events containing:
            - type: Event type ('start', 'data', 'tool_use', 'tool_result', 'end')
            - data: Event data (text chunk, tool info, etc.)
        """
        try:
            # Yield start event
            yield {
                "type": "start",
                "data": {"message": "Starting plan generation..."}
            }

            # Ensure MCP tools are initialized
            await self._ensure_mcp_initialized()

            # Stream agent response
            async for event in self.agent.stream_async(prompt):
                # Handle different event types
                if "messageStart" in event:
                    yield {
                        "type": "message_start",
                        "data": event["messageStart"]
                    }

                elif "contentBlockStart" in event:
                    start = event["contentBlockStart"]["start"]
                    if start.get("type") == "text":
                        yield {
                            "type": "text_start",
                            "data": {}
                        }
                    elif start.get("type") == "toolUse":
                        yield {
                            "type": "tool_use_start",
                            "data": {
                                "tool_name": start.get("name"),
                                "tool_use_id": start.get("toolUseId")
                            }
                        }

                elif "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]

                    if "text" in delta:
                        # Stream text chunk
                        yield {
                            "type": "data",
                            "data": {"text": delta["text"]}
                        }

                    elif "toolUse" in delta:
                        # Tool use in progress
                        yield {
                            "type": "tool_use_delta",
                            "data": {"input": delta["toolUse"].get("input", "")}
                        }

                elif "contentBlockStop" in event:
                    yield {
                        "type": "content_block_stop",
                        "data": {}
                    }

                elif "messageStop" in event:
                    stop_data = event["messageStop"]
                    yield {
                        "type": "end",
                        "data": {
                            "stop_reason": stop_data.get("stopReason", "end_turn")
                        }
                    }

                elif "metadata" in event:
                    # Include metadata (token usage, etc.)
                    yield {
                        "type": "metadata",
                        "data": event["metadata"]
                    }

        except Exception as e:
            logger.error(f"Error in streaming plan: {e}")
            yield {
                "type": "error",
                "data": {"error": str(e)}
            }

    def reset_conversation(self):
        """Reset the conversation history by recreating the agent."""
        # Recreate the agent to reset conversation history
        self.agent = Agent(
            model=self.model,
            tools=self.tools,
            system_prompt=AgentConfig.PLANNING_AGENT_SYSTEM_PROMPT,
            conversation_manager=self.conversation_manager
        )
        logger.info("Conversation history reset")

    async def close(self):
        """Close the agent and cleanup resources."""
        # Cleanup MCP connections
        if self.mcp_manager:
            try:
                await self.mcp_manager.cleanup()
                logger.info("MCP tools cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up MCP tools: {e}")

        # Close model
        if hasattr(self.model, 'close'):
            await self.model.close()

        logger.info("Planning agent closed")


# Singleton instance
_planning_agent_instance: Optional[PlanningAgent] = None


def get_planning_agent() -> PlanningAgent:
    """
    Get or create the planning agent singleton instance.

    Returns:
        PlanningAgent instance
    """
    global _planning_agent_instance

    if _planning_agent_instance is None:
        _planning_agent_instance = PlanningAgent()

    return _planning_agent_instance


async def close_planning_agent():
    """Close the planning agent singleton instance."""
    global _planning_agent_instance

    if _planning_agent_instance is not None:
        await _planning_agent_instance.close()
        _planning_agent_instance = None
