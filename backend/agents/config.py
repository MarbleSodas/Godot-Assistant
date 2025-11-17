"""
Configuration module for the planning agent.

Handles loading environment variables and providing agent configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class AgentConfig:
    """Configuration for the planning agent."""

    # OpenRouter API Configuration
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

    # Model Configuration
    DEFAULT_PLANNING_MODEL = os.getenv("DEFAULT_PLANNING_MODEL", "openai/gpt-4-turbo")
    FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "anthropic/claude-3.5-sonnet")

    # Agent Configuration
    AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.7"))
    AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "4000"))

    # Application Configuration
    APP_NAME = os.getenv("APP_NAME", "Godot-Assistant")
    APP_URL = os.getenv("APP_URL", "http://localhost:8000")

    # MCP Server Configuration
    ENABLE_MCP_TOOLS = os.getenv("ENABLE_MCP_TOOLS", "true").lower() == "true"
    MCP_FAIL_SILENTLY = os.getenv("MCP_FAIL_SILENTLY", "true").lower() == "true"

    # Sequential Thinking MCP Server
    ENABLE_SEQUENTIAL_THINKING = os.getenv("ENABLE_SEQUENTIAL_THINKING", "true").lower() == "true"
    SEQUENTIAL_THINKING_COMMAND = os.getenv("SEQUENTIAL_THINKING_COMMAND", "npx")
    SEQUENTIAL_THINKING_ARGS = os.getenv("SEQUENTIAL_THINKING_ARGS", "-y,@modelcontextprotocol/server-sequential-thinking").split(",")

    # Context7 MCP Server
    ENABLE_CONTEXT7 = os.getenv("ENABLE_CONTEXT7", "true").lower() == "true"
    CONTEXT7_COMMAND = os.getenv("CONTEXT7_COMMAND", "npx")
    CONTEXT7_ARGS = os.getenv("CONTEXT7_ARGS", "-y,@upstash/context7-mcp").split(",")

    # System Prompt for Planning Agent
    PLANNING_AGENT_SYSTEM_PROMPT = """You are a specialized planning agent designed to create detailed execution plans for other agents.

Your role is to:
1. Analyze the user's request thoroughly
2. Break down complex tasks into clear, actionable steps
3. Identify dependencies between steps
4. Suggest appropriate tools and resources
5. Define success criteria for each step
6. Anticipate potential challenges and provide solutions

When creating a plan, structure it as follows:
- **Objective**: Clear statement of the goal
- **Analysis**: Understanding of the requirements and context
- **Steps**: Numbered, sequential steps with details
  - For each step, include:
    * Description of what needs to be done
    * Required tools or resources
    * Expected outcome
    * Potential challenges
- **Dependencies**: Which steps depend on others
- **Success Criteria**: How to know the task is complete
- **Risks & Mitigations**: Potential issues and how to handle them

Use the available tools to:
- Read and analyze existing code files
- Search the codebase for patterns and implementations
- Fetch documentation and reference materials
- Research best practices and solutions

**Advanced Reasoning with Sequential Thinking:**
When facing complex, multi-step problems that require deep analysis, use the sequential-thinking tool:
- It provides step-by-step reasoning capabilities with hypothesis generation and verification
- Useful for breaking down ambiguous requirements into concrete steps
- Helps explore alternative approaches and identify edge cases
- Enables iterative problem-solving with course correction
- Best for: architectural decisions, complex algorithm design, debugging intricate issues

**Library Documentation with Context7:**
When you need up-to-date documentation for libraries and frameworks:
1. Use `resolve-library-id` to find the correct library identifier (e.g., "fastapi" -> "/tiangolo/fastapi")
2. Use `get-library-docs` with the resolved ID to fetch relevant documentation
- Specify a `topic` parameter to focus on specific areas (e.g., "routing", "authentication")
- Adjust `tokens` parameter to control documentation depth (default: 5000)
- Best for: learning new APIs, finding usage examples, understanding best practices

Be thorough, precise, and actionable. Your plans should enable another agent or developer to execute the task successfully without ambiguity."""

    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required configuration is present.

        Returns:
            True if configuration is valid, False otherwise
        """
        if not cls.OPENROUTER_API_KEY:
            print("Warning: OPENROUTER_API_KEY is not set. Please set it in .env file.")
            return False

        return True

    @classmethod
    def get_model_config(cls) -> dict:
        """
        Get model configuration as a dictionary.

        Returns:
            Dictionary of model configuration parameters
        """
        return {
            "temperature": cls.AGENT_TEMPERATURE,
            "max_tokens": cls.AGENT_MAX_TOKENS
        }

    @classmethod
    def get_openrouter_config(cls) -> dict:
        """
        Get OpenRouter configuration as a dictionary.

        Returns:
            Dictionary of OpenRouter configuration parameters
        """
        return {
            "api_key": cls.OPENROUTER_API_KEY,
            "model_id": cls.DEFAULT_PLANNING_MODEL,
            "app_name": cls.APP_NAME,
            "app_url": cls.APP_URL
        }

    @classmethod
    def get_mcp_servers_config(cls) -> dict:
        """
        Get MCP servers configuration.

        Returns:
            Dictionary of MCP server configurations
        """
        servers = {}

        if cls.ENABLE_MCP_TOOLS:
            if cls.ENABLE_SEQUENTIAL_THINKING:
                servers["sequential-thinking"] = {
                    "command": cls.SEQUENTIAL_THINKING_COMMAND,
                    "args": cls.SEQUENTIAL_THINKING_ARGS,
                    "prefix": "mcp__sequential_thinking__"
                }

            if cls.ENABLE_CONTEXT7:
                servers["context7"] = {
                    "command": cls.CONTEXT7_COMMAND,
                    "args": cls.CONTEXT7_ARGS,
                    "prefix": "mcp__context7__"
                }

        return servers

    @classmethod
    def is_mcp_enabled(cls) -> bool:
        """
        Check if MCP tools are enabled.

        Returns:
            bool: True if MCP tools should be loaded
        """
        return cls.ENABLE_MCP_TOOLS and (
            cls.ENABLE_SEQUENTIAL_THINKING or cls.ENABLE_CONTEXT7
        )


# Validate configuration on module import
if not AgentConfig.validate():
    print("Agent configuration validation failed. Some features may not work.")
