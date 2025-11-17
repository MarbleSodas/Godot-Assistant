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


# Validate configuration on module import
if not AgentConfig.validate():
    print("Agent configuration validation failed. Some features may not work.")
