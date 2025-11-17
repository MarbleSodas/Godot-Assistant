"""
Quick import test for the planning agent.
This test verifies that all modules can be imported without errors.
"""

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        # Test core imports
        print("  - Testing strands imports...")
        from strands import Agent, tool
        from strands.agent.conversation_manager import SlidingWindowConversationManager
        from strands.models import Model
        print("    ✓ Strands imports successful")

        # Test custom model
        print("  - Testing OpenRouter model...")
        from agents.models import OpenRouterModel
        print("    ✓ OpenRouter model import successful")

        # Test tools
        print("  - Testing agent tools...")
        from agents.tools import (
            read_file,
            list_files,
            search_codebase,
            search_documentation,
            fetch_webpage,
            get_godot_api_reference
        )
        print("    ✓ Tools import successful")

        # Test config
        print("  - Testing configuration...")
        from agents.config import AgentConfig
        print("    ✓ Configuration import successful")

        # Test planning agent
        print("  - Testing planning agent...")
        from agents import PlanningAgent, get_planning_agent
        print("    ✓ Planning agent import successful")

        # Test API routes
        print("  - Testing API routes...")
        from api import agent_router
        print("    ✓ API routes import successful")

        print("\n" + "=" * 60)
        print("✓ ALL IMPORTS SUCCESSFUL!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Add your OpenRouter API key to .env file")
        print("2. Run: python main.py")
        print("3. Test endpoints: python test_agent.py")
        print()

        return True

    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = test_imports()
    sys.exit(0 if success else 1)
