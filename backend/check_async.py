import inspect
import asyncio
from agents.tools.godot_debug_tools import get_project_overview

async def main():
    print(f"Tool: {get_project_overview}")
    print(f"Type: {type(get_project_overview)}")
    print(f"Is coroutine function: {inspect.iscoroutinefunction(get_project_overview)}")
    
    if hasattr(get_project_overview, '__wrapped__'):
        print(f"Wrapped: {get_project_overview.__wrapped__}")
        print(f"Wrapped is coroutine: {inspect.iscoroutinefunction(get_project_overview.__wrapped__)}")

if __name__ == "__main__":
    asyncio.run(main())
