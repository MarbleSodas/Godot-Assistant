"""
Test script for the planning agent.

This script tests both streaming and non-streaming endpoints.
Run this after starting the FastAPI server.
"""

import httpx
import asyncio
import json
import sys


async def test_health():
    """Test the health check endpoint."""
    print("=" * 60)
    print("Testing Agent Health Check")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/api/agent/health")
            response.raise_for_status()
            data = response.json()

            print(f"Status: {data['status']}")
            print(f"Agent Ready: {data['agent_ready']}")
            print(f"Model: {data.get('model', 'N/A')}")
            print("✓ Health check passed\n")
            return True

        except Exception as e:
            print(f"✗ Health check failed: {e}\n")
            return False


async def test_config():
    """Test the config endpoint."""
    print("=" * 60)
    print("Testing Agent Configuration")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/api/agent/config")
            response.raise_for_status()
            data = response.json()

            print(f"Status: {data['status']}")
            print(f"Model ID: {data['config']['model_id']}")
            print(f"Temperature: {data['config']['model_config']['temperature']}")
            print(f"Max Tokens: {data['config']['model_config']['max_tokens']}")
            print(f"Available Tools: {', '.join(data['config']['tools'])}")
            print(f"Conversation Manager: {data['config']['conversation_manager']}")
            print("✓ Configuration retrieved\n")
            return True

        except Exception as e:
            print(f"✗ Configuration test failed: {e}\n")
            return False


async def test_plan_simple():
    """Test simple plan generation (non-streaming)."""
    print("=" * 60)
    print("Testing Simple Plan Generation (Non-Streaming)")
    print("=" * 60)

    prompt = "Create a brief plan for implementing a health bar in a 2D game."

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                "http://localhost:8000/api/agent/plan",
                json={
                    "prompt": prompt,
                    "reset_conversation": True
                }
            )
            response.raise_for_status()
            data = response.json()

            print(f"Status: {data['status']}")
            print(f"\nGenerated Plan:\n{'-' * 60}")
            print(data['plan'])
            print("-" * 60)
            print("✓ Plan generation successful\n")
            return True

        except Exception as e:
            print(f"✗ Plan generation failed: {e}\n")
            return False


async def test_plan_streaming():
    """Test streaming plan generation."""
    print("=" * 60)
    print("Testing Streaming Plan Generation")
    print("=" * 60)

    prompt = "Create a simple plan for adding a pause menu to a game."

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            print("Streaming response:")
            print("-" * 60)

            async with client.stream(
                "POST",
                "http://localhost:8000/api/agent/plan/stream",
                json={
                    "prompt": prompt,
                    "reset_conversation": True
                }
            ) as response:
                response.raise_for_status()

                full_text = ""
                event_type = None

                async for line in response.aiter_lines():
                    line = line.strip()

                    if not line:
                        continue

                    # Parse SSE format
                    if line.startswith("event: "):
                        event_type = line[7:]  # Remove "event: " prefix
                    elif line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Handle different event types
                        if event_type == "start":
                            print(f"[START] {data.get('message', '')}")

                        elif event_type == "data":
                            if "text" in data:
                                text_chunk = data["text"]
                                full_text += text_chunk
                                print(text_chunk, end="", flush=True)

                        elif event_type == "tool_use_start":
                            print(f"\n[TOOL] Using: {data.get('tool_name')}")

                        elif event_type == "metadata":
                            usage = data.get("usage", {})
                            if usage:
                                print(f"\n[METADATA] Input tokens: {usage.get('inputTokens', 0)}, "
                                      f"Output tokens: {usage.get('outputTokens', 0)}")

                        elif event_type == "end":
                            print(f"\n[END] Stop reason: {data.get('stop_reason', 'unknown')}")

                        elif event_type == "error":
                            print(f"\n[ERROR] {data.get('error', 'Unknown error')}")
                            return False

                        elif event_type == "done":
                            break

            print("\n" + "-" * 60)
            print("✓ Streaming test successful\n")
            return True

        except Exception as e:
            print(f"\n✗ Streaming test failed: {e}\n")
            return False


async def test_reset():
    """Test conversation reset."""
    print("=" * 60)
    print("Testing Conversation Reset")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post("http://localhost:8000/api/agent/reset")
            response.raise_for_status()
            data = response.json()

            print(f"Status: {data['status']}")
            print(f"Message: {data['message']}")
            print("✓ Reset successful\n")
            return True

        except Exception as e:
            print(f"✗ Reset failed: {e}\n")
            return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PLANNING AGENT TEST SUITE")
    print("=" * 60)
    print()

    # Check if server is running
    print("Checking if server is running...")
    try:
        async with httpx.AsyncClient() as client:
            await client.get("http://localhost:8000/api/health", timeout=5.0)
        print("✓ Server is running\n")
    except Exception:
        print("✗ Server is not running!")
        print("\nPlease start the server first:")
        print("  cd backend")
        print("  python main.py")
        print()
        sys.exit(1)

    # Run tests
    results = []

    results.append(("Health Check", await test_health()))
    results.append(("Configuration", await test_config()))
    results.append(("Simple Plan", await test_plan_simple()))
    results.append(("Streaming Plan", await test_plan_streaming()))
    results.append(("Reset", await test_reset()))

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<40} {status}")

    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)

    print()
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    print("=" * 60)
    print()

    # Exit with appropriate code
    sys.exit(0 if passed_tests == total_tests else 1)


if __name__ == "__main__":
    asyncio.run(main())
