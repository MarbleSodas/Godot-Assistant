# Planning Agent Documentation

## Overview

The Planning Agent is a Strands-based AI agent that uses OpenRouter's API to generate detailed execution plans for other agents. It features:

- **Streaming responses** via Server-Sent Events (SSE)
- **Tool calling** with file system and web search capabilities
- **Custom OpenRouter model provider** for Strands Agents
- **RESTful API** endpoints via FastAPI

## Setup

### 1. Install Dependencies

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_actual_api_key_here
DEFAULT_PLANNING_MODEL=openai/gpt-4-turbo
FALLBACK_MODEL=openai/gpt-4-turbo
AGENT_TEMPERATURE=0.7
AGENT_MAX_TOKENS=4000
APP_NAME=Godot-Assistant
APP_URL=http://localhost:8000
```

**Note:** The model `openai/gpt-5.1-codex` was specified but may not be available yet. The configuration defaults to `openai/gpt-4-turbo` as a reliable alternative.

### 3. Run the Application

```bash
python main.py
```

The FastAPI server will start on `http://127.0.0.1:8000`

## API Endpoints

### Health Check

Check if the agent is ready:

```bash
GET /api/agent/health
```

**Response:**
```json
{
  "status": "healthy",
  "agent_ready": true,
  "model": "openai/gpt-4-turbo"
}
```

### Create Plan (Non-Streaming)

Generate a plan without streaming:

```bash
POST /api/agent/plan
Content-Type: application/json

{
  "prompt": "Create a plan for building a 2D platformer game in Godot",
  "reset_conversation": false
}
```

**Response:**
```json
{
  "status": "success",
  "plan": "# Plan for 2D Platformer Game...",
  "metadata": null
}
```

### Create Plan (Streaming)

Generate a plan with real-time streaming:

```bash
POST /api/agent/plan/stream
Content-Type: application/json

{
  "prompt": "Create a plan for implementing user authentication",
  "reset_conversation": false
}
```

**Response:** Server-Sent Events (SSE)

Events you'll receive:
- `event: start` - Plan generation started
- `event: message_start` - Message started
- `event: data` - Text chunks as they're generated
- `event: tool_use_start` - Agent is using a tool
- `event: tool_use_delta` - Tool input being sent
- `event: metadata` - Token usage information
- `event: end` - Plan generation completed
- `event: done` - Stream finished

### Reset Conversation

Clear the conversation history:

```bash
POST /api/agent/reset
```

**Response:**
```json
{
  "status": "success",
  "message": "Conversation history reset"
}
```

### Get Configuration

View current agent configuration:

```bash
GET /api/agent/config
```

**Response:**
```json
{
  "status": "success",
  "config": {
    "model_id": "openai/gpt-4-turbo",
    "model_config": {
      "temperature": 0.7,
      "max_tokens": 4000
    },
    "tools": [
      "read_file",
      "list_files",
      "search_codebase",
      "search_documentation",
      "fetch_webpage",
      "get_godot_api_reference"
    ],
    "conversation_manager": "SlidingWindowConversationManager"
  }
}
```

## Available Tools

The planning agent has access to the following tools:

### File System Tools

1. **read_file(file_path: str)**
   - Read contents of a file
   - Example: `read_file("./main.py")`

2. **list_files(directory: str, pattern: str)**
   - List files in a directory with optional glob pattern
   - Example: `list_files("./agents", "*.py")`

3. **search_codebase(pattern: str, directory: str, file_pattern: str, max_results: int)**
   - Search for regex patterns in the codebase
   - Example: `search_codebase("class.*Agent", ".", "*.py")`

### Web Tools

4. **search_documentation(query: str, source: str)**
   - Search for documentation on a topic
   - Sources: "general", "godot", "python", "fastapi", "strands"
   - Example: `search_documentation("Node2D", "godot")`

5. **fetch_webpage(url: str, extract_text: bool)**
   - Fetch and extract content from a webpage
   - Example: `fetch_webpage("https://docs.godotengine.org", True)`

6. **get_godot_api_reference(class_name: str)**
   - Get Godot API documentation for a specific class
   - Example: `get_godot_api_reference("CharacterBody2D")`

## Testing with cURL

### Non-Streaming Request

```bash
curl -X POST http://localhost:8000/api/agent/plan \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a plan for implementing a save/load system in Godot",
    "reset_conversation": false
  }'
```

### Streaming Request

```bash
curl -X POST http://localhost:8000/api/agent/plan/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a plan for adding multiplayer support to a game",
    "reset_conversation": false
  }' \
  --no-buffer
```

## Testing with Python

```python
import httpx
import asyncio
import json

async def test_streaming():
    url = "http://localhost:8000/api/agent/plan/stream"
    data = {
        "prompt": "Create a plan for building a physics-based puzzle game",
        "reset_conversation": False
    }

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=data, timeout=120.0) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event_data = line[6:]  # Remove "data: " prefix
                    if event_data and event_data != "{}":
                        data = json.loads(event_data)
                        if "text" in data:
                            print(data["text"], end="", flush=True)
            print()  # New line at end

asyncio.run(test_streaming())
```

## Architecture

### Project Structure

```
backend/
├── agents/                      # Agent module
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── planning_agent.py       # Main planning agent
│   ├── models/                 # Custom model providers
│   │   ├── __init__.py
│   │   └── openrouter.py      # OpenRouter integration
│   └── tools/                  # Agent tools
│       ├── __init__.py
│       ├── file_system_tools.py
│       └── web_tools.py
├── api/                        # API routes
│   ├── __init__.py
│   └── agent_routes.py        # Agent endpoints
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (gitignored)
└── .env.example               # Environment template
```

### Key Components

1. **OpenRouterModel** (`agents/models/openrouter.py`)
   - Custom Strands model provider
   - Handles OpenAI SSE format → Strands StreamEvents conversion
   - Tool calling format conversion
   - Error handling and retries

2. **PlanningAgent** (`agents/planning_agent.py`)
   - Main agent implementation
   - Integrates model, tools, and conversation management
   - Provides sync, async, and streaming interfaces

3. **Tools** (`agents/tools/`)
   - File system operations
   - Web search and documentation fetching
   - Godot-specific API reference

4. **API Routes** (`api/agent_routes.py`)
   - FastAPI endpoints
   - SSE streaming support
   - Request/response models

## Troubleshooting

### Agent won't start

- Check that `OPENROUTER_API_KEY` is set in `.env`
- Verify the model ID is correct and available on OpenRouter
- Check logs for initialization errors

### Streaming not working

- Ensure your client supports Server-Sent Events
- Check that nginx or proxy doesn't buffer responses
- Verify `X-Accel-Buffering: no` header is set

### Tool calls failing

- Check file permissions for file system tools
- Verify network connectivity for web tools
- Review tool execution logs in console

### High token usage

- Reduce `AGENT_MAX_TOKENS` in `.env`
- Use conversation reset more frequently
- Consider using a cheaper model for simple plans

## Future Enhancements

- [ ] Add support for more models (Anthropic, Google, etc.)
- [ ] Implement persistent session storage (Redis/SQLite)
- [ ] Add rate limiting and cost tracking
- [ ] Create frontend UI for agent interaction
- [ ] Add more Godot-specific tools
- [ ] Implement plan validation and execution
- [ ] Add structured output for plans (JSON schema)
- [ ] Create agent swarms for complex planning
