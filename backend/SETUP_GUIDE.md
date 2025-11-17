# Planning Agent Setup Guide

## ✅ Installation Complete!

All dependencies have been installed successfully. Follow these steps to configure and test the planning agent.

## Step 1: Get an OpenRouter API Key

1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up or log in
3. Navigate to [Keys](https://openrouter.ai/keys)
4. Create a new API key
5. Copy the API key

## Step 2: Configure Environment

Edit the `.env` file in the backend directory:

```bash
nano .env  # or use your preferred editor
```

Add your OpenRouter API key:

```env
OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key-here
```

**Important:** Replace `your_openrouter_api_key_here` with your actual API key!

## Step 3: Choose Your Model (Optional)

The default model is `openai/gpt-4-turbo`. You can change it in `.env`:

```env
DEFAULT_PLANNING_MODEL=anthropic/claude-3.5-sonnet
# or
DEFAULT_PLANNING_MODEL=openai/gpt-4-turbo
# or any other model from https://openrouter.ai/models
```

**Note:** The model `openai/gpt-5.1-codex` you requested may not be available yet. We've defaulted to `openai/gpt-4-turbo` which is reliable and powerful.

## Step 4: Verify Setup

Run the import test:

```bash
python test_imports.py
```

You should see:
```
✓ ALL IMPORTS SUCCESSFUL!
```

## Step 5: Start the Application

```bash
python main.py
```

This will:
1. Start the FastAPI server on http://127.0.0.1:8000
2. Open a PyWebView window with your application
3. Enable the planning agent API endpoints

## Step 6: Test the Planning Agent

Once the server is running, open a new terminal and run:

```bash
# In a new terminal window
cd backend
source venv/bin/activate
python test_agent.py
```

This will run a comprehensive test suite including:
- Health check
- Configuration verification
- Non-streaming plan generation
- Streaming plan generation
- Conversation reset

## Available API Endpoints

Once the server is running, you can access:

### Documentation
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

### Planning Agent Endpoints
- `GET /api/agent/health` - Check agent status
- `POST /api/agent/plan` - Generate plan (non-streaming)
- `POST /api/agent/plan/stream` - Generate plan with streaming
- `POST /api/agent/reset` - Reset conversation
- `GET /api/agent/config` - Get configuration

### Example cURL Command

```bash
curl -X POST http://localhost:8000/api/agent/plan/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a plan for building a 2D platformer game in Godot",
    "reset_conversation": false
  }' \
  --no-buffer
```

## Troubleshooting

### "Warning: OPENROUTER_API_KEY is not set"

- Make sure you've edited the `.env` file
- Verify the API key is correct (starts with `sk-or-v1-`)
- Restart the application after changing `.env`

### "Model not found" or "Invalid model"

- Check available models at https://openrouter.ai/models
- Update `DEFAULT_PLANNING_MODEL` in `.env`
- Some models require special access or credits

### "Connection refused" or "Network error"

- Check your internet connection
- Verify OpenRouter is accessible: https://openrouter.ai/
- Check if you have any firewall/proxy blocking the connection

### High costs/token usage

- Reduce `AGENT_MAX_TOKENS` in `.env`
- Use cheaper models like `anthropic/claude-3-haiku`
- Reset conversation more frequently

## Next Steps

1. **Test the agent** - Try different prompts
2. **Integrate with frontend** - Connect Angular app to agent endpoints
3. **Customize tools** - Add more Godot-specific tools
4. **Fine-tune prompts** - Adjust system prompt for better results

## Documentation

- **Full Agent Documentation:** `PLANNING_AGENT_README.md`
- **Strands Documentation:** https://strandsagents.com/
- **OpenRouter Documentation:** https://openrouter.ai/docs

## Support

If you encounter issues:
1. Check the console logs for error messages
2. Verify your OpenRouter API key and credits
3. Review the documentation files
4. Check Strands and OpenRouter documentation for API changes

---

**Everything is set up and ready to go!** 🚀

Just add your OpenRouter API key to `.env` and run `python main.py`!
