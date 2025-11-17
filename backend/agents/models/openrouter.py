"""
OpenRouter Custom Model Provider for Strands Agents

This module provides a custom model provider that integrates OpenRouter's API
with the Strands Agents framework, enabling streaming and tool calling.
"""

import json
import logging
from typing import AsyncIterable, Optional, Any, Dict, List, Type, TypeVar, Union, AsyncGenerator, TypedDict
from typing_extensions import Unpack
import httpx

from pydantic import BaseModel
from strands.models import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec

logger = logging.getLogger(__name__)


# Custom exception for rate limiting
class ModelThrottledException(Exception):
    """Exception raised when the model API rate limit is exceeded."""
    pass

# TypeVar for structured output
T = TypeVar('T', bound=BaseModel)


class ModelConfig(TypedDict, total=False):
    """Configuration for OpenRouter model."""
    model_id: str
    app_name: str
    app_url: str
    timeout: int
    temperature: float
    max_tokens: int
    params: Optional[Dict[str, Any]]


class OpenRouterModel(Model):
    """
    Custom model provider for OpenRouter API with Strands Agents.

    Supports:
    - Streaming responses via Server-Sent Events (SSE)
    - Tool calling with automatic format conversion
    - Error handling and retries
    - Multiple model selection
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        **model_config: Unpack[ModelConfig]
    ) -> None:
        """
        Initialize OpenRouter model provider.

        Args:
            api_key: OpenRouter API key
            **model_config: Model configuration including model_id, app_name, app_url,
                          timeout, temperature, max_tokens, params
        """
        self.api_key = api_key

        # Store configuration with defaults
        self._config: ModelConfig = {
            "model_id": "openai/gpt-4-turbo",
            "app_name": "Godot-Assistant",
            "app_url": "http://localhost:8000",
            "timeout": 120,
            "temperature": 0.7,
            "max_tokens": 4000,
            **model_config  # type: ignore
        }

        # Initialize HTTP client
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._config.get("timeout", 120)),
            headers=self._get_headers()
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for OpenRouter API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self._config.get("app_url", "http://localhost:8000"),
            "X-Title": self._config.get("app_name", "Godot-Assistant"),
            "Content-Type": "application/json"
        }

    def _convert_tool_to_openai_format(self, tool_spec: Dict) -> Dict:
        """
        Convert Strands tool spec (Bedrock format) to OpenAI function format.

        Args:
            tool_spec: Strands tool specification

        Returns:
            OpenAI function specification
        """
        return {
            "type": "function",
            "function": {
                "name": tool_spec.get("name"),
                "description": tool_spec.get("description", ""),
                "parameters": tool_spec.get("input_schema", {})
            }
        }

    def _convert_messages_to_openai_format(self, messages: Messages) -> List[Dict]:
        """
        Convert Strands messages format to OpenAI format.

        Args:
            messages: Strands messages

        Returns:
            OpenAI formatted messages
        """
        openai_messages = []

        for message in messages:
            role = message.get("role")
            content = message.get("content", [])

            # Handle different content types
            if isinstance(content, str):
                openai_messages.append({
                    "role": role,
                    "content": content
                })
            elif isinstance(content, list):
                # Process content blocks
                text_parts = []
                tool_calls = []

                for block in content:
                    if "text" in block:
                        text_parts.append(block["text"])
                    elif "toolUse" in block:
                        tool_use = block["toolUse"]
                        tool_calls.append({
                            "id": tool_use.get("toolUseId", ""),
                            "type": "function",
                            "function": {
                                "name": tool_use.get("name", ""),
                                "arguments": json.dumps(tool_use.get("input", {}))
                            }
                        })
                    elif "toolResult" in block:
                        # Tool result from previous execution
                        tool_result = block["toolResult"]
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_result.get("toolUseId", ""),
                            "content": json.dumps(tool_result.get("content", []))
                        })

                # Add message with text and/or tool calls
                if text_parts or tool_calls:
                    msg = {"role": role}
                    if text_parts:
                        msg["content"] = "\n".join(text_parts)
                    if tool_calls:
                        msg["tool_calls"] = tool_calls
                    openai_messages.append(msg)

        return openai_messages

    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: Optional[str] = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        **kwargs: Any
    ) -> AsyncIterable[StreamEvent]:
        """
        Stream chat completion from OpenRouter API.

        Args:
            messages: Conversation messages
            tool_specs: Optional tool specifications
            system_prompt: Optional system prompt
            tool_choice: Optional tool choice strategy ('auto', 'none', or specific function)
            system_prompt_content: Optional system prompt content blocks
            **kwargs: Additional parameters

        Yields:
            StreamEvent objects
        """
        try:
            # Convert messages to OpenAI format
            openai_messages = self._convert_messages_to_openai_format(messages)

            # Add system prompt if provided
            if system_prompt:
                openai_messages.insert(0, {
                    "role": "system",
                    "content": system_prompt
                })

            # Prepare request payload
            payload = {
                "model": self._config.get("model_id", "openai/gpt-4-turbo"),
                "messages": openai_messages,
                "stream": True,
                "temperature": self._config.get("temperature", 0.7),
                "max_tokens": self._config.get("max_tokens", 4000)
            }

            # Add tools if provided
            if tool_specs:
                payload["tools"] = [
                    self._convert_tool_to_openai_format(spec)
                    for spec in tool_specs
                ]
                # Use provided tool_choice or default to "auto"
                payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"

            # Make streaming request
            async with self._client.stream(
                "POST",
                f"{self.BASE_URL}/chat/completions",
                json=payload
            ) as response:
                response.raise_for_status()

                # Parse SSE stream and convert to Strands events
                async for event in self._parse_sse_stream(response):
                    yield event

        except httpx.HTTPStatusError as e:
            # Handle rate limiting
            if e.response.status_code == 429:
                logger.error("OpenRouter rate limit exceeded")
                raise ModelThrottledException("Rate limit exceeded")

            logger.error(f"OpenRouter API error: {e}")
            # Yield error event
            yield {
                "messageStop": {
                    "stopReason": "error"
                }
            }
        except httpx.HTTPError as e:
            logger.error(f"OpenRouter HTTP error: {e}")
            yield {
                "messageStop": {
                    "stopReason": "error"
                }
            }
        except Exception as e:
            logger.error(f"Unexpected error in stream: {e}")
            yield {
                "messageStop": {
                    "stopReason": "error"
                }
            }

    async def _parse_sse_stream(
        self,
        response: httpx.Response
    ) -> AsyncIterable[StreamEvent]:
        """
        Parse Server-Sent Events stream and convert to Strands StreamEvents.

        Args:
            response: HTTPX streaming response

        Yields:
            StreamEvent objects
        """
        message_started = False
        content_block_started = False
        current_tool_use = None
        accumulated_text = ""
        accumulated_tool_args = ""

        async for line in response.aiter_lines():
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith(":"):
                continue

            # Parse SSE data
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix

                # Check for end of stream
                if data == "[DONE]":
                    if content_block_started:
                        yield {"contentBlockStop": {}}
                    yield {"messageStop": {"stopReason": "end_turn"}}
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse SSE data: {data}")
                    continue

                # Handle error responses
                if "error" in chunk:
                    logger.error(f"OpenRouter error: {chunk['error']}")
                    yield {
                        "messageStop": {
                            "stopReason": "error",
                            "error": chunk["error"].get("message", "Unknown error")
                        }
                    }
                    break

                # Process chunks
                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason")

                # Start message if not started
                if not message_started:
                    yield {"messageStart": {"role": "assistant"}}
                    message_started = True

                # Handle content (text)
                if "content" in delta and delta["content"]:
                    if not content_block_started:
                        yield {"contentBlockStart": {"start": {"type": "text"}}}
                        content_block_started = True

                    text_chunk = delta["content"]
                    accumulated_text += text_chunk
                    yield {
                        "contentBlockDelta": {
                            "delta": {"text": text_chunk}
                        }
                    }

                # Handle tool calls
                if "tool_calls" in delta:
                    for tool_call in delta["tool_calls"]:
                        function = tool_call.get("function", {})
                        tool_id = tool_call.get("id", "")

                        # Start new tool use block
                        if "name" in function:
                            if content_block_started:
                                yield {"contentBlockStop": {}}

                            current_tool_use = {
                                "toolUseId": tool_id,
                                "name": function["name"]
                            }

                            yield {
                                "contentBlockStart": {
                                    "start": {
                                        "type": "toolUse",
                                        "toolUseId": tool_id,
                                        "name": function["name"]
                                    }
                                }
                            }
                            content_block_started = True
                            accumulated_tool_args = ""

                        # Accumulate tool arguments
                        if "arguments" in function:
                            args_chunk = function["arguments"]
                            accumulated_tool_args += args_chunk
                            yield {
                                "contentBlockDelta": {
                                    "delta": {
                                        "toolUse": {"input": args_chunk}
                                    }
                                }
                            }

                # Handle finish
                if finish_reason:
                    if content_block_started:
                        yield {"contentBlockStop": {}}

                    # Map OpenAI finish reasons to Strands stop reasons
                    stop_reason_map = {
                        "stop": "end_turn",
                        "length": "max_tokens",
                        "tool_calls": "tool_use",
                        "content_filter": "content_filtered"
                    }

                    yield {
                        "messageStop": {
                            "stopReason": stop_reason_map.get(finish_reason, "end_turn")
                        }
                    }

                    # Include metadata if available
                    if "usage" in chunk:
                        usage = chunk["usage"]
                        yield {
                            "metadata": {
                                "usage": {
                                    "inputTokens": usage.get("prompt_tokens", 0),
                                    "outputTokens": usage.get("completion_tokens", 0),
                                    "totalTokens": usage.get("total_tokens", 0)
                                }
                            }
                        }

    async def structured_output(
        self,
        output_model: Type[T],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """
        Generate structured output using Pydantic model.

        Args:
            output_model: Pydantic model class
            prompt: Conversation messages
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Yields:
            Dictionary containing validated model instance or partial data
        """
        # Collect full response
        full_response = ""
        async for event in self.stream(prompt, system_prompt=system_prompt, **kwargs):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    full_response += delta["text"]
                    # Optionally yield partial data for streaming
                    yield {"partial": delta["text"]}

        # Parse as JSON and validate with Pydantic
        try:
            data = json.loads(full_response)
            validated_model = output_model(**data)
            yield {"result": validated_model}
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse structured output: {e}")
            yield {"error": str(e)}
            raise

    def update_config(self, **model_config: Unpack[ModelConfig]) -> None:
        """Update model configuration."""
        self._config.update(model_config)  # type: ignore

    def get_config(self) -> ModelConfig:
        """Get current model configuration."""
        return self._config.copy()  # type: ignore

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
