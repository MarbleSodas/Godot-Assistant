"""
OpenRouter Custom Model Provider for Strands Agents

This module provides a custom model provider that integrates the official OpenRouter Python SDK
with the Strands Agents framework, enabling streaming, tool calling, and usage tracking.
"""

import json
import logging
import httpx
from typing import AsyncIterable, Optional, Any, Dict, List, Type, TypeVar, Union, AsyncGenerator, TypedDict, cast
from typing_extensions import Unpack

from pydantic import BaseModel
from strands.models import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec

from openrouter import OpenRouter
from openrouter import components

logger = logging.getLogger(__name__)
metric_logger = logging.getLogger("metrics")
metric_logger.setLevel(logging.INFO)

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
    Custom model provider for OpenRouter API with Strands Agents using the official OpenRouter SDK.
    
    Acts as an adapter between Strands 'Model' interface and OpenRouter SDK.

    Supports:
    - Streaming responses
    - Tool calling
    - Usage tracking (Metrics)
    - Error handling
    """

    def __init__(
        self,
        api_key: str,
        metrics_callback: Optional[Any] = None,
        **model_config: Unpack[ModelConfig]
    ) -> None:
        """
        Initialize OpenRouter model provider.

        Args:
            api_key: OpenRouter API key
            metrics_callback: Optional callback function(cost, tokens, model_name)
            **model_config: Model configuration
        """
        if not api_key or not api_key.strip():
            raise ValueError("OpenRouter API key cannot be empty.")
        
        self.api_key = api_key.strip()
        self.metrics_callback = metrics_callback

        # Store configuration with defaults
        self._config: ModelConfig = {
            "model_id": "openai/gpt-4-turbo",
            "app_name": "Godoty",
            "app_url": "http://localhost:8000",
            "timeout": 120,
            "temperature": 0.7,
            "max_tokens": 4000,
            **model_config  # type: ignore
        }

        # Initialize HTTP client with headers
        headers = {
            "HTTP-Referer": self._config.get("app_url", "http://localhost:8000"),
            "X-Title": self._config.get("app_name", "Godoty"),
        }
        
        # Initialize Async Client for the SDK
        self._async_http_client = httpx.AsyncClient(
            headers=headers,
            timeout=float(self._config.get("timeout", 120))
        )

        # Initialize OpenRouter SDK Client
        # Explicitly pass security object to ensure it is picked up
        self._client = OpenRouter(
            api_key=self.api_key,
            async_client=self._async_http_client
        )
        
        logger.info(f"OpenRouterModel initialized with model: {self._config.get('model_id')}")

    def _convert_tool_to_sdk_format(self, tool_spec: Dict) -> Dict:
        """
        Convert Strands tool spec to OpenRouter SDK tool format (standard OpenAI schema).
        """
        # Strands uses 'inputSchema' or 'input_schema' depending on version/context
        schema = tool_spec.get("inputSchema") or tool_spec.get("input_schema") or {}
        
        # Strands wraps the actual JSON schema in a 'json' key
        if "json" in schema:
            parameters = schema["json"]
        else:
            parameters = schema

        return {
            "type": "function",
            "function": {
                "name": tool_spec.get("name"),
                "description": tool_spec.get("description", ""),
                "parameters": parameters
            }
        }

    def _convert_messages_to_sdk_format(self, messages: Messages) -> List[Dict]:
        """
        Convert Strands messages format to OpenRouter SDK format.
        """
        sdk_messages = []

        for message in messages:
            role = message.get("role")
            content = message.get("content", [])

            if isinstance(content, str):
                sdk_messages.append({
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
                        tool_result = block["toolResult"]
                        sdk_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_result.get("toolUseId", ""),
                            "content": json.dumps(tool_result.get("content", []))
                        })

                # Add the main message part (text + tool_calls)
                if text_parts or tool_calls:
                    msg = {"role": role}
                    if text_parts:
                        msg["content"] = "\n".join(text_parts)
                    if tool_calls:
                        msg["tool_calls"] = tool_calls
                    
                    # Avoid adding empty assistant message if we only added tool results above
                    if role != "tool": 
                        sdk_messages.append(msg)

        return sdk_messages

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
        Stream chat completion from OpenRouter API using the SDK.
        """
        try:
            # Convert messages
            sdk_messages = self._convert_messages_to_sdk_format(messages)

            # Add system prompt
            if system_prompt:
                sdk_messages.insert(0, {
                    "role": "system",
                    "content": system_prompt
                })

            # Prepare tools
            tools = None
            if tool_specs:
                tools = [
                    self._convert_tool_to_sdk_format(spec)
                    for spec in tool_specs
                ]

            # Prepare stream options for usage tracking
            stream_opts = components.ChatStreamOptions(include_usage=True)

            # Call SDK
            stream_response = await self._client.chat.send_async(
                model=self._config.get("model_id"),
                messages=cast(Any, sdk_messages),
                stream=True,
                temperature=self._config.get("temperature"),
                max_tokens=self._config.get("max_tokens"),
                tools=cast(Any, tools) if tools else None,
                tool_choice=tool_choice if tools and tool_choice else ("auto" if tools else None),
                stream_options=stream_opts
            )
            
            # Process the stream
            async for event in self._process_sdk_stream(stream_response):
                yield event

        except Exception as e:
            logger.error(f"Error in OpenRouter SDK stream: {e}")
            yield {
                "messageStop": {
                    "stopReason": "error",
                    "error": str(e)
                }
            }

    async def _process_sdk_stream(self, stream) -> AsyncIterable[StreamEvent]:
        """
        Process OpenRouter SDK EventStream.
        """
        message_started = False
        content_block_started = False
        final_finish_reason = "end_turn"

        # Track usage data from OpenRouter
        usage_data = {}

        # Track active tool state
        active_tools = {}
        
        async for chunk in stream:
            # Check for usage data in chunk
            if chunk.usage:
                usage = chunk.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens

                # Extract actual cost from OpenRouter (in credits)
                actual_cost = getattr(usage, 'cost', None)

                # Store usage data for messageStop event
                usage_data = {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }

                # Include actual cost if available
                if actual_cost is not None:
                    usage_data["actual_cost"] = actual_cost

                # Invoke metrics callback if provided
                if self.metrics_callback:
                    try:
                        self.metrics_callback(
                            cost=actual_cost if actual_cost is not None else 0.0,
                            tokens=total_tokens,
                            model_name=self._config.get("model_id", "unknown")
                        )
                    except Exception as e:
                        logger.error(f"Error in metrics callback: {e}")

                # Log metrics including actual cost when available
                cost_str = f"Cost: ${actual_cost:.6f}" if actual_cost is not None else "Cost: N/A"
                metric_logger.info(
                    f"LLM_CALL | Model: {self._config.get('model_id')} | "
                    f"Tokens: {total_tokens} (P:{prompt_tokens}, C:{completion_tokens}) | {cost_str}"
                )
            
            if not chunk.choices:
                continue
                
            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason

            if not message_started:
                yield {"messageStart": {"role": "assistant"}}
                message_started = True
            
            if finish_reason:
                if finish_reason == "tool_calls":
                    final_finish_reason = "tool_use"
                elif finish_reason == "stop":
                    final_finish_reason = "end_turn"
                else:
                    final_finish_reason = finish_reason

            # Handle content
            if delta.content:
                if not content_block_started:
                    yield {"contentBlockStart": {"start": {}}}
                    content_block_started = True
                
                yield {
                    "contentBlockDelta": {
                        "delta": {"text": delta.content}
                    }
                }

            # Handle tool calls
            if delta.tool_calls:
                if content_block_started:
                    yield {"contentBlockStop": {}}
                    content_block_started = False
                
                for tool_call in delta.tool_calls:
                    idx = tool_call.index
                    
                    if idx not in active_tools:
                        active_tools[idx] = {"id": "", "name": "", "has_started": False}
                    
                    state = active_tools[idx]
                    
                    if tool_call.id:
                        state["id"] = tool_call.id
                    if tool_call.function and tool_call.function.name:
                        state["name"] = tool_call.function.name
                    
                    if not state["has_started"] and state["name"] and state["id"]:
                        yield {
                            "contentBlockStart": {
                                "start": {
                                    "toolUse": {
                                        "name": state["name"],
                                        "toolUseId": state["id"]
                                    }
                                }
                            }
                        }
                        state["has_started"] = True
                    
                    if state["has_started"] and tool_call.function and tool_call.function.arguments:
                        yield {
                            "contentBlockDelta": {
                                "delta": {
                                    "toolUse": {
                                        "input": tool_call.function.arguments
                                    }
                                }
                            }
                        }

            # Handle finish
            if finish_reason:
                if content_block_started:
                    yield {"contentBlockStop": {}}
                    content_block_started = False
                
                for idx, state in active_tools.items():
                    if state["has_started"]:
                        yield {"contentBlockStop": {}}
                        state["has_started"] = False

        # Include usage data in messageStop event if available
        if usage_data:
            yield {
                "messageStop": {
                    "stopReason": final_finish_reason, 
                    "usage": usage_data,
                    "model_id": self._config.get("model_id")
                },
                "model_id": self._config.get("model_id")  # Top-level as well for easy access
            }
        else:
            yield {
                "messageStop": {"stopReason": final_finish_reason},
                "model_id": self._config.get("model_id")
            }

    async def complete(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: Optional[str] = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Get a complete chat completion using streaming (to support usage metrics via SDK).
        """
        # We reuse the stream method to ensure we get usage data which is supported
        # in the SDK via stream_options. The non-streaming method doesn't support
        # 'usage' param in this SDK version without 'extra_body'.
        
        full_content = []
        current_text = ""
        tool_calls = {} # Map ID -> {name, arguments_str}
        stop_reason = "end_turn"
        final_usage = {} # Track usage from messageStop event

        async for event in self.stream(
            messages, 
            tool_specs, 
            system_prompt, 
            tool_choice=tool_choice, 
            system_prompt_content=system_prompt_content, 
            **kwargs
        ):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    current_text += delta["text"]
                elif "toolUse" in delta:
                    # Strands streaming doesn't provide ID in delta usually, 
                    # it assumes we are in a block.
                    # But we need to reconstruct the full tool call.
                    # We'll rely on the fact that Strands emits 'contentBlockStart' with ID.
                    pass
            
            elif "contentBlockStart" in event:
                start = event["contentBlockStart"]["start"]
                if "toolUse" in start:
                    t_id = start["toolUse"]["toolUseId"]
                    t_name = start["toolUse"]["name"]
                    tool_calls[t_id] = {"name": t_name, "arguments": ""}
                    # Set current tool ID for deltas (simplified logic)
                    self._current_tool_id = t_id
            
            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "toolUse" in delta and hasattr(self, '_current_tool_id'):
                    tool_calls[self._current_tool_id]["arguments"] += delta["toolUse"]["input"]

            elif "messageStop" in event:
                stop_reason = event["messageStop"]["stopReason"]
                # Extract usage data if available
                final_usage = event["messageStop"].get("usage", {})

        # Reconstruct response structure
        response_content = []
        if current_text:
            response_content.append({"text": current_text})
        
        for t_id, t_data in tool_calls.items():
            try:
                args = json.loads(t_data["arguments"])
            except:
                args = {} # Error parsing
            response_content.append({
                "toolUse": {
                    "toolUseId": t_id,
                    "name": t_data["name"],
                    "input": args
                }
            })

        return {
            "message": {
                "content": response_content,
                "role": "assistant"
            },
            "stop_reason": stop_reason,
            "usage": final_usage, # Now includes actual usage data from OpenRouter
            "id": "streaming-simulated",
            "model": self._config.get("model_id")
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
        """
        full_response = ""
        async for event in self.stream(prompt, system_prompt=system_prompt, **kwargs):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    full_response += delta["text"]
                    yield {"partial": delta["text"]}
        
        try:
            data = json.loads(full_response)
            validated_model = output_model(**data)
            yield {"result": validated_model}
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse structured output: {e}")
            yield {"error": str(e)}
            raise

    def update_config(self, **model_config: Unpack[ModelConfig]) -> None:
        self._config.update(model_config)  # type: ignore

    def get_config(self) -> ModelConfig:
        return self._config.copy()  # type: ignore

    async def close(self):
        await self._async_http_client.aclose()
