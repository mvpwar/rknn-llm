"""OpenAI tool-calling adapter for the RKLLM demo.

This module is intentionally dependency-free. It exposes the standard
OpenAI response shape and leaves tool execution to the caller (for example,
claw). The caller must send the returned assistant tool_calls and subsequent
role=tool messages back in the next request.
"""
import json
import re
import time
import uuid


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(x.get("text", "")) for x in content
            if isinstance(x, dict) and x.get("type") == "text"
        )
    return ""


def _extract_tool_call(text):
    """Parse common RKLLM tool-call formats into OpenAI's representation."""
    candidates = []
    for pattern in (
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        r"<|tool_call|>\s*(.*?)\s*<\|tool_call_end\|>",
    ):
        candidates.extend(re.findall(pattern, text, flags=re.S))
    candidates.append(text.strip())
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
            value = value["tool_calls"][0] if value["tool_calls"] else None
        if not isinstance(value, dict):
            continue
        function = value.get("function", value)
        name = function.get("name") if isinstance(function, dict) else None
        if not name:
            continue
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return {
            "id": value.get("id", "call_" + uuid.uuid4().hex),
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
    return None


def build_prompt(messages, tools):
    """Serialize the complete OpenAI conversation and tools for a text model."""
    lines = [
        "You are an assistant. If a tool is needed, output exactly:",
        '<tool_call>{"name":"function_name","arguments":{}}</tool_call>',
        "Do not add markdown around a tool call.",
    ]
    if tools:
        lines.append("Available tools: " + json.dumps(tools, ensure_ascii=False))
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "user")
        if role == "assistant" and message.get("tool_calls"):
            content = json.dumps(message["tool_calls"], ensure_ascii=False)
        elif role == "tool":
            content = json.dumps(message.get("content", ""), ensure_ascii=False)
        else:
            content = _text(message.get("content", ""))
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


def completion_response(messages, tools, generate, model="rkllm", stream=False):
    """Generate a completion using ``generate(prompt)`` and return OpenAI JSON.

    ``generate`` must return either a string or an iterable of text chunks.
    No tool is executed here; the returned tool_calls are for claw/client use.
    """
    prompt = build_prompt(messages, tools)
    result = generate(prompt)
    text = result if isinstance(result, str) else "".join(result)
    tool_call = _extract_tool_call(text) if tools else None
    message = {"role": "assistant", "content": None if tool_call else text}
    if tool_call:
        message["tool_calls"] = [tool_call]
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_call else "stop",
        }],
    }
