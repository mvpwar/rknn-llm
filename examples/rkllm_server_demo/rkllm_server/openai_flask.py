import argparse
import ctypes
import json
import os
import re
import resource
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, request, stream_with_context

app = Flask(__name__)

# ---------------------------------------------------------------------------
# RKLLM ctypes bindings
# ---------------------------------------------------------------------------

rkllm_lib = ctypes.CDLL("lib/librkllmrt.so")

RKLLM_Handle_t = ctypes.c_void_p
LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL = 0
LLMCallState.RKLLM_RUN_WAITING = 1
LLMCallState.RKLLM_RUN_FINISH = 2
LLMCallState.RKLLM_RUN_ERROR = 3

RKLLMInputType = ctypes.c_int
RKLLMInputType.RKLLM_INPUT_PROMPT = 0
RKLLMInputType.RKLLM_INPUT_TOKEN = 1
RKLLMInputType.RKLLM_INPUT_EMBED = 2
RKLLMInputType.RKLLM_INPUT_MULTIMODAL = 3

RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0
RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLMInferMode.RKLLM_INFER_GET_LOGITS = 2


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("ignore_eos_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p)]


class RKLLMInput(ctypes.Structure):
    _anonymous_ = ("input_data",)
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", RKLLMInputType),
        ("input_data", RKLLMInputUnion),
    ]


class RKLLMSamplingParam(ctypes.Structure):
    _fields_ = [
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("sampling_params", ctypes.POINTER(RKLLMSamplingParam)),
        ("keep_history", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int32),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", ctypes.c_byte * 16),
        ("logits", ctypes.c_byte * 16),
        ("perf", ctypes.c_byte * 20),
    ]


# ---------------------------------------------------------------------------
# Callback + global buffers
# ---------------------------------------------------------------------------

GLOBAL_TEXT: List[str] = []
GLOBAL_STATE = -1


def callback_impl(result, userdata, state):
    global GLOBAL_TEXT, GLOBAL_STATE
    GLOBAL_STATE = state
    if state == LLMCallState.RKLLM_RUN_NORMAL and result and result.contents.text:
        try:
            text = result.contents.text.decode("utf-8", errors="replace")
            GLOBAL_TEXT.append(text)
        except Exception:
            pass
    return 0


LLMResultCallback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
LLMTokenizerCallback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32)
LLMGetEmbedCallback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64)


class RKLLMCallback(ctypes.Structure):
    _fields_ = [
        ("result_callback", LLMResultCallback_type),
        ("result_userdata", ctypes.c_void_p),
        ("tokenizer_callback", LLMTokenizerCallback_type),
        ("tokenizer_userdata", ctypes.c_void_p),
        ("embed_callback", LLMGetEmbedCallback_type),
        ("embed_userdata", ctypes.c_void_p),
    ]


CALLBACK = LLMResultCallback_type(callback_impl)
MODEL_LOCK = threading.Lock()


class RKLLMModel:
    def __init__(self, model_path: str, platform: str):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path).rsplit(".", 1)[0]

        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), ctypes.POINTER(RKLLMCallback)]
        self.rkllm_init.restype = ctypes.c_int

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int

        self.rkllm_set_function_tools = rkllm_lib.rkllm_set_function_tools
        self.rkllm_set_function_tools.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.rkllm_set_function_tools.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

        p = RKLLMParam()
        p.model_path = model_path.encode("utf-8")
        p.max_context_len = 4096
        p.max_new_tokens = 4096
        p.n_keep = -1
        p.top_k = 1
        p.top_p = 0.9
        p.temperature = 0.8
        p.repeat_penalty = 1.1
        p.frequency_penalty = 0.0
        p.presence_penalty = 0.0
        p.skip_special_token = True
        p.ignore_eos_token = False
        p.is_async = False
        p.extend_param.base_domain_id = 0
        p.extend_param.embed_flash = 1
        p.extend_param.n_batch = 1
        p.extend_param.use_cross_attn = 0
        p.extend_param.enabled_cpus_num = 4
        if platform.lower() in ("rk3576", "rk3588"):
            p.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
        else:
            p.extend_param.enabled_cpus_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)

        self.handle = RKLLM_Handle_t()
        self.callback = RKLLMCallback(
            CALLBACK,
            None,
            LLMTokenizerCallback_type(),
            None,
            LLMGetEmbedCallback_type(),
            None,
        )

        ret = self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(p), ctypes.byref(self.callback))
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed with code {ret}")

        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(self.rkllm_infer_params))
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self.rkllm_infer_params.keep_history = 0

        self.current_tools = None

    def set_function_tools(self, system_prompt: str, tools: List[Dict[str, Any]]):
        raw = json.dumps(tools, ensure_ascii=False).encode("utf-8")
        if self.current_tools != tools:
            self.current_tools = tools
            self.rkllm_set_function_tools(self.handle, system_prompt.encode("utf-8"), raw, b"tool_response")

    def run(self, prompt: str, role: str = "user", enable_thinking: bool = False, sampling: Optional[RKLLMSamplingParam] = None, max_new_tokens: Optional[int] = None):
        rkllm_input = RKLLMInput()
        rkllm_input.role = (role or "user").encode("utf-8")
        rkllm_input.enable_thinking = ctypes.c_bool(bool(enable_thinking))
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
        rkllm_input.prompt_input = prompt.encode("utf-8")

        if sampling is not None:
            self.rkllm_infer_params.sampling_params = ctypes.pointer(sampling)
        if max_new_tokens is not None:
            self.rkllm_infer_params.max_new_tokens = int(max_new_tokens)

        rc = self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)

        if sampling is not None:
            self.rkllm_infer_params.sampling_params = None
        if max_new_tokens is not None:
            self.rkllm_infer_params.max_new_tokens = 0

        if rc != 0:
            raise RuntimeError(f"rkllm_run failed with code {rc}")

    def release(self):
        self.rkllm_destroy(self.handle)


# ---------------------------------------------------------------------------
# OpenAI-compatible helpers
# ---------------------------------------------------------------------------

def json_dumps(value):
    return json.dumps(value, ensure_ascii=False)


def content_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        text_parts = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return " ".join(text_parts)
    return ""


def error_response(message: str, error_type: str = "invalid_request_error", status_code: int = 400):
    return jsonify({"error": {"message": message, "type": error_type, "param": None, "code": None}}), status_code


def build_openai_response(model_name: str, message: Dict[str, Any], finish_reason: str = "stop", usage: Optional[Dict[str, int]] = None):
    if usage is None:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish_reason}],
        "usage": usage,
    }


def build_stream_chunk(model_name: str, delta: Dict[str, Any], finish_reason: Optional[str] = None, index: int = 0):
    chunk = {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": index, "delta": delta, "logprobs": None, "finish_reason": finish_reason}],
    }
    return "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"


def build_model_prompt(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None):
    """Convert OpenAI-style messages into a plain-text prompt for the model."""
    lines = [
        "You are a helpful assistant.",
        "If a tool call is needed, return a JSON object with a 'tool_calls' array. Example:",
        '{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":{"city":"Beijing"}}}]}',
        "Do not wrap the result in Markdown.",
    ]
    if tools:
        lines.append("Available tools:\n" + json.dumps(tools, ensure_ascii=False))

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        if role == "assistant" and msg.get("tool_calls"):
            v = json.dumps(msg["tool_calls"], ensure_ascii=False)
        elif role == "tool":
            payload = {
                "tool_call_id": msg.get("tool_call_id"),
                "name": msg.get("name"),
                "content": content_text(msg.get("content", "")),
            }
            v = json.dumps(payload, ensure_ascii=False)
        else:
            v = content_text(msg.get("content", ""))
        lines.append(f"{role}: {v}")

    lines.append("assistant:")
    return "\n".join(lines)


def extract_tool_calls(raw_text: str):
    """Parse raw model output into OpenAI tool_calls structure."""
    candidates: List[str] = []

    # Normalize common output templates.
    for pattern in (
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        r"<\|tool_call\|>\s*(.*?)\s*<\|tool_call_end\|>",
    ):
        candidates.extend(re.findall(pattern, raw_text, flags=re.S))

    if not candidates:
        candidates.append(raw_text.strip())

    result: List[Dict[str, Any]] = []
    seen = set()

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError):
            continue

        if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
            items = value["tool_calls"]
            for item in items:
                if isinstance(item, dict):
                    result.append(item)
            continue

        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    result.append(item)
            continue

        if isinstance(value, dict):
            result.append(value)

    normalized: List[Dict[str, Any]] = []
    for item in result:
        fn = item.get("function", item)
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        arguments = fn.get("arguments", {})
        if isinstance(arguments, (dict, list)):
            args_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        else:
            args_json = str(arguments)
        call_id = item.get("id") or "call_" + uuid.uuid4().hex
        key = (call_id, name, args_json)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": args_json},
        })

    return normalized


def parse_sampling_params(data: Dict[str, Any]):
    sampling = RKLLMSamplingParam()
    sampling.top_k = int(data.get("top_k", 1))
    sampling.top_p = float(data.get("top_p", 0.9))
    sampling.temperature = float(data.get("temperature", 0.8))
    sampling.repeat_penalty = float(data.get("repeat_penalty", 1.1))
    sampling.frequency_penalty = float(data.get("frequency_penalty", 0.0))
    sampling.presence_penalty = float(data.get("presence_penalty", 0.0))
    sampling.mirostat = 0
    sampling.mirostat_tau = 5.0
    sampling.mirostat_eta = 0.1
    return sampling


def run_inference(model: RKLLMModel, prompt: str, role: str = "user", enable_thinking: bool = False, sampling: Optional[RKLLMSamplingParam] = None, max_new_tokens: Optional[int] = None):
    global GLOBAL_TEXT, GLOBAL_STATE
    GLOBAL_TEXT = []
    GLOBAL_STATE = -1

    def worker():
        model.run(prompt, role, enable_thinking, sampling, max_new_tokens)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    chunks: List[str] = []
    while thread.is_alive() or GLOBAL_TEXT:
        while GLOBAL_TEXT:
            chunks.append(GLOBAL_TEXT.pop(0))
        thread.join(timeout=0.01)
    return "".join(chunks)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [{
            "id": rkllm_model.model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "rkllm",
        }],
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error_response("Request body must be JSON.")
    if "messages" not in data or not isinstance(data["messages"], list):
        return error_response("Missing 'messages' array.")

    messages = data["messages"]
    model_name = data.get("model", rkllm_model.model_name)
    stream = bool(data.get("stream", False))
    tools = data.get("tools") or []
    tool_choice = data.get("tool_choice", "auto")
    max_tokens = int(data.get("max_tokens", 4096))
    temperature = float(data.get("temperature", 0.8))
    top_p = float(data.get("top_p", 0.9))
    top_k = int(data.get("top_k", 1))
    enable_thinking = bool(data.get("enable_thinking", False))

    if tools and not isinstance(tools, list):
        return error_response("'tools' must be an array.")

    system_prompt = ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            system_prompt = content_text(msg.get("content", ""))
            break

    with MODEL_LOCK:
        if tools and tool_choice != "none":
            rkllm_model.set_function_tools(system_prompt, tools)

        prompt = build_model_prompt(messages, tools if tool_choice != "none" else [])
        sampling = parse_sampling_params({
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "repeat_penalty": data.get("repeat_penalty", 1.1),
            "frequency_penalty": data.get("frequency_penalty", 0.0),
            "presence_penalty": data.get("presence_penalty", 0.0),
        })
        output = run_inference(rkllm_model, prompt, "user", enable_thinking, sampling, max_tokens)

    tool_calls = extract_tool_calls(output) if tools and tool_choice != "none" else []
    if tool_choice == "required" and not tool_calls:
        return error_response("The model did not produce the required tool call.", "server_error", 500)

    prompt_tokens = max(1, len(prompt.split()))
    completion_tokens = max(1, len(output.split()))
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}

    if not stream:
        assistant_message = {"role": "assistant", "content": None if tool_calls else output}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        response = build_openai_response(
            model_name=model_name,
            message=assistant_message,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage=usage,
        )
        return jsonify(response)

    def generate_stream():
        base_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        initial_delta = {"role": "assistant"}
        yield build_stream_chunk(model_name, initial_delta, None)

        if tool_calls:
            for index, call in enumerate(tool_calls):
                yield build_stream_chunk(
                    model_name,
                    {"tool_calls": [{"index": index, "id": call["id"], "type": "function", "function": {"name": call["function"]["name"], "arguments": ""}}]},
                    None,
                    index=index,
                )
                yield build_stream_chunk(
                    model_name,
                    {"tool_calls": [{"index": index, "function": {"arguments": call["function"]["arguments"]}}]},
                    None,
                    index=index,
                )
            finish_reason = "tool_calls"
        else:
            # Split text into sensible chunks for OpenAI-compatible streaming.
            for piece in re.findall(r".{1,64}", output, flags=re.S):
                yield build_stream_chunk(model_name, {"content": piece}, None)
            finish_reason = "stop"

        final_chunk = {
            "content": "",
            "role": "assistant",
        }
        yield build_stream_chunk(model_name, final_chunk, finish_reason)
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate_stream()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI-compatible RKLLM Flask server")
    parser.add_argument("--rkllm_model_path", type=str, required=True, help="Absolute path to the RKLLM model on device")
    parser.add_argument("--target_platform", type=str, required=True, help="rk3588 / rk3576 / rv1126b / rk3562")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if not os.path.exists(args.rkllm_model_path):
        raise SystemExit("Invalid rkllm_model_path")
    if args.target_platform not in ("rk3588", "rk3576", "rv1126b", "rk3562"):
        raise SystemExit("Invalid target_platform")

    subprocess.run(f"sudo bash fix_freq_{args.target_platform}.sh", shell=True)
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    rkllm_model = RKLLMModel(args.rkllm_model_path, args.target_platform)

    def shutdown(signum, frame):
        try:
            rkllm_model.release()
        except Exception:
            pass
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        app.run(host=args.host, port=args.port, threaded=False, debug=False)
    finally:
        try:
            rkllm_model.release()
        except Exception:
            pass
