import argparse
import ctypes
import json
import logging
import os
import re
import resource
import signal
import subprocess
import threading
import time
import uuid
from flask import Flask, Response, jsonify, request, stream_with_context

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rkllm-server")
rkllm_lib = ctypes.CDLL("lib/librkllmrt.so")

MAX_CONTEXT_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 512
MAX_MAX_NEW_TOKENS = 1024
GENERATION_TIMEOUT_SECONDS = 120
DISABLE_THINKING = True
PROMPT_BUDGET = max(1024, MAX_CONTEXT_TOKENS - DEFAULT_MAX_NEW_TOKENS)

RKLLM_Handle_t = ctypes.c_void_p
LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL = 0
LLMCallState.RKLLM_RUN_WAITING = 1
LLMCallState.RKLLM_RUN_FINISH = 2
LLMCallState.RKLLM_RUN_ERROR = 3
RKLLMInputType = ctypes.c_int
RKLLMInputType.RKLLM_INPUT_PROMPT = 0
RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32), ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8), ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8), ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p), ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32), ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32), ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float), ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float), ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32), ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float), ("skip_special_token", ctypes.c_bool),
        ("ignore_eos_token", ctypes.c_bool), ("is_async", ctypes.c_bool),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMSamplingParam(ctypes.Structure):
    _fields_ = [
        ("top_k", ctypes.c_int32), ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float), ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float), ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32), ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p)]


class RKLLMInput(ctypes.Structure):
    _anonymous_ = ("input_data",)
    _fields_ = [
        ("role", ctypes.c_char_p), ("enable_thinking", ctypes.c_bool),
        ("input_type", RKLLMInputType), ("input_data", RKLLMInputUnion),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode), ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("sampling_params", ctypes.POINTER(RKLLMSamplingParam)),
        ("keep_history", ctypes.c_int), ("max_new_tokens", ctypes.c_int32),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p), ("token_id", ctypes.c_int),
        ("last_hidden_layer", ctypes.c_byte * 16), ("logits", ctypes.c_byte * 16),
        ("perf", ctypes.c_byte * 20),
    ]


lock = threading.Lock()
callback_lock = threading.Lock()
global_text = []
global_state = -1


def callback_impl(result, userdata, state):
    global global_state
    with callback_lock:
        global_state = state
        if state == LLMCallState.RKLLM_RUN_NORMAL and result and result.contents.text:
            global_text.append(result.contents.text.decode("utf-8", errors="replace"))
    return 0


LLMResultCallback_type = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int
)
LLMTokenizerCallback_type = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
)
LLMGetEmbedCallback_type = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
    ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64,
)


class RKLLMCallback(ctypes.Structure):
    _fields_ = [
        ("result_callback", LLMResultCallback_type), ("result_userdata", ctypes.c_void_p),
        ("tokenizer_callback", LLMTokenizerCallback_type), ("tokenizer_userdata", ctypes.c_void_p),
        ("embed_callback", LLMGetEmbedCallback_type), ("embed_userdata", ctypes.c_void_p),
    ]


_callback = LLMResultCallback_type(callback_impl)


class RKLLM:
    def __init__(self, model_path, platform):
        p = RKLLMParam()
        p.model_path = model_path.encode()
        p.max_context_len = MAX_CONTEXT_TOKENS
        p.max_new_tokens = MAX_MAX_NEW_TOKENS
        p.n_keep, p.top_k, p.top_p, p.temperature, p.repeat_penalty = -1, 1, .9, .8, 1.1
        p.skip_special_token = True
        p.extend_param.embed_flash, p.extend_param.n_batch = 1, 1
        p.extend_param.enabled_cpus_num = 4
        p.extend_param.enabled_cpus_mask = 0xf0 if platform in ("rk3588", "rk3576") else 0x0f
        self.handle = RKLLM_Handle_t()
        self.callback = RKLLMCallback(_callback, None, LLMTokenizerCallback_type(), None, LLMGetEmbedCallback_type(), None)
        self.init = rkllm_lib.rkllm_init
        self.init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), ctypes.POINTER(RKLLMCallback)]
        self.init.restype = ctypes.c_int
        if self.init(ctypes.byref(self.handle), ctypes.byref(p), ctypes.byref(self.callback)) != 0:
            raise RuntimeError("rkllm_init failed")
        self.run_fn = rkllm_lib.rkllm_run
        self.run_fn.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.run_fn.restype = ctypes.c_int
        self.abort_fn = rkllm_lib.rkllm_abort
        self.abort_fn.argtypes = [RKLLM_Handle_t]
        self.abort_fn.restype = ctypes.c_int
        self.set_tools_fn = rkllm_lib.rkllm_set_function_tools
        self.set_tools_fn.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.set_tools_fn.restype = ctypes.c_int
        self.destroy = rkllm_lib.rkllm_destroy
        self.destroy.argtypes = [RKLLM_Handle_t]
        self.model_name = os.path.basename(model_path).rsplit('.', 1)[0]
        self.infer = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.infer), 0, ctypes.sizeof(self.infer))
        self.infer.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self.infer.keep_history = 0
        self._tools_key = None

    def configure_tools(self, system, tools):
        raw = json.dumps(tools, ensure_ascii=False, separators=(",", ":")).encode()
        system_raw = system.encode()
        key = system_raw + b"\0" + raw
        if key == self._tools_key:
            log.info("Function tools unchanged; skipping reconfiguration")
            return 0
        rc = self.set_tools_fn(self.handle, system_raw, raw, b"tool_response")
        if rc == 0:
            self._tools_key = key
        return rc

    def abort(self):
        rc = self.abort_fn(self.handle)
        log.warning("rkllm_abort returned %s", rc)
        return rc

    def run(self, prompt, role, thinking, sampling, max_tokens):
        inp = RKLLMInput()
        inp.role = (role or "user").encode()
        inp.enable_thinking = bool(thinking)
        inp.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
        inp.prompt_input = prompt.encode()
        self.infer.sampling_params = ctypes.pointer(sampling) if sampling else None
        self.infer.max_new_tokens = int(max_tokens or 0)
        try:
            rc = self.run_fn(self.handle, ctypes.byref(inp), ctypes.byref(self.infer), None)
            if rc != 0:
                raise RuntimeError("rkllm_run failed: %s" % rc)
        finally:
            self.infer.sampling_params = None
            self.infer.max_new_tokens = 0

    def release(self):
        self.destroy(self.handle)


def text_content(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(str(x.get("text", "")) for x in value if isinstance(x, dict) and x.get("type") == "text").strip()
    return ""


def compact_tool_call(call):
    if not isinstance(call, dict):
        return call
    fn = call.get("function", call)
    if not isinstance(fn, dict):
        return call
    args = fn.get("arguments", {})
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    return {"name": fn.get("name", ""), "arguments": args}


def compact_message(message):
    role = message.get("role", "user")
    if role == "assistant" and message.get("tool_calls"):
        return {"role": "assistant", "tool_calls": [compact_tool_call(x) for x in message["tool_calls"]]}
    if role == "tool":
        return {"role": "tool", "content": text_content(message.get("content", ""))}
    return {"role": role, "content": text_content(message.get("content", ""))}


def message_value(message):
    if message.get("role") == "assistant" and message.get("tool_calls"):
        return json.dumps(message["tool_calls"], ensure_ascii=False, separators=(",", ":"))
    return text_content(message.get("content", ""))


def prompt_from_messages(messages):
    lines = []
    for message in messages:
        if isinstance(message, dict):
            lines.append("{}: {}".format(message.get("role", "user"), message_value(message)))
    lines.append("assistant:")
    return "\n".join(lines)


def token_estimate(text):
    return max(1, int(sum(1 if ord(c) >= 0x2E80 else 0.25 for c in text))) if text else 0


def compact_tools(tools):
    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        item = {"name": fn["name"]}
        if fn.get("description"):
            item["description"] = fn["description"]
        if fn.get("parameters"):
            item["parameters"] = fn["parameters"]
        result.append({"type": "function", "function": item})
    return result


def semantic_compact(messages, tools, budget=None):
    if budget is None:
        budget = PROMPT_BUDGET

    valid = [compact_message(m) for m in messages if isinstance(m, dict)]
    system = next((m for m in valid if m.get("role") == "system"), None)
    rest = [m for m in valid if m.get("role") != "system"]
    compacted_tools = compact_tools(tools)
    native_system = bool(compacted_tools and system)
    keep = list(rest if native_system else (([system] if system else []) + rest))

    def cost(items):
        prompt_cost = token_estimate(prompt_from_messages(items))
        tools_cost = token_estimate(json.dumps(compacted_tools, ensure_ascii=False, separators=(",", ":")))
        system_cost = 0
        if native_system and system:
            system_cost = token_estimate(text_content(system.get("content", "")))
        return prompt_cost + tools_cost + system_cost

    latest_user_index = -1
    for i, msg in enumerate(keep):
        if msg.get("role") == "user":
            latest_user_index = i

    protected_from = latest_user_index if latest_user_index >= 0 else len(keep)

    def tool_call_block_end(items, start):
        if start >= len(items):
            return start
        if items[start].get("role") == "assistant" and items[start].get("tool_calls"):
            end = start + 1
            while end < len(items) and items[end].get("role") == "tool":
                end += 1
            return end
        return start + 1

    removed = 0
    while cost(keep) > budget and len(keep) > 1:
        deleted_any = False
        for i in range(min(len(keep), protected_from)):
            if keep[i].get("role") == "assistant" and keep[i].get("tool_calls"):
                end = tool_call_block_end(keep, i)
                if end > i:
                    del keep[i:end]
                    removed += end - i
                    deleted_any = True
                    break
            del keep[i]
            removed += 1
            deleted_any = True
            break
        if not deleted_any:
            raise ValueError("prompt remains too large after safe compaction; reduce tools, system prompt, or max_tokens")

    if removed:
        log.info("semantic compaction removed %d old history messages; latest request and tool results preserved", removed)
    return keep, compacted_tools, system, native_system


def diagnostic_breakdown(messages, tools, system, native_system):
    prompt = prompt_from_messages(messages)
    tool_text = json.dumps(tools, ensure_ascii=False, separators=(",", ":")) if tools else ""
    system_text = text_content(system.get("content", "")) if system and native_system else ""
    parts = [(i, m.get("role", "user"), "{}: {}\n".format(m.get("role", "user"), message_value(m))) for i, m in enumerate(messages)]
    if system_text:
        parts.append(("native", "system", system_text))
    total = token_estimate(prompt) + token_estimate(tool_text) + token_estimate(system_text)
    log.info("prompt breakdown: messages=%d chars=%d estimated_tokens=%d tools_chars=%d tools_estimated_tokens=%d native_system_injected=%s",
             len(messages), len(prompt), token_estimate(prompt), len(tool_text), token_estimate(tool_text), native_system)
    for index, role, text in parts:
        log.info("prompt part index=%s role=%s chars=%d estimated_tokens=%d ratio=%.1f%%",
                 index, role, len(text), token_estimate(text), token_estimate(text) * 100 / max(total, 1))


def extract_tool_calls(raw):
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.S)
    if not blocks:
        blocks = re.findall(r"<\|tool_call\|>\s*(.*?)\s*<\|tool_call_end\|>", raw, re.S)
    values = []
    for block in blocks or [raw.strip()]:
        try:
            value = json.loads(block)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
            values.extend(value["tool_calls"])
        elif isinstance(value, list):
            values.extend(value)
        elif isinstance(value, dict):
            values.append(value)
    calls = []
    for value in values:
        if not isinstance(value, dict):
            continue
        fn = value.get("function", value)
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        args = fn.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        calls.append({"id": value.get("id") or "call_" + uuid.uuid4().hex,
                      "type": "function", "function": {"name": fn["name"], "arguments": args}})
    return calls


def completion_id():
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def sse(obj):
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def error(message, typ="invalid_request_error", code=400):
    return jsonify({"error": {"message": message, "type": typ, "param": None, "code": None}}), code


def run_model(model, prompt, role, thinking, sampling, max_tokens, timeout=GENERATION_TIMEOUT_SECONDS):
    global global_text, global_state
    global_text, global_state = [], -1
    worker_error = []

    def worker():
        try:
            model.run(prompt, role, thinking, sampling, max_tokens)
        except BaseException as exc:
            worker_error.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    result = []
    started = time.monotonic()
    while thread.is_alive() or global_text:
        while global_text:
            result.append(global_text.pop(0))
        if thread.is_alive() and time.monotonic() - started > timeout:
            log.error("RKLLM generation timeout after %ss; aborting", timeout)
            model.abort()
            thread.join(5)
            raise TimeoutError("RKLLM generation timed out after %s seconds" % timeout)
        thread.join(.01)
    if worker_error:
        raise worker_error[0]
    if global_state == LLMCallState.RKLLM_RUN_ERROR:
        raise RuntimeError("rkllm_run failed: RKLLM_RUN_ERROR")
    return "".join(result)


def make_sampling(data):
    p = RKLLMSamplingParam()
    p.top_k = int(data.get("top_k", 1))
    p.top_p = float(data.get("top_p", .9))
    p.temperature = float(data.get("temperature", .8))
    p.repeat_penalty = float(data.get("repeat_penalty", 1.1))
    p.frequency_penalty = float(data.get("frequency_penalty", 0.0))
    p.presence_penalty = float(data.get("presence_penalty", 0.0))
    p.mirostat = int(data.get("mirostat", 0))
    p.mirostat_tau = float(data.get("mirostat_tau", 5.0))
    p.mirostat_eta = float(data.get("mirostat_eta", 0.1))
    return p


@app.route('/v1/models', methods=['GET'])
def models():
    return jsonify({"object": "list", "data": [{"id": rkllm_model.model_name, "object": "model", "created": int(time.time()), "owned_by": "rkllm"}]})


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    request_id = uuid.uuid4().hex[:8]
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        return error("'messages' must be an array")
    raw_messages, tools = data["messages"], data.get("tools") or []
    if tools and not isinstance(tools, list):
        return error("'tools' must be an array")
    choice = data.get("tool_choice", "auto")
    tools_for_model = [] if choice == "none" else tools
    log.info("request start id=%s stream=%s messages=%d tools=%d", request_id, data.get("stream", False), len(raw_messages), len(tools_for_model))
    try:
        messages, compacted_tools, system, native_system = semantic_compact(raw_messages, tools_for_model, budget=PROMPT_BUDGET)
        diagnostic_breakdown(messages, compacted_tools, system, native_system)
        prompt = prompt_from_messages(messages)
        requested = int(data.get("max_tokens", DEFAULT_MAX_NEW_TOKENS))
        max_tokens = max(1, min(requested, MAX_MAX_NEW_TOKENS))
        with lock:
            if compacted_tools:
                system_text = text_content(system.get("content", "")) if system else ""
                rc = rkllm_model.configure_tools(system_text, compacted_tools)
                if rc != 0:
                    raise RuntimeError("rkllm_set_function_tools failed: %s" % rc)
            raw = run_model(rkllm_model, prompt, "user", False if DISABLE_THINKING else bool(data.get("enable_thinking", False)), make_sampling(data), max_tokens)
    except Exception as exc:
        log.exception("inference failed request_id=%s", request_id)
        return error(str(exc), "server_error", 500)
    calls = extract_tool_calls(raw) if tools_for_model and choice != "none" else []
    if choice == "required" and not calls:
        return error("The model did not return a tool call", "server_error", 500)
    ident, created, model = completion_id(), int(time.time()), data.get("model", rkllm_model.model_name)
    finish = "tool_calls" if calls else "stop"
    message = {"role": "assistant", "content": None if calls else raw}
    if calls:
        message["tool_calls"] = calls
    usage = {"prompt_tokens": token_estimate(prompt), "completion_tokens": len(raw.split())}
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    log.info("request finish id=%s output_chars=%d tool_calls=%d", request_id, len(raw), len(calls))
    if not data.get("stream", False):
        return jsonify({"id": ident, "object": "chat.completion", "created": created, "model": model,
                        "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish}], "usage": usage})

    def stream_response():
        base = {"id": ident, "object": "chat.completion.chunk", "created": created, "model": model}
        yield sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "logprobs": None, "finish_reason": None}]})
        if calls:
            for i, call in enumerate(calls):
                delta = {"tool_calls": [{"index": i, "id": call["id"], "type": "function", "function": {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}}]}
                yield sse({**base, "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": None}]})
        else:
            for part in re.findall(r".{1,64}", raw, re.S):
                yield sse({**base, "choices": [{"index": 0, "delta": {"content": part}, "logprobs": None, "finish_reason": None}]})
        yield sse({**base, "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": finish}], "usage": usage})
        yield "data: [DONE]\n\n"
    return Response(stream_with_context(stream_response()), content_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rkllm_model_path', required=True)
    ap.add_argument('--target_platform', required=True)
    args = ap.parse_args()
    if not os.path.exists(args.rkllm_model_path):
        raise SystemExit("invalid model path")
    if args.target_platform not in ("rk3588", "rk3576", "rv1126b", "rk3562"):
        raise SystemExit("invalid target platform")
    subprocess.run("sudo bash fix_freq_{}.sh".format(args.target_platform), shell=True)
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))
    rkllm_model = RKLLM(args.rkllm_model_path, args.target_platform)

    def shutdown(signum, frame):
        rkllm_model.release()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        app.run(host='0.0.0.0', port=8080, threaded=False, debug=False)
    finally:
        rkllm_model.release()
