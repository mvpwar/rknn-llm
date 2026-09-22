import argparse
import ctypes
import json
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
rkllm_lib = ctypes.CDLL("lib/librkllmrt.so")
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
    _fields_ = [("base_domain_id", ctypes.c_int32), ("embed_flash", ctypes.c_int8), ("enabled_cpus_num", ctypes.c_int8), ("enabled_cpus_mask", ctypes.c_uint32), ("n_batch", ctypes.c_uint8), ("use_cross_attn", ctypes.c_int8), ("reserved", ctypes.c_uint8 * 104)]
class RKLLMParam(ctypes.Structure):
    _fields_ = [("model_path", ctypes.c_char_p), ("max_context_len", ctypes.c_int32), ("max_new_tokens", ctypes.c_int32), ("top_k", ctypes.c_int32), ("n_keep", ctypes.c_int32), ("top_p", ctypes.c_float), ("temperature", ctypes.c_float), ("repeat_penalty", ctypes.c_float), ("frequency_penalty", ctypes.c_float), ("presence_penalty", ctypes.c_float), ("mirostat", ctypes.c_int32), ("mirostat_tau", ctypes.c_float), ("mirostat_eta", ctypes.c_float), ("skip_special_token", ctypes.c_bool), ("ignore_eos_token", ctypes.c_bool), ("is_async", ctypes.c_bool), ("extend_param", RKLLMExtendParam)]
class RKLLMSamplingParam(ctypes.Structure):
    _fields_ = [("top_k", ctypes.c_int32), ("top_p", ctypes.c_float), ("temperature", ctypes.c_float), ("repeat_penalty", ctypes.c_float), ("frequency_penalty", ctypes.c_float), ("presence_penalty", ctypes.c_float), ("mirostat", ctypes.c_int32), ("mirostat_tau", ctypes.c_float), ("mirostat_eta", ctypes.c_float)]
class RKLLMInputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p)]
class RKLLMInput(ctypes.Structure):
    _anonymous_ = ("input_data",)
    _fields_ = [("role", ctypes.c_char_p), ("enable_thinking", ctypes.c_bool), ("input_type", RKLLMInputType), ("input_data", RKLLMInputUnion)]
class RKLLMInferParam(ctypes.Structure):
    _fields_ = [("mode", RKLLMInferMode), ("lora_params", ctypes.c_void_p), ("prompt_cache_params", ctypes.c_void_p), ("sampling_params", ctypes.POINTER(RKLLMSamplingParam)), ("keep_history", ctypes.c_int), ("max_new_tokens", ctypes.c_int32)]
class RKLLMResult(ctypes.Structure):
    _fields_ = [("text", ctypes.c_char_p), ("token_id", ctypes.c_int), ("last_hidden_layer", ctypes.c_byte * 16), ("logits", ctypes.c_byte * 16), ("perf", ctypes.c_byte * 20)]

lock = threading.Lock()
global_text, global_state = [], -1

def callback_impl(result, userdata, state):
    global global_state
    global_state = state
    if state == LLMCallState.RKLLM_RUN_NORMAL and result and result.contents.text:
        global_text.append(result.contents.text.decode("utf-8", errors="replace"))
    return 0

LLMResultCallback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
LLMTokenizerCallback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32)
LLMGetEmbedCallback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64)
class RKLLMCallback(ctypes.Structure):
    _fields_ = [("result_callback", LLMResultCallback_type), ("result_userdata", ctypes.c_void_p), ("tokenizer_callback", LLMTokenizerCallback_type), ("tokenizer_userdata", ctypes.c_void_p), ("embed_callback", LLMGetEmbedCallback_type), ("embed_userdata", ctypes.c_void_p)]
_callback = LLMResultCallback_type(callback_impl)

class RKLLM:
    def __init__(self, model_path, platform):
        p = RKLLMParam(); p.model_path = model_path.encode(); p.max_context_len = p.max_new_tokens = 4096
        p.n_keep, p.top_k, p.top_p, p.temperature, p.repeat_penalty = -1, 1, .9, .8, 1.1; p.skip_special_token = True
        p.extend_param.embed_flash, p.extend_param.n_batch, p.extend_param.enabled_cpus_num = 1, 1, 4
        p.extend_param.enabled_cpus_mask = 0xf0 if platform in ("rk3588", "rk3576") else 0x0f
        self.handle = RKLLM_Handle_t(); self.callback = RKLLMCallback(_callback, None, LLMTokenizerCallback_type(), None, LLMGetEmbedCallback_type(), None)
        self.init = rkllm_lib.rkllm_init; self.init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), ctypes.POINTER(RKLLMCallback)]; self.init.restype = ctypes.c_int
        if self.init(ctypes.byref(self.handle), ctypes.byref(p), ctypes.byref(self.callback)) != 0: raise RuntimeError("rkllm_init failed")
        self.run_fn = rkllm_lib.rkllm_run; self.run_fn.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]; self.run_fn.restype = ctypes.c_int
        self.clear_kv_cache_fn = rkllm_lib.rkllm_clear_kv_cache; self.clear_kv_cache_fn.argtypes = [RKLLM_Handle_t, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]; self.clear_kv_cache_fn.restype = ctypes.c_int
        self.set_tools_fn = rkllm_lib.rkllm_set_function_tools; self.set_tools_fn.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]; self.set_tools_fn.restype = ctypes.c_int
        self.destroy = rkllm_lib.rkllm_destroy; self.destroy.argtypes = [RKLLM_Handle_t]
        self.model_name = os.path.basename(model_path).rsplit('.', 1)[0]; self.infer = RKLLMInferParam(); ctypes.memset(ctypes.byref(self.infer), 0, ctypes.sizeof(self.infer)); self.infer.mode = RKLLMInferMode.RKLLM_INFER_GENERATE; self.infer.keep_history = 0
        self.clear_context()

    def clear_context(self):
        rc = self.clear_kv_cache_fn(self.handle, 1, None, None)
        if rc != 0: raise RuntimeError("rkllm_clear_kv_cache failed: %s" % rc)
    def configure_tools(self, system, tools):
        return self.set_tools_fn(self.handle, system.encode(), json.dumps(tools, ensure_ascii=False).encode(), b"tool_response")
    def run(self, prompt, role, thinking, sampling, max_tokens):
        inp = RKLLMInput(); inp.role = (role or "user").encode(); inp.enable_thinking = bool(thinking); inp.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT; inp.prompt_input = prompt.encode()
        self.infer.sampling_params = ctypes.pointer(sampling) if sampling else None; self.infer.max_new_tokens = int(max_tokens or 0)
        try:
            rc = self.run_fn(self.handle, ctypes.byref(inp), ctypes.byref(self.infer), None)
            if rc != 0: raise RuntimeError("rkllm_run failed: %s" % rc)
        finally:
            self.infer.sampling_params = None; self.infer.max_new_tokens = 0
    def release(self): self.destroy(self.handle)

def text_content(value):
    if isinstance(value, str): return value
    if isinstance(value, list): return " ".join(str(x.get("text", "")) for x in value if isinstance(x, dict) and x.get("type") == "text")
    return ""

def message_value(m):
    role = m.get("role", "user")
    if role == "assistant" and m.get("tool_calls"): return json.dumps(m["tool_calls"], ensure_ascii=False)
    if role == "tool": return json.dumps({"tool_call_id": m.get("tool_call_id"), "content": text_content(m.get("content", ""))}, ensure_ascii=False)
    return text_content(m.get("content", ""))

def prompt_from_messages(messages):
    out = [f'{m.get("role", "user")}: {message_value(m)}' for m in messages if isinstance(m, dict)]
    out.append("assistant:"); return "\n".join(out)

# RKLLM does not expose its tokenizer through this server. Use a deliberately
# conservative estimate and keep headroom below 4096, rather than passing an
# overlong prompt to the native runtime. This is needed because clearing KV
# cache cannot make an individual prompt shorter.
def prompt_token_estimate(text):
    return sum(1 if ord(c) >= 0x2E80 else 0.25 for c in text)

def limit_messages(messages, limit=3500):
    valid = [m for m in messages if isinstance(m, dict)]
    system = [m for m in valid if m.get("role") == "system"][:1]
    rest = [m for m in valid if m.get("role") != "system"]
    selected = []
    used = prompt_token_estimate(prompt_from_messages(system + [{"role": "user", "content": ""}])) if system else 0
    for m in reversed(rest):
        cost = prompt_token_estimate(f'{m.get("role", "user")}: {message_value(m)}\n')
        if selected and used + cost > limit: break
        if not selected and used + cost > limit:
            value = message_value(m)
            # Preserve the newest request even when it alone is large.
            budget = max(256, int((limit - used) * 3.5))
            if len(value) > budget: m = dict(m); m["content"] = value[-budget:]
            cost = prompt_token_estimate(f'{m.get("role", "user")}: {message_value(m)}\n')
        selected.append(m); used += cost
    selected.reverse()
    result = system + selected
    if prompt_token_estimate(prompt_from_messages(result)) > limit:
        # Final hard cap protects against tokenization differences, especially
        # for CJK text and JSON tool results.
        prompt = prompt_from_messages(result)
        prompt = prompt[-int(limit * 3.5):]
        return [{"role": "user", "content": prompt}]
    return result

def extract_tool_calls(raw):
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.S) or re.findall(r"<\|tool_call\|>\s*(.*?)\s*<\|tool_call_end\|>", raw, re.S)
    values = []
    for block in blocks or [raw.strip()]:
        try: value = json.loads(block)
        except (TypeError, ValueError): continue
        if isinstance(value, dict) and isinstance(value.get("tool_calls"), list): values.extend(value["tool_calls"])
        elif isinstance(value, list): values.extend(value)
        elif isinstance(value, dict): values.append(value)
    calls = []
    for value in values:
        if not isinstance(value, dict): continue
        fn = value.get("function", value)
        if not isinstance(fn, dict) or not fn.get("name"): continue
        args = fn.get("arguments", {}); args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        calls.append({"id": value.get("id") or "call_" + uuid.uuid4().hex, "type": "function", "function": {"name": fn["name"], "arguments": args}})
    return calls

def completion_id(): return "chatcmpl-" + uuid.uuid4().hex[:24]
def sse(obj): return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
def error(message, typ="invalid_request_error", code=400): return jsonify({"error": {"message": message, "type": typ, "param": None, "code": None}}), code

def run_model(model, prompt, role, thinking, sampling, max_tokens):
    global global_text, global_state
    global_text, global_state = [], -1; worker_error = []
    def worker():
        try: model.run(prompt, role, thinking, sampling, max_tokens)
        except BaseException as exc: worker_error.append(exc)
    t = threading.Thread(target=worker, daemon=True); t.start(); result = []
    while t.is_alive() or global_text:
        while global_text: result.append(global_text.pop(0))
        t.join(.01)
    if worker_error: raise worker_error[0]
    if global_state == LLMCallState.RKLLM_RUN_ERROR: raise RuntimeError("rkllm_run failed: RKLLM_RUN_ERROR")
    return "".join(result)

def make_sampling(data):
    p = RKLLMSamplingParam(); p.top_k = int(data.get("top_k", 1)); p.top_p = float(data.get("top_p", .9)); p.temperature = float(data.get("temperature", .8)); p.repeat_penalty = float(data.get("repeat_penalty", 1.1)); p.frequency_penalty = float(data.get("frequency_penalty", 0)); p.presence_penalty = float(data.get("presence_penalty", 0)); return p

@app.route('/v1/models', methods=['GET'])
def models(): return jsonify({"object": "list", "data": [{"id": rkllm_model.model_name, "object": "model", "created": int(time.time()), "owned_by": "rkllm"}]})

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list): return error("'messages' must be an array")
    messages = limit_messages(data["messages"]); tools = data.get("tools") or []
    if tools and not isinstance(tools, list): return error("'tools' must be an array")
    choice = data.get("tool_choice", "auto"); tools_for_model = [] if choice == "none" else tools
    system = next((text_content(m.get("content", "")) for m in messages if m.get("role") == "system"), "")
    try:
        with lock:
            if tools_for_model: rkllm_model.configure_tools(system, tools_for_model)
            raw = run_model(rkllm_model, prompt_from_messages(messages), "user", data.get("enable_thinking", False), make_sampling(data), data.get("max_tokens", 4096))
    except Exception as exc: return error(str(exc), "server_error", 500)
    calls = extract_tool_calls(raw) if tools_for_model and choice != "none" else []
    if choice == "required" and not calls: return error("The model did not return a tool call", "server_error", 500)
    ident, created, model = completion_id(), int(time.time()), data.get("model", rkllm_model.model_name); finish = "tool_calls" if calls else "stop"; message = {"role": "assistant", "content": None if calls else raw}
    if calls: message["tool_calls"] = calls
    prompt = prompt_from_messages(messages); usage = {"prompt_tokens": int(prompt_token_estimate(prompt)), "completion_tokens": len(raw.split())}; usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    if not data.get("stream", False): return jsonify({"id": ident, "object": "chat.completion", "created": created, "model": model, "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish}], "usage": usage})
    def stream_response():
        base = {"id": ident, "object": "chat.completion.chunk", "created": created, "model": model}; yield sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "logprobs": None, "finish_reason": None}]})
        if calls:
            for i, call in enumerate(calls):
                yield sse({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, "id": call["id"], "type": "function", "function": {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}}]}, "logprobs": None, "finish_reason": None}]})
        else:
            for part in re.findall(r".{1,64}", raw, re.S): yield sse({**base, "choices": [{"index": 0, "delta": {"content": part}, "logprobs": None, "finish_reason": None}]})
        yield sse({**base, "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": finish}], "usage": usage}); yield "data: [DONE]\n\n"
    return Response(stream_with_context(stream_response()), content_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--rkllm_model_path', required=True); ap.add_argument('--target_platform', required=True); args = ap.parse_args()
    if not os.path.exists(args.rkllm_model_path): raise SystemExit("invalid model path")
    if args.target_platform not in ("rk3588", "rk3576", "rv1126b", "rk3562"): raise SystemExit("invalid target platform")
    subprocess.run("sudo bash fix_freq_{}.sh".format(args.target_platform), shell=True); resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400)); rkllm_model = RKLLM(args.rkllm_model_path, args.target_platform)
    def shutdown(signum, frame): rkllm_model.release(); raise SystemExit(0)
    signal.signal(signal.SIGINT, shutdown); signal.signal(signal.SIGTERM, shutdown)
    try: app.run(host='0.0.0.0', port=8080, threaded=False, debug=False)
    finally: rkllm_model.release()
