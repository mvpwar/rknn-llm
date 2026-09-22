import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import signal
import json
import re
import uuid
import gradio as gr
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Keep the original Gradio configuration and RKLLM model path unchanged.
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = "8080"
rkllm_lib = ctypes.CDLL('lib/librkllmrt.so')

RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)
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
    _fields_ = [("base_domain_id", ctypes.c_int32), ("embed_flash", ctypes.c_int8), ("enabled_cpus_num", ctypes.c_int8), ("enabled_cpus_mask", ctypes.c_uint32), ("n_batch", ctypes.c_uint8), ("use_cross_attn", ctypes.c_int8), ("reserved", ctypes.c_uint8 * 104)]
class RKLLMParam(ctypes.Structure):
    _fields_ = [("model_path", ctypes.c_char_p), ("max_context_len", ctypes.c_int32), ("max_new_tokens", ctypes.c_int32), ("top_k", ctypes.c_int32), ("n_keep", ctypes.c_int32), ("top_p", ctypes.c_float), ("temperature", ctypes.c_float), ("repeat_penalty", ctypes.c_float), ("frequency_penalty", ctypes.c_float), ("presence_penalty", ctypes.c_float), ("mirostat", ctypes.c_int32), ("mirostat_tau", ctypes.c_float), ("mirostat_eta", ctypes.c_float), ("skip_special_token", ctypes.c_bool), ("ignore_eos_token", ctypes.c_bool), ("is_async", ctypes.c_bool), ("extend_param", RKLLMExtendParam)]
class RKLLMLoraAdapter(ctypes.Structure):
    _fields_ = [("lora_adapter_path", ctypes.c_char_p), ("lora_adapter_name", ctypes.c_char_p), ("scale", ctypes.c_float)]
class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [("embed", ctypes.POINTER(ctypes.c_float)), ("n_tokens", ctypes.c_size_t)]
class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [("input_ids", ctypes.POINTER(ctypes.c_int32)), ("n_tokens", ctypes.c_size_t)]
class RKLLMImageInput(ctypes.Structure):
    _fields_ = [("image_embed", ctypes.POINTER(ctypes.c_float)), ("n_image_tokens", ctypes.c_size_t), ("n_image", ctypes.c_size_t), ("image_start", ctypes.c_char_p), ("image_end", ctypes.c_char_p), ("image_content", ctypes.c_char_p), ("image_width", ctypes.c_size_t), ("image_height", ctypes.c_size_t)]
class RKLLMVideoInput(ctypes.Structure):
    _fields_ = [("video_embed", ctypes.POINTER(ctypes.c_float)), ("n_frame_tokens", ctypes.c_size_t), ("n_frame_per_video", ctypes.c_size_t), ("n_video", ctypes.c_size_t), ("video_start", ctypes.c_char_p), ("video_end", ctypes.c_char_p), ("video_content", ctypes.c_char_p), ("frame_width", ctypes.c_size_t), ("frame_height", ctypes.c_size_t)]
class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [("prompt", ctypes.c_char_p), ("image", RKLLMImageInput), ("video", RKLLMVideoInput)]
class RKLLMInputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p), ("embed_input", RKLLMEmbedInput), ("token_input", RKLLMTokenInput), ("multimodal_input", RKLLMMultiModalInput)]
class RKLLMInput(ctypes.Structure):
    _anonymous_ = ("input_data",)
    _fields_ = [("role", ctypes.c_char_p), ("enable_thinking", ctypes.c_bool), ("input_type", RKLLMInputType), ("input_data", RKLLMInputUnion)]
class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]
class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [("save_prompt_cache", ctypes.c_int), ("prompt_cache_path", ctypes.c_char_p)]
class RKLLMSamplingParam(ctypes.Structure):
    _fields_ = [("top_k", ctypes.c_int32), ("top_p", ctypes.c_float), ("temperature", ctypes.c_float), ("repeat_penalty", ctypes.c_float), ("frequency_penalty", ctypes.c_float), ("presence_penalty", ctypes.c_float), ("mirostat", ctypes.c_int32), ("mirostat_tau", ctypes.c_float), ("mirostat_eta", ctypes.c_float)]
class RKLLMInferParam(ctypes.Structure):
    _fields_ = [("mode", RKLLMInferMode), ("lora_params", ctypes.POINTER(RKLLMLoraParam)), ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)), ("sampling_params", ctypes.POINTER(RKLLMSamplingParam)), ("keep_history", ctypes.c_int), ("max_new_tokens", ctypes.c_int32)]
class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [("hidden_states", ctypes.POINTER(ctypes.c_float)), ("embd_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]
class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [("logits", ctypes.POINTER(ctypes.c_float)), ("vocab_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]
class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [("prefill_time_ms", ctypes.c_float), ("prefill_tokens", ctypes.c_int), ("generate_time_ms", ctypes.c_float), ("generate_tokens", ctypes.c_int), ("memory_usage_mb", ctypes.c_float)]
class RKLLMResult(ctypes.Structure):
    _fields_ = [("text", ctypes.c_char_p), ("token_id", ctypes.c_int), ("last_hidden_layer", RKLLMResultLastHiddenLayer), ("logits", RKLLMResultLogits), ("perf", RKLLMPerfStat)]

global_text = []
global_state = -1
split_byte_data = bytes()

def callback_impl(result, userdata, state):
    global global_text, global_state
    global_state = state
    if state == LLMCallState.RKLLM_RUN_NORMAL and result.contents.text:
        # The original callback stores output in the shared streaming buffer.
        global_text.append(result.contents.text.decode('utf-8'))
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        print("run error")
        sys.stdout.flush()
    return 0

LLMResultCallback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
LLMTokenizerCallback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32)
LLMGetEmbedCallback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64)
class RKLLMCallback(ctypes.Structure):
    _fields_ = [("result_callback", LLMResultCallback_type), ("result_userdata", ctypes.c_void_p), ("tokenizer_callback", LLMTokenizerCallback_type), ("tokenizer_userdata", ctypes.c_void_p), ("embed_callback", LLMGetEmbedCallback_type), ("embed_userdata", ctypes.c_void_p)]
callback = LLMResultCallback_type(callback_impl)

class RKLLM:
    def __init__(self, model_path, lora_model_path=None, prompt_cache_path=None, platform="rk3588"):
        p = RKLLMParam()
        p.model_path = bytes(model_path, 'utf-8')
        p.max_context_len, p.max_new_tokens, p.skip_special_token, p.n_keep = 4096, 4096, True, -1
        p.top_k, p.top_p, p.temperature, p.repeat_penalty = 1, .9, .8, 1.1
        p.frequency_penalty = p.presence_penalty = 0.0
        p.mirostat, p.mirostat_tau, p.mirostat_eta = 0, 5.0, .1
        p.is_async, p.ignore_eos_token = False, False
        p.extend_param.base_domain_id, p.extend_param.embed_flash = 0, 1
        p.extend_param.n_batch, p.extend_param.use_cross_attn, p.extend_param.enabled_cpus_num = 1, 0, 4
        p.extend_param.enabled_cpus_mask = ((1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)) if platform.lower() in ["rk3576", "rk3588"] else 15
        self.handle = RKLLM_Handle_t()
        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), ctypes.POINTER(RKLLMCallback)]
        self.rkllm_init.restype = ctypes.c_int
        self.callback = RKLLMCallback(callback, None, LLMTokenizerCallback_type(), None, LLMGetEmbedCallback_type(), None)
        if self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(p), ctypes.byref(self.callback)) != 0:
            raise RuntimeError("rkllm init failed")
        print("\nrkllm init success!\n")
        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int
        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int
        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(self.rkllm_infer_params))
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self.rkllm_infer_params.keep_history = 0
        if lora_model_path:
            adapter = RKLLMLoraAdapter(ctypes.c_char_p(lora_model_path.encode()), ctypes.c_char_p(b"test"), 1.0)
            load = rkllm_lib.rkllm_load_lora
            load.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
            load.restype = ctypes.c_int
            load(self.handle, ctypes.byref(adapter))
            lp = RKLLMLoraParam(ctypes.c_char_p(b"test"))
            self.rkllm_infer_params.lora_params = ctypes.pointer(lp)
            self._lora_param = lp
        if prompt_cache_path:
            load_cache = rkllm_lib.rkllm_load_prompt_cache
            load_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
            load_cache.restype = ctypes.c_int
            load_cache(self.handle, ctypes.c_char_p(prompt_cache_path.encode()))

    def run(self, prompt, sampling_params=None, max_new_tokens=None):
        inp = RKLLMInput()
        inp.role, inp.enable_thinking, inp.input_type = b"user", False, RKLLMInputType.RKLLM_INPUT_PROMPT
        inp.prompt_input = ctypes.c_char_p(prompt.encode('utf-8'))
        if sampling_params is not None:
            self.rkllm_infer_params.sampling_params = ctypes.pointer(sampling_params)
        if max_new_tokens is not None:
            self.rkllm_infer_params.max_new_tokens = max_new_tokens
        try:
            return self.rkllm_run(self.handle, ctypes.byref(inp), ctypes.byref(self.rkllm_infer_params), None)
        finally:
            self.rkllm_infer_params.sampling_params = None
            self.rkllm_infer_params.max_new_tokens = 0

    def release(self):
        self.rkllm_destroy(self.handle)

# ---------------- OpenAI-compatible API, including tools/tool_calls ----------------
model_lock = threading.Lock()

def content_text(value):
    if isinstance(value, str): return value
    if isinstance(value, list):
        return " ".join(str(x.get("text", "")) for x in value if isinstance(x, dict) and x.get("type") == "text")
    return ""

def tool_call_from_text(text):
    patterns = [r"<tool_call>\s*(.*?)\s*</tool_call>", r"<\|tool_call\|>\s*(.*?)\s*<\|tool_call_end\|>"]
    candidates = sum((re.findall(p, text, re.S) for p in patterns), []) + [text.strip()]
    for raw in candidates:
        try: value = json.loads(raw)
        except (TypeError, ValueError): continue
        if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
            value = value["tool_calls"][0] if value["tool_calls"] else {}
        if not isinstance(value, dict): continue
        fn = value.get("function", value)
        if not isinstance(fn, dict) or not fn.get("name"): continue
        args = fn.get("arguments", {})
        if not isinstance(args, str): args = json.dumps(args, ensure_ascii=False)
        return {"id": value.get("id", "call_" + uuid.uuid4().hex), "type": "function", "function": {"name": fn["name"], "arguments": args}}
    return None

def messages_prompt(messages, tools):
    lines = ["You are a helpful assistant. If a tool is needed, output only <tool_call>{\"name\":\"function_name\",\"arguments\":{}}</tool_call>."]
    if tools: lines.append("Available tools: " + json.dumps(tools, ensure_ascii=False))
    for m in messages:
        if not isinstance(m, dict): continue
        role, value = m.get("role", "user"), m.get("content", "")
        if role == "assistant" and m.get("tool_calls"): value = json.dumps(m["tool_calls"], ensure_ascii=False)
        lines.append(role + ": " + content_text(value))
    lines.append("assistant:")
    return "\n".join(lines)

def generate_text(model, prompt, max_tokens=None):
    global global_text, global_state
    with model_lock:
        global_text, global_state = [], -1
        worker = threading.Thread(target=model.run, args=(prompt, None, max_tokens), daemon=True)
        worker.start()
        chunks = []
        while worker.is_alive() or global_text:
            while global_text: chunks.append(global_text.pop(0))
            worker.join(.01)
        return "".join(chunks)

class OpenAIHandler(BaseHTTPRequestHandler):
    server_version = "RKLLMOpenAI/1.0"
    def log_message(self, fmt, *args): pass
    def body(self):
        try: return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except (ValueError, TypeError, json.JSONDecodeError): return None
    def send_json(self, status, value):
        raw = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self.send_json(200, {"object":"list", "data":[{"id":self.server.model_name,"object":"model","created":int(time.time()),"owned_by":"rkllm"}]})
        else: self.send_json(404, {"error":{"message":"Not found","type":"invalid_request_error"}})
    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_json(404, {"error":{"message":"Not found","type":"invalid_request_error"}}); return
        req = self.body()
        if not isinstance(req, dict) or not isinstance(req.get("messages"), list) or not req["messages"]:
            self.send_json(400, {"error":{"message":"messages must be a non-empty array","type":"invalid_request_error"}}); return
        messages, tools = req["messages"], req.get("tools", [])
        text = generate_text(self.server.model, messages_prompt(messages, tools), req.get("max_tokens"))
        tc = tool_call_from_text(text) if tools else None
        rid, now = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
        if tc:
            message, finish = {"role":"assistant","content":None,"tool_calls":[tc]}, "tool_calls"
        else:
            message, finish = {"role":"assistant","content":text}, "stop"
        if not req.get("stream"):
            self.send_json(200, {"id":rid,"object":"chat.completion","created":now,"model":self.server.model_name,"choices":[{"index":0,"message":message,"finish_reason":finish}]}); return
        self.send_response(200); self.send_header("Content-Type","text/event-stream"); self.send_header("Cache-Control","no-cache"); self.end_headers()
        delta = {"role":"assistant","content":None,"tool_calls":message["tool_calls"]} if tc else {"role":"assistant","content":text}
        chunk = {"id":rid,"object":"chat.completion.chunk","created":now,"model":self.server.model_name,"choices":[{"index":0,"delta":delta,"finish_reason":None}]}
        self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode()); self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()

class OpenAIServer(ThreadingHTTPServer):
    allow_reuse_address = True
    def __init__(self, address, model, model_name="rkllm"):
        self.model, self.model_name = model, model_name
        super().__init__(address, OpenAIHandler)

def start_openai_server(model, host="0.0.0.0", port=8000):
    server = OpenAIServer((host, port), model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("OpenAI API listening on http://{}:{}/v1".format(host, port))
    return server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', type=str, required=True)
    parser.add_argument('--target_platform', type=str, required=True)
    parser.add_argument('--lora_model_path', type=str)
    parser.add_argument('--prompt_cache_path', type=str)
    parser.add_argument('--openai_host', type=str, default='0.0.0.0')
    parser.add_argument('--openai_port', type=int, default=8000, help='Set 0 to disable the OpenAI API.')
    args = parser.parse_args()
    if not os.path.exists(args.rkllm_model_path): raise SystemExit("Invalid rkllm model path")
    if args.target_platform not in ["rk3588", "rk3576", "rv1126b", "rk3562"]: raise SystemExit("Invalid target platform")
    for path, label in [(args.lora_model_path, "lora_model"), (args.prompt_cache_path, "prompt_cache")]:
        if path and not os.path.exists(path): raise SystemExit("Invalid {} path".format(label))
    subprocess.run("sudo bash fix_freq_{}.sh".format(args.target_platform), shell=True)
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))
    print("=========init....===========")
    rkllm_model = RKLLM(args.rkllm_model_path, args.lora_model_path, args.prompt_cache_path, args.target_platform)
    openai_server = start_openai_server(rkllm_model, args.openai_host, args.openai_port) if args.openai_port else None

    def shutdown_handler(signum, frame):
        if openai_server: openai_server.shutdown(); openai_server.server_close()
        rkllm_model.release(); sys.exit(0)
    signal.signal(signal.SIGINT, shutdown_handler); signal.signal(signal.SIGTERM, shutdown_handler)

    def get_user_input(user_message, history): return "", history + [{"role":"user","content":user_message}]
    def get_RKLLM_output(history):
        global global_text, global_state
        user_content = next((content_text(x.get("content", "")) for x in reversed(history) if isinstance(x, dict) and x.get("role") == "user"), "")
        history = history + [{"role":"assistant","content":""}]
        with model_lock:
            global_text, global_state = [], -1
            worker = threading.Thread(target=rkllm_model.run, args=(user_content,), daemon=True); worker.start()
            while worker.is_alive() or global_text:
                while global_text: history[-1]["content"] += global_text.pop(0); yield history
                worker.join(.005)

    with gr.Blocks(title="Chat with RKLLM") as chatRKLLM:
        gr.Markdown("<div align='center'><font size='70'> Chat with RKLLM </font></div>")
        gr.Markdown("### Enter your question in the inputTextBox and press the Enter key to chat with the RKLLM model.")
        rkllmServer = gr.Chatbot(height=600); msg = gr.Textbox(placeholder="Please input your question here...", label="inputTextBox"); clear = gr.Button("Clear")
        msg.submit(get_user_input, [msg, rkllmServer], [msg, rkllmServer], queue=False).then(get_RKLLM_output, rkllmServer, rkllmServer)
        clear.click(lambda: None, None, rkllmServer, queue=False)
    chatRKLLM.queue()
    try: chatRKLLM.launch()
    finally:
        if openai_server:
            openai_server.shutdown(); openai_server.server_close()
        rkllm_model.release()
