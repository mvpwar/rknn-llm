"""OpenAI-compatible HTTP API for the RKLLM Gradio demo.

This module is intentionally independent from Gradio. It is started by
``gradio_server.py`` and receives the already initialized RKLLM instance.
"""

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class OpenAIRequestHandler(BaseHTTPRequestHandler):
    """Small dependency-free subset of the OpenAI API."""

    server_version = "RKLLMOpenAI/1.0"

    def log_message(self, fmt, *args):
        # Keep the existing application's console output readable.
        print("[openai-api] " + fmt % args)

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _content_to_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return ""

    def _prompt_from_messages(self, messages):
        # Keep the same behavior as the original Gradio UI: RKLLM receives
        # the most recent user question, while the API accepts normal chat
        # message arrays for OpenAI client compatibility.
        if not isinstance(messages, list):
            return ""
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                return self._content_to_text(message.get("content", ""))
        return ""

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": self.server.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "rkllm",
                }],
            })
            return
        self._send_json(404, {"error": {"message": "Not found", "type": "invalid_request_error"}})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "Not found", "type": "invalid_request_error"}})
            return

        request = self._read_json()
        if not isinstance(request, dict):
            self._send_json(400, {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}})
            return

        prompt = self._prompt_from_messages(request.get("messages", []))
        if not prompt:
            self._send_json(400, {"error": {"message": "messages must contain a user message", "type": "invalid_request_error"}})
            return

        sampling = RKLLMSamplingParam()
        sampling.top_k = int(request.get("top_k", 1))
        sampling.top_p = float(request.get("top_p", 0.9))
        sampling.temperature = float(request.get("temperature", 0.8))
        sampling.repeat_penalty = float(request.get("repeat_penalty", 1.1))
        sampling.frequency_penalty = float(request.get("frequency_penalty", 0.0))
        sampling.presence_penalty = float(request.get("presence_penalty", 0.0))
        sampling.mirostat = int(request.get("mirostat", 0))
        sampling.mirostat_tau = float(request.get("mirostat_tau", 5.0))
        sampling.mirostat_eta = float(request.get("mirostat_eta", 0.1))
        max_tokens = request.get("max_tokens")
        max_tokens = int(max_tokens) if max_tokens is not None else None
        stream = bool(request.get("stream", False))
        request_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())

        if stream:
            self._stream_response(request_id, created, prompt, sampling, max_tokens)
        else:
            text = "".join(self.server.generate(prompt, sampling, max_tokens))
            self._send_json(200, {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": self.server.model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
            })

    def _stream_response(self, request_id, created, prompt, sampling, max_tokens):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def send(payload):
            self.wfile.write(("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8"))
            self.wfile.flush()

        try:
            for text in self.server.generate(prompt, sampling, max_tokens):
                send({"id": request_id, "object": "chat.completion.chunk", "created": created,
                      "model": self.server.model_name,
                      "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                                   "finish_reason": None}]})
            send({"id": request_id, "object": "chat.completion.chunk", "created": created,
                  "model": self.server.model_name,
                  "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class OpenAICompatibleServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, address, model, model_name="rkllm"):
        super().__init__(address, OpenAIRequestHandler)
        self.model = model
        self.model_name = model_name
        self.inference_lock = threading.Lock()

    def generate(self, prompt, sampling_params=None, max_new_tokens=None):
        """Yield callback output while serializing access to the RKLLM handle."""
        global global_text, global_state
        with self.inference_lock:
            global_text = []
            global_state = -1
            worker = threading.Thread(
                target=self.model.run,
                args=(prompt, sampling_params, max_new_tokens),
                daemon=True,
            )
            worker.start()
            while worker.is_alive() or global_text:
                while global_text:
                    yield global_text.pop(0)
                worker.join(timeout=0.005)


def start_openai_server(model, host="0.0.0.0", port=8000, model_name="rkllm"):
    server = OpenAICompatibleServer((host, port), model, model_name)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print("OpenAI-compatible API listening on http://{}:{}/v1".format(host, port))
    return server


# These names are assigned by gradio_server.py after importing this module.
global_text = []
global_state = -1
RKLLMSamplingParam = None
