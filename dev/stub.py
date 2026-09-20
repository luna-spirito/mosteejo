#!/usr/bin/env python3
"""Stub Telegram Bot API (:8081) and OpenAI-compatible LLM (:8082) for a
local smoke run of rp-bot. Records outbound Telegram sends to /tmp/rp-smoke/sends.log
and LLM request summaries to /tmp/rp-smoke/llm_requests.log."""
import json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TG_SEED = [
    {
        "update_id": 100,
        "message": {
            "message_id": 11,
            "from": {"id": 1, "is_bot": False, "first_name": "Alice", "username": "alice"},
            "chat": {"id": -100123, "type": "supergroup", "title": "RP"},
            "message_thread_id": 42,
            "date": 1758000000,
            "text": "Привет! Начнём сцену в таверне.",
        },
    },
    {
        "update_id": 101,
        "message": {
            "message_id": 12,
            "from": {"id": 1, "is_bot": False, "first_name": "Alice", "username": "alice"},
            "chat": {"id": -100123, "type": "supergroup", "title": "RP"},
            "message_thread_id": 42,
            "date": 1758000001,
            "text": "/model glm-5.3:max",
        },
    },
]
state = {"updates_sent": False, "llm_calls": 0}

class Tg(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        method = self.path.rsplit("/", 1)[-1]
        if method == "getMe":
            result = {"id": 7, "is_bot": True, "first_name": "Stub", "username": "rp_bot"}
        elif method == "getUpdates":
            if not state["updates_sent"]:
                state["updates_sent"] = True
                result = TG_SEED
            else:
                time.sleep(2)  # emulate long poll
                result = []
        elif method == "sendMessage":
            with open("/tmp/rp-smoke/sends.log", "a") as f:
                f.write(json.dumps(body, ensure_ascii=False) + "\n")
            result = {"message_id": 555}
        elif method == "sendChatAction":
            result = True
        else:
            result = None
        resp = json.dumps({"ok": True, "result": result}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

class Llm(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        reasoning_in_history = sum(
            1 for m in req.get("messages", []) if m.get("reasoning_content")
        )
        with open("/tmp/rp-smoke/llm_requests.log", "a") as f:
            f.write(json.dumps({
                "model": req.get("model"),
                "reasoning_effort": req.get("reasoning_effort"),
                "thinking": req.get("thinking"),
                "clear_thinking": req.get("clear_thinking"),
                "tools": len(req.get("tools", [])),
                "assistant_msgs_with_reasoning": reasoning_in_history,
            }, ensure_ascii=False) + "\n")
        state["llm_calls"] += 1
        if state["llm_calls"] == 1:
            message = {"role": "assistant", "reasoning_content": "(Думаю: поприветствовать гостей таверны.)",
                       "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "send_message", "arguments": json.dumps({
                    "chat_id": -100123, "thread_id": 42, "reply_to_message_id": 11,
                    "text": "Трактирщик поднимает взгляд: *«Проходите, садитесь»*",
                }, ensure_ascii=False)}}]}
        else:
            message = {"role": "assistant", "reasoning_content": "(Гость сел, сцена начата.)",
                       "content": "(сцена начата)"}
        resp = json.dumps({
            "choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 30, "cached_tokens": 870},
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

for handler, port in [(Tg, 8081), (Llm, 8082)]:
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(3600)
