#!/usr/bin/env python3
"""集成测试用的 mock OpenAI 兼容上游（SSE 流式 + 非流式）

只为 scripts/integration_stream_test.sh 服务：不依赖第三方库（标准库
http.server 足够），行为由请求体里最后一条 user 消息的文本决定：

  content 含 "B-长流"   -> 每 0.7s 发一个 delta 事件，共 12 个（≈8.4s 总时长），
                           末尾带 usage 事件 + [DONE]
  content 含 "C-静默"   -> 先发一个 delta 事件，然后静默 10s（不发任何字节），
                           用于验证网关的空闲死线会中停上游
  content 含 "usage"    -> 普通 3 个 delta 事件 + usage 事件（token 计数用）
  其它                  -> 3 个 delta 事件（非流式时返回完整 JSON）

usage 事件只在请求体里出现 `"include_usage": true`（且 stream=true）时才发 ——
与真实 OpenAI 兼容实现一致。这样"网关是否解析 usage"与"上游是否提供 usage"
两件事能被分开验证。

启动后向 stdout 打印 "LISTENING <port>"，脚本据此同步。
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DELTA_PAUSE = 0.7  # 长流模式下每个事件的间隔（秒）
LONG_EVENTS = 12
SILENT_SECONDS = 10.0


def sse_event(obj) -> bytes:
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()


def delta_event(text: str) -> bytes:
    return sse_event({"choices": [{"delta": {"content": text}, "index": 0}]})


def usage_event(prompt_tokens: int = 128, completion_tokens: int = 57) -> bytes:
    return sse_event({
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # 静默默认访问日志（脚本会重定向到文件）
        sys.stderr.write("upstream: " + (fmt % args) + "\n")
        sys.stderr.flush()

    # ---- helpers ----
    def _read_body(self):
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            return {}

    @staticmethod
    def _last_user_text(req) -> str:
        for m in reversed(req.get("messages") or []):
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return "".join(p.get("text", "") for p in c
                                   if isinstance(p, dict))
        return ""

    def _begin_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        # 无 Content-Length -> 服务端用 chunked 或 connection close；
        # BaseHTTPRequestHandler 在未显式设置长度时按 close 语义处理，
        # 这里用 chunked 更接近真实上游
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk(self, payload: bytes):
        self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
        self.wfile.flush()

    def _end_chunked(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_POST(self):  # noqa: N802
        req = self._read_body()
        text = self._last_user_text(req)
        stream = bool(req.get("stream"))
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))

        answer = f"mock-answer-for[{text}]"

        if not stream:
            body = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": answer},
                             "finish_reason": "stop", "index": 0}],
                "usage": {"prompt_tokens": 128, "completion_tokens": 57,
                          "total_tokens": 185},
            }, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        try:
            if "C-静默" in text:
                # 先给一个事件（证明流已经开始），然后长时间静默
                self._begin_sse()
                self._chunk(delta_event("静默前的一个事件"))
                time.sleep(SILENT_SECONDS)
                self._chunk(delta_event("静默之后（网关应已中停）"))
                self._chunk(b"data: [DONE]\n\n")
                self._end_chunked()
                return

            if "B-长流" in text:
                self._begin_sse()
                for i in range(LONG_EVENTS):
                    self._chunk(delta_event(f"长流片段{i}"))
                    # 最后一个事件后不再等待，避免总时长无谓地拉长
                    if i != LONG_EVENTS - 1:
                        time.sleep(DELTA_PAUSE)
                if include_usage:
                    self._chunk(usage_event(600, 400))
                self._chunk(b"data: [DONE]\n\n")
                self._end_chunked()
                return

            # 默认：3 个 delta 事件（间隔很小），可选 usage
            self._begin_sse()
            for i in range(3):
                self._chunk(delta_event(f"片段{i}"))
                time.sleep(0.05)
            if include_usage:
                self._chunk(usage_event())
            self._chunk(b"data: [DONE]\n\n")
            self._end_chunked()
        except (BrokenPipeError, ConnectionResetError):
            # 网关中停上游（空闲死线 / 客户端断开）时就是这条路径，属预期行为
            sys.stderr.write("upstream: client aborted the stream (expected)\n")
            sys.stderr.flush()


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()
    port = args.port or free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    print(f"LISTENING {port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    threading.stack_size(1 << 20)
    raise SystemExit(main())
