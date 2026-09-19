#!/usr/bin/env python3
"""集成测试探针：真实 ai-gateway 二进制 + 真实 TCP 客户端。

只为 scripts/integration_pipeline_test.sh 服务（那个脚本负责起进程、配端口、
发信号、判 PASS/FAIL）。这里只做两件事：
  1. upstream —— 一个可控的 OpenAI 兼容 mock 上游（回显 system/user，可延迟、
     可静默、可对流式请求回非 SSE 的 JSON）
  2. 客户端探针 —— 按场景发真实请求并打印机器可读的结果，供 shell 断言：
       in-flight-sigterm  在途请求期间发 SIGTERM，观察客户端拿到什么
       merge-probe        A/B 两个不同 system 的并发请求（间隔可调），观察 B 是否
                          拿到了 A 的上游答案
       keepalive-probe    同一条 TCP 连接上连发两个请求，观察第一个响应的
                          Connection 头与"第二个请求是否还能发出去"

设计约束（与项目既有约定一致）：
  - 只用标准库；不依赖第三方
  - 不按进程名 kill（线上有同名实例）：所有进程由调用方按 PID 管理
  - 输出带 `KEY=value` 前缀，便于 shell 侧用 grep 断言
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# ─────────────────────────── mock 上游 ───────────────────────────
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: N802
        sys.stderr.write("upstream: " + (fmt % args) + "\n")
        sys.stderr.flush()

    # ---- 解析请求 ----
    def _read_body(self):
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            return {}

    @staticmethod
    def _field(req, role):
        for m in req.get("messages") or []:
            if m.get("role") == role:
                c = m.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return "".join(p.get("text", "") for p in c
                                   if isinstance(p, dict))
        return ""

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _begin_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk(self, payload: bytes) -> None:
        self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
        self.wfile.flush()

    def _end_chunked(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    @staticmethod
    def _sse(obj) -> bytes:
        return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()

    def do_POST(self):  # noqa: N802
        req = self._read_body()
        # 供集成测试统计"上游被真实调用了几次"：流式缓存命中必须零回源
        sys.stderr.write("REQ\n")
        sys.stderr.flush()
        user = self._field(req, "user")
        system = self._field(req, "system")
        stream = bool(req.get("stream"))
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        # 回显 system 与 user：调用方据此判断"这份答案是不是给我的"
        answer = f"ANSWER[sys={system}][user={user}]"

        delay = float(self.server.probe_delay)
        if delay > 0:
            time.sleep(delay)

        try:
            # --- 触发"上游静默"：发一个事件后长时间不发数据 ---
            if "PROBE-SILENT" in user:
                self._begin_sse()
                self._chunk(self._sse({"choices": [{"delta": {"content": "静默前"}}]}))
                time.sleep(float(self.server.probe_silent))
                self._chunk(self._sse({"choices": [{"delta": {"content": "静默后"}}]}))
                self._end_chunked()
                return
            # --- 触发"上游对 stream:true 回非 SSE"：网关的 stream_fallback 分支 ---
            if "PROBE-FALLBACK" in user:
                self._send_json({
                    "error": {"message": "upstream does not support streaming"},
                    "choices": [{"message": {"role": "assistant", "content": answer}}],
                })
                return
            if not stream:
                self._send_json({
                    "choices": [{"message": {"role": "assistant", "content": answer},
                                 "finish_reason": "stop", "index": 0}],
                    "usage": {"prompt_tokens": 128, "completion_tokens": 57,
                              "total_tokens": 185},
                })
                return
            self._begin_sse()
            self._chunk(self._sse({"choices": [{"delta": {"content": answer}}]}))
            if include_usage:
                self._chunk(self._sse({
                    "choices": [],
                    "usage": {"prompt_tokens": 128, "completion_tokens": 57,
                              "total_tokens": 185},
                }))
            self._chunk(b"data: [DONE]\n\n")
            self._end_chunked()
        except (BrokenPipeError, ConnectionResetError):
            sys.stderr.write("upstream: client aborted the stream (expected)\n")
            sys.stderr.flush()


def cmd_upstream(args) -> int:
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.daemon_threads = True
    srv.probe_delay = args.delay
    srv.probe_silent = args.silent
    print(f"LISTENING {args.port}", flush=True)
    srv.serve_forever()
    return 0


# ─────────────────────────── 客户端小工具 ───────────────────────────
def post_chat(port: int, user: str, system: str = "", stream: bool = False,
              timeout: float = 30.0) -> tuple[int, str, float]:
    """发一个 POST /v1/chat/completions，返回 (curl_rc, body, seconds)。

    curl_rc 的口径与 curl(1) 一致地简化：0 = 拿到完整响应；
    52 = 服务端没发响应就关了连接（Empty reply）；
    28 = 超时；56 = 连接被重置。
    """
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    body = json.dumps({"model": "probe", "messages": msgs,
                       "stream": stream}, ensure_ascii=False)
    req = (f"POST /v1/chat/completions HTTP/1.1\r\n"
           f"Host: 127.0.0.1\r\nContent-Type: application/json\r\n"
           f"Content-Length: {len(body.encode())}\r\nConnection: close\r\n\r\n").encode() \
        + body.encode()
    t0 = time.time()
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    except OSError as e:
        print(f"CONNECT_ERROR={e}")
        return (7, "", time.time() - t0)
    try:
        s.sendall(req)
        chunks = []
        while True:
            b = s.recv(65536)
            if not b:
                break
            chunks.append(b)
        raw = b"".join(chunks)
    except socket.timeout:
        return (28, "", time.time() - t0)
    except ConnectionResetError:
        return (56, "", time.time() - t0)
    finally:
        s.close()
    elapsed = time.time() - t0
    if not raw:
        return (52, "", elapsed)  # Empty reply from server
    head, _, payload = raw.partition(b"\r\n\r\n")
    if b"Transfer-Encoding: chunked" in head:
        payload = decode_chunked(payload)
    return (0, payload.decode("utf-8", "replace"), elapsed)


def decode_chunked(data: bytes) -> bytes:
    out = bytearray()
    while True:
        nl = data.find(b"\r\n")
        if nl < 0:
            break
        try:
            size = int(data[:nl].split(b";")[0], 16)
        except ValueError:
            break
        if size == 0:
            break
        out += data[nl + 2:nl + 2 + size]
        data = data[nl + 2 + size + 2:]
    return bytes(out)


def split_http(raw: bytes) -> tuple[bytes, bytes]:
    head, _, rest = raw.partition(b"\r\n\r\n")
    return head, rest


# ─────────────────────────── 场景 1：在途请求 + SIGTERM（H1） ───────────────────────────
def cmd_in_flight_sigterm(args) -> int:
    """在途请求期间给网关发 SIGTERM，观察客户端拿到 200 完整响应还是被掐断。

    修复前（main 从未调用 drain()）：reactor 先退出、main 立刻统计/落盘，
    worker 还在跑 → 走到这里时连接被关，客户端拿 rc=52 / body 为空。
    修复后：drain() 等 worker 跑完 → 200 + 完整 body。
    """
    result: dict = {}

    def worker():
        result["rc"], result["body"], result["sec"] = post_chat(
            args.gateway_port, "PROBE-INFLIGHT 在途请求", timeout=30)

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(args.sigterm_after)
    gw_pid = int(args.gateway_pid)
    import os
    os.kill(gw_pid, signal.SIGTERM)
    print(f"SIGTERM_SENT={gw_pid}", flush=True)
    t.join(timeout=30)
    rc = result.get("rc", -1)
    body = result.get("body", "")
    print(f"CURL_RC={rc}")
    print(f"ELAPSED_MS={int(result.get('sec', 0) * 1000)}")
    print(f"BODY_LEN={len(body)}")
    print(f"BODY={body[:200]}")
    return 0


# ─────────────────────────── 场景 2：跨 ns 合并（H2） ───────────────────────────
def cmd_merge_probe(args) -> int:
    """A（system=PROBE-ALPHA）与 B（system=PROBE-BETA）同 user，间隔 0.25s。

    同一份 user message 让向量的余弦恒为 1.0，因此"语义合并"必然触发——
    修复前 B 会拿到 A 的上游答案（_cache=merged，正文里的 sys=PROBE-ALPHA）。
    """
    user = "PROBE-MERGE 今天天气怎么样"
    out: dict = {}

    def run_a():
        out["a"] = post_chat(args.gateway_port, user, system="PROBE-ALPHA")

    ta = threading.Thread(target=run_a)
    ta.start()
    time.sleep(args.gap)
    out["b"] = post_chat(args.gateway_port, user, system="PROBE-BETA")
    ta.join(timeout=40)
    for tag in ("a", "b"):
        rc, body, sec = out.get(tag, (-1, "", 0.0))
        print(f"{tag.upper()}_RC={rc}")
        print(f"{tag.upper()}_MS={int(sec * 1000)}")
        print(f"{tag.upper()}_BODY={body[:220]}")
        try:
            print(f"{tag.upper()}_CACHE={json.loads(body).get('_cache', '')}")
        except Exception:
            print(f"{tag.upper()}_CACHE=")
    return 0


# ─────────────────────────── 场景 3：keep-alive 决策（M3） ───────────────────────────
def cmd_keepalive_probe(args) -> int:
    """同一条 TCP 连接上连发两个请求，观察第一个响应的 Connection 头与
    "第二个请求还能不能发出去"。

    场景是流式请求遇上游非 SSE → 网关的 stream_fallback 分支，它明确返回
    keep_alive=false。修复前该决策在路由层被丢弃：响应头写 keep-alive 且连接
    真被复用（第二个请求成功）；修复后应当是 Connection: close 且连接被关闭。
    """
    body = json.dumps({"model": "probe", "stream": True,
                       "messages": [{"role": "user",
                                     "content": "PROBE-FALLBACK 你好"}]},
                      ensure_ascii=False).encode()
    req1 = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n\r\n").encode() + body
    body2 = json.dumps({"model": "probe",
                        "messages": [{"role": "user", "content": "unused"}]}).encode()
    req2 = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body2)}\r\n\r\n").encode() + body2

    s = socket.create_connection(("127.0.0.1", args.gateway_port), timeout=10)
    s.settimeout(10)
    s.sendall(req1)
    raw = b""
    while b"\r\n\r\n" not in raw:
        b = s.recv(4096)
        if not b:
            break
        raw += b
    head, rest = split_http(raw)
    # 读满第一个响应的 body（Content-Length）
    clen = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            clen = int(line.split(b":")[1].strip())
    while len(rest) < clen:
        b = s.recv(4096)
        if not b:
            break
        rest += b
    print(f"FIRST_HEAD={head.decode('utf-8', 'replace').replace(chr(13) + chr(10), ' | ')}")
    print(f"FIRST_BODY={rest[:120].decode('utf-8', 'replace')}")

    # 第二个请求：连接若已被服务端关闭，这里会读到 EOF（recv 返回空）
    reused = False
    try:
        s.sendall(req2)
        second = s.recv(4096)
        reused = bool(second)
        print(f"SECOND_BYTES={len(second)}")
    except (BrokenPipeError, ConnectionResetError) as e:
        print(f"SECOND_ERROR={type(e).__name__}")
    except socket.timeout:
        print("SECOND_ERROR=timeout")
    print(f"CONNECTION_REUSED={1 if reused else 0}")
    s.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upstream")
    up.add_argument("--port", type=int, required=True)
    up.add_argument("--delay", type=float, default=0.0)
    up.add_argument("--silent", type=float, default=10.0)
    up.set_defaults(func=cmd_upstream)

    s1 = sub.add_parser("in-flight-sigterm")
    s1.add_argument("--gateway-port", type=int, required=True)
    s1.add_argument("--gateway-pid", type=int, required=True)
    s1.add_argument("--sigterm-after", type=float, default=1.0)
    s1.set_defaults(func=cmd_in_flight_sigterm)

    s2 = sub.add_parser("merge-probe")
    s2.add_argument("--gateway-port", type=int, required=True)
    s2.add_argument("--gap", type=float, default=0.25)
    s2.set_defaults(func=cmd_merge_probe)

    s3 = sub.add_parser("keepalive-probe")
    s3.add_argument("--gateway-port", type=int, required=True)
    s3.set_defaults(func=cmd_keepalive_probe)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
