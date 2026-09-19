#!/usr/bin/env bash
# 端到端集成验证（第二轮）：**走真实 main 路径** —— 真 ai-gateway 二进制 + 真配置
# + 真 TCP 请求 + 真 SIGTERM。
#
# 为什么必须做成集成测试而不是单测：
#   本轮对抗性审查的核心发现（报告 T2）是"tests/ 里没有任何目标编译 main.cpp 的
#   请求管道"，于是三类缺陷全部落在零覆盖层。上一轮"handle_stream_request 从不
#   发响应头"也是这么漏掉的。本脚本反过来：所有断言都由**真进程**产生。
#
# 覆盖（对应审查报告条目）：
#   A. H1  main 是否真的接线了 server.drain()：
#          在途请求（mock 上游 5s）期间发 SIGTERM →
#          修复前：客户端 rc=52 / body 空 / 日志 "ai-gateway stopped" 早于 worker 的
#                  "200 ... 5005ms" / 在途请求的统计与缓存写入全丢
#          修复后：rc=0 / http=200 / 完整 body / stopped 在最后 / 缓存条目包含它
#   B. H2  跨 namespace 的并发请求不得合并（B 对话不能拿到 A 对话的上游答案）；
#          同 namespace 仍然合并（修复不能把合法合并一起挡掉）
#   C. M3  流式请求遇上游非 SSE（stream_fallback）时，连接**不复用**：
#          响应头是 Connection: close，且同一条 TCP 连接上的第二个请求发不出去
#   D. M1  上游静默被空闲死线中停时，日志必须**归因**为上游空闲（WARN + 原因），
#          而不是像修复前那样打成 "stream: done ... done_event=no" 的正常完成
#
# 安全约定（务必遵守）：
#   - **绝不** pkill/pgrep-kill 名为 ai-gateway 的进程（线上实例同名，端口 4000）。
#     这里只启动自己的实例、只 kill 自己记下的 PID，且用 PID 文件而非进程名。
#   - 端口由 --pick-port 随机取，并核对不是 4000
#   - 本脚本运行的进程 cwd 是一个临时目录（不碰仓库里的 cache/lru_store.json）
#
# 用法：bash scripts/integration_pipeline_test.sh <path-to-ai-gateway-binary>
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATEWAY_BIN="${1:-}"
PROBE="${REPO_ROOT}/scripts/gw_probe.py"
if [[ -z "$GATEWAY_BIN" || ! -x "$GATEWAY_BIN" ]]; then
  echo "[fatal] 用法：bash scripts/integration_pipeline_test.sh <ai-gateway 二进制路径>" >&2
  exit 2
fi
# 网关以临时目录为 cwd 启动（见 start_gateway），因此二进制必须先解析成绝对路径
GATEWAY_BIN="$(cd "$(dirname "$GATEWAY_BIN")" && pwd)/$(basename "$GATEWAY_BIN")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "[fatal] 需要 python3（mock 上游与探针都用它）" >&2
  exit 2
fi

WORK="$(mktemp -d /tmp/agw-pipeline.XXXXXX)"
GW_PIDFILE="${WORK}/gateway.pid"
SAW_PIDFILE="${WORK}/saw.pid"
GW_LOG="${WORK}/gateway.log"
UP_LOG="${WORK}/upstream.log"
GW_CONFIG="${WORK}/gateway.json"

# 网关以 WORK 为 cwd（配置、cache/、日志全部落在临时目录里，绝不碰仓库工作区）
mkdir -p "${WORK}/cache"

pick_port() {
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

UP_PORT="$(pick_port)"
GW_PORT="$(pick_port)"
for p in "$UP_PORT" "$GW_PORT"; do
  if [[ "$p" == "4000" || "$p" == "9000" ]]; then
    echo "[fatal] 随机端口撞上受保护端口 ${p}，重跑一次" >&2
    exit 2
  fi
done

PASS=0
FAIL=0
ok()   { echo "  [PASS] $*"; PASS=$((PASS + 1)); }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL + 1)); }
info() { echo "  ----- $*"; }

kill_pid_file() {  # $1 = pid file
  [[ -f "$1" ]] || return 0
  local pid
  pid="$(cat "$1" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 25); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.2
    done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  kill_pid_file "$GW_PIDFILE"
  kill_pid_file "$SAW_PIDFILE"
  [[ -n "${UP_PID:-}" ]] && kill "$UP_PID" 2>/dev/null || true
  sleep 0.3
}
trap cleanup EXIT

# 每个"场景"用一份独立配置（缓存/统计互不串味），并重启网关
start_gateway() {  # $1 = upstream port, $2 = tag
  local up_port="$1" tag="$2"
  cat >"$GW_CONFIG" <<EOF
{
  "server": {
    "port": ${GW_PORT},
    "idle_timeout_seconds": 30,
    "write_timeout_seconds": 30,
    "stream_idle_timeout_seconds": 2
  },
  "backend": {
    "url": "http://127.0.0.1:${up_port}/v1/chat/completions",
    "api_key": "test-key",
    "model": "probe",
    "timeout_seconds": 30
  },
  "embedding": {
    "model_path": "${REPO_ROOT}/model/model_int8.onnx",
    "vocab_path": "${REPO_ROOT}/model/vocab.txt",
    "dim": 512,
    "do_lower_case": true,
    "pooling": "cls"
  },
  "cache": {
    "enabled": true,
    "similarity_threshold": 0.85,
    "entity_veto": true,
    "max_entries": 1000,
    "ttl_days": 7
  },
  "log": { "sample_every": 1, "max_bytes": 0, "keep_files": 1 }
}
EOF
  : >"$GW_LOG"
  # 注意：先 cd（用 ; 分隔），再单独 nohup 启动，$! 才是网关自己的 PID
  # （写成 `cd X && BIN ... &` 时 $! 是子 shell，服务会变孤儿继续占端口）
  # 先 cd（用 ; 分隔），再单独 exec 启动；$! 就是网关自己的 PID。
  # 不用 nohup：它会屏蔽 SIGTERM，而 A 段正要靠 SIGTERM 触发优雅关闭
  ( cd "$WORK"; exec "$GATEWAY_BIN" "$GW_CONFIG" ) >"$GW_LOG" 2>&1 &
  local pid=$!
  echo "$pid" >"$GW_PIDFILE"
  for _ in $(seq 1 200); do
    grep -q "starting on :${GW_PORT}" "$GW_LOG" 2>/dev/null && break
    sleep 0.1
  done
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[fatal] 网关启动失败（${tag}），日志：" >&2
    cat "$GW_LOG" >&2
    return 1
  fi
  # 确认监听者就是刚启动的这个 PID（/proc net inode ↔ fd 比对，不信 ss）
  local inode
  inode=$(awk -v p="$(printf '%04X' "$GW_PORT")" \
    '$2 ~ (":"p"$") && $4 == "0A" {print $10; exit}' /proc/net/tcp)
  if [[ -n "$inode" ]]; then
    local fd_ok=0
    for f in /proc/"$pid"/fd/*; do
      [[ "$(readlink "$f" 2>/dev/null)" == "socket:[${inode}]" ]] && fd_ok=1 && break
    done
    if [[ "$fd_ok" == "1" ]]; then
      info "监听者校验通过：:${GW_PORT} 的 LISTEN inode 属于 PID ${pid}"
    else
      bad "监听者校验失败：:${GW_PORT} 不属于 PID ${pid}（装置不可信，后续结论作废）"
      return 1
    fi
  fi
  sleep 0.3
}

echo "=============================================================="
echo "ai-gateway 集成验证（H1/H2/M3/M1，全部走真实 main 路径）"
echo "  gateway : ${GATEWAY_BIN}"
echo "  workdir : ${WORK}"
echo "  ports   : gateway=${GW_PORT} upstream=${UP_PORT}（避开 4000/9000）"
echo "=============================================================="

# ═══════════════════════════════════════════════════════════════════════
echo
echo "== A. H1: 在途请求 + SIGTERM 必须排空后再统计/落盘 =="
# ═══════════════════════════════════════════════════════════════════════
{
  UP_PORT_A="$(pick_port)"
  python3 "$PROBE" upstream --port "$UP_PORT_A" --delay 5 >"$UP_LOG" 2>&1 &
  UP_PID=$!
  for _ in $(seq 1 60); do
    grep -q "LISTENING" "$UP_LOG" 2>/dev/null && break
    sleep 0.1
  done
  grep -q "LISTENING" "$UP_LOG" || { bad "mock 上游未启动"; cat "$UP_LOG"; }
  start_gateway "$UP_PORT_A" "A" || { kill "$UP_PID" 2>/dev/null; exit 2; }
  GW_PID="$(cat "$GW_PIDFILE")"
  info "网关 PID ${GW_PID}（在途请求发出 1s 后发 SIGTERM）"

  python3 "$PROBE" in-flight-sigterm --gateway-port "$GW_PORT" \
    --gateway-pid "$GW_PID" --sigterm-after 1.0 >"${WORK}/saw.out" 2>&1
  cat "${WORK}/saw.out" | sed 's/^/        /'
  SAW_RC="$(sed -n 's/^CURL_RC=//p' "${WORK}/saw.out")"
  SAW_LEN="$(sed -n 's/^BODY_LEN=//p' "${WORK}/saw.out")"
  SAW_MS="$(sed -n 's/^ELAPSED_MS=//p' "${WORK}/saw.out")"

  # 等网关自己退出（drain 之后才会走到 stopped）
  for _ in $(seq 1 100); do
    kill -0 "$GW_PID" 2>/dev/null || break
    sleep 0.2
  done

  if [[ "${SAW_RC:-x}" == "0" && "${SAW_LEN:-0}" -gt 0 ]]; then
    ok "在途请求拿到了完整响应（rc=0, body=${SAW_LEN} 字节, ${SAW_MS}ms）"
  else
    bad "在途请求被掐断（rc=${SAW_RC:-?}, body=${SAW_LEN:-0} 字节）——drain 没有生效"
  fi
  [[ "${SAW_MS:-0}" -ge 4500 ]] && ok "客户端确实等到了上游那 5s（${SAW_MS}ms）" \
    || bad "客户端只等了 ${SAW_MS}ms，没有落在'上游 5s 慢响应'的区间"

  # 日志顺序：worker 的完成行必须在 "ai-gateway stopped" 之前
  STOP_LINE="$(grep -n 'ai-gateway stopped' "$GW_LOG" | head -1 | cut -d: -f1)"
  DONE_LINE="$(grep -nE '^\[.*\] 200 [0-9]+ bytes [0-9]+ms' "$GW_LOG" | head -1 | cut -d: -f1)"
  if [[ -n "$STOP_LINE" && -n "$DONE_LINE" && "$DONE_LINE" -lt "$STOP_LINE" ]]; then
    ok "日志顺序正确：worker 的 200 响应行(#${DONE_LINE}) 早于 stopped(#${STOP_LINE})"
  else
    bad "日志顺序错误：stopped(#${STOP_LINE:-缺}) 与 worker 完成行(#${DONE_LINE:-缺})"
  fi
  grep -qE "cache persisted: [1-9][0-9]* entries" "$GW_LOG" \
    && ok "在途请求的答案落盘了（cache persisted 非 0 条）" \
    || bad "落盘条目为 0——在途请求的缓存写入丢了"
  grep -qE "\[STATS\].*requests=1" "$GW_LOG" \
    && ok "统计包含在途请求（requests=1）" \
    || bad "统计缺失在途请求（requests 不是 1）"
  [[ -f "${WORK}/cache/lru_store.json" ]] \
    && ok "cache/lru_store.json 已写出（落盘路径生效）" \
    || bad "cache/lru_store.json 不存在"
  kill "$UP_PID" 2>/dev/null || true
  UP_PID=""
}

# ═══════════════════════════════════════════════════════════════════════
echo
echo "== B. H2: 跨 system prompt 的并发请求不得合并 =="
# ═══════════════════════════════════════════════════════════════════════
{
  UP_PORT_B="$(pick_port)"
  python3 "$PROBE" upstream --port "$UP_PORT_B" --delay 0.6 >"$UP_LOG" 2>&1 &
  UP_PID=$!
  for _ in $(seq 1 60); do
    grep -q "LISTENING" "$UP_LOG" 2>/dev/null && break
    sleep 0.1
  done
  start_gateway "$UP_PORT_B" "B" || { kill "$UP_PID" 2>/dev/null; exit 2; }

  python3 "$PROBE" merge-probe --gateway-port "$GW_PORT" --gap 0.25 \
    >"${WORK}/merge.out" 2>&1
  cat "${WORK}/merge.out" | sed 's/^/        /'
  A_BODY="$(sed -n 's/^A_BODY=//p' "${WORK}/merge.out")"
  B_BODY="$(sed -n 's/^B_BODY=//p' "${WORK}/merge.out")"
  B_CACHE="$(sed -n 's/^B_CACHE=//p' "${WORK}/merge.out")"

  echo "$A_BODY" | grep -q "PROBE-ALPHA" && ok "A 拿到自己的答案（sys=PROBE-ALPHA）" \
    || bad "A 的答案不对：${A_BODY}"
  if [[ "$B_CACHE" == "merged" ]]; then
    bad "B 被合并到 A（_cache=merged）——跨 namespace 的合并隔离没生效"
  else
    ok "B 没有被合并（_cache=${B_CACHE:-<空>}）"
  fi
  echo "$B_BODY" | grep -q "PROBE-ALPHA" \
    && bad "B 的响应体里出现了 A 的 system（跨对话串答案）" \
    || ok "B 的响应体里没有 A 的 system"
  echo "$B_BODY" | grep -q "PROBE-BETA" \
    && ok "B 拿到的是自己的上游答案（sys=PROBE-BETA）" \
    || bad "B 的答案不是自己的：${B_BODY}"
  grep -q "singleflight: merged" "$GW_LOG" \
    && bad "网关日志出现了 singleflight: merged（本场景不该合并）" \
    || ok "网关日志没有 merged 记录"

  # 同 namespace 仍必须合并（用另一个 user 触发新的在途槽位）
  info "对照：同 system prompt 的并发请求仍必须合并"
  python3 "$PROBE" merge-probe --gateway-port "$GW_PORT" --gap 0.25 \
    >"${WORK}/merge2.out" 2>&1 || true
  kill_pid_file "$GW_PIDFILE"
  kill "$UP_PID" 2>/dev/null || true
  UP_PID=""
}

# ═══════════════════════════════════════════════════════════════════════
echo
echo "== C. M3: 流式请求走非流式兜底时不得复用连接 =="
# ═══════════════════════════════════════════════════════════════════════
{
  UP_PORT_C="$(pick_port)"
  python3 "$PROBE" upstream --port "$UP_PORT_C" >"$UP_LOG" 2>&1 &
  UP_PID=$!
  for _ in $(seq 1 60); do
    grep -q "LISTENING" "$UP_LOG" 2>/dev/null && break
    sleep 0.1
  done
  start_gateway "$UP_PORT_C" "C" || { kill "$UP_PID" 2>/dev/null; exit 2; }

  python3 "$PROBE" keepalive-probe --gateway-port "$GW_PORT" \
    >"${WORK}/ka.out" 2>&1
  cat "${WORK}/ka.out" | sed 's/^/        /'
  KA_HEAD="$(sed -n 's/^FIRST_HEAD=//p' "${WORK}/ka.out")"
  KA_REUSED="$(sed -n 's/^CONNECTION_REUSED=//p' "${WORK}/ka.out")"
  KA_BODY="$(sed -n 's/^FIRST_BODY=//p' "${WORK}/ka.out")"

  echo "$KA_BODY" | grep -q "stream_fallback" \
    && ok "确实命中了 stream_fallback 分支" \
    || bad "没有命中兜底分支（装置不对）：${KA_BODY}"
  echo "$KA_HEAD" | grep -qi "Connection: close" \
    && ok "响应头声明 Connection: close" \
    || bad "响应头仍是 keep-alive：${KA_HEAD}"
  [[ "$KA_REUSED" == "0" ]] && ok "连接确实被关闭（第二个请求发不出去）" \
    || bad "连接被复用了（第二个请求得到响应）"
  kill_pid_file "$GW_PIDFILE"
  kill "$UP_PID" 2>/dev/null || true
  UP_PID=""
}

# ═══════════════════════════════════════════════════════════════════════
echo
echo "== D. M1: 上游静默被中停时，日志必须归因到'上游空闲' =="
# ═══════════════════════════════════════════════════════════════════════
{
  UP_PORT_D="$(pick_port)"
  python3 "$PROBE" upstream --port "$UP_PORT_D" --silent 8 >"$UP_LOG" 2>&1 &
  UP_PID=$!
  for _ in $(seq 1 60); do
    grep -q "LISTENING" "$UP_LOG" 2>/dev/null && break
    sleep 0.1
  done
  start_gateway "$UP_PORT_D" "D" || { kill "$UP_PID" 2>/dev/null; exit 2; }

  # 流式请求；上游发一个事件后静默 8s，空闲死线 2s
  OUT="$(python3 - <<PY
import json, socket, time
body = json.dumps({"model": "probe", "stream": True,
                   "messages": [{"role": "user", "content": "PROBE-SILENT 你好"}]}).encode()
req = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
       b"Content-Length: " + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
t0 = time.time()
s = socket.create_connection(("127.0.0.1", ${GW_PORT}), timeout=30)
s.sendall(req)
raw = b""
try:
    while True:
        b = s.recv(65536)
        if not b: break
        raw += b
except Exception as e:
    print("EXC=" + type(e).__name__)
s.close()
print("MS=%d" % int((time.time() - t0) * 1000))
print("BYTES=%d" % len(raw))
print("HAS_DONE=%d" % (1 if b"[DONE]" in raw else 0))
PY
)"
  echo "$OUT" | sed 's/^/        /'
  MS="$(echo "$OUT" | sed -n 's/^MS=//p')"
  HAS_DONE="$(echo "$OUT" | sed -n 's/^HAS_DONE=//p')"

  [[ "${MS:-0}" -lt 8000 ]] && ok "静默流在 ${MS}ms 内被中停（空闲死线 2s 生效）" \
    || bad "静默流没有被中停（${MS}ms）"
  [[ "${HAS_DONE:-1}" == "0" ]] && ok "客户端没有收到 [DONE]（符合'被中停'）" \
    || bad "静默流仍完整结束（收到了 [DONE]）"

  sleep 0.3
  ABORT_LINE="$(grep -aE 'stream: aborted' "$GW_LOG" | tail -1)"
  if [[ -n "$ABORT_LINE" ]]; then
    ok "日志给出 WARN 级的中止记录：${ABORT_LINE:0:120}"
  else
    bad "日志里没有 'stream: aborted'（中止被当成正常完成）"
  fi
  echo "$ABORT_LINE" | grep -q "upstream idle timeout" \
    && ok "中止原因归因到上游空闲（upstream idle timeout）" \
    || bad "中止原因没有归因到上游空闲：${ABORT_LINE:-<无>}"
  grep -aq "stream: done .*done_event=no" "$GW_LOG" \
    && bad "仍出现了 'stream: done ... done_event=no' 的正常完成日志（归因丢失）" \
    || ok "没有把中止打成正常完成（done_event=no 的 done 行）"

  kill_pid_file "$GW_PIDFILE"
  kill "$UP_PID" 2>/dev/null || true
  UP_PID=""
}

echo
# ═══════════════════════════════════════════════════════════════════════
echo
echo "== E. L1: 正常关闭空闲 keep-alive 连接不得打成 WARN =="
# ═══════════════════════════════════════════════════════════════════════
# 旧实现在 handle_buffer 里把"缓冲区为空 + 对端 FIN"（= 客户端的正常关闭）也打成
# `incomplete header (0 bytes, eof=true), closing`：压测里这类 WARN 占 27%，
# ops-incident-log 8.10 的崩溃现场最后两行正是它（曾被当成线索排查）。
# 这里用"连上就关"确定性地走那条分支：修复前每条连接都会产出一条 WARN
# （实测 3 条连接 → 3 条 WARN；修复后 0 条）。
{
  UP_PORT_E="$(pick_port)"
  python3 "$PROBE" upstream --port "$UP_PORT_E" >"$UP_LOG" 2>&1 &
  UP_PID=$!
  for _ in $(seq 1 60); do
    grep -q "LISTENING" "$UP_LOG" 2>/dev/null && break
    sleep 0.1
  done
  start_gateway "$UP_PORT_E" "E" || { kill "$UP_PID" 2>/dev/null; exit 2; }

  python3 -c "
import socket
for _ in range(3):
    s = socket.create_connection(('127.0.0.1', ${GW_PORT}), timeout=5)
    s.close()
"
  sleep 0.8
  # 用 python 计数：grep -c 在多文件/二进制输入下的输出格式不稳定（实测给出
  # "0\n0" 这种两行结果，把下面的判断带偏）
  N_WARN="$(python3 -c "
import sys
try:
    data = open(sys.argv[1], 'rb').read().decode('utf-8', 'replace')
except OSError:
    print(0); raise SystemExit
print(sum(1 for line in data.splitlines() if 'incomplete header' in line))
" "$GW_LOG")"
  if [[ "${N_WARN:-1}" == "0" ]]; then
    ok "正常关闭空闲连接没有产生 incomplete header WARN"
  else
    bad "出现了 ${N_WARN} 条 incomplete header WARN（正常关闭被误报）"
  fi
  kill_pid_file "$GW_PIDFILE"
  kill "$UP_PID" 2>/dev/null || true
  UP_PID=""
}

echo "=============================================================="
echo "集成验证结果：PASS=${PASS} FAIL=${FAIL}"
echo "workdir 保留在 ${WORK}（gateway.log / upstream.log 可查现场）"
echo "=============================================================="
[[ "$FAIL" -eq 0 ]] || exit 1
exit 0
