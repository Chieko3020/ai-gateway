#!/usr/bin/env bash
# 端到端集成验证：**走真实 main 路径**（真 main.cpp + 真配置 + 真 HTTP 请求 + curl）
#
# 为什么要这个脚本（而不是再加一个 ctest 单测）：
#   上一轮的真 bug（`handle_stream_request` 从不发响应头）就是这么漏掉的——
#   单元测试自己注册路由（server.set_handler(...)）绕过了 main.cpp 的组装路径，
#   17/17 全绿却完全没覆盖真实请求。本脚本反过来：启动真正的 ai-gateway 二进制，
#   用 curl 发真实请求，验证本轮四项改动在端到端链路上的行为。
#
# 覆盖：
#   A. 流式 + stream_options.include_usage -> /metrics 与日志里 token 不再是 0（第 6 项）
#   B. 流式长回答：总时长 > write_timeout_seconds 且持续有事件 -> **不被切断**（第 5 项）
#   C. 流式空闲超时：上游长时间静默 -> 中停（第 5 项的反面，证明死线仍然有效）
#   D. 非流式：连续两个请求走同一连接（keep-alive 未被写死线改动破坏）
#
# 安全约定（务必遵守）：
#   - **绝不** pkill/pgrep-kill 名为 ai-gateway 的进程（线上实例同名，端口 4000）。
#     这里只启动自己的实例、只 kill 自己记下的 PID。
#   - 端口从高位随机挑，避开 4000（线上）与 9000（示例配置）
#
# 用法：bash scripts/integration_stream_test.sh [build_dir]
#   默认 build_dir = build（本仓库常见为 build-fix/dbg2）
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-}"
if [[ -z "$BUILD_DIR" ]]; then
  for cand in build build-fix/dbg2 build-fix/rel2; do
    if [[ -x "${REPO_ROOT}/${cand}/src/ai-gateway" ]]; then
      BUILD_DIR="${REPO_ROOT}/${cand}"
      break
    fi
  done
fi
[[ -n "$BUILD_DIR" && -x "${BUILD_DIR}/src/ai-gateway" ]] || {
  echo "[fatal] 找不到 ai-gateway 二进制，请先构建，或把构建目录作为第一个参数" >&2
  exit 2
}
GATEWAY_BIN="${BUILD_DIR}/src/ai-gateway"

WORK="$(mktemp -d /tmp/agw-integration.XXXXXX)"
PIDFILE="${WORK}/gateway.pid"
LOG="${WORK}/gateway.log"
UPSTREAM_LOG="${WORK}/upstream.log"
CONFIG="${WORK}/gateway.json"

# 端口：从 43000 起随机挑，核对 4000/9000 不在其中
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

cleanup() {
  # 恢复被临时移开的缓存文件（见下方"隔离缓存"）
  if [[ -n "${CACHE_BAK:-}" && -f "$CACHE_BAK" && ! -f "$REPO_ROOT/cache/lru_store.json" ]]; then
    mv "$CACHE_BAK" "$REPO_ROOT/cache/lru_store.json" 2>/dev/null || true
  fi
  # 只 kill 自己写下的 PID；绝不按进程名 kill
  if [[ -f "$PIDFILE" ]]; then
    local pid
    pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.2
      done
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  [[ -n "${UP_PID:-}" ]] && kill "$UP_PID" 2>/dev/null || true
  sleep 0.3
}
trap cleanup EXIT

cat >"$CONFIG" <<EOF
{
  "server": {
    "port": ${GW_PORT},
    "idle_timeout_seconds": 30,
    "write_timeout_seconds": 5,
    "stream_idle_timeout_seconds": 3
  },
  "backend": {
    "url": "http://127.0.0.1:${UP_PORT}/v1/chat/completions",
    "api_key": "test-key",
    "model": "mock-model",
    "timeout_seconds": 120
  },
  "embedding": {
    "model_path": "model/model_int8.onnx",
    "vocab_path": "model/vocab.txt",
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

echo "== 启动 mock 上游（SSE）与网关（真实 main 路径） =="
echo "  gateway  : ${GATEWAY_BIN}（端口 ${GW_PORT}，PID 文件 ${PIDFILE}）"
echo "  upstream : 端口 ${UP_PORT}"
echo "  workdir  : ${WORK}"

# ---- mock 上游：脚本目录下的 Python mock（见 scripts/mock_sse_upstream.py）----
python3 "${REPO_ROOT}/scripts/mock_sse_upstream.py" --port "$UP_PORT" \
  >"$UPSTREAM_LOG" 2>&1 &
UP_PID=$!
for _ in $(seq 1 50); do
  grep -q "LISTENING" "$UPSTREAM_LOG" 2>/dev/null && break
  sleep 0.1
done
grep -q "LISTENING" "$UPSTREAM_LOG" || { echo "[fatal] mock 上游未启动" >&2; cat "$UPSTREAM_LOG"; exit 2; }

# ---- 隔离缓存 ----
# 网关以仓库根为 cwd，会读取 cache/lru_store.json。仓库里遗留的条目会让"实体否决"
# 这类断言失真——实测：旧条目里已存在 DMA，于是"什么是DMA"命中了它自己那条
# （sim=1.0 且实体一致，这是正确行为），看起来却像否决规则失效。
# 测试期间先把它移开，退出时（cleanup）恢复。
CACHE_BAK=""
if [[ -f "$REPO_ROOT/cache/lru_store.json" ]]; then
  CACHE_BAK="${WORK}/lru_store.json.bak"
  mv "$REPO_ROOT/cache/lru_store.json" "$CACHE_BAK"
fi

# ---- 网关：在仓库根目录启动（模型与 cache/ 路径相对 cwd）----
( cd "$REPO_ROOT" && exec "$GATEWAY_BIN" "$CONFIG" ) >"$LOG" 2>&1 &
GW_PID=$!
echo "$GW_PID" >"$PIDFILE"
for _ in $(seq 1 100); do
  # 网关日志里出现 listening 即可（main 在 server.run 前打 "starting on :port"）
  grep -q "starting on :${GW_PORT}" "$LOG" 2>/dev/null && break
  sleep 0.1
done
if ! kill -0 "$GW_PID" 2>/dev/null; then
  echo "[fatal] 网关启动失败，日志：" >&2; cat "$LOG" >&2; exit 2
fi
sleep 0.5

BASE="http://127.0.0.1:${GW_PORT}/v1/chat/completions"

# ---------------------------------------------------------------------------
echo
echo "== A. 流式 + include_usage：token 统计从 0 变为真实值（第 6 项） =="
# ---------------------------------------------------------------------------
T0=$(date +%s%N)
STREAM_BODY="$(curl -sS -N -X POST "$BASE" -H 'Content-Type: application/json' -d '{
  "model":"mock","stream":true,
  "stream_options":{"include_usage":true},
  "messages":[{"role":"user","content":"A-流式usage测试"}]}' 2>&1)"
STREAM_RC=$?
T1=$(date +%s%N)
if [[ $STREAM_RC -ne 0 ]]; then
  bad "curl 流式请求失败：${STREAM_BODY}"
else
  info "流式响应（前 3 行 / 末 2 行）："
  echo "$STREAM_BODY" | head -3 | sed 's/^/        /'
  echo "$STREAM_BODY" | tail -2 | sed 's/^/        /'
  echo "$STREAM_BODY" | grep -q "usage" && ok "响应里含 usage 事件（透传未被破坏）" \
    || bad "响应里没有 usage 事件"
  echo "$STREAM_BODY" | grep -q "\[DONE\]" && ok "响应含 [DONE] 终止事件" \
    || bad "响应缺少 [DONE]"
  # 从网关日志里读 token（record_stream 的落日志口径）
  sleep 0.3
  LINE="$(grep -o 'stream: done .*tokens_in=[0-9]* tokens_out=[0-9]* usage_seen=[a-z]*' "$LOG" | tail -1)"
  if [[ -n "$LINE" ]]; then
    info "网关日志：${LINE}"
    TIN="$(echo "$LINE" | sed -n 's/.*tokens_in=\([0-9]*\).*/\1/p')"
    TOUT="$(echo "$LINE" | sed -n 's/.*tokens_out=\([0-9]*\).*/\1/p')"
    if [[ "${TIN:-0}" -gt 0 && "${TOUT:-0}" -gt 0 ]]; then
      ok "流式 token 计数非 0（in=${TIN} out=${TOUT}）——修复前这两项恒为 0"
    else
      bad "流式 token 计数仍为 0（in=${TIN} out=${TOUT}）"
    fi
    echo "$LINE" | grep -q "usage_seen=yes" && ok "usage 解析标记为 seen=yes" \
      || bad "usage_seen 不是 yes"
  else
    bad "网关日志里找不到 'stream: done ... tokens_in=' 行"
    info "网关日志尾部："; tail -15 "$LOG" | sed 's/^/        /'
  fi
  # /metrics 里的 token 总量
  M="$(curl -sS "http://127.0.0.1:${GW_PORT}/metrics" 2>/dev/null)"
  if echo "$M" | grep -qE 'ai_gateway_tokens_(prompt|completion)_total'; then
    echo "$M" | grep -E 'ai_gateway_tokens_(prompt|completion)_total' | sed 's/^/        /'
    TP="$(echo "$M" | grep -E 'ai_gateway_tokens_prompt_total' | awk '{print $2}' | head -1)"
    TC="$(echo "$M" | grep -E 'ai_gateway_tokens_completion_total' | awk '{print $2}' | head -1)"
    if [[ "${TP:-0}" != "0" && "${TC:-0}" != "0" ]]; then
      ok "/metrics 的 prompt=${TP} completion=${TC}（都非 0）"
    else
      bad "/metrics 的 token 计数仍为 0（prompt=${TP} completion=${TC}）"
    fi
  else
    bad "/metrics 未暴露 token 指标"
  fi
  info "流式整段耗时 $(( (T1 - T0) / 1000000 ))ms"
fi

# ---------------------------------------------------------------------------
echo
echo "== B. 流式长回答：总时长超过 write_timeout_seconds(5s) 不被切断（第 5 项） =="
# ---------------------------------------------------------------------------
# mock 上游对 "B-长流" 会每 0.7s 发一个事件、共 12 个事件（≈8.4s）：
#   总时长 8.4s > write_timeout_seconds=5s（旧实现：整条响应的总死线，第 5s 被切断）
#   事件间隔 0.7s < stream_idle_timeout_seconds=3s（新实现：空闲死线，不触发）
T0=$(date +%s%N)
LONG_BODY="$(curl -sS -N -X POST "$BASE" -H 'Content-Type: application/json' -d '{
  "model":"mock","stream":true,
  "stream_options":{"include_usage":true},
  "messages":[{"role":"user","content":"B-长流"}]}' 2>&1)"
LONG_RC=$?
T1=$(date +%s%N)
LONG_MS=$(( (T1 - T0) / 1000000 ))
N_EVENTS="$(echo "$LONG_BODY" | grep -c '^data: ' || true)"
info "耗时 ${LONG_MS}ms，收到 data: 事件 ${N_EVENTS} 个"
if [[ $LONG_RC -ne 0 ]]; then
  bad "curl 长流请求失败：${LONG_BODY}"
else
  [[ "$LONG_MS" -gt 5000 ]] && ok "整段耗时 ${LONG_MS}ms 已超过 write_timeout_seconds=5s" \
    || bad "整段耗时只有 ${LONG_MS}ms，没有落在'旧实现会切断'的区间"
  echo "$LONG_BODY" | grep -q '\[DONE\]' && ok "长流完整收到 [DONE]（未被切断）" \
    || bad "长流被截断：没有 [DONE]"
  [[ "$N_EVENTS" -ge 12 ]] && ok "收到全部 ${N_EVENTS} 个事件（mock 发 12 个 + usage）" \
    || bad "只收到 ${N_EVENTS} 个 data 事件（应 >= 12）"
  grep -q 'stream: aborted' "$LOG" && bad "网关日志里出现了 aborted（长流被中途放弃）" \
    || ok "网关日志无 aborted 记录"
  grep -o 'stream: done .*tokens_in=[0-9]* tokens_out=[0-9]*' "$LOG" | tail -1 | sed 's/^/        /'
fi

# ---------------------------------------------------------------------------
echo
echo "== C. 流式空闲超时：上游静默超过 stream_idle_timeout_seconds(3s) 必须中停 =="
# ---------------------------------------------------------------------------
# mock 对 "C-静默" 先发一个事件，然后静默 10s：
#   空闲死线 3s 到点 -> 网关中停上游、关闭连接（curl 提前结束，拿不到 [DONE]）
T0=$(date +%s%N)
SILENT_BODY="$(timeout 20 curl -sS -N -X POST "$BASE" -H 'Content-Type: application/json' -d '{
  "model":"mock","stream":true,
  "messages":[{"role":"user","content":"C-静默"}]}' 2>&1)"
SILENT_RC=$?
T1=$(date +%s%N)
SILENT_MS=$(( (T1 - T0) / 1000000 ))
info "耗时 ${SILENT_MS}ms（rc=${SILENT_RC}），收到：$(echo "$SILENT_BODY" | tr '\n' ' ' | head -c 120)"
if [[ "$SILENT_MS" -lt 10000 ]]; then
  ok "静默流在 ${SILENT_MS}ms 内被中停（空闲死线 3s 生效，没有一直挂着）"
else
  bad "静默流没有被中停（耗时 ${SILENT_MS}ms，说明空闲死线没生效）"
fi
echo "$SILENT_BODY" | grep -q '\[DONE\]' && bad "静默流仍收到 [DONE]（不该完整结束）" \
  || ok "静默流未收到 [DONE]（符合'中停'预期）"
# ---- 归因断言（本轮修复：此处原来是 info，属于**假覆盖**） ----
# 此前这一行写成 info，于是"日志给出 idle 原因"这条断言从未真正生效：修复前
# SsePassthroughSink 用 `(void)reason` 丢掉了 StreamAbortReason，日志变成
# "stream: done ... done_event=no keep_alive=yes"（不打 WARN、不计 aborted、
# 还把超时值算进 TTFT 样本池）——而脚本照样打印一行 info 看起来"通过"。
ABORT_LINE="$(grep -aE 'stream: aborted' "$LOG" | tail -1)"
if [[ -n "$ABORT_LINE" ]]; then
  info "中止日志：${ABORT_LINE}"
  ok "日志给出 WARN 级中止记录（stream: aborted）"
else
  bad "日志里没有 'stream: aborted'（中止被当成正常完成）"
fi
echo "$ABORT_LINE" | grep -q 'upstream idle timeout' \
  && ok "中止原因归因到上游空闲（upstream idle timeout）" \
  || bad "中止原因未归因到上游空闲：${ABORT_LINE:-<无>}"
grep -aq 'stream: done .*done_event=no' "$LOG" \
  && bad "出现了 'stream: done ... done_event=no' 的正常完成日志（归因丢失）" \
  || ok "没有把中止打成正常完成（无 done_event=no 的 done 行）"
# TTFT 样本池不得被中止样本污染：A 段 1 条 + B 段 1 条正常流，C 段的中止流不计
BSAMP="$(curl -sS "http://127.0.0.1:${GW_PORT}/metrics" 2>/dev/null \
  | awk '/^ai_gateway_bypass_latency_samples/ {print $2}' | head -1)"
if [[ -n "$BSAMP" ]]; then
  info "TTFT/旁路样本数=${BSAMP}"
  [[ "${BSAMP:-0}" -le 2 ]] && ok "中止流没有进入 TTFT 样本池（样本数=${BSAMP} ≤ 2）" \
    || bad "中止流污染了 TTFT 样本池（样本数=${BSAMP} > 2）"
fi

# ---------------------------------------------------------------------------
echo
echo "== D. 非流式：同一连接两个请求（写死线改动未破坏 keep-alive） =="
# ---------------------------------------------------------------------------
NL="$(curl -sS -o /dev/null -w '%{http_code}\n%{num_connects}\n' \
  -X POST "$BASE" -H 'Content-Type: application/json' \
  --next -X POST "$BASE" -H 'Content-Type: application/json' \
  -d '{"model":"mock","messages":[{"role":"user","content":"D-非流式-1"}]}' \
  -d '{"model":"mock","messages":[{"role":"user","content":"D-非流式-2"}]}' 2>&1)"
echo "$NL" | sed 's/^/        /'
CODE_COUNT="$(echo "$NL" | grep -c '^200$' || true)"
[[ "$CODE_COUNT" -ge 1 ]] && ok "非流式请求返回 200（共 ${CODE_COUNT} 个）" \
  || bad "非流式请求未返回 200：${NL}"

# ---------------------------------------------------------------------------
echo
echo "== E. 缓存 + 实体否决（真实 main 路径上的 DMA/DNS） =="
# ---------------------------------------------------------------------------
# 先写入 "什么是DNS" 的答案，再用 "什么是DMA" 查询：
#   向量层：lower+cls 下两者余弦 0.58（低于 0.85），实体否决是第二道闸；
#   这里断言的是"不得返回 DNS 的答案"
Q1="$(curl -sS -X POST "$BASE" -H 'Content-Type: application/json' -d '{
  "model":"mock","messages":[{"role":"user","content":"什么是DNS"}]}')"
sleep 0.2
Q2="$(curl -sS -X POST "$BASE" -H 'Content-Type: application/json' -d '{
  "model":"mock","messages":[{"role":"user","content":"什么是DMA"}]}')"
info "第一次（什么是DNS）：$(echo "$Q1" | head -c 160)"
info "第二次（什么是DMA）：$(echo "$Q2" | head -c 160)"
echo "$Q2" | grep -q '"_cache":"hit"' && bad "DMA 查询命中了缓存（应未命中）" \
  || ok "DMA 查询未命中缓存（_cache 不是 hit）"
# 同一问题再问一次必须命中（证明缓存本身在工作）
Q3="$(curl -sS -X POST "$BASE" -H 'Content-Type: application/json' -d '{
  "model":"mock","messages":[{"role":"user","content":"什么是DNS"}]}')"
echo "$Q3" | grep -q '"_cache":"hit"' && ok "重复的 DNS 查询命中缓存（缓存功能正常）" \
  || bad "重复的 DNS 查询未命中缓存：$(echo "$Q3" | head -c 160)"
grep -o 'cache: VETO by entity mismatch.*' "$LOG" | tail -2 | sed 's/^/        /'

# ---------------------------------------------------------------------------
echo
echo "=============================================================="
echo "集成验证结果：PASS=${PASS} FAIL=${FAIL}"
echo "workdir 保留在 ${WORK}（gateway.log 可查完整现场）"
echo "=============================================================="
[[ "$FAIL" -eq 0 ]] || exit 1
exit 0
