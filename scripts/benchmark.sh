#!/bin/bash
# 简单压测脚本：并发请求  统计延迟和命中率
# 用法: ./scripts/benchmark.sh [并发数] [请求数]
set -e

CONCURRENCY=${1:-5}
REQUESTS=${2:-20}
# 默认打本地测试实例（4100）；线上实例是 4000，打它之前用 GATEWAY_URL 显式指定并确认影响
URL="${GATEWAY_URL:-http://localhost:4100/v1/chat/completions}"
BODY='{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"你好"}]}'

echo "=== AI Gateway Benchmark ==="
echo "并发: $CONCURRENCY  请求: $REQUESTS"
echo ""

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

START=$(date +%s%N)

# 并发发送请求
for i in $(seq 1 $REQUESTS); do
  (curl -s -o "$TMPDIR/req_$i.json" -w "%{http_code} %{time_total}\n" \
    "$URL" -H "Content-Type: application/json" -d "$BODY" \
    > "$TMPDIR/status_$i.txt" 2>/dev/null) &
  if (( i % CONCURRENCY == 0 )); then wait; fi
done
wait

END=$(date +%s%N)
ELAPSED=$(( (END - START) / 1000000 ))

# 统计
SUCCESS=0
FAIL=0
TOTAL_TIME=0
for f in "$TMPDIR"/status_*.txt; do
  read CODE TIME < "$f"
  if [ "$CODE" = "200" ]; then
    ((SUCCESS++))
    TOTAL_TIME=$(echo "$TOTAL_TIME + $TIME" | bc)
  else
    ((FAIL++))
  fi
done

echo "总耗时: ${ELAPSED}ms"
echo "成功: $SUCCESS  失败: $FAIL"
if [ $SUCCESS -gt 0 ]; then
  AVG=$(echo "scale=1; $TOTAL_TIME / $SUCCESS * 1000" | bc)
  echo "平均延迟: ${AVG}ms"
fi
echo "QPS: $(echo "scale=1; $SUCCESS * 1000 / $ELAPSED" | bc)"
