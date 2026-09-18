#!/usr/bin/env python3
"""AI Gateway 压测脚本 — 命中判定基于网关返回的 `_cache` 字段
（旧版用"延迟 < 800ms"猜测命中，阈值一旦失准命中率就整体偏移）

目标地址：默认 http://127.0.0.1:4100/v1/chat/completions，即本地测试实例。
指向线上实例（监听 4000）之前必须显式传 --url 并确认影响：压测流量会真实写进
线上缓存（污染命中率与缓存内容），并消耗真实上游 token 与费用。
历史事故：默认值曾写成 4000，一次压测把流量打进了线上缓存。

用法:
  python3 scripts/benchmark.py --level 2 --dataset scripts/datasets/synthetic.jsonl
  python3 scripts/benchmark.py --level 2 --dataset scripts/datasets/real.jsonl --out results/replay_real.json
  python3 scripts/benchmark.py --level 3 --concurrency 4
  # 只有在明确要打线上时才显式指定（端口 4000）：
  python3 scripts/benchmark.py --level 2 --url http://127.0.0.1:4000/v1/chat/completions
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

# 默认打本地测试实例（测试实例端口 4100；线上实例是 4000，不要用它当默认值）
DEFAULT_URL = "http://127.0.0.1:4100/v1/chat/completions"
MODEL = "deepseek-v4-flash"


def call(url, msg, max_tokens=30, timeout=90):
    """发一次请求。返回 (latency_ms, cache_status, body)

    cache_status 取自响应体 `_cache` 字段：hit / miss / bypass / merged / None(未知)
    """
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": msg}],
                       "max_tokens": max_tokens})
    t0 = time.time()
    try:
        r = subprocess.run(
            ["curl", "-s", "-X", "POST", url,
             "-H", "Content-Type: application/json", "-d", body],
            capture_output=True, text=True, timeout=timeout)
        lat = int((time.time() - t0) * 1000)
        try:
            status = json.loads(r.stdout).get("_cache")
        except Exception:
            status = None
        return lat, status, r.stdout
    except subprocess.TimeoutExpired:
        return timeout * 1000, "timeout", ""


def load_dataset(path, limit=None):
    msgs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                msgs.append(obj.get("q", "") if isinstance(obj, dict) else str(obj))
            except Exception:
                msgs.append(line)
    return msgs[:limit] if limit else msgs


def latency_stats(lats):
    if not lats:
        return {}
    xs = sorted(lats)
    n = len(xs)
    return {
        "n": n,
        "min": xs[0],
        "p50": xs[n // 2],
        "p95": xs[min(n - 1, int(n * 0.95))],
        "max": xs[-1],
        "avg": round(sum(xs) / n, 1),
    }


def replay(url, msgs, rounds=2, label="", quiet=False, first_round_sleep=0.0):
    """回放数据集 rounds 轮，返回每轮统计"""
    per_round = []
    for rd in range(1, rounds + 1):
        lat_hit, lat_miss, lat_bypass, lat_merged, lat_unknown = [], [], [], [], []
        for i, m in enumerate(msgs):
            lat, st, _ = call(url, m)
            if st == "hit":
                lat_hit.append(lat)
            elif st == "merged":
                # 请求合并：与并发的同义请求共享一次上游调用，本就不是缓存命中，
                # 因此不计入命中率分子（口径与网关 /stats 的 report() 一致）
                lat_merged.append(lat)
            elif st == "bypass":
                lat_bypass.append(lat)
            elif st in ("miss", "timeout"):
                lat_miss.append(lat)
            else:
                lat_unknown.append(lat)
            if not quiet and (i + 1) % 40 == 0:
                print(f"  {label}R{rd}: {i+1}/{len(msgs)} "
                      f"hit={len(lat_hit)} miss={len(lat_miss)}")
            if rd == 1 and first_round_sleep:
                time.sleep(first_round_sleep)

        n_hit, n_miss = len(lat_hit), len(lat_miss)
        n_bypass, n_merged = len(lat_bypass), len(lat_merged)
        n_unknown = len(lat_unknown)
        # 命中率分母排除旁路流量（工具调用/流式本不可缓存）与合并命中（本就不是缓存命中）
        denom = n_hit + n_miss
        row = {
            "round": rd,
            "n": len(msgs),
            "hit": n_hit,
            "miss": n_miss,
            "bypass": n_bypass,
            "merged": n_merged,
            "unknown": n_unknown,
            "hit_rate": round(n_hit / denom * 100, 1) if denom else 0.0,
            "latency_hit": latency_stats(lat_hit),
            "latency_miss": latency_stats(lat_miss),
        }
        per_round.append(row)
        print(f"  {label}R{rd}: hit={n_hit} miss={n_miss} bypass={n_bypass} "
              f"merged={n_merged} unknown={n_unknown} 命中率={row['hit_rate']}%"
              + (f"  命中p50={row['latency_hit'].get('p50')}ms "
                 f"未命中p50={row['latency_miss'].get('p50')}ms"
                 if row["latency_hit"] else ""))
    return per_round


def level_concurrent(url, msgs, concurrency=4, rounds_per_thread=3):
    """并发验证：多线程同时发相同消息，检查 singleflight 合并与错误率"""
    results = []
    lock = threading.Lock()

    def worker(idx):
        local = []
        for _ in range(rounds_per_thread):
            for m in msgs:
                lat, st, _ = call(url, m)
                local.append((lat, st))
        with lock:
            results.extend(local)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    lats = [r[0] for r in results]
    hit = sum(1 for r in results if r[1] == "hit")
    miss = sum(1 for r in results if r[1] in ("miss", "timeout"))
    # 并发场景下一次上游调用被多条同义请求共享，非 leader 拿到 _cache:"merged"
    merged = sum(1 for r in results if r[1] == "merged")
    return {
        "concurrency": concurrency,
        "requests_per_thread": rounds_per_thread * len(msgs),
        "total_requests": len(results),
        "wall_seconds": round(wall, 2),
        "throughput_rps": round(len(results) / wall, 1) if wall > 0 else 0,
        "hit": hit,
        "miss": miss,
        "merged": merged,
        "latency": latency_stats(lats),
    }


def warn_if_not_local(url):
    """非本机目标（尤其线上部署实例）时给出显式警告：这类压测会真实写缓存、烧 token"""
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""
    if host in ("127.0.0.1", "localhost", "::1"):
        return
    print(f"⚠️  目标 {url} 不是本机（host={host}）：压测流量会写入该实例的缓存并"
          f"消耗真实上游 token/费用；若不是有意为之，请改回 "
          f"--url {DEFAULT_URL}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL,
                    help=f"网关地址（默认 {DEFAULT_URL}，本地测试实例；"
                         f"指向线上实例前请确认影响）")
    ap.add_argument("--dataset", default="scripts/datasets/synthetic.jsonl")
    ap.add_argument("--level", type=int, choices=[1, 2, 3], default=2)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="仅取数据集前 N 条（0=全部）")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    msgs = load_dataset(args.dataset, args.limit or None)
    if not msgs:
        print(f"❌ 数据集为空: {args.dataset}", file=sys.stderr)
        return 1

    warn_if_not_local(args.url)
    print(f"{'='*64}\n数据集: {args.dataset} ({len(msgs)} 条) | 目标: {args.url}\n{'='*64}")

    report = {"url": args.url, "dataset": args.dataset, "n": len(msgs)}

    if args.level in (1, 2):
        rounds = args.rounds if args.level == 2 else 2
        report["replay"] = replay(args.url, msgs, rounds=rounds, quiet=args.quiet)
        # 两轮合计
        tot_hit = sum(r["hit"] for r in report["replay"])
        tot_miss = sum(r["miss"] for r in report["replay"])
        tot_bypass = sum(r["bypass"] for r in report["replay"])
        tot_merged = sum(r.get("merged", 0) for r in report["replay"])
        denom = tot_hit + tot_miss
        report["combined"] = {
            "rounds": rounds,
            "total_requests": sum(r["n"] for r in report["replay"]),
            "hit": tot_hit,
            "miss": tot_miss,
            "bypass": tot_bypass,
            "merged": tot_merged,
            "hit_rate": round(tot_hit / denom * 100, 1) if denom else 0.0,
        }
        print(f"\n合计: hit={tot_hit} miss={tot_miss} bypass={tot_bypass} "
              f"merged={tot_merged} 命中率={report['combined']['hit_rate']}%")

    if args.level == 3:
        probe = msgs[:3]
        report["concurrent"] = level_concurrent(args.url, probe, args.concurrency)
        c = report["concurrent"]
        print(f"\n并发: {c['concurrency']} 线程 × {c['requests_per_thread']} 请求 "
              f"= {c['total_requests']} | 吞吐={c['throughput_rps']} rps "
              f"| p50={c['latency'].get('p50')}ms p95={c['latency'].get('p95')}ms "
              f"| merged={c.get('merged', 0)}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 结果已写入 {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
