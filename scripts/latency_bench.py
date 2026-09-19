#!/usr/bin/env python3
"""延迟测量（复用 keep-alive 连接，测网关处理本身）

命中：发送一条**已在缓存里**的问题（精确命中）
未命中：发送一批**互不相同**的新问题（每次都回源 + 写缓存）

输出 p50/p95/p99/min/max，单位毫秒。
"""
import argparse
import http.client
import json
import statistics
import time


def bench(port: int, prompts, n: int):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=60)
    lat = []
    caches = []
    for i in range(n):
        prompt = prompts(i)
        body = json.dumps({'model': 'mock',
                           'messages': [{'role': 'user', 'content': prompt}]})
        t0 = time.perf_counter()
        conn.request('POST', '/v1/chat/completions', body,
                     {'Content-Type': 'application/json'})
        resp = conn.getresponse()
        payload = resp.read()
        lat.append((time.perf_counter() - t0) * 1000.0)
        try:
            caches.append(json.loads(payload).get('_cache'))
        except Exception:  # noqa: BLE001
            caches.append('?')
    return lat, caches


def report(label: str, lat, caches) -> None:
    lat_sorted = sorted(lat)
    n = len(lat_sorted)
    def pct(p): return lat_sorted[min(n - 1, int(n * p))]
    from collections import Counter
    print(f'{label}: n={n} min={lat_sorted[0]:.1f} p50={pct(0.50):.1f} '
          f'p95={pct(0.95):.1f} p99={pct(0.99):.1f} max={lat_sorted[-1]:.1f} '
          f'avg={statistics.mean(lat_sorted):.1f} (ms)  _cache={dict(Counter(caches))}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=4110)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--hit-prompt', default='MEMTEST 第 5000 号独立问题，用于灌满缓存')
    ap.add_argument('--miss-prefix', default='MISSPROBE')
    args = ap.parse_args()

    lat, caches = bench(args.port, lambda i: args.hit_prompt, args.n)
    report('命中（同一问题，精确命中）', lat, caches)

    lat, caches = bench(args.port,
                        lambda i: f'{args.miss_prefix} 第 {i} 号全新问题（未见过）',
                        args.n)
    report('未命中（互不相同，回源+写缓存）', lat, caches)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
