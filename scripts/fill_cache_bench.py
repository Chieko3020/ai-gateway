#!/usr/bin/env python3
"""缓存灌入 + RSS 采样（README 性能数据重测用）

每条请求的 user message 都不同 ⇒ 必然 miss ⇒ 必然写缓存，用来把缓存灌到指定条数；
在指定条数处记 STEADY_MARK，灌完后静默观察 idle-seconds 秒以捕获定时落盘的 RSS 窗口。
只读 /proc/<pid>/status 的 VmRSS，不干扰被测进程。
"""
import argparse
import http.client
import json
import sys
import time


def rss_mb(pid: int) -> float:
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('VmRSS'):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return -1.0
    return -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--total', type=int, default=10000)
    ap.add_argument('--marks', default='5000,10000')
    ap.add_argument('--idle-seconds', type=int, default=90)
    ap.add_argument('--prefix', default='MEMTEST')
    args = ap.parse_args()

    marks = sorted({int(x) for x in args.marks.split(',') if x.strip()})
    host = '127.0.0.1'
    conn = http.client.HTTPConnection(host, args.port, timeout=60)

    t0 = time.time()
    cur = rss_mb(args.pid)
    peak = cur
    mark_idx = 0
    errors = 0
    print('elapsed_s\tcount\tRSS_MB\tevent', flush=True)
    print(f'0.0\t0\t{cur:.1f}\tstart(idle)', flush=True)

    for i in range(1, args.total + 1):
        body = json.dumps({
            'model': 'mock',
            'messages': [{'role': 'user',
                          'content': f'{args.prefix} 第 {i} 号独立问题，用于灌满缓存'}],
        })
        try:
            conn.request('POST', '/v1/chat/completions', body,
                         {'Content-Type': 'application/json'})
            conn.getresponse().read()
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f'request failed at {i}: {exc}', file=sys.stderr, flush=True)
            conn.close()
            conn = http.client.HTTPConnection(host, args.port, timeout=60)
            continue

        cur = rss_mb(args.pid)
        peak = max(peak, cur)
        if i % 500 == 0:
            print(f'{time.time() - t0:.1f}\t{i}\t{cur:.1f}\tprogress', flush=True)
        if mark_idx < len(marks) and i == marks[mark_idx]:
            print(f'{time.time() - t0:.1f}\t{i}\t{cur:.1f}\tSTEADY_MARK', flush=True)
            mark_idx += 1

    print(f'{time.time() - t0:.1f}\t{args.total}\t{rss_mb(args.pid):.1f}\tfill_done', flush=True)

    # 静默观察：定时落盘（60s）期间的 RSS 窗口
    t1 = time.time()
    last = rss_mb(args.pid)
    while time.time() - t1 < args.idle_seconds:
        time.sleep(2)
        cur = rss_mb(args.pid)
        peak = max(peak, cur)
        if abs(cur - last) >= 0.5:
            print(f'{time.time() - t0:.1f}\t-\t{cur:.1f}\tidle_change', flush=True)
            last = cur

    print(f'{time.time() - t0:.1f}\t-\t{rss_mb(args.pid):.1f}\tfinal', flush=True)
    print(f'PEAK_MB {peak:.1f}', flush=True)
    print(f'REQUESTS {args.total} ERRORS {errors}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
