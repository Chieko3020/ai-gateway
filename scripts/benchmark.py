#!/usr/bin/env python3
"""AI Gateway 综合 Benchmark — L1 函数级 + L2 回放 + L3 并发

用法:
  python3 scripts/benchmark.py             # 默认 30/200/10 消息
  python3 scripts/benchmark.py --level 1   # 仅函数级
  python3 scripts/benchmark.py --level 2   # 仅回放
  python3 scripts/benchmark.py --level 3   # 仅并发
"""

import subprocess, json, time, sys, threading, argparse

URL = "http://127.0.0.1:3003/v1/chat/completions"
MODEL = "deepseek-v4-flash"

def call(msg, max_tokens=30):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": msg}], "max_tokens": max_tokens})
    t0 = time.time()
    r = subprocess.run(["curl", "-s", "-X", "POST", URL, "-H", "Content-Type: application/json", "-d", body],
                       capture_output=True, text=True, timeout=60)
    return int((time.time() - t0) * 1000)


def level1(num=30):
    """函数级: N 条消息 × 2 轮"""
    msgs = [
        "你好","嗨","早上好","晚上好","hello","在吗","嗨喽","你好呀","下午好","好久不见",
        "今天天气怎么样","现在几点了","你是谁","帮我写一首诗","讲一个笑话",
        "推荐一本书","1+1等于几","你叫什么名字","介绍一下Linux","中国首都是哪里",
        "Python怎么读取文件","什么是C++","解释一下递归","Linux怎么查看进程","git怎么回退",
        "怎么用vim","docker命令大全","nginx怎么配置","ssh怎么免密登录","什么是SQL注入",
    ][:num]

    print(f"\n{'='*60}")
    print(f"Level 1: 函数级 ({len(msgs)} msgs × 2 rounds)")
    print(f"{'='*60}")

    for rd, label in [(1, "R1-populate"), (2, "R2-replay")]:
        hits = misses = 0
        for i, msg in enumerate(msgs):
            lat = call(msg)
            if lat < 500: hits += 1
            else: misses += 1
            if (i + 1) % 10 == 0:
                print(f"  {label}: {i+1}/{len(msgs)} done, HIT={hits} MISS={misses}")
        print(f"  {label}: HIT={hits}/{len(msgs)} ({hits/len(msgs)*100:.0f}%)")


def level2(num=200):
    """回放: N 条消息 × 2 轮，10 语义簇"""
    msgs = [
        "你好","你好","嗨","嗨","hello","hi","您好","早上好","晚上好","下午好",
        "好久不见","在吗","嗨喽","你好呀","good morning","good evening",
        "hey there","how are you","最近怎么样","你还好吗",
        "今天天气怎么样","外面冷不冷","明天会下雨吗","天气如何","what's the weather",
        "今天温度多少","需要带伞吗","这几天会降温吗","夏天什么时候结束","今天有太阳吗",
        "会不会刮风","空气质量如何","适合出门吗","天气预报","今天湿度大吗",
        "Python怎么读取文件","Python读文件","reading files in python","read file with python",
        "什么是C++","C++和C的区别","解释一下递归","recursion explained",
        "Linux怎么查看进程","ps aux","how to list processes linux",
        "git怎么回退","git revert","git reset","怎么用vim","vim基本操作","vim basics",
        "docker命令大全","docker常用命令","nginx怎么配置","ssh怎么免密登录",
        "什么是SQL注入","sql injection explained","Python装饰器","async await python",
        "多线程vs多进程","REST API设计原则",
        "中国首都是哪里","中国的首都","北京是哪个国家的","1+1等于几","1+1","what is 1+1",
        "什么是黑洞","黑洞是怎么形成的","black hole explained","介绍一下Linux",
        "什么是Linux","傅里叶变换是什么","相对论简单解释","什么是AI","机器学习",
        "TCP三次握手","数据库索引","什么是DNS",
        "讲一个笑话","讲笑话","tell me a joke","joke","你叫什么名字","你是谁",
        "推荐一本书","推荐电影","失眠怎么办","睡不着","怎么放松",
        "今天是什么日子","喜欢的颜色","兴趣爱好","现在几点了",
        "服务器状态","在线人数","TPS多少","内存占用","服务器性能",
        "备份信息","最近备份","服务器日志","启动服务器","关闭服务器",
        "翻译hello","翻译成中文good morning","how to say 你好 in English",
        "translate apple","英语怎么说谢谢","中译英","翻译成英文",
        "推荐外卖","附近有什么好吃的","怎么煮面","简单食谱","怎么做饭",
        "怎么学英语","英语怎么提高","学编程从哪里开始","怎么找工作",
        "面试技巧","简历怎么写","健身建议","旅游推荐","周末去哪玩",
        "Python和Java哪个好","java vs python","Ubuntu好用吗","Windows还是Linux",
        "Linux怎么装docker","微服务架构","什么是k8s","CI/CD是什么",
        "GitHub怎么用","VSCode插件推荐","深度学习框架","pytorch和tensorflow",
    ][:num]

    print(f"\n{'='*60}")
    print(f"Level 2: 回放 ({len(msgs)} msgs × 2 rounds, 10 语义簇)")
    print(f"{'='*60}")

    for rd, label in [(1, "R1-populate"), (2, "R2-replay")]:
        hits = misses = 0
        for i, msg in enumerate(msgs):
            lat = call(msg)
            if lat < 500: hits += 1
            else: misses += 1
            if (i + 1) % 50 == 0:
                print(f"  {label}: {i+1}/{len(msgs)} done, HIT={hits} MISS={misses}")
        print(f"  {label}: HIT={hits}/{len(msgs)} ({hits/len(msgs)*100:.0f}%)")


def level3(threads_num=4, requests=10):
    """并发: M 线程 × N 请求"""
    msg = "你好"

    print(f"\n{'='*60}")
    print(f"Level 3: 并发 ({threads_num} threads, cached HIT)")
    print(f"{'='*60}")

    results = {"ok": 0, "fail": 0, "lats": []}
    lock = threading.Lock()

    def worker():
        for _ in range(requests):
            lat = call(msg)
            with lock:
                if lat < 1000: results["ok"] += 1
                else: results["fail"] += 1
                results["lats"].append(lat)

    # Sequential baseline
    t0 = time.time()
    for _ in range(requests): call(msg)
    seq_time = int((time.time() - t0) * 1000)
    print(f"  Sequential: {requests} req × {seq_time/requests:.0f}ms = {seq_time}ms")

    # Concurrent
    results = {"ok": 0, "fail": 0, "lats": []}
    t0 = time.time()
    threads = [threading.Thread(target=worker) for _ in range(threads_num)]
    for t in threads: t.start()
    for t in threads: t.join()
    total_time = int((time.time() - t0) * 1000)
    total_req = threads_num * requests
    qps = total_req / (total_time / 1000)

    lats = sorted(results["lats"])
    p50 = lats[len(lats)//2]
    p99 = lats[min(int(len(lats)*0.99), len(lats)-1)]

    print(f"  Concurrent: {total_req} req in {total_time}ms ({qps:.1f} QPS)")
    print(f"  P50={p50}ms P99={p99}ms")
    print(f"  Speedup: {seq_time*threads_num/max(total_time,1):.1f}x")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AI Gateway Benchmark")
    p.add_argument("--level", type=int, choices=[1, 2, 3], default=0,
                   help="1=函数级 2=回放 3=并发 (0=全部)")
    p.add_argument("--count", type=int, default=0,
                   help="消息数 (L1默认30, L2默认200, L3默认10/线程)")
    args = p.parse_args()

    if args.level in (0, 1):
        level1(args.count if args.count else 30)
    if args.level in (0, 2):
        level2(args.count if args.count else 200)
    if args.level in (0, 3):
        level3(requests=args.count if args.count else 10)
