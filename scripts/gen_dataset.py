#!/usr/bin/env python3
"""数据集生成：合成语义簇集 + Hermes 真实问答集

用途：为 ai-gateway 的语义缓存压测提供可复现的数据集。
  - 合成集：15 个语义簇 × 20 条同义表达 = 300 条，用于测"缓存能力上限"
    （同簇内语义相近，理应命中；跨簇不应命中）
  - 真实集：从 Hermes state.db 导出真实用户提问，用于测"真实可用性"

用法:
  python3 scripts/gen_dataset.py --out-dir scripts/datasets
  python3 scripts/gen_dataset.py --hermes ~/.hermes/state.db --out-dir scripts/datasets
"""

import argparse
import json
import os
import sqlite3
import sys

# 15 个语义簇，每簇 20 条同义表达（同簇内语义相近，应命中）
CLUSTERS = {
    "greeting": ["你好", "您好", "嗨", "hi", "hello", "嘿", "在吗", "有人吗", "你好呀", "早上好",
                 "早安", "晚上好", "下午好", "好久不见", "最近怎么样", "嗨喽", "哈喽", "您好呀", "喂", "在不在"],
    "weather": ["今天天气怎么样", "外面冷不冷", "明天会下雨吗", "天气如何", "今天温度多少", "需要带伞吗",
                "这几天会降温吗", "今天有太阳吗", "会不会刮风", "空气质量如何", "适合出门吗", "天气预报",
                "今天热不热", "外面在下雨吗", "明天天气好吗", "湿度大吗", "要穿外套吗", "今天多少度",
                "晚上冷不冷", "周末天气怎么样"],
    "time": ["现在几点了", "现在什么时间", "今天几号", "今天是周几", "当前时间是多少", "现在时刻",
             "今天日期", "这个月几号", "现在几点钟", "距离今天结束还有多久", "现在是上午还是下午",
             "今天星期几", "当前日期是", "告诉我现在的时间", "现在时间戳", "今天是什么日子",
             "几点了现在", "今天几月几号", "本周第几天", "现在时刻是几点"],
    "identity": ["你是谁", "你叫什么名字", "你是什么", "介绍一下你自己", "你的名字是什么", "你是什么模型",
                 "你是AI吗", "你能做什么", "你的能力范围", "你由谁开发", "你的身份", "自我介绍一下",
                 "你叫什么", "你是什么产品", "你有哪些功能", "你是聊天机器人吗", "你的定位是什么",
                 "你擅长什么", "你可以帮我做什么", "你是什么助手"],
    "py_file": ["Python怎么读取文件", "Python读文件", "用python打开文件", "python 文件读取方式",
                "Python如何写入文件", "python读取txt文件", "python open 用法", "Python 文件读写示例",
                "python怎么打开一个文件", "用 Python 读文本文件", "python读取csv", "Python 写文件",
                "python 按行读取文件", "python with open 语法", "Python 读取 json 文件",
                "python怎么保存文件", "python 文件操作", "Python 逐行读文件", "python 读文件内容",
                "Python 文件处理"],
    "cpp_basic": ["什么是C++", "C++是什么语言", "介绍一下C++", "C++ 的特点", "C++和C的区别",
                  "C++ 是面向对象的吗", "C++ 有什么用", "C++ 能做什么", "C++ 应用场景",
                  "为什么要学C++", "C++ 语言简介", "C++ 支持哪些范式", "C++ 版本有哪些",
                  "C++ 的优缺点", "C++ 适合做什么", "C++ 是什么级别的语言", "C++ 和 Java 区别",
                  "C++ 主要特性", "C++ 由谁设计", "C++ 发展历史"],
    "linux_cmd": ["Linux怎么查看进程", "linux 查看进程命令", "如何列出系统进程", "ps 命令怎么用",
                  "Linux 查看磁盘占用", "linux 查看端口占用", "top 命令怎么看", "Linux 查找文件",
                  "linux 查看内存使用", "grep 命令用法", "Linux 查看日志", "linux 解压文件",
                  "如何查看 CPU 使用率", "Linux 杀进程", "linux 查看目录大小", "awk 命令用法",
                  "Linux 设置环境变量", "linux 后台运行程序", "查看网络连接的命令", "Linux 权限修改"],
    "git_op": ["git怎么回退", "git 回退版本", "如何撤销 git 提交", "git reset 用法", "git 回滚代码",
               "git 怎么切换分支", "git 合并分支", "git 提交规范", "git 撤销修改", "git rebase 怎么用",
               "git 查看提交历史", "git 冲突怎么解决", "git 删除分支", "git 拉取远程代码",
               "git 推送失败怎么办", "git stash 用法", "git 打标签", "git 忽略文件配置",
               "git 查看差异", "git 恢复删除的文件"],
    "network": ["HTTP和HTTPS区别", "什么是TCP协议", "TCP 三次握手", "HTTP 状态码有哪些",
                "HTTPS 加密原理", "什么是DNS", "TCP 和 UDP 区别", "HTTP 请求方法",
                "什么是长连接", "Cookie 和 Session 区别", "什么是 CDN", "HTTP2 有什么改进",
                "什么是负载均衡", "OSI 七层模型", "什么是反向代理", "WebSocket 是什么",
                "TCP 粘包怎么解决", "什么是跨域", "HTTP 缓存机制", "什么是 RESTful"],
    "os_mem": ["什么是虚拟内存", "进程和线程的区别", "什么是死锁", "内存泄漏怎么排查",
               "什么是僵尸进程", "用户态和内核态区别", "什么是协程", "CPU 调度算法有哪些",
               "什么是页面置换", "进程间通信方式", "什么是内存对齐", "什么是写时复制",
               "文件描述符是什么", "什么是上下文切换", "堆和栈的区别", "什么是信号量",
               "什么是线程池", "什么是零拷贝", "什么是 DMA", "中断和轮询区别"],
    "sql": ["MySQL 索引原理", "什么是数据库事务", "SQL 查询优化", "MVCC 是什么",
            "什么是慢查询", "MySQL 主从复制", "SQL 注入怎么防", "什么是间隙锁",
            "数据库隔离级别", "什么是范式", "MySQL 存储引擎区别", "什么是回表",
            "如何设计索引", "什么是分库分表", "Redis 和 MySQL 区别", "什么是乐观锁",
            "SQL 执行计划怎么看", "什么是死锁检测", "数据库连接池作用", "什么是读写分离"],
    "recommend": ["推荐一本书", "有什么好书推荐", "推荐一本小说", "好看的书有哪些", "值得读的书",
                  "推荐一本技术书", "有哪些经典书籍", "推荐点书给我", "好书推荐一下",
                  "适合入门看的书", "推荐一本书看看", "有什么值得看的书", "书单推荐",
                  "推荐几本书", "想看书求推荐", "有哪些必读书", "推荐一本悬疑小说",
                  "推荐一本历史书", "有哪些好书值得读", "给我推荐一本书"],
    "math": ["1+1等于几", "一加一等于多少", "1加1", "计算 1+1", "1+1=?", "2的10次方是多少",
             "100除以4等于多少", "求 15 的平方", "3乘以7等于几", "100以内质数有哪些",
             "圆的面积怎么算", "什么是斐波那契数列", "计算 12*12", "9的平方根是多少",
             "sin30度等于多少", "一元二次方程求根公式", "什么是质数", "计算 25% 是多少",
             "三角形面积公式", "排列组合怎么算"],
    "translate": ["把这句话翻译成英文", "翻译成英文", "这句怎么翻译", "帮我翻译一下",
                  "中译英", "英文怎么说", "翻译这句话", "请翻译成英语", "这句话英文是什么",
                  "帮我翻译成中文", "翻译一下这段话", "怎么用英文表达", "这句话怎么用英语说",
                  "翻译这个单词", "英文翻译中文", "请把下面这句翻译", "英译中", "翻译求助",
                  "这句英文什么意思", "帮忙翻译一句"],
    "chat": ["谢谢", "多谢", "感谢", "辛苦了", "谢谢你的帮助", "太感谢了", "thanks", "谢了",
             "麻烦你了", "感谢解答", "好的谢谢", "明白了谢谢", "可以了谢谢", "谢谢啦",
             "感谢你", "非常感谢", "谢啦", "多亏了你", "谢谢解答", "感谢帮忙"],
}

# 跨簇的独立问题（不应命中任何簇）
EXTRA = [
    "怎么学习算法", "什么是动态规划", "推荐学习路线", "如何准备面试", "简历怎么写",
    "什么是微服务", "什么是容器化", "Docker 和虚拟机的区别", "什么是 K8s", "CI/CD 是什么",
    "什么是单元测试", "代码审查怎么做", "什么是设计模式", "什么是重构", "如何提高代码质量",
    "什么是敏捷开发", "技术债是什么", "如何做技术选型", "什么是灰度发布", "如何排查线上问题",
]


def gen_synthetic(out_path):
    rows = []
    for cluster, msgs in CLUSTERS.items():
        for m in msgs:
            rows.append({"q": m, "cluster": cluster})
    for m in EXTRA:
        rows.append({"q": m, "cluster": "extra"})
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows), len(CLUSTERS)


def gen_real(db_path, out_path, min_len=5, max_len=200):
    """从 Hermes state.db 导出真实用户提问（去重、过滤上下界）"""
    if not os.path.exists(db_path):
        print(f"⚠️ 未找到 {db_path}，跳过真实集", file=sys.stderr)
        return 0
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    seen, rows = set(), []
    for (content,) in con.execute(
            "SELECT content FROM messages WHERE role='user' ORDER BY timestamp"):
        if not content:
            continue
        q = content.strip()
        if not (min_len <= len(q) <= max_len):
            continue
        # 过滤含随机标识（长十六进制/数字 ID）的指令类消息
        if any(tok.isdigit() and len(tok) >= 8 for tok in q.split()):
            continue
        if q in seen:
            continue
        seen.add(q)
        rows.append({"q": q, "source": "hermes"})
    con.close()
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="scripts/datasets")
    ap.add_argument("--hermes", default=os.path.expanduser("~/.hermes/state.db"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    n_syn, n_cluster = gen_synthetic(os.path.join(args.out_dir, "synthetic.jsonl"))
    print(f"✅ synthetic.jsonl: {n_syn} 条（{n_cluster} 语义簇 × 20 + 20 独立问题）")

    n_real = gen_real(args.hermes, os.path.join(args.out_dir, "real.jsonl"))
    if n_real:
        print(f"✅ real.jsonl: {n_real} 条（来自 Hermes 真实提问，去重后）")


if __name__ == "__main__":
    main()
