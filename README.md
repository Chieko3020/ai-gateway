# AI Gateway — C++20 语义缓存透明代理

- 部署在客户端与 LLM API 之间，通过语义缓存减少重复的 API 调用节省 token
- epoll ET + 非阻塞 I/O + 单 Reactor + 线程池架构，适配并发场景
- HNSW 图索引进行缓存查询，近似 O(log N) 检索
- 本地 ONNX Runtime 运行 INT8 量化 bge-small-zh 模型，进程内推理
- LLM 后端独立配置，支持任意 OpenAI 兼容 API
- LRU + TTL 内存缓存 + JSON 持久化，重启不丢缓存，默认 7 天自动过期
- 开发环境：Ubuntu 24.04 LTS, GCC 13.3, CMake 3.28

## 项目简介

客户端每日处理大量用户消息，其中语义重复或相似的请求反复调用 LLM API，造成不必要的费用。AI Gateway 作为透明代理层插入客户端与 LLM 后端之间，通过 C++ ONNX Runtime 进程内向量化用户消息、HNSW 图索引检索历史缓存、余弦相似度判断是否命中，命中时直接返回缓存，未命中时转发 API 处理消息并将结果写入缓存。相较于现有方案（GPTCache / LiteLLM），本项目的 C++ 实现更为轻量。

## 功能特性

### 核心功能
- **语义缓存**: 基于 bge-small-zh Embedding 的向量检索 + 余弦相似度匹配，非精确文本也能命中
- **安全过滤**: 请求/响应双向过滤（反注入检测、URL 拦截、关键词屏蔽、长度截断）
- **实时统计**: 每 60 秒输出命中率/token 消耗/费用节省/延迟统计到 systemd journal
- **命名空间隔离**: 不同 system prompt 缓存自动隔离，防止不同对话交叉污染
- **定时持久化**: 每 60 秒自动保存缓存到磁盘，意外宕机不丢数据

### 技术特性
- **并发模型**: 单 Reactor + 线程池，主线程管理连接，线程池处理缓存和 LLM 转发
- **向量检索**: 按论文实现 HNSW 图索引（分层结构、几何分布层数、启发式邻居选择、邻居满时收缩重选）。实测（见 `tests/recall_bench`）：320 向量下自检索 top-1 **99.4%**、top-3 召回率 99.7%；1000 / 5000 / 10000 向量下自检索 top-1 均为 **100%**，top-3 召回率 99.7% / 93.8% / 81.7%（`ef_search=50` 固定时规模增大导致束搜索覆盖不足，属 HNSW 固有特性；缓存只需 top-1，故不影响命中），相对暴力检索加速比 1.16x / 2.74x / 4.75x
- **存储引擎**: LRU + TTL 缓存管理，JSON 持久化
- **嵌入推理**: C++ ONNX Runtime 进程内 INT8 量化推理，零外部依赖
- **模型量化**: `scripts/quantize.py` — HuggingFace → FP32 ONNX → 动态量化 INT8 (~90MB→~23MB)
- **连接池**: libcurl Keep-Alive 复用 TCP 连接

## 技术栈

| 组件 | 选型 |
|------|------|
| 编程语言 | C++20 |
| JSON | nlohmann/json |
| HTTP 客户端 | libcurl |
| Embedding 推理 | ONNX Runtime C API + bge-small-zh-v1.5 INT8 (512d) |
| 构建系统 | CMake 3.16+ |

## 项目结构

```
ai-gateway/
├── include/                     # 头文件
│   ├── common/                  #   公共组件
│   │   ├── config.h             #     JSON 配置加载
│   │   ├── logger.h             #     日志宏
│   │   ├── log_file.h           #     日志输出
│   │   ├── types.h              #     公共类型（ErrorCode）
│   │   ├── curl_client.h        #     RAII libcurl 封装
│   │   └── thread_pool.h        #     线程池
│   ├── server/                  #   模块1: HTTP 接入
│   │   ├── http_server.h        #     epoll ET + 线程池调度
│   │   ├── connection_handler.h #     请求管道（解析、路由、处理、响应）
│   │   ├── request.h            #     HTTP/1.1 请求解析
│   │   ├── response.h           #     HTTP 响应构造
│   │   ├── filter.h             #     安全过滤器
│   │   └── router.h             #     URL 路由
│   ├── cache/                   #   模块2: 语义缓存
│   │   ├── cache_engine.h       #     缓存协调器（根据命名空间隔离）
│   │   ├── lru_store.h          #     LRU + TTL 缓存存储
│   │   ├── hnsw_index.h         #     HNSW 图索引（header-only）
│   │   ├── onnx_embedding.h     #     ONNX Runtime 嵌入推理 + 贪心分词
│   │   ├── embedding.h          #     HTTP Embedding 客户端（备用）
│   │   └── vector_index.h       #     暴力搜索索引（已弃用）
│   ├── backend/                 #   模块3: LLM 后端
│   │   └── llm_client.h         #     LLM API 客户端（Keep-Alive 连接池）
│   └── stats/                   #   模块4: 统计
│       └── stats.h              #     命中率/token/费用/延迟统计
├── src/                         # 源文件
│   ├── main.cpp                 #   入口：组装 + 启动 + 信号处理
│   ├── common/                  #   config.cpp, log_file.cpp
│   ├── server/                  
│   ├── cache/                   
│   ├── backend/                 
│   └── stats/                   
├── model/                       # 内嵌模型文件
│   ├── model_int8.onnx          #   INT8 量化 bge-small-zh (23MB)
│   └── vocab.txt                #   词表 (107KB)
├── tests/                       # 单元测试
├── config/
│   ├── gateway.example.json     #   配置模板
│   └── gateway.env              #   API Key
├── scripts/
│   ├── ai-gateway.service       #   systemd 服务
│   └── benchmark.py             #   综合压测脚本
├── .gitignore
├── CMakeLists.txt
└── README.md
```

### 核心模块

#### server — HTTP 接入层
- **HttpServer**: epoll ET 主循环 + 线程池调度
- **ConnectionHandler**: 请求管道（解析、路由、处理、响应）
- **Filter**: 安全过滤（反注入/URL/关键词/长度）
- **Router**: URL 路由表（METHOD + path 精确匹配）
- **Request/Response**: HTTP/1.1 解析与响应构造

#### cache — 语义缓存引擎
- **CacheEngine**: 编排缓存流程（Embedding 搜索，命中/未命中判断，命名空间隔离）
- **HnswIndex**: HNSW 图索引（header-only；按论文实现启发式邻居选择与邻居收缩，`shared_mutex` 保护读写并发，visited 标记线程本地复用）
- **OnnxEmbedding**: C++ ONNX Runtime 进程内推理 + WordPiece 词表贪心分词
- **LruStore**: LRU + TTL 内存缓存 + JSON 持久化

#### backend — LLM 转发
- **LlmClient**: LLM API 客户端（Keep-Alive 连接池复用 TCP）

#### stats — 统计
- **Stats**: 命中率/token/费用/延迟统计，每 60s 定期输出

## 架构设计

```
客户端 (QQ Bot / Web / API)
    │  POST /v1/chat/completions
    ▼
┌────────────────────────────────────────────────────────────────┐
│  HTTP 接入层 (epoll ET + 非阻塞 IO)                              │
│  模块1: ConnectionHandler (解析、路由、过滤)                      │
├────────────────────────────────────────────────────────────────┤
│  安全过滤器 (MessageFilter)                                      │
│  模块1: 反注入/URL/关键词/长度 (双向)                              │
├────────────────────────────────────────────────────────────────┤
│  语义缓存                                                        │
│  模块2: CacheEngine                                              │
│    1. C++ ONNX Runtime 进程内向量化 (WordPiece 词表 + INT8 推理, 512d)  │
│    2. HNSW 图索引检索 Top-K 相似条目                              │
│    3. max(cosine) >= 0.80                                        │
│       命中: 返回 LruStore 中的缓存回复                             │
│       未命中: 继续                                                │
├────────────────────────────────────────────────────────────────┤
│  LLM 后端转发                                                    │
│  模块3: LlmClient (Keep-Alive 连接池)                             │
│  模块4: Stats (命中率/token/费用/延迟)                            │
└────────────────────────────────────────────────────────────────┘
    │  ↑
    │  缓存写入（未命中后：HNSW 索引 + LruStore 双向存入）
    ▼
┌──────────────────────────────────────┐
│  LLM 后端 (可替换)                    │
│  DeepSeek / OpenAI / Ollama / vLLM   │
└──────────────────────────────────────┘
```

## 编译和运行

### 依赖

```bash
# 系统库
sudo apt install build-essential cmake g++-13 \
    libcurl4-openssl-dev nlohmann-json3-dev

# ONNX Runtime C 库（pip 安装后自动可用）
pip install onnxruntime
# 头文件从 GitHub Release 获取
sudo cp onnxruntime_c_api.h onnxruntime_error_code.h /usr/local/include/
# 动态库通常位于 Python site-packages
export LD_LIBRARY_PATH=$(python3 -c "import onnxruntime; print(onnxruntime.__path__[0])/capi"):$LD_LIBRARY_PATH
```

### 构建

```bash
cmake -B build -S . && cmake --build build -j$(nproc)
```

### 配置

```bash
cp config/gateway.example.json config/gateway.json
echo "LLM_API_KEY=sk-your-key" > config/gateway.env
chmod 600 config/gateway.env
```

### 启动

```bash
# 网关（模型文件在 model/ 内，无需额外服务）
./build/src/ai-gateway config/gateway.json
```

### 集成到现有客户端

```python
# 将 LLM API URL 指向网关
LLM_URL = "http://127.0.0.1:3003/v1/chat/completions"
# 原来的 URL 可以移除，API Key 由网关统一管理
```

### systemd 部署

```bash
sudo cp scripts/ai-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-gateway
```

## 配置项

| 字段 | 默认 | 说明 |
|------|------|------|
| `server.port` | 3003 | 监听端口 |
| `backend.url` | DeepSeek API | LLM 后端 URL（OpenAI 兼容） |
| `backend.model` | deepseek-v4-flash | 模型名称 |
| `backend.timeout_seconds` | 60 | 请求超时 |
| `embedding.url` | 127.0.0.1:8081 | 已弃用（C++ ONNX 替代） |
| `cache.enabled` | true | 启用语义缓存 |
| `cache.similarity_threshold` | 0.80 | 余弦相似度阈值 |
| `cache.max_entries` | 10000 | 最大缓存条目 |
| `cache.ttl_days` | 7 | 缓存过期天数 |
| `filter.max_input_chars` | 500 | 输入最大字符 |
| `filter.max_output_chars` | 600 | 输出最大字符 |
| `filter.block_urls` | true | 拦截 URL |

API Key 通过 `config/gateway.env`（systemd `EnvironmentFile`）或环境变量 `LLM_API_KEY` 注入。

## 测试

```bash
cmake --build build --target test_filter test_lru_store test_request \
                                          test_response test_router \
                                          test_stats test_config

for t in build/tests/test_*; do $t; done
# 7/7 模块, 50/50 用例

# HNSW 召回率基准（手动运行，不纳入 ctest）
cmake --build build --target recall_bench
# 真实数据集模式（ONNX 编码后建索引）
./build/tests/recall_bench scripts/datasets/synthetic.jsonl model/model_int8.onnx model/vocab.txt
# 合成向量模式（规模曲线：抽样 200 查询与暴力检索对比）
./build/tests/recall_bench --synthetic 10000 --dim 512 --queries 200
```

## 实测性能

> 测试环境：本机 2 vCPU / 2GB 内存（**压测端与被测服务同机环回**，服务 `taskset -c 0`、压测端 `taskset -c 1`）
> 后端：DeepSeek v4-flash（公网 API）；Embedding：ONNX bge-small-zh-v1.5 INT8 (512d)，进程内推理
> 数据集：`scripts/datasets/synthetic.jsonl` 320 条（15 语义簇 × 20 条同义改写 + 20 条独立问题）、`scripts/datasets/real.jsonl` 92 条（真实提问），各 2 轮
> 原始输出：`results/replay_synthetic_fixed.json`、`results/replay_real_fixed.json`（HNSW 修复前：`replay_synthetic.json`、`replay_real.json`）

| 指标 | 合成集 | 真实集 | 说明 |
|------|--------|--------|------|
| 命中率 R1 / R2 | 25.6% / **100.0%** | 15.4% / **100.0%** | 两轮合计 62.8% / 57.7% |
| 命中延迟 p50 / p95 | 13ms / 18ms | 22ms / 51ms | 本地 ONNX 推理 + 图检索 |
| 未命中延迟 p50 | 808ms | 731ms | 含公网 LLM API 往返 |
| 网关自身处理延迟 | < 1ms | < 1ms | 不含 LLM 与 Embedding |

**读法**：R1 是"首次提问"——每个语义簇的首条必然未命中，同义改写还要跨过
`similarity_threshold`（默认 0.8）才算命中，因此 R1 反映的是**语义匹配的严格程度**；
R2 是"重复提问"，命中率 100% 说明**缓存写入与检索链路完全正常**。

**HNSW 召回修复（本次重测的主要产出）**：修复前 `tests/recall_bench` 实测自研 HNSW 在 320 向量规模下
**自检索 top-1 仅 30.9%、top-3 召回率 31.7%**，导致 R2 命中率被压在 48.4%（字面完全相同的消息也检索不回来）。
根因是**邻居饱和时直接放弃反向连接**，使大量节点没有入边、在图中不可达（缺失论文的启发式剪枝）。
补上 `SELECT-NEIGHBORS-HEURISTIC`（多样性筛选）与**满时收缩重选**后：
**自检索 top-1 30.9% → 99.4%、top-3 召回率 31.7% → 99.6%**，R2 命中率随之升到 100%。

